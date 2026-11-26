"""TCM (Liu et al., CVPR 2023; third_party/tcm, N = 64, M = 320, MSE weights) in the 8-bit file chain.

Codec interface (TcmOps, mean-scale base): z = h_a(y), z_hat by the EntropyBottleneck medians; y in 5 channel slices,
each slice's mean and scale from the hyper means / scales and the post-LRP y_hat of the earlier slices; the transmitted
y_hat is the pre-LRP value round(y - mu) + mu, the synthesis reads post = y_hat + 0.5 tanh(lrp(...)). The wrapper acts
on g_a (pre-LRP y space), f = 16, k = 320, c = 10. Methods: "tcm_adapted" (round 1 the codec's own record, rounds >= 2
the wrapped record; decoder: y* = select_target, x_raw = base_decode(y_hat), escalation ladder from y*) and "tcm_plain".
File: b"TCM1" + (h, w, zH, zW, len_z, len_y) as ">HHHHII", then the z and y rANS strings (payload = the two strings)."""
import hashlib
import os
import sys

import struct

import numpy as np
import torch

from adapted_codec.chain import Adapter, Decoded
from adapted_codec.codec_ops import MeanScaleOps, encode_bytes
from adapted_codec.determinism import set_exact_math
from adapted_codec.encoder import encode, symbol_record
from adapted_codec.protocols.eight_bit.escalation import escalation_write
from adapted_codec.settings import S
from adapted_codec.target_search import select_target
from adapted_codec.wrapper import build_wrapper

TCM_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party",
                       "tcm", "src")
TCM_PY_MD5 = "afde7202326d76f11196750752cf637e"      # third_party/tcm/src/models/tcm.py (upstream, unchanged)
MAGIC = b"TCM1"
HDR = ">HHHHII"
HLEN = 4 + struct.calcsize(HDR)
if TCM_SRC not in sys.path:
    sys.path.insert(0, TCM_SRC)


def load_tcm(path, dev):
    from models.tcm import TCM
    import models.tcm as mt
    if hashlib.md5(open(mt.__file__, "rb").read()).hexdigest() != TCM_PY_MD5:
        raise RuntimeError(f"tcm.py md5 mismatch {mt.__file__}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net = TCM(config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
    net.load_state_dict(sd)
    net = net.eval().to(dev)
    net.update(force=True)
    return net


class TcmOps(MeanScaleOps):
    def _slices(self, z_hat, get_slice, shape):
        net = self.net
        lsc, lmu = net.h_scale_s(z_hat), net.h_mean_s(z_hat)
        H, W = shape
        post, pre, scales, means = [], [], [], []
        for i in range(net.num_slices):
            support = post if net.max_support_slices < 0 else post[:net.max_support_slices]
            ms = net.atten_mean[i](torch.cat([lmu] + support, 1))
            mu = net.cc_mean_transforms[i](ms)[:, :, :H, :W]
            ss = net.atten_scale[i](torch.cat([lsc] + support, 1))
            sc = net.cc_scale_transforms[i](ss)[:, :, :H, :W]
            yq = get_slice(i, mu, sc)
            lrp = 0.5 * torch.tanh(net.lrp_transforms[i](torch.cat([ms, yq], 1)))
            pre.append(yq)
            post.append(yq + lrp)
            scales.append(sc)
            means.append(mu)
        return torch.cat(pre, 1), torch.cat(scales, 1), torch.cat(means, 1), torch.cat(post, 1)

    @torch.no_grad()
    def quantize_main(self, y, z_hat):
        ys = y.chunk(self.net.num_slices, 1)
        gc = self.gaussian_conditional
        yh, scales, means, _ = self._slices(z_hat, lambda i, mu, sc: gc.quantize(ys[i], "symbols", mu) + mu,
                                    y.shape[-2:])
        return yh, scales, means

    @torch.no_grad()
    def post_lrp(self, y, z_hat):
        ys = y.chunk(self.net.num_slices, 1)
        return self._slices(z_hat, lambda i, mu, sc: ys[i], y.shape[-2:])[3]

    def base_decode(self, y_hat, z_hat=None):
        if z_hat is None:
            z_hat = self.hyper(y_hat)[1]
        return self.net.g_s(self.post_lrp(y_hat, z_hat)).clamp(0, 1)

    @torch.no_grad()
    def decode_y_string(self, y_string, zd):
        from compressai.ans import RansDecoder
        gc = self.gaussian_conditional
        cdf = gc.quantized_cdf.tolist()
        cdf_lengths = gc.cdf_length.reshape(-1).int().tolist()
        offsets = gc.offset.reshape(-1).int().tolist()
        H, W = zd.shape[2] * 4, zd.shape[3] * 4
        dec = RansDecoder()
        dec.set_stream(y_string)

        def dslice(i, mu, sc):
            rv = dec.decode_stream(gc.build_indexes(sc).reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.tensor(rv, dtype=torch.float32, device=mu.device).reshape(1, -1, H, W)
            return gc.dequantize(rv, mu)
        yd, _, _, post = self._slices(zd, dslice, (H, W))
        return yd, post

    @torch.no_grad()
    def encode_bytes(self, z_hat, y_hat):
        from compressai.ans import BufferedRansEncoder
        net, gc, eb = self.net, self.gaussian_conditional, self.entropy_bottleneck
        z_strings = eb.compress(z_hat)
        zd = eb.decompress(z_strings, z_hat.shape[-2:])
        cdf = gc.quantized_cdf.tolist()
        cdf_lengths = gc.cdf_length.reshape(-1).int().tolist()
        offsets = gc.offset.reshape(-1).int().tolist()
        ys = y_hat.chunk(net.num_slices, 1)
        syms, idxs = [], []

        def enc(i, mu, sc):
            q = gc.quantize(ys[i], "symbols", mu)
            syms.extend(q.reshape(-1).int().tolist())
            idxs.extend(gc.build_indexes(sc).reshape(-1).tolist())
            return q + mu
        ye, _, _, _ = self._slices(zd, enc, y_hat.shape[-2:])
        encoder = BufferedRansEncoder()
        encoder.encode_with_indexes(syms, idxs, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        yd, post = self.decode_y_string(y_string, zd)
        xo = net.g_s(post).clamp(0, 1)
        ok = bool(torch.equal(zd, z_hat) and torch.equal(ye, y_hat) and torch.equal(yd, y_hat)
                  and torch.equal(xo, self.base_decode(y_hat, z_hat)))
        return (z_strings[0], y_string), len(y_string) + len(z_strings[0]), ok


class TcmBackend:
    """TCM (N = 64, M = 320, MSE) at one lambda: model (exact math), PCA basis, wrapper, codec interface."""

    def __init__(self, rate, pca, c, weights):
        self.rate, self.c = rate, c
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        set_exact_math()
        self.net = load_tcm(weights.format(rate), self.dev)
        p = torch.load(pca, map_location="cpu", weights_only=False)
        self.wrapper = build_wrapper(self.net, p, c)
        self.ops = TcmOps(self.net, self.wrapper)
        S.image_hw = None


class TcmAdapter(Adapter):
    def __init__(self, backend, method):
        if method not in ("tcm_adapted", "tcm_plain"):
            raise ValueError(method)
        self.b, self.name, self.rate = backend, method, backend.rate

    def device_name(self):
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    def bitstream_of(self, data):
        return data[HLEN:]

    def _tensor(self, x_u8, g):
        if g == 1:
            a = np.asarray(x_u8, dtype=np.float32) / 255.0
            return torch.from_numpy(a).permute(2, 0, 1)[None].to(self.b.dev)
        t = torch.from_numpy(np.ascontiguousarray(x_u8)).permute(2, 0, 1).float().div(255.0)
        return torch.stack([t]).to(device=self.b.dev, dtype=torch.float32)

    def encode_to_file(self, x_u8, g, path):
        b = self.b
        with torch.no_grad():
            x = self._tensor(x_u8, g)
            h, w = x.shape[2:]
            if h % 64 or w % 64:
                raise SystemExit(f"image {h}x{w} not a multiple of 64 (no pad path in this port)")
            if self.name == "tcm_plain":
                e = symbol_record(b.ops, b.ops.base_encode(x))
                tag = "plain"
            else:
                e = encode(b.ops, x, check=None, gen=g)
                tag = e.fp_stop
            (z_s, y_s), nbytes, ok = b.ops.encode_bytes(e.z_hat, e.y_hat)
            zH, zW = e.z_hat.shape[-2:]
            data = MAGIC + struct.pack(HDR, h, w, zH, zW, len(z_s), len(y_s)) + z_s + y_s
            meta = {"enc_tag": tag if ok else f"{tag}/decode_ok0", "bpp_est": round(nbytes * 8 / (h * w), 7)}
            del e, x
        with open(path, "wb") as f:
            f.write(data)
        return meta

    def decode_from_file(self, path):
        b = self.b
        ops = b.ops
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != MAGIC:
            raise RuntimeError(f"{path}: not a TCM1 file")
        h, w, zH, zW, lz, ly = struct.unpack_from(HDR, data, 4)
        z_s, y_s = data[HLEN:HLEN + lz], data[HLEN + lz:HLEN + lz + ly]
        if HLEN + lz + ly != len(data):
            raise RuntimeError(f"{path}: length mismatch")
        with torch.no_grad():
            zd = ops.entropy_bottleneck.decompress([z_s], (zH, zW))
            yd, post = ops.decode_y_string(y_s, zd)
            if self.name == "tcm_plain":
                xd = b.net.g_s(post).clamp(0, 1)
                diag = "plain"
            else:
                yt, it = select_target(ops, zd, yd)
                xr = ops.base_decode(yd, zd)

                xd, stage = escalation_write(ops, zd, yd, yt, xr)

                q0 = symbol_record(ops, ops.wrapped_encode(xd))
                fpass = int(torch.equal(q0.z_hat, zd) and torch.equal(q0.y_hat, yd))
                nz, ny = int((q0.z_hat != zd).sum()), int((q0.y_hat != yd).sum())
                diag = f"zs{it}" if yt is not None else "zsNONE"
                diag += f"/pass{fpass}/nz{nz}/ny{ny}/esc{stage}"
            if xd.shape[-2:] != (h, w):
                raise RuntimeError(f"decoded {tuple(xd.shape[-2:])} != header {(h, w)}")
            syms = [zd.float().cpu().numpy(), yd.float().cpu().numpy()]
        return Decoded(xd, "01", syms, diag)
