"""mbt2018-mean (CompressAI zoo, MSE; the mean-scale hyperprior) in the 8-bit file chain (adapted_codec/chain.py).

As adapters/hyperprior/adapter.py, with the mean-scale quantizers (adapted_codec/codec_ops.py MeanScaleOps: z = h_a(y),
(scales, means) = h_s(z_hat).chunk(2, 1), y_hat = round(y - mu) + mu) and the file magic b"BMB1"."""
import struct

import numpy as np
import torch

from adapted_codec.chain import Adapter, Decoded
from adapted_codec.codec_ops import MeanScaleOps, encode_bytes
from adapted_codec.encoder import encode, symbol_record
from adapted_codec.protocols.eight_bit.escalation import escalation_write
from adapted_codec.settings import S
from adapted_codec.target_search import select_target
from adapted_codec.wrapper import build_wrapper

MAGIC = b"BMB1"
HDR = ">HHHHII"
HLEN = 4 + struct.calcsize(HDR)


def load_mean_scale(quality, dev):
    from compressai.zoo import mbt2018_mean
    from adapted_codec.determinism import set_exact_math
    net = mbt2018_mean(quality=quality, metric="mse", pretrained=True, progress=False).eval()
    set_exact_math()
    return net.to(dev)


class MeanScaleBackend:
    """One CompressAI mbt2018-mean quality (MSE): model (exact math), PCA basis, wrapper, codec interface."""

    def __init__(self, rate, pca, c):
        self.rate, self.c = rate, c
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = load_mean_scale(int(rate), self.dev)
        if type(self.net).__name__ != "MeanScaleHyperprior":
            raise RuntimeError(f"model class {type(self.net).__name__} != MeanScaleHyperprior")
        p = torch.load(pca, map_location="cpu", weights_only=False)
        self.wrapper = build_wrapper(self.net, p, c)
        self.ops = MeanScaleOps(self.net, self.wrapper)
        S.image_hw = None


def decode_pair(ops, net, z_s, y_s, zH, zW):
    zd = ops.entropy_bottleneck.decompress([z_s], (zH, zW))
    sc, mu = net.h_s(zd).chunk(2, 1)
    yd = ops.gaussian_conditional.decompress([y_s], ops.gaussian_conditional.build_indexes(sc), zd.dtype, means=mu)
    return zd, yd


class MeanScaleAdapter(Adapter):
    def __init__(self, backend, method):
        if method not in ("mbt_adapted", "mbt_plain"):
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
            if self.name == "mbt_plain":
                e = symbol_record(b.ops, b.ops.base_encode(x))
                tag = "plain"
            else:
                e = encode(b.ops, x, check=None, gen=g)
                tag = e.fp_stop
            strings, nbytes, ok = encode_bytes(b.ops, e.z_hat, e.y_hat)
            y_s, z_s = strings[0][0], strings[1][0]
            zH, zW = e.z_hat.shape[-2:]
            data = MAGIC + struct.pack(HDR, h, w, zH, zW, len(z_s), len(y_s)) + z_s + y_s
            meta = {"enc_tag": tag if ok else f"{tag}/decode_ok0", "bpp_est": round(nbytes * 8 / (h * w), 7)}
            del e, x, strings
        with open(path, "wb") as f:
            f.write(data)
        return meta

    def decode_from_file(self, path):
        b = self.b
        ops = b.ops
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != MAGIC:
            raise RuntimeError(f"{path}: not a BMB1 file")
        h, w, zH, zW, lz, ly = struct.unpack_from(HDR, data, 4)
        z_s, y_s = data[HLEN:HLEN + lz], data[HLEN + lz:HLEN + lz + ly]
        if HLEN + lz + ly != len(data):
            raise RuntimeError(f"{path}: length mismatch")
        with torch.no_grad():
            zd, yd = decode_pair(ops, b.net, z_s, y_s, zH, zW)
            if self.name == "mbt_plain":
                xd = ops.base_decode(yd, zd)
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
