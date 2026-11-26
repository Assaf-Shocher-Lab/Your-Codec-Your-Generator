"""MS-ILLM (Muckley et al., ICML 2023; NeuralCompression HiFiC autoencoder with the MS-ILLM weights) in the 8-bit file
chain. The model gets the CompressAI-style names the codec interface reads (g_a, g_s, h_a, h_s = cat(scales, means),
gaussian_conditional, entropy_bottleneck, M = 220), i.e. the mean-scale hyperprior of HiFiCAutoencoder.compress:
z = h_a(y), z_hat = round(z - m) + m, y_hat = round(y - mu) + mu, output clamp(0, 1). Wrapper f = 16, k = 220, c = 10.
Methods: "msillm_adapted" (stage-0 decoder: y* = select_target, corrected 8-bit write) and "msillm_plain".
File: b"MSI1" + (h, w, zH, zW, len_z, len_y) as ">HHHHII", then the z and y rANS strings (payload = the two strings)."""
import os
import sys

import torch.nn as nn

import struct

import numpy as np
import torch

from adapted_codec.chain import Adapter, Decoded
from adapted_codec.codec_ops import MeanScaleOps, encode_bytes
from adapted_codec.determinism import set_exact_math
from adapted_codec.encoder import encode, symbol_record
from adapted_codec.protocols.eight_bit.target_correction import corrected_write
from adapted_codec.settings import S
from adapted_codec.target_search import select_target
from adapted_codec.wrapper import build_wrapper

NC_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party",
                      "neuralcompression", "src")
MAGIC = b"MSI1"
HDR = ">HHHHII"
HLEN = 4 + struct.calcsize(HDR)
if NC_SRC not in sys.path:
    sys.path.insert(0, NC_SRC)


class _HS(nn.Module):
    def __init__(self, scale, mean):
        super().__init__()
        self.scale, self.mean = scale, mean

    def forward(self, z_hat):
        return torch.cat([self.scale(z_hat), self.mean(z_hat)], 1)


def load_hific(name, device=None):
    from compressai.models.utils import update_registered_buffers
    from neuralcompression.models import HiFiCAutoencoder
    net = HiFiCAutoencoder()
    sd = torch.load(name, map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd)
    net.eval()
    net.g_a, net.g_s, net.h_a = net.encoder, net.decoder, net.hyper_analysis
    net.h_s = _HS(net.hyper_synthesis_scale, net.hyper_synthesis_mean)
    net.gaussian_conditional, net.entropy_bottleneck = net.latent_bottleneck, net.hyper_bottleneck
    net.M = 220
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return net.to(dev)


class MsillmBackend:
    """MS-ILLM at one target rate: HiFiC autoencoder with the released weights (exact math), PCA basis (k = M = 220),
    wrapper, mean-scale codec interface (z = h_a(y), no context model)."""

    def __init__(self, rate, pca, c, weights):
        self.rate, self.c = rate, c
        set_exact_math()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = load_hific(weights.format(rate), self.dev)
        self.net.update(force=True)
        p = torch.load(pca, map_location="cpu", weights_only=False)
        self.wrapper = build_wrapper(self.net, p, c)
        self.ops = MeanScaleOps(self.net, self.wrapper)
        S.image_hw = None


class MsillmAdapter(Adapter):
    def __init__(self, backend, method):
        if method not in ("msillm_adapted", "msillm_plain"):
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
            if self.name == "msillm_plain":
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
            raise RuntimeError(f"{path}: not an MSI1 file")
        h, w, zH, zW, lz, ly = struct.unpack_from(HDR, data, 4)
        z_s, y_s = data[HLEN:HLEN + lz], data[HLEN + lz:HLEN + lz + ly]
        if HLEN + lz + ly != len(data):
            raise RuntimeError(f"{path}: length mismatch")
        with torch.no_grad():
            zd = ops.entropy_bottleneck.decompress([z_s], (zH, zW))
            sc, mu = b.net.h_s(zd).chunk(2, 1)
            yd = ops.gaussian_conditional.decompress([y_s], ops.gaussian_conditional.build_indexes(sc), zd.dtype, means=mu)
            if self.name == "msillm_plain":
                xd = ops.base_decode(yd, zd)
                diag = "plain"
            else:
                yt, it = select_target(ops, zd, yd)
                xr = ops.base_decode(yd, zd)
                xd = corrected_write(ops, zd, yd, yd if yt is None else yt, xr)
                q0 = symbol_record(ops, ops.wrapped_encode(xd))
                fpass = int(torch.equal(q0.z_hat, zd) and torch.equal(q0.y_hat, yd))
                nz, ny = int((q0.z_hat != zd).sum()), int((q0.y_hat != yd).sum())
                diag = f"zs{it}" if yt is not None else "zsNONE"
                diag += f"/pass{fpass}/nz{nz}/ny{ny}"
            if xd.shape[-2:] != (h, w):
                raise RuntimeError(f"decoded {tuple(xd.shape[-2:])} != header {(h, w)}")
            syms = [zd.float().cpu().numpy(), yd.float().cpu().numpy()]
        return Decoded(xd, "01", syms, diag)
