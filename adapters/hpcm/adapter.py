"""HPCM-Large in the file chain (adapted_codec/chain.py).

Methods: "hpcm_adapted" (the adapted codec: round 1 sends HPCM's own record, rounds >= 2 the wrapped record; decoder
D_8 of adapted_codec/protocols/eight_bit) and "hpcm_plain" (HPCM as released, every round). The input tensor is built
exactly as the reported runs built it, because convolution algorithms see the memory layout: round 1 from a float32
numpy array (channels-last strides), rounds >= 2 contiguous (u8 / 255 in float32 in both cases)."""
import struct

import numpy as np
import torch

from adapted_codec import pad_repair
from adapted_codec.chain import Adapter, Decoded
from adapted_codec.determinism import set_exact_math
from adapted_codec.encoder import encode, symbol_record
from adapted_codec.protocols.eight_bit.decoder import decode
from adapted_codec.settings import S
from adapted_codec.wrapper import build_wrapper
from adapters.hpcm import codec
from adapters.hpcm.file_format import Coder


class HpcmBackend:
    """One HPCM-Large rate (lambda): exact math, PCA basis, model, wrapper, codec interface and file coder."""

    def __init__(self, rate, pca, c, weights):
        set_exact_math()
        self.rate = rate
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        p = torch.load(pca, map_location="cpu", weights_only=False)
        codec.WEIGHTS = weights
        self.net = codec.load_hpcm(rate, self.dev)
        self.wrapper = build_wrapper(self.net, p, c)
        self.ops = codec.HpcmOps(self.net, self.wrapper)
        self.coder, self.zp = Coder(self.net), pad_repair
        self.pm = 256


class HpcmAdapter(Adapter):
    def __init__(self, backend, method):
        if method not in ("hpcm_adapted", "hpcm_plain"):
            raise ValueError(method)
        self.b, self.name, self.rate = backend, method, backend.rate
        self._enc = None

    def device_name(self):
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    def assert_no_encoder_state(self):
        if self._enc is not None or getattr(self.b.net, "pair_only", False):
            raise RuntimeError("encoder state is still held before the decode call")

    def bitstream_of(self, data):
        return data[12:]

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
            H, W = b.zp.pad_to(h, w, b.pm)
            S.image_hw = (h, w)
            xp = b.zp.reflect_pad(x, H, W)
            if self.name == "hpcm_plain":
                e = symbol_record(b.ops, b.ops.base_encode(xp))
                tag = "plain"
            else:
                e = encode(b.ops, xp, check=None, gen=g)
                tag = e.fp_stop
            strings, nbytes, ok = b.ops.encode_bytes(e.z_hat, e.y_hat)
            data = b.coder.encode(strings, h, w)
            meta = {"enc_tag": tag if ok else f"{tag}/decode_ok0", "bpp_est": round(nbytes * 8 / (h * w), 7)}
            del e, x, xp, strings
        with open(path, "wb") as f:
            f.write(data)
        self._enc = None
        return meta

    def decode_from_file(self, path):
        b = self.b
        if getattr(b.net, "pair_only", False):
            raise RuntimeError("pair-only flag set: not a fresh decode")
        with open(path, "rb") as f:
            data = f.read()
        h, w, _ = struct.unpack(">III", data[:12])
        with torch.no_grad():
            z_hat, y_hat, hw = b.coder.decode(data)
            if tuple(hw) != (h, w):
                raise RuntimeError(f"header {(h, w)} != decoded {hw}")
            H, W = b.zp.pad_to(h, w, b.pm)
            S.image_hw = (h, w)
            if self.name == "hpcm_plain":
                xd = b.ops.base_decode(y_hat, z_hat)[:, :, :h, :w]
                diag = "plain"
            else:
                xd, _, yt, it = decode(b.ops, z_hat, y_hat)
                diag = f"zs{it}" if yt is not None else "zsNONE"
                if (H, W) != (h, w):
                    tgt = y_hat if yt is None else yt
                    xd, ptag = b.zp.pad_repair(b.ops, xd, (h, w), tgt, y_hat, z_hat)
                    diag += f"/pad{ptag}"
            syms = [z_hat.float().cpu().numpy(), y_hat.float().cpu().numpy()]
        return Decoded(xd, "01", syms, diag)
