"""The HPCM file: height, width and the z-stream length (3 x uint32, big-endian), then the z rANS stream, then the y
rANS stream. bpp of the paper = 8 x (both streams) / (h w) (payload, header excluded). Decoding pads (h, w) to multiples
of 256 and reads a hyper latent of (H / 64, W / 64)."""
import struct

import torch

from adapted_codec.pad_repair import pad_to
from adapters.hpcm.upstream_copies import HPCMCoderCopy

HEADER = ">III"
PAD_MULTIPLE = 256
Z_STRIDE = 64


class Coder:
    def __init__(self, net):
        self.net = net

    @staticmethod
    def encode(strings, h, w):
        y_string, z_string = strings[0][0], strings[1][0]
        return struct.pack(HEADER, h, w, len(z_string)) + z_string + y_string

    def decode(self, data):
        h, w, nz = struct.unpack(HEADER, data[:12])
        z_string, y_string = data[12:12 + nz], data[12 + nz:]
        H, W = pad_to(h, w, PAD_MULTIPLE)
        self.net.pair_only = True
        try:
            out = HPCMCoderCopy.decompress(self.net, [y_string, z_string], (H // Z_STRIDE, W // Z_STRIDE))
        finally:
            self.net.pair_only = False
        return out["z_hat"], out["y_hat"], (h, w)

    def columns(self, strings, z_hat, y_hat, h, w, prev_bytes):
        data = self.encode(strings, h, w)
        z_dec, y_dec, hw = self.decode(data)
        ok = hw == (h, w) and torch.equal(z_dec, z_hat) and torch.equal(y_dec, y_hat)
        return dict(file_bytes=len(data), bpp_coded=round(8 * len(data) / (h * w), 7), roundtrip=int(ok),
                    same_bytes_prev="" if prev_bytes is None else int(data == prev_bytes)), data


def own_codec_check(net, x_padded, strings, x_plain, h, w):
    out = net.compress(x_padded)
    same = int(out["strings"][0] == strings[0][0] and out["strings"][1] == strings[1][0])
    x_own = net.decompress(out["strings"], out["shape"])["x_hat"][:, :, :h, :w]
    return [f"cc{same}", f"x{float((x_own - x_plain).abs().max()):.1e}"]
