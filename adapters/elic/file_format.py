"""The ELIC file: height and width (2 x uint32, big-endian), then the 11 entropy-coded strings (z, then the anchor and
non-anchor strings of the 5 slices), each preceded by its length (uint32). Decoding pads (h, w) to multiples of 64 and
reads a hyper latent of (H / 64, W / 64). bpp of the paper = 8 x (the 11 strings) / (h w) (payload).
The decoder reads the record with ELIC's own decompress, patched to return (z_hat, y_hat) when `pair_only` is set
(patches/elic_decompress.patch)."""
import struct

from adapted_codec.pad_repair import pad_to

HEADER = ">II"
PAD_MULTIPLE = 64
Z_STRIDE = 64
N_Y = 10


def core_streams(strings):
    ys, zs = strings[0][0], strings[1]
    assert len(ys) == N_Y and len(zs) == 1, (len(ys), len(zs))
    return [zs[0]] + list(ys)


def serialize_core(strings):
    return b"".join(struct.pack(">I", len(s)) + s for s in core_streams(strings))


class Coder:
    def __init__(self, net):
        self.net = net

    @staticmethod
    def encode(strings, h, w):
        return struct.pack(HEADER, h, w) + serialize_core(strings)

    def decode(self, data):
        h, w = struct.unpack(HEADER, data[:8])
        pos, streams = 8, []
        for _ in range(1 + N_Y):
            n, = struct.unpack(">I", data[pos:pos + 4])
            streams.append(data[pos + 4:pos + 4 + n])
            pos += 4 + n
        assert pos == len(data), (pos, len(data))
        z, ys = streams[0], streams[1:]
        H, W = pad_to(h, w, PAD_MULTIPLE)
        self.net.pair_only = True
        try:
            y_strings = [[[ys[2 * i]], [ys[2 * i + 1]]] for i in range(N_Y // 2)]
            out = self.net.decompress([y_strings, [z]], (H // Z_STRIDE, W // Z_STRIDE))
        finally:
            self.net.pair_only = False
        return out["z_hat"], out["y_hat"], (h, w)
