"""The AEIC file: height, width and stream length (3 x uint32, big-endian; AEIC's my_write_body), AEIC's rANS stream of
the record (z_hat, y_hat), and for images > 768^2 the 6 colour integers (6 x uint16) of AEIC's colour fix. The coder is
AEIC's own (third_party/aeic/src/cpp, built by scripts/build_aeic_coder.sh) with the copies F6-F8
(upstream_copies.AEICCoderCopy): it codes exactly the record that is sent, every element. Payload bpp = 8 (stream +
colour bytes) / (h w); the 12-byte header is framing."""
import math
import struct
import types

import torch

from adapters.aeic.upstream_copies import AEICCoderCopy, AdaptedRequest

HEADER = ">III"
SIDE = ">6H"


class Coder:
    def __init__(self, net):
        self.codec = net.codec
        self.codec.update(force=True)

        self.codec.my_build_indexes = types.MethodType(AEICCoderCopy.my_build_indexes, self.codec)

    def encode(self, cfg, pair, x_padded, h, w):
        codec = self.codec
        codec.entropy_coder.reset()
        codec.adapted_request = AdaptedRequest(code_pair=(pair.z_hat, pair.y_hat))
        try:
            AEICCoderCopy.compress(codec, x_padded)
        finally:
            codec.adapted_request = None
        stream = codec.entropy_coder.get_encoded_stream()
        data = struct.pack(HEADER, h, w, len(stream)) + stream
        if pair.side_info is not None:
            data += struct.pack(SIDE, *pair.side_info)
        return data

    def decode(self, data):
        codec = self.codec
        h, w, n = struct.unpack(HEADER, data[:12])
        padded_y_h, padded_y_w = math.ceil(h / 64) * 2, math.ceil(w / 64) * 2
        z_size = (1, codec.y_channel // 2, math.ceil(padded_y_h / 4), math.ceil(padded_y_w / 4))
        codec.entropy_coder.reset()
        codec.entropy_coder.set_stream(data[12:12 + n])
        request = AdaptedRequest()
        codec.adapted_request = request
        try:
            AEICCoderCopy.decompress(codec, z_size, padded_y_h, padded_y_w)
        finally:
            codec.adapted_request = None
        side = struct.unpack(SIDE, data[12 + n:24 + n]) if len(data) > 12 + n else None
        return request.z_hat, request.y_hat, side
