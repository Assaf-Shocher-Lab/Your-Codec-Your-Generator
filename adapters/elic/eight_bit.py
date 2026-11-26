"""ELIC in the 8-bit file chain (adapted_codec/chain.py): Kodak (6 rates) and the 256 x 256 sets (Xu et al.'s weights).

Sender, round g: x = ((u8 / 255 - 0.5) / 0.5 + 1) / 2, the ELIC evaluation's normalization through [-1, 1] (computed on
the CPU up to [-1, 1], then on the device); round 1 sends Q(E(x)), rounds >= 2 Q(E'(x)) (adapted_codec/encoder.py).
File: b"E3M1" + (h, w, zH, zW, ny, nz) as "<HHHHBB", then the ny = 10 y strings (5 slices x anchor / non-anchor) and
the z string, each preceded by its length (uint32, little-endian). Payload (bpp) = the strings.

Receiver (method "elic_adapted"): (z_hat, y_hat) from the file; the float target y*_F (margins of the float protocol,
m_y = `float_margin_y`) and the float right inverse x_F = W(y*_F, b(D(y_hat))) on the codec's [-1, 1] image path; the
8-bit target y*_8 (m_y = `margin_y`; y*_F when none is found); then the 8-bit stage: the escalation ladder from y*_8
on x_F (`ladder` = 1; stage 0 = the corrected write) or the corrected write alone (`ladder` = 0). The 8-bit repair
therefore starts from the float D' output, not from D(y_hat). Method "elic_plain": the codec as released."""
import struct

import torch

from adapted_codec.chain import Adapter, Decoded
from adapted_codec.encoder import encode, symbol_record
from adapted_codec.protocols.eight_bit.escalation import escalation_write
from adapted_codec.protocols.eight_bit.target_correction import corrected_write
from adapted_codec.settings import S
from adapted_codec.target_search import select_target
from adapters.elic.codec import _pack, _unpack
from adapters.elic.file_format import PAD_MULTIPLE

MAGIC = b"E3M1"
HDR = "<HHHHBB"
HLEN = 4 + struct.calcsize(HDR)


def from_uint8(u8):
    """uint8 -> [-1, 1] as the ELIC evaluation loads an image: ToTensor (/ 255), Normalize(0.5, 0.5)."""
    return (u8.float().div(255) - 0.5) / 0.5


def to01(x):
    return (x.float() + 1) / 2


def file_bytes(hw, z_hat, strings):
    """The E3M1 file of a record: header, then every y string and the z string with uint32 lengths."""
    yl, zl = strings
    ys, zs = list(yl[0]), list(zl)
    h, w = hw
    zH, zW = z_hat.shape[-2:]
    b_ = MAGIC + struct.pack(HDR, h, w, zH, zW, len(ys), len(zs))
    for s_ in ys + zs:
        b_ += struct.pack("<I", len(s_)) + s_
    return b_


def read_file(path):
    """(h, w), (zH, zW), y strings, z strings, the file bytes. The file holds nothing else."""
    with open(path, "rb") as f_:
        b_ = f_.read()
    if b_[:4] != MAGIC:
        raise RuntimeError(f"{path}: not an E3M1 file")
    h, w, zH, zW, ny, nz_ = struct.unpack_from(HDR, b_, 4)
    o_, ss = HLEN, []
    for _ in range(ny + nz_):
        (L,) = struct.unpack_from("<I", b_, o_)
        ss.append(b_[o_ + 4:o_ + 4 + L])
        o_ += 4 + L
    if o_ != len(b_):
        raise RuntimeError(f"{path}: length mismatch")
    return (h, w), (zH, zW), ss[:ny], ss[ny:], b_


def entropy_decode(ops, zshape, ys, zs):
    """(y_hat, z_hat) of the strings: the hyper string, then the slice / checkerboard passes of ElicOps._slices."""
    zd = ops.entropy_bottleneck.decompress(zs, zshape)
    it_ = iter(ys)

    def dec(i, mu, sc, anchor):
        idx = ops.gaussian_conditional.build_indexes(_pack(sc, anchor))
        return _unpack(ops.gaussian_conditional.decompress([next(it_)], idx, means=_pack(mu, anchor)), anchor, mu)
    yd, _, _ = ops._slices(zd, dec)
    return yd, zd


def float_right_inverse(ops, net, y_hat, yt):
    """x_F = W(y*_F, b(D(y_hat))) on the [-1, 1] image path of the codec (D(y) = 2 g_s(y).clamp(0, 1) - 1)."""
    x_dec = net.g_s(y_hat).clamp(0, 1) * 2 - 1
    return ops.write(to01(x_dec), yt) * 2 - 1


def float_target(ops, z_hat, y_hat):
    """y*_F: the selection target under the float margins (m_y = float_margin_y); y_hat if none is found."""
    my8 = S.margin_y
    try:
        S.margin_y = S.float_margin_y
        yt, _ = select_target(ops, z_hat, y_hat)
    finally:
        S.margin_y = my8
    return y_hat.clone() if yt is None else yt


class ElicFileAdapter(Adapter):
    def __init__(self, backend, method):
        if method not in ("elic_adapted", "elic_plain"):
            raise ValueError(method)
        if S.float_margin_y is None:
            raise ValueError("the ELIC 8-bit receiver needs float_margin_y (m_y of the float target)")
        self.b, self.name, self.rate = backend, method, backend.rate

    def device_name(self):
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    def bitstream_of(self, data):
        _, _, ys, zs, _ = _split(data)
        return b"".join(ys + zs)

    def encode_to_file(self, x_u8, g, path):
        b = self.b
        with torch.no_grad():
            x = from_uint8(torch.from_numpy(x_u8).permute(2, 0, 1)[None]).to(b.dev)
            h, w = x.shape[2:]
            if h % PAD_MULTIPLE or w % PAD_MULTIPLE:
                raise SystemExit(f"image {h}x{w} not a multiple of {PAD_MULTIPLE} (no pad path in the 8-bit chain)")
            xf = to01(x)
            if self.name == "elic_plain":
                e = symbol_record(b.ops, b.ops.base_encode(xf))
                tag = "plain"
            else:
                e = encode(b.ops, xf, gen=g)
                tag = e.fp_stop
            strings, nbytes, ok = b.ops.encode_bytes(e.z_hat, e.y_hat)
            data = file_bytes((h, w), e.z_hat, strings)
            meta = {"enc_tag": tag if ok else f"{tag}/decode_ok0", "bpp_est": round(nbytes * 8 / (h * w), 7)}
            del e, x, xf
        with open(path, "wb") as f:
            f.write(data)
        return meta

    def decode_from_file(self, path):
        b = self.b
        ops = b.ops
        hw, zshape, ys_, zs_, _ = read_file(path)
        h, w = hw
        with torch.no_grad():
            yd, zd = entropy_decode(ops, zshape, ys_, zs_)
            if self.name == "elic_plain":
                xd = ops.base_decode(yd, zd)[:, :, :h, :w].clamp(0, 1)
                diag = "plain"
            else:
                yt = float_target(ops, zd, yd)
                xin = to01(float_right_inverse(ops, b.net, yd, yt))
                yt8, it = select_target(ops, zd, yd)
                yt8 = yt if yt8 is None else yt8
                if S.ladder:
                    xd, stage = escalation_write(ops, zd, yd, yt8, xin)
                else:
                    xd, stage = corrected_write(ops, zd, yd, yt8, xin), 0
                q0 = symbol_record(ops, ops.wrapped_encode(xd))
                fpass = int(torch.equal(q0.z_hat, zd) and torch.equal(q0.y_hat, yd))
                nz, ny = int((q0.z_hat != zd).sum()), int((q0.y_hat != yd).sum())
                diag = f"zs{it}/pass{fpass}/nz{nz}/ny{ny}/esc{stage}"
            if xd.shape[-2:] != (h, w):
                raise RuntimeError(f"decoded {tuple(xd.shape[-2:])} != header {(h, w)}")
            syms = [zd.float().cpu().numpy(), yd.float().cpu().numpy()]
        return Decoded(xd, "01", syms, diag)


def _split(data):
    h, w, zH, zW, ny, nz_ = struct.unpack_from(HDR, data, 4)
    o_, ss = HLEN, []
    for _ in range(ny + nz_):
        (L,) = struct.unpack_from("<I", data, o_)
        ss.append(data[o_ + 4:o_ + 4 + L])
        o_ += 4 + L
    return (h, w), (zH, zW), ss[:ny], ss[ny:], data
