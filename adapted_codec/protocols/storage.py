"""The storage step of each recompression protocol (App. A.2), applied to every decoded image before it is scored,
stored and re-encoded.

  none   float chain: identity
  clamp  clipped float: clip to the valid range
  png8   8-bit: S_8(x) = rint(255 clip(x, 0, 1)) / 255 (ties to even), with a lossless PNG round trip (checked)

Two float paths of S_8 exist and both are kept because rounding decisions downstream depend on the last float digit:
the unsigned path returns float32(u8) / 255; the signed path returns the value the ELIC encoder computes from the stored
file, ((u8 / 255 - 0.5) / 0.5 + 1) / 2 (`s8_signed_path`). The search trials can evaluate S_8 on the device with a lookup
table (`fast_s8`, the same values; `fast_s8_audit` compares against the PNG path every N calls)."""
import io

import numpy as np
import torch
from PIL import Image

from adapted_codec.settings import S

_FAST_S8_STATE = {"lut": None, "calls": 0, "audits": 0}


def _png8_roundtrip(x01):
    out = []
    for im in x01.detach().float().clamp(0, 1).cpu():
        a = np.rint(im.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG")
        buf.seek(0)
        b = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
        if not np.array_equal(a, b):
            raise RuntimeError("PNG round trip changed pixels")
        out.append(torch.from_numpy(b).permute(2, 0, 1).float().div(255.0))
    return torch.stack(out)


def _store01_unsigned(x):
    if S.chain_format == "none":
        return x
    if S.chain_format == "clamp":
        return x.clamp(0, 1)
    return _png8_roundtrip(x).to(device=x.device, dtype=x.dtype)


def store11(x):
    if S.chain_format == "none":
        return x
    if S.chain_format == "clamp":
        return x.clamp(-1, 1)
    return (_png8_roundtrip((x + 1) / 2) * 2 - 1).to(device=x.device, dtype=x.dtype)


def to_u8(x01):
    return [np.rint(im.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            for im in x01.detach().float().clamp(0, 1).cpu()]


def png_bytes(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def u8_to01(b, device):
    u8 = torch.from_numpy(np.ascontiguousarray(b)).permute(2, 0, 1)[None]
    x11 = (u8.float().div(255) - 0.5) / 0.5
    return (x11.to(device).float() + 1) / 2


def _png8_roundtrip_signed(x01):
    out = []
    for a in to_u8(x01):
        b = np.array(Image.open(io.BytesIO(png_bytes(a))).convert("RGB"), dtype=np.uint8)
        if not np.array_equal(a, b):
            raise RuntimeError("PNG round trip changed pixels")
        out.append(u8_to01(b, x01.device)[0])
    return torch.stack(out)


def _store01_signed(x):
    if S.chain_format == "none":
        return x
    if S.chain_format == "clamp":
        return x.clamp(0, 1)
    return _png8_roundtrip_signed(x).to(device=x.device, dtype=x.dtype)


def store01(x):
    """Storage map for images in [0, 1] (chain_format: none / clamp / png8)."""
    return (_store01_signed if S.s8_signed_path else _store01_unsigned)(x)


def _fast_s8(x):
    from adapted_codec.protocols.storage import store01
    if S.chain_format != "png8":
        raise RuntimeError(f"fast_s8 needs chain_format png8, not {S.chain_format}")
    lut = _FAST_S8_STATE["lut"]
    if lut is None or lut.device != x.device:
        import numpy as np
        lut = torch.from_numpy(np.arange(256, dtype=np.uint8)).float().div(255.0).to(x.device)
        _FAST_S8_STATE["lut"] = lut
    u = torch.round(x.detach().float().clamp(0, 1) * 255.0)
    out = lut[u.long()].to(dtype=x.dtype)
    _FAST_S8_STATE["calls"] += 1
    if S.fast_s8_audit > 0 and _FAST_S8_STATE["calls"] % S.fast_s8_audit == 0:
        _FAST_S8_STATE["audits"] += 1
        if not torch.equal(out, store01(x)):
            raise RuntimeError(f"fast_s8 audit failed at call {_FAST_S8_STATE['calls']}: lookup table != PNG path")
    return out


def search_s8(store01):
    return _fast_s8 if S.fast_s8 else store01


def tensor_md5(*ts):
    import hashlib
    h = hashlib.md5()
    for t in ts:
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:8]
