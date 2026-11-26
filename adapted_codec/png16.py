"""16-bit RGB PNG files for float reconstructions (standard library + NumPy; PIL cannot write 16-bit RGB).

A value v in [0, 1] is stored as round(65535 v) (step 1.5e-5; metrics recomputed from the file match the float image to
about 1e-4 dB). Pixels outside [0, 1] are clipped."""
import os
import struct
import zlib

import numpy as np

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_png16(path, img01):
    a = img01.detach().float().cpu().numpy() if hasattr(img01, "detach") else np.asarray(img01, dtype=np.float32)
    a = a.reshape(a.shape[-3:])
    q = np.rint(np.clip(a, 0.0, 1.0) * 65535.0).astype(">u2").transpose(1, 2, 0)
    h, w = q.shape[:2]
    raw = b"".join(b"\x00" + q[i].tobytes() for i in range(h))
    png = (_SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
           + _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b""))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(png)
    os.replace(tmp, path)


def read_png16(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != _SIGNATURE:
        raise ValueError(f"{path}: not a PNG file")
    pos, idat, w, h = 8, [], None, None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, colour = struct.unpack(">IIBB", body[:10])
            if (depth, colour) != (16, 2):
                raise ValueError(f"{path}: not a 16-bit RGB PNG")
        elif kind == b"IDAT":
            idat.append(body)
        pos += 12 + n
    rows = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8).reshape(h, 1 + 6 * w)
    if rows[:, 0].any():
        raise ValueError(f"{path}: filtered rows (write_png16 writes filter 0 only)")
    return rows[:, 1:].copy().view(">u2").reshape(h, w, 3).astype(np.float32) / 65535.0
