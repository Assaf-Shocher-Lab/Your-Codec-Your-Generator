"""The in-memory recompression chain of the float and clipped-float protocols (App. A.2).

Round g = 1 .. G, with x_0 = the 8-bit original / 255 (float32) and the image kept in memory between rounds:
  record   s_g = Q(E(x_{g-1})) at g = 1 (the codec's own record), Q(E'(x_{g-1})) for g >= 2 (adapted_codec/encoder.py);
           the base codec ("plain") sends Q(E(x_{g-1})) every round
  file     the codec's file of s_g, written to F/g###.bin and decoded back (roundtrip = the file reproduces s_g)
  decode   x_g = storage(D'(s_g)) with D' = the protocol's decoder (float_chain.decode_float or
           clipped_float.range_repair.decode_clipped); the base codec: x_g = storage(D(y_hat)).
           storage = store01 of the chain format: identity (float), clip to [0, 1] (clipped float)
Images whose size is not a multiple of the codec's padding unit are reflect-padded for the encoder and their border is
repaired after the decode (adapted_codec/pad_repair.py); Kodak images are not padded.
Outputs per chain (out_dir): rows.csv (one row per round), F/g###.bin, recon/g###.png (16-bit PNG of x_g for the
rounds in save_rounds), DONE.
"""
import csv
import hashlib
import os
import time

import numpy as np
import torch
from PIL import Image

from adapted_codec.encoder import encode, symbol_record
from adapted_codec.pad_repair import pad_repair, pad_to, reflect_pad
from adapted_codec.png16 import write_png16
from adapted_codec.protocols.storage import store01
from adapted_codec.settings import S

FIELDS = ["protocol", "method", "rate", "image", "gen", "h", "w", "bpp", "file_bytes", "md5_file", "roundtrip",
          "sym_eq_g1", "sym_eq_prev", "file_eq_g1", "file_eq_prev", "z_changed", "y_changed", "stable", "enc_tag",
          "psnr", "ms_ssim", "lpips", "dists", "enc_s", "dec_s"]


def load_image(path):
    """The 8-bit image as float32 / 255, shape (1, 3, H, W) (a permuted view: the encoder sees these strides)."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def run_memory_chain(backend, decode, image_path, out_dir, stem, rounds, method, metrics, protocol, save_rounds=(),
                     log=print):
    """One chain. backend: ops (codec interface), coder (file encode / decode), pm (padding unit), dev, rate.
    decode: the protocol's D'. method: "adapted" or "plain". metrics: {name: f(x, ref)} on images in [0, 1]."""
    ops, coder = backend.ops, backend.coder
    os.makedirs(os.path.join(out_dir, "F"), exist_ok=True)
    x = load_image(image_path).to(backend.dev)
    h, w = x.shape[2:]
    H, W = pad_to(h, w, backend.pm)
    S.image_hw = (h, w)
    rows, xi, prev, first = [], x, None, None
    for gen in range(1, rounds + 1):
        _sync()
        t0 = time.time()
        xp = reflect_pad(xi, H, W)
        if method == "plain":
            e = symbol_record(ops, ops.base_encode(xp))
            tag = "plain"
        else:
            e = encode(ops, xp, gen=gen)
            tag = e.fp_stop
        strings, nbytes, _ = ops.encode_bytes(e.z_hat, e.y_hat)
        _sync()
        enc_s, t0 = time.time() - t0, time.time()
        if method == "plain":
            xd = ops.base_decode(e.y_hat, e.z_hat)[:, :, :h, :w]
        else:
            xd, _, yt, _ = decode(ops, e.z_hat, e.y_hat)
            if (H, W) != (h, w):
                xd, ptag = pad_repair(ops, xd, (h, w), e.y_hat if yt is None else yt, e.y_hat, e.z_hat)
                tag = f"{tag}/pad{ptag}"
        xd = store01(xd)
        _sync()
        dec_s = time.time() - t0
        if not bool(torch.isfinite(xd).all()):
            log(f"{stem} round {gen}: non-finite decode, chain stopped")
            break
        m = {k: float(f(xd.clamp(0, 1).float(), x).item()) for k, f in metrics.items()}
        data = coder.encode(strings, h, w)
        z_dec, y_dec, hw = coder.decode(data)
        roundtrip = int(hw == (h, w) and torch.equal(z_dec, e.z_hat) and torch.equal(y_dec, e.y_hat))
        with open(os.path.join(out_dir, "F", f"g{gen:03d}.bin"), "wb") as f:
            f.write(data)
        md5 = hashlib.md5(data).hexdigest()
        cur = (e.z_hat, e.y_hat, data)
        same = lambda a, b: int(b is not None and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))  # noqa: E731
        r = dict(protocol=protocol, method=method, rate=backend.rate, image=stem, gen=gen, h=h, w=w,
                 bpp=nbytes * 8 / (h * w), file_bytes=len(data), md5_file=md5, roundtrip=roundtrip,
                 sym_eq_g1=1 if first is None else same(cur, first), sym_eq_prev="" if prev is None else same(cur, prev),
                 file_eq_g1=1 if first is None else int(data == first[2]),
                 file_eq_prev="" if prev is None else int(data == prev[2]),
                 z_changed="" if prev is None else int((e.z_hat != prev[0]).sum()),
                 y_changed="" if prev is None else int((e.y_hat != prev[1]).sum()),
                 stable=int(xd.dtype == xi.dtype and xd.shape == xi.shape and bool(torch.equal(xd, xi))),
                 enc_tag=tag, enc_s=round(enc_s, 3), dec_s=round(dec_s, 3))
        r.update({k: m.get(k, "") for k in ("psnr", "ms_ssim", "lpips", "dists")})
        rows.append(r)
        if gen in save_rounds:
            write_png16(os.path.join(out_dir, "recon", f"g{gen:03d}.png"), xd)
        log(f"{stem} round {gen}: bpp {r['bpp']:.5f} file {md5[:8]} psnr {m.get('psnr', float('nan')):.4f} "
            f"tag {tag} enc {enc_s:.1f}s dec {dec_s:.1f}s")
        if first is None:
            first = cur
        prev, xi = cur, xd
    with open(os.path.join(out_dir, "rows.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out_dir, "DONE"), "w") as f:
        f.write(f"{len(rows)} rounds\n")
    return rows
