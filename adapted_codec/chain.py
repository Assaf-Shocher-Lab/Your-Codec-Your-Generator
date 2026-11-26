"""The recompression file chain (App. A.2), identical for every codec and method.

One round g:
  x_{g-1}: uint8 RGB array read from a PNG file (g = 1: the original image)
  adapter.encode_to_file(x_{g-1}, g, F_g)   the codec's encoder writes the compressed file (round 1: base record,
                                            rounds >= 2: wrapped record, adapted_codec/encoder.py)
  adapter.decode_from_file(F_g)             a fresh decode that reads only F_g
  u8 = rint(clip(x, 0, 1) * 255) (images in [-1, 1] are mapped by (x + 1) / 2 first), written as PNG, read back and
  checked -> x_g
Per round: md5 of the symbols, the entropy-coded payload, the file and the stored image; equality with round 1 and with
the previous round; payload and file bpp; times. Outputs per chain: rows.csv, F/g###.bin, png/g###.png, sym/g###.md5,
DONE. Every round runs the encoder and the decoder."""
import csv
import hashlib
import io
import os
import socket
import time

import numpy as np
from PIL import Image

PROTOCOL = "file-chain"
RUNNER_VERSION = "chain v1"

FIELDS = [
    "protocol", "runner", "method", "rate", "image", "gen", "computed", "attempt", "H", "W",
    "file_bytes", "bitstream_bytes", "bpp", "bpp_bitstream", "bpp_est",
    "md5_sym", "md5_bits", "md5_file", "md5_img", "md5_png_file",
    "sym_eq_g1", "sym_eq_prev", "bits_eq_g1", "bits_eq_prev", "file_eq_g1", "file_eq_prev", "img_eq_g1", "img_eq_prev",
    "png_roundtrip", "fresh_decode", "on_grid", "psnr255", "enc_s", "dec_s", "png_s", "node", "gpu", "job_id",
    "enc_tag", "dec_diag", "png_saved",
]
INRUN_FIELDS = ["m_psnr", "m_ms_ssim", "m_lpips", "m_dists"]


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def md5_arrays(arrays):
    h = hashlib.md5()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def load_png_u8(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def to_u8(x01):
    a = np.asarray(x01)
    if a.dtype != np.float32:
        raise TypeError(f"to_u8 needs float32 (the _png8 rule), got {a.dtype}")
    return np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


def decoded_to_x01(x, value_range):
    if hasattr(x, "detach"):
        import torch
        t = x.detach().float()
        if t.dim() == 4:
            if t.shape[0] != 1:
                raise ValueError("one image per decode")
            t = t[0]
        if value_range == "m11":
            t = (t + 1) / 2
        elif value_range != "01":
            raise ValueError(value_range)
        t = t.clamp(0, 1).cpu()
        return t.permute(1, 2, 0).numpy().astype(np.float32, copy=False)
    a = np.asarray(x, dtype=np.float32)
    if value_range == "m11":
        a = ((a + np.float32(1)) / np.float32(2)).astype(np.float32)
    elif value_range != "01":
        raise ValueError(value_range)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def png_roundtrip(u8, path=None):
    if u8.dtype != np.uint8 or u8.ndim != 3 or u8.shape[2] != 3:
        raise ValueError("png_roundtrip needs uint8 H x W x 3")
    buf = io.BytesIO()
    Image.fromarray(u8, mode="RGB").save(buf, format="PNG")
    data = buf.getvalue()
    if path is not None:
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        back = load_png_u8(path)
    else:
        back = np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    if back.shape != u8.shape or not np.array_equal(back, u8):
        raise RuntimeError("PNG round trip changed pixels")
    return back, data


def psnr255(a_u8, b_u8):
    d = a_u8.astype(np.float64) - b_u8.astype(np.float64)
    mse = float(np.mean(d * d))
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


class Decoded:
    def __init__(self, x, value_range, symbols, diag=""):
        self.x, self.value_range, self.symbols, self.diag = x, value_range, symbols, diag


class Adapter:
    name = rate = "?"

    def bitstream_of(self, data):
        return data

    def assert_no_encoder_state(self):
        pass

    def device_name(self):
        return "cpu"


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def chain_done(out_dir, generations):
    p = os.path.join(out_dir, "DONE")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        return f.read().split()[:1] == [str(generations)]


def run_chain(adapter, src_png, out_dir, image, generations, save_png=None, log=print, inrun=None):
    if chain_done(out_dir, generations):
        return
    attempt = 1
    ap = os.path.join(out_dir, "ATTEMPTS")
    if os.path.exists(ap):
        with open(ap) as f:
            attempt = int(f.read().strip() or 0) + 1

    os.makedirs(out_dir, exist_ok=True)
    stale_rows = 0
    rp = os.path.join(out_dir, "rows.csv")
    if os.path.exists(rp):
        with open(rp) as f:
            stale_rows = max(0, sum(1 for _ in f) - 1)
    stale = [n for n in ("rows.csv", "F", "png", "sym", "feat") if os.path.exists(os.path.join(out_dir, n))]
    if stale:
        import shutil
        for n in stale:
            p = os.path.join(out_dir, n)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    with open(os.path.join(out_dir, "RESTARTS.log"), "a") as f:
        f.write(f"{time.strftime('%F %T')} attempt {attempt} job {os.environ.get('SLURM_JOB_ID', '-')} restart_count "
                f"{os.environ.get('SLURM_RESTART_COUNT', '0')}: restart at g=1 in a clean folder; discarded "
                f"{stale if stale else 'nothing'} ({stale_rows} rows of attempt {attempt - 1})\n")

    for d in ("F", "png", "sym"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    with open(ap, "w") as f:
        f.write(str(attempt))
    rows_path = os.path.join(out_dir, "rows.csv")
    node, gpu, job = socket.gethostname(), adapter.device_name(), os.environ.get("SLURM_JOB_ID", "-")
    x_prev = load_png_u8(src_png)
    x_orig = x_prev.copy()
    H, W = x_prev.shape[:2]
    first = {}
    prev = {}
    with open(rows_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS + (INRUN_FIELDS if inrun is not None else []))
        wr.writeheader()
        for g in range(1, generations + 1):
            fpath = os.path.join(out_dir, "F", f"g{g:03d}.bin")
            ppath = os.path.join(out_dir, "png", f"g{g:03d}.png")
            if os.path.exists(fpath):
                os.remove(fpath)

            _sync()
            t0 = time.perf_counter()
            meta = adapter.encode_to_file(x_prev, g, fpath)
            _sync()
            enc_s = time.perf_counter() - t0
            enc_fields = {k: meta.get(k, "") for k in ("enc_tag", "bpp_est")}
            del meta
            adapter.assert_no_encoder_state()
            if not os.path.isfile(fpath):
                raise RuntimeError(f"g{g}: the encoder wrote no file {fpath}")

            _sync()
            t0 = time.perf_counter()
            dec = adapter.decode_from_file(fpath)
            x01 = decoded_to_x01(dec.x, dec.value_range)
            _sync()
            dec_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            u8 = to_u8(x01)
            if u8.shape != (H, W, 3):
                raise RuntimeError(f"g{g}: decoded shape {u8.shape} != original {(H, W, 3)}")
            keep = save_png is None or g in save_png
            x_g, png_bytes = png_roundtrip(u8, ppath)
            if not keep:
                os.remove(ppath)
            png_s = time.perf_counter() - t0

            with open(fpath, "rb") as f:
                data = f.read()
            bits = adapter.bitstream_of(data)
            m = dict(sym=md5_arrays(dec.symbols), bits=md5_bytes(bits), file=md5_bytes(data),
                     img=md5_arrays([x_g]))
            with open(os.path.join(out_dir, "sym", f"g{g:03d}.md5"), "w") as f:
                f.write(f"{m['sym']}  symbols ({len(dec.symbols)} arrays: "
                        + ", ".join(f"{a.dtype}{tuple(a.shape)}" for a in dec.symbols) + ")\n")
            if g == 1:
                first = dict(m)
            row = dict(protocol=PROTOCOL, runner=RUNNER_VERSION, method=adapter.name, rate=adapter.rate, image=image,
                       gen=g, computed=1, attempt=attempt, H=H, W=W, file_bytes=len(data), bitstream_bytes=len(bits),
                       bpp=round(8 * len(data) / (H * W), 7), bpp_bitstream=round(8 * len(bits) / (H * W), 7),
                       bpp_est=enc_fields["bpp_est"],
                       md5_sym=m["sym"], md5_bits=m["bits"], md5_file=m["file"], md5_img=m["img"],
                       md5_png_file=md5_bytes(png_bytes),
                       png_roundtrip=1, fresh_decode=1, on_grid=1, psnr255=round(psnr255(x_g, x_orig), 6),
                       enc_s=round(enc_s, 4), dec_s=round(dec_s, 4), png_s=round(png_s, 4), node=node, gpu=gpu,
                       job_id=job, enc_tag=enc_fields["enc_tag"], dec_diag=dec.diag, png_saved=int(keep))
            for lv in ("sym", "bits", "file", "img"):
                row[f"{lv}_eq_g1"] = int(m[lv] == first[lv])
                row[f"{lv}_eq_prev"] = "" if g == 1 else int(m[lv] == prev[lv])
            if not (enc_s > 0 and dec_s > 0):
                raise RuntimeError(f"g{g}: non-positive time enc {enc_s} dec {dec_s}")
            if inrun is not None:
                row.update({k: v for k, v in inrun(x_g, x_orig, g, out_dir).items() if k in INRUN_FIELDS})
            wr.writerow(row)
            fh.flush()
            prev = m
            del dec, x01, u8
            x_prev = x_g
    with open(os.path.join(out_dir, "DONE"), "w") as f:
        f.write(f"{generations} generations, attempt {attempt}, {time.strftime('%F %T')} {node}\n")
