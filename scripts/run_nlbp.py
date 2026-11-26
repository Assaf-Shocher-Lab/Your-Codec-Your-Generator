"""NLBP first decodes of one reported configuration (nlbp/configs/<config>.yaml); one row per image.

    python scripts/run_nlbp.py --config nlbp/configs/elic_ffhq_rate4.yaml --images 00000 --data data/ffhq256 \
        --out runs/nlbp
    python scripts/run_nlbp.py --config nlbp/configs/hpcm_kodak_pixnerd_s045.yaml --images kodim05 --data data/kodak \
        --out runs/nlbp

ELIC on FFHQ (8-bit storage), per image:
  sender     x = the 8-bit image mapped to [-1, 1]; round 1 sends the codec's own record Q(E(x)) as the E3M1 file
             <out>/<config>/nlbp/<rate>/<image>/nlbp_g1.bin
  receiver   reads only that file: entropy decoding -> (z_hat, y_hat); y* = the selection target under the float
             margins (adapters/elic/eight_bit.float_target); the state of nlbp/receiver.py (x_base = W(y*, b(D(y_hat))),
             the bitstream seed); the NLBP decode (nlbp/sampler.py) with the frozen prior; the 8-bit target y*_8 and
             the corrected 8-bit write of the NLBP output (adapted_codec/protocols/eight_bit/target_correction.py)
  storage    the PNG nlbp_g1.png (written, read back, compared)
  check      fmt_pass = Sigma(E'(x_png)) == the received record (the operational check of App. E.1)
rows.csv: payload bytes / bpp, md5 of the file and of the PNG, sym_eq_file (the file decodes to the sent record),
fmt_pass, PSNR of the PNG vs the original (images in [-1, 1], 4 decimals), times.

HPCM on Kodak (float storage), per image:
  sender     x = the 8-bit image mapped to [-1, 1] (evaluation/png_io.from_uint8); round 1 sends HPCM's own record
             Q(E(x)) as the HPCM file <out>/<config>/nlbp/<rate>/<image>/nlbp_g1.bin
  receiver   reads only that file: (z_hat, y_hat); y* = the selection target under the float margins of the HPCM float
             chain; the state of nlbp/receiver.py; the NLBP decode with the frozen prior (PixNerd-XXL/16 T2I in pixel
             space: nlbp_decode; Z-Image / FLUX.2 [dev] in latent space behind a child process: nlbp_decode_latent)
  storage    the float output x_out as the 8-bit PNG nlbp_g1.png of evaluation/png_io.py
  check      the float output re-encoded as round 2 (Q(E'(x_out))): full_sym_eq = same record, full_bytes_eq = same
             entropy-coded bytes (the re-encode check of App. E.1)
rows.csv: payload bytes / bpp, digest (sha256 of the two rANS streams, 16 hex), md5 of the file and of the PNG,
sym_eq_file, full_sym_eq, full_bytes_eq, psnr (PNG vs original) and psnr_f (float output clamped to [-1, 1]) at full
precision, times.
Weights, prior and data are not part of the repository (README, "Weights and data")."""
import argparse
import csv
import hashlib
import os
import socket
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapted_codec.settings import S  # noqa: E402

FIELDS = ["config", "rate", "image", "gen", "steps", "H", "W", "nbytes", "bpp", "md5_file", "md5_png_file",
          "sym_eq_file", "fmt_pass", "psnr", "enc_s", "dec_s", "write_s", "node", "gpu"]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _md5(b):
    return hashlib.md5(b).hexdigest()


def load_configs(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, cfg["codec_config"])) as f:
        ccfg = yaml.safe_load(f)
    if cfg["codec"] != ccfg["codec"]:
        raise SystemExit(f"{path}: codec {cfg['codec']} but its codec_config is for {ccfg['codec']}")
    return cfg, ccfg


def sampler_params(cfg):
    keys = ("variant", "target", "upsample", "prompt", "cfg", "t0", "stride", "min_bp", "lambda2", "lambda1", "kz",
            "opt_K", "opt_lr", "s0", "seed_off", "latent", "lat_up", "vae_residual", "n_steps", "min_bp_frac")
    return {k: cfg[k] for k in keys if k in cfg}


def run_elic_ffhq(cfg, ccfg, name, a):
    from adapted_codec.encoder import encode, symbol_record
    from adapted_codec.protocols.eight_bit.target_correction import corrected_write
    from adapted_codec.protocols.storage import png_bytes, to_u8
    from adapted_codec.target_search import select_target
    from adapters.elic.backend import ElicBackend
    from adapters.elic.eight_bit import entropy_decode, file_bytes, float_target, from_uint8, read_file, to01
    from adapters.elic.file_format import PAD_MULTIPLE
    from nlbp.codec_interface import NlbpCodec
    from nlbp.priors.ddpm import snap_ddpm
    from nlbp.priors.ffhq_dps import load_ffhq_dps_prior
    from nlbp.receiver import nlbp_state, psnr
    from nlbp.sampler import nlbp_decode, resolve

    if cfg["prior"] != "ffhq_10m" or cfg["storage"] != "8bit" or cfg["variant"] != "R1" or int(cfg["upsample"]) != 1:
        raise SystemExit("ELIC NLBP runner: prior ffhq_10m, 8-bit storage, variant R1, upsample 1 only")
    S.load(ccfg["settings"])
    rate = str(cfg["rate"])
    b = ElicBackend(rate, os.path.join(ROOT, ccfg["pca"]), float(ccfg["gain"]),
                    os.path.join(ROOT, ccfg["weights"][rate]))
    ops, dev = b.ops, b.dev
    codec = NlbpCodec(b.net, b.wrapper)
    prior = load_ffhq_dps_prior(os.path.join(ROOT, a.prior_repo or ccfg["prior_repo"]),
                                os.path.join(ROOT, a.prior_weights or ccfg["prior_weights"]), dev)
    snap, shift = snap_ddpm, float(cfg.get("shift", 1.0))
    prm = resolve(sampler_params(cfg), snap, shift)
    node = socket.gethostname()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    for stem in a.images.split(","):
        out = os.path.join(a.out, name, "nlbp", rate, stem)
        os.makedirs(out, exist_ok=True)
        u8o = torch.from_numpy(np.array(Image.open(os.path.join(a.data, stem + ".png")).convert("RGB"))) \
            .permute(2, 0, 1)[None]
        x = from_uint8(u8o).to(dev)
        h, w = x.shape[2:]
        if h % PAD_MULTIPLE or w % PAD_MULTIPLE:
            raise SystemExit(f"{stem}: {h}x{w} not a multiple of {PAD_MULTIPLE} (no pad path in this runner)")
        with torch.no_grad():
            _sync()
            t1 = time.time()
            e = encode(ops, to01(x), gen=1)
            strings, nbytes, ok = ops.encode_bytes(e.z_hat, e.y_hat)
            if not ok:
                raise RuntimeError(f"{stem}: entropy coding does not decode back to the record")
            data = file_bytes((h, w), e.z_hat, strings)
            fpath = os.path.join(out, "nlbp_g1.bin")
            with open(fpath, "wb") as f:
                f.write(data)
            y_s, z_s = e.y_hat.clone(), e.z_hat.clone()
            del e
            _sync()
            enc_s = time.time() - t1
            t1 = time.time()
            hw, zshape, ys_, zs_, _ = read_file(fpath)
            yd, zd = entropy_decode(ops, zshape, ys_, zs_)
            sym_file = int(torch.equal(yd, y_s) and torch.equal(zd, z_s))
            yt = float_target(ops, zd, yd)
            st = nlbp_state(stem, SimpleNamespace(y_hat=yd, z_hat=zd, y_t=yt), codec)
            v, n_steps = nlbp_decode(st, prm, prior, codec, snap, shift, dev, opt_lr=float(ccfg["opt_lr"]))
            _sync()
            dec_s = time.time() - t1
            if not bool(torch.isfinite(v).all()):
                raise RuntimeError(f"{stem}: non-finite NLBP output")
            t1 = time.time()
            yt8 = select_target(ops, zd, yd)[0]
            if yt8 is None:
                yt8 = yt
            xq = corrected_write(ops, zd, yd, yt8, (v.float() + 1) / 2)
            pb = png_bytes(to_u8(xq)[0])
            ppath = os.path.join(out, "nlbp_g1.png")
            with open(ppath, "wb") as f:
                f.write(pb)
            b8 = np.array(Image.open(ppath).convert("RGB"), dtype=np.uint8)
            if not np.array_equal(b8, to_u8(xq)[0]):
                raise RuntimeError("PNG file round trip changed pixels")
            x_png = from_uint8(torch.from_numpy(b8).permute(2, 0, 1)[None]).to(dev)
            q0 = symbol_record(ops, ops.wrapped_encode((x_png.float() + 1) / 2))
            fmt_pass = int(torch.equal(q0.z_hat, zd) and torch.equal(q0.y_hat, yd))
            write_s = time.time() - t1
            p = torch.from_numpy(b8).permute(2, 0, 1)[None].float().div(255).to(dev)
            ps = psnr(p * 2 - 1, x)
        row = dict(config=name, rate=rate, image=stem, gen=1, steps=n_steps, H=h, W=w, nbytes=nbytes,
                   bpp=f"{nbytes * 8 / (h * w):.6f}", md5_file=_md5(data), md5_png_file=_md5(pb), sym_eq_file=sym_file,
                   fmt_pass=fmt_pass, psnr=f"{ps:.4f}", enc_s=f"{enc_s:.2f}", dec_s=f"{dec_s:.2f}",
                   write_s=f"{write_s:.2f}", node=node, gpu=gpu)
        with open(os.path.join(out, "rows.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerow(row)
        print("ROW", row, flush=True)


HPCM_FIELDS = ["config", "rate", "image", "gen", "steps", "H", "W", "nbytes", "bpp", "digest", "md5_file",
               "md5_png_file", "sym_eq_file", "full_sym_eq", "full_bytes_eq", "re_digest", "psnr", "psnr_f", "enc_s",
               "dec_s", "check_s", "node", "gpu"]


def _digest(strings):
    return hashlib.sha256(b"".join(s for part in strings for s in part)).hexdigest()[:16]


def load_prior(cfg, ccfg, dev):
    """(prior, client): the pixel prior of nlbp_decode, or the child-process client of nlbp_decode_latent."""
    pri = ccfg["priors"]
    if int(cfg.get("latent", 0)):
        from nlbp.priors.latent_client import start_prior
        return None, start_prior(cfg["prior"], os.path.join(ROOT, pri["latent_models"]))
    if cfg["prior"] != "pixnerd_xxl16_t2i" or cfg["prompt"] != "empty" or float(cfg["cfg"]) != 1.0:
        raise SystemExit("HPCM NLBP runner: pixel prior pixnerd_xxl16_t2i with the empty prompt and cfg 1 only")
    from nlbp.priors.pixnerd_t2i import load_pixnerd_t2i_flow_prior
    return load_pixnerd_t2i_flow_prior(dev, os.path.join(ROOT, pri["pixnerd_xxl16_t2i"]),
                                       os.path.join(ROOT, pri["qwen3_1p7b"]), prompt="", cfg=1.0, bf16=False), None


def run_hpcm_kodak(cfg, ccfg, name, a):
    from adapted_codec.encoder import encode
    from adapted_codec.target_search import select_target
    from adapters.hpcm.adapter import HpcmBackend
    from evaluation.png_io import from_uint8, save_png, to_uint8
    from nlbp.codec_interface import NlbpCodec, to01
    from nlbp.priors.ddpm import snap_ddpm
    from nlbp.receiver import nlbp_state, psnr
    from nlbp.sampler import nlbp_decode, nlbp_decode_latent, resolve
    from nlbp.schedule import snap_s

    if cfg["storage"] != "float" or cfg["variant"] != "R1" or int(cfg["upsample"]) != 1:
        raise SystemExit("HPCM NLBP runner: float storage, variant R1, upsample 1 only")
    S.load(ccfg["settings"])
    rate = str(cfg["rate"])
    b = HpcmBackend(rate, os.path.join(ROOT, ccfg["pca"]), float(ccfg["gain"]),
                    os.path.join(ROOT, ccfg["weights"][rate]))
    ops, dev, coder = b.ops, b.dev, b.coder
    codec = NlbpCodec(b.net, b.wrapper)
    prior, client = load_prior(cfg, ccfg, dev)
    snap = snap_ddpm if cfg["snap"] == "ddpm" else snap_s
    shift = float(cfg["shift"])
    prm = resolve(sampler_params(cfg), snap, shift)
    opt_lr = float(cfg.get("opt_lr", ccfg["opt_lr"]))
    node = socket.gethostname()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    for stem in a.images.split(","):
        out = os.path.join(a.out, name, "nlbp", rate, stem)
        os.makedirs(out, exist_ok=True)
        u8 = torch.from_numpy(np.array(Image.open(os.path.join(a.data, stem + ".png")).convert("RGB"))) \
            .permute(2, 0, 1)[None]
        x = from_uint8(u8).to(dev)
        h, w = x.shape[2:]
        if h % b.pm or w % b.pm:
            raise SystemExit(f"{stem}: {h}x{w} not a multiple of {b.pm} (no pad path in this runner)")
        S.image_hw = (h, w)
        with torch.no_grad():
            _sync()
            t1 = time.time()
            e = encode(ops, to01(x), gen=1)
            strings, nbytes, ok = ops.encode_bytes(e.z_hat, e.y_hat)
            if not ok:
                raise RuntimeError(f"{stem}: entropy coding does not decode back to the record")
            data = coder.encode(strings, h, w)
            fpath = os.path.join(out, "nlbp_g1.bin")
            with open(fpath, "wb") as f:
                f.write(data)
            y_s, z_s = e.y_hat.clone(), e.z_hat.clone()
            del e
            _sync()
            enc_s = time.time() - t1
            t1 = time.time()
            with open(fpath, "rb") as f:
                zd, yd, hw = coder.decode(f.read())
            sym_file = int(hw == (h, w) and torch.equal(yd, y_s) and torch.equal(zd, z_s))
            yt = select_target(ops, zd, yd)[0]
            if yt is None:
                yt = yd.clone()
            st = nlbp_state(stem, SimpleNamespace(y_hat=yd, z_hat=zd, y_t=yt), codec)
            if client is None:
                v, n_steps = nlbp_decode(st, prm, prior, codec, snap, shift, dev, opt_lr=opt_lr)
            else:
                v, n_steps = nlbp_decode_latent(st, prm, client, codec, dev, opt_lr=opt_lr)
            _sync()
            dec_s = time.time() - t1
            if not bool(torch.isfinite(v).all()):
                raise RuntimeError(f"{stem}: non-finite NLBP output")
            t1 = time.time()
            e2 = encode(ops, to01(v.float()), gen=2)
            s2, _, _ = ops.encode_bytes(e2.z_hat, e2.y_hat)
            full_sym = int(torch.equal(e2.z_hat, zd) and torch.equal(e2.y_hat, yd))
            full_bytes = int(s2 == strings)
            _sync()
            check_s = time.time() - t1
            ppath = os.path.join(out, "nlbp_g1.png")
            save_png(v, ppath)
            with open(ppath, "rb") as f:
                pb = f.read()
            p = to_uint8(v).float().div(255).to(dev)
            ps, ps_f = psnr(p * 2 - 1, x), psnr(v.clamp(-1, 1), x)
        row = dict(config=name, rate=rate, image=stem, gen=1, steps=n_steps, H=h, W=w, nbytes=nbytes,
                   bpp=repr(nbytes * 8 / (h * w)), digest=_digest(strings), md5_file=_md5(data), md5_png_file=_md5(pb),
                   sym_eq_file=sym_file, full_sym_eq=full_sym, full_bytes_eq=full_bytes, re_digest=_digest(s2),
                   psnr=repr(ps), psnr_f=repr(ps_f), enc_s=f"{enc_s:.2f}", dec_s=f"{dec_s:.2f}",
                   check_s=f"{check_s:.2f}", node=node, gpu=gpu)
        with open(os.path.join(out, "rows.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=HPCM_FIELDS)
            wr.writeheader()
            wr.writerow(row)
        print("ROW", row, flush=True)
    if client is not None:
        client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="nlbp/configs/<config>.yaml")
    ap.add_argument("--images", required=True, help="comma-separated image stems")
    ap.add_argument("--data", required=True, help="folder with <stem>.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prior-repo", dest="prior_repo", default=None, help="override the codec config's prior_repo")
    ap.add_argument("--prior-weights", dest="prior_weights", default=None, help="override prior_weights")
    a = ap.parse_args()
    cfg, ccfg = load_configs(a.config)
    name = os.path.splitext(os.path.basename(a.config))[0]
    if cfg["codec"] == "elic":
        return run_elic_ffhq(cfg, ccfg, name, a)
    if cfg["codec"] == "hpcm":
        return run_hpcm_kodak(cfg, ccfg, name, a)
    raise SystemExit(f"codec {cfg['codec']!r}: no NLBP runner")


if __name__ == "__main__":
    main()
