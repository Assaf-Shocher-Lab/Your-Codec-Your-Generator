"""Score stored 8-bit chains from their files only (no codec run): per image and round the paired metrics of the stored
PNG against the original file and the payload bpp of the compressed file; per (method, rate, round) the patch FID / KID
over all images of the set.

    python evaluation/score_chains.py --chains runs/hpcm_8bit --orig-dir data/kodak --rounds 1,50 \
        --codec hpcm --protocol kodak64 --out runs/score

Layout read: <chains>/<method>/<rate>/<image>/png/g###.png and F/g###.bin (adapted_codec/chain.py).
Payload = the entropy-coded streams of the file without its framing (the codec's file format), bpp = 8 |payload| / (h w).
FID / KID: float16 features (the reported values) and float32 features; images in sorted order.
Outputs: per_image.csv, set.csv."""
import argparse
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapted_codec.chain import load_png_u8  # noqa: E402

# bytes of file framing before the payload, per codec file format (adapters/<codec>/)
HEADER_BYTES = {"hpcm": 12, "hyperprior": 20, "mean_scale": 20}


def payload(codec, data):
    if codec not in HEADER_BYTES:
        raise SystemExit(f"payload slice of {codec!r} files not defined")
    return data[HEADER_BYTES[codec]:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", required=True)
    ap.add_argument("--orig-dir", required=True)
    ap.add_argument("--rounds", default="1,50")
    ap.add_argument("--codec", required=True)
    ap.add_argument("--protocol", default="kodak64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fid", action="store_true")
    a = ap.parse_args()
    import torch

    from evaluation.patch_metrics import ImageScorer, fid_kid_both
    rounds = [int(g) for g in a.rounds.split(",")]
    scorer = ImageScorer()
    per, real, fake = [], {}, {}
    for d in sorted(glob.glob(os.path.join(a.chains, "*", "*", "*"))):
        if not os.path.isdir(os.path.join(d, "png")):
            continue
        method, rate, image = d.split(os.sep)[-3:]
        x0 = load_png_u8(os.path.join(a.orig_dir, image + ".png"))
        h, w = x0.shape[:2]
        for g in rounds:
            pp, fp = os.path.join(d, "png", f"g{g:03d}.png"), os.path.join(d, "F", f"g{g:03d}.bin")
            if not os.path.exists(pp):
                continue
            x = load_png_u8(pp)
            row = dict(method=method, rate=rate, image=image, gen=g, H=h, W=w)
            if os.path.exists(fp):
                with open(fp, "rb") as f:
                    data = f.read()
                n = len(payload(a.codec, data))
                row.update(file_bytes=len(data), payload_bytes=n, bpp_payload=8 * n / (h * w))
            row.update(scorer.score(x, x0))
            per.append(row)
            if not a.no_fid:
                if image not in real:
                    real[image] = scorer.patch_features(x0, a.protocol, real=True)
                fake.setdefault((method, rate, g), {})[image] = scorer.patch_features(x, a.protocol, ref_u8=x0)
        print(f"scored {method} {rate} {image}", flush=True)
    sets = []
    for (method, rate, g), ff in sorted(fake.items()):
        imgs = sorted(ff)
        r_ = torch.cat([real[i] for i in imgs])
        f_ = torch.cat([ff[i] for i in imgs])
        (f32, k32, ks32), (f16, k16, ks16) = fid_kid_both(r_, f_)
        sets.append(dict(method=method, rate=rate, gen=g, n_images=len(imgs), n_patches=len(f_), protocol=a.protocol,
                         fid_fp16=f16, kid_fp16=k16, kid_std_fp16=ks16, fid_fp32=f32, kid_fp32=k32, kid_std_fp32=ks32))
    os.makedirs(a.out, exist_ok=True)
    for name, rows in (("per_image.csv", per), ("set.csv", sets)):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(a.out, name), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=keys)
                wr.writeheader()
                wr.writerows(rows)
    print(f"{len(per)} image rows, {len(sets)} set rows -> {a.out}")


if __name__ == "__main__":
    main()
