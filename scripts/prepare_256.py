"""256 x 256 inputs of the ELIC 256^2 comparison (configs/elic_256_8bit.yaml): centre crop of each 8-bit RGB image to
its short edge (offsets rounded up), then PIL bicubic resize to 256 x 256, saved as PNG and checked after reloading.

    python scripts/prepare_256.py --src data/kodak --out data/kodak256
    python scripts/prepare_256.py --src data/clic_pro_valid41 --out data/clic256

Writes <out>/<stem>.png and <out>/MD5SUMS (md5 of each PNG file)."""
import argparse
import hashlib
import math
import os
import sys

import numpy as np
from PIL import Image


def centre_crop_square(img):
    """The central s x s block, s = min(h, w); the left / top offsets are ceil((w - s) / 2), ceil((h - s) / 2)."""
    h, w = img.shape[:2]
    s = min(h, w)
    left, top = int(math.ceil((w - s) / 2)), int(math.ceil((h - s) / 2))
    return img[top:top + s, left:left + s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines, bad = [], 0
    for fn in sorted(f for f in os.listdir(a.src) if f.lower().endswith(".png")):
        x = np.asarray(Image.open(os.path.join(a.src, fn)).convert("RGB"))
        c = centre_crop_square(x)
        y = Image.fromarray(np.ascontiguousarray(c)).resize((256, 256), Image.BICUBIC)
        op = os.path.join(a.out, os.path.splitext(fn)[0] + ".png")
        y.save(op)
        r = np.asarray(Image.open(op).convert("RGB"))
        bad += int(not (r.shape == (256, 256, 3) and np.array_equal(r, np.asarray(y))))
        lines.append(f"{hashlib.md5(open(op, 'rb').read()).hexdigest()}  {os.path.basename(op)}")
    with open(os.path.join(a.out, "MD5SUMS"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} images, reload failures {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
