"""FFHQ 256 x 256 test images of the NLBP ELIC FFHQ configurations (nlbp/configs/elic_ffhq_rate<r>.yaml): rows 0..N-1
of the first shard of the Hugging Face dataset bitmind/ffhq-256 (train-00000-of-00016.parquet, md5
e7f8b94f61c3c787d4a9fde2bf7ad265 of the file the reported runs used), written as <out>/<row:05d>.png. PNG bytes are
written unchanged; any other image format is decoded and saved as an RGB PNG. The test set is rows 0-999.

    python scripts/prepare_ffhq256.py --parquet train-00000-of-00016.parquet --out data/ffhq256 --n 1000

Writes <out>/MD5SUMS (md5 of each PNG file). Needs pyarrow."""
import argparse
import hashlib
import io
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    t = pq.read_table(a.parquet)
    col = [c for c in t.column_names if c.lower() in ("image", "img", "png")] or t.column_names[:1]
    os.makedirs(a.out, exist_ok=True)
    lines = []
    for i, v in enumerate(t.column(col[0]).slice(0, a.n).to_pylist()):
        b = v["bytes"] if isinstance(v, dict) else v
        p = os.path.join(a.out, f"{i:05d}.png")
        im = Image.open(io.BytesIO(b))
        if im.format == "PNG":
            with open(p, "wb") as f:
                f.write(b)
        else:
            im.convert("RGB").save(p)
        lines.append(f"{hashlib.md5(open(p, 'rb').read()).hexdigest()}  {os.path.basename(p)}")
    with open(os.path.join(a.out, "MD5SUMS"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} images written to {a.out}")


if __name__ == "__main__":
    main()
