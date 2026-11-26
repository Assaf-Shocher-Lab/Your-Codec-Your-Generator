"""Compare smoke-run rows with the stored expectations per round: every md5 column of the expectation (compressed
file, stored image) must be equal, every PSNR column (psnr255 of the 8-bit file chain, psnr of the in-memory chains)
equal at the expectation's printed precision. `md5_pixels` = md5 of the raw H x W x 3 uint8 pixels of the stored PNG
of the round (the md5 the ELIC 8-bit reference rows store); `psnr` of such rows = psnr255 of the run;
`sha256_file` = sha256 of the round's file F/g###.bin (the hash the AEIC DIV2K float reference rows store). A column
`check_<name>` must equal the run's column <name> (e.g. check_fmt_pass of the NLBP rows).
    python scripts/compare_smoke.py tests/smoke_expectations/<file>.csv <runs root>"""
import csv
import glob
import os
import sys


def main(exp_path, root):
    bad = n = 0
    for e in csv.DictReader(open(exp_path)):
        hits = glob.glob(os.path.join(root, e["variant"], "*", e["rate"], e["image"], "rows.csv"))
        if not hits:
            print("MISSING", e["variant"], e["rate"], e["image"])
            bad += 1
            n += 1
            continue
        rows = {r["gen"]: r for r in csv.DictReader(open(hits[0]))}
        r = rows.get(e["gen"])
        if r is not None and "md5_pixels" in e:
            import hashlib
            import numpy as np
            from PIL import Image
            png = os.path.join(os.path.dirname(hits[0]), "png", f"g{int(e['gen']):03d}.png")
            r = dict(r, md5_pixels=hashlib.md5(np.array(Image.open(png).convert("RGB"), dtype=np.uint8).tobytes())
                     .hexdigest())
            if "psnr" in e and "psnr" not in r:
                r["psnr"] = r["psnr255"]
        if r is not None and "sha256_file" in e:
            import hashlib
            fb = os.path.join(os.path.dirname(hits[0]), "F", f"g{int(e['gen']):03d}.bin")
            r = dict(r, sha256_file=hashlib.sha256(open(fb, "rb").read()).hexdigest())
        md5s = [k for k in e if k.startswith(("md5_", "sha256_"))]
        psnrs = [k for k in e if k.startswith("psnr")]
        checks = [k for k in e if k.startswith("check_")]

        def same(a, b):
            nd = len(b.split(".")[1]) if "." in b else 0
            return round(float(a), nd) == float(b)
        ok = r is not None and all(r[k] == e[k] for k in md5s) and all(same(r[k], e[k]) for k in psnrs) and             all(r[k[6:]] == e[k] for k in checks)
        n += 1
        bad += not ok
        print("PASS" if ok else "FAIL", e["variant"], e["rate"], e["image"], "g" + e["gen"],
              "" if ok or r is None else " ".join(f"{k} {r[k]}" for k in md5s + psnrs + [c[6:] for c in checks]))
    print(f"{n - bad}/{n} rounds equal")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(*sys.argv[1:3]) else 0)
