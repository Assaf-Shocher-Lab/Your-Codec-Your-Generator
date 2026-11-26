"""The ELIC clone (third_party/elic/src) is the pinned commit with patches/elic_decompress.patch applied: sha256 of
every file used (line endings normalized to LF, so a clone with git's autocrlf passes too), and the patched TestModel.decompress minus the fenced lines equals the upstream function (read from
the clone's git HEAD). Skipped when the clone is missing."""
import ast
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "third_party", "elic", "src")
SHA = {
    "Network.py": "8931f6e5b7c457e34aa9a23bdf88973df623f582df2cbdca3ef78f551216a95c",
    "ELICUtilis/layers/__init__.py": "d55722c3524f2b55b3e9deacd4388e26a90ba585adf3ba593d3a9df0563cc379",
    "ELICUtilis/layers/gdn.py": "878679e1f82ae3726f3ec41b93bd80557abcca82ff6ce178ea55b217aa1c74a6",
    "ELICUtilis/layers/layers.py": "2885c495292cf96265dea9754c64bfd1f1c72be667ab88847f7fdc92bcb27abf",
}
UPSTREAM_NETWORK = "9bcd23a1f7f668804c38a343923d2dbd49dd54573d5c2875b5055a59313c721a"


def _sha(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _decompress(src):
    t = ast.parse(src)
    c = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "TestModel")
    return ast.dump(next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "decompress"))


def _strip_fences(src):
    out, skip = [], False
    for ln in src.splitlines():
        if "# --- adapted codec: begin" in ln:
            skip = True
        elif "# --- adapted codec: end" in ln:
            skip = False
        elif not skip:
            out.append(ln)
    return "\n".join(out)


def main():
    if not os.path.isdir(SRC):
        print("SKIP: third_party/elic/src missing (third_party/elic/UPSTREAM.md)")
        return 0
    bad = 0
    for f, want in SHA.items():
        got = _sha(open(os.path.join(SRC, f), "rb").read())
        if got != want:
            print("FAIL sha256", f, got)
            bad += 1
    patched = open(os.path.join(SRC, "Network.py"), encoding="utf-8").read()
    try:
        head = subprocess.run(["git", "-C", SRC, "show", "HEAD:Network.py"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        head = None
    if head is not None:
        if _sha(head) != UPSTREAM_NETWORK:
            print("FAIL: the clone's HEAD is not the pinned commit")
            bad += 1
        elif _decompress(_strip_fences(patched)) != _decompress(head.decode("utf-8")):
            print("FAIL: the patched decompress differs from upstream outside the fenced lines")
            bad += 1
    print("PASS" if not bad else f"{bad} failures", "test_elic_patch")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
