"""Every injected upstream copy equals the vendored upstream text once our fenced lines are removed and the
"# upstream:" lines restored (whitespace-only lines ignored)."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPIES = [("adapters/hpcm/upstream_copies.py", "third_party/hpcm"),
          ("adapters/aeic/upstream_copies.py", "third_party/aeic")]


def _restore(lines):
    out, skip = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("# --- adapted codec: begin"):
            skip = True
            continue
        if s.startswith("# --- adapted codec: end"):
            skip = False
            continue
        m = re.match(r"^(\s*)# upstream: (.*)$", ln)
        if m:
            out.append(m.group(1) + m.group(2))
        elif not skip:
            out.append(ln)
    return out


def check(copy_path, upstream_root):
    lines = open(os.path.join(ROOT, copy_path), encoding="utf-8").read().splitlines()
    n, i = 0, 0
    while i < len(lines):
        m = re.match(r"^# upstream copy: (\S+) L(\d+)-(\d+)", lines[i])
        if not m:
            i += 1
            continue
        j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("# end of upstream copy"))
        got = [ln.rstrip() for ln in _restore(lines[i + 1:j]) if ln.strip()]
        up = open(os.path.join(ROOT, upstream_root, m.group(1)), encoding="utf-8").read().splitlines()
        want = [ln.rstrip() for ln in up[int(m.group(2)) - 1:int(m.group(3))] if ln.strip()]
        assert got == want, f"{copy_path}: block {m.group(1)} L{m.group(2)}-{m.group(3)} differs from upstream"
        n += 1
        i = j + 1
    assert n > 0, copy_path
    return n


def test_upstream_copies():
    for c, u in COPIES:
        check(c, u)


if __name__ == "__main__":
    for c, u in COPIES:
        print(c, check(c, u), "blocks equal upstream")
