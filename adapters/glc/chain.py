"""GLC float chain (in memory; symbols and estimated rate, GLC has no entropy coder).

Round g: replicate pad to a multiple of 64 (Kodak: none); round 1 (and the plain codec) sends the record of E_q(x),
rounds >= 2 the record of E'(x); the symbol file is written and decoded back; D(y_hat) by GLC's generator; the adapted
decoder searches y* over the codeword regions from the received y_hat and writes x = W(y*, b(D(y_hat))); padded images
get the border repair. The next round encodes x unchanged (float protocol, no clamp). Images are in [-1, 1]; PSNR is
GLC's on [-1, 1] (adapters/glc/codec.py psnr), the other metrics on (x + 1) / 2."""
import csv
import hashlib
import os

import torch

from adapted_codec.pca_basis import with_k
from adapted_codec.png16 import write_png16
from adapted_codec.wrapper import PCAWrapper
from adapters.glc.codec import EQ, GLC, load_image, psnr, replicate_pad
from adapters.glc.symbols import pack, sha256, unpack, y_hat_from_symbols

K_WEAK = 256
PATCH = 16


class GlcBackend:
    """The GLC image model (one checkpoint for q = 0..3) and the wrapper of quality q."""

    def __init__(self, rate, pca, c, weights):
        self.rate, self.q, self.c = rate, int(rate), c
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.glc = GLC(weights, self.dev)
        self.eq = EQ(self.glc.net, self.q)
        p = torch.load(pca, map_location="cpu", weights_only=False)
        self.wrapper = PCAWrapper.from_pca(self.eq, with_k(p, K_WEAK), c=c, f=PATCH).to(self.dev)


def write(wrapper, x_raw, t):
    """W(t, b(x_raw)) = J((t - f(b)) / c, b)."""
    _, u1 = wrapper.split(x_raw)
    y1 = wrapper.join(torch.zeros(1, wrapper.k, *u1.shape[2:], device=x_raw.device), u1)
    return wrapper.join((t - wrapper.E(y1)) / wrapper.c.view(1, -1, 1, 1), u1)


@torch.no_grad()
def run_glc_chain(backend, image_path, out_dir, stem, rounds, method, metrics, save_rounds=(), log=print):
    glc, q, wrapper = backend.glc, backend.q, backend.wrapper
    os.makedirs(os.path.join(out_dir, "symbols"), exist_ok=True)
    x = load_image(image_path).to(backend.dev)
    h, w = x.shape[2:]
    H, W = (h + 63) // 64 * 64, (w + 63) // 64 * 64
    xi, prev, first, rows = x, None, None, []
    for gen in range(1, rounds + 1):
        xi = replicate_pad(xi, H, W)
        y = backend.eq(xi) if (method == "plain" or gen == 1) else wrapper.wrapped_encoder(xi)
        idx, zs = glc.z_of(y)
        y_q, y_hat, bits = glc.main(y, idx, zs)
        blob = pack(idx, y_q, q, h, w, H, W, zs)
        s_ = unpack(blob, glc.dev)
        zs_d = (1, glc.net.z_vq.emb_dim, s_["zshape"][2], s_["zshape"][3])
        y_hat_d, qs_d = y_hat_from_symbols(glc.net, s_["idx"], zs_d, s_["y_q"])
        xd = glc.D(y_hat_d, q)
        tag = "plain"
        if method != "plain":
            tgt, it = glc.search(s_["idx"], zs_d, s_["y_q"], y_hat_d, qs_d)
            tag = ("zsA_plain" if gen == 1 else "zsA_wrapped") + f"/ds{it if tgt is not None else 'FAIL'}"
            t = y_hat_d if tgt is None else tgt
            xd = write(wrapper, xd, t)
            if (H, W) != (h, w):
                qs_ = glc.main(y, idx, zs, with_qs=True)[3]
                xd, ptag = glc.pad_repair(wrapper, xd, (h, w), t, idx, zs, y_q, y_hat, qs_)
                tag += f"/pad{ptag}"
        xd = xd[:, :, :h, :w]
        if not bool(torch.isfinite(xd).all()):
            log(f"{stem} round {gen}: non-finite decode, chain stopped")
            break
        roundtrip = int(torch.equal(y_hat_d, y_hat) and torch.equal(s_["idx"], idx) and torch.equal(s_["y_q"], y_q))
        cur = (idx, y_q)
        same = lambda a, b: int(b is not None and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))  # noqa: E731
        x01, ref01 = ((xd.clamp(-1, 1) + 1) / 2).float(), (x + 1) / 2
        m = {k: float(f(x01, ref01).item()) for k, f in metrics.items() if k != "psnr"}
        with open(os.path.join(out_dir, "symbols", f"g{gen:03d}.glcs"), "wb") as f:
            f.write(blob)
        r = dict(protocol="float", method=method, rate=backend.rate, image=stem, gen=gen, h=h, w=w,
                 bpp=bits / h / w, symbols_bytes=len(blob), symbols_sha256=sha256(blob),
                 md5_symbols=hashlib.md5(blob).hexdigest(), roundtrip=roundtrip,
                 sym_eq_g1=1 if first is None else same(cur, first), sym_eq_prev="" if prev is None else same(cur, prev),
                 psnr=psnr(xd, x), enc_tag=tag)
        r.update({k: m.get(k, "") for k in ("ms_ssim", "lpips", "dists")})
        rows.append(r)
        if gen in save_rounds:
            write_png16(os.path.join(out_dir, "recon", f"g{gen:03d}.png"), x01)
        log(f"{stem} round {gen}: bpp {r['bpp']:.5f} symbols {r['md5_symbols'][:8]} psnr {r['psnr']:.4f} tag {tag}")
        if first is None:
            first = cur
        prev, xi = cur, xd
    with open(os.path.join(out_dir, "rows.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows
