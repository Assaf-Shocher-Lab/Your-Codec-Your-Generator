"""Per-symbol hyper repair (App. B.3): for up to 16 wrong hyper symbols per round (at most 12 rounds, 400 trials),
move the pixels with the largest |d z_j / d x| by one grey level against the violation; a trial is accepted only if the
number of wrong hyper symbols decreases and every main coordinate stays inside its cell (pair-aware variant: pixels that
push a newly broken symbol toward its wall are excluded and the move is retried)."""
import torch

from adapted_codec.protocols.storage import search_s8
from adapted_codec.settings import S


def repair_hyper_symbols(ops, x, zh, yh, off, k, gM):
    """Repair the wrong hyper symbols of the 8-bit image x one at a time. Returns (x, E'(x), #wrong hyper symbols)."""
    from adapted_codec.protocols.storage import store01
    store01 = search_s8(store01)
    lo_, hi_ = -0.5 / 255, 1 + 0.5 / 255

    def ev(x_):
        with torch.no_grad():
            y_ = ops.wrapped_encode(x_)
            zv_, zq_ = ops.hyper(y_)
            return y_, zv_, int((zq_ != zh).sum()), int(((y_ - yh).abs() >= 0.5).sum())

    y_, zv, nz, nyc = ev(x)
    tries, rnd = 0, 0
    while nz > 0 and rnd < 12 and tries < 400:
        rnd += 1
        u = (zv - off - k)
        wrong = ((u.abs() >= 0.5) | (ops.hyper(y_)[1] != zh)).flatten().nonzero().flatten().tolist()
        G = []
        for j in wrong[:16]:
            xv = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                zj = ops.hyper(ops.wrapped_encode(xv))[0].flatten()[j]
                g, = torch.autograd.grad(zj, xv)
            G.append(g.detach())
        ym = 0.5 - (y_ - yh).abs()
        prog = False
        for a_, j in enumerate(wrong[:16]):
            uj = float(u.flatten()[j])
            dj = max(abs(uj) - 0.5, 0.0)
            s = 1.0 if uj > 0 else -1.0
            g = G[a_]
            sg = torch.sign(g) * s
            xn = x - sg / 255.0
            okm = ((xn > lo_) & (xn < hi_) & (g != 0)).flatten()
            aa = g.abs().flatten() * okm
            srt, idx = torch.sort(aa, descending=True)
            nnz = int((srt > 0).sum())
            if nnz == 0:
                continue

            excl = torch.zeros_like(aa, dtype=torch.bool)
            u_old_bad = ((zv - off - k).abs() >= 0.5).flatten()
            for f in (1.0, 2.0, 4.0, 0.5, 0.25, 8.0):
              for pr in range(4):
                aa2 = aa * (~excl)
                srt2, idx2 = torch.sort(aa2, descending=True)
                nnz2 = int((srt2 > 0).sum())
                if nnz2 == 0:
                    break
                K02 = int(torch.searchsorted(torch.cumsum(srt2, 0) / 255.0, torch.tensor([dj + 1e-3], device=srt2.device,
                          dtype=srt2.dtype))) + 1
                K = max(1, min(nnz2, int(K02 * f)))
                tries += 1
                m = torch.zeros_like(aa)
                m[idx2[:K]] = 1
                xt = store01(x - (sg * m.view_as(x)) / 255.0).clone()
                yt_, zvt, nzt, nyct = ev(xt)
                if nzt < nz and nyct == 0:
                    x, y_, zv, nz, nyc = xt, yt_, zvt, nzt, nyct
                    prog = True
                    break
                if nyct != 0:
                    break
                ut = (zvt - off - k).flatten()
                Bn = (((ut.abs() >= 0.5) & ~u_old_bad).nonzero().flatten().tolist())[:8]
                if not Bn:
                    break
                dxp = (-sg).flatten()
                for b in Bn:
                    xv = x.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        zb = ops.hyper(ops.wrapped_encode(xv))[0].flatten()[b]
                        gb, = torch.autograd.grad(zb, xv)
                    sb = 1.0 if float(ut[b]) > 0 else -1.0
                    excl |= (sb * gb.flatten() * dxp > 0)
              if prog:
                  break

            if prog:
                break
        if not prog:
            break
    return x, y_, nz
