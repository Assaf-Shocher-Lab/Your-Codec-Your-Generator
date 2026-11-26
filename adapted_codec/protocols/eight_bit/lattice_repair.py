"""Lattice search on the 8-bit grid (App. B.3). Merit
    M(x) = sum_j [|h_a(E'(x))_j - z_hat_j| - (1/2 - eta_z)]_+^2 + sum_i [|E'(x)_i - y_hat_i| - (1/2 - eta_y)]_+^2,
eta_z = 0.003, eta_y = 0.005. Moves of one grey level against sign(dM/dx) on the K pixels with the largest |dM/dx|
(K from the first-order prediction, then smaller), single-pixel scan and coupled group moves; a move is accepted only if
the exactly evaluated merit decreases (lexicographic: never more wrong hyper symbols). The budget is a count of exactly
evaluated trials (`lattice_trials`, 9,056 per call); the image never leaves the 8-bit lattice."""
import torch

from adapted_codec.protocols.eight_bit.hyper_symbol_repair import repair_hyper_symbols
from adapted_codec.protocols.storage import search_s8
from adapted_codec.settings import S


def _lattice_better(Mt, nzt, M, nz):
    return nzt < nz or (nzt == nz and float(Mt) < float(M))


def _lattice_budget_left(n_try, t0):
    import time
    if time.time() - t0 > S.lattice_time_cap:
        raise RuntimeError(f"lattice search exceeded the safety cap lattice_time_cap={S.lattice_time_cap} s at n_try={n_try}")
    return n_try < S.lattice_trials


def lattice_search(ops, xq0, z_hat, y_hat):
    """Monotone descent of the merit M on the 8-bit lattice from xq0. Returns (image, #mismatched symbols)."""
    import time
    from adapted_codec.protocols.storage import store01
    store01 = search_s8(store01)
    t0 = time.time()
    d = ops.wrapper
    with torch.inference_mode(False), torch.enable_grad():
        c = d.c.view(1, -1, 1, 1).clone()
        yh, zh, off = y_hat.clone(), z_hat.clone(), ops.z_offset().clone()
        k = zh - off

        def merit(x_):
            with torch.no_grad():
                y_ = ops.wrapped_encode(x_).clone()
            yv_ = y_.detach().clone().requires_grad_(True)
            zv, zqh = ops.hyper(yv_)
            vz = ((zv - off - k).abs() - (0.5 - S.eta_z)).clamp_min(0)
            oy = ((yv_ - yh).abs() - (0.5 - S.eta_y)).clamp_min(0)

            return vz.pow(2).sum() + oy.pow(2).sum(), yv_, int((zqh != zh).sum()), y_

        x = xq0.clone()
        M, yv, nz, y_ = merit(x)
        n_try = 0
        for _ in range(S.lattice_steps):

            if float(M) == 0.0:
                break

            if not _lattice_budget_left(n_try, t0):
                break

            gy, = torch.autograd.grad(M, yv)
            gy = gy.detach()
            try:
                xv = x.detach().clone().requires_grad_(True)
                gx, = torch.autograd.grad((c * d.split(xv)[0] * gy).sum(), xv)
            except Exception:
                with torch.no_grad():
                    a0, b0 = d.split(x)
                    gx = d.join(c * gy, torch.zeros_like(b0)) - d.join(torch.zeros_like(a0), torch.zeros_like(b0))

            try:
                xv = x.detach().clone().requires_grad_(True)
                gx, = torch.autograd.grad((ops.wrapped_encode(xv) * gy).sum(), xv)
            except Exception as ex:
                pass

            with torch.no_grad():
                gx = gx.detach().to(x.dtype)
                sg = torch.sign(gx)
                xn = x - sg / 255.0
                okm = (xn > -0.5 / 255) & (xn < 1 + 0.5 / 255) & (gx != 0)
                a = (gx.abs() * okm).flatten()
                srt, idx = torch.sort(a, descending=True)
                nnz = int((srt > 0).sum())
                if nnz == 0:
                    break
                cs = torch.cumsum(srt, 0) / 255.0
                K = int(torch.searchsorted(cs, torch.tensor([2.0 * float(M)], device=cs.device, dtype=cs.dtype))) + 1
                K = max(1, min(K, nnz))
            acc = False

            while K >= 1 and _lattice_budget_left(n_try, t0):
                n_try += 1
                with torch.no_grad():
                    m = torch.zeros_like(a)
                    m[idx[:K]] = 1
                    xt = store01(x - (sg * m.view_as(x)) / 255.0).clone()
                Mt, yvt, nzt, yt_ = merit(xt)

                if _lattice_better(Mt, nzt, M, nz):
                    x, M, yv, nz, y_ = xt, Mt, yvt, nzt, yt_
                    acc = True
                    break
                if K == 1:
                    break
                K = max(1, K // 4)

            r = 1

            while not acc and nz > 0 and r < min(S.lattice_scan, nnz) and _lattice_budget_left(n_try, t0):
                n_try += 1
                with torch.no_grad():
                    m = torch.zeros_like(a)
                    m[idx[r]] = 1
                    xt = store01(x - (sg * m.view_as(x)) / 255.0).clone()
                Mt, yvt, nzt, yt_ = merit(xt)

                if _lattice_better(Mt, nzt, M, nz):
                    x, M, yv, nz, y_ = xt, Mt, yvt, nzt, yt_
                    acc = True
                r += 1

            for g in (2, 4, 8):
                s = 0
                while (not acc and nz > 0 and s + g <= min(S.lattice_group, nnz) and _lattice_budget_left(n_try, t0)):
                    n_try += 1
                    with torch.no_grad():
                        m = torch.zeros_like(a)
                        m[idx[s:s + g]] = 1
                        xt = store01(x - (sg * m.view_as(x)) / 255.0).clone()
                    Mt, yvt, nzt, yt_ = merit(xt)
                    if _lattice_better(Mt, nzt, M, nz):
                        x, M, yv, nz, y_ = xt, Mt, yvt, nzt, yt_
                        acc = True
                    s += g // 2

            if not acc:

                break

        if nz > 0:
            _gM = None
            try:
                gy, = torch.autograd.grad(M, yv)
                xv = x.detach().clone().requires_grad_(True)
                _gM, = torch.autograd.grad((ops.wrapped_encode(xv) * gy.detach()).sum(), xv)
            except Exception as ex:
                pass
            x, y_, nz = repair_hyper_symbols(ops, x, zh, yh, off, k, _gM)
            M, yv, nz, y_ = merit(x)

        with torch.no_grad():
            ny = int((ops.quantize_main(y_, zh)[0] != yh).sum())

    return x, nz + ny
