"""Escalation stage 3: projected image-space box step (App. B.3). Start x = clip(W(y*, b(x_raw)), 0, 1); minimize the
lattice merit M (guards gamma_y, gamma_z) over the pixel box [0, 1]^n by projected gradient with the full pull-back
J_E'(x)^T dM/dy, first step M / ||g_P||^2, halved until M decreases; stop at M = 0, after `box_steps` iterations or
when the step no longer changes x. Then S_8 and the unchanged lattice search."""
import torch

from adapted_codec.protocols.eight_bit.lattice_repair import lattice_search
from adapted_codec.protocols.storage import search_s8
from adapted_codec.settings import S


def box_step_decode(ops, z_hat, y_hat, yt, xr):
    """Projected image-space descent of the merit inside the pixel box, then S_8 and the lattice search."""
    from adapted_codec.protocols.storage import store01
    store01 = search_s8(store01)
    if S.chain_format != "png8" or S.gamma_y != 0.02 or S.gamma_z != 0.02:
        raise RuntimeError(f"box step needs png8 and gamma 0.02 / 0.02; got {S.chain_format} {S.gamma_y} {S.gamma_z}")
    with torch.inference_mode(False), torch.enable_grad():
        yh, zh, off = y_hat.clone(), z_hat.clone(), ops.z_offset().clone()
        k = zh - off

        def merit(x_):
            with torch.no_grad():
                y_ = ops.wrapped_encode(x_).clone()
            yv_ = y_.detach().clone().requires_grad_(True)
            zv, zqh = ops.hyper(yv_)
            vz = ((zv - off - k).abs() - (0.5 - S.gamma_z)).clamp_min(0)
            oy = ((yv_ - yh).abs() - (0.5 - S.gamma_y)).clamp_min(0)
            return (vz.pow(2).sum() + oy.pow(2).sum(), yv_, int((zqh != zh).sum()),
                    int(((y_ - yh).abs() >= 0.5).sum()))

        with torch.no_grad():
            xw = ops.write(xr.clone(), yt.clone())
            x = xw.clamp(0, 1).clone()
        M, yv, nz, nyc = merit(x)
        for it in range(S.box_steps):
            if float(M) == 0.0:
                break
            gy, = torch.autograd.grad(M, yv)
            xv = x.detach().clone().requires_grad_(True)
            gx, = torch.autograd.grad((ops.wrapped_encode(xv) * gy.detach()).sum(), xv)
            with torch.no_grad():
                gx = gx.detach().to(x.dtype)
                free = ~(((x <= 0) & (gx > 0)) | ((x >= 1) & (gx < 0)))
                gg = float((gx * free).double().pow(2).sum())
            if gg == 0.0:
                break
            al, acc = float(M) / gg, False
            while True:
                with torch.no_grad():
                    xt = (x - al * gx).clamp(0, 1)
                    if torch.equal(xt, x):
                        break
                Mt, yvt, nzt, nyct = merit(xt)
                if float(Mt) < float(M):
                    x, M, yv, nz, nyc, acc = xt, Mt, yvt, nzt, nyct, True
                    break
                al *= 0.5
            if not acc:
                break
        with torch.no_grad():
            xq = store01(x).clone()
            yq = ops.wrapped_encode(xq).clone()
            nzq = int((ops.hyper(yq)[1] != zh).sum())
            nyq = int((ops.quantize_main(yq, zh)[0] != yh).sum())
            nycq = int(((yq - yh).abs() >= 0.5).sum())
    bad = nzq + (nycq if S.context_free_count else nyq)
    best = (bad, xq)
    if nzq + nyq > 0:
        xl, badl = lattice_search(ops, xq, z_hat, y_hat)
        if S.context_free_count and badl > 0:
            with torch.no_grad():
                _yl = ops.wrapped_encode(xl)
                badl = int((ops.hyper(_yl)[1] != z_hat).sum()) + int(((_yl - y_hat).abs() >= 0.5).sum())
        if badl < best[0]:
            best = (badl, xl)
    return best[1]
