"""Closed-loop 8-bit write with target correction (Sec. 3.3, App. B.3).

t_0 = y*, x_k = S_8(W(t_k, b(x_raw))), accepted iff Sigma(E'(x_k)) = s. With v_k = E'(x_k):
    r_y(v) = sign(v - y_hat) * [|v - y_hat| - (1/2 - gamma_y)]_+,
    L_z(v) = sum_j [|h_a(v)_j - z_hat_j| - (1/2 - gamma_z)]_+^2,
    t_{k+1} = t_k - r_y(v_k) - L_z(v_k) / ||grad L_z(v_k)||^2 * grad L_z(v_k)   (last term omitted if 0),
gradients at the realized latent, no differentiation through rounding; gamma_y = gamma_z = 0.02, at most
`correction_budget` updates. The round with the fewest mismatched symbols is kept and finished by the lattice search.
decode_once: target selection + this write (the stage-0 8-bit decode)."""
import torch

from adapted_codec.protocols.eight_bit.lattice_repair import lattice_search
from adapted_codec.protocols.storage import search_s8
from adapted_codec.settings import S
from adapted_codec.target_search import select_target


def corrected_write(ops, z_hat, y_hat, yt, xr):
    """Closed-loop write of the target yt on the raw decode xr, in the chain's storage format; returns the stored
    image of the best round (finished by the lattice search when no round passes).
    """
    from adapted_codec.protocols.storage import store01
    store01 = search_s8(store01)
    with torch.inference_mode(False), torch.enable_grad():
        yh, zh, off = y_hat.clone(), z_hat.clone(), ops.z_offset().clone()
        k = zh - off
        t, xr_ = yt.clone(), xr.clone()
        best = None
        for n in range(S.correction_budget + 1):
            with torch.no_grad():
                x = ops.write(xr_, t)
                xq = store01(x).clone()
                yq = ops.wrapped_encode(xq).clone()
                zq, zq_hat = ops.hyper(yq)
                nz = int((zq_hat != zh).sum())
                ny = int((ops.quantize_main(yq, zh)[0] != yh).sum())

                nyc = int(((yq - yh).abs() >= 0.5).sum())

                _bad = nz + (nyc if S.context_free_count else ny)
                if best is None or _bad < best[0]:
                    best = (_bad, xq.clone(), n)

                if nz == 0 and ny == 0:
                    break
                if n == S.correction_budget:
                    break
                over = ((yq - yh).abs() - (0.5 - S.gamma_y)).clamp_min(0) * torch.sign(yq - yh)
                step = -over
            yv = yq.detach().clone().requires_grad_(True)
            viol = ((ops.hyper(yv)[0] - off - k).abs() - (0.5 - S.gamma_z)).clamp_min(0)
            L = viol.pow(2).sum()
            if float(L) > 0:
                g, = torch.autograd.grad(L, yv)
                gg = float(g.pow(2).sum())
                if gg > 0:
                    step = step - (float(L) / gg) * g.detach()
            with torch.no_grad():
                t = t + step

    if best[0] > 0 and S.chain_format == "png8":
        xl, badl = lattice_search(ops, best[1], z_hat, y_hat)

        if S.context_free_count and badl > 0:
            with torch.no_grad():
                _yl = ops.wrapped_encode(xl)
                badl = int((ops.hyper(_yl)[1] != z_hat).sum()) + int(((_yl - y_hat).abs() >= 0.5).sum())

        if badl < best[0]:
            best = (badl, xl, best[2])

    return best[1]


def decode_once(ops, z_hat, y_hat):
    """Stage-0 decode: target selection and the closed-loop write (plain write when correction_budget = 0).
    Returns (x, x_raw, y*, it).
    """
    yt, it = select_target(ops, z_hat, y_hat)
    xr = ops.base_decode(y_hat, z_hat)

    if S.correction_budget > 0 and yt is not None:
        return corrected_write(ops, z_hat, y_hat, yt, xr), xr, yt, it

    return ops.write(xr, y_hat if yt is None else yt), xr, yt, it
