"""Target search over the margin-constrained feasible set F_m(s) (Sec. 3.2, App. B.2).

search_target: projected Adam inside the shrunk main-cell box |t - y_hat|_inf <= 1/2 - m_y (every iterate is projected
back), objective on the hyper constraint sum_j (v_j + 0.01 [v_j > 0])^2 with v_j = [|h_a(t)_j - z_hat_j| - (1/2 - m_z)]_+;
the first accepted iterate is refined toward the start (refinement.py).
select_target: start at the projection of E'(D(y_hat)) (search_start = "proj"); if that fails restart at y_hat; if both
fail, y_hat itself when its record is consistent, else no target."""
import torch

from adapted_codec.refinement import refine_target
from adapted_codec.settings import S


def search_target(ops, z_hat, y_hat, y_start=None):
    """y* in the shrunk cells of y_hat with Q_z(h_a(y*)) = z_hat; returns (y* or None, iterations)."""
    with torch.inference_mode(False), torch.enable_grad():
        yh, zh, off = y_hat.clone(), z_hat.clone(), ops.z_offset().clone()
        lo, hi = yh - 0.5 + S.margin_y, yh + 0.5 - S.margin_y
        k = zh - off
        ys = (yh.clone() if y_start is None
              else torch.minimum(torch.maximum(y_start.clone(), lo), hi)).requires_grad_(True)
        a = ys.detach().clone()
        opt = torch.optim.Adam([ys], lr=S.search_lr)
        for it in range(S.search_steps + 1):
            z = ops.hyper(ys)[0] - off
            viol = ((z - k).abs() - (0.5 - S.margin_z)).clamp_min(0)
            if float(viol.max()) == 0.0:
                with torch.no_grad():
                    y_ = ys.detach()
                    if torch.equal(ops.hyper(y_)[1], zh) and torch.equal(ops.quantize_main(y_, zh)[0], yh):
                        if S.refine_steps > 0 and y_start is not None:
                            with torch.enable_grad():
                                y_ = refine_target(ops, y_.clone(), a, lo, hi, off, k, zh, yh)
                        return y_.clone(), it
            if it == S.search_steps:
                break
            loss = (viol + 0.01 * (viol > 0)).pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                ys.copy_(torch.minimum(torch.maximum(ys, lo), hi))
    return None, S.search_steps


def select_target(ops, z_hat, y_hat):
    """The decoder's selection target y* (or y_hat, or None); returns (target, iterations)."""
    ys = None
    if S.search_start == "proj":
        with torch.no_grad():
            s = ops.wrapped_encode(ops.base_decode(y_hat, z_hat).float()).clone()
        ys, it = search_target(ops, z_hat, y_hat, s)
    if ys is None:
        ys, it = search_target(ops, z_hat, y_hat)
    if ys is None and torch.equal(ops.hyper(y_hat.clone())[1], z_hat):
        return y_hat.clone(), -1
    return ys, it
