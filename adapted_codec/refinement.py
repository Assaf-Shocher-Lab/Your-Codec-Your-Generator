"""Refinement of a feasible target toward the projected start (App. B.2): 12 bisection steps on the segment
start -> y_f, then `refine_steps` bounded Adam steps on ||t - start||^2 + w * sum (v + 0.01 [v > 0])^2 with an extra
hyper-cell guard of 0.01; the nearest accepted iterate is kept."""
import torch

from adapted_codec.acceptance import is_feasible
from adapted_codec.settings import S


def refine_target(ops, y_f, a, lo, hi, off, k, zh, yh):
    """The accepted point nearest the start `a`: bisection a -> y_f, then `refine_steps` bounded Adam steps."""
    with torch.no_grad():
        best, t_lo, t_hi = y_f.clone(), 0.0, 1.0
        for _ in range(12):
            t = 0.5 * (t_lo + t_hi)
            y_ = a + t * (y_f - a)
            if is_feasible(ops, y_, off, k, zh, yh):
                best, t_hi = y_.clone(), t
            else:
                t_lo = t
        bd = float((best - a).pow(2).sum())
    ys = best.clone().requires_grad_(True)
    opt = torch.optim.Adam([ys], lr=S.refine_lr)
    for _ in range(S.refine_steps):
        z = ops.hyper(ys)[0] - off
        viol = ((z - k).abs() - (0.5 - S.margin_z - 0.01)).clamp_min(0)
        loss = (ys - a).pow(2).sum() + S.refine_weight * (viol + 0.01 * (viol > 0)).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            ys.copy_(torch.minimum(torch.maximum(ys, lo), hi))
            d = float((ys - a).pow(2).sum())
            if d < bd and is_feasible(ops, ys.detach(), off, k, zh, yh):
                best, bd = ys.detach().clone(), d
    return best
