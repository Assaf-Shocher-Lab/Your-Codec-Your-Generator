"""Clipped-float range repair (App. B.3): x_0 = W(y*, b), x_{k+1} = W(y*, b(Pi(x_k))) with Pi the clip to the valid
range shrunk inward by `range_guard` of its width, until x_k is inside the range or `range_repair_steps` (200) are
used."""
import torch

from adapted_codec.settings import S
from adapted_codec.target_search import select_target


def range_repair(write, x, lo, hi, U0=None):
    """Range repair of the written image x with the exact write `write` (x <- write(clip(x)) until in range)."""
    if S.range_repair_steps <= 0:
        return x
    e = S.range_guard * (hi - lo)
    n = 0
    while True:
        exc = torch.maximum(x - hi, lo - x)
        m = float(exc.max())
        if m <= 0 or n == S.range_repair_steps:
            break
        x = write(x.clamp(lo + e, hi - e))
        n += 1
    return x


@torch.no_grad()
def decode_clipped(ops, z_hat, y_hat):
    """D' for the clipped-float chain: the float decode followed by the range repair. Returns (x, x_raw, y*, it)."""
    yt, it = select_target(ops, z_hat, y_hat)
    xr = ops.base_decode(y_hat, z_hat)

    tgt = y_hat if yt is None else yt
    return range_repair(lambda v: ops.write(v, tgt), ops.write(xr, tgt), 0.0, 1.0, ops.wrapper.U[:ops.wrapper.k]), xr, yt, it
