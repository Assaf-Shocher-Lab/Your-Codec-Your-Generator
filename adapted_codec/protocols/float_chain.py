"""Float chain (App. A.2): the adapted decoder D'(s) = W(y*, b(D(y_hat))) is passed to the next encoder unclamped."""
import torch

from adapted_codec.target_search import select_target


@torch.no_grad()
def decode_float(ops, z_hat, y_hat):
    """D'(s): codec decode x_raw = D(y_hat), target y*, exact write W(y*, b(x_raw)). Returns (x, x_raw, y*, it)."""
    yt, it = select_target(ops, z_hat, y_hat)
    xr = ops.base_decode(y_hat, z_hat)
    return ops.write(xr, y_hat if yt is None else yt), xr, yt, it
