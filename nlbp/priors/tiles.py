"""Two overlapping 512 x 512 tiles on the long side of a 768 x 512 image ([0:512], [L-512:L]) and their linear blending
ramp across the overlap: the PixNerd-XXL/16 T2I prior sees its native 512 grid on every tile."""
import math

import torch


def axis_starts(L, T=512):
    if L <= 2 * T:
        return sorted({0, max(L - T, 0)})
    n = int(math.ceil((L - T) / (0.75 * T))) + 1
    return [int(round(k * (L - T) / (n - 1))) for k in range(n)]


def tiles512(H, W, T=512):
    return [(i, j) for i in axis_starts(H, T) for j in axis_starts(W, T)]


def ramp_weight(H, W, i0, j0, T, dev):
    def axis_many(L, a0):
        st = axis_starts(L, T)
        k = st.index(a0)
        w = torch.ones(T, device=dev)
        if k > 0:
            ov = st[k - 1] + T - a0
            w[:ov] = (torch.arange(ov, device=dev, dtype=torch.float32) + 0.5) / ov
        if k < len(st) - 1:
            ov = a0 + T - st[k + 1]
            w[T - ov:] = 1 - (torch.arange(ov, device=dev, dtype=torch.float32) + 0.5) / ov
        return w

    def axis(L, a0):
        if L > 2 * T:
            return axis_many(L, a0)
        w = torch.ones(T, device=dev)
        if L > T:
            ov = 2 * T - L
            if ov > 0:
                r = (torch.arange(ov, device=dev, dtype=torch.float32) + 0.5) / ov
                if a0 == 0:
                    w[T - ov:] = 1 - r
                else:
                    w[:ov] = r
        return w
    return axis(H, i0)[:, None] * axis(W, j0)[None, :]
