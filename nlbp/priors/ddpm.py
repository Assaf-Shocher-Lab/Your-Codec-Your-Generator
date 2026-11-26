"""DDPM priors in the sampler's flow convention: x_s = (1 - s) x_0 + s eps; the DDPM state is
x_i = sqrt(abar_i) x_0 + sqrt(1 - abar_i) eps = x_s / sqrt((1 - s)^2 + s^2) with abar_i = abar_flow(s); linear betas
1e-4 .. 0.02, T = 1000. snap_ddpm: nearest integer training index, back to s."""
import math

import torch

_BETAS = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
_ABAR = torch.cumprod(1 - _BETAS, 0)


def s_to_index(s):
    nsr = s / (1 - s)
    ab = 1 / (1 + nsr ** 2)
    la = torch.log(_ABAR)
    lt = math.log(ab)
    if lt >= float(la[0]):
        return 0.0
    k = int(torch.searchsorted(-la, torch.tensor(-lt, dtype=torch.float64)))
    k = min(max(k, 1), 999)
    return (k - 1) + (float(la[k - 1]) - lt) / (float(la[k - 1]) - float(la[k]))


def index_to_s(i):
    ab = float(_ABAR[int(i)])
    nsr = math.sqrt((1 - ab) / ab)
    return nsr / (1 + nsr)


def snap_ddpm(s, shift=None):
    return index_to_s(max(0, round(s_to_index(s))))
