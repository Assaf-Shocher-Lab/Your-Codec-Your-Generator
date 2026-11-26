"""Time grid of the NLBP sampler (Algorithm "NLBP decoding"). Flow convention: x_s = (1 - s) x_0 + s eps, s in (0, 1];
a DDPM index i maps to s with abar_flow(s) = abar_i (linear betas 1e-4 .. 0.02, T = 1000). snap maps a time to the
nearest training time of the prior (snap_s for shifted flow priors, priors/ddpm.snap_ddpm for DDPM priors). t0_of_s:
the DDPM start index whose snapped time is nearest the start noise s0 = omega. seed_from_bitstream: the sampler's seed
from the received symbols (the same bitstream gives the same decode)."""
import hashlib
import math

MINBP_FRAC = 354 / 550           # the early / late switch of lambda_t as a fraction of t0


def shift_fn(u, shift):
    return shift * u / (1 + (shift - 1) * u)


def unshift_fn(s, shift):
    return s / (shift - (shift - 1) * s)


def snap_s(s, shift, N=1000):
    i = max(1, round(N * unshift_fn(s, shift)))
    return shift_fn(i / N, shift)


def ddpm_abar(i, T=1000, b0=1e-4, b1=0.02):
    betas = [b0 + (b1 - b0) * k / (T - 1) for k in range(T)]
    a = 1.0
    for k in range(i + 1):
        a *= 1 - betas[k]
    return a


def ddpm_to_flow(i):
    ab = ddpm_abar(i)
    nsr = math.sqrt((1 - ab) / ab)
    return nsr / (1 + nsr)


def abar_flow(s):
    nsr = s / (1 - s)
    return 1 / (1 + nsr ** 2)


def seed_from_bitstream(y_hat, z_hat):
    h = hashlib.sha256()
    for t in (z_hat, y_hat):
        h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return int(h.hexdigest()[:15], 16)


def t0_of_s(s0, snap, shift):
    return min(range(1000), key=lambda i: abs(snap(ddpm_to_flow(i), shift) - float(s0)))
