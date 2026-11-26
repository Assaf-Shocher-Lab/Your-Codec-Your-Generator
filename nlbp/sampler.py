"""NLBP decoding (Sec. 3.4, App. E.1, Algorithm "NLBP decoding") for a pixel-space prior.

Inputs per image (state): geo = nlbp.final_write.TargetWrite(codec, y*) with geo.r_t = b(D(y_hat)); Dp = the base
x_base = W(y*, b(D(y_hat))); r_ys = b(D(y*)) (used when v_ref = y*); seed = schedule.seed_from_bitstream(y_hat, z_hat).
Configuration (prm, nlbp/configs/*.yaml): t0 (DDPM start index; omega = the snapped flow time of t0), stride, min_bp
(steps with index >= min_bp use lambda2 = the early lambda_t, the others lambda1 = the late one), kz = kappa,
target (yhat | ystar = v_ref), upsample (1 for the reported runs), opt_K = K, opt_lr.

    schedule   s_k = snap(flow time of DDPM index i), i = t0, t0 - stride, ..., 0 (duplicates dropped), then 0
    start      x_{s_0} = (1 - s_0) x_base + s_0 eps                       (start noise from the seeded generator)
    step k     v = prior(x_s, s);  u = x_s - s v (clean estimate);  e = x_s + (1 - s) v
               u^+ = nlbp_update(u) if lambda_t > 0 else u
               e <- sqrt(1 - rho^2) e + rho eps',  rho = min(1, sqrt((1 - abar_s / abar_s') / (1 - abar_s')))  (s' > 0)
               x_{s'} = (1 - s') u^+ + s' e
    final      out = geo.opt(x_0, K, lr): K Adam steps on b toward x_0, exact write W(y*, b) (E'(out) = y*)
The 8-bit chains then run the 8-bit repair on out (adapted_codec/protocols/eight_bit) and the check Sigma(E'(x)) = s.

Latent-space priors (nlbp_decode_latent; the prior runs behind a client with encode / decode / velocity): the state is
the latent z, z_{s_0} = (1 - s_0) Enc(x_base) + s_0 eps, steps s_k = s_0 (1 - k / n); per step the clean latent
z_0 = z - s v is decoded first, x_0 = Dec(z_0) (vae_residual: x_0 = x_base + Dec(z_0) - Dec(Enc(x_base))), the image-space
update acts on x_0 (lambda_t = lambda2 while s >= min_bp_frac s_0, else lambda1), and the corrected image is encoded
back (z_0^+ = Enc(x_0^+), residual form likewise) before the same re-noising; the last clean image goes to the final
write."""
import math

import torch
import torch.nn.functional as F

from nlbp.schedule import MINBP_FRAC, abar_flow, ddpm_to_flow, t0_of_s
from nlbp.update import nlbp_update


def make_noise(gen, dev, noise=None):
    """Draws from gen in call order, or pops pre-drawn tensors in order."""
    if noise is None:
        return lambda shape: torch.randn(shape, generator=gen, device=dev)
    it = iter(noise)
    return lambda shape: next(it).to(dev).reshape(shape)


def resolve(prm, snap, shift):
    """A configuration given by s0 (= omega) instead of t0 -> t0 = the DDPM index whose snapped flow time is nearest s0;
    min_bp at the fraction 354/550 of t0 and the number of steps, unless the configuration gives them."""
    prm = dict(prm)
    if "s0" in prm and "t0" not in prm:
        prm["t0"] = t0_of_s(prm["s0"], snap, shift)
    if "t0" in prm:
        t0_, st_ = int(prm["t0"]), int(prm.get("stride", 10))
        prm.setdefault("min_bp", int(round(MINBP_FRAC * t0_)))
        prm.setdefault("min_bp_frac", MINBP_FRAC)
        prm.setdefault("n_steps", len(range(t0_, -1, -st_)))
        prm.setdefault("s0", float(snap(ddpm_to_flow(t0_), shift)))
    return prm


def nlbp_decode(state, prm, prior, codec, snap, shift, dev, noise=None, opt_lr=0.003):
    """The NLBP decode of one image (module docstring). Returns (output image, number of sampling steps)."""
    t0, lam2, lam1, min_bp, kz, target = (prm["t0"], prm["lambda2"], prm["lambda1"], prm["min_bp"], prm["kz"],
                                          prm["target"])
    stride, K, opt_K = int(prm["stride"]), int(prm["upsample"]), int(prm["opt_K"])
    up = (lambda u: F.interpolate(u, scale_factor=K, mode="bicubic", align_corners=False)) if K > 1 else (lambda u: u)
    down = (lambda u: F.avg_pool2d(u, K)) if K > 1 else (lambda u: u)

    def lift_bp(u_hi, fn):
        if K == 1:
            return fn(u_hi)
        lo = down(u_hi)
        return u_hi + F.interpolate(fn(lo) - lo, scale_factor=K, mode="nearest")

    geo, gen = state["geo"], torch.Generator(device=dev).manual_seed(state["seed"] + int(prm.get("seed_off", 0)))
    yh = geo.y_hat.float()
    rt_arm = state["r_ys"] if target == "ystar" else None

    def soft(u, lam, kz, rt=None):
        return nlbp_update(u, lam, kz, codec, yh, geo.r_t if rt is None else rt)

    sched, seen = [], set()
    for i in range(t0, -1, -stride):
        s = snap(ddpm_to_flow(i), shift)
        if s not in seen:
            seen.add(s)
            sched.append((s, "L2" if i >= min_bp else "L1"))
    lam_map = {"L2": lam2, "L1": lam1}
    nz = make_noise(gen, dev, noise)
    Dp = up(state["Dp"])
    eps0 = nz(Dp.shape)
    s0 = sched[0][0]
    xt = (1 - s0) * Dp + s0 * eps0
    ss = [s for s, _ in sched] + [0.0]
    for k, (s, s_next) in enumerate(zip(ss[:-1], ss[1:])):
        v = prior(xt, s)
        x0 = xt - s * v
        et = xt + (1 - s) * v
        lam = lam_map[sched[k][1]]
        if lam > 0:
            x0h = lift_bp(x0, lambda u: soft(u, lam, kz, rt_arm))
        else:
            x0h = x0
        if s_next > 0:
            ab_s, ab_n = abar_flow(s), abar_flow(s_next)
            rho = min(1.0, math.sqrt(max(0.0, (1 - ab_s / ab_n) / (1 - ab_n))))
            et = math.sqrt(1 - rho ** 2) * et + rho * nz(et.shape)
        xt = (1 - s_next) * x0h + s_next * et
    out = geo.opt(down(xt), opt_K, float(prm.get("opt_lr", opt_lr)))
    return out, len(sched)


def nlbp_decode_latent(state, prm, client, codec, dev, noise=None, opt_lr=0.003):
    """The NLBP decode with a latent-space prior (module docstring); prm resolved (n_steps, min_bp_frac), R1 arm.
    Returns (output image, number of sampling steps)."""
    s0, nst = float(prm["s0"]), int(prm["n_steps"])
    resid = int(prm.get("vae_residual", 0))
    opt_K = int(prm.get("opt_K", 10))
    cfg = float(prm.get("cfg", 1.0))
    prompt = prm.get("prompt", "empty")
    geo, gen = state["geo"], torch.Generator(device=dev).manual_seed(state["seed"] + int(prm.get("seed_off", 0)))
    Dp = state["Dp"]
    LC = latent_start(state, client)
    ZD, R = LC["ZD"], LC["R"]
    enc = (lambda x: client.encode(x - Dp + R)) if resid else client.encode
    dec = (lambda z: Dp + client.decode(z) - R) if resid else client.decode
    yh = geo.y_hat.float()
    lam2, lam1, kz, target = prm["lambda2"], prm["lambda1"], prm["kz"], prm["target"]
    mbf = float(prm.get("min_bp_frac", 0.0))
    rt_arm = state["r_ys"] if target == "ystar" else None

    def soft(u, lam, kz, rt=None):
        return nlbp_update(u, lam, kz, codec, yh, geo.r_t if rt is None else rt)
    sched = [s0 * (1 - k / nst) for k in range(nst)]
    ss = sched + [0.0]
    nz = make_noise(gen, dev, noise)
    eps0 = nz(ZD.shape)
    z = (1 - s0) * ZD + s0 * eps0
    x_fin = None
    for k, (s, s_next) in enumerate(zip(ss[:-1], ss[1:])):
        v = client.velocity(z, s, prompt, cfg)
        z0 = z - s * v
        et = z + (1 - s) * v
        x0 = dec(z0)
        lam = lam2 if s >= mbf * s0 else lam1
        x0h = soft(x0, lam, kz, rt_arm) if lam > 0 else x0
        if s_next <= 0:
            x_fin = x0h
            break
        z0h = enc(x0h) if x0h is not x0 else z0
        ab_s, ab_n = abar_flow(s), abar_flow(s_next)
        rho = min(1.0, math.sqrt(max(0.0, (1 - ab_s / ab_n) / (1 - ab_n))))
        et = math.sqrt(1 - rho ** 2) * et + rho * nz(et.shape)
        z = (1 - s_next) * z0h + s_next * et
    out = geo.opt(x_fin, opt_K, opt_lr)
    return out, nst


def latent_start(state, client):
    """Enc(x_base) and Dec(Enc(x_base)), once per image (kept in the state)."""
    key = ("lat", int(getattr(client, "upsample", 1)), 1)
    if key not in state:
        Dh = state["Dp"]
        ZD = client.encode(Dh)
        state[key] = dict(ZD=ZD, R=client.decode(ZD), Dh=Dh)
    return state[key]
