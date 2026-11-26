"""DeCo ImageNet-512 (class-conditional pixel DiT, flow matching) as an unconditional flow prior of the NLBP sampler
through its CFG null class 1000 (the AEIC Kodak controls): prior(x_s, s) -> v with x_0 = x_s - s v. DeCo's convention
x_tau = tau x_0 + (1 - tau) n, net(x, tau, y) = x_0 - n, so tau = 1 - s and v = -net; pixels in [-1, 1].
size_mode "tile512" (the reported runs): overlapping 512 x 512 tiles blended with a 64-px linear ramp; "native": one
token grid. Network: PixNerDiT of DeCo's dit_c2i_DeCo.py (external, third_party/nlbp_nets/UPSTREAM.md); weights
CKPT = NLBP_MODELS/deco/imagenet512_epoch340.ckpt (state_dict "ema_denoiser.*")."""
import os
import sys

import torch

NETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party",
                    "nlbp_nets", "src")          # the priors' network files (external, third_party/nlbp_nets/UPSTREAM.md)
if NETS not in sys.path:
    sys.path.insert(0, NETS)

CKPT = os.path.join(os.environ.get("NLBP_MODELS", "models"), "deco", "imagenet512_epoch340.ckpt")


def tiles_1d(n, T=512):
    if n <= T:
        return [0]
    k = -(-n // T) if n % T else n // T
    k = max(k, 2)
    return [round(i * (n - T) / (k - 1)) for i in range(k)]


def tiled(fn, x, T=512):
    B, C, H, W = x.shape
    if H <= T and W <= T:
        return fn(x)
    out = torch.zeros_like(x, dtype=torch.float32)
    wsum = torch.zeros((1, 1, H, W), device=x.device, dtype=torch.float32)
    for y0 in tiles_1d(H, T):
        for x0 in tiles_1d(W, T):
            h, w = min(T, H), min(T, W)
            wy = torch.ones(h, device=x.device)
            wx = torch.ones(w, device=x.device)
            r = torch.linspace(0.02, 1, 64, device=x.device)
            if y0 > 0:
                wy[:64] = r
            if y0 + h < H:
                wy[-64:] = r.flip(0)
            if x0 > 0:
                wx[:64] = r
            if x0 + w < W:
                wx[-64:] = r.flip(0)
            wt = (wy[:, None] * wx[None, :])[None, None]
            out[:, :, y0:y0 + h, x0:x0 + w] += wt * fn(x[:, :, y0:y0 + h, x0:x0 + w]).float()
            wsum[:, :, y0:y0 + h, x0:x0 + w] += wt
    return out / wsum


def build_deco(dev, ckpt=CKPT, dtype=torch.float32, log=print):
    p = NETS
    if p not in sys.path:
        sys.path.insert(0, p)
    import dit_c2i_DeCo as M
    net = M.PixNerDiT(in_channels=3, patch_size=16, num_groups=16, hidden_size=1152, hidden_size_x=32, num_blocks=31,
                      num_cond_blocks=28, nerf_mlpratio=2, num_classes=1000)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    pref = [k.split(".")[0] for k in sd]
    for cand in ("ema_denoiser.", "denoiser."):
        sub = {k[len(cand):]: v for k, v in sd.items() if k.startswith(cand)}
        if sub:
            missing, unexpected = net.load_state_dict(sub, strict=False)
            assert not missing, missing
            break
    else:
        raise RuntimeError("no ema_denoiser./denoiser. keys in the DeCo ckpt")
    return net.eval().requires_grad_(False).to(dev, dtype)


def load_deco_prior(dev, size_mode="native", dtype=torch.float32, cls=1000, log=print):
    net = build_deco(dev, dtype=dtype, log=log)

    @torch.no_grad()
    def v_of(x, tau):
        y = torch.full((x.shape[0],), cls, device=x.device, dtype=torch.long)
        t = torch.full((x.shape[0],), tau, device=x.device, dtype=torch.float32)
        return net(x.to(dtype), t.to(dtype), y).float()

    @torch.no_grad()
    def prior(xt, s):
        tau = 1.0 - float(s)
        f = lambda z: v_of(z, tau)
        v = f(xt) if size_mode == "native" else tiled(f, xt, 512)
        return -v

    prior.state = {}
    return prior
