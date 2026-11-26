"""PixNerd-XXL/16 T2I (512) as a flow prior of the NLBP sampler: prior(x_s, s) -> v with x_0 = x_s - s v. The network
convention is x_t = t x + (1 - t) eps, v_net = x - eps, so t = 1 - s and v = -v_net; empty prompt = the Qwen3 embedding
of "" (nlbp/priors/text_embedding.py), cfg 1. 768 x 512 images: two 512 tiles with a linear ramp (nlbp/priors/tiles.py).
Network: PixNerDiT of the PixNerd repository (external, third_party/nlbp_nets/UPSTREAM.md); weights
PixNerd-XXL-P16-T2I model.ckpt (state_dict "ema_denoiser.*")."""
import os
import sys

import torch

NETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party",
                    "nlbp_nets", "src")          # the priors' network files (external, third_party/nlbp_nets/UPSTREAM.md)
if NETS not in sys.path:
    sys.path.insert(0, NETS)

from pixnerd_t2i import PixNerDiT  # noqa: E402
from nlbp.priors.text_embedding import text_embedding  # noqa: E402
from nlbp.priors.tiles import ramp_weight, tiles512  # noqa: E402


def build_pixnerd_t2i():
    return PixNerDiT(in_channels=3, patch_size=16, num_groups=24, hidden_size=1536, txt_embed_dim=2048,
                     txt_max_length=128, num_text_blocks=4, decoder_hidden_size=64, num_encoder_blocks=16,
                     num_decoder_blocks=2)


def load_net(ckpt, dev, log=print):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    sd = sd.get("state_dict", sd)
    pre = "ema_denoiser."
    if not any(k.startswith(pre) for k in sd):
        pre = "denoiser."
    sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    if not sub:
        pre, sub = "", {k: v for k, v in sd.items() if hasattr(v, "shape")}
    prefixes = sorted({k.split(".")[0] for k in sd})
    net = build_pixnerd_t2i()
    missing, unexpected = net.load_state_dict(sub, strict=False)
    if missing:
        raise RuntimeError(f"PixNerd T2I weights incomplete: {missing[:10]}")
    return net.eval().requires_grad_(False).to(dev).float()


def load_pixnerd_t2i_flow_prior(dev, ckpt, qwen_dir, prompt="", cfg=1.0, bf16=False, T=512, log=print):
    yc = text_embedding(qwen_dir, prompt, dev, log=log)
    yu = yc if (prompt == "" or cfg == 1.0) else text_embedding(qwen_dir, "", dev, log=log)
    net = load_net(ckpt, dev, log=log)
    yc = yc.to(dev)
    yu = yu.to(dev)

    @torch.no_grad()
    def run(x, t):
        n = x.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(bf16)):
            if cfg == 1.0:
                return net(x, t, yc.expand(n, -1, -1).float()).float()
            out = net(torch.cat([x, x]), torch.cat([t, t]),
                      torch.cat([yu.expand(n, -1, -1), yc.expand(n, -1, -1)]).float()).float()
            u, c = out.chunk(2)
            return u + cfg * (c - u)

    @torch.no_grad()
    def prior(xt, s):
        xt = xt.float()
        n, _, H, W = xt.shape
        t = torch.full((n,), 1.0 - float(s), device=xt.device, dtype=torch.float32)
        if H == T and W == T:
            v = run(xt, t)
        else:
            assert H >= T and W >= T, (H, W)
            v = torch.zeros_like(xt)
            wsum = torch.zeros((1, 1, H, W), device=xt.device)
            for (i0, j0) in tiles512(H, W, T):
                w = ramp_weight(H, W, i0, j0, T, xt.device)[None, None]
                v[..., i0:i0 + T, j0:j0 + T] += w * run(xt[..., i0:i0 + T, j0:j0 + T].contiguous(), t)
                wsum[..., i0:i0 + T, j0:j0 + T] += w
            v = v / wsum
        return -v
    return prior
