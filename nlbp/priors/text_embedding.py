"""The Qwen3-1.7B embedding of a prompt for the PixNerd T2I prior, computed once by nlbp/priors/qwen3_embed_child.py in
the prior environment (NLBP_PRIOR_PY; the codec environment's transformers has no Qwen3Model) and cached in EMB_DIR
(NLBP_WORK/cache/t2i_emb); the embedding depends only on the prompt."""
import os
import sys

import torch

PRIOR_PY = os.environ.get("NLBP_PRIOR_PY", sys.executable)
EMB_DIR = os.path.join(os.environ.get("NLBP_WORK", os.getcwd()), "cache", "t2i_emb")


@torch.no_grad()
def text_embedding(qwen_dir, prompt, dev, max_length=128, embed_dim=2048, log=print):
    import hashlib
    import subprocess
    os.makedirs(EMB_DIR, exist_ok=True)
    f = os.path.join(EMB_DIR, "qwen3_" + hashlib.md5(f"{prompt}|{max_length}".encode()).hexdigest()[:12] + ".pt")
    if not os.path.exists(f):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
        child = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_embed_child.py")
        r = subprocess.run([PRIOR_PY, child, qwen_dir, prompt, f + ".tmp", str(max_length)], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("qwen3_embed_child failed")
        os.replace(f + ".tmp", f)
    d = torch.load(f, map_location="cpu")
    y = d["y"].to(dev)
    if not torch.isfinite(y).all():
        raise RuntimeError("Qwen3 text embedding is not finite")
    if y.shape[2] < embed_dim:
        y = torch.cat([y, torch.zeros(y.shape[0], y.shape[1], embed_dim - y.shape[2]).to(y.device, y.dtype)], dim=-1)
    if y.shape[2] > embed_dim:
        y = y[:, :, :embed_dim]
    return y
