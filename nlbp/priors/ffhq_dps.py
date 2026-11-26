"""The FFHQ prior of the ELIC FFHQ runs (`ffhq_10m`: the DPS authors' FFHQ 256 x 256 DDPM, as used by Xu et al.) in
the sampler's flow convention (nlbp/priors/ddpm.py).

Network: guided-diffusion's UNet as shipped in Xu et al.'s repository (an external dependency, not redistributed;
third_party/xu/UPSTREAM.md), created from that repository's configs/ffhq_model_config.yaml with model_path = the
checkpoint; the state dict is loaded again with strict=True (create_model only prints a load error). fp32, no class
label, no x_0 clipping, the learned variance channels unused.
    prior(x_s, s) -> v with x_0 = x_s - s v:   i = round(s_to_index(s)),  x_i = x_s / sqrt((1 - s)^2 + s^2),
    eps = UNet(x_i, i)[:, :3],  x_0 = (x_i - sqrt(1 - abar_i) eps) / sqrt(abar_i),  v = (x_s - x_0) / s."""
import importlib
import importlib.util
import math
import os
import sys

import numpy as np
import torch
import yaml

from nlbp.priors import ddpm as DF


def _unet_module(repo):
    """guided_diffusion/unet.py of the external repository as package 'xu_guided_diffusion' (no sys.path change: the
    repository's top-level packages must not shadow anything; unet.py needs only .fp16_util and .nn)."""
    name = "xu_guided_diffusion"
    if name + ".unet" in sys.modules:
        return sys.modules[name + ".unet"]
    pkg_dir = os.path.join(repo, "guided_diffusion")
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, "__init__.py"),
                                                  submodule_search_locations=[pkg_dir])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[name] = pkg
    spec.loader.exec_module(pkg)
    unet = importlib.import_module(name + ".unet")
    assert os.path.realpath(unet.__file__) == os.path.realpath(os.path.join(pkg_dir, "unet.py")), unet.__file__
    return unet


def xu_abar(T=1000):
    """The external repository's schedule: linear betas (get_named_beta_schedule('linear', 1000)) and cumprod."""
    scale = 1000 / T
    betas = np.linspace(scale * 0.0001, scale * 0.02, T, dtype=np.float64)
    return np.cumprod(1.0 - betas, axis=0)


def load_unet(repo, ckpt, dev):
    with open(os.path.join(repo, "configs", "ffhq_model_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["model_path"] = ckpt
    unet = _unet_module(repo)
    model = unet.create_model(**cfg)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    del sd
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def load_ffhq_dps_prior(repo, ckpt, dev):
    """prior(x_s, s) -> v (module docstring). repo = the external repository folder, ckpt = ffhq_10m.pt."""
    model, cfg = load_unet(repo, ckpt, dev)
    d_ab = float(np.abs(xu_abar() - DF._ABAR.numpy()).max())
    assert d_ab < 1e-12, d_ab
    out_ch = 6 if cfg["learn_sigma"] else 3

    @torch.no_grad()
    def prior(xt, s):
        assert xt.shape[-2] % 32 == 0 and xt.shape[-1] % 32 == 0, xt.shape
        i = float(round(DF.s_to_index(s)))
        ab = float(DF._ABAR[int(i)])
        xd = xt.float() / math.sqrt((1 - s) ** 2 + s ** 2)
        t = torch.full((xt.shape[0],), i, device=xt.device, dtype=torch.float32)
        out = model(xd, t).float()
        assert out.shape[1] == out_ch, out.shape
        eps = out[:, :3]
        x0 = (xd - math.sqrt(1 - ab) * eps) / math.sqrt(ab)
        return (xt.float() - x0) / s
    prior.name = "ffhq_10m"
    return prior
