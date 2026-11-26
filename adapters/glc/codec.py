"""GLC (Jia et al., CVPR 2024; third_party/glc, GLC_Image) behind the adapted codec.

E_q(x) = enc(vqgan.encoder(x), q_enc[q]) is the analysis transform the wrapper acts on (f = 16, k = 256, c = 10).
Record s = (VQ indices of hyper_enc(y), y_q of the four-part prior given the indices); y_hat and the quantization step
follow from s. GLC has no entropy coder: the rate is the estimated one (Gaussian bits of y_q + log2(codebook) per index).
Decoder-side search (Sec. 3.2, GLC row of the settings table): start at the received y_hat, Adam steps inside the shrunk
cells |t - y_hat| <= (1/2 - m_y) quant_step, until VQ(hyper_enc(t)) = the received indices with a squared-distance gap
m_z to the 8 nearest other codewords and the prior reproduces y_q. Images are in [-1, 1]."""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from adapted_codec.settings import S

GLC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party",
                       "glc")
if GLC_DIR not in sys.path:
    sys.path.insert(0, GLC_DIR)
from src.models.image_model import GLC_Image  # noqa: E402
from src.utils.test_utils import get_state_dict  # noqa: E402


def replicate_pad(x, H, W):
    h, w = x.shape[2:]
    if H > h:
        x = torch.cat([x, x[:, :, h - 1:h].expand(-1, -1, H - h, -1)], 2)
    if W > w:
        x = torch.cat([x, x[:, :, :, w - 1:w].expand(-1, -1, -1, W - w)], 3)
    return x


def exact_math():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EQ(nn.Module):
    dtype = torch.float32

    def __init__(self, net, q):
        super().__init__()
        self.net, self.q = net, q
        self.config = type("C", (), {"scaling_factor": 1.0})()

    def forward(self, x):
        n = self.net
        return n.enc(n.vqgan.encoder(x), n.q_enc[self.q:self.q + 1])

    def encode(self, x, return_dict=True):
        m = self.forward(x)
        return type("O", (), {"latent_dist": type("L", (), {"mean": m})()})()


class GLC:
    def __init__(self, ckpt, device="cuda"):
        exact_math()
        net = GLC_Image(inplace=False)
        net.load_state_dict(get_state_dict(ckpt), strict=True)
        self.net = net.to(device).eval()
        self.dev = device

    def z_of(self, y):
        n = self.net
        z = n.hyper_enc(y)
        idx = n.z_vq.get_indices(z)
        return idx, z.shape

    def main(self, y, idx, zshape, with_qs=False):
        n = self.net
        z_hat = n.z_vq.get_quan_feat(idx, (zshape[0], zshape[2], zshape[3], zshape[1]))
        params = n.y_prior_fusion(n.hyper_dec(z_hat))
        _, y_q, y_hat, scales_hat = n.forward_four_part_prior(
            y, params, n.y_spatial_prior_adaptor_1, n.y_spatial_prior_adaptor_2, n.y_spatial_prior_adaptor_3,
            n.y_spatial_prior, y_spatial_prior_reduction=n.y_spatial_prior_reduction)
        bit_y = n.get_y_gaussian_bits(y_q, scales_hat).sum().item()
        bit_z = zshape[2] * zshape[3] * math.log2(n.codebook_size)
        if with_qs:
            return y_q, y_hat, bit_y + bit_z, n.separate_prior(params)[1]
        return y_q, y_hat, bit_y + bit_z

    def search(self, idx, zs, y_q, y_hat, qs):
        n = self.net
        E = n.z_vq.embedding.weight.detach()
        e2 = (E ** 2).sum(1)
        tgt = idx.view(-1)
        with torch.inference_mode(False), torch.enable_grad():
            yh, q_s = y_hat.clone(), qs.clone()
            lo, hi = yh - (0.5 - S.margin_y) * q_s, yh + (0.5 - S.margin_y) * q_s
            ys = yh.clone().requires_grad_(True)
            opt = torch.optim.Adam([ys], lr=S.search_lr)
            for it in range(S.search_steps + 1):
                z = n.hyper_enc(ys).permute(0, 2, 3, 1).reshape(-1, E.shape[1])
                d = (z ** 2).sum(1, keepdim=True) + e2 - 2 * z @ E.t()
                d_t = d.gather(1, tgt[:, None])
                d_o = d.scatter(1, tgt[:, None], float("inf")).topk(8, dim=1, largest=False).values
                viol = (d_t - d_o + S.margin_z).clamp_min(0)
                if float(viol.max()) == 0.0:
                    with torch.no_grad():
                        y_ = ys.detach()
                        if torch.equal(self.z_of(y_)[0], idx) and torch.equal(self.main(y_, idx, zs)[0], y_q):
                            return y_.clone(), it
                if it == S.search_steps:
                    break
                loss = (viol + 0.01 * (viol > 0)).pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    ys.copy_(torch.minimum(torch.maximum(ys, lo), hi))
        return None, S.search_steps

    def pad_repair(self, wrapper, x_pad, hw, y_t, idx, zs, y_q, y_hat, qs):
        h, w = hw
        H, W = x_pad.shape[2:]
        n = self.net
        E = n.z_vq.embedding.weight.detach()
        e2 = (E ** 2).sum(1)
        tgt = idx.view(-1)
        m = torch.zeros(1, 1, h, w, device=x_pad.device)
        if H > h:
            m[:, :, max(0, h - S.pad_rows):] = 1
        if W > w:
            m[:, :, :, max(0, w - S.pad_rows):] = 1
        ok = False
        with torch.inference_mode(False), torch.enable_grad():
            m, yh, q_s, yt = m.clone(), y_hat.clone(), qs.clone(), y_t.clone()
            ylim = ((yt - yh).abs() / q_s + 1e-3).clamp_min(0.5 - S.pad_cell_margin).clamp_max(0.5 - 1e-3)
            base = x_pad[:, :, :h, :w].clone()
            delta = torch.zeros_like(base, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=S.pad_lr)
            for it in range(S.pad_iters + 1):
                v = replicate_pad(base + delta * m, H, W)
                y2 = wrapper.wrapped_encoder(v)
                ec = (y2 - yh).abs() / q_s - ylim
                z = n.hyper_enc(y2).permute(0, 2, 3, 1).reshape(-1, E.shape[1])
                d = (z ** 2).sum(1, keepdim=True) + e2 - 2 * z @ E.t()
                d_o = d.scatter(1, tgt[:, None], float("inf")).topk(8, dim=1, largest=False).values
                viol = (d.gather(1, tgt[:, None]) - d_o + S.margin_z).clamp_min(0)
                if float(ec.max()) < 0 and float(viol.max()) == 0.0:
                    with torch.no_grad():
                        y_ = y2.detach()
                        if torch.equal(self.z_of(y_)[0], idx) and torch.equal(self.main(y_, idx, zs)[0], y_q):
                            ok = True
                            break
                if it == S.pad_iters:
                    break
                loss = 10 * (ec + 0.03).clamp_min(0).pow(2).sum() + (viol + 0.01 * (viol > 0)).pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            out = (base + delta.detach() * m).clone()
        return out, f"{it}{'ok' if ok else 'X'}"

    def D(self, y_hat, q):
        n = self.net
        return n.vqgan.generator(n.dec(y_hat, n.q_dec[q:q + 1])).clamp(-1, 1)


def psnr(a, b):
    mse = float(((a.clamp(-1, 1) - b) / 2).pow(2).mean())
    return 10 * math.log10(1 / max(mse, 1e-12))


def load_image(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
