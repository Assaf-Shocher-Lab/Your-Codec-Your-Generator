"""AEIC-ME (third_party/aeic) behind the adapted codec: codec steps, AEIC's colour fix, and the adapted decoder.

Notation (images in [-1, 1], reflect-padded to a multiple of 64 as AEIC's compress.py; h, w = unpadded size):
  y = g_a(x)          continuous latent, 320 channels, one vector per 32 x 32 cell
  z, z_hat            hyper latent z = h_a(y), z_hat = round(z - m) + m (m: AEIC's per-channel medians)
  mu, sigma           mean and scale of every y element from z_hat and the y_hat parts already decoded (four parts)
  y_hat               y_hat = round(y - mu) + mu; the record s = (z_hat, y_hat)
  E', W               adapters/aeic/wrapper.py;  y* = the selected target (select_target)
Decoder D' (AdaptedAeic): x_raw = AEIC's decoder output for y_hat (decode_symbols); y* = search_target from the
projected start E'(x_raw), then from the cell centre, refined toward the start; if both fail, y_hat when it is
hyper-consistent. consistent_decode: crop, AEIC's colour fix (images > 768^2), pad64, W(y*) as the last image
operation; clipped float: range_repair (App. B.3); 8-bit: corrected_write (closed-loop target correction through S_8,
Sec. 3.3); padded images: pad_repair. Encoder (round-selected): round 1 Q(g_a(x)), rounds >= 2 Q(E'(x)).
AEIC's own functions in which the method acts are in adapters/aeic/upstream_copies.py (verbatim, fenced)."""
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from adapted_codec.pad_repair import reflect_pad
from adapted_codec.protocols.clipped_float.range_repair import range_repair
from adapted_codec.settings import S
from adapters.aeic.upstream_copies import (AEICCodecCopy, AEICModelCopy, AdaptedRequest,
                                           adaptive_instance_normalization_side_info, hyper_latent, pad64,
                                           quantize_given_z)
from color_fix import calc_mean_std_quant           # third_party/aeic/src/color_fix.py, unchanged

COLOUR_FIX_MIN_PIXELS = 768 ** 2     # compress.py L136: AEIC colour-fixes images with ori_h * ori_w > 768 ** 2
COLOUR_FIX_BITS = 16                 # compress.py L139: adain_color_fix_quant(output_pil, img_pil, 16)
COLOUR_SIDE_BITS = 96                # 6 numbers x 16 bits sent besides (z_hat, y_hat)
Q16 = 65535.0                        # calc_mean_std_quant(bits=16) returns k / 65535
ENCODERS = ("variant_a",)


@dataclass(frozen=True)
class AeicSettings:
    """Settings of the AEIC decoder (configs/aeic_*.yaml, section aeic). Paper names: search_iters / search_lr = steps /
    learning rate of the target search, margin_y / margin_z = m_y / m_z, start "proj" = projected start, polish_* = the
    refinement toward the projected start (12 bisection steps, then polish_iters bounded Adam steps, extra hyper guard 0.01),
    pad_* = the border repair of padded images (pad_repair off on the high-resolution canvas chain).
    """
    search_iters: int = 400
    search_lr: float = 0.01
    margin_y: float = 0.02
    margin_z: float = 0.02
    start: str = "proj"
    polish_iters: int = 1000
    polish_lr: float = 0.003
    polish_weight: float = 100.0
    pad_repair: bool = True
    pad_iters: int = 1000
    pad_rows: int = 96
    pad_lr: float = 2e-3
    pad_y_target: float = 0.35
    pad_z_target: float = 0.4
    pad_z_stop: float = 0.01
    pad_cell_margin: float = 0.08
    log_distances: bool = True
    log_pad_psnr: bool = True
    encoder: str = "variant_a"

    def __post_init__(self):

        if self.encoder not in ENCODERS:
            raise ValueError(f"aeic.encoder must be one of {ENCODERS}, not {self.encoder!r}")


@dataclass
class Pair:
    z_hat: Tensor
    y_hat: Tensor
    y: Tensor
    bpp: float
    tag: str = ""


def encode_pair(net, x_padded, h, w, latent=None):
    request = AdaptedRequest(latent=latent, pair_only=True)
    net.codec.adapted_request = request
    try:
        with torch.no_grad():
            AEICCodecCopy.inference(net.codec, x_padded, ori_h=h, ori_w=w)
    finally:
        net.codec.adapted_request = None
    return Pair(z_hat=request.z_hat, y_hat=request.y_hat, y=request.y, bpp=float(request.rate.quantized_total_bpp))


def decode_symbols(net, y_hat):
    net.codec.adapted_request = AdaptedRequest(decode_y_hat=y_hat)
    try:
        x_hat, _ = AEICModelCopy.inference(net, None)
    finally:
        net.codec.adapted_request = None
    return x_hat


def needs_colour_fix(h, w):
    return h * w > COLOUR_FIX_MIN_PIXELS


def colour_stats(x_m11):
    return calc_mean_std_quant((x_m11 * 0.5 + 0.5).float(), bits=COLOUR_FIX_BITS)


def stats_to_side(stats):
    return tuple(int(round(float(t) * Q16)) for t in torch.cat([stats[0].flatten(), stats[1].flatten()]))


def colour_side_info(x_in, h, w):
    if not needs_colour_fix(h, w):
        return None
    return stats_to_side(colour_stats(x_in))


def crop_and_colour_fix(x_pad, h, w, side_info):
    x = x_pad[:, :, :h, :w]
    if side_info is not None:
        stats = torch.tensor(side_info, dtype=torch.float32, device=x.device) / 65535
        source_stats = (stats[:3].view(1, 3, 1, 1), stats[3:].view(1, 3, 1, 1))
        r = adaptive_instance_normalization_side_info((x * 0.5 + 0.5).float(), source_stats,
                                                      COLOUR_FIX_BITS).clamp(0.0, 1.0)
        x = r * 2 - 1
    return x


class AdaptedAeic:
    """The adapted AEIC decoder D' and the round-selected encoder for one AEIC model (net = AEIC_practical.AEIC,
    wrapped_encoder = E', settings = AeicSettings). D': select_target (Sec. 3.2) then consistent_decode (exact write W as the
    last image operation; clipped float: range repair; 8-bit: corrected_write; padded images: pad_repair).
    """
    def __init__(self, net, wrapped_encoder, settings: AeicSettings):
        self.net, self.wrapped_encoder, self.settings = net, wrapped_encoder, settings
        self._start_cache = []

    def start_point(self, y_hat, x_raw=None):
        for yk, s in self._start_cache:
            if yk.shape == y_hat.shape and torch.equal(yk, y_hat):
                return s
        with torch.no_grad():
            s = self.wrapped_encoder((decode_symbols(self.net, y_hat) if x_raw is None else x_raw).float()).clone()

        self._start_cache.insert(0, (y_hat.clone(), s))
        del self._start_cache[4:]
        return s

    def is_feasible(self, y, off, k, z_hat, y_hat):
        codec = self.net.codec
        z, zq = hyper_latent(codec, y)
        viol = ((z - off - k).abs() - (0.5 - self.settings.margin_z)).clamp_min(0)
        return (float(viol.max()) == 0.0 and torch.equal(zq, z_hat)
                and torch.equal(quantize_given_z(codec, y, z_hat)[0], y_hat))

    def refine_target(self, y_found, a, lo, hi, off, k, z_hat, y_hat):
        s, codec = self.settings, self.net.codec
        with torch.no_grad():
            best, t_lo, t_hi = y_found.clone(), 0.0, 1.0
            for _ in range(12):
                t = 0.5 * (t_lo + t_hi)
                y_ = a + t * (y_found - a)
                if self.is_feasible(y_, off, k, z_hat, y_hat):
                    best, t_hi = y_.clone(), t
                else:
                    t_lo = t
            bd = float((best - a).pow(2).sum())
        b0, n_acc = float((best - a).abs().mean()), 0
        ys = best.clone().requires_grad_(True)
        opt = torch.optim.Adam([ys], lr=s.polish_lr)
        for _ in range(s.polish_iters):
            z = hyper_latent(codec, ys)[0] - off
            viol = ((z - k).abs() - (0.5 - s.margin_z - 0.01)).clamp_min(0)
            loss = (ys - a).pow(2).sum() + s.polish_weight * (viol + 0.01 * (viol > 0)).pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                ys.copy_(torch.minimum(torch.maximum(ys, lo), hi))
                d = float((ys - a).pow(2).sum())
                if d < bd and self.is_feasible(ys.detach(), off, k, z_hat, y_hat):
                    best, bd, n_acc = ys.detach().clone(), d, n_acc + 1
        return best, f"b{b0:.3f}n{n_acc}"

    def search_target(self, z_hat, y_hat, y_start=None):
        s, codec = self.settings, self.net.codec
        off = codec.entropy_bottleneck._get_medians()
        with torch.inference_mode(False), torch.enable_grad():
            yh, zh, off = y_hat.clone(), z_hat.clone(), off.clone()
            lo, hi = yh - 0.5 + s.margin_y, yh + 0.5 - s.margin_y
            k = zh - off
            ys = (yh.clone() if y_start is None
                  else torch.minimum(torch.maximum(y_start.clone(), lo), hi)).requires_grad_(True)
            a = ys.detach().clone()
            opt = torch.optim.Adam([ys], lr=s.search_lr)
            for it in range(s.search_iters + 1):
                z = hyper_latent(codec, ys)[0] - off
                viol = ((z - k).abs() - (0.5 - s.margin_z)).clamp_min(0)
                if float(viol.max()) == 0.0:
                    with torch.no_grad():
                        y_ = ys.detach()
                        if torch.equal(hyper_latent(codec, y_)[1], zh) and \
                                torch.equal(quantize_given_z(codec, y_, zh)[0], yh):
                            refine_tag = ""
                            if s.polish_iters > 0 and y_start is not None:
                                with torch.enable_grad():
                                    y_, refine_tag = self.refine_target(y_.clone(), a, lo, hi, off, k, zh, yh)
                            tag = ""
                            if s.log_distances:
                                c = float((y_start - a).abs().mean()) if y_start is not None else 0.0
                                tag = f"d{float((y_ - a).abs().mean()):.3f}c{c:.3f}{refine_tag}"
                            return y_.clone(), it, tag
                if it == s.search_iters:
                    break
                loss = (viol + 0.01 * (viol > 0)).pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    ys.copy_(torch.minimum(torch.maximum(ys, lo), hi))
        return None, s.search_iters, ""

    def select_target(self, z_hat, y_hat, x_raw=None):
        y_star, it, tag = None, 0, ""
        if self.settings.start == "proj":
            y_star, it, tag = self.search_target(z_hat, y_hat, self.start_point(y_hat, x_raw))

        if y_star is None:
            y_star, it, tag = self.search_target(z_hat, y_hat)
        if y_star is None and torch.equal(hyper_latent(self.net.codec, y_hat.clone())[1], z_hat):
            return y_hat.clone(), -1, ""
        return y_star, it, tag

    def pad_repair(self, x_pad, y_star, y_hat, z_hat, h, w):
        s, codec = self.settings, self.net.codec
        H, W = x_pad.shape[2:]
        ok = False
        off = codec.entropy_bottleneck._get_medians()
        T = torch.round(z_hat - off)
        z0 = hyper_latent(codec, y_star)[0]
        zlim = ((z0 - off - T).abs() + 1e-3).clamp_min(s.pad_z_target)
        m = torch.zeros(1, 1, h, w, device=x_pad.device)
        if H > h:
            m[:, :, max(0, h - s.pad_rows):] = 1
        if W > w:
            m[:, :, :, max(0, w - s.pad_rows):] = 1
        with torch.inference_mode(False), torch.enable_grad():
            m, T, zlim = m.clone(), T.clone(), zlim.clone()
            zh, yh, yt = z_hat.clone(), y_hat.clone(), y_star.clone()
            ylim = ((yt - yh).abs() + 1e-3).clamp_min(0.5 - s.pad_cell_margin).clamp_max(0.5 - 1e-3)
            base = x_pad[:, :, :h, :w].clone()
            delta = torch.zeros_like(base, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=s.pad_lr)
            for it in range(s.pad_iters + 1):
                v = reflect_pad(base + delta * m, H, W)
                y2 = self.wrapped_encoder(v)
                e = (y2 - yt).abs()
                d = hyper_latent(codec, y2)[0] - off - T
                ez = float((d.abs() - zlim).max())
                ec = (y2 - yh).abs() - ylim
                if float(ec.max()) < 0 and ez < s.pad_z_stop:
                    with torch.no_grad():
                        if torch.equal(hyper_latent(codec, y2.detach())[1], zh) and \
                                torch.equal(quantize_given_z(codec, y2.detach(), zh)[0], yh):
                            ok = True
                            break
                if it == s.pad_iters:
                    break
                loss = (e - s.pad_y_target + 0.05).clamp_min(0).pow(2).sum() + \
                    (d.abs() - zlim + 0.05).clamp_min(0).pow(2).sum()
                loss = loss + 10 * (ec + 0.03).clamp_min(0).pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            out = reflect_pad(base + delta.detach() * m, H, W)
        psnr_tag = ""
        if s.log_pad_psnr:
            mse = float((delta.detach() * m).pow(2).mean()) / 4
            psnr_tag = f"p{-10 * math.log10(mse) if mse > 0 else 99:.1f}"
        return out.detach().clone(), f"{it}{psnr_tag}{'ok' if ok else 'X'}"

    def corrected_write(self, xp0, pair, yt):
        from adapted_codec.protocols.storage import store11
        we, codec, s = self.wrapped_encoder, self.net.codec, self.settings
        with torch.inference_mode(False), torch.enable_grad():
            yh, zh = pair.y_hat.clone(), pair.z_hat.clone()
            off = codec.entropy_bottleneck._get_medians().clone()
            k = zh - off
            goal, t, x0 = yt.clone(), yt.clone(), xp0.clone()
            lo, hi = yh - 0.5 + S.gamma_y, yh + 0.5 - S.gamma_y
            best, hist = None, []
            for n in range(S.correction_budget + 1):
                with torch.no_grad():
                    x = we.write(x0, t)
                    xq = store11(x).clone()
                    yq = we(xq).clone()
                    zq, zq_hat = hyper_latent(codec, yq)
                    nz = int((zq_hat != zh).sum())
                    ny = int((quantize_given_z(codec, yq, zh)[0] != yh).sum())
                    hist.append(f"{nz}/{ny}")
                    if best is None or (nz + ny) < best[0]:
                        best = (nz + ny, xq.clone(), n)
                    if nz == 0 and ny == 0:
                        break
                    if n == S.correction_budget:
                        break
                    over = ((yq - yh).abs() - (0.5 - S.gamma_y)).clamp_min(0) * torch.sign(yq - yh)
                    step = -over
                yv = yq.detach().clone().requires_grad_(True)
                viol = ((hyper_latent(codec, yv)[0] - off - k).abs() - (0.5 - S.gamma_z)).clamp_min(0)
                L = viol.pow(2).sum()
                if float(L) > 0:
                    g, = torch.autograd.grad(L, yv)
                    gg = float(g.pow(2).sum())
                    if gg > 0:
                        step = step - (float(L) / gg) * g.detach()
                with torch.no_grad():
                    t = t + step
                    goal = goal + step
        self.last_write = dict(ok=int(best[0] == 0), rounds=n, best_round=best[2], best_bad=best[0],
                            hist=",".join(hist[:12]) + (".." + hist[-1] if len(hist) > 12 else ""))
        return best[1]

    def consistent_decode(self, x_raw, pair, y_target, side_info, h, w, canvas=False):
        x = crop_and_colour_fix(x_raw, h, w, side_info)
        xp = pad64(x.float())
        xp0 = xp
        xp = self.wrapped_encoder.write(xp, y_target)
        if tuple(xp.shape[2:]) == (h, w):
            xp = range_repair(lambda v: self.wrapped_encoder.write(v, y_target), xp, -1.0, 1.0,
                              self.wrapped_encoder.pca.U[:self.wrapped_encoder.k])
        if S.correction_budget > 0 and ((tuple(xp.shape[2:]) == (h, w) and side_info is None) or canvas):
            xp = self.corrected_write(xp0, pair, y_target)
        pad_tag = ""
        if self.settings.pad_repair and tuple(xp.shape[2:]) != (h, w):
            xp, pad_tag = self.pad_repair(xp, y_target, pair.y_hat, pair.z_hat, h, w)
        out = xp[:, :, :h, :w]
        exc = (out.abs() - 1).clamp_min(0)
        if canvas:
            return out, int((exc > 0).sum()), float(exc.max()), pad_tag, xp
        return out, int((exc > 0).sum()), float(exc.max()), pad_tag

    def encode(self, x_padded, h, w, gen):
        if not isinstance(gen, int) or isinstance(gen, bool) or gen < 1:
            raise ValueError(f"encoder variant_a needs the generation number gen >= 1, got {gen!r}")
        if gen == 1:
            q = encode_pair(self.net, x_padded, h, w)
            q.tag = "zsA_plain"
        else:
            q = encode_pair(self.net, x_padded, h, w, latent=self.wrapped_encoder(x_padded))
            q.tag = "zsA_wrapped"
        return q
