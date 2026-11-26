"""Padded images (sizes not a multiple of the codec's padding unit): reflect padding as the codecs pad, and the border
repair that moves the latents of the padded border strip back inside their cells so that the re-encoder's own reflect
padding of the crop encodes inside the cells. Kodak images are never padded."""
import torch

from adapted_codec.settings import S

PAD_ZTGT = 0.4
PAD_ZSTOP = 0.01
PAD_TGT = 0.35


def reflect_pad(x, H, W):
    h, w = x.shape[2:]
    if H > h:
        x = torch.cat([x, x[:, :, h - 1 - (H - h):h - 1].flip(2)], 2)
    if W > w:
        x = torch.cat([x, x[:, :, :, w - 1 - (W - w):w - 1].flip(3)], 3)
    return x


def pad_to(h, w, m=64):
    return (h + m - 1) // m * m, (w + m - 1) // m * m


def pad_repair(ops, x_pad, hw, y_star, y_hat, z_hat):
    h, w = hw
    H, W = x_pad.shape[2:]
    if (H, W) == (h, w):
        return x_pad, "0ok"
    ok = False
    off = ops.z_offset()
    T = torch.round(z_hat - off)
    z0 = ops.hyper(y_star)[0]
    zlim = ((z0 - off - T).abs() + 1e-3).clamp_min(PAD_ZTGT)
    m = torch.zeros(1, 1, h, w, device=x_pad.device)
    if H > h:
        m[:, :, max(0, h - S.pad_rows):] = 1
    if W > w:
        m[:, :, :, max(0, w - S.pad_rows):] = 1
    with torch.inference_mode(False), torch.enable_grad():
        m, T, zlim, off = m.clone(), T.clone(), zlim.clone(), off.clone()
        zh, yh, yt = z_hat.clone(), y_hat.clone(), y_star.clone()
        ylim = ((yt - yh).abs() + 1e-3).clamp_min(0.5 - S.pad_cell_margin).clamp_max(0.5 - 1e-3)
        base = x_pad[:, :, :h, :w].clone()
        delta = torch.zeros_like(base, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=S.pad_lr)
        for it in range(S.pad_iters + 1):
            v = reflect_pad(base + delta * m, H, W)
            y2 = ops.wrapped_encode(v)
            e = (y2 - yt).abs()
            d = ops.hyper(y2)[0] - off - T
            ez = float((d.abs() - zlim).max())
            ec = (y2 - yh).abs() - ylim
            if float(ec.max()) < 0 and ez < PAD_ZSTOP:
                with torch.no_grad():
                    if torch.equal(ops.hyper(y2.detach())[1], zh) and \
                            torch.equal(ops.quantize_main(y2.detach(), zh)[0], yh):
                        ok = True
                        break
            if it == S.pad_iters:
                break
            loss = (e - PAD_TGT + 0.05).clamp_min(0).pow(2).sum() + (d.abs() - zlim + 0.05).clamp_min(0).pow(2).sum() \
                + 10 * (ec + 0.03).clamp_min(0).pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        out = (base + delta.detach() * m).clone()
    return out, f"{it}{'ok' if ok else 'X'}"
