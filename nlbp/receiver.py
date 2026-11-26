"""Receiver preparation of one NLBP decode (App. E.1), from the received record q (y_hat, z_hat) and the selected
target q.y_t = y* (adapted_codec/target_search.py, float margins):

    geo.x_dec = D(y_hat),  geo.r_t = r_ref = b(D(y_hat))   (v_ref = y_hat; geo = TargetWrite(codec, y*))
    Dp        = x_base = W(y*, b(D(y_hat)))                (the adapted decoder's output, the sampler's start)
    r_ys      = b(D(y*))                                   (r_ref when v_ref = y*)
    seed      = seed_from_bitstream(y_hat, z_hat)          (the same symbols give the same decode)
codec: the NLBP codec interface (nlbp/codec_interface.py)."""
import math

from nlbp.final_write import TargetWrite
from nlbp.schedule import seed_from_bitstream


def nlbp_state(name, q, codec):
    """The sampler's per-image state (module docstring)."""
    geo = TargetWrite(codec, q.y_t)
    geo.x_dec = codec.base_decode(q.y_hat)
    geo.r_t = codec.complement(geo.x_dec)
    Dp = geo.raw(geo.x_dec)
    r_ys = codec.complement(codec.base_decode(q.y_t))
    return dict(name=name, q=q, geo=geo, Dp=Dp, r_ys=r_ys, seed=seed_from_bitstream(q.y_hat, q.z_hat))


def psnr(a, b):
    """PSNR of two images in [-1, 1] (peak-to-peak 2), as the NLBP runs report it."""
    mse = float(((a - b) / 2).pow(2).mean())
    return 100.0 if mse == 0 else 10 * math.log10(1 / mse)
