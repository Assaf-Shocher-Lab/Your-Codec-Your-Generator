"""Escalation ladder (Sec. 3.3, App. B.3). Stage 0 = the preset corrected write; stage 1 = a doubled correction budget
(60); stage 2 = also a wider hyper guard (gamma_z 0.10); stage 3 = the box step. A stage runs only after every earlier
stage failed the same acceptance test; if all stages fail the stage-0 image is returned, not certified.

Two entry points with the same stages:
  escalation_decode  from the received record (HPCM rerun): stage 0 = decode_once (target selection included);
                     acceptance = the round-2 encoder reproduces (z_hat, y_hat) from the stored image.
  escalation_write   from a selected target y* and the raw decode x_raw (CompressAI-zoo codecs, TCM): acceptance =
                     nz + #(|E'(x) - y_hat| >= 1/2) = 0 on the stored image (context-free count, exact for these
                     codecs since z_hat fixes the means slice by slice); the box step needs a target."""
import torch

from adapted_codec.encoder import encode
from adapted_codec.protocols.eight_bit.box_step import box_step_decode
from adapted_codec.protocols.eight_bit.target_correction import corrected_write, decode_once
from adapted_codec.protocols.storage import tensor_md5
from adapted_codec.settings import S


@torch.no_grad()
def escalation_decode(ops, z_hat, y_hat):
    """Stages 0-3 of the escalation ladder; returns the first stage whose stored image passes the acceptance test."""
    from adapted_codec import pad_repair as zp
    from adapted_codec.protocols.storage import store01
    fix0, gz0, sel0 = S.correction_budget, S.gamma_z, S.context_free_count
    S.context_free_count = S.stabilization_candidates[0]

    def accept(x, H, W):
        q2 = encode(ops, zp.reflect_pad(store01(x), H, W), gen=2)
        return (torch.equal(q2.z_hat, z_hat) and torch.equal(q2.y_hat, y_hat), int((q2.z_hat != z_hat).sum()),
                int((q2.y_hat != y_hat).sum()), tensor_md5((store01(x).double() * 255).round()))

    try:
        out0 = None
        for si, (fx, gz) in enumerate(S.ladder_stages):
            S.correction_budget = fix0 if fx is None else fx
            S.gamma_z = gz0 if gz is None else gz
            if si == 0:
                out = out0 = decode_once(ops, z_hat, y_hat)
                H, W = out0[0].shape[2:]
                if S.image_hw is not None and tuple(S.image_hw) != (H, W):
                    return out0
                if out0[2] is None or fix0 <= 0:
                    return out0
            else:
                out = (corrected_write(ops, z_hat, y_hat, out0[2], out0[1]), out0[1], out0[2], out0[3])
            ok, nz, ny, xm = accept(out[0], H, W)
            if ok:
                return out
        S.correction_budget, S.gamma_z = fix0, gz0
        if S.box_step:
            xb = box_step_decode(ops, z_hat, y_hat, out0[2], out0[1])
            ok, nz, ny, xm = accept(xb, H, W)
            if ok:
                return (xb, out0[1], out0[2], out0[3])
        return out0
    finally:
        S.correction_budget, S.gamma_z, S.context_free_count = fix0, gz0, sel0


@torch.no_grad()
def escalation_write(ops, z_hat, y_hat, yt, xr):
    """Stages 0-3 from the target yt (None: no target found, stages 0-2 write y_hat, no box step) and x_raw = xr.
    Returns (stored image, stage that passed or -1)."""
    if S.box_step and (S.context_free_count != 1 or S.correction_budget <= 0):
        raise RuntimeError("escalation_write: the box stage needs context_free_count = 1 and correction_budget > 0")
    tgt = y_hat if yt is None else yt
    fix0, gz0 = S.correction_budget, S.gamma_z
    xd, stage = None, -1
    try:
        for si, (fx, gz) in enumerate(S.ladder_stages):
            S.correction_budget = fix0 if fx is None else fx
            S.gamma_z = gz0 if gz is None else gz
            xs = corrected_write(ops, z_hat, y_hat, tgt.clone(), xr.clone())
            ys = ops.wrapped_encode(xs)
            bad = int((ops.hyper(ys)[1] != z_hat).sum()) + int(((ys - y_hat).abs() >= 0.5).sum())
            if si == 0:
                xd = xs
            if bad == 0:
                xd, stage = xs, si
                break
            del ys
    finally:
        S.correction_budget, S.gamma_z = fix0, gz0
    if S.box_step and stage == -1 and yt is not None:
        xb = box_step_decode(ops, z_hat, y_hat, tgt.clone(), xr.clone())
        yb = ops.wrapped_encode(xb)
        if int((ops.hyper(yb)[1] != z_hat).sum()) + int(((yb - y_hat).abs() >= 0.5).sum()) == 0:
            xd, stage = xb, 3
    return xd, stage
