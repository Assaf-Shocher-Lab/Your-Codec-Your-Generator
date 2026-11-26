"""The 8-bit adapted decoder D_8: escalation ladder, outer stabilization (HPCM, App. B.3) or the stage-0 decode.

Outer stabilization (`stabilization_states` = 4): s_0 = s; at state s_i try each candidate setting
(`stabilization_candidates`) and accept x when the round-2 encoder reproduces s_i from S_8(x); otherwise
s_{i+1} = Sigma(E'(D_8(s_i))). On success D_8(Sigma(E'(x*))) = x*: the image repeats from round 1, the file may change
once at round 2."""
import torch

from adapted_codec.encoder import encode
from adapted_codec.protocols.eight_bit.escalation import escalation_decode
from adapted_codec.protocols.eight_bit.target_correction import decode_once
from adapted_codec.settings import S


@torch.no_grad()
def decode(ops, z_hat, y_hat):
    """D_8(s): the escalation ladder, the outer stabilization or the stage-0 decode. Returns (x, x_raw, y*, it)."""
    if S.ladder:
        return escalation_decode(ops, z_hat, y_hat)
    if S.stabilization_states <= 0:
        return decode_once(ops, z_hat, y_hat)
    from adapted_codec import pad_repair as zp
    from adapted_codec.protocols.storage import store01
    zh, yh = z_hat, y_hat
    sel0 = S.context_free_count
    try:
        for i in range(S.stabilization_states):
            nxt = None
            for a, sel in enumerate(S.stabilization_candidates):
                S.context_free_count = sel
                out = decode_once(ops, zh, yh)
                H, W = out[0].shape[2:]
                if S.image_hw is not None and tuple(S.image_hw) != (H, W):
                    return out
                q2 = encode(ops, zp.reflect_pad(store01(out[0]), H, W), gen=2)
                ok = torch.equal(q2.z_hat, zh) and torch.equal(q2.y_hat, yh)
                if ok:
                    return out
                if a == 0:
                    nxt, last = q2, out

            zh, yh = nxt.z_hat, nxt.y_hat
        return last
    finally:
        S.context_free_count = sel0
