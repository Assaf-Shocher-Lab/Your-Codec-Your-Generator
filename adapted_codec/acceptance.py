"""Acceptance of a candidate target t (Sec. 3.2): the codec's own quantizers reproduce the received record,
Sigma(t) = s, and the hyper margin holds: |h_a(t) - z_offset - s_z| <= 1/2 - m_z for every hyper element."""
import torch

from adapted_codec.settings import S


def is_feasible(ops, y_, off, k, zh, yh):
    """True iff t passes the margin test and Sigma(t) = (z_hat, y_hat)."""
    viol = ((ops.hyper(y_)[0] - off - k).abs() - (0.5 - S.margin_z)).clamp_min(0)
    return (float(viol.max()) == 0.0 and torch.equal(ops.hyper(y_)[1], zh)
            and torch.equal(ops.quantize_main(y_, zh)[0], yh))
