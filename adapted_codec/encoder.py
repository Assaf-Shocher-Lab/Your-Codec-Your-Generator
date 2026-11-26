"""Encoder side of the chain (Sec. 4 convention): round 1 sends the base codec's record Q(E(x)); every later round
sends the wrapped record Q(E'(x)). The encoder runs no search; the decoder does all the work."""
from types import SimpleNamespace

import torch

from adapted_codec.settings import S


def pair_bits(ops, z_hat, y_hat, scales, means):
    """Estimated bits of the record under the codec's entropy models."""
    return float(-torch.log2(ops.gaussian_conditional(y_hat, scales, means, training=False)[1]).sum()
                 - torch.log2(ops.entropy_bottleneck(z_hat, training=False)[1]).sum())


def symbol_record(ops, y):
    """The transmitted record of a continuous latent y: (z_hat, y_hat) and its bits."""
    z, z_hat = ops.hyper(y)
    y_hat, sc, mu = ops.quantize_main(y, z_hat)
    return SimpleNamespace(z_hat=z_hat, y_hat=y_hat, y=y, fp_steps=0, fp_residual=0,
                           est_bits=pair_bits(ops, z_hat, y_hat, sc, mu))


@torch.no_grad()
def encode(ops, x, fp_iters=100, check=None, gen=None):
    """Round `gen` of the chain: gen 1 sends Q(E(x)) (the codec's own record), gen >= 2 sends Q(E'(x))."""
    if S.encoder == "variant_a":
        if not isinstance(gen, int) or isinstance(gen, bool) or gen < 1:
            raise ValueError(f"encoder variant_a needs the generation number gen >= 1, got {gen!r}")
        q = symbol_record(ops, ops.base_encode(x) if gen == 1 else ops.wrapped_encode(x))
        q.fp_stop, q.y_t = ("zsA_plain" if gen == 1 else "zsA_wrapped"), None
        return q
    raise ValueError(f"unknown encoder {S.encoder!r}: only the round-based encoder is included")
