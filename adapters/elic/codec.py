"""ELIC (the re-implementation of He et al., CVPR 2022; external dependency, third_party/elic/UPSTREAM.md) behind the
codec interface (adapted_codec/codec_ops.py).

Quantizers: z_hat = EntropyBottleneck median rounding of h_a(y); y in 5 uneven channel slices [16, 16, 32, 64, 192],
each with a checkerboard pass: anchor means from h_s(z_hat) and the channel context (y_hat of slice 0 and of the
previous slice), then non-anchor means from the anchors of the same slice; y_hat = round(y - mu) + mu. mu depends only
on z_hat and earlier y_hat, as the target search assumes. quantize_main = TestModel.compress's quantizer (batched);
encode_bytes = TestModel.compress's coding loop on the record (10 y strings + z), decoded back with its decompress
loop and with net.decompress (decode_ok). Wrapper: f = 16 (g_a has stride 16), k = net.M = 320."""
import os
import sys
import types

import torch

from adapted_codec.codec_ops import MeanScaleOps

ELIC_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party",
                        "elic", "src")
WEIGHTS = os.path.join("weights", "elic", "ELIC_lambda{}.pth.tar")     # set by the runner from the variant's yaml

for _m in ("thop", "ptflops"):          # profiling-only imports of Network.py (not needed at inference)
    if _m not in sys.modules:
        _s = types.ModuleType(_m)
        _s.profile = _s.get_model_complexity_info = None
        sys.modules[_m] = _s
if ELIC_SRC not in sys.path:
    sys.path.insert(0, ELIC_SRC)


def load_elic(lam, dev):
    from compressai.zoo import load_state_dict

    import types
    import compressai.entropy_models
    import compressai.models.priors as _pri
    if not hasattr(_pri, "GaussianConditional"):
        _pri.GaussianConditional = compressai.entropy_models.GaussianConditional

    import compressai.ops as _ops
    if not hasattr(_ops, "ste_round"):
        _ops.ste_round = lambda x: (torch.round(x) - x).detach() + x
    for mod, attr in (("thop", "profile"), ("ptflops", "get_model_complexity_info")):
        try:
            __import__(mod)
        except ImportError:
            m = types.ModuleType(mod)
            setattr(m, attr, lambda *a, **k: None)
            sys.modules[mod] = m
    from Network import TestModel
    ck = torch.load(os.path.normpath(WEIGHTS.format(lam)), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = load_state_dict(sd)
    net = TestModel.from_state_dict(sd).eval().to(dev)
    net.update(force=True)
    return net


def _ab(t):
    m = torch.zeros_like(t, dtype=torch.bool)
    m[:, :, 0::2, 0::2] = True
    m[:, :, 1::2, 1::2] = True
    return m


def _pack(t, anchor):
    B, C, H, W = t.shape
    o = torch.zeros(B, C, H, W // 2, device=t.device, dtype=t.dtype)
    if anchor:
        o[:, :, 0::2, :], o[:, :, 1::2, :] = t[:, :, 0::2, 0::2], t[:, :, 1::2, 1::2]
    else:
        o[:, :, 0::2, :], o[:, :, 1::2, :] = t[:, :, 0::2, 1::2], t[:, :, 1::2, 0::2]
    return o


def _unpack(p, anchor, like):
    o = torch.zeros_like(like)
    if anchor:
        o[:, :, 0::2, 0::2], o[:, :, 1::2, 1::2] = p[:, :, 0::2, :], p[:, :, 1::2, :]
    else:
        o[:, :, 0::2, 1::2], o[:, :, 1::2, 0::2] = p[:, :, 0::2, :], p[:, :, 1::2, :]
    return o


class ElicOps(MeanScaleOps):
    def _support(self, lm, ls, yhs, i):
        net = self.net
        if i == 0:
            return torch.cat([lm, ls], 1)
        support = yhs[0] if i == 1 else torch.cat([yhs[0], yhs[i - 1]], 1)
        cm, cs = net.cc_transforms[i - 1](support).chunk(2, 1)
        return torch.cat([cm, cs, lm, ls], 1)

    def _slices(self, z_hat, get_slice):
        net = self.net
        lm, ls = net.h_s(z_hat).chunk(2, 1)
        B, _, H, W = lm.shape
        yhs, S, MU = [], [], []
        for i, g in enumerate(net.groups[1:]):
            support = self._support(lm, ls, yhs, i)
            ma, sa = net.ParamAggregation[i](torch.cat([torch.zeros(B, 2 * g, H, W, device=lm.device), support],
                                                       1)).chunk(2, 1)
            ya = get_slice(i, ma, sa, True)
            mn, sn = net.ParamAggregation[i](torch.cat([net.context_prediction[i](ya), support], 1)).chunk(2, 1)
            yn = get_slice(i, mn, sn, False)
            a = _ab(ma)
            yhs.append(ya + yn)
            S.append(torch.where(a, sa, sn))
            MU.append(torch.where(a, ma, mn))
        return torch.cat(yhs, 1), torch.cat(S, 1), torch.cat(MU, 1)

    @torch.no_grad()
    def quantize_main(self, y, z_hat):
        ys = torch.split(y, self.net.groups[1:], 1)
        gc = self.gaussian_conditional

        def q(i, mu, sc, anchor):
            p = gc.quantize(_pack(ys[i], anchor), "symbols", _pack(mu, anchor)) + _pack(mu, anchor)
            return _unpack(p, anchor, ys[i])
        return self._slices(z_hat, q)

    @torch.no_grad()
    def encode_bytes(self, z_hat, y_hat):
        net, gc = self.net, self.gaussian_conditional
        z_strings = self.entropy_bottleneck.compress(z_hat)
        zd = self.entropy_bottleneck.decompress(z_strings, z_hat.shape[-2:])
        ys = torch.split(y_hat, net.groups[1:], 1)
        y_strings = []

        def enc(i, mu, sc, anchor):
            idx = gc.build_indexes(_pack(sc, anchor))
            s = gc.compress(_pack(ys[i], anchor), idx, means=_pack(mu, anchor))
            y_strings.append(s)
            p = gc.decompress(s, idx, means=_pack(mu, anchor))
            return _unpack(p, anchor, ys[i])
        ye, _, _ = self._slices(zd, enc)
        it = iter(y_strings)

        def dec(i, mu, sc, anchor):
            idx = gc.build_indexes(_pack(sc, anchor))
            return _unpack(gc.decompress(next(it), idx, means=_pack(mu, anchor)), anchor, ys[i])
        yd, _, _ = self._slices(zd, dec)
        strings = [[[y_strings[2 * i], y_strings[2 * i + 1]] for i in range(len(ys))], z_strings]
        xo = net.decompress(strings, z_hat.shape[-2:])["x_hat"]
        ok = bool(torch.equal(zd, z_hat) and torch.equal(ye, y_hat) and torch.equal(yd, y_hat)
                  and torch.equal(xo, net.g_s(y_hat).clamp(0, 1)))
        flat = tuple(s[0] for s in y_strings)
        return [[flat], z_strings], sum(len(s) for s in flat) + sum(len(s) for s in z_strings), ok
