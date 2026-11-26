"""HPCM-Large (MSE) behind the codec interface (adapted_codec/codec_ops.py).

Quantizers: z_hat = round(z - means_hyper) + means_hyper; y_hat = round(y - mu) + mu with mu and the scale from h_s(z_hat)
and the hierarchical progressive context (2 + 3 + 6 coding steps, each reading z_hat and earlier y_hat only).
quantize_main = the model's forward_hpcm in eval mode, which is what compress() quantizes. The wrapper uses f = 16
patches (g_a has stride 16), k = 320. Entropy coder: the upstream repository ships its rANS only as cpython-3.8
binaries, so it is bound to CompressAI's rANS and pmf_to_quantized_cdf (same signatures, 16-bit CDFs); the CDF tables
are HPCM's own (scale table exp(linspace(log .12, log 64, 60)) as in its test.py)."""
import math
import os
import sys
import types

import torch

from adapted_codec.codec_ops import MeanScaleOps

HPCM_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party",
                        "hpcm")
WEIGHTS = os.path.join("weights", "hpcm", "HPCM_Large_mse_{}.pth")    # set by the runner from the variant's yaml


def _install_shims():
    from compressai._CXX import pmf_to_quantized_cdf
    from compressai.ans import BufferedRansEncoder, RansDecoder
    cxx = types.ModuleType("src.entropy_models._CXX")
    cxx.pmf_to_quantized_cdf = pmf_to_quantized_cdf

    def _l(a):
        return a.tolist() if hasattr(a, "tolist") else list(a)

    class ubransEncoder:
        def __init__(self):
            self.e = BufferedRansEncoder()

        def encode_with_indexes(self, sym, idx, cdf, ln, off):
            self.e.encode_with_indexes(_l(sym), _l(idx), _l(cdf), _l(ln), _l(off))

        def flush(self):
            return self.e.flush()

    class ubransDecoder:
        def __init__(self):
            self.wrapper = RansDecoder()

        def set_stream(self, s):
            self.wrapper.set_stream(s)

        def decode_stream(self, idx, cdf, ln, off):
            return self.wrapper.decode_stream(_l(idx), _l(cdf), _l(ln), _l(off))

    ua = types.ModuleType("src.entropy_models.unbounded_ans")
    ua.ubransEncoder, ua.ubransDecoder = ubransEncoder, ubransDecoder

    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    for name in ("src", "src.models", "src.utils"):
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(HPCM_SRC, *name.split("."))]
        sys.modules[name] = m
    sys.modules["src.entropy_models._CXX"] = cxx
    sys.modules["src.entropy_models.unbounded_ans"] = ua
    try:
        import einops
    except ImportError:
        ein = types.ModuleType("einops")

        def rearrange(x, pat, **k):
            pat = " ".join(pat.split())
            if pat == "b c (w1 p1) (w2 p2) -> (b w1 w2) c p1 p2":
                b, c, H, W = x.shape
                p1, p2 = k["p1"], k["p2"]
                w1, w2 = H // p1, W // p2
                return x.reshape(b, c, w1, p1, w2, p2).permute(0, 2, 4, 1, 3, 5).reshape(b * w1 * w2, c, p1, p2)
            if pat == "(b w1 w2) c p1 p2 -> b c (w1 p1) (w2 p2)":
                n, c, p1, p2 = x.shape
                w1, w2 = k["w1"], k["w2"]
                b = n // (w1 * w2)
                return x.reshape(b, w1, w2, c, p1, p2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, w1 * p1, w2 * p2)
            raise NotImplementedError(pat)
        ein.rearrange = rearrange
        sys.modules["einops"] = ein


_install_shims()
if HPCM_SRC not in sys.path:
    sys.path.insert(0, HPCM_SRC)


def load_hpcm(lam, dev):
    from adapters.hpcm.upstream_copies import HPCM

    ck = torch.load(os.path.normpath(WEIGHTS.format(lam)), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    net = HPCM()
    bufs = {n for n, _ in net.named_buffers()}

    miss, unex = torch.nn.Module.load_state_dict(net,{k: v for k, v in sd.items() if k not in bufs}, strict=False)
    miss = [k for k in miss if k not in bufs]

    miss = [k for k in miss if not k.startswith("adaptive_params_list.")]
    assert not miss and not unex, (miss[:5], unex[:5])
    net = net.eval().to(dev)

    import scipy.stats
    from src.entropy_models import entropy_models as _em
    for _cls in vars(_em).values():
        if isinstance(_cls, type) and "_standardized_quantile" in vars(_cls) and "beta" in \
                _cls._standardized_quantile.__code__.co_names:
            _cls._standardized_quantile = lambda self, q: torch.as_tensor(scipy.stats.gennorm.ppf(
                q, torch.as_tensor(self.beta).detach().cpu().numpy()), dtype=torch.float32,
                device=torch.as_tensor(self.beta).device)
    net.update(torch.exp(torch.linspace(math.log(0.12), math.log(64), 60)).to(dev))
    net.M = 320
    return net


class _GC:
    def __init__(self, net):
        self.net = net

    def __call__(self, y, scales, means=None, training=False):
        return None, self.net.entropy_estimation(y, scales, means)


class _EB:
    def __init__(self, net):
        self.net = net

    def __call__(self, z_hat, training=False):
        n = self.net
        return None, n.entropy_estimation(z_hat - n.means_hyper, n.scales_hyper.expand_as(z_hat))


class HpcmOps(MeanScaleOps):
    def __init__(self, net, wrapper):
        self.net, self.wrapper = net, wrapper
        self.gaussian_conditional, self.entropy_bottleneck = _GC(net), _EB(net)

    def z_offset(self):
        return self.net.means_hyper

    def _ctx(self, y, params):
        n = self.net
        return n.forward_hpcm(y, params, n.y_spatial_prior_adaptor_list_s1, n.y_spatial_prior_s1_s2,
                              n.y_spatial_prior_adaptor_list_s2, n.y_spatial_prior_s1_s2,
                              n.y_spatial_prior_adaptor_list_s3, n.y_spatial_prior_s3,
                              n.adaptive_params_list, n.context_net)

    @torch.no_grad()
    def quantize_main(self, y, z_hat):
        _, y_q, y_hat, scales = self._ctx(y, self.net.h_s(z_hat))
        return y_hat, scales, y_hat - y_q

    @torch.no_grad()
    def encode_bytes(self, z_hat, y_hat):
        from src.entropy_models import ubransEncoder
        n = self.net
        z_res = torch.round(z_hat - n.means_hyper)
        ez = ubransEncoder()
        n.compress_symbols(z_res, n.build_indexes_z(z_res.size()), n.quantized_cdf_z.cpu().numpy(),
                           n.cdf_length_z.cpu().numpy(), n.offset_z.cpu().numpy(), ez)
        zs = ez.flush()
        params = n.h_s(z_res + n.means_hyper)
        yq_l, sc_l = n.compress_hpcm(y_hat, params, n.y_spatial_prior_adaptor_list_s1, n.y_spatial_prior_s1_s2,
                                     n.y_spatial_prior_adaptor_list_s2, n.y_spatial_prior_s1_s2,
                                     n.y_spatial_prior_adaptor_list_s3, n.y_spatial_prior_s3,
                                     n.adaptive_params_list, n.context_net)
        ey = ubransEncoder()
        for q, s in zip(yq_l, sc_l):
            n.compress_symbols(q, n.build_indexes_conditional(s), n.quantized_cdf_y.cpu().numpy(),
                               n.cdf_length_y.cpu().numpy(), n.offset_y.cpu().numpy(), ey)
        ys = ey.flush()
        xo = n.decompress([ys, zs], z_res.size()[2:])["x_hat"]
        ok = bool(torch.equal(z_res + n.means_hyper, z_hat) and torch.equal(xo, n.g_s(y_hat).clamp(0, 1)))
        return [[ys], [zs]], len(ys) + len(zs), ok
