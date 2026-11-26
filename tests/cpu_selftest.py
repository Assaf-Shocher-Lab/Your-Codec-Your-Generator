"""CPU self-test with a stand-in codec (random weights, no downloads, no real images).

Checks: imports; wrapper identities E'(W(v, b)) = v and E'(D'(v)) = v; S_8 (values, ties to even, idempotence, PNG
round trip); one decode per protocol (float, clipped float, 8-bit stage 0) returns a record-consistent image; one
round of the file chain with a stand-in adapter.  Run: python tests/cpu_selftest.py"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapted_codec.chain import Adapter, Decoded, run_chain  # noqa: E402
from adapted_codec.codec_ops import MeanScaleOps  # noqa: E402
from adapted_codec.encoder import encode  # noqa: E402
from adapted_codec.pca_basis import fit_patch_pca  # noqa: E402
from adapted_codec.protocols import storage  # noqa: E402
from adapted_codec.protocols.clipped_float.range_repair import decode_clipped  # noqa: E402
from adapted_codec.protocols.eight_bit.decoder import decode as decode_8bit  # noqa: E402
from adapted_codec.protocols.float_chain import decode_float  # noqa: E402
from adapted_codec.settings import S  # noqa: E402
from adapted_codec.wrapper import build_wrapper  # noqa: E402

K, F = 8, 16


class _Likelihood:
    def __call__(self, v, *a, training=False):
        return None, torch.full_like(v, 0.5)


class StandInCodec(nn.Module):
    """Mean-scale hyperprior shape: g_a (stride 16), h_a (stride 2), h_s -> (scales, means), g_s."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.M = K
        self.g_a = nn.Conv2d(3, K, F, stride=F)
        self.h_a = nn.Conv2d(K, 4, 2, stride=2)
        self.h_s = nn.ConvTranspose2d(4, 2 * K, 2, stride=2)
        self.g_s = nn.ConvTranspose2d(K, 3, F, stride=F)
        for m in (self.g_a, self.h_a, self.h_s, self.g_s):
            nn.init.normal_(m.weight, std=0.05)
        self.gaussian_conditional = _Likelihood()

        class _EB:
            def __call__(self, z, training=False):
                return None, torch.full_like(z, 0.5)

            def _get_medians(self):
                return torch.zeros(1, 4, 1, 1)
        self.entropy_bottleneck = _EB()


class StandInOps(MeanScaleOps):
    def z_offset(self):
        return torch.zeros(1, 4, 1, 1)

    def hyper(self, y):
        z = self.net.h_a(y)
        return z, torch.round(z)

    def quantize_main(self, y, z_hat):
        scales, means = self.net.h_s(z_hat).chunk(2, 1)
        means = 0.1 * means
        return torch.round(y - means) + means, scales, means

    def base_decode(self, y_hat, z_hat=None):
        return self.net.g_s(y_hat).clamp(0, 1)


def build():
    net = StandInCodec().eval()
    g = torch.Generator().manual_seed(1)
    imgs = [torch.rand(1, 3, 64, 64, generator=g) * 2 - 1 for _ in range(4)]
    pca = fit_patch_pca(imgs, k=K, f=F, device="cpu")
    w = build_wrapper(net, pca, 10.0)
    return net, w, StandInOps(net, w)


def test_wrapper_identities():
    net, w, ops = build()
    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(2))
    v = torch.randn(1, K, 4, 4, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        assert torch.allclose(w.wrapped_encoder(ops.write(x, v)), v, atol=1e-4), "E'(W(v, b)) != v"
        d_prime = ops.write(ops.base_decode(v), v)                       # D'(v) = W(v, b(D(v)))
        assert torch.allclose(w.wrapped_encoder(d_prime), v, atol=1e-4), "E'(D'(v)) != v"


def test_s8():
    S.load({"chain_format": "png8"})
    x = torch.tensor([[[[-0.1, 0.0, 0.5 / 255, 1.5 / 255, 254.5 / 255, 1.2]]]]).expand(1, 3, 1, 6).contiguous()
    q = storage.store01(x)
    want = torch.tensor([0.0, 0.0, 0.0, 2 / 255, 254 / 255, 1.0])        # ties to even: 0.5 -> 0, 1.5 -> 2, 254.5 -> 254
    assert torch.allclose(q[0, 0, 0], want), q[0, 0, 0]
    assert torch.equal(storage.store01(q), q), "S_8 not idempotent"
    S.load({"chain_format": "png8", "fast_s8": 1})
    assert torch.equal(storage.search_s8(storage.store01)(x), q), "lookup-table S_8 != PNG path"


def _check_record(ops, x, q, gen=2):
    r = encode(ops, x, gen=gen)
    return torch.equal(r.z_hat, q.z_hat) and torch.equal(r.y_hat, q.y_hat)


def test_protocol_decodes():
    net, w, ops = build()
    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(4))
    base = dict(encoder="variant_a", search_start="proj", refine_steps=20)
    S.load(base)
    q = encode(ops, x, gen=1)
    xf, _, yt, _ = decode_float(ops, q.z_hat, q.y_hat)
    assert yt is not None and _check_record(ops, xf, q), "float: Sigma(E'(x)) != s"
    S.load(dict(base, chain_format="clamp", range_repair_steps=200))
    xc, _, yt, _ = decode_clipped(ops, q.z_hat, q.y_hat)
    assert float(xc.min()) >= 0 and float(xc.max()) <= 1 or yt is None
    S.load(dict(base, chain_format="png8", margin_y=0.06, correction_budget=30, context_free_count=1,
                lattice_steps=200, lattice_trials=200, lattice_time_cap=600.0, lattice_scan=8))
    x8, _, _, _ = decode_8bit(ops, q.z_hat, q.y_hat)
    assert torch.equal(storage.store01(x8), x8), "8-bit output not on the 8-bit grid"


def test_escalation_write():
    """The ladder from a selected target (CompressAI-zoo codecs, TCM, ELIC): stored image on the 8-bit grid, the stage
    index reported, settings restored; the ELIC start x_F = W(y*_F, b(D(y_hat))) re-encodes to y*_F."""
    from adapted_codec.protocols.eight_bit.escalation import escalation_write
    from adapted_codec.target_search import select_target
    net, w, ops = build()
    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(5))
    S.load(dict(encoder="variant_a", search_start="proj", refine_steps=20, chain_format="png8", margin_y=0.06,
                correction_budget=30, context_free_count=1, lattice_steps=200, lattice_trials=200,
                lattice_time_cap=600.0, lattice_scan=8, box_step=1, box_steps=20))
    q = encode(ops, x, gen=1)
    with torch.no_grad():
        yt, _ = select_target(ops, q.z_hat, q.y_hat)
        xr = ops.base_decode(q.y_hat, q.z_hat)
        xd, stage = escalation_write(ops, q.z_hat, q.y_hat, yt, xr)
    assert torch.equal(storage.store01(xd), xd), "ladder output not on the 8-bit grid"
    assert stage in (-1, 0, 1, 2, 3) and S.correction_budget == 30 and S.gamma_z == 0.02
    if yt is not None:
        with torch.no_grad():
            xf = ops.write(xr, yt)
            err = float((ops.wrapped_encode(xf) - yt).abs().max())
        assert err < 1e-3, f"E'(W(y*, b(x_raw))) != y* ({err})"


class StandInAdapter(Adapter):
    name, rate = "stand_in", "0"

    def __init__(self):
        self.net, self.w, self.ops = build()

    def encode_to_file(self, x_u8, g, path):
        x = torch.from_numpy(np.ascontiguousarray(x_u8)).permute(2, 0, 1)[None].float().div(255)
        q = encode(self.ops, x, gen=g)
        np.savez(path + ".npz", z=q.z_hat.numpy(), y=q.y_hat.numpy())
        os.replace(path + ".npz", path)
        return {"enc_tag": q.fp_stop, "bpp_est": ""}

    def decode_from_file(self, path):
        d = np.load(path)
        z, y = torch.from_numpy(d["z"]), torch.from_numpy(d["y"])
        x, _, _, _ = decode_8bit(self.ops, z, y)
        return Decoded(x, "01", [d["z"], d["y"]], "stand-in")


def test_file_chain():
    S.load(dict(encoder="variant_a", search_start="proj", refine_steps=20, chain_format="png8", margin_y=0.06,
                correction_budget=30, lattice_steps=200, lattice_trials=200, lattice_time_cap=600.0))
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        a = (np.random.default_rng(5).random((64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(a).save(os.path.join(d, "img.png"))
        run_chain(StandInAdapter(), os.path.join(d, "img.png"), os.path.join(d, "chain"), "img", 2, log=lambda m: None)
        assert os.path.exists(os.path.join(d, "chain", "DONE"))


class StandInBackend:
    """In-memory chain backend: the stand-in codec, a pickled-tensor file format, no padding."""
    rate, pm, dev = "0", 16, "cpu"

    def __init__(self):
        self.net, self.w, self.ops = build()
        self.ops.encode_bytes = lambda z, y: (([y.numpy().tobytes()], [z.numpy().tobytes()]),
                                              (z.numel() + y.numel()) // 8, True)
        shapes = {}

        class Coder:
            @staticmethod
            def encode(strings, h, w):
                return strings[1][0] + strings[0][0]

            @staticmethod
            def decode(data):
                z, y = shapes["z"], shapes["y"]
                nz = 4 * int(np.prod(z))
                return (torch.from_numpy(np.frombuffer(data[:nz], np.float32).reshape(z).copy()),
                        torch.from_numpy(np.frombuffer(data[nz:], np.float32).reshape(y).copy()), (64, 64))
        enc = self.ops.encode_bytes

        def encode_bytes(z, y):
            shapes.update(z=tuple(z.shape), y=tuple(y.shape))
            return enc(z, y)
        self.ops.encode_bytes = encode_bytes
        self.coder = Coder()


def test_memory_chain():
    from PIL import Image

    from adapted_codec.memory_chain import run_memory_chain
    base = dict(encoder="variant_a", search_start="proj", refine_steps=20)
    psnr = {"psnr": lambda a, b: -10 * torch.log10(((a - b) ** 2).mean())}
    with tempfile.TemporaryDirectory() as d:
        a = (np.random.default_rng(7).random((64, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(a).save(os.path.join(d, "img.png"))
        for proto, dec, extra in (("float", decode_float, {}),
                                  ("clipped_float", decode_clipped, dict(chain_format="clamp", range_repair_steps=200))):
            S.load(dict(base, **extra))
            rows = run_memory_chain(StandInBackend(), dec, os.path.join(d, "img.png"), os.path.join(d, proto), "img",
                                    3, "adapted", psnr, proto, save_rounds={1}, log=lambda m: None)
            assert len(rows) == 3 and all(r["roundtrip"] == 1 for r in rows), proto
            assert os.path.exists(os.path.join(d, proto, "recon", "g001.png"))


def test_aeic_wrapper():
    """AEIC wrapper (32 x 32 patches, the released basis, a stand-in strided g_a): E'(W(x, t)) = t."""
    from adapters.aeic.wrapper import PatchPCA, WrappedEncoder
    pca = torch.load(os.path.join(ROOT, "pca", "pca_clic585_f32.pt"), map_location="cpu", weights_only=True)
    g_a = torch.nn.Conv2d(3, 320, 32, stride=32)
    torch.nn.init.normal_(g_a.weight, std=0.01, generator=torch.Generator().manual_seed(4))
    we = WrappedEncoder(g_a, PatchPCA(pca, k=320, patch=32), c=10.0)
    x = torch.rand(1, 3, 64, 96, generator=torch.Generator().manual_seed(5)) * 2 - 1
    t = torch.randn(1, 320, 2, 3, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        assert torch.allclose(we(we.write(x, t)), t, atol=1e-3), "AEIC E'(W(x, t)) != t"


def test_nlbp_decode():
    """NLBP with the released codec interface on the stand-in codec and a stand-in prior (clean estimate = the base):
    E'(W(y, b)) = y of the interface, then E'(x_out) = y* after the decode (images in [-1, 1])."""
    from nlbp.codec_interface import NlbpCodec
    from nlbp.priors.ddpm import snap_ddpm
    from nlbp.receiver import nlbp_state
    from nlbp.sampler import nlbp_decode, resolve
    from types import SimpleNamespace
    net, w, ops = build()
    codec = NlbpCodec(net, w)
    y_hat = torch.randn(1, K, 4, 4, generator=torch.Generator().manual_seed(8))
    y_star = y_hat + 0.05 * torch.randn(1, K, 4, 4, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        b = codec.complement(torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(10)) * 2 - 1)
        assert torch.allclose(codec.wrapper.wrapped_encoder(codec.write_step(y_star, b)), y_star, atol=1e-3),             "NLBP codec interface: E'(W(y, b)) != y"
        state = nlbp_state("x", SimpleNamespace(y_hat=y_hat, z_hat=torch.zeros(1, 4, 2, 2), y_t=y_star), codec)
        base = state["Dp"]
        prm = resolve(dict(t0=20, stride=5, lambda2=0.3, lambda1=0.1, kz=0.2, target="yhat", upsample=1, opt_K=3),
                      snap_ddpm, 1.0)
        out, n = nlbp_decode(state, prm, lambda x, s: (x - base) / s, codec, snap_ddpm, 1.0, "cpu")
    assert n == 5 and torch.allclose(codec.wrapper.wrapped_encoder(out), y_star, atol=1e-3), "NLBP: E'(x_out) != y*"


def test_nlbp_decode_latent():
    """NLBP with a latent-space prior behind a stand-in client (Enc = Dec = identity, clean estimate = the base) and the
    two-tile blending of the 512 priors: E'(x_out) = y*; the tile weights sum to 1 on a 768 x 512 image."""
    from nlbp.codec_interface import NlbpCodec
    from nlbp.priors.tiles import ramp_weight, tiles512
    from nlbp.receiver import nlbp_state
    from nlbp.sampler import nlbp_decode_latent, resolve
    from nlbp.schedule import snap_s
    from types import SimpleNamespace
    net, w, ops = build()
    codec = NlbpCodec(net, w)
    y_hat = torch.randn(1, K, 4, 4, generator=torch.Generator().manual_seed(11))
    y_star = y_hat + 0.05 * torch.randn(1, K, 4, 4, generator=torch.Generator().manual_seed(12))
    with torch.no_grad():
        state = nlbp_state("x", SimpleNamespace(y_hat=y_hat, z_hat=torch.zeros(1, 4, 2, 2), y_t=y_star), codec)
        base = state["Dp"]

        class Client:
            def encode(self, x):
                return x.clone()

            def decode(self, z):
                return z.clone()

            def velocity(self, z, s, prompt, cfg):
                return (z - base) / s
        prm = resolve(dict(s0=0.45, stride=10, lambda2=1.0, lambda1=0.35, kz=1.0, target="yhat", opt_K=3,
                           vae_residual=1, latent=1), snap_s, 4.0)
        out, n = nlbp_decode_latent(state, prm, Client(), codec, "cpu")
    assert n == prm["n_steps"] and torch.allclose(codec.wrapper.wrapped_encoder(out), y_star, atol=1e-3),         "latent NLBP: E'(x_out) != y*"
    wsum = torch.zeros(512, 768)
    for i0, j0 in tiles512(512, 768, 512):
        wsum[i0:i0 + 512, j0:j0 + 512] += ramp_weight(512, 768, i0, j0, 512, "cpu")
    assert torch.allclose(wsum, torch.ones_like(wsum)), "tile weights do not sum to 1"


if __name__ == "__main__":
    for t in (test_wrapper_identities, test_s8, test_protocol_decodes, test_escalation_write, test_file_chain,
              test_memory_chain, test_aeic_wrapper, test_nlbp_decode, test_nlbp_decode_latent):
        t()
        print("PASS", t.__name__)
