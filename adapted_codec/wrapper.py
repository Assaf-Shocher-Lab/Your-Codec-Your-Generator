"""Right inverses and the pretrained wrapper (Sec. 3.1).

Per f x f x 3 pixel patch, an orthonormal patch-PCA basis U = [U_0; U_1] with mean mu (U_0: the k lowest-variance
rows, k = the codec's latent channels):
    a(x) = U_0 (x - mu)                 weak coordinates            split(x)[0]
    b(x) = U_1 (x - mu)                 complementary coordinates   split(x)[1]
    J(a, b) = mu + U_0^T a + U_1^T b                                join(a, b)
    f(b) = E(J(0, b))
    E'(x) = f(b(x)) + c a(x)            wrapped encoder             wrapped_encoder(x)
    W(v, b) = J((v - f(b)) / c, b)      latent write, E'(W(v, b)) = v; the codec interface's write(x, v) = W(v, b(x))
    D'(v) = W(v, b(D(v)))               right inverse, E' D' = Id: write(base_decode(v), v) of the codec interface
E is the codec's analysis transform (AnalysisTransform exposes g_a with the interface the wrapper calls). Only the
basis is fitted; c > 0 is the gain."""
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapted_codec.pca_basis import pca_unit_range, with_k

PATCH = 16          # patch size f of every codec except AEIC (stride-16 analysis transforms)


def mix(w, v):
    """(out, in) matrix applied over the channel axis of a (B, in, H, W) map."""
    return torch.einsum("oc,bchw->bohw", w, v)


class PCAWrapper(nn.Module):
    def __init__(self, vae, U, mu, *, k, c, f):
        super().__init__()
        assert U.shape == (3 * f * f, 3 * f * f) and mu.shape == (3 * f * f,)
        self.vae = vae.eval().requires_grad_(False)
        self.k, self.f = k, f
        self.register_buffer("U", U.detach().float().clone())
        self.register_buffer("mu", mu.detach().float().clone())
        c = torch.as_tensor(c, dtype=torch.float32)
        self.register_buffer("c", (c.expand(k) if c.ndim == 0 else c).clone())
        assert (self.c > 0).all()

    @classmethod
    def from_pca(cls, vae, pca, **kw):
        return cls(vae, pca["U"], pca["mu"], k=pca["k"], **kw)

    def split(self, x):
        u = mix(self.U, F.pixel_unshuffle(x.float(), self.f) - self.mu.view(1, -1, 1, 1))
        return u[:, :self.k], u[:, self.k:]

    def join(self, a, b):
        p = mix(self.U.T, torch.cat([a, b], 1)) + self.mu.view(1, -1, 1, 1)
        return F.pixel_shuffle(p, self.f)

    def E(self, x):
        return self.vae.encode(x.to(self.vae.dtype)).latent_dist.mean.float()

    def wrapped_encoder(self, x):
        a, b = self.split(x)
        return self.E(self.join(torch.zeros_like(a), b)) + self.c.view(1, -1, 1, 1) * a


class AnalysisTransform(nn.Module):
    dtype = torch.float32

    def __init__(self, g_a):
        super().__init__()
        self.g_a = g_a
        self.config = SimpleNamespace(scaling_factor=1.0)

    def encode(self, x, return_dict=True):
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=self.g_a(x)))


def build_wrapper(net, pca, c):
    """The wrapper around net.g_a for images in [0, 1]: basis fitted on [-1, 1] images, mapped exactly to [0, 1],
    k = net.M weak coefficients, gain c, patch size PATCH.
    """
    t = PCAWrapper.from_pca(AnalysisTransform(net.g_a), with_k(pca_unit_range(pca), net.M), c=c, f=PATCH)
    return t.to(next(net.parameters()).device)
