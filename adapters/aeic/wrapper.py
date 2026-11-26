"""The patch-PCA wrapper around AEIC's analysis transform g_a (Sec. 3.1), in the arithmetic of the reported AEIC runs.

One f x f x 3 patch (f = 32 = one AEIC latent cell) has 3 f^2 = 3072 coefficients; U, mu = patch PCA (rows =
directions), fitted on 585 CLIC 2020 Professional training images (pca/pca_clic585_f32.pt). a = the k = 320 weak
coefficients (the lowest-variance directions, the shape of AEIC's latent), b = the other 2752. J(a, b) joins them.
    E'(x)    = g_a(J(0, b(x))) + c a(x)                 wrapped encoder
    W(x, t)  = J((t - g_a(J(0, b(x)))) / c, b(x))       write: keeps b of x, sets a so that E'(W(x, t)) = t
Same formula as adapted_codec/wrapper.py; this file keeps the operation order of the AEIC runs (einsum over channels,
pixel_unshuffle / pixel_shuffle, float32 casts), so its outputs are bit-identical to them."""
import torch
import torch.nn.functional as F


def channel_mix(w, v):
    return torch.einsum("oc,bchw->bohw", w, v)


def weak_first(pca, k):
    order = pca["std"].argsort()
    idx = torch.cat([order[:k], order[k:].flip(0)])
    return {**pca, "U": pca["U"][idx], "std": pca["std"][idx], "k": k}


class PatchPCA:
    def __init__(self, pca, k, patch):
        pca = weak_first(pca, k)
        assert pca["U"].shape == (3 * patch * patch, 3 * patch * patch) and pca["mu"].shape == (3 * patch * patch,)
        self.k, self.patch = k, patch
        self.U = pca["U"].detach().float().clone()
        self.mu = pca["mu"].detach().float().clone()

    def to(self, device):
        self.U, self.mu = self.U.to(device), self.mu.to(device)
        return self

    def split(self, x):
        u = channel_mix(self.U, F.pixel_unshuffle(x.float(), self.patch) - self.mu.view(1, -1, 1, 1))
        return u[:, :self.k], u[:, self.k:]

    def join(self, a, b):
        p = channel_mix(self.U.T, torch.cat([a, b], 1)) + self.mu.view(1, -1, 1, 1)
        return F.pixel_shuffle(p, self.patch)


class WrappedEncoder:
    def __init__(self, g_a, pca, c=1.0):
        self.g_a, self.pca = g_a, pca
        c = torch.as_tensor(c, dtype=torch.float32)
        self.c = (c.expand(pca.k) if c.ndim == 0 else c).clone()
        assert (self.c > 0).all()

    def to(self, device):
        self.pca.to(device)
        self.c = self.c.to(device)
        return self

    @property
    def k(self):
        return self.pca.k

    def E(self, x):
        return self.g_a(x.to(torch.float32)).float()

    def __call__(self, x):
        a, b = self.pca.split(x)
        return self.E(self.pca.join(torch.zeros_like(a), b)) + self.c.view(1, -1, 1, 1) * a

    def write(self, x, y_target):
        _, b = self.pca.split(x)
        x_no_weak = self.pca.join(torch.zeros(len(y_target), self.k, *b.shape[2:], device=y_target.device), b)
        return self.pca.join((y_target - self.E(x_no_weak)) / self.c.view(1, -1, 1, 1), b)
