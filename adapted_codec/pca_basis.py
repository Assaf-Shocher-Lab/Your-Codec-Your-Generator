"""The patch-PCA basis, the only fitted component (Sec. 3.1): covariance of all f x f x 3 patches of the fitting
images, eigenvectors sorted with the k lowest-variance directions first (U_0), then the rest by decreasing variance
(U_1). Fitted once per patch size (32: 585 CLIC 2020 Professional training images, AEIC; 16: 1,000 Vimeo90K frames,
all other codecs) and reused across rates, protocols and generative decoders."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def fit_patch_pca(batches, k=4, f=8, device="cuda"):
    d = 3 * f * f
    s = torch.zeros(d, dtype=torch.float64, device=device)
    ss = torch.zeros(d, d, dtype=torch.float64, device=device)
    n = 0
    for x in batches:
        p = F.pixel_unshuffle(x.to(device, torch.float64), f).permute(0, 2, 3, 1).reshape(-1, d)
        s += p.sum(0); ss += p.T @ p; n += len(p)
    mu = s / n
    cov = ss / n - torch.outer(mu, mu)
    lam, V = torch.linalg.eigh(cov)
    U = torch.cat([V[:, :k], V[:, k:].flip(1)], 1).T
    std = torch.cat([lam[:k], lam[k:].flip(0)]).clamp(min=0).sqrt()
    return {"U": U.float().cpu(), "mu": mu.float().cpu(), "k": k, "std": std.float().cpu(), "n_patches": n}


def with_k(pca, k):
    order = pca["std"].argsort()
    idx = torch.cat([order[:k], order[k:].flip(0)])
    return {**pca, "U": pca["U"][idx], "std": pca["std"][idx], "k": k}


def pca_unit_range(pca):
    return {**pca, "mu": (pca["mu"] + 1) / 2, "std": pca["std"] / 2,
            "source": f"{pca.get('source', '?')}; mapped to [0, 1]"}
