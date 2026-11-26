"""Patch FID / KID (App. A): Inception features (torchmetrics' FID extractor) of image patches cut by update_patch_fid
(third_party/neuralcompression), FID and KID with torchmetrics' formulas from the features (KID: subset size
min(1000, n), 100 subsets, seed 0). The reported values use float16 features (fid_kid_both in patch_metrics.py)."""
import os
import sys
from collections import defaultdict

import torch
from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party",
                                "neuralcompression"))
from _update_patch_fid import update_patch_fid  # noqa: E402,F401  (NeuralCompression's function, unchanged)


class _Sink:
    def __init__(self, inception, keep_real, out):
        self.inception, self.keep_real, self.out = inception, keep_real, out

    def update(self, imgs, real):
        if real == self.keep_real:
            self.out.append(self.inception(imgs).float().cpu())


class PatchFeatureStore:
    def __init__(self, device):
        self.inception = FrechetInceptionDistance().inception.to(device).eval()
        self.feats = defaultdict(list)
        self._gt_of = None


def fid_kid(real, fake, seed=0):
    fid = FrechetInceptionDistance(feature=2048)
    r, f = real.double(), fake.double()
    fid.real_features_sum = r.sum(0); fid.real_features_cov_sum = r.t().mm(r); fid.real_features_num_samples = torch.tensor(len(r))
    fid.fake_features_sum = f.sum(0); fid.fake_features_cov_sum = f.t().mm(f); fid.fake_features_num_samples = torch.tensor(len(f))
    fid.orig_dtype = torch.float32
    fid._update_count = 1
    fid_v = float(fid.compute())
    kid = KernelInceptionDistance(subset_size=min(1000, len(real), len(fake)))
    kid.real_features, kid.fake_features = [real.float()], [fake.float()]
    torch.manual_seed(seed)
    km, ks = kid.compute()
    return fid_v, float(km), float(ks)
