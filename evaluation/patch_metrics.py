"""The scorer of stored images (8-bit chains): the image as stored (u8 / 255) against the original file (u8 / 255).

  paired   PSNR (peak 1 on u8 / 255 = peak 255), MS-SSIM, LPIPS, DISTS (evaluation/metrics.py PairedMetrics)
  patches  kodak64: non-overlapping 64 x 64 patches, no shift (96 per 768 x 512 image, 2,304 for Kodak's 24 images);
           2k256: 256 x 256 patches plus the half-patch shifted grid (high-resolution sets)
FID / KID of a set: fid_kid_both(real, fake) -> (float32 features), (float16 features = the reported values)."""
import numpy as np
import torch

PROTOCOLS = {"kodak64": dict(patch_size=64, split_patch_num=1), "2k256": dict(patch_size=256, split_patch_num=2)}


def u8_to_t01(u8, device):
    t = torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).contiguous().float().div(255.0)
    return t.unsqueeze(0).to(device)


class ImageScorer:
    def __init__(self, device=None):
        from evaluation.fid_kid import PatchFeatureStore
        from evaluation.metrics import PairedMetrics
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.paired = PairedMetrics(self.device)

        for _m in self.paired.metric_paired_dict.values():
            if hasattr(_m, "dummy_param"):
                _m.dummy_param = _m.dummy_param.to(self.device)
                _m.device = torch.device(self.device)

        self.inception = PatchFeatureStore(self.device).inception

    @torch.no_grad()
    def score(self, x_u8, x0_u8):
        m = self.paired.metric_paired_dict
        a, b = u8_to_t01(x_u8, self.device), u8_to_t01(x0_u8, self.device)
        out = {}
        for key in ("psnr", "ms_ssim", "lpips", "dists"):
            out[key] = float(m[key](a, b).item())
        return out

    @torch.no_grad()
    def patch_features(self, img_u8, protocol, real=False, ref_u8=None):
        from evaluation import fid_kid as fk
        from _update_patch_fid import update_patch_fid
        p = PROTOCOLS[protocol]
        img = u8_to_t01(img_u8, self.device)
        ref = img if ref_u8 is None else u8_to_t01(ref_u8, self.device)
        out = []
        sink = fk._Sink(self.inception, real, out)
        if real:
            update_patch_fid(img, img, kid_metric=sink, **p)
        else:
            update_patch_fid(ref, img, kid_metric=sink, **p)
        return torch.cat(out)


def fid_kid_both(real32, fake32):
    from evaluation.fid_kid import fid_kid
    return fid_kid(real32, fake32), fid_kid(real32.half(), fake32.half())
