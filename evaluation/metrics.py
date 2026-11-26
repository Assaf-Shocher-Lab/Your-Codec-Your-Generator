"""Per-image quality against the original (App. A): pyiqa PSNR, MS-SSIM and DISTS, torchmetrics LPIPS (AlexNet,
normalize=True), inputs in [0, 1]. Each chain scores x_g.clamp(0, 1) against x_0 = the 8-bit original / 255."""
import pyiqa
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

PAIRED = ("psnr", "ms_ssim", "lpips", "dists")


def build_metrics(device):
    m = {}
    m["psnr"] = pyiqa.create_metric('psnr').to(device)
    m["dists"] = pyiqa.create_metric('dists').to(device)
    m["ms_ssim"] = pyiqa.create_metric('ms_ssim').to(device)
    m["lpips"] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    return m


class PairedMetrics:
    def __init__(self, device):
        metric_paired_dict = {}

        metric_paired_dict["psnr"] = pyiqa.create_metric('psnr').to(device)
        metric_paired_dict["dists"] = pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ms_ssim"] = pyiqa.create_metric('ms_ssim').to(device)
        metric_paired_dict["lpips"] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

        self.metric_paired_dict = metric_paired_dict

    def score(self, img_m11, gt01):
        metric_paired_dict = self.metric_paired_dict
        recon_tensor, gt_tensor = (img_m11 * 0.5 + 0.5).clamp(0, 1), gt01
        result = {}

        for key, metric in metric_paired_dict.items():
            value = metric(recon_tensor, gt_tensor).item()
            result[key] = result.get(key, 0) + value

        return {k: float(result[k]) for k in PAIRED}


def build(names, device):
    """The metric objects of `names` (a subset of PAIRED), built as build_metrics builds them."""
    if set(names) == set(PAIRED):
        return build_metrics(device)
    make = {"psnr": lambda: pyiqa.create_metric("psnr"), "dists": lambda: pyiqa.create_metric("dists"),
            "ms_ssim": lambda: pyiqa.create_metric("ms_ssim"),
            "lpips": lambda: LearnedPerceptualImagePatchSimilarity(normalize=True)}
    return {k: make[k]().to(device) for k in names}
