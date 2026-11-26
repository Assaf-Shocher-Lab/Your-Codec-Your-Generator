"""The 8-bit PNG convention of the HPCM and AEIC NLBP runs (AEIC's compress.py): an image x in [-1, 1] is saved as
(x * 0.5 + 0.5).clamp(0, 1) * 255 truncated to uint8 (torchvision's ToPILImage); an 8-bit image is loaded as
(u8 / 255 - 0.5) / 0.5. The NLBP rows score this PNG."""
from PIL import Image


def to_uint8(x):
    return (x * 0.5 + 0.5).clamp(0, 1).mul(255).byte()


def from_uint8(u8):
    return (u8.float().div(255) - 0.5) / 0.5


def save_png(x_m11, path):
    u8 = to_uint8(x_m11)
    Image.fromarray(u8[0].permute(1, 2, 0).cpu().numpy()).save(path)
    return u8
