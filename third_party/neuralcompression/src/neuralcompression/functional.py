# Stand-in for neuralcompression.functional.pad_image_to_factor (reflect pad to a multiple of factor).
import torch.nn.functional as F


def pad_image_to_factor(image, factor):
    height, width = image.shape[-2:]
    pad_height = (factor - (height % factor)) % factor
    pad_width = (factor - (width % factor)) % factor
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
    return image, (height, width)
