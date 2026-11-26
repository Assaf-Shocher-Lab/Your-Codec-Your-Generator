"""AEIC functions in which the adapted codec acts, copied verbatim from third_party/aeic (AEIC-main, MIT); our lines
are between "adapted codec" fences and every upstream line they replace is kept as "# upstream: <line>".
tests/test_upstream_copies.py restores the upstream text from this file and compares it with third_party/aeic.

  AEICCodecCopy    PixelCodec.inference: quantize a given latent (F1), return the record (F3)
  AEICModelCopy    AEIC.inference: decode received symbols (F4)
  AEICCodecParts   the quantizer lines of PixelCodec.inference on a given latent (hyper latent; four-part rounding)
  adaptive_instance_normalization_side_info   color_fix.adaptive_instance_normalization_quant with the source
                   statistics as side information (F5)
  pad64            compress.py's reflect padding to a multiple of 64
  AEICCoderCopy    PixelCodec.compress / decompress / my_build_indexes: code a given record (F6), return the decoded
                   record (F7), code every element (F8)
  preprocess_image, aeic_transform   compress.py's image loading and input transform"""
import math

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms

from color_fix import calc_mean_std_quant           # third_party/aeic/src/color_fix.py, unchanged
from codec.codec_practical import SCALES_LEVELS     # third_party/aeic/src/codec/codec_practical.py, unchanged


class AdaptedRequest:
    """What the adapted codec asks of the copied AEIC functions (fences F1, F3, F4, F6, F7), and what they hand
    back."""

    def __init__(self, latent=None, pair_only=True, decode_y_hat=None, code_pair=None):
        self.latent = latent                  # quantize this latent instead of g_a(x) (F1)
        self.pair_only = pair_only            # stop after the quantizer and return the bits (F3)
        self.decode_y_hat = decode_y_hat      # decode these received symbols (F4)
        self.code_pair = code_pair            # entropy-code this pair (z_hat, y_hat) with AEIC's rANS coder (F6)
        self.z_hat = self.y_hat = self.y = self.rate = None


class AEICCodecCopy:
    """Namespace for AEIC's PixelCodec.inference (the codec step): AEICCodecCopy.inference(net.codec, x, ...)."""

# upstream copy: src/codec/codec_practical.py L59-124 (sha256 2a1c3e57...)
    def inference(self, x, ori_h=None, ori_w=None):
        
        y_h, y_w = x.shape[2:]
        y_h, y_w = y_h // 32, y_w // 32
        z_h, z_w = math.ceil(y_h / 4), math.ceil(y_w / 4)
        pad_h, pad_w = z_h * 4 - y_h, z_w * 4 - y_w

        # --- adapted codec: begin --- quantize the latent the adapted codec passes (E'(x)) instead of g_a(x)
        request = getattr(self, "adapted_request", None)
        if request is not None and request.latent is not None:
            y = request.latent
        else:
            y = self.g_a(x)
        # upstream: y = self.g_a(x)
        # --- adapted codec: end ---

        y_padded = F.pad(y, pad=(0, pad_w, 0, pad_h), mode='constant')
        z = self.h_a(y_padded)

        _, quantized_z_likelihoods = self.entropy_bottleneck(z, training=False)

        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = torch.round(z - z_offset) + z_offset

        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        base = self.h_s(z_hat)[:, :, 0 : y_h, 0 : y_w]
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        means_0 = means_0_supp * mask_0
        scales_0 = scales_0_supp * mask_0
        y_0 = y * mask_0
        y_hat_0 = torch.round(y_0 - means_0) + means_0

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        means_1 = means_1_supp * mask_1
        scales_1 = scales_1_supp * mask_1
        y_1 = y * mask_1
        y_hat_1 = torch.round(y_1 - means_1) + means_1

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        means_2 = means_2_supp * mask_2
        scales_2 = scales_2_supp * mask_2
        y_2 = y * mask_2
        y_hat_2 = torch.round(y_2 - means_2) + means_2

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        means_3 = means_3_supp * mask_3
        scales_3 = scales_3_supp * mask_3
        y_3 = y * mask_3
        y_hat_3 = torch.round(y_3 - means_3) + means_3

        scales_all = scales_0 + scales_1 + scales_2 + scales_3
        means_all = means_0 + means_1 + means_2 + means_3

        _, quantized_y_likelihoods = self.gaussian_conditional(y, scales_all, means_all, training=False)

        y_hat = base * (1 - mask_3) + y_hat_3
        # --- adapted codec: begin --- hand back the record (z_hat, y_hat) and its latent; for encoding only, stop before the synthesis with the bits of AEIC's own rate()
        if request is not None:
            request.z_hat, request.y_hat, request.y = z_hat, y_hat, y
            if request.pair_only:
                request.rate = self.rate(
                    latent_likelihoods=quantized_y_likelihoods,
                    quantized_latent_likelihoods=quantized_y_likelihoods,
                    hyper_latent_likelihoods=quantized_z_likelihoods,
                    quantized_hyper_latent_likelihoods=quantized_z_likelihoods,
                    ori_h=ori_h,
                    ori_w=ori_w,
                )
                return None, request.rate, None
        # --- adapted codec: end ---
        x_hat, res1 = self.g_s(y_hat)

        RateLossOutput = self.rate(
            latent_likelihoods=quantized_y_likelihoods,
            quantized_latent_likelihoods=quantized_y_likelihoods,
            hyper_latent_likelihoods=quantized_z_likelihoods,
            quantized_hyper_latent_likelihoods=quantized_z_likelihoods,
            ori_h=ori_h,
            ori_w=ori_w,
        )

        return x_hat, RateLossOutput, res1
# end of upstream copy


class AEICModelCopy:
    """Namespace for AEIC's AEIC.inference (codec step + one-step UNet + VAE decoder):
    AEICModelCopy.inference(net, x, ...)."""

# upstream copy: src/AEIC_practical.py L75-81 (sha256 c7a898b4...)
    @torch.inference_mode()
    def inference(self, x, ori_h=None, ori_w=None):
        '''Fast Inference without Bitstreams'''
        # --- adapted codec: begin --- decoding a received record starts from its symbols with AEIC's synthesis of y_hat (their codec_practical.py L113); encoding runs the copied codec step above
        request = getattr(self.codec, "adapted_request", None)
        if request is not None and request.decode_y_hat is not None:
            (l_T, residual), RateLossOutput = self.codec.g_s(request.decode_y_hat), None
        else:
            l_T, RateLossOutput, residual = AEICCodecCopy.inference(self.codec, x, ori_h=ori_h, ori_w=ori_w)
        # upstream: l_T, RateLossOutput, residual = self.codec.inference(x, ori_h=ori_h, ori_w=ori_w)
        # --- adapted codec: end ---
        l_0 = self.inference_unet(l_T) + residual
        x_hat = self.vae.decoder(l_0).clamp(-1, 1)
        return x_hat, RateLossOutput
# end of upstream copy


class AEICCodecParts:
    """AEIC's quantizer lines, copied verbatim from PixelCodec.inference, as the two pieces the search repeats on
    a given latent y: its hyper latent, and the four-part checkerboard rounding of y under a given z_hat
    (self = net.codec)."""

    def hyper_latent(self, y):
        """z = h_a(y padded to a multiple of 4 latent cells) and z_hat = round(z - m) + m, for a given latent y."""
        y_h, y_w = y.shape[2:]                      # ours: the latent size from y (their L61-62 take it from x)
# upstream copy: src/codec/codec_practical.py L63-64
        z_h, z_w = math.ceil(y_h / 4), math.ceil(y_w / 4)
        pad_h, pad_w = z_h * 4 - y_h, z_w * 4 - y_w
# end of upstream copy
# upstream copy: src/codec/codec_practical.py L68-69
        y_padded = F.pad(y, pad=(0, pad_w, 0, pad_h), mode='constant')
        z = self.h_a(y_padded)
# end of upstream copy
# upstream copy: src/codec/codec_practical.py L73-74
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = torch.round(z - z_offset) + z_offset
# end of upstream copy
        return z, z_hat

    def quantize_given_z(self, y, z_hat):
        """y_hat = the 4-part checkerboard rounding of y around the means predicted from z_hat; also the scales and the
        means (for the bits)."""
        y_h, y_w = y.shape[2:]                      # ours: the latent size from y (their L61-62 take it from x)
# upstream copy: src/codec/codec_practical.py L76-108
        B, C, H, W = y.shape
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, device=y.device)

        base = self.h_s(z_hat)[:, :, 0 : y_h, 0 : y_w]
        means_0_supp, scales_0_supp = self.adapter_out[0](self.g_c(self.adapter_in[0](base))).chunk(2, 1)
        means_0 = means_0_supp * mask_0
        scales_0 = scales_0_supp * mask_0
        y_0 = y * mask_0
        y_hat_0 = torch.round(y_0 - means_0) + means_0

        base = base * (1 - mask_0) + y_hat_0
        means_1_supp, scales_1_supp = self.adapter_out[1](self.g_c(self.adapter_in[1](base))).chunk(2, 1)
        means_1 = means_1_supp * mask_1
        scales_1 = scales_1_supp * mask_1
        y_1 = y * mask_1
        y_hat_1 = torch.round(y_1 - means_1) + means_1

        base = base * (1 - mask_1) + y_hat_1
        means_2_supp, scales_2_supp = self.adapter_out[2](self.g_c(self.adapter_in[2](base))).chunk(2, 1)
        means_2 = means_2_supp * mask_2
        scales_2 = scales_2_supp * mask_2
        y_2 = y * mask_2
        y_hat_2 = torch.round(y_2 - means_2) + means_2

        base = base * (1 - mask_2) + y_hat_2
        means_3_supp, scales_3_supp = self.adapter_out[3](self.g_c(self.adapter_in[3](base))).chunk(2, 1)
        means_3 = means_3_supp * mask_3
        scales_3 = scales_3_supp * mask_3
        y_3 = y * mask_3
        y_hat_3 = torch.round(y_3 - means_3) + means_3

        scales_all = scales_0 + scales_1 + scales_2 + scales_3
        means_all = means_0 + means_1 + means_2 + means_3
# end of upstream copy
# upstream copy: src/codec/codec_practical.py L112-112
        y_hat = base * (1 - mask_3) + y_hat_3
# end of upstream copy
        return y_hat, scales_all, means_all


hyper_latent = AEICCodecParts.hyper_latent
quantize_given_z = AEICCodecParts.quantize_given_z


# upstream copy: src/color_fix.py L52-64 (sha256 21b6870f...)
# --- adapted codec: begin --- renamed; the source statistics arrive as side information (colour_side_info)
def adaptive_instance_normalization_side_info(content_feat:Tensor, source_stats, bits: int):
# upstream: def adaptive_instance_normalization_quant(content_feat:Tensor, style_feat:Tensor, bits: int):
# --- adapted codec: end ---
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    # --- adapted codec: begin --- (mean, std) of the source, quantized to `bits` by the encoder with their calc_mean_std_quant
    style_mean, style_std = source_stats
    # upstream: style_mean, style_std = calc_mean_std_quant(style_feat, bits=bits)
    # --- adapted codec: end ---
    content_mean, content_std = calc_mean_std_quant(content_feat, bits=bits)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
# end of upstream copy


# upstream copy: src/compress.py L102-106 (sha256 4b118e31...)
# --- adapted codec: begin --- compress.py's padding step as a function of the image
def pad64(img):
# --- adapted codec: end ---
        ori_h, ori_w = img.shape[2:]

        pad_h = (math.ceil(ori_h / 64)) * 64 - ori_h
        pad_w = (math.ceil(ori_w / 64)) * 64 - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')
        # --- adapted codec: begin --- return the padded image
        return img_padded
        # --- adapted codec: end ---
# end of upstream copy


class AEICCoderCopy:
    """Namespace for AEIC's PixelCodec.compress / decompress with their rANS entropy coder (compiled from
    third_party/aeic/src/cpp by scripts/build_aeic_coder.sh). Call after net.codec.update(force=True)
    (their compress.py L67); adapters/aeic/file_format.py does both."""

# upstream copy: src/codec/codec_practical.py L220-271 (sha256 2a1c3e57...)
    @torch.no_grad()
    def compress(self, x):

        B, C, H, W = x.shape
        y_h, y_w = H // 32, W // 32
        z_h, z_w = math.ceil(y_h / 4), math.ceil(y_w / 4)
        pad_h = z_h * 4 - y_h
        pad_w = z_w * 4 - y_w

        masks = self.get_mask_four_parts(1, self.y_channel, y_h, y_w, device=x.device)

        indexes, z_offset = self.entropy_bottleneck.get_compress_info([1, self.y_channel // 2, z_h, z_w])

        # --- adapted codec: begin --- code the given record (z_hat, y_hat) instead of the record of g_a(x) (the estimate path pads the hyper latent's input with zeros, this function by reflection: the given record keeps the estimate path's z_hat)
        request = getattr(self, "adapted_request", None)
        if request is not None and request.code_pair is not None:
            z_hat_sent, y = request.code_pair
            z_q = torch.round(z_hat_sent - z_offset)
        else:
            y = self.g_a(x)
            y_padded = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
            z = self.h_a(y_padded)
            z_q = torch.round(z - z_offset)
        # upstream: y = self.g_a(x)
        # upstream: y_padded = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")
        # upstream: z = self.h_a(y_padded)
        # upstream: z_q = torch.round(z - z_offset)
        # --- adapted codec: end ---
        z_hat = z_q + z_offset
        z_q = z_q.flatten().cpu().numpy()

        base = self.h_s(z_hat)[:, :, :y_h, :y_w]
        y_q_list = []
        scales_list = []

        for i in range(4):

            mask = masks[i]
            out = self.adapter_out[i](self.g_c(self.adapter_in[i](base)))
            means_supp, scales_supp = out.chunk(2, 1)
            means = means_supp * mask
            scales = scales_supp * mask
            y_part = y * mask
            y_q = torch.round(y_part - means)
            y_q_list.append(y_q)
            scales_list.append(scales)

            if i < 3:
                y_hat = y_q + means
                base = base * (1 - mask) + y_hat

        y_q_0, y_q_1, y_q_2, y_q_3, indexes_0, indexes_1, indexes_2, indexes_3 = self.compress_group(y_q_list, scales_list)

        self.entropy_coder.encode_with_z4y_indexes(
            z_q, indexes, self.z_cdf_group_index,
            y_q_0, indexes_0,
            y_q_1, indexes_1,
            y_q_2, indexes_2,
            y_q_3, indexes_3,
            self.y_cdf_group_index)
        self.entropy_coder.flush()
        
        return
# end of upstream copy

# upstream copy: src/codec/codec_practical.py L274-301 (sha256 2a1c3e57...)
    @torch.no_grad()
    def decompress(self, z_size, padded_y_h, padded_y_w):

        indexes, z_offset = self.entropy_bottleneck.get_compress_info(z_size)
        z_q = self.entropy_coder.decode_stream(indexes.reshape(-1), self.z_cdf_group_index)
        z_hat = z_q.reshape(indexes.shape).to(z_offset.device, dtype=z_offset.dtype) + z_offset

        masks = self.get_mask_four_parts(1, self.y_channel, padded_y_h, padded_y_w, device=z_hat.device)
        base = self.h_s(z_hat)[:, :, :padded_y_h, :padded_y_w]

        y_parts = []
        for i in range(4):

            mask = masks[i]
            out = self.adapter_out[i](self.g_c(self.adapter_in[i](base)))
            means_supp, scales_supp = out.chunk(2, 1)
            means = means_supp * mask
            scales = scales_supp * mask
            y_hat = self.decompress_group_with_mask(scales, means, mask)
            y_parts.append(y_hat)

            if i < 3:
                base = base * (1 - mask) + y_hat

        y_hat = base * (1 - masks[3]) + y_parts[3]
        # --- adapted codec: begin --- hand the decoded record back instead of synthesizing the image (the file check needs only the record)
        request = getattr(self, "adapted_request", None)
        if request is not None:
            request.z_hat, request.y_hat = z_hat, y_hat
            return None, None
        x_hat, res1 = self.g_s(y_hat)
        # upstream: x_hat, res1 = self.g_s(y_hat)
        # --- adapted codec: end ---

        return x_hat, res1
# end of upstream copy

# upstream copy: src/codec/codec_practical.py L212-217 (sha256 2a1c3e57...)
    def my_build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.my_log_scale_min) / self.my_log_scale_step
        indexes = indexes.clamp_(0, SCALES_LEVELS - 1)
        # --- adapted codec: begin --- code every element (AEIC's coder skips elements whose scale is below 0.08 and its decoder sets them to the mean); the file and the estimated bpp then describe the same record
        # upstream: indexes = torch.where(scales < 0.08, torch.zeros_like(indexes) - 1, indexes)
        # --- adapted codec: end ---
        return indexes.int()
# end of upstream copy


# upstream copy: src/compress.py L20-23 (sha256 4b118e31...)
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor
# end of upstream copy


def aeic_transform():
    """AEIC's input transform (compress.py): [0, 1] tensor, then normalized to [-1, 1]."""
# upstream copy: src/compress.py L76-79
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
# end of upstream copy
    return transform
