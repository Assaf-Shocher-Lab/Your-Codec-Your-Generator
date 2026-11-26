"""AEIC-ME built as AEIC's compress.py builds it (LoRA rank 32 merged, tiled VAE decoder 160, tiled UNet 96 / 32),
except compile_model = False (speed only) and exact float32 math (adapted_codec/determinism.py, after AEIC's import
switched TF32 on). One checkpoint per process: a second AEIC model in the same process changes the shared UNet."""
from types import SimpleNamespace

import torch

from adapted_codec.determinism import set_exact_math
from adapters.aeic.wrapper import PatchPCA, WrappedEncoder


def load_aeic(codec_path, sd_path, vae_decoder_path, codec_cfg):
    from AEIC_practical import AEIC
    me = codec_cfg["type"] == "AEIC-ME"
    args = SimpleNamespace(codec_type=codec_cfg["type"], codec_path=codec_path, vae_decoder_path=vae_decoder_path,
                           lora_rank_unet=codec_cfg["lora_rank_unet"], merge_LoRA=codec_cfg["merge_lora"],
                           compile_model=codec_cfg["compile_model"], use_tiled_vae=codec_cfg["use_tiled_vae"],
                           vae_decoder_tiled_size=codec_cfg["vae_decoder_tiled_size"],
                           use_tiled_unet=codec_cfg["use_tiled_unet"],
                           latent_tiled_size=codec_cfg["latent_tiled_size"] if me else 192,
                           latent_tiled_overlap=codec_cfg["latent_tiled_overlap"] if me else 64)
    net = AEIC(sd_path=sd_path, args=args)
    set_exact_math()

    return net.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def load_wrapped_encoder(net, pca_path, wrapper_cfg):
    pca = torch.load(pca_path, map_location="cpu", weights_only=True)
    k = wrapper_cfg["weak_coefficients"]
    assert k == net.codec.y_channel, (k, net.codec.y_channel)
    device = next(net.codec.parameters()).device
    return WrappedEncoder(net.codec.g_a, PatchPCA(pca, k=k, patch=wrapper_cfg["patch"]),
                          c=wrapper_cfg["gain_c"]).to(device)
