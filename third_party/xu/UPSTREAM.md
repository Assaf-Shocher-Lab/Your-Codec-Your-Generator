# Xu et al. (external dependency, not redistributed)

T. Xu et al., "Idempotence and Perceptual Image Compression", ICLR 2024. Their public code repository has no licence
file, so no file of it is included here.

- Repository: https://github.com/tongdaxu/Idempotence-and-Perceptual-Image-Compression (the origin of the clone the
  reported runs used), commit `028cae51e27c6cba5e1a17a308614ac77003e304`.
- Used for: the ELIC weights of the 256 x 256 and FFHQ experiments (`configs/elic_256_8bit.yaml`,
  `nlbp/configs/elic_ffhq_codec.yaml`: rate 1-5 files `ELIC_0008 / 0016 / 0032 / 0150 / 0450_ft_3980_Plateau.pth.tar`,
  md5 `7f1f4d95`, `63f8a6b4`, `be554bc1`, `e5b1a77c`, `35f11222`; rates 1-3 are byte-identical to the
  re-implementation's 0.008 / 0.016 / 0.032 files), and the FFHQ prior
  network: `guided_diffusion/{__init__,unet,fp16_util,nn}.py` and `configs/ffhq_model_config.yaml`, imported read-only
  by `nlbp/priors/ffhq_dps.py` (`load_ffhq_dps_prior(repo, ckpt, dev)`).
- FFHQ prior weights: `ffhq_10m.pt` (FFHQ 256, from the DPS authors; Xu et al. use the same file; md5
  `d2d4694c11bcdfb6bf7f2dcd673ea73d`). Put it under `weights/priors/ffhq_10m.pt`.

Setup:
```
git clone https://github.com/tongdaxu/Idempotence-and-Perceptual-Image-Compression third_party/xu/src && git -C third_party/xu/src checkout 028cae51
```
`third_party/xu/src` is ignored by git.
