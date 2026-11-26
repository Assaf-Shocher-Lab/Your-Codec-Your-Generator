# Your Codec, Your Generator: Idempotent Generative Compression without Retraining

Raz Haim and Assaf Shocher, Technion - Israel Institute of Technology

[Paper (arXiv)](https://arxiv.org/abs/2611.31520) | [Project page](https://assafshocher.github.io/your-codec/)

The method makes a frozen, pretrained learned image codec idempotent under recompression, without any training, and
lets a frozen generative prior decode the same bitstream (NLBP, nonlinear back-projection). This repository holds the
code of the reported results: a codec-agnostic core, thin per-codec adapters, NLBP, the evaluation, and one
configuration per reported variant.

## The method (plug-and-play)

A codec is used only through a small interface: its analysis transform $`E`$, synthesis $`D`$, hyper-analysis
$`h_a`$, its own quantizers and entropy coder, and its file format. Around $`E`$ we put a fixed linear wrapper built from
a patch-PCA basis (the only fitted component; the codec is not trained):

```math
E'(x) = f(b(x)) + c\,a(x).
```

The wrapper has an exact write $`W(v, b)`$ with $`E'(W(v, b)) = v`$, hence a right inverse $`D'(v) = W(v, b(D(v)))`$
with $`E'(D'(E'(x))) = E'(x)`$. The encoder sends the codec's own record in round 1 and the record of $`E'(x)`$ in later
rounds. The decoder never sees the source image: from the received symbols it selects a target latent $`y^\star`$ inside
the cells of the record (target search with margins), writes the image with $`W`$, and, depending on the storage
protocol, repairs the image so that re-encoding reproduces the same symbols (clipped-float range repair; 8-bit target
correction, lattice search, per-symbol hyper repair, escalation ladder; HPCM outer stabilization). NLBP replaces the
image $`W(y^\star, b(D(\hat y)))`$ by a sample of a frozen prior, corrected at every step so that the output still
writes $`y^\star`$. Everything after the codec interface is codec-agnostic: adding a codec = writing its adapter.

## Notation and code names

| Paper | Code | Where |
|---|---|---|
| $`E`$, $`D`$ | `base_encode`, `base_decode` | `adapted_codec/codec_ops.py`, `adapters/<codec>/` |
| $`h_a`$, $`Q_z`$, $`Q_y`$ | `hyper`, `quantize_main` | codec interface |
| $`s = (s_y, s_z)`$, $`\Sigma`$ | `symbol_record` (z_hat, y_hat and bits) | `adapted_codec/encoder.py` |
| $`\mu_{\mathrm{PCA}}`$, $`U_0`$, $`U_1`$ | `PCAWrapper.mu`, `U` (rows split at k) | `adapted_codec/wrapper.py`, `adapted_codec/pca_basis.py` |
| $`a(x)`$, $`b(x)`$, $`J(a, b)`$ | `split(x)` -> (a, b), `join(a, b)` | `adapted_codec/wrapper.py` (AEIC: `adapters/aeic/wrapper.py`) |
| $`f(b) = E(J(0, b))`$, gain $`c`$ | `E(join(0, b))`, `c` | wrapper |
| $`E'(x) = f(b(x)) + c\,a(x)`$ | `wrapped_encoder`, `wrapped_encode` | wrapper, codec interface |
| $`W(v, b)`$, $`D'(v)`$ | `write(x_raw, v)` = $`W(v, b(x_{\mathrm{raw}}))`$ | `adapted_codec/codec_ops.py`, `adapters/aeic/wrapper.py` |
| $`F_m(s)`$, $`y^\star`$, margins $`m_y`$, $`m_z`$ | `select_target`, `search_target`, `margin_y`, `margin_z` | `adapted_codec/target_search.py`, `adapters/aeic/codec.py` |
| refinement toward the projected start | `refine_target` | `adapted_codec/refinement.py`, `adapters/aeic/codec.py` |
| acceptance $`\Sigma(E'(x)) = s`$ | `is_feasible` | `adapted_codec/acceptance.py` |
| $`S_8`$ | `store01`, `store11` | `adapted_codec/protocols/storage.py` |
| clipped-float range repair | `range_repair`, `decode_clipped` | `adapted_codec/protocols/clipped_float/range_repair.py` |
| 8-bit closed loop, target update ($`\gamma_y`$, $`\gamma_z`$) | `corrected_write` | `adapted_codec/protocols/eight_bit/target_correction.py`, `adapters/aeic/codec.py` |
| lattice merit ($`\eta_y`$, $`\eta_z`$), 9,056 trials | `lattice_search` | `.../eight_bit/lattice_repair.py` |
| per-symbol hyper repair | `repair_hyper_symbols` | `.../eight_bit/hyper_symbol_repair.py` |
| escalation ladder, box step | `escalation_decode`, `escalation_write`, `box_step_decode`; AEIC: `EscalatedAeic` | `.../eight_bit/escalation.py`, `box_step.py`, `adapters/aeic/chain.py` |
| HPCM outer stabilization | `decode` (stabilization_states = 4) | `.../eight_bit/decoder.py` |
| NLBP update ($`v_t^+`$, $`b_t^+`$), $`\lambda_t`$, $`\kappa`$, $`v_{\mathrm{ref}}`$ | `nlbp_update`, `lambda2` -> `lambda1`, `kz`, `target` | `nlbp/update.py` |
| NLBP sampler, start noise $`\omega`$, steps | `nlbp_decode`, `t0` ($`\omega`$ = snapped time of `t0`), `stride` | `nlbp/sampler.py`, `nlbp/schedule.py` |
| final $`b`$ refinement ($`K`$ Adam steps), final write | `TargetWrite.opt`, `opt_K` | `nlbp/final_write.py` |

Every setting is a field of `adapted_codec/settings.py` (AEIC: also `AeicSettings` in `adapters/aeic/codec.py`; NLBP:
the configuration keys of `nlbp/sampler.py`) and is set per reported variant in `configs/*.yaml` and
`nlbp/configs/*.yaml`; no setting is chosen per image or per chain.

## Layout

```
adapted_codec/     codec-agnostic core: wrapper, PCA basis, target search, refinement, acceptance, encoder,
                   protocols (float, clipped float, 8-bit), file chain (chain.py), in-memory chain (memory_chain.py)
adapters/<codec>/  codec interface + file format per codec: aeic, hpcm, elic, hyperprior, mean_scale, tcm, msillm, glc
                   (AEIC keeps the arithmetic of its own runs for the decoder, wrapper and chains: adapters/aeic/)
nlbp/              NLBP: update equations, sampler (pixel and latent priors), final write, receiver state, codec
                   interface, time grid, prior adapters (latent priors: client + child-process server), configurations
third_party/       upstream code, vendored unchanged with its LICENSE and UPSTREAM.md: aeic, hpcm, tcm, glc,
                   neuralcompression; external dependencies (not redistributed, see UPSTREAM.md): elic, xu, nlbp_nets
patches/           the one injected change of the ELIC re-implementation (fenced)
configs/           one yaml per reported variant
evaluation/        PSNR, MS-SSIM, LPIPS, DISTS, patch FID / KID, payload bpp (score_chains.py), 8-bit PNG of NLBP
scripts/           run_chain.py, run_nlbp.py, prepare_256.py, prepare_ffhq256.py, build_aeic_coder.sh, compare_smoke.py
tests/             CPU self-test, upstream-copy and patch tests, smoke expectations
pca/               the patch-PCA bases: 16 x 16 Vimeo90K (all codecs but AEIC), 32 x 32 CLIC 2020 (AEIC)
```

Where the method acts inside upstream code, the upstream function is copied verbatim into the adapter
(`adapters/<codec>/upstream_copies.py`) and our lines are between `# --- adapted codec: begin/end ---` fences;
`tests/test_upstream_copies.py` restores the upstream text and compares it with `third_party/`.

## Install

Python 3.10+ and PyTorch 2.x with CUDA and the matching torchvision (the reported numbers were produced with PyTorch
2.7 and 2.9 on NVIDIA RTX PRO 6000 and B200 GPUs). Install PyTorch first, then

```
pip install -r requirements.txt
```

`requirements.txt` pins the versions of the reported runs, which keep the installed PyTorch (einops / timm: TCM;
torchmetrics / pyiqa / torch-fidelity: evaluation). AEIC also needs
`pip install diffusers==0.25.1 peft==0.13.2 pytorch_lightning==2.4.0` (the versions of the reported runs) and its rANS
coder, compiled once from `third_party/aeic/src/cpp` with g++:

```
PYBIN=python bash scripts/build_aeic_coder.sh
```

NLBP with the PixNerd and latent priors also needs a second Python environment for the text encoder and the latent
priors (the reported runs: `diffusers==0.42.1 transformers==5.22.1 accelerate safetensors` over the same PyTorch); set
`NLBP_PRIOR_PY` to its python and `NLBP_WORK` to a writable folder (prompt-embedding caches). The prior network files
are external (`third_party/nlbp_nets/UPSTREAM.md`).

ELIC and Xu et al.'s code are external dependencies: `third_party/elic/UPSTREAM.md`, `third_party/xu/UPSTREAM.md`.
Run everything from the repository root. The runners set exact float math (no TF32, deterministic cuDNN); bit-exact
reproduction of the reported files needs the GPU type of the original runs.

## Weights and data (not included)

| Codec / prior | Weights | Place as |
|---|---|---|
| AEIC-ME | official checkpoints of the five rates and AEIC's half VAE decoder; Stability AI's sd-turbo (diffusers folder) | `weights/aeic/AEIC_ME_ft{2,4,8,16,32}.pkl`, `weights/aeic/halfDecoder.ckpt`, `weights/sd-turbo/` |
| HPCM-Large (MSE) | official checkpoints | `weights/hpcm/HPCM_Large_mse_<lambda>.pth` |
| ELIC | re-implementation's checkpoints; Xu et al.'s ELIC weights for lambda 0.15 / 0.45 and the FFHQ rates 1-5 | `configs/elic_*.yaml`, `nlbp/configs/elic_ffhq_codec.yaml` (`weights/elic_xu/ELIC_0xxx_ft_3980_Plateau.pth.tar`) |
| Hyperprior, Mean-scale | CompressAI zoo (downloaded by CompressAI) | `TORCH_HOME/hub/checkpoints` |
| TCM | official N = 64 MSE checkpoints | `weights/tcm/TCM_N64_mse_<lambda>.pth.tar` |
| MS-ILLM | NeuralCompression MS-ILLM release | `weights/msillm/msillm_target_<rate>bpp.ckpt` |
| GLC | official image checkpoint | `weights/glc/GLC_image.pth.tar` |
| NLBP prior ffhq_10m | the DPS authors' FFHQ 256 DDPM (`ffhq_10m.pt`, md5 `d2d4694c...`), with the UNet code of Xu et al.'s repository (`third_party/xu/UPSTREAM.md`) | `weights/priors/ffhq_10m.pt`, `third_party/xu/src` |
| NLBP PixNerd-XXL/16 T2I | authors' checkpoint (`third_party/nlbp_nets/UPSTREAM.md`) and Qwen3-1.7B for the empty-prompt embedding (computed in the prior environment, `NLBP_PRIOR_PY`) | `weights/priors/pixnerd/PixNerd-XXL-P16-T2I_model.ckpt`, `weights/priors/qwen3_1p7b/` (`nlbp/configs/hpcm_kodak_codec.yaml`) |
| NLBP DeCo-512 | authors' checkpoint (`third_party/nlbp_nets/UPSTREAM.md`) | `NLBP_MODELS/deco/imagenet512_epoch340.ckpt` |
| NLBP Z-Image, FLUX.2 [dev] | Z-Image (Tongyi-MAI/Z-Image, revision `04cc4abb`, diffusers folder); FLUX.2 [dev] (black-forest-labs/FLUX.2-dev, revision `26afe3a7`, diffusers folder) is gated: accept its licence on its Hugging Face model page and download it yourself | `weights/priors/zimage/`, `weights/priors/flux2_dev/`; run by the child-process prior server `nlbp/priors/latent_server.py` in the prior environment |

Data: Kodak (24 PNGs) in `data/kodak`; DIV2K validation (100 PNGs, `0801.png` ... `0900.png`) in `data/div2k`; CLIC 2020
Professional validation (41 PNGs) in `data/clic`; 256 x 256 sets with `scripts/prepare_256.py` (Kodak, CLIC); FFHQ
256 x 256 for NLBP: rows 0-999 of the first shard of the Hugging Face dataset `bitmind/ffhq-256`
(`train-00000-of-00016.parquet`, md5 `e7f8b94f61c3c787d4a9fde2bf7ad265`) with
`python scripts/prepare_ffhq256.py --parquet <shard> --out data/ffhq256 --n 1000` (needs pyarrow with NumPy >= 2).

## Reproduce

One chain per (variant, rate, image); `--rounds 50` = 1 compression + 49 recompressions. AEIC: one rate per process.

```
python scripts/run_chain.py --config configs/<variant>.yaml --rate <rate> --images kodim01,kodim02 --rounds 50 \
    --data data/kodak --out runs/<variant>
python evaluation/score_chains.py --chains runs/<variant>/<method> --orig-dir data/kodak --rounds 1,50 \
    --codec <codec> --protocol kodak64 --out runs/score_<variant>
```

| Reported results | Variant (protocol) | Config | Rates |
|---|---|---|---|
| recompression, 8-bit | AEIC Kodak (8-bit image in memory) | `aeic_kodak_8bit` | ft2 ... ft32 |
| | HPCM Kodak (the method, with the escalation ladder; the reported run used the outer stabilization) | `hpcm_kodak_8bit_ladder`, `hpcm_kodak_8bit_stabilized` | 0.0018 ... 0.0483 |
| | ELIC Kodak (preset; ladder chains at lambda 0.45) | `elic_kodak_8bit`, `elic_kodak_8bit_ladder` | 0.004 ... 0.45 |
| | MS-ILLM Kodak | `msillm_kodak_8bit` | 0.035 ... 0.45 |
| | Hyperprior, Mean-scale Kodak q1-q7 | `hyperprior_kodak_8bit`, `mean_scale_kodak_8bit` | 1 ... 7 |
| | TCM Kodak (lambda 0.0025-0.025; 0.05 with the box step) | `tcm_kodak_8bit`, `tcm_kodak_8bit_box` | 0.0025 ... 0.05 |
| | AEIC DIV2K 2K (canvas file chain, escalation) | `aeic_div2k_8bit` (`--data data/div2k`) | ft2 ... ft32 |
| | AEIC CLIC 2K (canvas file chain) | `aeic_clic_8bit` (`--data data/clic`) | ft2 ... ft32 |
| | ELIC 256 x 256 Kodak / CLIC (Xu et al.'s weights) | `elic_256_8bit` | 0.45 |
| recompression, float | AEIC Kodak, HPCM Kodak, ELIC Kodak | `aeic_kodak_float`, `hpcm_kodak_float`, `elic_kodak_float` | |
| | GLC Kodak (symbols, estimated rate) | `glc_kodak_float` | 0 ... 3 |
| | AEIC DIV2K (padded canvas, colour side information frozen after round 1) | `aeic_div2k_float` (`--data data/div2k`) | ft2 ... ft32 |
| recompression, clipped float | AEIC Kodak, HPCM Kodak (lambda <= 0.013) | `aeic_kodak_clipped_float`, `hpcm_kodak_clipped_float` | |
| NLBP (runner: `scripts/run_nlbp.py`) | ELIC FFHQ first decode, 8-bit, rates 1-5 | `nlbp/configs/elic_ffhq_rate<r>.yaml` | xu1 ... xu5 |
| | HPCM Kodak lambda .0018 first decode, float: PixNerd-XXL/16 T2I (start noise .45; also .20 / .30 / .35 / .65), Z-Image, FLUX.2 [dev] | `nlbp/configs/hpcm_kodak_{pixnerd_s045, zimage_s045, flux2dev_s045, pixnerd_s0xx}.yaml` | 0.0018 |
| NLBP (configurations and prior adapters only; see NLBP below) | AEIC Kodak ft2 controls: Z-Image 1 step, DeCo-512 1 step and 4 steps | `nlbp/configs/aeic_kodak_ft2_*.yaml` | ft2 |

The rates of each variant are listed in its config (`rates:`). The plain codec of a variant: `--method plain`
(in-memory and AEIC chains) or the 8-bit adapter's plain method (`hpcm_plain`, `elic_plain`, `hp_plain`, `mbt_plain`,
`tcm_plain`, `msillm_plain`). Metrics: PSNR, MS-SSIM, LPIPS, DISTS per image; patch FID / KID per set (Kodak: 2,304
patches of 64 x 64, `--protocol kodak64`; high resolution: 256 x 256 patches, `--protocol 2k256`). Rates are payload
bpp: all entropy-coded streams and transmitted side information, without container framing.

HPCM, 8-bit: the reported run (`hpcm_kodak_8bit_stabilized`) kept the first file in 137 of 144 chains; in the other
7 the outer stabilization accepted a different record, so the file changed once, at round 2, and the image stayed
fixed. With the escalation ladder (`hpcm_kodak_8bit_ladder`, the method), these 7 chains keep their first file as
well, so all 144 chains (6 rates x 24 images) are file-exact and image-exact over 50 rounds. The other 137 chains
never reach the ladder and decode identically under both configurations.

## NLBP

`nlbp/` holds the decoder-side generative method. From the received symbols the receiver selects $`y^\star`$ and forms
$`x_{\mathrm{base}} = W(y^\star, b(D(\hat y)))`$; sampling starts from
$`x = (1 - \omega)\,x_{\mathrm{base}} + \omega\,\epsilon`$. At every step the frozen prior's clean estimate $`u_t`$
is corrected, with $`y_t = E'(u_t)`$:

```math
\begin{aligned}
v_t^+ &= (1 - \lambda_t)\,y_t + \lambda_t\,y^\star,
\\ b_t^+ &= b(u_t) + \kappa\lambda_t\,\big[r_{\mathrm{ref}} - b(D_{\mathrm{ref}}(y_t))\big],
\\ u_t^+ &= W(v_t^+, b_t^+),
\end{aligned}
```

where $`D_{\mathrm{ref}}`$ is the codec's own synthesis and $`r_{\mathrm{ref}} = b(D_{\mathrm{ref}}(v_{\mathrm{ref}}))`$
for a fixed $`v_{\mathrm{ref}} \in \{\hat y, y^\star\}`$. After the last step, $`K`$ Adam steps refine $`b`$ and the
output is the exact write $`W(y^\star, b)`$.

| File | Contents |
|---|---|
| `update.py` | the update above |
| `sampler.py` | schedule, start, the loop on the prior's clean estimate, the final write; `nlbp_decode_latent` for latent-space priors (decodes the clean latent, then the image-space update) |
| `final_write.py` | the $`K`$ Adam steps on $`b`$ and the exact write $`W(y^\star, b)`$ |
| `receiver.py` | $`x_{\mathrm{base}}`$, $`r_{\mathrm{ref}}`$, the seed derived from the received symbols |
| `codec_interface.py` | $`D_{\mathrm{ref}}`$, $`b`$, $`W`$, $`E'`$ on images in $`[-1, 1]`$ |
| `schedule.py` | time grid |
| `priors/` | prior adapters, $`(x_s, s) \mapsto v`$ |
| `configs/` | the reported configurations, equal to the stored ones of the reported runs |

ELIC on FFHQ (first decode, 8-bit; weights and data above, Xu et al.'s repository in `third_party/xu/src`):

```
python scripts/run_nlbp.py --config nlbp/configs/elic_ffhq_rate1.yaml --images 00000,00001 --data data/ffhq256 --out runs/nlbp
python scripts/run_nlbp.py --config nlbp/configs/elic_ffhq_rate2.yaml --images 00000,00001 --data data/ffhq256 --out runs/nlbp
python scripts/run_nlbp.py --config nlbp/configs/elic_ffhq_rate3.yaml --images 00000,00001 --data data/ffhq256 --out runs/nlbp
python scripts/run_nlbp.py --config nlbp/configs/elic_ffhq_rate4.yaml --images 00000,00001 --data data/ffhq256 --out runs/nlbp
python scripts/run_nlbp.py --config nlbp/configs/elic_ffhq_rate5.yaml --images 00000,00001 --data data/ffhq256 --out runs/nlbp
python scripts/compare_smoke.py tests/smoke_expectations/nlbp_elic_ffhq.csv runs/nlbp
```

Per image the sender writes the codec's own file (`nlbp_g1.bin`); the receiver reads only that file, selects $`y^\star`$,
runs NLBP, applies the 8-bit repair toward $`y^\star_8`$ and stores `nlbp_g1.png`; `rows.csv` holds the file and PNG md5, the
symbol check of the file, `fmt_pass` (re-encoding the stored PNG gives the received record) and the PSNR. The test set
is 00000 ... 00999.

HPCM on Kodak (first decode, float; `export NLBP_PRIOR_PY=<python of the prior environment> NLBP_WORK=work`):

```
python scripts/run_nlbp.py --config nlbp/configs/hpcm_kodak_pixnerd_s045.yaml --images kodim01,kodim02 --data data/kodak --out runs/nlbp
python scripts/run_nlbp.py --config nlbp/configs/hpcm_kodak_zimage_s045.yaml --images kodim01,kodim02 --data data/kodak --out runs/nlbp
python scripts/run_nlbp.py --config nlbp/configs/hpcm_kodak_flux2dev_s045.yaml --images kodim01,kodim02 --data data/kodak --out runs/nlbp
python scripts/compare_smoke.py tests/smoke_expectations/nlbp_hpcm_kodak.csv runs/nlbp
```

(`hpcm_kodak_pixnerd_s0{20,30,35,65}.yaml`: the other start-noise points.) The sender writes HPCM's own file
(`nlbp_g1.bin`); the receiver reads only that file, selects $`y^\star`$ under the float margins, runs NLBP with the prior
(PixNerd in pixel space; Z-Image and FLUX.2 [dev] in latent space, served by `nlbp/priors/latent_server.py` in the
prior environment) and stores the float output as the 8-bit PNG `nlbp_g1.png` (`evaluation/png_io.py`). `rows.csv`
holds the digest of the sent streams, `full_sym_eq` / `full_bytes_eq` (re-encoding the float output gives the received
record and the same bytes), the PSNR of the PNG (`psnr`) and of the float output (`psnr_f`).

AEIC Kodak ft2 controls: the configurations (`nlbp/configs/aeic_kodak_ft2_*.yaml`) and their prior adapters
(`priors/deco512.py`, the latent priors above) are included. Their runs used an earlier AEIC codec interface (another
implementation of the sender, the target search and the wrapper than `adapters/aeic/`); no runner for them is
included, since the equivalence of the two AEIC implementations has not been shown.

## Tests

`python tests/cpu_selftest.py` (stand-in codec and prior, no downloads), `python tests/test_upstream_copies.py`,
`python tests/test_elic_patch.py` (after the ELIC setup), and
`python scripts/compare_smoke.py tests/smoke_expectations/<file>.csv runs/<smoke>`, which compares a run with stored
reference rounds of the reported runs (file md5, image md5, PSNR).

## Licence

Our code: MIT (`LICENSE`). Code under `third_party/` keeps its own licence (`third_party/<name>/LICENSE`,
`UPSTREAM.md`); external dependencies are not redistributed.

## Citation

```bibtex
@article{haim2026yourcodec,
  title   = {Your Codec, Your Generator: Idempotent Generative Compression without Retraining},
  author  = {Haim, Raz and Shocher, Assaf},
  journal = {arXiv preprint arXiv:2611.31520},
  year    = {2026}
}
```
