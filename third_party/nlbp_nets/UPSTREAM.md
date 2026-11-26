# Network code of the PixNerd and DeCo priors (external dependency, not redistributed)

The NLBP prior adapters `nlbp/priors/pixnerd_t2i.py` (HPCM on Kodak) and `nlbp/priors/deco512.py` (AEIC Kodak controls)
import the authors' network classes from `third_party/nlbp_nets/src/` (ignored by git). Their licences were not checked
for this release, so no file is included; place the files as the reported runs used them:

| file in `src/` | origin | change made for the runs |
|---|---|---|
| `pixnerd_t2i.py` | PixNerd (https://github.com/MCG-NJU/PixNerd), commit `6da060bd`, `src/models/transformer/pixnerd_t2i.py` (class `PixNerDiT`) | the 6 `from src.models.layers...` import lines replaced by imports of the same layer definitions from `dit_t2i_DeCo.py` (they are character-identical to PixNerd's `src/models/layers/{attention_op,rope,time_embed,patch_embed,swiglu,rmsnorm}.py`); md5 of the run copy `91ff84fe9ce37115ac569900bc87eccd` |
| `dit_t2i_DeCo.py` | DeCo (https://github.com/Zehong-Ma/DeCo), commit `0792af05`, `src/models/transformer/dit_t2i_DeCo.py` | the 6 layer import lines replaced by verbatim copies of those layer definitions; md5 of the run copy `60a89ecd7e08d41a94eacb81cf1a372c` |
| `dit_c2i_DeCo.py` | DeCo, commit `0792af05`, `src/models/transformer/dit_c2i_DeCo.py` | none (unchanged; md5 of the run copy `240819dbfa685e6c05b350a3d95d94e7`) |

Weights (not included): PixNerd-XXL-P16-T2I `model.ckpt` (https://huggingface.co/MCG-NJU/PixNerd-XXL-P16-T2I, sha256
`7d14aecb54f2e6bfc17683ce453fd517fe7a5948d75fe2e66e3b854abb17169e`) with the Qwen3-1.7B text encoder
(https://huggingface.co/Qwen/Qwen3-1.7B; shard sha256 `169ad53e...`, `912becff...`) for the empty-prompt embedding;
DeCo-XL/16 ImageNet-512 `imagenet512_epoch340.ckpt` under `NLBP_MODELS/deco/`.

Latent priors (weights not included; diffusers folders, loaded by `nlbp/priors/latent_server.py` in the prior
environment): Z-Image (https://huggingface.co/Tongyi-MAI/Z-Image, revision `04cc4abb`) as `weights/priors/zimage/`;
FLUX.2 [dev] (https://huggingface.co/black-forest-labs/FLUX.2-dev, revision `26afe3a7`) as `weights/priors/flux2_dev/`. FLUX.2 [dev] is
gated: accept its licence on the model page, then download it with your own Hugging Face account (for example
`hf download black-forest-labs/FLUX.2-dev --local-dir weights/priors/flux2_dev`).
