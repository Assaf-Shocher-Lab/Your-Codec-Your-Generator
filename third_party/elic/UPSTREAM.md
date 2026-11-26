# ELIC re-implementation (external dependency, not included)

Unofficial PyTorch re-implementation of ELIC ("ELIC: Efficient Learned Image Compression with Unevenly Grouped
Space-Channel Contextual Adaptive Coding", CVPR 2022).
- Repository: https://github.com/VincentChandelier/ELiC-ReImplemetation
- Commit: `92a9ece1e6e1a188a12dfd7a58f9b51c554f9f2d`
- Licence: the repository has no licence file, so no file of it is included here. Clone it yourself:

```
git clone https://github.com/VincentChandelier/ELiC-ReImplemetation third_party/elic/src
git -C third_party/elic/src checkout 92a9ece1e6e1a188a12dfd7a58f9b51c554f9f2d
git -C third_party/elic/src apply ../../../patches/elic_decompress.patch
python tests/test_elic_patch.py
```

The patch adds four lines to `TestModel.decompress` (between `# --- adapted codec: begin/end ---`): when the file
reader sets `net.pair_only`, the function returns the decoded record (z_hat, y_hat) instead of synthesizing the image.
With `pair_only` unset the function is unchanged. `tests/test_elic_patch.py` checks the sha256 of every file used and
that the patched function minus the fenced lines is the upstream function.

Files used at run time (imported by `adapters/elic/codec.py`): `Network.py` (TestModel), `ELICUtilis/layers/*`.
`Network.py` also imports `thop` and `ptflops` (FLOP counters, unused at inference); the adapter stubs them.

| file | sha256 at the commit (LF line endings) | after the patch |
|---|---|---|
| `Network.py` | `9bcd23a1f7f668804c38a343923d2dbd49dd54573d5c2875b5055a59313c721a` | `8931f6e5b7c457e34aa9a23bdf88973df623f582df2cbdca3ef78f551216a95c` |
| `ELICUtilis/layers/__init__.py` | `d55722c3524f2b55b3e9deacd4388e26a90ba585adf3ba593d3a9df0563cc379` | |
| `ELICUtilis/layers/gdn.py` | `878679e1f82ae3726f3ec41b93bd80557abcca82ff6ce178ea55b217aa1c74a6` | |
| `ELICUtilis/layers/layers.py` | `2885c495292cf96265dea9754c64bfd1f1c72be667ab88847f7fdc92bcb27abf` | |

Weights (not included): the released checkpoints of the repository (lambda 0.004, 0.008, 0.016, 0.032, 0.15, 0.45),
placed as `weights/elic/ELIC_lambda<lambda>.pth.tar`. The 256 x 256 and FFHQ experiments use the ELIC weights of
Xu et al. (README, "Weights and data").
