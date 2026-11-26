# HPCM (upstream, unchanged)

"Learned Image Compression with Hierarchical Progressive Context Modeling", ICCV 2025.
- Repository: https://github.com/lyq133/LIC-HPCM
- Commit: `c271cfe2412071e7cacf4186825f1bbff4434664`
- Licence: MIT (`LICENSE`)
- Only the files imported by the model are included, byte-identical to that commit. Functions in which the adapted
  codec acts are copied into `adapters/hpcm/upstream_copies.py` (fenced); `tests/test_upstream_copies.py` checks them.
- Weights (not included): HPCM-Large MSE checkpoints from the repository's README, placed as
  `weights/hpcm/HPCM_Large_mse_<lambda>.pth`.

| file | sha256 |
|---|---|
| `LICENSE` | `2c514166581afe6ce33e991a7ae992ce86254f05bf9c0830ca27aadded254f1a` |
| `src/entropy_models/__init__.py` | `a7aebf988e4e715ecbc2b1e29b880ea32e383adaccaea7093de8afc7638901cf` |
| `src/entropy_models/bound_ops.py` | `16290f0516d496569ceacf8fdf15778ee4f26f91fd3a9adb184921a5f295a59c` |
| `src/entropy_models/entropy_models.py` | `237dfb7bf92c69170fcacae07ca75d1409757d3d769f2242cf2ec9262fbee231` |
| `src/layers/__init__.py` | `b2e66adc26c1372ec59158553a898712255736543c69a948c765fdb19b967385` |
| `src/layers/conv.py` | `36d13e0cff17d3efdee3bf1bfe8d7178ff62866a9272859b7720907babf01442` |
| `src/layers/res_blk.py` | `2b010f38e79a319eb3d1fcb1971ee0eefa2726f178f0b8f0d849e90af698e2a3` |
| `src/models/HPCM_Large.py` | `6b62db5d4873649121d8fad3809c43a24efe5b8035c4ee9380ebb7e3b3dae33c` |
| `src/models/base.py` | `c619a834ac1dafbbd238837a22427e64ba5b4eb9f5c96f1841e7e22b4affefd6` |
| `src/utils/utils.py` | `025206ebd0c58ed56f9ce94c1ea6d8c214964b572dcd199a7e5832f7db2a9f35` |
