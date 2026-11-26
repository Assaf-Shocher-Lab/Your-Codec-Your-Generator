# TCM (upstream, unchanged)

"Learned Image Compression with Mixed Transformer-CNN Architectures", CVPR 2023.
- Repository: https://github.com/jmliu206/LIC_TCM
- Commit: `6c6d7a364817a4c372bd210dbbcb175f9172caa0`
- Licence: MIT (`LICENSE`, copied from the repository)
- Only the model package is included (`src/models`), byte-identical to that commit. The adapted codec does not change
  any upstream function: `adapters/tcm/adapter.py` reads the model's layers to form the record (TcmOps).
- Python dependencies of `tcm.py`: compressai, einops, timm.
- Weights (not included): the six N = 64 MSE checkpoints of the repository's README, placed as
  `weights/tcm/TCM_N64_mse_<lambda>.pth.tar` (lambda 0.0025, 0.0035, 0.0067, 0.013, 0.025, 0.05).

| file | sha256 | git blob |
|---|---|---|
| `LICENSE` | `c7631abd8900f6f3afe2d4e322ea35ae868373cc51b545b17c3ac91abc06de55` | `49711f8ce3be02b519e337ea5c85e087475a04aa` |
| `src/models/__init__.py` | `046936d83e70438041753b543157c54701ee9e43cd8e24b641d4e489b1677f0f` | `8423a4d1ad2b84e21f248e61b9bf10b2ded33016` |
| `src/models/tcm.py` | `766e1d08299cdb0aac20caeed77a2527c282bf89acb677ec72ab6fffe11a61ee` | `90293160301477fe7172dfc1e18a35b6ac2f9ed2` |
