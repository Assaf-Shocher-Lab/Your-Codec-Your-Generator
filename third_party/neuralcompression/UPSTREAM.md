# NeuralCompression (upstream files, unchanged)

Meta's NeuralCompression, https://github.com/facebookresearch/NeuralCompression, MIT licence (file headers; `LICENSE`). `LICENSE` is the standard MIT text with the copyright line of
the file headers: no verbatim copy of the upstream LICENSE file was available when these files were packaged.
Two parts are used, both without installing the package:

1. `_update_patch_fid.py`: function `update_patch_fid` (`neuralcompression/metrics/_update_patch_fid.py`), the copy
   shipped with GLC (`src/utils/_update_patch_fid.py`), byte-identical to the file the reported patch FID / KID used.
2. `src/neuralcompression/`: the HiFiC autoencoder that MS-ILLM's released weights load into
   (`neuralcompression.models.HiFiCAutoencoder`). Upstream files (branch `main`): `models/_hific_autoencoder.py`,
   `models/_hific_encoder_decoder.py`, `models/_hyperprior_autoencoder.py`, and `layers/_channel_norm.py` (here
   `layers.py`). Stand-ins (not upstream; code as run, header comments only): `__init__.py` (the two output containers
   the model files import), `functional.py` (`pad_image_to_factor`), `models/__init__.py` (one import line).
   Python dependency: compressai. Weights (not included): `msillm_target_<rate>bpp.ckpt` from the NeuralCompression
   MS-ILLM release (CC-BY-NC 4.0), placed as `weights/msillm/msillm_target_<rate>bpp.ckpt`.

| file | sha256 |
|---|---|
| `_update_patch_fid.py` | `b741f45ae6231c3ad8535f73a4c23cac8785a20eeb85201a6633ab992fffb3d6` |
| `src/neuralcompression/models/_hific_autoencoder.py` | `181e41345b568de65619af7c4aa1abd1155516b2011d7bb86ac3bfea3f53a262` |
| `src/neuralcompression/models/_hific_encoder_decoder.py` | `2b189909d24b63a14592b2ecc2ad4c54d22a215325779d39748955b6d59ef8f6` |
| `src/neuralcompression/models/_hyperprior_autoencoder.py` | `18b4a621508dd8f7bd5502f46bea49ed47d325ac56e16476b01861ef69ceda10` |
| `src/neuralcompression/layers.py` | `19d310cca660a66cc29ee51016f78cd0f1b2c2ef3e4b054f4235e7332626e74f` |
| `src/neuralcompression/__init__.py` (stand-in) | `413cf8baafcbba829d9c7d5c2d14a56628e3d27a510ea160b07be3ad7090d28e` |
| `src/neuralcompression/functional.py` (stand-in) | `3bcfb9067fab3c8d1e5927b0b6740959eb771d5c9bdddbc2d47b684033b4e9da` |
| `src/neuralcompression/models/__init__.py` (stand-in) | `b77401f86f204fe608f9f1b75ea16ac9107e3889868e64b9c131c5f4b70be6ed` |
