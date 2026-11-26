# GLC (upstream, unchanged)

"Generative Latent Coding for Ultra-Low Bitrate Image Compression", CVPR 2024; image model `GLC_Image`
(`src/models/image_model.py`). Copyright (c) Microsoft Corporation, MIT licence (`LICENSE`).
- Source: https://github.com/jzyustc/GLC, obtained as a source archive (no git metadata). Every included file is
  byte-identical to commit `126db25f9c093508cd0c99cee32b53fd60074f9a` (2025-12-24; git blob hashes compared). `src/__init__.py` (one newline) makes `src` a regular package.
- Only the files of `GLC_Image`'s import chain are included (image_model -> common_model, layers, vqgan_arch, loss ->
  utils/lpips) and `get_state_dict` (utils/test_utils.py), byte-identical to that release.
- The adapted codec changes no upstream function; `adapters/glc/symbols.py` re-implements the decoder side of the
  four-part prior from the symbols (`y_hat_from_symbols`).
- Weights (not included): the GLC image checkpoint of the release, placed as `weights/glc/GLC_image.pth.tar` (one file
  for the four qualities q = 0..3).

| file | sha256 | used for |
|---|---|---|
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | MIT |
| `README.md` | `41795517810c8aca003bdea93d2930709e85414fab2247d0b1b2326b0cdfad52` | the papers; the checkpoint name |
| `requirements.txt` | `56b232ca31426f3e7c8572ebeab1eef5d4fd330ef2cd9d12adfaeba7dcc8590f` | upstream's (unpinned) list |
| `src/__init__.py` | `334359b90efed75da5f0ada1d5e6b256f4a6bd0aee7eb39c0f90182a021ffc8b` | makes `src` a package (see above) |
| `src/models/image_model.py` | `0e8adee05dc2122bef6218acca2a17b9595b9ad89c78e06893b0b4aab3575ea8` | `GLC_Image`: enc, hyper_enc / hyper_dec, z_vq, four-part prior, dec, vqgan, test() |
| `src/models/common_model.py` | `50fb8d604b11dcefde8a08dae766f52717e05952814eb02f87e42364a39d0251` | `CompressionModel`: quant, separate_prior, masks, forward_four_part_prior, get_y_gaussian_bits |
| `src/models/layers.py` | `98217d7a035f39ecf5930e8caa5da149ea207f4a6d9322fef2697d2d616b2fb6` | DepthConvBlock(2), ResidualBlockUpsample |
| `src/models/vqgan_arch.py` | `94d4240e2bdf0258aaa4c15fb355a5ec44f2dd60c40c982313d188038acbdfe4` | VQAutoEncoder (encoder, generator), VectorQuantizer (get_indices, get_quan_feat) |
| `src/models/loss.py` | `4af29014a3bd7ec17ea351d05422ca320e630508eb1db8fe620579aae674b334` | `CodePredictionLoss` (built by the constructor; not used at test time) |
| `src/utils/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | |
| `src/utils/test_utils.py` | `3fa82ad198140dd4b29afe627f092f4d058221bf04618b6e354be999dd882f74` | `get_state_dict` (checkpoint loading) |
| `src/utils/lpips/__init__.py` | `c77534c0af4d449e8746152c41ac00ba782cee0217a5e5d19a4554630509c5af` | imported by loss.py |
| `src/utils/lpips/lpips.py` | `0d7226f4da0936b7e43ea93db3b0b9e5bdd38d273982629550fdcc91197ca48f` | imported by loss.py |
| `src/utils/lpips/pretrained_networks.py` | `fca334fa5282d0f6846aaacc84021ab763fdaca829a2ead670f5785bd6573bcb` | |
| `src/utils/lpips/trainer.py` | `4c4aec383c30d7b61044612ff5459f4d54a4ee1915da53ffca6e102c603a8698` | |
| `src/utils/lpips/utils.py` | `745467a4e4f977415763b4849038526d06216a2846b14b1307974cb6a4c14a6f` | |
| `src/utils/lpips/weights/v0.0/alex.pth` | `18720f55913d0af89042f13faa7e536a6ce1444a0914e6db9461355ece1e8cd5` | |
| `src/utils/lpips/weights/v0.0/squeeze.pth` | `c27abd3a0145541baa50990817df58d3759c3f8154949f42af3b59b4e042d0bf` | |
| `src/utils/lpips/weights/v0.0/vgg.pth` | `b9e4236260c3dd988fc79d2a48d645d885afcbb21f9fd595e6744cf7419b582c` | |
| `src/utils/lpips/weights/v0.1/alex.pth` | `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` | |
| `src/utils/lpips/weights/v0.1/squeeze.pth` | `4a5350f23600cb79923ce65bb07cbf57dca461329894153e05a1346bd531cf76` | |
| `src/utils/lpips/weights/v0.1/vgg.pth` | `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868` | |
