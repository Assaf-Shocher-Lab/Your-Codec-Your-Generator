# AEIC (vendored, unchanged)

AEIC, "Ultra-Low Bitrate Perceptual Image Compression with Shallow Encoder" (Zhang, Liu, Chen; CVPR 2026).
- Repository: https://github.com/LuizScarlet/AEIC, branch `main`, downloaded as a zip archive
  (sha256 `722058d28411c7def37def4df91cca5f6952e163abed68a09635ec10bef59ea9`).
- Licence: MIT (`LICENSE`, unchanged).

Only the files the model's import chain needs, the C++ sources of AEIC's rANS coder (`src/cpp`, compiled by
`scripts/build_aeic_coder.sh` into `src/codec/`), and `src/compress.py` (not imported; the functions copied from it into
`adapters/aeic/upstream_copies.py` are checked against it by `tests/test_upstream_copies.py`). Every file is
byte-identical to the archive:

| file | sha256 |
|---|---|
| `LICENSE` | `83ed66e2cf9e4ff9db4bf7e0c5f255d3afb4164e33f430708e01b611ab8ebad5` |
| `src/AEIC_practical.py` | `c7a898b4ea8a61389dd8a62c07b1b9d44ade1f1313cb75f3f535ea99c8e786eb` |
| `src/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `src/__pycache__/__init__.cpython-311.pyc` | `7621a106d250ffbb95164cceddec5b540effdfe1fee8946fdb36ceb2c81c25fc` |
| `src/__pycache__/color_fix.cpython-311.pyc` | `22eb158a84c460afefc46500851dcbf9f05167ca7b83d2db4e90a9ce3c84a63e` |
| `src/base_model.py` | `050beda4ad3239472c335ea23e8bd5197cc141324d73c76c25ba7306646c724a` |
| `src/codec/__pycache__/base_entropy_coder.cpython-311.pyc` | `1c78c79348c13a09451ea712f2ba8347c12d89abc4498640f1fa765bb785c649` |
| `src/codec/__pycache__/base_module.cpython-311.pyc` | `96d738b6a5bf7574a04823366ecb9e8cf26e3e97b5924db7f6c76dd83374c8a7` |
| `src/codec/__pycache__/codec_practical.cpython-311.pyc` | `59fb8c1237e9b04077c507e0a1eee5a29de0d890b972f353cb2e042a6deebc51` |
| `src/codec/base_entropy_coder.py` | `bf5a59f35b5507076f46baa147a951940418925e7380f3da79eb70bf5c7a135e` |
| `src/codec/base_module.py` | `a04a45f067bb4fd215ebe0bdd4c0b7536793ef1747d18eda1d7e919a8998b6fb` |
| `src/codec/codec_practical.py` | `2a1c3e577c58c90a5453ed1d6bc0be763fa56f2dcc8f8ece98144f67c0c17977` |
| `src/color_fix.py` | `21b6870f9e0b9a86e0fca199e5490bff67596f7cc8195f94a8e60aa9334e160d` |
| `src/compress.py` | `4b118e310216c9f048684a28783019be945ffafa0075de188112c22717c7fd79` |
| `src/cpp/3rdparty/CMakeLists.txt` | `7d4c84d3a126bb564cb34dadc1fba9cf18fd08fde04eeddff5b4a314a7b7a4e9` |
| `src/cpp/3rdparty/pybind11/CMakeLists.txt` | `f44f9515487dbcec493a900806c57d01f765aca754cb7cd624bae5d81e7680ac` |
| `src/cpp/3rdparty/pybind11/CMakeLists.txt.in` | `f14d71058caef7ea25c09af4d8c01a32fca0f7dddc0c8bff3ae42a81f660b43c` |
| `src/cpp/CMakeLists.txt` | `84c7b4119a9f4fa20e4c4fdbb35de066403c9323369d5e9dd1d6b4673b57b2eb` |
| `src/cpp/ops/CMakeLists.txt` | `28011a9a20c993e07739dcae6ddbea053638b56104fd736635b63901aa11eef1` |
| `src/cpp/ops/ops.cpp` | `d771daa4da7dc1176ad1ec2fc511f78966840d471879ddd625402888ea8ca410` |
| `src/cpp/py_rans/CMakeLists.txt` | `69ab934cfcfe5fca7218d04a06bdaa5837bb48bb242eaf8b2dc30ed9f6221968` |
| `src/cpp/py_rans/py_rans.cpp` | `e101d31c624f4e98ac36ada3260449dcad8077afaf585344dc3a1ff749c5d89c` |
| `src/cpp/py_rans/py_rans.h` | `261839ac7319de59ff24b431d8e6a6ea82005833911ca7e8e990bc71371a3d72` |
| `src/cpp/rans/CMakeLists.txt` | `af5dff61b964e2b0e6953f277204500b7e0c90b3daeb630494e5725a8fa0ef34` |
| `src/cpp/rans/rans.cpp` | `44f66c0770ca8789611b7810f76bbfc1573099fdeb9fc6cea601391b4c3f41bf` |
| `src/cpp/rans/rans.h` | `205528617b3b86cb8d0cda61701e772f7264e69eedb27dd546cf96ce155e77ca` |
| `src/cpp/rans/rans_byte.h` | `585b070b12ea8638ecadaf50980e982552e8f4f20ea237fec480892dc09e247a` |
| `src/custom_forward.py` | `d524907958738ec1e37e6ab4594b02df66bc93e005b88f18ea9b03655c48c5d0` |
| `src/my_utils/devices.py` | `2b51bc63afd173cc129459258b1b334c7b0ec1f9c36e44db9f2213412b6d4f1a` |
| `src/my_utils/vaehook.py` | `250c3c3600fd9d4bd0cd82eea8170f530c067cefa817e23aa0dbce14a1d864cb` |

AEIC's files import each other both as `src.codec...` and as `codec...` / `base_model` / `my_utils...`; importing
`adapters.aeic` puts this folder and its `src/` on `sys.path`.
