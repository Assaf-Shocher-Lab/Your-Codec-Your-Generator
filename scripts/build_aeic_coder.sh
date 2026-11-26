#!/bin/bash
# Build AEIC's rANS entropy coder from AEIC's own C++ sources (third_party/aeic/src/cpp, unchanged) into
# third_party/aeic/src/codec/, where AEIC's codec/base_entropy_coder.py imports it (MLCodec_rans, MLCodec_CXX).
# AEIC's CMake build downloads pybind11 at configure time; this compiles the same sources directly with the pybind11
# headers that ship with PyTorch. Run from the repository root with the Python that runs the chains:
#   PYBIN=python bash scripts/build_aeic_coder.sh
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/aeic/src/cpp"
OUT="$ROOT/third_party/aeic/src/codec"
PYTHON=${PYBIN:-python3}
INC=$("$PYTHON" -c "import sysconfig, torch.utils.cpp_extension as c; print(' '.join('-I' + p for p in c.include_paths() + [sysconfig.get_paths()['include']]))")
EXT=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
FLAGS="-O3 -DNDEBUG -std=c++17 -fPIC -shared -pthread"
g++ $FLAGS $INC -I"$SRC/rans" "$SRC/rans/rans.cpp" "$SRC/py_rans/py_rans.cpp" -o "$OUT/MLCodec_rans$EXT"
g++ $FLAGS $INC "$SRC/ops/ops.cpp" -o "$OUT/MLCodec_CXX$EXT"
"$PYTHON" -c "import sys; sys.path.insert(0, '$OUT/..'); from codec.MLCodec_rans import RansEncoder; from codec.MLCodec_CXX import pmf_to_quantized_cdf; print('AEIC coder import OK')"
