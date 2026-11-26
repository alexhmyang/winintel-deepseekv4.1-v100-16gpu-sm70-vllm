#!/usr/bin/env bash
# Author: 数萌AI服务器 (winintel)
set -euo pipefail
root="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
python="${WININTEL_REFERENCE_ENV:-$root/.venv}/bin/python"
export PATH="$(dirname "$python"):${CUDA_HOME:-/usr/local/cuda-12.8}/bin:$PATH"
export PYTHONPATH="$root:$root/upstream"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCH_CUDA_ARCH_LIST=7.0
if [[ -n "${WININTEL_REFERENCE_SYSROOT:-}" ]]; then
    lib="$WININTEL_REFERENCE_SYSROOT/usr/lib/x86_64-linux-gnu"
    exec uv run --offline --no-project --no-python-downloads \
        "$lib/ld-linux-x86-64.so.2" \
        --library-path "$lib:$(dirname "$(dirname "$python")")/lib/python3.12/site-packages/torch/lib" \
        "$python" "$@"
fi
exec uv run --offline --no-project --python "$python" python "$@"
