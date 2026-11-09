#!/usr/bin/env bash
# Use an existing B300-compatible Torch read-only, with isolated project packages.
set -euo pipefail
BASE=${DEEPPROBE_CACHE:-/home/jinhaopeng/.cache/Paper/DeepProbe-VLM}
TORCH_ENV=${DEEPPROBE_GPU_RUNTIME:?Set DEEPPROBE_GPU_RUNTIME to an authorized compatible GPU environment}
EXPERIMENT=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONNOUSERSITE=1
mkdir -p "$BASE/bootstrap"
[[ -x "$BASE/env/bin/python" ]] || "$TORCH_ENV/bin/python" -m venv "$BASE/env"
"$BASE/env/bin/python" - "$TORCH_ENV" <<'PY'
import json, pathlib, subprocess, sys, sysconfig
base=pathlib.Path(sys.argv[1])
site=subprocess.check_output([str(base/'bin/python'), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip()
(pathlib.Path(sysconfig.get_path('purelib'))/'inherited_gpu_runtime.pth').write_text(site+'\n')
PY
# Refuse a missing Torch before pip can resolve/download a replacement.
"$BASE/env/bin/python" -c 'import torch; assert torch.__version__ == "2.10.0+cu130", torch.__version__'
"$BASE/env/bin/python" -m pip install -r "$EXPERIMENT/requirements/train.txt"
"$BASE/env/bin/python" -m pip install --no-deps -e "$EXPERIMENT"
"$BASE/env/bin/python" -m pip freeze > "$BASE/bootstrap/resolved-packages.txt"
