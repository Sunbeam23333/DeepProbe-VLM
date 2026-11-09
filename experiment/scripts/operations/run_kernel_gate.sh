#!/usr/bin/env bash
# Finite single-node B300 compatibility gate; no training or model download.
set -euo pipefail
RUN=${1:?usage: run_kernel_gate.sh RUN_ID}
[[ "$RUN" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
EXPERIMENT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM
PY=$BASE/env/bin/python
OUT=$BASE/runs/$RUN
export PATH=/home/jinhaopeng/.local/bin:$PATH
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 73
[[ ! -e "$OUT/GATE_COMPLETE.json" ]] || exit 0
exec > >(tee -a "$OUT/gate.log") 2>&1
archive() {
  code=$?
  trap - EXIT
  "$PY" - "$EXPERIMENT" "$OUT" "$RUN" "$code" <<'PY'
import hashlib, importlib.util, json, shutil, sys
from pathlib import Path
root, out, run, code = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
spec = importlib.util.spec_from_file_location('assets', root/'scripts/operations/stage_assets.py')
assets = importlib.util.module_from_spec(spec); spec.loader.exec_module(assets)
uri = 'oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/runs/' + run
record = {'exit_code':code, 'uri':uri, 'scope':'BF16 kernels and NCCL only; not pretrained-model training'}
shutil.copyfile(out/'gate.log', out/'gate.snapshot.log')
files = [p for p in out.iterdir() if p.is_file() and p.name not in {'.lock','ARCHIVED.json','gate.log'}]
for p in files:
    assets.upload_file(p, uri+'/'+p.name)
record['files'] = [{'path':p.name, 'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(out/'ARCHIVED.json').write_text(json.dumps(record,indent=2)+'\n')
assets.upload_file(out/'ARCHIVED.json', uri+'/ARCHIVED.json')
PY
  archive_code=$?
  if [[ "$archive_code" != 0 ]]; then exit "$archive_code"; fi
  exit "$code"
}
trap archive EXIT
cd "$EXPERIMENT"
"$PY" -c 'import torch; assert torch.cuda.device_count() == 8, "Expected a full eight-GPU node"'
timeout --signal=TERM --kill-after=30 300 "$PY" scripts/preflight.py \
  --expected-gpu B300 --output "$OUT/preflight-single.json"
timeout --signal=TERM --kill-after=30 600 "$PY" -m torch.distributed.run \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=29853 scripts/preflight.py \
  --expected-gpu B300 --output "$OUT/preflight-ddp.json"
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
single=json.loads((out/'preflight-single.json').read_text())
ddp=json.loads((out/'preflight-ddp.json').read_text())
assert single['status']==ddp['status']=='gpu_gate_passed'
assert ddp['world_size']==8 and len(ddp['rank_reports'])==8
assert all(r['distributed_all_reduce']=='pass' and not r['errors'] for r in ddp['rank_reports'])
(out/'GATE_COMPLETE.json').write_text(json.dumps({'status':'gpu_gate_passed','training_completed':False})+'\n')
PY
