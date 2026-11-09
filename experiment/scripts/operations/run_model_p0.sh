#!/usr/bin/env bash
# Submit one finite phase through sslaunch -p low; this script never submits jobs.
# DDP requires the successfully archived single-phase P0_COMPLETE.json as arg 6.
# The P0 runtime records one allowlisted override: DataLoader workers 2 -> 0.
# Normal errors/signals attempt bounded archival. SIGKILL/node loss cannot do so.
set -euo pipefail
if [[ ${1:-} == --help || $# -lt 5 || $# -gt 6 ]]; then
  echo 'Usage: run_model_p0.sh RUN_ID single|ddp DATA_ROOT MEDIA_ROOT CONFIG_PATH [SINGLE_P0_COMPLETE_JSON]'
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
[[ "$1" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$2" == single || "$2" == ddp ]] || exit 2
[[ "$2" != ddp || $# == 6 ]] || { echo 'DDP requires an archived single-phase completion receipt.' >&2; exit 2; }
EXPERIMENT=$(cd "$(dirname "$0")/../.." && pwd)
BASE=/home/jinhaopeng/.cache/Paper/DeepProbe-VLM
PY=$BASE/env/bin/python
[[ -x "$PY" ]] || { echo "Missing project runtime: $PY" >&2; exit 2; }
mkdir -p "$BASE/locks"
exec 9>"$BASE/locks/model-p0.lock"
flock -n 9 || { echo 'Another model P0 phase holds the project lock.' >&2; exit 73; }
export PATH=/home/jinhaopeng/.local/bin:$PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME="$BASE/hf" HF_HUB_CACHE="$BASE/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
exec "$PY" - "$EXPERIMENT" "$BASE" "$@" <<'PY'
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

experiment, base = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
run_id, phase = sys.argv[3:5]
data, media, config = [Path(value).resolve() for value in sys.argv[5:8]]
predecessor = Path(sys.argv[8]).resolve() if len(sys.argv) == 9 else None
out = base / 'runs' / run_id
uri = 'oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/runs/' + run_id
limit = 100_000_000_000
if out.exists():
    raise FileExistsError('Run directory already exists; never retry automatically or overwrite a prior run')
for path in (experiment, base, data, media, config):
    if not path.is_relative_to(Path('/home/jinhaopeng')):
        raise ValueError(f'All inputs, code and staging must be in the personal home: {path}')
out.mkdir(parents=True, exist_ok=False)
for directory in ('tmp', 'cache'):
    (out / directory).mkdir()
os.environ.update(TMPDIR=str(out/'tmp'), TMP=str(out/'tmp'), TEMP=str(out/'tmp'),
                  XDG_CACHE_HOME=str(out/'cache'), CUDA_CACHE_PATH=str(out/'cache/cuda'),
                  TRITON_CACHE_DIR=str(out/'cache/triton'), TORCH_HOME=str(out/'cache/torch'),
                  TORCH_EXTENSIONS_DIR=str(out/'cache/extensions'), TORCHINDUCTOR_CACHE_DIR=str(out/'cache/inductor'))
for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK'):
    os.environ.pop(key, None)
original_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
first_device = original_devices.split(',')[0] if original_devices else '0'
log = (out/'stdout.log').open('x', encoding='utf-8')
log_lock = threading.Lock()
active = None
stages = []
source = out/'source'
inputs = out/'inputs'
training = out/'training'

def emit(value):
    text = str(value).rstrip('\n')
    with log_lock:
        log.write(text+'\n'); log.flush()
        print(text, flush=True)

def dump(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()

def cache_bytes():
    # Only this project's cache; never recurse through the million-file Paper tree.
    total = 0
    for directory, folders, names in os.walk(base, followlinks=False):
        folders[:] = [name for name in folders if not (Path(directory)/name).is_symlink()]
        for name in names:
            path = Path(directory)/name
            try:
                if not path.is_symlink(): total += path.stat().st_size
            except FileNotFoundError:
                pass  # Own transient caches may rotate during a stage.
    return total

def stop_process(process):
    if process is not None and process.poll() is None:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: return
        try: process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=15)

def stage(name, command, seconds, *, one_gpu=False, monitor_storage=True):
    global active
    env = dict(os.environ)
    if one_gpu: env['CUDA_VISIBLE_DEVICES'] = first_device
    env['PYTHONPATH'] = str(source/'src')
    started = time.monotonic()
    record = {'name': name, 'command': [str(item) for item in command], 'timeout_seconds': seconds,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    stages.append(record)
    emit(f'START {name} (timeout {seconds}s)')
    active = subprocess.Popen(record['command'], cwd=source if source.exists() else experiment, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              errors='replace', start_new_session=True)
    process = active
    def stream_output():
        for line in process.stdout:
            emit(line)
    pump = threading.Thread(target=stream_output, daemon=True)
    pump.start()
    next_space_check = started + 60
    try:
        while process.poll() is None:
            now = time.monotonic()
            if now-started > seconds:
                raise TimeoutError(f'{name} exceeded its finite time budget')
            if monitor_storage and now >= next_space_check:
                used = cache_bytes()
                if used >= limit-1_000_000_000:
                    raise RuntimeError('Project staging reached the 99 GB guard; stopping before additional checkpoints')
                next_space_check = now+60
            time.sleep(1)
        record['returncode'] = process.returncode
        if process.returncode:
            raise RuntimeError(f'{name} exited with {process.returncode}')
    finally:
        stop_process(process)
        pump.join(timeout=30)
        active = None
        record['returncode'] = process.returncode
        record['elapsed_seconds'] = time.monotonic()-started
        emit(f'END {name}: code={process.returncode}, elapsed={record["elapsed_seconds"]:.1f}s')

def require_status(path, expected):
    report = json.loads(path.read_text())
    if report.get('status') != expected:
        raise ValueError(f'{path.name}: expected {expected}')
    return report

def diagnostic(path):
    report = require_status(path, 'engineering_checks_passed')
    checks = report['checks']
    if not checks.get('loss_finite') or checks['cache_equivalence']['status'] != 'passed':
        raise ValueError('Finite loss / cache equivalence diagnostic failed')
    if any(value['status'] not in ('passed','expected_frozen') for value in checks['gradient_groups'].values()):
        raise ValueError('An expected parameter group did not pass the gradient diagnostic')
    if report.get('checkpoint') and report['checkpoint_reload']['status'] != 'passed':
        raise ValueError('Checkpoint reload diagnostic failed')
    return report

def weights_hashes():
    return {str(path.relative_to(training/'final')): digest(path)
            for path in sorted((training/'final').rglob('*.safetensors'))}

def verify_step(path, expected):
    state = json.loads(path.read_text())
    if state.get('global_step') != expected:
        raise ValueError(f'{path}: optimizer step must be {expected}')
    for record in state.get('log_history', []):
        for name, value in record.items():
            if ('loss' in name) and isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f'Nonfinite logged loss: {name}')
    return state

def reserve_storage(counts):
    total = sum(group['total'] for group in counts.values())
    trainable = sum(group['trainable'] for group in counts.values())
    # Two FP32 model+Adam checkpoints, one FP32 final, and bounded log/cache margin.
    reserve = 12*total + 16*trainable + 3_000_000_000
    used = cache_bytes()
    free = shutil.disk_usage(base).free
    dump(out/'storage_admission.json', {'project_cache_bytes': used, 'limit_bytes': limit,
         'reserved_additional_bytes': reserve, 'filesystem_free_bytes': free, 'parameter_counts': counts,
         'formula': '2*(4*all_parameters + 8*trainable_parameters) + 4*all_parameters + 3 GB'})
    if used+reserve > limit or free < reserve:
        raise RuntimeError('Insufficient project staging budget for this complete phase; archive/clean prior owned artifacts first')

def snapshot_and_verify():
    source.mkdir(); inputs.mkdir()
    for name in ('src','scripts','configs','requirements'):
        for path in sorted((experiment/name).rglob('*')):
            if '__pycache__' in path.parts or path.suffix == '.pyc': continue
            if path.is_symlink(): raise ValueError(f'Source snapshot rejects symlinks: {path}')
            if path.is_file():
                target = source/path.relative_to(experiment)
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path,target)
    for name in ('pyproject.toml','README.md'):
        shutil.copyfile(experiment/name, source/name)
    dump(out/'source_manifest.json', {'files': {str(p.relative_to(source)):digest(p) for p in sorted(source.rglob('*')) if p.is_file()}})
    ready = json.loads((data/'READY.json').read_text())
    report_path = data/'preparation_report.json'
    if ready.get('status') != 'p0_media_and_offline_token_preflight_passed' or ready['report_sha256'] != digest(report_path):
        raise ValueError('DATA_ROOT READY/preparation report is unverified or changed')
    report = json.loads(report_path.read_text())
    if report['config_sha256'] != digest(config): raise ValueError('Config differs from successful token preflight')
    if Path(report['media_root']).resolve() != media: raise ValueError('Media root differs from verified preparation')
    for name in ('train.jsonl','validation.jsonl'):
        if ready['manifest_sha256'].get(name) != digest(data/name): raise ValueError(f'Changed manifest: {name}')
        shutil.copyfile(data/name, inputs/name)
    for name in ('READY.json','preparation_report.json','selected_media.json','benchmark_source_registry.json','coverage.jsonl','token_budget.json'):
        if name in report.get('output_sha256', {}) and digest(data/name) != report['output_sha256'][name]:
            raise ValueError(f'Changed preparation evidence: {name}')
        shutil.copyfile(data/name, inputs/name)
    shutil.copyfile(config, inputs/'config.json')
    for entry in json.loads((inputs/'selected_media.json').read_text()):
        path = (media/entry['video_path']).resolve()
        if not path.is_relative_to(media) or path.stat().st_size != entry['size_bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Selected media bytes differ from prepared data')
    raw = json.loads((inputs/'config.json').read_text())
    original = json.loads((source/'configs/train/ouro_pilot.json').read_text())
    if raw['training'] != original['training']:
        raise ValueError('Use the unchanged 20-step Ouro pilot training/optimizer schedule')
    comparable = {key:value for key,value in raw['model'].items() if key != 'max_text_tokens'}
    expected = {key:value for key,value in original['model'].items() if key != 'max_text_tokens'}
    if comparable != expected: raise ValueError('P0 model recipe differs from Ouro R4/F8/G4 baseline')
    if raw['training'].get('dataloader_num_workers') != 2:
        raise ValueError('Expected the prepared Ouro pilot to declare two DataLoader workers')
    runtime = json.loads(json.dumps(raw))
    runtime['training']['dataloader_num_workers'] = 0
    restored = json.loads(json.dumps(runtime))
    restored['training']['dataloader_num_workers'] = raw['training']['dataloader_num_workers']
    if restored != raw:
        raise ValueError('P0 runtime config may only change training.dataloader_num_workers')
    runtime_path = inputs/'config.runtime.json'
    dump(runtime_path, runtime)
    override_path = inputs/'runtime_overrides.json'
    dump(override_path, {
        'status':'single_allowlisted_runtime_override',
        'field':'training.dataloader_num_workers',
        'requested':raw['training']['dataloader_num_workers'],
        'runtime':runtime['training']['dataloader_num_workers'],
        'reason':'Avoid multiprocessing resource_sharer UNIX-socket IPC on the shared home filesystem during finite P0 engineering validation.',
        'scope':'P0 train and eval loading only; model, optimizer, schedule, data and token budgets are unchanged.',
        'requested_config_sha256':digest(inputs/'config.json'),
        'runtime_config_sha256':digest(runtime_path),
    })
    return {'config_sha256':digest(inputs/'config.json'),
            'runtime_config_sha256':digest(runtime_path),
            'runtime_overrides_sha256':digest(override_path),
            'manifest_sha256':{name:digest(inputs/name) for name in ('train.jsonl','validation.jsonl')},
            'source_manifest_sha256':digest(out/'source_manifest.json'),
            'preparation_report_sha256':digest(inputs/'preparation_report.json'),
            'data_gate_scope':{'purpose':ready.get('purpose', report.get('purpose')),
                              'formal_source_gate':ready.get('benchmark_source_identity_gate',report.get('benchmark_source_identity_gate','not_certified_by_p0_runner')),
                              'formal_training_ready':ready.get('formal_training_ready',False),
                              'certification':'P0 engineering only; no formal benchmark contamination guarantee'}}

def check_predecessor(identity):
    if predecessor is None: raise ValueError('DDP requires an explicit single-phase receipt')
    if not predecessor.is_relative_to(base/'runs'): raise ValueError('Predecessor must be this project\'s local archived run receipt')
    prior = json.loads(predecessor.read_text())
    if prior.get('phase') != 'single' or prior.get('status') != 'p0_engineering_complete':
        raise ValueError('Single phase has not completed successfully')
    for name in ('config_sha256','runtime_config_sha256','runtime_overrides_sha256'):
        if prior.get(name) != identity[name]:
            raise ValueError(f'DDP {name} must match the completed single phase')
    if prior['manifest_sha256'] != identity['manifest_sha256']:
        raise ValueError('DDP data must match the completed single phase')
    if prior['source_manifest_sha256'] != identity['source_manifest_sha256']:
        raise ValueError('DDP code snapshot differs from the completed single phase')
    if prior['preparation_report_sha256'] != identity['preparation_report_sha256']:
        raise ValueError('DDP preparation/source audit differs from the completed single phase')
    archived = require_status(predecessor.parent/'ARCHIVED.json','archive_verified')
    manifest_path = predecessor.parent/'ARCHIVE_MANIFEST.json'
    if digest(manifest_path) != archived['archive_manifest_sha256']: raise ValueError('Predecessor archive manifest changed')
    manifest = json.loads(manifest_path.read_text())
    entry = next((entry for entry in manifest['files'] if entry['path'] == predecessor.name), None)
    if entry is None or entry['sha256'] != digest(predecessor): raise ValueError('Single completion receipt is not in its verified archive')
    for name in ('before.json','after.json'):
        entry = next(entry for entry in manifest['files'] if entry['path'] == name)
        if digest(predecessor.parent/name) != entry['sha256']: raise ValueError('Predecessor diagnostics changed')
        diagnostic(predecessor.parent/name)
    for path in (predecessor, predecessor.parent/'ARCHIVED.json', manifest_path):
        shutil.copyfile(path, inputs/('predecessor-'+path.name))
    return prior['parameter_counts']

def archive():
    # Exclude active archive stdout and reproducible per-run temporary/kernel caches.
    log.flush(); shutil.copyfile(out/'stdout.log', out/'stdout.snapshot.log')
    helper = out/'archive_payload.py'
    helper.write_text(ARCHIVE_HELPER)
    stage('archive', [sys.executable, helper, source, out, uri, run_id, phase], 7200, monitor_storage=False)
    require_status(out/'ARCHIVED.json','archive_verified')

ARCHIVE_HELPER = r'''
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
source,out,uri,run_id,phase=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]
def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): d.update(b)
 return d.hexdigest()
spec=importlib.util.spec_from_file_location('p0_assets',source/'scripts/operations/stage_assets.py')
assets=importlib.util.module_from_spec(spec); spec.loader.exec_module(assets)
listing=subprocess.run(['ossutil','ls',uri+'/'],check=True,capture_output=True,text=True).stdout
if any(line.split() and line.split()[-1].startswith(uri+'/') for line in listing.splitlines()):
 raise FileExistsError('Run bucket prefix already contains objects; do not overwrite/retry automatically')
files=[]
for path in sorted(out.rglob('*')):
 relative=path.relative_to(out)
 if relative.parts[0] in ('tmp','cache') or relative.as_posix() in ('stdout.log','ARCHIVE_MANIFEST.json','ARCHIVED.json'): continue
 if path.is_symlink(): raise ValueError('Unexpected symlink in run archive: '+str(relative))
 if path.is_file(): files.append({'path':relative.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
manifest={'run_id':run_id,'phase':phase,'uri':uri,'files':files,'file_count':len(files),
          'total_bytes':sum(f['bytes'] for f in files),'verification':'OSS cp CRC + exact-key length; local SHA256 inventory',
          'excluded':['active stdout.log (stdout.snapshot.log archived)','tmp/','cache/']}
manifest_path=out/'ARCHIVE_MANIFEST.json'
with manifest_path.open('x') as f: json.dump(manifest,f,indent=2); f.write('\n')
for entry in files:
 path=out/entry['path']; assets.upload_file(path,uri+'/'+entry['path'])
 if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']: raise RuntimeError('Payload changed while archiving')
assets.upload_file(manifest_path,uri+'/ARCHIVE_MANIFEST.json')
# Read back the authoritative small manifest and completion record as SHA checks.
for name in ['ARCHIVE_MANIFEST.json']+(['P0_COMPLETE.json'] if (out/'P0_COMPLETE.json').exists() else []):
 back=out/'tmp'/('readback-'+name)
 subprocess.run(['ossutil','cp',uri+'/'+name,str(back),'-f','--no-progress'],check=True)
 if sha(back)!=sha(out/name): raise RuntimeError('Archive key readback SHA256 mismatch: '+name)
receipt={'status':'archive_verified','run_id':run_id,'phase':phase,'uri':uri,
         'archive_manifest_sha256':sha(manifest_path),'file_count':len(files),'total_bytes':manifest['total_bytes'],
         'remote_manifest_and_completion_sha256_readback':True}
pending=out/'ARCHIVED.json.pending'
with pending.open('x') as f: json.dump(receipt,f,indent=2); f.write('\n')
assets.upload_file(pending,uri+'/ARCHIVED.json')
pending.replace(out/'ARCHIVED.json')
'''

def cleanup_verified_weights():
    # Only this successful run's archived, unchanged weight/optimizer payloads.
    archived=require_status(out/'ARCHIVED.json','archive_verified')
    manifest_path=out/'ARCHIVE_MANIFEST.json'
    if digest(manifest_path)!=archived['archive_manifest_sha256']: raise ValueError('Archive manifest changed before cleanup')
    removed=[]
    for entry in json.loads(manifest_path.read_text())['files']:
        relative=Path(entry['path'])
        if len(relative.parts)<3 or relative.parts[0]!='training': continue
        if relative.parts[1] not in ('final','checkpoint-10','checkpoint-20'): continue
        if relative.suffix not in ('.safetensors','.bin','.pt','.pth'): continue
        path=out/relative
        if path.is_symlink() or not path.resolve().is_relative_to(training.resolve()): raise ValueError('Unexpected cleanup path')
        if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']: raise ValueError('Archived file changed; refusing cleanup')
        path.unlink(); removed.append(entry)
    dump(out/'CLEANED_WEIGHTS.json',{'status':'only_verified_archived_weights_removed','files':removed,
                                  'archive_manifest_sha256':archived['archive_manifest_sha256']})

def on_signal(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}; stop owned process and attempt bounded archival')

signal.signal(signal.SIGTERM,on_signal); signal.signal(signal.SIGINT,on_signal)
exit_code=1
try:
    identity=snapshot_and_verify()
    if cache_bytes()>=limit-3_000_000_000: raise RuntimeError('Project cache has no bounded working space')
    stage('full-node-inventory',[sys.executable,'-c','import torch; assert torch.cuda.device_count()==8, "Expected exactly eight visible GPUs from one scheduled full node"'],120)
    stage('preflight-single',[sys.executable,source/'scripts/preflight.py','--expected-gpu','B300','--output',out/'preflight-single.json'],300,one_gpu=True)
    require_status(out/'preflight-single.json','gpu_gate_passed')
    common=[sys.executable,source/'scripts/operations/verify_p0_checkpoint.py','--config',inputs/'config.runtime.json',
            '--train-manifest',inputs/'train.jsonl','--media-root',media]
    if phase=='single':
        stage('before-diagnostic',common+['--output',out/'before.json'],1800,one_gpu=True)
        counts=diagnostic(out/'before.json')['parameter_counts']
    else:
        counts=check_predecessor(identity)
        port=str(29000+int(hashlib.sha256(run_id.encode()).hexdigest()[:4],16)%5000)
        launch=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--node-rank=0','--nproc-per-node=8',
                '--master-addr=127.0.0.1','--master-port='+port]
        stage('preflight-ddp',launch+[source/'scripts/preflight.py','--expected-gpu','B300','--output',out/'preflight-ddp.json'],600)
        report=require_status(out/'preflight-ddp.json','gpu_gate_passed')
        if report['world_size']!=8 or len(report.get('rank_reports',[]))!=8 or any(
            row['distributed_all_reduce']!='pass' or row['errors'] for row in report['rank_reports']):
            raise ValueError('Eight-rank NCCL gate is incomplete')
    reserve_storage(counts)
    train_args=['--config',inputs/'config.runtime.json','--train-manifest',inputs/'train.jsonl',
                '--eval-manifest',inputs/'validation.jsonl','--media-root',media,'--output-dir',training]
    train_command=([sys.executable,'-m','deepprobe_vlm.train'] if phase=='single' else launch+['-m','deepprobe_vlm.train'])+train_args
    stage('train-20',train_command,5400,one_gpu=phase=='single')
    verify_step(training/'trainer_state.json',20)
    if phase=='single':
        verify_step(training/'checkpoint-10/trainer_state.json',10)
        for name in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_provenance.json','deepprobe_config.json'):
            path=training/'checkpoint-10'/name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f'Checkpoint-10 is not a complete optimizer/RNG resume point: missing {name}')
        before=weights_hashes()
        if not before: raise ValueError('Final model has no saved safetensors weights')
        dump(out/'before_resume_final_weights.json',before)
        shutil.copyfile(training/'run_manifest.json',out/'before_resume_run_manifest.json')
        stage('resume-checkpoint10-to20',train_command+['--resume',training/'checkpoint-10'],3600,one_gpu=True)
        verify_step(training/'trainer_state.json',20)
        after=weights_hashes()
        dump(out/'resume_comparison.json',{'engineering_resume_succeeded':True,'unchanged_config_and_data':True,
             'resume_from_optimizer_step':10,'final_optimizer_step':20,'final_weight_file_hashes_identical':before==after,
             'before':before,'after':after,'limitation':'Different serialization hashes do not certify exact numerical resume; this is not a tensorwise comparison.'})
    stage('after-diagnostic',common+['--checkpoint',training/'final','--output',out/'after.json'],1800,one_gpu=True)
    diagnostic(out/'after.json')
    dump(out/'P0_COMPLETE.json',{'status':'p0_engineering_complete','run_id':run_id,'phase':phase,**identity,
         'parameter_counts':counts,'optimizer_steps':20,'world_size':1 if phase=='single' else 8,
         'effective_global_batch':1 if phase=='single' else 8,'single_resume_replay_verified':phase=='single',
         'quality_result':False,'purpose':'Engineering connectivity, gradients, cache and existing-checkpoint reload only'})
    exit_code=0
except BaseException as exc:
    emit(traceback.format_exc())
    exit_code=130 if isinstance(exc,KeyboardInterrupt) else 1
finally:
    stop_process(active)
    dump(out/'EXIT_STATUS.json',{'exit_code':exit_code,'run_id':run_id,'phase':phase,'stages':stages,
         'automatic_retry':False,'signal_limit':'SIGKILL or node disappearance can prevent final archival; inspect exact job before any manual recovery.'})
    try:
        # Use the snapshot when available; pre-snapshot failures still archive via
        # the original known source without pretending the snapshot was complete.
        if not (source/'scripts/operations/stage_assets.py').is_file():
            target=source/'scripts/operations'; target.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(experiment/'scripts/operations/stage_assets.py',target/'stage_assets.py')
            (source/'scripts/data').mkdir(parents=True,exist_ok=True)
            shutil.copyfile(experiment/'scripts/data/download_hf.py',source/'scripts/data/download_hf.py')
            shutil.copytree(experiment/'src',source/'src',dirs_exist_ok=True)
        archive()
        if exit_code==0:
            cleanup_verified_weights()
    except BaseException:
        emit('ARCHIVE/CLEANUP FAILURE: local evidence retained; no automatic retry.\n'+traceback.format_exc())
        exit_code=exit_code or 74
    log.close()
sys.exit(exit_code)
PY
