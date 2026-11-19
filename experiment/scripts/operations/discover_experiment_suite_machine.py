#!/usr/bin/env python3
"""Generate a private Ali machine candidate from existing real metadata only.

No scheduler, GPU, upload, cache decode or checkpoint restore is performed.
Historical source proofs provide candidate byte pins; full current validation
is deferred to complete_experiment_suite.py. This is never an admission receipt.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re


def pin(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError('Required existing metadata file unavailable: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def clean_job_template(original):
    """Extract the reviewed infrastructure fields, never old commands/server state."""
    if original.get('apiVersion') != 'kubeflow.org/v1' or original.get('kind') != 'PyTorchJob':
        raise ValueError('Expected the actual Kubeflow PyTorchJob template')
    replicas = original['spec']['pytorchReplicaSpecs']
    if set(replicas) != {'Master'}: raise ValueError('Template is not single-node')
    pod = replicas['Master']['template']['spec']
    if len(pod['containers']) != 1 or pod.get('nodeName') or pod.get('initContainers'):
        raise ValueError('Template has unexpected execution or node binding')
    selected = {name: deepcopy(pod[name]) for name in ('dnsPolicy', 'hostNetwork', 'imagePullSecrets',
                'nodeSelector', 'priorityClassName', 'tolerations', 'volumes') if name in pod}
    c = pod['containers'][0]
    selected['containers'] = [{name: deepcopy(c[name]) for name in ('image', 'imagePullPolicy', 'name',
        'ports', 'resources', 'securityContext', 'volumeMounts') if name in c}]
    return {'apiVersion': original['apiVersion'], 'kind': original['kind'], 'metadata': {},
        'spec': {'cleanPodPolicy': original['spec']['cleanPodPolicy'],
            'ttlSecondsAfterFinished': original['spec']['ttlSecondsAfterFinished'],
            'pytorchReplicaSpecs': {'Master': {'replicas': 1, 'restartPolicy': 'Never',
                'template': {'metadata': {}, 'spec': selected}}}}}


def discover(base, source_root, suite_id, workers, priority, *, unlimited_oss=False):
    base, source_root = Path(base).absolute(), Path(source_root).absolute()
    p1 = Path(str(base) + '-p1')
    if not 1 <= workers <= 4 or priority not in {'normal', 'low'} or not re.fullmatch('[a-z][a-z0-9-]*', suite_id):
        raise ValueError('Invalid suite identity, worker count or priority')
    if not isinstance(unlimited_oss, bool):
        raise ValueError('OSS quota opt-in must be an explicit boolean')
    control = p1 / ('suite-' + suite_id)
    home = base.parents[2]
    ready = p1 / 'formal-source-final-20261009-v1/FORMAL_READY.json'
    data = p1 / 'formal-training-data-20261009-v1'
    training = {'train_manifest': pin(data / 'train.jsonl'), 'eval_manifest': pin(data / 'validation.jsonl'),
                'formal_ready': pin(ready), 'data_dir': str(data), 'media_root': str(base / 'media/llava-p0'),
                'frozen_source_manifest': pin(p1 / 'releases/deepprobe-20261021-declared-v9/P1_SOURCE_MANIFEST.json')}
    proofs = {'smollm2': p1 / 'formal-preflight-v9-smollm2-s29-alignment-20261027-prepare-1/CPU_INPUTS_PREPARED.json',
              'ouro': p1 / 'formal-preflight-v9-r4-s29-alignment-20261021-prepare-1/CPU_INPUTS_VERIFIED_PENDING_CAPACITY.json'}
    backbones, sources = {}, {}
    for family, path in proofs.items():
        proof = json.loads(path.read_text()); sources[family] = pin(path)
        offline = proof['offline']
        assets = {'files': []}
        for label in ('language', 'vision'):
            value = offline[label]; root = Path(value['resolved_path'])
            assets[label + '_root'] = str(root.resolve())
            assets['files'].extend({'path': str(root.resolve() / name), **row} for name, row in sorted(value['files'].items()))
            identifier = value['configured_id']; revision = value['revision']
            hub = base / 'hf/hub' / ('models--' + identifier.replace('/', '--')) / 'snapshots' / revision
            if hub.resolve() != hub:
                links = [{'path': str(x), 'target': os.readlink(x)} for x in [*reversed(hub.parents), hub] if x.is_symlink()]
                canonical = json.dumps(value['files'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
                assets.setdefault('snapshot_aliases', {})[label] = {'hub_snapshot': str(hub), 'physical_root': str(root.resolve()),
                    'links': links, 'inventory_sha256': hashlib.sha256(canonical).hexdigest()}
        for name in (('smollm2',) if family == 'smollm2' else ('ouro_r1', 'ouro_r2', 'ouro_r4')):
            backbones[name] = deepcopy(assets)
    tools = {'kubectl': str(home / '.local/bin/kubectl'), 'kubeconfig': '/mnt/vepfs/base/infra/.kube/aliconfig',
             'ossutil': str(home / '.local/bin/ossutil'), 'oss_config': str(home / '.ossutilconfig')}
    profile = {'schema_version': 2, 'fleet_id': suite_id, 'priority': priority, 'gpu_product': 'NVIDIA-B300',
               'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
               'workers': [{'worker_id': 'worker-' + str(i).zfill(2), 'job_name': 'jinhaopeng-dpvlm-' + suite_id + '-worker-' + str(i).zfill(2),
                            'namespace': 'default'} for i in range(1, workers + 1)]}
    # Accounting policy is shared with maintenance and remains fixed even when
    # executable source moves into a new immutable release below BASE.
    roots = [str(base), str(p1), str(home / 'Paper/DeepProbe-VLM')]
    if (home / '.config/deepprobe-vlm').is_dir(): roots.append(str(home / '.config/deepprobe-vlm'))
    env = {'schema_version': 1, 'suite_id': suite_id, 'experiment_root': str(source_root), 'python': str(base / 'env/bin/python'),
        'machine_identity': {'uid': os.getuid(), 'gid': os.getgid()},
        'run_root': str(base / 'runs' / suite_id), 'archive_root': 'oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/' + suite_id,
        'execution_profile': profile, 'source_files': [], 'tools': tools,
        'allocation_identity': {'job_name': profile['workers'][0]['job_name'], 'namespace': 'default',
                                'project_name_prefixes': ['jinhaopeng-dpvlm-', 'deepprobe-']},
        'runtime_environment': {'HF_HOME': str(base / 'hf'), 'HF_HUB_CACHE': str(base / 'hf/hub')},
        'storage': {'schema_version': 2, 'roots': roots, 'ledger_root': str(base / 'locks/suite-storage-ledger'),
            'home_limit_bytes': 150000000000, 'oss_limit_bytes': None if unlimited_oss else 150000000000,
            'oss_prefixes': ['oss://ss-devbox-stream/jinhaopeng/Paper/DeepProbe-VLM/'],
            'home_safety_bytes': 2000000000, 'oss_safety_bytes': 2000000000, 'free_guard_bytes': 1000000000},
        'task_reserve_bytes': {'startup': 1000000000, 'gate': 1000000000, 'train_alignment': 35000000000,
            'train_sft': 65000000000, 'evaluate': 3000000000, 'latency': 1000000000, 'compare_latency': 100000000,
            'archive': 67108864, 'cleanup': 100000000, 'aggregate': 1000000000},
        'assets': {'training': training, 'backbones': backbones, 'benchmarks': {},
            'development': {'manifest': training['eval_manifest'], 'media_root': training['media_root']}}, 'history': {}}
    if not unlimited_oss:
        env['modelscope_token_file'] = str(home / '.config/deepprobe-vlm/modelscope-token')
    paths = {'videomme899': (p1 / 'videomme-reviewed-cpu-20261021-a1/manifest/videomme.jsonl',
                            p1 / 'benchmark-media-20261012/videomme/FRAME_CACHE_RECEIPT.json'),
             'tempcompass_mc': (p1 / 'benchmark-media-tempcompass-20261021-a1/manifest.jsonl',
                               p1 / 'benchmark-media-tempcompass-20261021-a1/FRAME_CACHE_RECEIPT.json')}
    preparations = []
    for benchmark, (manifest, receipt) in paths.items():
        for frames in (4, 8, 16):
            request = {'benchmark': benchmark, 'frames': frames, 'source_manifest': pin(manifest),
                'accepted_decode_receipt': pin(receipt), 'output_dir': str(control / 'media-metadata' / (benchmark + '-f' + str(frames))),
                'oss_prefix': 'jinhaopeng/Paper/DeepProbe-VLM/benchmark-cache/shape-ablation/' + suite_id + '/' + benchmark + '/uniform' + str(frames) + '/',
                'media_root': str(base / 'media' / suite_id / (benchmark + '-f' + str(frames))),
                'max_png_bytes': (8000000000 if benchmark == 'videomme899' else 4000000000) * frames // 8,
                'metadata_reserve_bytes': 128 * 1024 * 1024}
            if benchmark == 'videomme899':
                request['archive_audit'] = pin(p1 / 'benchmark-videomme-oss-decode-20261013/release-v1/ACCEPTED_ARCHIVE_AUDIT.json')
            preparations.append(request)
    actual_source_available = (source_root / 'src/deepprobe_vlm/suite_workflow.py').is_file()
    template_source = pin(p1 / 'smollm2-s29-align-a5-submit-20261118-v1/exact_job.stdout')
    return {'schema_version': 1, 'environment': env, 'state_root': str(control / 'state'), 'benchmark_preparations': preparations,
            'history_imports': {}, 'job_template_source': template_source,
            'job_template_document': clean_job_template(json.loads(Path(template_source['path']).read_text())),
            'discovery': {'status': 'REAL_MACHINE_CANDIDATE_NOT_ADMITTED', 'source_metadata_proofs': sources,
                          'suite_source_deployed': actual_source_available, 'current_model_payload_sha_verified': False,
                          'historical_acceptances_imported': False, 'cache_build_executed': False, 'gpu_submitted': False},
            'recovery_candidates': {'train__smollm2_s29__alignment': {'checkpoint': str(base / 'runs/dp-p1-smollm2-f8-g4-s29-alignment-a5/training/checkpoint-350'),
                'checkpoint_step': 350, 'archive_required': True, 'independent_import_required': True},
                'train__ouro_r4_s17__sft': {'checkpoint_step': 600, 'archive_restore_required': True, 'independent_import_required': True}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/home/jinhaopeng/.cache/Paper/DeepProbe-VLM'))
    parser.add_argument('--source-root', type=Path, default=Path('/home/jinhaopeng/Paper/DeepProbe-VLM/experiment'))
    parser.add_argument('--suite-id', required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--priority', choices=('normal', 'low'), default='normal')
    parser.add_argument('--unlimited-oss', action='store_true',
                        help='Explicitly remove only the OSS capacity quota; retain home limits and exact archive integrity checks')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--capacity-snapshot', type=Path, help='Pinned observed storage evidence; never replaces fresh admission')
    parser.add_argument('--capacity-recovery-plan', type=Path, help='Exact reviewable retirement/migration plan; no automatic deletion')
    parser.add_argument('--runtime-python', type=Path, help='Existing isolated suite runtime; original training runtime remains frozen')
    parser.add_argument('--history-import-index', type=Path, help='Reviewed task-to-entry/review pin mapping')
    parser.add_argument('--recovery-preparation-index', type=Path, help='Explicit reviewed archive/restore request list')
    parser.add_argument('--history-preparation-index', type=Path, help='Reviewed accepted checkpoint full-GET/import request list')
    args = parser.parse_args()
    result = discover(args.base, args.source_root, args.suite_id, args.workers, args.priority,
                      unlimited_oss=args.unlimited_oss)
    for name in ('capacity_snapshot', 'capacity_recovery_plan'):
        if getattr(args, name): result[name] = pin(getattr(args, name))
    if args.runtime_python:
        if not args.runtime_python.is_file(): raise ValueError('Suite runtime executable unavailable')
        result['environment']['python'] = str(args.runtime_python.absolute())
    indexes = {}
    for arg, field, expected in ((args.history_import_index, 'history_imports', dict),
                                 (args.recovery_preparation_index, 'recovery_preparations', list),
                                 (args.history_preparation_index, 'history_preparations', list)):
        if arg:
            proof = pin(arg); value = json.loads(arg.read_text())
            if not isinstance(value, expected): raise ValueError('Invalid ' + field + ' index')
            result[field] = value; indexes[field] = proof
    if indexes: result['input_indexes'] = indexes
    raw = json.dumps(result, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as stream: stream.write(raw)
    else: print(raw, end='')

if __name__ == '__main__': main()
