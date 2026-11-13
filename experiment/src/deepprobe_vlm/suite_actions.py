"""Concrete task adapters for the serial experiment suite.

The public DAG is scientific intent. A byte-pinned environment supplies local
assets and independently audited historical inputs. This module never submits
jobs, changes priority, or deletes unowned historical runs. A low allocation may
be preempted; only explicitly resumed tasks may reuse partial artifacts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import suite_runner as runner
from . import checkpoint_retention as retention
from .experiment_suite import checkpoint_consumers
from .distributed_launch import single_node_torchrun, single_node_torchrun_environment

require, digest, sha, safe = runner.require, runner.digest, runner.sha, runner.safe
pin, verify_pin, read = runner.artifact, runner.verify_pin, runner.read


def save(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, 'Immutable action artifact differs: ' + str(path))
    else:
        runner.write_once(path, value)
    return pin(path)


def inventory(root):
    root = safe(root)
    require(root.is_dir(), 'Missing model directory')
    rows = []
    for path in sorted(root.rglob('*')):
        safe(path)
        if path.is_file():
            rows.append({**retention.file_pin(path), 'path': path.relative_to(root).as_posix()})
    require(rows, 'Empty checkpoint inventory')
    return rows


def verify_inventory(root, rows, *, exact=True):
    root = safe(root)
    require(rows and len({r['path'] for r in rows}) == len(rows), 'Duplicate/empty file inventory')
    for row in rows:
        relative = retention.relative(row['path'])
        retention.check_pin({**row, 'path': str(root / relative)})
    if exact:
        require(inventory(root) == rows, 'Checkpoint inventory differs')
    return digest(rows)


def finite_metrics(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_metrics(item)
    elif isinstance(value, list):
        for item in value:
            finite_metrics(item)
    elif isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite measured training/evaluation metric')


def load_operation(root, name):
    path = safe(root) / 'scripts/operations' / (name + '.py')
    spec = importlib.util.spec_from_file_location('suite_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def allocation_contract(job):
    """Verify actual scheduler spec, rather than trusting launch CLI flags."""
    replicas = job['spec']['pytorchReplicaSpecs']
    require(set(replicas) == {'Master'}, 'Exactly one Master node is required')
    master = replicas['Master']
    spec = master['template']['spec']
    require(master.get('replicas') == 1 and master.get('restartPolicy') == 'Never', 'Wrong replica/restart contract')
    require(spec.get('nodeSelector', {}).get('sslaunch-pool') == 'stream'
            and spec.get('priorityClassName') == 'low' and not spec.get('nodeName'), 'Wrong stream/low contract')
    require(not spec.get('initContainers') and len(spec.get('containers', [])) == 1,
            'Unexpected execution containers')
    resources = spec['containers'][0].get('resources', {})
    require(int(resources.get('limits', {}).get('nvidia.com/gpu', 0)) == 8
            and int(resources.get('requests', {}).get('nvidia.com/gpu', 8)) == 8, 'Eight GPU allocation required')
    return dict(runner.ALLOCATION)


def terminal_job(job):
    status = job.get('status', {})
    true_conditions = {c.get('type') for c in status.get('conditions', []) if c.get('status') == 'True'}
    terminal = true_conditions & {'Succeeded', 'Failed'}
    # A contradictory/terminating object is not proof that another allocation
    # is inactive. Pending jobs without a terminal condition remain active too.
    return (len(terminal) == 1 and true_conditions <= {'Created', 'Succeeded', 'Failed'}
            and not any(c.get('status') not in {'True', 'False'} for c in status.get('conditions', []))
            and not job.get('metadata', {}).get('deletionTimestamp')
            and all(value.get('active', 0) == 0 for value in status.get('replicaStatuses', {}).values()))


def validate_training_output(output, task, training, frozen_source_sha):
    output = safe(output)
    complete = read(output / 'FORMAL_TRAIN_COMPLETE.json')
    expected = {'status': 'formal_training_complete', 'stage': task['stage'], 'world_size': 8,
                'optimizer_step': task['expected_optimizer_steps'],
                'train_manifest_sha256': training['train_manifest']['sha256'],
                'eval_manifest_sha256': training['eval_manifest']['sha256'],
                'formal_ready_sha256': training['formal_ready']['sha256'],
                'frozen_source_manifest_sha256': frozen_source_sha}
    require(all(complete.get(k) == v for k, v in expected.items()), 'Formal completion differs from task/data/source')
    require(complete.get('media_before') == complete.get('media_after') and complete.get('media_before'),
            'Training media changed or was not checked')
    outputs = [pin(output / name) for name in ('FORMAL_TRAIN_COMPLETE.json', 'run_manifest.json',
                'trainer_state.json', 'train_results.json', 'eval_results.json')]
    require(sha(output / 'trainer_state.json') == complete['trainer_state_sha256']
            and sha(output / 'run_manifest.json') == complete['run_manifest_sha256'], 'Formal evidence changed')
    require(read(output / 'trainer_state.json')['global_step'] == task['expected_optimizer_steps'], 'Wrong final optimizer step')
    for name in ('train_results.json', 'eval_results.json'):
        finite_metrics(read(output / name))
    final_files = inventory(output / 'final')
    from .resume_checkpoint import verify_resume_checkpoint
    verify_resume_checkpoint(output / ('checkpoint-' + str(task['expected_optimizer_steps'])),
                             world_size=8, expected_max_steps=task['expected_optimizer_steps'])
    return outputs, final_files


def latest_complete_checkpoint(output, *, expected_steps=718):
    """Choose the newest structurally complete owned save, preserving torn saves."""
    from .resume_checkpoint import verify_resume_checkpoint
    import re
    candidates = sorted((p for p in Path(output).glob('checkpoint-*')
                         if re.fullmatch(r'checkpoint-[1-9][0-9]*', p.name)),
                        key=lambda p: int(p.name.split('-')[-1]), reverse=True)
    for candidate in candidates:
        try:
            verify_resume_checkpoint(candidate, world_size=8, expected_max_steps=expected_steps)
            rows = inventory(candidate)
            require(any(r['path'].endswith('.safetensors') for r in rows), 'Recovery model weights missing')
            return candidate
        except (ValueError, FileNotFoundError):
            continue
    return None


def validate_benchmark(data, task):
    """Require the actual frame count; never let load_frames repeat an F8 cache."""
    from .manifest import load_manifest
    for key in ('manifest', 'acceptance'):
        verify_pin(data[key])
    records = load_manifest(data['manifest']['path'], media_root=data['media_root'])
    count = 2697 if task['benchmark'] == 'videomme899' else 1580
    require(len(records) == count and len({r['sample_id'] for r in records}) == count, 'Incomplete benchmark QA set')
    frames = task['frames']
    require(all(len(r.get('image_paths', [])) == frames for r in records), 'Frame cache shape differs from requested original-video sampling')
    names = sorted({p for r in records for p in r['image_paths']})
    files = data['media_files']
    require(sorted(p['path'] for p in files) == names and len(files) == len(names), 'Incomplete frame inventory')
    for item in files:
        retention.relative(item['path'])
        retention.check_pin({**item, 'path': str(safe(data['media_root']) / item['path'])})
    acceptance = read(data['acceptance']['path'])
    require(acceptance.get('status') == 'SUITE_BENCHMARK_READY'
            and acceptance.get('manifest_sha256') == data['manifest']['sha256']
            and acceptance.get('num_frames') == frames
            and acceptance.get('media_inventory_sha256') == digest(files)
            and acceptance.get('sampling') == 'uniform_indices_from_original_video'
            and acceptance.get('qa') == count, 'Benchmark acceptance does not bind actual frame protocol')
    require(acceptance.get('evidence'), 'Original-source media acceptance evidence missing')
    for item in acceptance['evidence']:
        verify_pin(item)
    if task['benchmark'] == 'videomme899':
        from .evaluate import validate_videomme_subset
        validate_videomme_subset(records)
    else:
        require(len({r['video_id'] for r in records}) == 410, 'TempCompass clip coverage differs')
        require(all(r.get('original_answer') and r.get('underlying_source_video_ids') for r in records),
                'Missing official answers or connected-source identities')
    return records


def merge_predictions(output, records, *, manifest_sha, provenance_sha, engine_protocol, config):
    from . import evaluate
    merged = {}
    for rank in range(8):
        path = evaluate.shard_output(Path(output), rank, 8)
        protocol = read(path.with_suffix('.protocol.json'))
        expected = {'manifest_sha256': manifest_sha, 'run_provenance_sha256': provenance_sha,
                    'protocol': engine_protocol, 'checkpoint_config': config,
                    'num_shards': 8, 'shard_index': rank, 'max_new_tokens': 16,
                    'cache': True, 'decoding': 'greedy_argmax', 'subtitles': False, 'audio': False}
        require(all(protocol.get(k) == v for k, v in expected.items()), 'Shard protocol mismatch')
        assigned = evaluate.shard_records(records, rank, 8)
        rows = evaluate.read_prediction_rows(path, assigned, digest(protocol))
        summary = read(path.with_suffix('.summary.json'))
        require(len(rows) == len(assigned) and summary.get('num_examples') == len(rows)
                and summary.get('predictions_sha256') == sha(path)
                and summary.get('protocol_sha256') == digest(protocol), 'Incomplete or changed prediction shard')
        for row in rows:
            require(row['sample_id'] not in merged, 'Duplicate prediction')
            merged[row['sample_id']] = row
    require(set(merged) == {r['sample_id'] for r in records}, 'Incomplete prediction coverage')
    return [merged[r['sample_id']] for r in records]


class Actions:
    def __init__(self, plan, environment, state_root, *, resume=False):
        self.plan, self.env = plan, environment
        self.state = safe(state_root)
        self.root = safe(environment['experiment_root'])
        self.run_root = safe(environment['run_root'])
        self.tasks = {t['id']: t for t in plan['tasks']}
        self.resume = resume
        self.source_sha = runner.source_identity({'source_files': environment['source_files']})

    def task_dir(self, task_id):
        require(task_id in self.tasks or task_id == 'startup', 'Unknown task')
        path = self.state / 'artifacts' / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def completed(self, task_id):
        receipt = runner.completion_for(self.state, task_id)['receipts'][0]
        value = read(receipt['path'])
        require(value.get('status') == 'SUITE_TASK_ACCEPTED' and value.get('task_id') == task_id
                and value.get('task_sha256') == digest(self.tasks[task_id]), 'Invalid task acceptance')
        for item in value['outputs']:
            verify_pin(item)
        return value

    def budget(self, extra):
        return runner.capacity({'storage': self.env['storage']}, extra)

    def command(self, argv, directory, label, *, source_root=None, extra_env=None):
        directory = safe(directory)
        directory.mkdir(parents=True, exist_ok=True)
        logs = directory / (label + '-' + os.urandom(6).hex())
        env = os.environ.copy()
        env.update(self.env['runtime_environment'])
        env.update(runner.OFFLINE)
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [str((source_root or self.root) / 'src'),
                                                       self.env['runtime_environment'].get('PYTHONPATH', '')]))
        if extra_env:
            env.update(extra_env)
        if list(argv[1:4]) == ['-B', '-m', 'torch.distributed.run']:
            env = single_node_torchrun_environment(env)
        save(logs.with_suffix('.intent.json'), {'argv': list(map(str, argv)), 'source_root': str(source_root or self.root)})
        with logs.with_suffix('.stdout').open('xb') as out, logs.with_suffix('.stderr').open('xb') as err:
            # Inherit runner process group so timeout/SIGTERM covers all ranks.
            result = subprocess.run(list(map(str, argv)), cwd=source_root or self.root, env=env, stdout=out, stderr=err)
        save(logs.with_suffix('.exit.json'), {'returncode': result.returncode})
        require(result.returncode == 0, 'Task subprocess failed; retained logs at ' + str(logs))

    def live(self, *, require_startup=True):
        tools, identity = self.env['tools'], self.env['allocation_identity']
        argv = [tools['kubectl'], '--kubeconfig', tools['kubeconfig'], '--request-timeout=30s',
                'get', 'pytorchjobs,pods', '-A', '-o', 'json']
        result = subprocess.run(argv, capture_output=True, timeout=40)
        require(result.returncode == 0, 'Cannot verify exact allocation and project jobs')
        items = json.loads(result.stdout)['items']
        jobs = [x for x in items if x['kind'] == 'PyTorchJob']
        target = [x for x in jobs if x['metadata']['name'] == identity['job_name']
                  and x['metadata']['namespace'] == identity['namespace']]
        require(len(target) == 1 and not terminal_job(target[0]), 'Exact live suite job not found')
        job = target[0]
        allocation_contract(job)
        prefixes = identity['project_name_prefixes']
        require(prefixes and all(p.startswith('jinhaopeng-dp') for p in prefixes), 'Project job scope missing')
        others = [x['metadata']['name'] for x in jobs if any(x['metadata']['name'].startswith(p) for p in prefixes)
                  and x['metadata']['uid'] != job['metadata']['uid'] and not terminal_job(x)]
        pods = [x for x in items if x['kind'] == 'Pod']
        owned = [x for x in pods if any(r.get('uid') == job['metadata']['uid'] for r in x['metadata'].get('ownerReferences', []))]
        require(len(owned) == 1 and owned[0].get('status', {}).get('phase') == 'Running', 'Suite requires one live owned pod')
        others.extend(x['metadata']['name'] for x in pods if any(x['metadata']['name'].startswith(p) for p in prefixes)
                      and x not in owned and x.get('status', {}).get('phase') not in {'Failed', 'Succeeded'})
        require(not others, 'Another project job or orphan pod is active')
        if require_startup:
            invocations = sorted((self.state / 'invocations').glob('invocation-*'))
            require(invocations, 'No current allocation invocation')
            started = read(invocations[-1] / 'STARTUP.json')
            require(started.get('status') == 'SUITE_STARTUP_ACCEPTED'
                    and started.get('source_identity_sha256') == self.source_sha
                    and started.get('job_uid') == job['metadata']['uid']
                    and started.get('pod_uid') == owned[0]['metadata']['uid'],
                    'Allocation UID changed since this invocation startup')
        return {'job_uid': job['metadata']['uid'], 'pod_uid': owned[0]['metadata']['uid'],
                'node': owned[0]['spec']['nodeName'], 'other_active_project_jobs': others}

    def startup(self, output):
        for item in self.env['source_files']:
            verify_pin(item)
        live = self.live(require_startup=False)
        from .suite_bindings import validate_environment
        validate_environment(self.env, self.plan)
        self.budget(0)
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
        uuids = gpu.stdout.strip().splitlines()
        require(gpu.returncode == 0 and len(uuids) == len(set(uuids)) == 8, 'Eight actual GPU UUIDs required')
        hardware = safe(output).parent / 'HARDWARE.json'
        self.command(single_node_torchrun(self.env['python']) +
                     [self.root / 'scripts/preflight.py', '--output', hardware],
                     safe(output).parent, 'hardware')
        report = read(hardware)
        require(report.get('status') == 'gpu_gate_passed' and report.get('world_size') == 8
                and report.get('distributed_all_reduce') == 'pass' and len(report.get('gpus', [])) == 8, 'Hardware gate failed')
        save(output, {'status': 'SUITE_STARTUP_ACCEPTED', 'allocation': dict(runner.ALLOCATION),
                      'source_identity_sha256': self.source_sha, **live, 'gpu_count': 8, 'gpu_uuids': uuids,
                      'hardware_preflight_passed': True, 'hardware': pin(hardware)})

    def owner(self, task_id):
        run_id = self.env['suite_id'] + '--' + task_id
        root = self.run_root / run_id
        if (root / '.suite-owner.json').exists():
            owner = read(root / '.suite-owner.json')
            require(owner['suite_id'] == self.env['suite_id'] and owner['source_sha256'] == self.source_sha
                    and owner['run_id'] == run_id and owner['run_root'] == str(root)
                    and owner['runs_root'] == str(self.run_root),
                    'Existing run belongs to another suite/source')
            retention.check_owner(owner)
            return owner
        return retention.claim_owned_run(self.run_root, run_id, self.env['suite_id'], self.source_sha)

    def guard(self, owner):
        self.live()
        root = str(retention.check_owner(owner))
        writers = []
        proc = Path('/proc')
        require(proc.is_dir(), 'Live Linux process verification required')
        for item in proc.iterdir():
            if not item.name.isdigit() or int(item.name) == os.getpid():
                continue
            try:
                args = (item / 'cmdline').read_bytes().split(b'\0')
                if any(root.encode() in arg for arg in args) and any(
                        token in b' '.join(args) for token in (b'deepprobe_vlm.train', b'torch.distributed.run')):
                    writers.append(int(item.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
            except PermissionError:
                # Processes of other users cannot write this user's owned run.
                require(item.stat().st_uid != os.getuid(), 'Cannot inspect an owned process')
        return {'run_id': owner['run_id'], 'source_sha256': owner['source_sha256'], 'active_writers': writers}

    def store(self, owner):
        from .checkpoint_store import OssutilStore
        prefix = self.env['archive_root'].rstrip('/') + '/runs/' + owner['run_id'] + '/'
        return OssutilStore(self.env['tools']['ossutil'], self.env['tools']['oss_config'],
                            allowed_prefix=prefix, state_dir=Path(owner['run_root']) / '.retention' / 'oss'), prefix

    def history(self, task, owner, *, restore=True):
        """Copy only explicitly audited historical inputs; never adopt/delete old runs."""
        value = self.env['history'].get(task['id'])
        require(value and value['mode'] in {'accepted', 'resume'}, 'Historical candidate needs audited binding; no silent fresh restart')
        evidence = read(verify_pin(value['acceptance'])['path'])
        for item in evidence.get('evidence', []):
            verify_pin(item)
        expected = {'status': 'SUITE_HISTORY_ACCEPTED', 'task_id': task['id'], 'mode': value['mode'],
                    'config_sha256': value['config']['sha256'],
                    'frozen_source_manifest_sha256': value['frozen_source_manifest']['sha256'],
                    'checkpoint_inventory_sha256': digest(value['checkpoint']['files'])}
        require(evidence.get('evidence') and all(evidence.get(k) == v for k, v in expected.items()), 'Historical audit binding incomplete')
        verify_pin(value['config']); verify_pin(value['frozen_source_manifest'])
        if value['mode'] == 'resume':
            require(evidence.get('exact_resume_accepted') is True, 'Historical numerical resume audit not accepted')
        require(read(value['config']['path']) == read(self.root / task['config']['path']), 'Historical recipe differs from scientific declaration')
        src = safe(value['checkpoint']['path'])
        files = value['checkpoint']['files']
        if restore and not value['checkpoint'].get('oss_prefix'):
            verify_inventory(src, files)
        step = task['historical_hint'].get('checkpoint_step')
        if restore and value['mode'] == 'resume' and not value['checkpoint'].get('oss_prefix'):
            from .resume_checkpoint import verify_resume_checkpoint
            audit = verify_resume_checkpoint(src, world_size=8, expected_max_steps=task['expected_optimizer_steps'])
            require(audit['optimizer_step'] == step, 'Historical resume step differs')
        target = Path(owner['run_root']) / 'training' / ('final' if value['mode'] == 'accepted' else 'checkpoint-' + str(step))
        if not restore:
            return value, target
        if value['checkpoint'].get('oss_prefix'):
            prefix = value['checkpoint']['oss_prefix']
            require(evidence.get('checkpoint_oss_prefix') == prefix, 'Historical archive prefix differs')
            self.restore_files(prefix, target, files, Path(owner['run_root']) / '.retention' / 'restore')
        else:
            self.copy_files(src, target, files)
        if value['mode'] == 'resume':
            from .resume_checkpoint import verify_resume_checkpoint
            audit = verify_resume_checkpoint(target, world_size=8, expected_max_steps=718)
            require(audit['optimizer_step'] == step, 'Restored historical resume step differs')
        return value, target

    def restore_files(self, prefix, target, files, state_dir):
        """Stream only the declared payload from the stream bucket to node cache."""
        from .checkpoint_store import OssutilStore
        store = OssutilStore(self.env['tools']['ossutil'], self.env['tools']['oss_config'],
                             allowed_prefix=prefix, state_dir=state_dir)
        require(prefix.endswith('/') and '/runs/' in prefix, 'Exact historical run archive prefix required')
        self.budget(sum(r['bytes'] for r in files if not (target / r['path']).exists()))
        for row in files:
            relative = retention.relative(row['path'])
            dest = safe(target / relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                retention.check_pin({**row, 'path': str(dest)})
                continue
            temp = dest.with_name(dest.name + '.suite-restore')
            if temp.exists():
                require(self.resume, 'Interrupted owned restore requires explicit resume')
                temp.unlink()
            received = 0
            with temp.open('xb') as stream:
                for chunk in store.iter_object(prefix + relative.as_posix()):
                    received += len(chunk)
                    require(received <= row['bytes'], 'Archive object exceeds declared size')
                    stream.write(chunk)
                stream.flush(); os.fsync(stream.fileno())
            retention.check_pin({**row, 'path': str(temp)})
            os.link(temp, dest); temp.unlink()
        verify_inventory(target, files)

    def copy_files(self, src, target, files):
        self.budget(sum(r['bytes'] for r in files if not (target / r['path']).exists()))
        for row in files:
            relative = retention.relative(row['path'])
            dest = safe(target / relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                retention.check_pin({**row, 'path': str(dest)})
                continue
            temp = dest.with_name(dest.name + '.suite-copy')
            if temp.exists():
                # Only this known owned scratch file is recoverable; no silent
                # unverified partial checkpoint is ever used.
                require(self.resume, 'Partial owned copy requires explicit resume')
                temp.unlink()
            with safe(src / relative).open('rb') as source, temp.open('xb') as out:
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
                out.flush(); os.fsync(out.fileno())
            retention.check_pin({**row, 'path': str(temp)})
            os.link(temp, dest); temp.unlink()
        verify_inventory(target, files)

    def checkpoint(self, stage_id):
        accepted = self.completed(stage_id)
        root = safe(accepted['checkpoint'])
        verify_inventory(root, accepted['checkpoint_files'])
        return root, accepted

    def requirements(self, stage_id):
        out = []
        for cid in checkpoint_consumers(self.plan, stage_id):
            task = self.tasks[cid]
            kind = {'train': 'training', 'evaluate': 'evaluation', 'latency': 'latency'}[task['kind']]
            units = 718 if kind == 'training' else (32 * 13 if kind == 'latency' else
                     2697 if task['benchmark'] == 'videomme899' else 1580)
            out.append({'id': cid, 'kind': kind, 'checkpoint': 'training/final',
                        'protocol_sha256': digest(task), 'expected_units': units})
        return out

    def consume(self, stage_id, task, outputs):
        parent = self.completed(stage_id)
        rows = [{**r, 'path': 'training/final/' + r['path']} for r in parent['checkpoint_files']]
        requirement = next(r for r in self.requirements(stage_id) if r['id'] == task['id'])
        result = {'schema_version': 1, 'status': 'SUITE_CONSUMER_COMPLETE', 'consumer_id': task['id'],
                  'kind': requirement['kind'], 'source_sha256': self.source_sha,
                  'checkpoint_inventory_sha256': digest(rows), 'protocol_sha256': digest(task),
                  'completed_units': requirement['expected_units'], 'failed_units': 0,
                  'outputs': outputs, 'output_inventory_sha256': digest(outputs)}
        return save(self.task_dir(task['id']) / ('CONSUMER--' + stage_id + '.json'), result)

    def training_storage(self, task, output):
        """Compute FP32 weights/Adam rotation peak from real safetensor headers."""
        model = next(m for m in self.plan['models'] if m['id'] == task['model_id'])
        assets = self.env['assets']['backbones'][model['family']]
        require(assets.get('files'), 'Pinned offline backbone inventory required')
        for item in assets['files']:
            verify_pin(item)
        config = read(self.root / task['config']['path'])
        config['model']['language_model'] = assets['language_root']
        config['model']['vision_model'] = assets['vision_root']
        estimator = load_operation(self.root, 'run_p1_stage_v9')
        # Only the pure header estimator is called; its runner/cleanup is never invoked.
        estimate = estimator.estimate_stage_storage(config, task['stage'])
        resident = sum(r['bytes'] for r in inventory(output)) if output.is_dir() and any(output.iterdir()) else 0
        additional = max(0, estimate['required_additional_bytes'] - resident)
        observed = self.budget(additional)
        return {'estimate': estimate, 'owned_stage_resident_bytes': resident, 'capacity': observed}

    def train(self, task):
        owner = self.owner(task['id'])
        output = Path(owner['run_root']) / 'training'
        training = self.env['assets']['training']
        for name in ('train_manifest', 'eval_manifest', 'formal_ready', 'frozen_source_manifest'):
            verify_pin(training[name])
        config = self.root / task['config']['path']
        require(sha(config) == task['config']['sha256'], 'Recipe changed')
        source, frozen, resume_checkpoint = self.root, training['frozen_source_manifest'], None
        history = None
        owned_recovery = latest_complete_checkpoint(output) if self.resume else None
        if task['historical_hint']['status'] in {'resume_candidate', 'reuse_candidate'}:
            history, historical_checkpoint = self.history(task, owner, restore=owned_recovery is None)
            if history['mode'] == 'accepted':
                files = inventory(historical_checkpoint)
                audit = read(history['acceptance']['path'])
                require(audit.get('optimizer_step') == 718 and audit.get('formal_training_complete') is True,
                        'Historical success acceptance missing')
                outputs = [history['acceptance']]
            else:
                resume_checkpoint, source = owned_recovery or historical_checkpoint, safe(history['source_root'])
                config, frozen = Path(history['config']['path']), history['frozen_source_manifest']
        elif self.resume and output.exists() and any(output.iterdir()):
            require(owned_recovery, 'Interrupted training has no complete checkpoint; manual recovery required')
            resume_checkpoint = owned_recovery
        initializer = task['initializer'].get('checkpoint_task_id')
        if initializer and history and history['mode'] == 'accepted':
            ancestor = self.completed(initializer)
            audit = read(history['acceptance']['path'])
            require(audit.get('initializer_checkpoint_inventory_sha256') == digest(ancestor['checkpoint_files']),
                    'Historical SFT acceptance does not bind the matching alignment initializer')
        init_checkpoint = self.checkpoint(initializer)[0] if initializer and not resume_checkpoint else None
        if history is None or history['mode'] != 'accepted':
            save(self.task_dir(task['id']) / ('CAPACITY-' + os.urandom(6).hex() + '.json'),
                 self.training_storage(task, output))
            from .resume_checkpoint import verify_resume_checkpoint
            if resume_checkpoint:
                verify_resume_checkpoint(resume_checkpoint, world_size=8, expected_max_steps=718)
            if not (output / 'FORMAL_TRAIN_COMPLETE.json').exists():
                argv = runner.build_training_argv(self.env['python'], config=config,
                    train_manifest=training['train_manifest']['path'], eval_manifest=training['eval_manifest']['path'],
                    media_root=training['media_root'], output_dir=output, formal_ready=training['formal_ready']['path'],
                    formal_training_data_dir=training['data_dir'], frozen_source_manifest=frozen['path'],
                    frozen_source_manifest_sha256=frozen['sha256'], declared=task['entry'] == 'deepprobe_vlm.train_declared',
                    init_checkpoint=init_checkpoint, resume_checkpoint=resume_checkpoint,
                    no_alignment=task['initializer']['kind'] == 'pretrained_backbones_random_connector' and task['stage'] == 'sft')
                self.command(argv, self.task_dir(task['id']), 'training', source_root=source)
            outputs, files = validate_training_output(output, task, training, frozen['sha256'])
        summary = {'schema_version': 1, 'status': 'SUITE_TRAINING_COMPLETE', 'run_id': owner['run_id'],
                   'source_sha256': self.source_sha,
                   'final_checkpoint_inventory_sha256': digest([{**r, 'path': 'training/final/' + r['path']} for r in files]),
                   'optimizer_steps': 718, 'target_optimizer_steps': 718, 'exit_code': 0,
                   'outputs': outputs, 'output_inventory_sha256': digest(outputs)}
        complete = save(self.task_dir(task['id']) / 'TRAINING_COMPLETE.json', summary)
        consumers = [self.consume(initializer, task, [complete])] if initializer else []
        return {'owner': owner, 'checkpoint': str(output / 'final'), 'checkpoint_files': files,
                'training_complete': complete, 'outputs': [complete], 'consumer_receipts': consumers}

    def benchmark_data(self, task):
        data = self.env['assets']['benchmarks'][task['benchmark']][str(task['frames'])]
        if 'cache' not in data:
            return data
        from .suite_media import hydrate_frame_cache, OSSStore, PROJECT_PREFIX, _bound_cache_inventory
        _, _, items, _, _, _ = _bound_cache_inventory(data['cache'])
        needed = sum(r['bytes'] for r in items if not (Path(data['media_root']) / r['path']).exists())
        self.budget(needed)
        return hydrate_frame_cache(data['cache'], media_root=data['media_root'], max_bytes=data['max_bytes'],
            store=OSSStore(self.env['tools']['oss_config'], allowed_prefix=PROJECT_PREFIX + 'suite-media-readonly/'), resume=True)

    def evaluate(self, task):
        data = self.benchmark_data(task)
        records = validate_benchmark(data, task)
        checkpoint, parent = self.checkpoint(task['checkpoint_task_id'])
        provenance_sources = read(checkpoint / 'training_provenance.json')
        observed_sources = set(provenance_sources['training_sources']) | set(provenance_sources['tuning_sources'])
        benchmark_sources = {r['source_video_id'] for r in records} | {source for r in records
                             for source in r.get('underlying_source_video_ids', [])}
        require(not observed_sources & benchmark_sources, 'Benchmark underlying source occurred in training/tuning')
        config = read(checkpoint / 'deepprobe_config.json')
        require(config['num_frames'] == task['frames'] and config['pool_grid'] == task['pool_grid'], 'Model shape differs')
        directory = self.task_dir(task['id'])
        provenance = save(directory / 'PROVENANCE.json', {'checkpoint': str(checkpoint),
            'all_checkpoint_files_sha256_verified': True, 'checkpoint_config_sha256': sha(checkpoint / 'deepprobe_config.json'),
            'suite_task_sha256': digest(task), 'parent_acceptance': parent['training_complete'],
            'data_acceptance_sha256': data['acceptance']['sha256'], 'source_identity_sha256': self.source_sha})
        engine = task['protocol'] if task['protocol'] == 'videomme-899-no-subtitles-no-audio-v1' else 'reference'
        output = directory / 'raw' / 'predictions.jsonl'
        argv = single_node_torchrun(self.env['python']) + ['-m', 'deepprobe_vlm.evaluate',
                '--checkpoint', checkpoint, '--manifest', data['manifest']['path'], '--media-root', data['media_root'],
                '--output', output, '--max-new-tokens', '16', '--protocol', engine, '--distributed-env',
                '--run-provenance', provenance['path']]
        if self.resume:
            argv.append('--resume')
        self.command(argv, directory, 'evaluate')
        verify_inventory(checkpoint, parent['checkpoint_files'])
        validate_benchmark(data, task)
        rows = merge_predictions(output, records, manifest_sha=data['manifest']['sha256'],
                                 provenance_sha=provenance['sha256'], engine_protocol=engine, config=config)
        if task['benchmark'] == 'tempcompass_mc':
            scorer = load_operation(self.root, 'score_tempcompass_mc')
            bridge = [{'sample_id': r['sample_id'], 'prediction': p['prediction'], 'manifest_row_sha256': digest(r)}
                      for r, p in zip(records, rows)]
            scored, summary = scorer.score(records, bridge)
            by_id = {r['sample_id']: r for r in scored}
            rows = [{**p, 'correct': int(by_id[p['sample_id']]['correct']), 'metric': 'tempcompass_official_disable_llm',
                     'source_video_id': r['source_video_id'], 'underlying_source_video_ids': r['underlying_source_video_ids']}
                    for r, p in zip(records, rows)]
        else:
            summary = {'scope': 'videomme899_reference_multiple_choice', 'qa': len(rows),
                       'correct': sum(r['correct'] for r in rows), 'accuracy': sum(r['correct'] for r in rows) / len(rows)}
        rows = [{**row, 'question_identity_sha256': digest({k: record.get(k) for k in
                    ('sample_id', 'source_video_id', 'question', 'choices', 'answer', 'split', 'original_answer')})}
                for record, row in zip(records, rows)]
        result = save(directory / 'EVALUATION.json', {'status': 'SUITE_EVALUATION_COMPLETE',
                      'suite_protocol': task['protocol'], 'task_sha256': digest(task), 'summary': summary,
                      'qa': len(rows), 'failed_units': 0, 'rows': rows, 'data_acceptance': data['acceptance'],
                      'sampling_scope': 'original_video_uniform_frames', 'provenance': provenance})
        return {'outputs': [result], 'evaluation': result,
                'consumer_receipts': [self.consume(task['checkpoint_task_id'], task, [result])]}

    def latency(self, task, *, emit_consumer=True):
        from . import benchmark_latency as latency
        checkpoint, parent = self.checkpoint(task['checkpoint_task_id'])
        data = self.env['assets']['development']
        verify_pin(data['manifest']); verify_pin(data['source_metadata'])
        plan = latency.freeze_plan(data['manifest']['path'], checkpoint, data['media_root'],
                                   data['source_metadata']['path'], cache=task['cache'], implementation=task['implementation'])
        directory = self.task_dir(task['id'])
        plan_pin = save(directory / 'LATENCY_PLAN.json', plan)
        # Incomplete timing trials never mix allocations; retain each attempt.
        output = directory / ('measurement-' + os.urandom(6).hex())
        argv = [self.env['python'], '-B', '-m', 'deepprobe_vlm.benchmark_latency', '--plan', plan_pin['path'],
                '--plan-sha256', plan_pin['sha256'], '--execute', '--output', output]
        if task['separate_profile']:
            argv.append('--profile')
        self.command(argv, directory, 'latency')
        from .compare_latency import read_run
        read_run(output)
        verify_inventory(checkpoint, parent['checkpoint_files'])
        files = [pin(output / name) for name in ('PLAN.json', 'SUMMARY.json', 'RUNTIME.json', 'requests.jsonl')]
        if task['separate_profile']:
            profile = read(output / 'PROFILE.json')
            require(profile.get('status') == 'DIAGNOSTIC_PROFILE_COMPLETE' and profile.get('included_in_measurements') is False,
                    'Required separate CUDA profile did not complete')
            require(profile.get('trace_sha256') == sha(output / 'profile.trace.json')
                    and profile.get('operators_sha256') == sha(output / 'profile.operators.txt'), 'CUDA profile payload changed')
            files.extend(pin(output / name) for name in ('PROFILE.json', 'profile.trace.json', 'profile.operators.txt'))
        return {'measurement_root': str(output), 'outputs': files,
                'consumer_receipts': [self.consume(task['checkpoint_task_id'], task, files)] if emit_consumer else []}

    def archive(self, task):
        if task['archive_scope'] == 'results':
            return self.archive_results(task)
        train = self.completed(task['checkpoint_task_id'])
        owner = train['owner']
        store, prefix = self.store(owner)
        guard = lambda: self.guard(owner)
        manifest_path = Path(owner['run_root']) / '.retention/MANIFEST.json'
        if manifest_path.exists():
            manifest = retention.validate_manifest(read(manifest_path))
        else:
            diagnostic_root = self.task_dir(task['checkpoint_task_id'])
            self.copy_files(diagnostic_root, Path(owner['run_root']) / 'stage_evidence', inventory(diagnostic_root))
            manifest = retention.prepare_manifest(owner, archive_prefix=prefix,
                required_consumers=self.requirements(task['checkpoint_task_id']), guard=guard,
                training_completion=train['training_complete'])
        intermediate = [cp for cp in manifest['checkpoints'] if cp != 'training/final']
        missing = [row for row in manifest['files'] if retention.cleanable(row['path'])
                   and not (Path(owner['run_root']) / row['path']).exists()]
        outputs = [pin(manifest_path)]
        if missing:
            # An interruption can occur after intermediate cleanup but before
            # the task's final receipt. The retention transaction proves every
            # missing file and freshly checks the whole remote archive below.
            require(self.resume and all(not row['path'].startswith('training/final/') for row in missing),
                    'Missing archived weights require explicit intermediate-cleanup recovery')
        else:
            archived = retention.archive_and_verify(manifest, store, guard=guard)
            outputs.append(archived['receipt'])
        if intermediate:
            cleanup = retention.plan_cleanup(manifest, {}, checkpoints=intermediate)
            result = retention.cleanup(cleanup, store, guard=guard, execute=True)
            outputs.append(save(self.task_dir(task['id']) / 'INTERMEDIATE_CLEANUP.json', result))
        return {'outputs': outputs, 'manifest': pin(manifest_path), 'owner': owner}

    def archive_results(self, task):
        # Results use a distinct owned archive prefix; never modify model archives.
        owner = self.owner(task['id'])
        store, prefix = self.store(owner)
        rows = []
        for tid in task['result_task_ids']:
            result = self.completed(tid)
            for index, item in enumerate(result['outputs']):
                verify_pin(item)
                rows.append({**item, 'object': prefix + tid + '/' + str(index) + '-' + Path(item['path']).name})
        manifest = save(self.task_dir(task['id']) / 'RESULT_MANIFEST.json', {'schema_version': 1, 'files': rows})
        rows.append({**manifest, 'object': prefix + 'RESULT_MANIFEST.json'})
        expected = {r['object']: r for r in rows}
        existing = {r['uri']: r for r in store.list_prefix(prefix)}
        require(set(existing) <= set(expected), 'Unexpected result archive object')
        for uri, row in expected.items():
            if uri not in existing:
                store.put_if_absent(uri, row['path'], row['sha256'])
            actual = retention._object_digest(store, uri)
            require(actual == {'bytes': row['bytes'], 'sha256': row['sha256']}, 'Result archive readback differs')
        before = retention._listing(store, prefix)
        require({u: r['bytes'] for u, r in before.items()} == {u: r['bytes'] for u, r in expected.items()},
                'Result archive inventory differs')
        for uri, row in expected.items():
            require(retention._object_digest(store, uri) == {'bytes': row['bytes'], 'sha256': row['sha256']},
                    'Result archive changed during final verification')
        require(before == retention._listing(store, prefix), 'Result archive inventory changed during full readback')
        receipt = save(self.task_dir(task['id']) / 'RESULT_ARCHIVE_VERIFIED.json',
                       {'status': 'SUITE_RESULTS_ARCHIVED_FULL_GET_VERIFIED', 'files': rows, 'manifest': manifest})
        return {'outputs': [receipt], 'result_tasks': task['result_task_ids']}

    def cleanup(self, task):
        stage = task['checkpoint_task_id']
        arc = self.completed(stage.replace('train__', 'archive__', 1))
        manifest = read(arc['manifest']['path'])
        consumers = {}
        for cid in task['required_consumers']:
            value = self.completed(cid)
            match = [p for p in value.get('consumer_receipts', []) if read(p['path']).get('checkpoint_inventory_sha256')
                     == manifest['checkpoints']['training/final'] and read(p['path']).get('consumer_id') == cid]
            require(len(match) == 1, 'Consumer does not attest this exact checkpoint')
            consumers[cid] = match[0]
        store, _ = self.store(manifest['owner'])
        result = retention.cleanup(retention.plan_cleanup(manifest, consumers), store,
                                   guard=lambda: self.guard(manifest['owner']), execute=True)
        return {'outputs': [save(self.task_dir(task['id']) / 'CLEANUP.json', result)]}

    def run(self, task_id, output):
        if task_id == 'startup':
            return self.startup(output)
        task = self.tasks[task_id]
        for dependency in task['dependencies']:
            self.completed(dependency)
        for source in self.env['source_files']:
            verify_pin(source)
        self.live()
        kind = task['kind']
        if kind == 'gate':
            directory = self.task_dir(task_id)
            outputs = []
            for dtype in ('bfloat16', 'float32'):
                dest = directory / (dtype + '-' + os.urandom(6).hex())
                self.command([self.env['python'], '-B', '-m', 'deepprobe_vlm.benchmark_cuda_pool',
                              '--dtype', dtype, '--execute', '--output', dest], directory, dtype)
                summary = read(dest / 'SUMMARY.json')
                require(summary.get('status') == 'CUDA_POOL_MICROBENCHMARK_COMPLETE', 'CUDA microbenchmark incomplete')
                outputs.append(pin(dest / 'SUMMARY.json'))
            result = {'outputs': outputs}
        elif kind == 'compare_latency':
            from .compare_latency import compare, read_run, HARDWARE_AND_SOFTWARE
            reference = self.completed(task['reference_task_id'])['measurement_root']
            candidate = self.completed(task['candidate_task_id'])['measurement_root']
            ref_runtime, new_runtime = read_run(reference)[1], read_run(candidate)[1]
            extra_outputs = []
            if any(ref_runtime[k] != new_runtime[k] for k in HARDWARE_AND_SOFTWARE):
                # Cross-allocation timings cannot be compared. Repeat the exact
                # declared pair on this allocation, preserving earlier trials.
                # Token-parity failure below is never retried to seek a pass.
                ref = self.latency(self.tasks[task['reference_task_id']], emit_consumer=False)
                new = self.latency(self.tasks[task['candidate_task_id']], emit_consumer=False)
                reference, candidate = ref['measurement_root'], new['measurement_root']
                extra_outputs = ref['outputs'] + new['outputs']
            value = compare(reference, candidate)
            result_pin = save(self.task_dir(task_id) / 'COMPARISON.json', value)
            require(value['status'] == 'TOKEN_PARITY_PASSED', 'CUDA/cache output parity failed; weights retained')
            result = {'outputs': [result_pin] + extra_outputs,
                      'reference_measurement_root': reference, 'candidate_measurement_root': candidate,
                      'cross_allocation_pair_remeasured': bool(extra_outputs)}
        elif kind == 'aggregate':
            from .suite_results import aggregate
            evaluations = {t['id']: read(self.completed(t['id'])['evaluation']['path'])
                           for t in self.plan['tasks'] if t['kind'] == 'evaluate'}
            result = {'outputs': [save(self.task_dir(task_id) / 'RESULTS.json', aggregate(self.plan, evaluations))]}
        else:
            result = getattr(self, kind)(task)
        for item in result['outputs']:
            verify_pin(item)
        save(output, {'schema_version': 1, 'status': 'SUITE_TASK_ACCEPTED', 'task_id': task_id,
                      'task_sha256': digest(task), **result})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'environment', 'state-root', 'output-receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--environment-sha256', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    require(sha(args.plan) == args.plan_sha256 and sha(args.environment) == args.environment_sha256, 'Pinned execution input changed')
    from .experiment_suite import validate_plan
    environment = read(args.environment)
    from .suite_bindings import validate_environment
    plan = validate_plan(read(args.plan), config_root=environment['experiment_root'])
    validate_environment(environment, plan)
    Actions(plan, environment, args.state_root, resume=args.resume).run(args.task, args.output_receipt)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
