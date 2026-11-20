"""Synthetic savepoint payloads; no real cluster, model, bucket or training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import ast
import threading
import struct
import math
import os

import pytest

from deepprobe_vlm import checkpoint_retention as retention, suite_runner as runner
from deepprobe_vlm import suite_checkpoint_export as export


class Store:
    def __init__(self):
        self.objects, self.puts, self.gets = {}, [], []
        self.corrupt = None

    def list_prefix(self, prefix):
        return [{'uri': key, 'bytes': len(value), 'etag': hashlib.sha256(value).hexdigest()}
                for key, value in sorted(self.objects.items()) if key.startswith(prefix)]

    def put_if_absent(self, uri, source, sha):
        assert uri not in self.objects
        data = Path(source).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha
        self.objects[uri] = data
        self.puts.append(uri)

    def iter_object(self, uri):
        self.gets.append(uri)
        yield self.objects[uri] + (b'changed' if self.corrupt and uri.endswith(self.corrupt) else b'')


def safetensors(path, shapes):
    offset, header = 0, {}
    for name, shape in shapes.items():
        size = math.prod(shape) * 4
        header[name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [offset, offset + size]}
        offset += size
    data = json.dumps(header).encode()
    data += b' ' * (-len(data) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack('<Q', len(data)) + data + bytes(offset))


@pytest.fixture
def checkpoint(tmp_path):
    tmp_path = tmp_path.resolve()
    runs = tmp_path / 'runs'; runs.mkdir()
    owner = retention.claim_owned_run(runs, 'synthetic-run', 'synthetic-suite', 'a' * 64)
    path = Path(owner['run_root']) / 'training/checkpoint-17'; path.mkdir(parents=True)
    source = tmp_path / 'SOURCE.json'; source.write_text('{}')
    source_pin = runner.artifact(source)
    values = {'trainer_state.json': json.dumps({'global_step': 17, 'max_steps': 718}),
              'deepprobe_config.json': '{}', 'training_args.bin': 'synthetic-args',
              'training_provenance.json': json.dumps({'training_contract': {'frozen_source_manifest_sha256': source_pin['sha256']}}),
              'language_model/config.json': '{"hidden_size":2}', 'vision_model/config.json': '{"hidden_size":2}',
              'tokenizer/tokenizer_config.json': json.dumps({'tokenizer_class': 'PreTrainedTokenizerFast',
                  'unk_token': '[UNK]', 'model_max_length': 512}),
              'tokenizer/tokenizer.json': json.dumps({'version': '1.0', 'truncation': None, 'padding': None,
                  'added_tokens': [], 'normalizer': None, 'pre_tokenizer': {'type': 'Whitespace'},
                  'post_processor': None, 'decoder': None,
                  'model': {'type': 'WordLevel', 'vocab': {'[UNK]': 0, 'test': 1}, 'unk_token': '[UNK]'}}),
              'image_processor/preprocessor_config.json': '{"image_processor_type":"CLIPImageProcessor"}',
              'optimizer.pt': 'synthetic-optimizer',
              'scheduler.pt': 'synthetic-scheduler', **{f'rng_state_{i}.pth': str(i) for i in range(8)}}
    for name, value in values.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(value)
    for component in ('language_model', 'vision_model'):
        safetensors(path / component / 'model.safetensors', {'test.weight': [2, 2]})
    safetensors(path / 'projector.safetensors', {'gate.weight': [1, 2], 'gate.bias': [1],
        'projection.0.weight': [2, 2], 'projection.0.bias': [2], 'projection.2.weight': [2, 2], 'projection.2.bias': [2]})
    config = {'schema_version': 1, 'owner': owner, 'task_id': 'train__synthetic__sft', 'task_sha256': 'b' * 64,
              'expected_steps': 718, 'world_size': 8, 'frozen_source_manifest': source_pin,
              'archive_root': 'oss://ss-devbox-stream/synthetic', 'state_root': str(tmp_path / 'state'),
              'tools': {}, 'part_workers': 4}
    return config, path


def test_ready_is_last_and_every_payload_is_read_back_and_repeated_export_is_immutable(checkpoint):
    config, path = checkpoint
    store = Store()
    result = export.export_checkpoint(config, path, 17, store=store)
    assert store.puts[-1].endswith('/READY.json')
    assert '/savepoints/runs/synthetic-run/' in store.puts[-1]
    assert all(uri in store.gets for uri in store.puts)
    accepted = runner.read(result['path'])
    assert accepted['optimizer_step'] == 17
    before = deepcopy(store.objects)
    assert export.export_checkpoint(config, path, 17, store=store) == result
    assert store.objects == before
    ready = runner.read(accepted['ready']['path'])
    assert ready['control_backend'] == 'gloo' and ready['control_timeout_seconds'] == 21600


def test_corrupt_remote_payload_never_publishes_ready(checkpoint):
    config, path = checkpoint
    store = Store(); store.corrupt = 'optimizer.pt'
    with pytest.raises(ValueError, match='readback differs'):
        export.export_checkpoint(config, path, 17, store=store)
    assert not any(uri.endswith('/READY.json') for uri in store.objects)


@pytest.fixture
def post_exit_checkpoint(checkpoint, monkeypatch):
    from deepprobe_vlm import suite_legacy_private_resume as legacy, suite_training_process
    from deepprobe_vlm.suite_actions import save
    config, old = checkpoint
    checkpoint718 = old.with_name('checkpoint-718'); old.rename(checkpoint718)
    (checkpoint718 / 'trainer_state.json').write_text(json.dumps({'global_step': 718, 'max_steps': 718}))
    source = checkpoint718.parent.parent / 'legacy-source'; source.mkdir()
    monkeypatch.setattr(legacy, 'LEGACY_SOURCE_ROOT', str(source))
    monkeypatch.setattr(legacy, 'LEGACY_FROZEN_SHA', config['frozen_source_manifest']['sha256'])
    config['task_id'] = legacy.TASK_ID
    old_config = save(source / 'config.json', {'synthetic': 'original-legacy-config'})
    formal = save(checkpoint718.parent / 'FORMAL_TRAIN_COMPLETE.json', {
        'status': 'formal_training_complete', 'optimizer_step': 718, 'world_size': 8,
        'frozen_source_manifest_sha256': config['frozen_source_manifest']['sha256']})
    review = save(source / 'review.json', {'status': legacy.REVIEW_STATUS, 'approved': True,
        'production_evidence': True, 'task_id': config['task_id'], 'task_sha256': config['task_sha256'],
        'verified_facts': {'legacy_frozen_source_manifest': config['frozen_source_manifest'],
            'legacy_source_root': str(source), 'optimizer_step': 600, 'expected_optimizer_steps': 718,
            'legacy_config': old_config}})
    argv = [sys.executable, '-c', 'pass', '--output-dir', str(checkpoint718.parent),
            '--resume', str(checkpoint718.parent / 'checkpoint-600'), '--config', old_config['path']]
    # This is a real short CPU process tree, not an asserted mock exit code.
    exited = suite_training_process.run_training(argv, cwd=source, env=dict(os.environ),
        logs_prefix=source / 'logs' / 'synthetic-cpu-command')
    proof = save(Path(config['state_root']) / 'POST_TRAINING_EXPORT_INPUT.json', {
        'schema_version': 1, 'status': 'LEGACY_NATIVE_POST_TRAINING_EXPORT_INPUT',
        'task_id': config['task_id'], 'task_sha256': config['task_sha256'], 'optimizer_step': 718,
        'source_root': str(source), 'frozen_source_manifest': config['frozen_source_manifest'],
        'native_checkpoint': str(checkpoint718), 'training_exit': exited,
        'formal_completion': formal, 'legacy_review': review})
    return config, checkpoint718, proof


def test_postexit_native_export_truthfully_records_single_process_control(post_exit_checkpoint):
    config, checkpoint718, proof = post_exit_checkpoint
    result = export.export_checkpoint(config, checkpoint718, 718, store=Store(), post_training_exit=proof)
    accepted = runner.read(result['path']); ready = runner.read(accepted['ready']['path'])
    assert ready['control_backend'] == 'single_process_after_training_exit0'
    assert ready['control_timeout_seconds'] is None
    assert ready['post_training_exit_proof'] == accepted['post_training_exit_proof'] == proof
    assert ready['world_size'] == 8 and ready['optimizer_step'] == 718


@pytest.mark.parametrize('change', ['unquiet', 'formal_source', 'command_resume'])
def test_postexit_export_rejects_unbound_exit_or_native_completion_before_oss(post_exit_checkpoint, change):
    config, checkpoint718, proof_pin = post_exit_checkpoint
    proof = runner.read(proof_pin['path'])
    if change == 'unquiet':
        path = Path(proof['training_exit']['exit']['path']); value = runner.read(path)
        value['group_cleanup_verified'] = False
        path.write_text(json.dumps(value)); proof['training_exit']['exit'] = runner.artifact(path)
    elif change == 'formal_source':
        path = Path(proof['formal_completion']['path']); value = runner.read(path)
        value['frozen_source_manifest_sha256'] = 'f' * 64
        path.write_text(json.dumps(value)); proof['formal_completion'] = runner.artifact(path)
    else:
        path = Path(proof['training_exit']['intent']['path']); value = runner.read(path)
        value['argv'][value['argv'].index('--resume') + 1] = '/different/checkpoint-600'
        path.write_text(json.dumps(value)); proof['training_exit']['intent'] = runner.artifact(path)
    Path(proof_pin['path']).write_text(json.dumps(proof)); proof_pin = runner.artifact(proof_pin['path'])
    store = Store()
    with pytest.raises(ValueError):
        export.export_checkpoint(config, checkpoint718, 718, store=store, post_training_exit=proof_pin)
    assert not store.objects


def test_partial_native_checkpoint_and_changed_provenance_are_not_uploaded(checkpoint):
    config, path = checkpoint
    (path / 'rng_state_7.pth').unlink()
    store = Store()
    with pytest.raises(ValueError, match='RNG'):
        export.export_checkpoint(config, path, 17, store=store)
    assert store.objects == {}
    (path / 'rng_state_7.pth').write_text('7')
    (path / 'training_provenance.json').write_text('{}')
    with pytest.raises(ValueError, match='source differs'):
        export.export_checkpoint(config, path, 17, store=store)
    assert store.objects == {}


@pytest.mark.parametrize('missing', ['vision_model/model.safetensors', 'projector.safetensors',
                                   'tokenizer/tokenizer_config.json', 'image_processor/preprocessor_config.json'])
def test_missing_model_component_never_uploads(checkpoint, missing):
    config, path = checkpoint
    (path / missing).unlink()
    store = Store()
    with pytest.raises(ValueError, match='Incomplete|Unsharded'):
        export.export_checkpoint(config, path, 17, store=store)
    assert store.objects == {}


def test_index_references_every_required_shard(checkpoint):
    config, path = checkpoint
    (path / 'language_model/model.safetensors').unlink()
    safetensors(path / 'language_model/model-1.safetensors', {'first': [2, 2]})
    (path / 'language_model/model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': 32}, 'weight_map': {'first': 'model-1.safetensors', 'second': 'model-2.safetensors'}}))
    store = Store()
    with pytest.raises(ValueError, match='Incomplete sharded'):
        export.export_checkpoint(config, path, 17, store=store)
    assert store.objects == {}


def test_tokenizer_configuration_without_vocabulary_is_not_a_ready_savepoint(checkpoint):
    config, path = checkpoint
    (path / 'tokenizer/tokenizer.json').unlink()
    store = Store()
    with pytest.raises((ValueError, OSError, AttributeError)):
        export.export_checkpoint(config, path, 17, store=store)
    assert store.objects == {}


def test_equal_length_overwrite_after_verification_is_detected(checkpoint):
    config, path = checkpoint
    class ChangedAfterGet(Store):
        def iter_object(self, uri):
            data = self.objects[uri]
            self.gets.append(uri)
            yield data
            if uri.endswith('/optimizer.pt'):
                self.objects[uri] = bytes([data[0] ^ 1]) + data[1:]
    store = ChangedAfterGet()
    with pytest.raises(ValueError, match='final full readback|object identities'):
        export.export_checkpoint(config, path, 17, store=store)
    assert not any(uri.endswith('/READY.json') for uri in store.objects)


def fake_collective(monkeypatch, config, path, *, precheck_error=None, export_error=False):
    config_path = path.parent / 'EXPORT_CONFIG.json'; config_path.write_text(json.dumps(config))
    monkeypatch.setenv(export.CONFIG_ENV, str(config_path))
    monkeypatch.setenv(export.CONFIG_SHA_ENV, runner.sha(config_path))
    calls = []
    def new_group(**kwargs):
        assert kwargs['backend'] == 'gloo' and kwargs['timeout'].total_seconds() == 21600
        calls.append('new_group')
        return 'cpu-export-group'
    def gather(out, value, *, group):
        assert group == 'cpu-export-group'
        out[:] = [value] + [None] * 7
        out[7] = precheck_error
        calls.append('gather')
    def barrier(*, group):
        assert group == 'cpu-export-group'; calls.append('barrier')
    def broadcast(value, *, src, group):
        assert group == 'cpu-export-group' and src == 0; calls.append(('broadcast', value[0]))
    distributed = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, get_world_size=lambda: 8,
        new_group=new_group, all_gather_object=gather, barrier=barrier, broadcast_object_list=broadcast)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(distributed=distributed))
    monkeypatch.setitem(sys.modules, 'torch.distributed', distributed)
    def upload(*args, **kwargs):
        calls.append('upload')
        if export_error:
            raise OSError('synthetic redacted transport failure')
    monkeypatch.setattr(export, 'export_checkpoint', upload)
    trainer = SimpleNamespace(args=SimpleNamespace(output_dir=str(path.parent)), state=SimpleNamespace(global_step=17),
                              is_world_process_zero=lambda: True)
    return trainer, calls


def test_rank_precheck_error_is_gathered_before_any_payload_upload(checkpoint, monkeypatch):
    trainer, calls = fake_collective(monkeypatch, *checkpoint, precheck_error='FileNotFoundError')
    with pytest.raises(ValueError, match='rank precheck failed'):
        export.export_saved_checkpoint(trainer)
    assert calls == ['new_group', 'gather']


def test_local_pin_error_is_collectively_reported(checkpoint, monkeypatch):
    trainer, calls = fake_collective(monkeypatch, *checkpoint)
    monkeypatch.setenv(export.CONFIG_SHA_ENV, '0' * 64)
    with pytest.raises(ValueError, match='rank precheck failed'):
        export.export_saved_checkpoint(trainer)
    assert calls == ['new_group', 'gather']


def test_upload_failure_is_broadcast_using_cpu_control_group(checkpoint, monkeypatch):
    trainer, calls = fake_collective(monkeypatch, *checkpoint, export_error=True)
    with pytest.raises(ValueError, match='OSS export failed: OSError'):
        export.export_saved_checkpoint(trainer)
    assert calls == ['new_group', 'gather', 'barrier', 'upload', ('broadcast', 'OSError')]


def test_without_opt_in_the_hook_does_not_import_torch_or_create_collectives(monkeypatch):
    monkeypatch.delenv(export.CONFIG_ENV, raising=False); monkeypatch.delenv(export.CONFIG_SHA_ENV, raising=False)
    export.export_saved_checkpoint(SimpleNamespace())


def test_single_pending_export_overlaps_updates_and_final_flush_joins(checkpoint, monkeypatch):
    trainer, calls = fake_collective(monkeypatch, *checkpoint)
    started, release = threading.Event(), threading.Event()
    def upload(*args):
        started.set()
        assert release.wait(5)
    monkeypatch.setattr(export, 'export_checkpoint', upload)
    export.queue_checkpoint_export(trainer)
    try:
        assert started.wait(2) and trainer._suite_checkpoint_export_pending
        assert not trainer._suite_checkpoint_export_future.done()
        with pytest.raises(ValueError, match='Previous savepoint'):
            export.queue_checkpoint_export(trainer)
    finally:
        release.set()
        export.flush_checkpoint_export(trainer, final=True)
    assert trainer._suite_checkpoint_export_pending is False
    assert not hasattr(trainer, '_suite_checkpoint_export_pool')


def test_pending_export_failure_blocks_flush_and_preserves_unaccepted_state(checkpoint, monkeypatch):
    trainer, calls = fake_collective(monkeypatch, *checkpoint, export_error=True)
    export.queue_checkpoint_export(trainer)
    try:
        with pytest.raises(ValueError, match='before rotation/completion: OSError'):
            export.flush_checkpoint_export(trainer, final=True)
        assert trainer._suite_checkpoint_export_pending is True
        with pytest.raises(ValueError, match='Previous savepoint'):
            export.queue_checkpoint_export(trainer)
    finally:
        trainer._suite_checkpoint_export_pool.shutdown(wait=True)


@pytest.mark.parametrize('filename', ['train.py', 'train_declared.py'])
def test_actual_trainer_hook_flushes_before_native_rotation_and_failure_prevents_save(filename, monkeypatch):
    # Execute the actual two hook bodies without importing torch/model fixtures.
    source = Path(export.__file__).with_name(filename)
    tree = ast.parse(source.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'DeepProbeTrainer')
    methods = [node for node in klass.body if isinstance(node, ast.FunctionDef)
               and node.name in {'_save_checkpoint', 'save_final_resume_checkpoint'}]
    events = []
    class Base:
        def _save_checkpoint(self, *args, **kwargs):
            events.append('native_save_and_rotation')
    derived = ast.ClassDef(name='HookTrainer', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=methods,
                          decorator_list=[], type_params=[])
    namespace = {'Base': Base, '__name__': 'deepprobe_vlm.synthetic_hook_fixture', '__package__': 'deepprobe_vlm', 'Path': Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[])), str(source), 'exec'), namespace)
    trainer = namespace['HookTrainer']()
    trainer.args = SimpleNamespace(save_only_model=False, output_dir='/synthetic')
    trainer.state = SimpleNamespace(global_step=718)
    trainer.model = None
    trainer.accelerator = SimpleNamespace(wait_for_everyone=lambda: events.append('native_barrier'))
    monkeypatch.setattr(export, 'flush_checkpoint_export', lambda owner, final=False: events.append('final_flush' if final else 'flush'))
    monkeypatch.setattr(export, 'queue_checkpoint_export', lambda owner: events.append('queue'))
    trainer.save_final_resume_checkpoint()
    assert events == ['flush', 'native_save_and_rotation', 'queue', 'final_flush', 'native_barrier']
    events.clear()
    def failed(owner, **kwargs):
        raise OSError('synthetic previous export failure')
    monkeypatch.setattr(export, 'flush_checkpoint_export', failed)
    with pytest.raises(OSError, match='previous export'):
        trainer._save_checkpoint(None)
    assert events == []
