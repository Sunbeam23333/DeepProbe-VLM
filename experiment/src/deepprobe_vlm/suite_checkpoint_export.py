"""One pending immutable OSS savepoint for private-disk formal training.

The native save first drains the previous export, then queues the current save
after every rank finishes it. No training schedule, sample order, weights or
RNG state is changed. READY is published last after full payload readback; a
lost Pod can restore only an explicitly reviewed READY generation.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from . import checkpoint_retention as retention
from . import suite_runner as runner


CONFIG_ENV = 'DP_SUITE_CHECKPOINT_EXPORT_CONFIG'
CONFIG_SHA_ENV = 'DP_SUITE_CHECKPOINT_EXPORT_CONFIG_SHA256'
CONTROL_TIMEOUT_SECONDS = 6 * 60 * 60


def _post_training_control(config, checkpoint, step, proof_pin):
    """Authenticate the sole legacy post-exit route without claiming Gloo I/O."""
    from . import suite_legacy_private_resume as legacy
    from .suite_training_process import verify_training_exit
    runner.verify_pin(proof_pin)
    proof = runner.read(proof_pin['path'])
    runner.require(config['task_id'] == legacy.TASK_ID and step == 718
        and config['frozen_source_manifest']['sha256'] == legacy.LEGACY_FROZEN_SHA
        and all(proof.get(key) == value for key, value in {
            'schema_version': 1, 'status': 'LEGACY_NATIVE_POST_TRAINING_EXPORT_INPUT',
            'task_id': config['task_id'], 'task_sha256': config['task_sha256'], 'optimizer_step': 718,
            'source_root': legacy.LEGACY_SOURCE_ROOT, 'frozen_source_manifest': config['frozen_source_manifest'],
            'native_checkpoint': str(checkpoint)}.items()), 'Legacy post-exit export identity differs')
    runner.require(runner.safe(proof_pin['path']).is_relative_to(runner.safe(config['state_root'])),
                   'Post-exit export evidence must survive in shared state')
    verify_training_exit(proof['training_exit'], expected_source_root=legacy.LEGACY_SOURCE_ROOT)
    intent = runner.read(proof['training_exit']['intent']['path'])
    runner.verify_pin(proof['legacy_review'])
    review = runner.read(proof['legacy_review']['path'])
    facts = review.get('verified_facts', {})
    runner.require(review.get('status') == legacy.REVIEW_STATUS and review.get('approved') is True
        and review.get('production_evidence') is True and review.get('task_id') == config['task_id']
        and review.get('task_sha256') == config['task_sha256']
        and facts.get('legacy_frozen_source_manifest') == config['frozen_source_manifest']
        and facts.get('legacy_source_root') == legacy.LEGACY_SOURCE_ROOT
        and facts.get('optimizer_step') == 600 and facts.get('expected_optimizer_steps') == 718,
        'Post-exit export lost its exact independently reviewed legacy bridge')
    argv = intent.get('argv', [])
    for option, expected in (('--output-dir', str(checkpoint.parent)),
                             ('--resume', str(checkpoint.parent / 'checkpoint-600')),
                             ('--config', facts.get('legacy_config', {}).get('path'))):
        runner.require(argv.count(option) == 1 and argv.index(option) + 1 < len(argv)
                       and argv[argv.index(option) + 1] == expected,
                       'Post-exit export training command differs from the legacy resume')
    runner.verify_pin(proof['formal_completion'])
    formal = runner.read(proof['formal_completion']['path'])
    runner.require(all(formal.get(key) == value for key, value in {
        'status': 'formal_training_complete', 'optimizer_step': 718, 'world_size': 8,
        'frozen_source_manifest_sha256': legacy.LEGACY_FROZEN_SHA}.items())
        and proof['formal_completion']['sha256'] == runner.sha(checkpoint.parent / 'FORMAL_TRAIN_COMPLETE.json'),
        'Post-exit export requires the exact formal 718-step completion')
    return {'control_backend': 'single_process_after_training_exit0', 'control_timeout_seconds': None,
            'post_training_exit_proof': proof_pin}


def export_checkpoint(config, checkpoint, step, *, store=None, validate_native=None, post_training_exit=None):
    from .suite_actions import inventory, save, verify_inventory, load_operation
    from .resume_checkpoint import verify_resume_checkpoint
    runner.require(set(config) == {'schema_version', 'owner', 'task_id', 'task_sha256', 'expected_steps',
        'world_size', 'frozen_source_manifest', 'archive_root', 'state_root', 'tools', 'part_workers'}
        and config['schema_version'] == 1 and config['world_size'] == 8
        and type(step) is int and 0 < step <= config['expected_steps']
        and type(config['part_workers']) is int and 1 <= config['part_workers'] <= 4,
        'Invalid private checkpoint export contract')
    owner = config['owner']
    root = retention.check_owner(owner)
    checkpoint = runner.safe(checkpoint)
    runner.require(checkpoint == root / 'training' / ('checkpoint-' + str(step)),
                   'Savepoint is outside its exact owned native checkpoint directory')
    control = (_post_training_control(config, checkpoint, step, post_training_exit)
               if post_training_exit is not None else
               {'control_backend': 'gloo', 'control_timeout_seconds': CONTROL_TIMEOUT_SECONDS})
    runner.verify_pin(config['frozen_source_manifest'])
    (validate_native or verify_resume_checkpoint)(checkpoint, world_size=8, expected_max_steps=config['expected_steps'])
    # Reuse the existing tensor-header/architecture/sharded-index checker.
    # Its pure validation reads no tensors onto a GPU and runs no operations.
    load_operation(Path(__file__).resolve().parents[2], 'run_p1_stage_v9')._require_model_tree(checkpoint)
    runner.require((checkpoint / 'tokenizer/tokenizer.json').is_file(),
                   'Savepoint tokenizer vocabulary is missing from the supported fast-tokenizer format')
    from transformers import AutoImageProcessor, AutoTokenizer
    # Config files alone are insufficient: verify the saved vocabulary and
    # processor can be reopened offline, without loading model weights.
    AutoTokenizer.from_pretrained(str(checkpoint / 'tokenizer'), local_files_only=True, trust_remote_code=False)
    AutoImageProcessor.from_pretrained(str(checkpoint / 'image_processor'), local_files_only=True,
                                       trust_remote_code=False, use_fast=False)
    runner.require(runner.read(checkpoint / 'trainer_state.json')['global_step'] == step,
                   'Savepoint global step differs from its checkpoint name')
    rows = inventory(checkpoint)
    names = {row['path'] for row in rows}
    runner.require({'deepprobe_config.json', 'training_provenance.json', 'training_args.bin'} <= names
                   and any(name.endswith('.safetensors') for name in names), 'Savepoint lacks model/config/provenance payload')
    provenance = runner.read(checkpoint / 'training_provenance.json')
    runner.require(provenance.get('training_contract', {}).get('frozen_source_manifest_sha256')
                   == config['frozen_source_manifest']['sha256'], 'Savepoint training source differs from its export contract')
    generation = runner.digest(rows)
    # Keep generations outside the stage archive's exact /runs/<run>/ inventory.
    prefix = config['archive_root'].rstrip('/') + '/savepoints/runs/' + owner['run_id'] + '/checkpoint-' + str(step) + '/' + generation + '/'
    retention.relative(config['task_id'])
    output = runner.safe(config['state_root']) / 'checkpoint-exports' / config['task_id'] / ('checkpoint-' + str(step)) / generation
    output.mkdir(parents=True, exist_ok=True)
    from .suite_durable_metadata import retain_pin
    metadata = {name: retain_pin(runner.artifact(checkpoint / name), checkpoint, output / 'metadata',
                                max_bytes=4 * 1024 * 1024)
                for name in ('deepprobe_config.json', 'training_provenance.json', 'trainer_state.json')}
    manifest = {'schema_version': 1, 'status': 'SUITE_NATIVE_CHECKPOINT_MANIFEST', 'owner': owner,
        'task_id': config['task_id'], 'task_sha256': config['task_sha256'], 'optimizer_step': step,
        'expected_optimizer_steps': config['expected_steps'], 'world_size': 8,
        'frozen_source_manifest': config['frozen_source_manifest'], 'checkpoint_inventory_sha256': generation,
        'archive_prefix': prefix, 'files': rows}
    manifest_pin = save(output / 'MANIFEST.json', manifest)
    ready = {'schema_version': 1, 'status': 'SUITE_NATIVE_CHECKPOINT_OSS_READY',
        'manifest_sha256': manifest_pin['sha256'], 'manifest_bytes': manifest_pin['bytes'],
        'manifest_object': prefix + 'MANIFEST.json', 'checkpoint_inventory_sha256': generation,
        'optimizer_step': step, 'world_size': 8, 'payload_objects': len(rows),
        'payload_bytes': sum(row['bytes'] for row in rows), 'full_get_sha256_verified': True,
        **control,
        'automatic_resume_authorized': False}
    expected = {prefix + row['path']: {**row, 'path': str(checkpoint / row['path'])} for row in rows}
    expected[prefix + 'MANIFEST.json'] = manifest_pin
    if store is None:
        from .checkpoint_store import OssutilStore
        store = OssutilStore(config['tools']['ossutil'], config['tools']['oss_config'], allowed_prefix=prefix,
                             state_dir=root / '.retention' / 'savepoint-oss' / generation,
                             part_workers=config['part_workers'])
    before = retention._listing(store, prefix)
    runner.require(set(before) <= set(expected) | {prefix + 'READY.json'}, 'Unexpected savepoint archive object')
    for uri, row in expected.items():
        if uri not in before:
            store.put_if_absent(uri, row['path'], row['sha256'])
        runner.require(retention._object_digest(store, uri) == {'bytes': row['bytes'], 'sha256': row['sha256']},
                       'Savepoint full OSS readback differs')
    verify_inventory(checkpoint, rows)
    observed = retention._listing(store, prefix)
    runner.require({uri: row['bytes'] for uri, row in observed.items() if uri != prefix + 'READY.json'}
                   == {uri: row['bytes'] for uri, row in expected.items()}, 'Savepoint archive inventory differs')
    for uri, row in expected.items():
        runner.require(retention._object_digest(store, uri) == {'bytes': row['bytes'], 'sha256': row['sha256']},
                       'Savepoint changed during final full readback')
    runner.require(observed == retention._listing(store, prefix),
                   'Savepoint object identities changed during final full readback')
    ready_pin = save(output / 'READY.json', ready)
    if prefix + 'READY.json' not in observed:
        store.put_if_absent(prefix + 'READY.json', ready_pin['path'], ready_pin['sha256'])
    runner.require(retention._object_digest(store, prefix + 'READY.json')
                   == {'bytes': ready_pin['bytes'], 'sha256': ready_pin['sha256']}, 'Savepoint READY readback differs')
    final = retention._listing(store, prefix)
    runner.require({uri: row['bytes'] for uri, row in final.items()}
                   == {**{uri: row['bytes'] for uri, row in expected.items()}, prefix + 'READY.json': ready_pin['bytes']},
                   'Savepoint archive changed before acceptance')
    runner.require({uri: row for uri, row in final.items() if uri != prefix + 'READY.json'}
                   == {uri: row for uri, row in observed.items() if uri != prefix + 'READY.json'},
                   'Savepoint object identities changed before acceptance')
    return save(output / 'EXPORT_ACCEPTED.json', {'schema_version': 1, 'status': 'SUITE_SAVEPOINT_EXPORTED',
        'manifest': manifest_pin, 'ready': ready_pin, 'archive_prefix': prefix,
        'checkpoint_oss_prefix': prefix, 'all_objects_full_get_sha256_verified': True,
        'ready_published_last': True, 'stable_listing_sha256': runner.digest(final),
        'checkpoint_metadata': metadata, 'checkpoint_inventory_sha256': generation, 'optimizer_step': step,
        **({'post_training_exit_proof': post_training_exit} if post_training_exit is not None else {})})


def _collective_context(trainer):
    path = os.environ.get(CONFIG_ENV)
    if not path:
        runner.require(not os.environ.get(CONFIG_SHA_ENV), 'Checkpoint export SHA has no config path')
        return None
    import torch.distributed as distributed
    runner.require(distributed.is_available() and distributed.is_initialized()
                   and distributed.get_world_size() == 8, 'Savepoint export requires the exact eight-rank DDP world')
    # Archive I/O must not leave the other ranks inside a short-timeout NCCL
    # collective. This independent CPU control group changes no DDP settings.
    group = getattr(trainer, '_suite_checkpoint_export_group', None)
    if group is None:
        group = distributed.new_group(backend='gloo', timeout=timedelta(seconds=CONTROL_TIMEOUT_SECONDS))
        trainer._suite_checkpoint_export_group = group
    config, local_error = None, None
    try:
        pin = {'path': path, 'sha256': os.environ.get(CONFIG_SHA_ENV, '')}
        runner.verify_pin(pin)
        config = runner.read(path)
        runner.require(config.get('world_size') == 8, 'Savepoint config world size differs')
    except BaseException as exc:
        local_error = type(exc).__name__
    prechecks = [None] * 8
    distributed.all_gather_object(prechecks, local_error, group=group)
    runner.require(all(error is None for error in prechecks),
                   'Native checkpoint export rank precheck failed: ' + str(prechecks))
    return config, distributed, group


def export_saved_checkpoint(trainer):
    """Synchronous export primitive, also used by isolated contract tests."""
    context = _collective_context(trainer)
    if context is None:
        return
    config, distributed, group = context
    # All ranks reached this point after their native save returned.
    distributed.barrier(group=group)
    error = [None]
    if trainer.is_world_process_zero():
        try:
            export_checkpoint(config, Path(trainer.args.output_dir) / ('checkpoint-' + str(trainer.state.global_step)),
                              trainer.state.global_step)
        except BaseException as exc:
            # Only the class crosses the collective; SDK messages can contain credentials.
            error[0] = type(exc).__name__
    distributed.broadcast_object_list(error, src=0, group=group)
    runner.require(error[0] is None, 'Native checkpoint OSS export failed: ' + str(error[0]))
    distributed.barrier(group=group)


def flush_checkpoint_export(trainer, *, final=False):
    """Wait before the next native save/rotation and before formal completion."""
    if not getattr(trainer, '_suite_checkpoint_export_pending', False):
        return
    context = _collective_context(trainer)
    runner.require(context is not None, 'Pending checkpoint export lost its pinned configuration')
    _, distributed, group = context
    error = [None]
    if trainer.is_world_process_zero():
        try:
            trainer._suite_checkpoint_export_future.result()
        except BaseException as exc:
            error[0] = type(exc).__name__
    distributed.broadcast_object_list(error, src=0, group=group)
    runner.require(error[0] is None, 'Native checkpoint OSS export failed before rotation/completion: ' + str(error[0]))
    trainer._suite_checkpoint_export_pending = False
    if trainer.is_world_process_zero():
        trainer._suite_checkpoint_export_future = None
        if final:
            trainer._suite_checkpoint_export_pool.shutdown(wait=True)
            del trainer._suite_checkpoint_export_pool


def queue_checkpoint_export(trainer):
    """At most one immutable save exports while the following updates run.

    The next native save must call ``flush_checkpoint_export`` before invoking
    Transformers, whose checkpoint rotation could otherwise remove this input.
    Every formal completion also flushes, including the final native save.
    """
    context = _collective_context(trainer)
    if context is None:
        return
    config, distributed, group = context
    runner.require(not getattr(trainer, '_suite_checkpoint_export_pending', False),
                   'Previous savepoint must be durable before another native save can rotate it')
    distributed.barrier(group=group)
    error = [None]
    if trainer.is_world_process_zero():
        try:
            pool = getattr(trainer, '_suite_checkpoint_export_pool', None)
            if pool is None:
                pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint-oss-export')
                trainer._suite_checkpoint_export_pool = pool
            checkpoint = Path(trainer.args.output_dir) / ('checkpoint-' + str(trainer.state.global_step))
            trainer._suite_checkpoint_export_future = pool.submit(export_checkpoint, config, checkpoint, trainer.state.global_step)
        except BaseException as exc:
            error[0] = type(exc).__name__
    distributed.broadcast_object_list(error, src=0, group=group)
    runner.require(error[0] is None, 'Cannot start native checkpoint export: ' + str(error[0]))
    trainer._suite_checkpoint_export_pending = True


def verify_control_group(output):
    """Fresh eight-rank CPU control collective admission, with no model loading."""
    import socket
    import torch
    import torch.distributed as distributed
    runner.require(distributed.is_available() and distributed.is_gloo_available(),
                   'Private savepoint control requires a Gloo-enabled Torch runtime')
    runner.require(int(os.environ.get('WORLD_SIZE', '0')) == 8, 'Eight control verification ranks required')
    distributed.init_process_group(backend='gloo', timeout=timedelta(seconds=120))
    try:
        rank = distributed.get_rank()
        records = [None] * 8
        distributed.all_gather_object(records, {'rank': rank, 'pid': os.getpid(), 'host': socket.gethostname()})
        value = torch.tensor(rank, dtype=torch.int64)
        distributed.all_reduce(value)
        runner.require(value.item() == 28 and {row['rank'] for row in records} == set(range(8))
                       and len({row['pid'] for row in records}) == 8 and len({row['host'] for row in records}) == 1,
                       'Control collective does not cover one exact eight-process node')
        if rank == 0:
            runner.write_once(runner.safe(output), {'schema_version': 1, 'status': 'SUITE_EXPORT_CONTROL_GROUP_READY',
                'world_size': 8, 'control_backend': 'gloo', 'control_timeout_seconds': CONTROL_TIMEOUT_SECONDS,
                'startup_probe_timeout_seconds': 120, 'cpu_all_reduce': 'passed', 'ranks': records,
                'training_schedule_modified': False})
        distributed.barrier()
    finally:
        distributed.destroy_process_group()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-control-group', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    verify_control_group(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
