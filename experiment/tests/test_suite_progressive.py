"""Local synthetic readiness tests: no real bucket, GPU, model or job access."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest
from PIL import Image

from deepprobe_vlm import experiment_suite, formal_integrity, suite_media as media
from deepprobe_vlm import suite_preflight, suite_progressive as progressive, suite_runner as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical(value) + b'\n')
    return runner.artifact(path)


@pytest.fixture
def declared(tmp_path):
    root = tmp_path.resolve()
    source = root / 'producer-v2'
    for name in formal_integrity.SOURCE_DIRS:
        (source / name).mkdir(parents=True, exist_ok=True)
    for name in {*formal_integrity.REQUIRED_P1_SOURCE, *formal_integrity.SOURCE_FILES}:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# synthetic fixture source only\n')
    source_pin = write(root / 'control-v2/SOURCE.json', {'schema_version': 1,
        'purpose': 'DEEPPROBE_FROZEN_P1_SOURCE', 'files': formal_integrity.source_inventory(source)})
    env = {'suite_id': 'synthetic-progressive', 'experiment_root': str(root / 'runtime-v3'),
        'storage': {'schema_version': 2, 'roots': [str(root)], 'ledger_root': str(root / 'ledger'),
            'home_limit_bytes': 150_000_000_000, 'oss_limit_bytes': None,
            'oss_prefixes': ['oss://ss-devbox-stream/' + media.PROJECT_PREFIX],
            'home_safety_bytes': 1024, 'oss_safety_bytes': 1024, 'free_guard_bytes': 1024},
        'execution_profile': {'schema_version': 2, 'fleet_id': 'synthetic-progressive', 'priority': 'low',
            'gpu_product': 'NVIDIA-B300', 'node_selector': {'sslaunch-pool': 'stream',
                'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'}, 'workers': [{'worker_id': 'worker-01',
                'job_name': 'jinhaopeng-dpvlm-synthetic-progressive-worker-01', 'namespace': 'default'}]},
        'tools': {'oss_config': str(root / 'synthetic-no-real-config')}, 'benchmark_preparations': {}}
    for benchmark in progressive.BENCHMARKS:
        for frames in progressive.FRAMES:
            shape = benchmark + '-f' + str(frames)
            request = {'benchmark': benchmark, 'frames': frames,
                'source_manifest': write(root / 'inputs' / (benchmark + '.jsonl'), {'synthetic': benchmark}),
                'accepted_decode_receipt': write(root / 'inputs' / (benchmark + '-decode.json'),
                                                   {'complete': True, 'entries': []}),
                'archive_audit': write(root / 'inputs' / (benchmark + '-archive.json'), {'synthetic': True}),
                'output_dir': str(root / 'metadata' / shape), 'media_root': str(root / 'frames' / shape),
                'oss_prefix': media.PROJECT_PREFIX + 'shape-ablation/synthetic/' + benchmark + '/uniform' + str(frames) + '/',
                'max_png_bytes': 1_000_000, 'metadata_reserve_bytes': 64 * 1024 * 1024}
            env['benchmark_preparations'][progressive.key(benchmark, frames)] = progressive.make_declaration(
                env, request, source_pin, root / 'control-v3/cache-assets' / shape / 'READY.json',
                producer_source_root=source)
    return env, experiment_suite.default_plan(), source


@pytest.fixture
def cache(declared, monkeypatch):
    env, plan, producer = declared
    declaration = env['benchmark_preparations']['tempcompass_mc:f4']
    request = declaration['contract']['request']
    output = Path(request['output_dir'])
    inputs = output.with_name(output.name + '-inputs')
    normalized = runner.read(request['accepted_decode_receipt']['path'])
    binding = {'dataset': 'tempcompass', 'archives': [], 'sources': [],
        'accepted_original_decode_receipt_sha256': media.digest(media.encode(normalized)),
        'archive_catalog_sha256': media.digest(media.encode([])),
        'normalization_evidence': {'original_decode_receipt': request['accepted_decode_receipt'],
            'archive_audit': request['archive_audit'],
            'archive_bound_receipt': write(inputs / 'ARCHIVE_BOUND_SOURCE_RECEIPT.json', normalized)}}
    write(inputs / 'ARCHIVE_CATALOG.json', [])
    pins = {'source_manifest': request['source_manifest'], 'source_binding': write(inputs / 'SOURCE_BINDING.json', binding)}
    for name, filename in (('manifest', 'manifest.jsonl'), ('receipt', 'FRAME_CACHE_RECEIPT.json'),
                            ('complete', 'CACHE_COMPLETE.json'), ('plan', 'BUILD_PLAN.json')):
        pins[name] = write(output / filename, {'synthetic_only': name})
    image = Image.new('RGB', (2, 2), (17, 23, 41))
    stream = io.BytesIO(); image.save(stream, format='PNG'); png = stream.getvalue()
    row = {'path': 'tiny.png', 'key': request['oss_prefix'] + 'tiny.png', 'bytes': len(png),
        'sha256': media.digest(png), 'rgb_sha256': media.digest(image.tobytes()), 'width': 2, 'height': 2}
    asset = {'cache': {'kind': 'suite', 'prefix': request['oss_prefix'], **pins},
             'media_root': request['media_root'], 'max_bytes': request['max_png_bytes']}
    raw = {name: Path(pin['path']).read_bytes() for name, pin in pins.items()}
    # Replace only the full scientific fixture construction. Production pin,
    # receipt publication and remote PNG SHA/RGB readback code all run below.
    monkeypatch.setattr(media, '_bound_cache_inventory', lambda _: ([{'synthetic': True}], binding,
        [row], request['frames'], raw, list(raw)))
    objects = {row['key']: png}
    for name, filename in (('plan', 'BUILD_PLAN.json'), ('manifest', 'manifest.jsonl'),
                            ('receipt', 'FRAME_CACHE_RECEIPT.json'), ('complete', 'CACHE_COMPLETE.json')):
        objects[request['oss_prefix'] + filename] = raw[name]
    class Store:
        def __init__(self): self.gets = []
        def get(self, object_key, maximum):
            self.gets.append(object_key)
            return objects.get(object_key)
        def keys(self, prefix): return iter(k for k in objects if k.startswith(prefix))
    store = Store()
    monkeypatch.setattr(media, 'OSSStore', lambda *args, **kwargs: store)
    return env, declaration, asset, objects, store, producer


def test_complete_dag_can_declare_six_pending_shapes_without_network(declared, monkeypatch):
    env, plan, _ = declared
    forbidden = Mock(side_effect=AssertionError('No remote access in declarations or pending resolution'))
    monkeypatch.setattr(media, 'OSSStore', forbidden)
    original = runner.digest(plan)
    assert progressive.validate_declarations(env, plan) == env['benchmark_preparations']
    assert len(plan['tasks']) == 185 and runner.digest(plan) == original
    assert all(progressive.resolve_ready(env, b, f) is None for b in progressive.BENCHMARKS for f in progressive.FRAMES)
    forbidden.assert_not_called()


def test_publish_runs_actual_png_readback_then_resolves_locally_without_environment_mutation(cache, monkeypatch):
    env, declaration, asset, _, store, _ = cache
    original = runner.digest(env)
    result = progressive.publish_ready(env, declaration, asset)
    assert result['asset'] == asset and result['contract_sha256'] == declaration['contract_sha256']
    assert any(k.endswith('.png') for k in store.gets)
    ready = runner.read(result['ready_pin']['path'])
    assert ready['remote_verification']['all_png_full_get_and_rgb_verified'] is True
    assert ready['remote_verification']['formal_evaluation_completed'] is False
    assert not Path(asset['media_root']).exists()
    monkeypatch.setattr(media, 'OSSStore', Mock(side_effect=AssertionError('Resolution/reuse must not issue GETs')))
    assert progressive.resolve_ready(env, 'tempcompass_mc', 4, expected_pin=result['ready_pin']) == result
    assert progressive.publish_ready(env, declaration, asset) == result
    assert runner.digest(env) == original


@pytest.mark.parametrize('damage', ['missing', 'corrupt', 'extra'])
def test_failed_remote_acceptance_does_not_publish_ready(cache, damage):
    env, declaration, asset, objects, _, _ = cache
    png_key = next(k for k in objects if k.endswith('.png'))
    if damage == 'missing': del objects[png_key]
    elif damage == 'corrupt': objects[png_key] = b'changed'
    else: objects[declaration['contract']['request']['oss_prefix'] + 'unexpected.png'] = b'extra'
    with pytest.raises(ValueError, match='inventory|full-GET SHA'):
        progressive.publish_ready(env, declaration, asset)
    assert not Path(declaration['ready_path']).exists()


@pytest.mark.parametrize('damage', ['wrong_contract', 'false_readback', 'wrong_inventory', 'partial'])
def test_existing_invalid_ready_is_an_error_never_pending(cache, damage):
    env, declaration, asset, _, _, _ = cache
    ready = progressive.publish_ready(env, declaration, asset)['ready_pin']
    value = runner.read(ready['path'])
    if damage == 'wrong_contract': value['contract_sha256'] = '0' * 64
    elif damage == 'false_readback': value['remote_verification']['all_png_full_get_and_rgb_verified'] = False
    elif damage == 'wrong_inventory': value['inventory_sha256'] = '0' * 64
    else: value.pop('asset')
    write(Path(ready['path']), value)
    with pytest.raises(ValueError, match='READY'):
        progressive.resolve_ready(env, 'tempcompass_mc', 4)


def test_claimed_ready_cannot_be_replaced_or_disappear(cache):
    env, declaration, asset, _, _, _ = cache
    result = progressive.publish_ready(env, declaration, asset)
    path = Path(result['ready_pin']['path'])
    value = runner.read(path); value['generated_utc'] = '2001-01-01T00:00:00+00:00'
    write(path, value)
    with pytest.raises(ValueError, match='pinned artifact changed'):
        progressive.resolve_ready(env, 'tempcompass_mc', 4, expected_pin=result['ready_pin'])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        progressive.resolve_ready(env, 'tempcompass_mc', 4, expected_pin=result['ready_pin'])


@pytest.mark.parametrize('damage', ['prefix', 'frames', 'decode', 'source_manifest', 'metadata_path'])
def test_asset_must_match_its_exact_frozen_request(cache, damage):
    env, declaration, asset, _, _, _ = cache
    changed = deepcopy(asset)
    if damage == 'prefix': changed['cache']['prefix'] += 'different/'
    elif damage == 'source_manifest': changed['cache']['source_manifest'] = changed['cache']['receipt']
    elif damage == 'metadata_path': changed['cache']['plan'] = changed['cache']['receipt']
    elif damage == 'decode':
        Path(declaration['contract']['request']['accepted_decode_receipt']['path']).write_text('changed')
    else:
        declaration['contract']['request']['frames'] = 8
        declaration['contract_sha256'] = runner.digest(declaration['contract'])
    with pytest.raises(ValueError):
        progressive.publish_ready(env, declaration, changed)
    assert not Path(declaration['ready_path']).exists()


def test_producer_source_is_separate_from_runtime_but_must_remain_exact(cache):
    env, declaration, asset, _, _, source = cache
    assert declaration['contract']['producer']['source_root'] != env['experiment_root']
    result = progressive.publish_ready(env, declaration, asset)
    (source / 'src/deepprobe_vlm/train.py').write_text('changed producer source')
    with pytest.raises(ValueError, match='Frozen source tree differs'):
        progressive.resolve_ready(env, 'tempcompass_mc', 4, expected_pin=result['ready_pin'])


@pytest.mark.parametrize('damage', ['missing_shape', 'legacy_storage', 'legacy_fleet', 'shared_ready', 'shared_prefix', 'escape'])
def test_declaration_set_rejects_incomplete_or_overlapping_contracts(declared, damage):
    env, plan, _ = declared
    declarations = env['benchmark_preparations']
    first, second = declarations['videomme899:f4'], declarations['videomme899:f8']
    if damage == 'missing_shape': declarations.pop('videomme899:f16')
    elif damage == 'legacy_storage': env['storage']['schema_version'] = 1
    elif damage == 'legacy_fleet': env.pop('execution_profile')
    elif damage == 'shared_ready': second['ready_path'] = first['ready_path']
    elif damage == 'shared_prefix':
        second['contract']['request']['oss_prefix'] = first['contract']['request']['oss_prefix']
        second['contract_sha256'] = runner.digest(second['contract'])
    else: first['ready_path'] = str(Path(env['storage']['roots'][0]).parent / 'outside/READY.json')
    with pytest.raises(ValueError):
        progressive.validate_declarations(env, plan)


def test_unregistered_declaration_never_publishes(cache):
    env, declaration, asset, _, _, _ = cache
    changed = deepcopy(declaration); changed['ready_path'] += '.other.json'
    with pytest.raises(ValueError, match='not frozen'):
        progressive.publish_ready(env, changed, asset)
    assert not Path(changed['ready_path']).exists()


def admission(env):
    deferred = [{'status': 'DEFERRED_BENCHMARK', 'benchmark': d['contract']['request']['benchmark'],
                 'frames': d['contract']['request']['frames'], 'contract_sha256': d['contract_sha256'],
                 'ready_path': d['ready_path']} for d in env['benchmark_preparations'].values()]
    return {'status': 'SUITE_CPU_PROGRESSIVE_ADMITTED', 'benchmark_preparation_mode': 'external_progressive',
            'failures': [], 'deferred_benchmarks': deepcopy(deferred), 'checks': [
                {'check': 'training_data_and_live_media', 'status': 'passed', 'result': {'synthetic': True}},
                *({'check': 'benchmark/' + d['benchmark'] + '/' + str(d['frames']), 'status': 'deferred',
                   'result': d} for d in deferred)]}


def test_admission_records_deferred_snapshot_even_if_ready_appears_later(cache):
    env, declaration, asset, _, _, _ = cache
    report = admission(env)
    plan = experiment_suite.default_plan()
    assert progressive.validate_admission(report, env, plan)
    progressive.publish_ready(env, declaration, asset)
    assert progressive.validate_admission(report, env, plan)


def test_admission_checks_exact_ready_pin_and_deferred_inventory(cache):
    env, declaration, asset, _, _, _ = cache
    resolved = progressive.publish_ready(env, declaration, asset)
    report = admission(env)
    report['deferred_benchmarks'] = [d for d in report['deferred_benchmarks']
                                   if not (d['benchmark'] == 'tempcompass_mc' and d['frames'] == 4)]
    row = next(r for r in report['checks'] if r['check'] == 'benchmark/tempcompass_mc/4')
    row.update(status='passed', result={k: v for k, v in resolved.items() if k != 'asset'})
    assert progressive.validate_admission(report, env, experiment_suite.default_plan())
    Path(resolved['ready_pin']['path']).write_text('changed')
    with pytest.raises(ValueError, match='pinned artifact changed'):
        progressive.validate_admission(report, env, experiment_suite.default_plan())


@pytest.mark.parametrize('damage', ['status', 'mode', 'failure', 'missing_check', 'duplicate', 'deferred_inventory',
                                  'wrong_contract', 'wrong_ready_path', 'deferred_training', 'missing_pin'])
def test_admission_cannot_promote_incomplete_or_generic_gated_report(declared, damage):
    env, plan, _ = declared
    report = admission(env)
    if damage == 'status': report['status'] = 'SUITE_CPU_GATED'
    elif damage == 'mode': report['benchmark_preparation_mode'] = 'eager'
    elif damage == 'failure': report['failures'].append({'error': 'synthetic'})
    elif damage == 'missing_check': report['checks'].pop()
    elif damage == 'duplicate': report['checks'].append(deepcopy(report['checks'][-1]))
    elif damage == 'deferred_inventory': report['deferred_benchmarks'].pop()
    elif damage == 'wrong_contract': report['checks'][-1]['result']['contract_sha256'] = '0' * 64
    elif damage == 'wrong_ready_path': report['checks'][-1]['result']['ready_path'] += '.other'
    elif damage == 'deferred_training': report['checks'][0]['status'] = 'deferred'
    else: report['checks'][-1]['status'] = 'passed'
    with pytest.raises(ValueError):
        progressive.validate_admission(report, env, plan)


def test_ready_is_absent_until_atomic_link_and_concurrent_publisher_cannot_corrupt(cache, monkeypatch):
    env, declaration, asset, _, _, _ = cache
    ready_path = Path(declaration['ready_path'])
    linking, release = threading.Event(), threading.Event()
    original_link = runner.os.link
    def blocked_link(source, target, *args, **kwargs):
        if Path(target) == ready_path:
            # write_once has already flushed and fsynced the whole unpublished
            # file. No reader can interpret this temporary name as READY.
            assert runner.read(source)['status'] == progressive.STATUS
            linking.set()
            assert release.wait(5), 'Test publisher was not released'
        return original_link(source, target, *args, **kwargs)
    monkeypatch.setattr(runner.os, 'link', blocked_link)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(progressive.publish_ready, env, declaration, asset)
        try:
            assert linking.wait(5), 'Publisher did not reach atomic publication'
            assert progressive.resolve_ready(env, 'tempcompass_mc', 4) is None
            second = workers.submit(progressive.publish_ready, env, declaration, asset)
            # write_once's same-process O_EXCL temporary name refuses this
            # concurrent publication without touching the first writer's file.
            with pytest.raises(FileExistsError): second.result(timeout=5)
            assert progressive.resolve_ready(env, 'tempcompass_mc', 4) is None
        finally:
            release.set()
        result = first.result(timeout=5)
    assert progressive.resolve_ready(env, 'tempcompass_mc', 4, expected_pin=result['ready_pin']) == result
    assert not list(ready_path.parent.glob('READY.json.pending-*'))


def test_publisher_losing_final_link_preserves_winning_complete_receipt(cache, monkeypatch):
    env, declaration, asset, _, _, _ = cache
    ready_path = Path(declaration['ready_path'])
    original_link = runner.os.link
    def another_publisher_wins(source, target, *args, **kwargs):
        if Path(target) == ready_path:
            winner = Path(source).with_name('synthetic-other-producer.pending')
            value = runner.read(source)
            value['generated_utc'] = '2026-01-01T00:00:00+00:00'
            write(winner, value)
            try: original_link(winner, target)
            finally: winner.unlink()
        return original_link(source, target, *args, **kwargs)
    monkeypatch.setattr(runner.os, 'link', another_publisher_wins)
    with pytest.raises(FileExistsError): progressive.publish_ready(env, declaration, asset)
    result = progressive.resolve_ready(env, 'tempcompass_mc', 4)
    assert result['asset'] == asset
    assert runner.read(ready_path)['generated_utc'] == '2026-01-01T00:00:00+00:00'
    assert not list(ready_path.parent.glob('*.pending*'))


def evaluation_history(env, plan):
    task = next(t for t in plan['tasks'] if t['id'] == 'evaluate__ouro_r1_s17__tempcompass_mc')
    root = Path(env['storage']['roots'][0]) / 'synthetic-evaluation-history'
    source = write(root / 'SOURCE.json', {'scope': 'synthetic-unit-test-only'})
    result = write(root / 'RESULT.json', {'status': 'SUITE_EVALUATION_COMPLETE', 'suite_protocol': task['protocol'],
        'task_sha256': runner.digest(task), 'qa': 1580, 'failed_units': 0,
        'rows': [{'sample_id': 'synthetic-' + str(i), 'correct': i % 2, 'question_identity_sha256': runner.digest(i)}
                 for i in range(1580)]})
    acceptance = write(root / 'ACCEPTANCE.json', {'schema_version': 1, 'status': 'SUITE_HISTORY_ACCEPTED',
        'production_evidence': True, 'task_id': task['id'], 'task_sha256': runner.digest(task), 'mode': 'accepted',
        'evidence': [source], 'evaluation_sha256': result['sha256'], 'checkpoint_inventory_sha256': 'a' * 64,
        'protocol': task['protocol'], 'qa': 1580, 'failed_units': 0})
    history = {'mode': 'accepted', 'acceptance': acceptance, 'evaluation': result,
               'checkpoint_inventory_sha256': 'a' * 64}
    env['history'] = {task['id']: history}
    return task, history


def test_verified_evaluation_history_bypasses_pending_cache_without_network(declared, monkeypatch):
    env, plan, _ = declared
    task, _ = evaluation_history(env, plan)
    assert progressive.resolve_ready(env, task['benchmark'], task['frames']) is None
    forbidden = Mock(side_effect=AssertionError('Accepted evaluation must not build or resolve a new cache'))
    monkeypatch.setattr(progressive, 'resolve_ready', forbidden)
    monkeypatch.setattr(media, 'OSSStore', forbidden)
    assert progressive.task_readiness(env, plan, task) == {}
    forbidden.assert_not_called()


@pytest.mark.parametrize('damage', ['acceptance_changed', 'result_changed', 'wrong_mode', 'incomplete_rows', 'hint_only'])
def test_bad_evaluation_history_never_bypasses_cache_gate_or_silently_waits(declared, monkeypatch, damage):
    env, plan, _ = declared
    task, history = evaluation_history(env, plan)
    if damage == 'acceptance_changed': Path(history['acceptance']['path']).write_text('changed')
    elif damage == 'result_changed': Path(history['evaluation']['path']).write_text('changed')
    elif damage == 'wrong_mode': history['mode'] = 'resume'
    elif damage == 'hint_only': env['history'][task['id']] = {'mode': 'accepted'}
    else:
        result = runner.read(history['evaluation']['path']); result['rows'].pop()
        history['evaluation'] = write(Path(history['evaluation']['path']), result)
        review = runner.read(history['acceptance']['path'])
        review['evaluation_sha256'] = history['evaluation']['sha256']
        history['acceptance'] = write(Path(history['acceptance']['path']), review)
    forbidden = Mock(side_effect=AssertionError('Invalid history must fail before cache readiness'))
    monkeypatch.setattr(progressive, 'resolve_ready', forbidden)
    with pytest.raises(ValueError): progressive.task_readiness(env, plan, task)
    forbidden.assert_not_called()


def test_new_evaluation_waits_for_own_ready_and_non_evaluation_needs_no_cache(cache):
    env, declaration, asset, _, _, _ = cache
    plan = experiment_suite.default_plan()
    task = next(t for t in plan['tasks'] if t['kind'] == 'evaluate' and t['benchmark'] == 'tempcompass_mc' and t['frames'] == 4)
    assert progressive.task_readiness(env, plan, task) is None
    result = progressive.publish_ready(env, declaration, asset)
    assert progressive.task_readiness(env, plan, task) == {k: v for k, v in result.items() if k != 'asset'}
    assert progressive.task_readiness(env, plan, {'kind': 'train'}) == {}
    strict = deepcopy(env); strict.pop('benchmark_preparations')
    assert progressive.task_readiness(strict, plan, task) == {}


def test_preclaim_cache_failure_stops_other_workers_before_reservation_or_claim():
    from test_suite_fleet import FleetTest
    from deepprobe_vlm import suite_fleet
    fixture = FleetTest(); fixture.setUp()
    try:
        fixture.plan['tasks'] = [{'id': 'consumer', 'kind': 'evaluate', 'dependencies': []},
                                 {'id': 'first', 'kind': 'train', 'dependencies': []}]
        state = suite_fleet.FleetState(fixture.plan, fixture.binding, fixture.state,
            plan_sha256=runner.digest(fixture.plan), binding_sha256=runner.digest(fixture.binding))
        state.initialize()
        reserve = Mock(side_effect=AssertionError('Failure must precede reservation'))
        with pytest.raises(ValueError, match='corrupt READY'):
            state.claim('worker-01', 'synthetic-job-01', reserve,
                        readiness=Mock(side_effect=ValueError('corrupt READY')))
        assert not list(fixture.state.rglob('FAILED.json'))
        # The runner's outer exception handler writes this when failure occurs
        # before it has a task claim. Kubernetes may still report a Running Pod.
        stopped = fixture.state / 'invocations/failed-worker'
        stopped.mkdir()
        runner.write_once(stopped / 'STOPPED.json', {'status': 'EXPERIMENT_SUITE_STOPPED',
            'worker_id': 'worker-01', 'error_type': 'ValueError', 'error': 'corrupt READY'})
        with pytest.raises(ValueError, match='worker failure'):
            state.claim('worker-02', 'synthetic-job-02', reserve, readiness=lambda task: {})
        reserve.assert_not_called()
        assert not list(fixture.state.rglob('CLAIM.json'))
        assert not list((fixture.state / 'tasks').glob('*/attempts'))
    finally:
        fixture.doCleanups()


@pytest.mark.parametrize('tampered', [False, True])
def test_publication_failure_never_becomes_pending_for_consumer(declared, tampered):
    env, plan, _ = declared
    for frames in (4, 8):
        declaration = env['benchmark_preparations'][progressive.key('videomme899', frames)]
        failure = Path(declaration['ready_path']).with_name('READY.FAILED.json')
        write(failure, {'status': 'SUITE_BENCHMARK_PREPARATION_FAILED',
            'contract': declaration['contract'], 'contract_sha256': '0' * 64 if tampered else declaration['contract_sha256'],
            'evidence': []})
        task = next(t for t in plan['tasks'] if t['kind'] == 'evaluate'
                    and t['benchmark'] == 'videomme899' and t['frames'] == frames)
        with pytest.raises(ValueError, match='failure marker' if tampered else 'CPU cache preparation failed'):
            progressive.task_readiness(env, plan, task)
