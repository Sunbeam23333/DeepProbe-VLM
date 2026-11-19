"""Old P0 retirement tests use synthetic exact manifests and in-memory objects."""
from contextlib import nullcontext
from copy import deepcopy
import json

import pytest

from deepprobe_vlm import checkpoint_retention as r
from deepprobe_vlm import historical_retirement as h
from test_checkpoint_retention import MemoryStore


def fixture(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return r.file_pin(path)
    stores, manifests, objects = {}, [], []
    for run_id in ('p0-05-20261009-v3', h.KEPT_RUN):
        files = {'strict/control/final/vision_model/model.safetensors': b'vision',
                 'strict/control/final/language_model/model.safetensors': b'language',
                 'strict/control/final/config.json': b'configuration',
                 'strict/control/checkpoint-20/optimizer.pt': b'optimizer',
                 'strict/control/checkpoint-20/model.safetensors': b'intermediate',
                 'EVALUATION.json': b'engineering-evidence'}
        store = MemoryStore()
        rows = []
        for name, raw in files.items():
            import hashlib
            rows.append({'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
            store.objects[h.ROOT + run_id + '/' + name] = raw
        manifest = {'run_id': run_id, 'uri': h.ROOT + run_id, 'files': rows,
                    'file_count': len(rows), 'total_bytes': sum(row['bytes'] for row in rows)}
        pin = write(run_id + '.json', manifest)
        store.objects[h.ROOT + run_id + '/ARCHIVE_MANIFEST.json'] = (tmp_path / (run_id + '.json')).read_bytes()
        objects.extend(store.list_prefix(h.ROOT + run_id + '/'))
        manifests.append(pin)
        stores[run_id] = store
    accepted = write('accepted.json', {'run_id': h.KEPT_RUN, 'status': 'p0_05_engineering_gate_and_archive_verified',
        'job': {'scheduler_state': 'Succeeded', 'pod_container_exit_codes': [0]},
        'archive_manifest_sha256': manifests[-1]['sha256']})
    scientific = write('scientific.json', {'tasks': [{'id': 'train-r1', 'initializer': {'kind': 'backbones'}}]})
    snapshot = write('snapshot.json', {'oss_inventory': {'complete': True, 'objects': objects},
        'oss': {'bytes': sum(row['bytes'] for row in objects)}})
    plan = h.prepare(snapshot, manifests, accepted, scientific)
    class Budget:
        def lock(self):
            return nullcontext()
        def _read(self):
            return {'reservations': {}, 'artifacts': {}}
        def _admit(self, *args):
            pytest.fail('Decrement-only retirement must work while current usage exceeds its cap')
    guard = lambda value: {'plan_sha256': r.digest(value), 'active_project_jobs': [],
                           'active_writers': [], 'checkpoint_references': []}
    return plan, stores, Budget(), guard


def test_retirement_preserves_v8_final_results_and_resumes_without_new_storage(tmp_path):
    plan, stores, budget, guard = fixture(tmp_path)
    before = sum(len(store.objects) for store in stores.values())
    observed = h.execute(plan, budget=budget, stores=stores, guard=guard, transaction_root=tmp_path / 'tx', apply=True)
    assert observed['new_oss_bytes'] == 0
    assert sum(len(store.objects) for store in stores.values()) == before - len(plan['targets'])
    for row in plan['preserved_accepted_final']:
        assert row['uri'] in stores[row['run_id']].objects
    for run_id, store in stores.items():
        assert h.ROOT + run_id + '/EVALUATION.json' in store.objects
        assert h.ROOT + run_id + '/ARCHIVE_MANIFEST.json' in store.objects
    resumed = h.execute(plan, budget=budget, stores=stores, guard=guard, transaction_root=tmp_path / 'tx', apply=True)
    assert resumed['receipt'] == observed['receipt']


def test_corrupt_retained_recovery_and_pending_reference_block_all_deletions(tmp_path):
    plan, stores, budget, guard = fixture(tmp_path)
    row = plan['preserved_accepted_final'][0]
    stores[row['run_id']].objects[row['uri']] = b'corrupt'
    before = {run: dict(store.objects) for run, store in stores.items()}
    with pytest.raises(ValueError, match='Preserved P0 final'):
        h.execute(plan, budget=budget, stores=stores, guard=guard, transaction_root=tmp_path / 'tx', apply=True)
    assert {run: store.objects for run, store in stores.items()} == before
    blocking = lambda value: {**guard(value), 'checkpoint_references': ['unexpected-scientific-resume']}
    with pytest.raises(ValueError, match='active writer/job/reference'):
        h.execute(plan, budget=budget, stores=stores, guard=blocking, transaction_root=tmp_path / 'tx', apply=True)


def test_retirement_plan_cannot_expand_to_formal_models(tmp_path):
    plan, stores, budget, guard = fixture(tmp_path)
    changed = deepcopy(plan)
    changed['targets'][0]['uri'] = h.ROOT + 'dp-p1-smollm2-s29-alignment-a3/training/checkpoint-350/optimizer.pt'
    with pytest.raises(ValueError, match='plan changed'):
        h.execute(changed, budget=budget, stores=stores, guard=guard, transaction_root=tmp_path / 'tx', apply=True)
