"""Shared-node scheduler regressions; fixtures never use a real cluster or GPU."""
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import unittest

from deepprobe_vlm import suite_fleet as fleet, suite_runner as runner
from deepprobe_vlm.suite_storage import SharedBudget


def empty_remote():
    return {'complete': True, 'objects': [], 'multipart': []}


def child_run(plan, binding, state, worker, result_queue):
    try:
        result = runner.run_fleet_worker(plan, binding, plan_sha256=runner.digest(plan),
            binding_sha256=runner.digest(binding), state_root=state, worker_id=worker,
            poll_seconds=.02, budget=SharedBudget(binding['storage'], empty_remote), live_check=lambda: None)
        result_queue.put({'worker': worker, 'status': result['status']})
    except BaseException as exc:
        result_queue.put({'worker': worker, 'error': repr(exc)})
        raise


WORKER = r'''
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--environment');p.add_argument('--environment-sha256')
p.add_argument('--task');p.add_argument('--receipt');p.add_argument('--source');p.add_argument('--root')
a=p.parse_args();w=os.environ['DP_SUITE_WORKER_ID'];env=json.loads(Path(a.environment).read_text())
if a.task=='startup':
 r={'status':'SUITE_STARTUP_ACCEPTED','allocation':env['allocation'],'source_identity_sha256':a.source,
    'other_active_project_jobs':[],'gpu_count':8,'gpu_uuids':[w+'-'+str(i) for i in range(8)],
    'job_uid':'job-'+w,'pod_uid':'pod-'+w,'node':'node-'+w,'worker_id':w,
    'hardware_preflight_passed':True,'gpu_product':'NVIDIA-B300'}
else:
 root=Path(a.root);start=time.time()
 with (root/(a.task+'.start')).open('x') as f:f.write(str(start))
 time.sleep(.35)
 r={'status':'ACCEPTED','task':a.task,'worker_id':w,'start':start,'end':time.time()}
with Path(a.receipt).open('x') as f:json.dump(r,f)
'''


class FleetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.home = self.root / 'home'; self.home.mkdir()
        self.stage = self.root / 'stage'; self.stage.mkdir()
        self.profile = {'schema_version': 2, 'fleet_id': 'fixture-fleet', 'priority': 'normal', 'gpu_product': 'NVIDIA-B300',
            'node_selector': {'sslaunch-pool': 'stream', 'aliyun.accelerator/nvidia_name': 'NVIDIA-L20D'},
            'workers': [{'worker_id': 'worker-0' + str(i), 'job_name': 'jinhaopeng-dpvlm-fixture-worker-' + str(i), 'namespace': 'default'} for i in (1, 2)]}
        self.storage = {'schema_version': 2, 'roots': [str(self.home), str(self.stage)], 'ledger_root': str(self.home / 'ledger'),
            'home_limit_bytes': 150_000_000_000, 'oss_limit_bytes': 150_000_000_000,
            'oss_prefixes': ['oss://ss-devbox-stream/project/fixture/'],
            'home_safety_bytes': 1024, 'oss_safety_bytes': 1024, 'free_guard_bytes': 1024}
        self.plan = {'tasks': [
            {'id': 'first', 'kind': 'train', 'dependencies': []},
            {'id': 'second', 'kind': 'train', 'dependencies': []},
            {'id': 'consumer', 'kind': 'evaluate', 'dependencies': ['first', 'second']},
            {'id': 'cleanup', 'kind': 'cleanup', 'dependencies': ['consumer']}]}
        worker = self.home / 'worker.py'; worker.write_text(WORKER)
        env = {'storage': self.storage, 'execution_profile': self.profile, 'allocation': fleet.profile_allocation(self.profile)}
        self.environment = self.home / 'environment.json'; self.environment.write_text(json.dumps(env))
        self.binding = {'schema_version': 2, 'suite_id': 'fixture-fleet', 'plan_sha256': runner.digest(self.plan),
            'allocation': fleet.profile_allocation(self.profile), 'execution_profile': self.profile,
            'source_files': [runner.artifact(worker)], 'storage': self.storage, 'environment': {}, 'startup': {}, 'tasks': {}}
        def command(task, receipt):
            return {'mode': 'execute', 'argv': [sys.executable, '-B', str(worker), '--environment', str(self.environment),
                '--environment-sha256', runner.sha(self.environment), '--task', task, '--receipt', receipt,
                '--source', runner.source_identity(self.binding), '--root', str(self.home)],
                'cwd': str(self.home), 'reserve_bytes': 4096, 'oss_reserve_bytes': 4096, 'timeout_seconds': 20,
                'receipts': [{'path': receipt, 'checks': [{'pointer': '/status', 'equals': 'SUITE_STARTUP_ACCEPTED' if task == 'startup' else 'ACCEPTED'}]}]}
        self.binding['startup'] = command('startup', '{invocation_dir}/STARTUP.json'); self.binding['startup'].pop('mode')
        self.binding['tasks'] = {t['id']: command(t['id'], '{task_attempt_dir}/VERIFIED.json') for t in self.plan['tasks']}
        self.state = self.home / 'suite-state'

    def test_two_real_workers_overlap_but_consumers_wait_and_each_task_runs_once(self):
        ctx = multiprocessing.get_context('spawn'); queue = ctx.Queue()
        children = [ctx.Process(target=child_run, args=(self.plan, self.binding, str(self.state), w['worker_id'], queue)) for w in self.profile['workers']]
        for child in children: child.start()
        for child in children:
            child.join(30)
            if child.is_alive(): child.terminate(); child.join(); self.fail('Worker hung')
            self.assertEqual(child.exitcode, 0)
        reports = [queue.get(timeout=2) for _ in children]
        self.assertTrue(all(r.get('status') == 'EXPERIMENT_SUITE_COMPLETE' for r in reports), reports)
        records = {}
        for task in self.plan['tasks']:
            done = runner.completion_for(self.state, task['id'])
            records[task['id']] = runner.read(done['receipts'][0]['path'])
            self.assertEqual(len(list((self.state / 'tasks' / task['id'] / 'attempts').iterdir())), 1)
        a, b, c = (records[k] for k in ('first', 'second', 'consumer'))
        self.assertLess(max(a['start'], b['start']), min(a['end'], b['end']))
        self.assertGreaterEqual(c['start'], max(a['end'], b['end']))
        self.assertGreaterEqual(records['cleanup']['start'], c['end'])
        ledger = runner.read(self.home / 'ledger/LEDGER.json')
        self.assertTrue(all(row['state'] == 'RELEASED' for row in ledger['reservations'].values()))
        self.assertEqual(runner.read(self.state / 'COMPLETE.json')['completed_tasks'], 4)

    def test_four_real_workers_share_one_journal_and_storage_ledger(self):
        for i in (3, 4):
            self.profile['workers'].append({'worker_id': 'worker-0' + str(i),
                'job_name': 'jinhaopeng-dpvlm-fixture-worker-' + str(i), 'namespace': 'default'})
        environment = runner.read(self.environment)
        environment['execution_profile'] = self.profile
        self.environment.write_text(json.dumps(environment))
        for command in [self.binding['startup'], *self.binding['tasks'].values()]:
            argv = command['argv']; argv[argv.index('--environment-sha256') + 1] = runner.sha(self.environment)
        ctx = multiprocessing.get_context('spawn'); queue = ctx.Queue()
        children = [ctx.Process(target=child_run, args=(self.plan, self.binding, str(self.state), worker['worker_id'], queue))
                    for worker in self.profile['workers']]
        for child in children: child.start()
        for child in children:
            child.join(30)
            if child.is_alive(): child.terminate(); child.join(); self.fail('Worker hung')
            self.assertEqual(child.exitcode, 0)
        self.assertTrue(all(queue.get(timeout=2)['status'] == 'EXPERIMENT_SUITE_COMPLETE' for _ in children))
        rendezvous = runner.read(self.state / 'FLEET_RENDEZVOUS.json')
        self.assertEqual(len(rendezvous['workers']), 4)
        self.assertEqual(len(list((self.state / 'tasks').glob('*/COMPLETED.json'))), 4)
        self.assertTrue(all(row['state'] == 'RELEASED' for row in runner.read(self.home / 'ledger/LEDGER.json')['reservations'].values()))

    def test_shared_budget_can_throttle_two_workers_to_one_training_task(self):
        self.storage['home_limit_bytes'] = 100_000_000
        environment = runner.read(self.environment); environment['storage'] = self.storage
        self.environment.write_text(json.dumps(environment))
        for command in [self.binding['startup'], *self.binding['tasks'].values()]:
            argv = command['argv']; argv[argv.index('--environment-sha256') + 1] = runner.sha(self.environment)
        for task_id in ('first', 'second'):
            self.binding['tasks'][task_id]['reserve_bytes'] = 60_000_000
        ctx = multiprocessing.get_context('spawn'); queue = ctx.Queue()
        children = [ctx.Process(target=child_run, args=(self.plan, self.binding, str(self.state), w['worker_id'], queue))
                    for w in self.profile['workers']]
        for child in children: child.start()
        for child in children:
            child.join(30)
            if child.is_alive(): child.terminate(); child.join(); self.fail('Budget-throttled worker hung')
            self.assertEqual(child.exitcode, 0)
        first, second = [runner.read(runner.completion_for(self.state, task_id)['receipts'][0]['path']) for task_id in ('first', 'second')]
        self.assertLessEqual(first['end'], second['start'])
        self.assertTrue(all(queue.get(timeout=2)['status'] == 'EXPERIMENT_SUITE_COMPLETE' for _ in children))

    def test_stale_claim_never_expires_and_duplicate_worker_cannot_join(self):
        state = fleet.FleetState(self.plan, self.binding, self.state, plan_sha256=runner.digest(self.plan), binding_sha256=runner.digest(self.binding))
        state.initialize(); state.join('worker-01')
        self.assertFalse(state.workers_ready())
        with self.assertRaisesRegex(ValueError, 'already claimed'): state.join('worker-01')
        reservation = lambda task, owner: owner['claim_id']
        first = state.claim('worker-01', 'original-job', reservation)
        self.assertEqual(first['task']['id'], 'first')
        second = state.claim('worker-02', 'another-job', reservation)
        self.assertEqual(second['task']['id'], 'second')
        self.assertEqual(state.claim('worker-02', 'another-job', reservation)['status'], 'wait')
        state.fail(first['task'], first['claim'], RuntimeError('preempted'))
        with self.assertRaisesRegex(ValueError, 'failed task'): state.claim('worker-02', 'another-job', reservation)

    def test_global_lease_blocks_another_state_or_fleet(self):
        state = fleet.FleetState(self.plan, self.binding, self.state, plan_sha256=runner.digest(self.plan), binding_sha256=runner.digest(self.binding))
        state.initialize()
        other = fleet.FleetState(self.plan, self.binding, self.home / 'other-state', plan_sha256=runner.digest(self.plan), binding_sha256=runner.digest(self.binding))
        with self.assertRaisesRegex(ValueError, 'lease differs'): other.initialize()
        other_binding = deepcopy(self.binding); other_binding['suite_id'] = 'different'; other_binding['execution_profile']['fleet_id'] = 'different'
        other = fleet.FleetState(self.plan, other_binding, self.home / 'other-state', plan_sha256=runner.digest(self.plan), binding_sha256=runner.digest(other_binding))
        with self.assertRaisesRegex(ValueError, 'unfinished global'): other.initialize()

    def test_profile_rejects_fifth_worker_wrong_gpu_high_priority_and_duplicate(self):
        for change in ({'priority': 'high'}, {'gpu_product': 'NVIDIA-H100'}, {'workers': self.profile['workers'] * 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                fleet.validate_execution_profile({**self.profile, **change})
        duplicate = deepcopy(self.profile); duplicate['workers'][1]['job_name'] = duplicate['workers'][0]['job_name']
        with self.assertRaisesRegex(ValueError, 'Duplicate'): fleet.validate_execution_profile(duplicate)


if __name__ == '__main__': unittest.main()
