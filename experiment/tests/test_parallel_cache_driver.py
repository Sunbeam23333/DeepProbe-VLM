"""The CPU cache child cannot adopt an old or unrelated supervisor receipt."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deepprobe_vlm import suite_runner as runner

DRIVER = Path(__file__).resolve().parents[1] / 'scripts/operations/run_parallel_suite_cache.py'
spec = importlib.util.spec_from_file_location('parallel_cache_driver_under_test', DRIVER)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class SupervisorIdentityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.control = self.root / 'control'
        self.path = self.control / 'cpu-cache-prepare' / ('a' * 32) / 'START.json'
        self.path.parent.mkdir(parents=True)
        self.proof = {key: {key: 'test'} for key in ('candidate', 'driver', 'source_manifest', 'execution')}
        self.start = {'status': 'CPU_CACHE_PREPARATION_STARTED', 'output_root': str(self.path.parent),
                      'pid': 123, 'linux_process_start': '456', **self.proof}
        patcher = patch.object(driver, 'CONTROL', self.control)
        patcher.start()
        self.addCleanup(patcher.stop)

    def publish(self, *, path=None, document=None, active=None):
        path = path or self.path
        document = self.start if document is None else document
        runner.write_once(path, document)
        runner.write_once(self.control / 'CPU_CACHE_PREPARE_ACTIVE.json', document if active is None else active)
        return runner.artifact(path), document

    def verify(self, pin, start, *, parent=123, quiet=False):
        with patch.object(driver.os, 'getppid', return_value=parent), \
             patch.object(driver, 'exact_process_quiet', return_value=quiet):
            driver.verify_supervisor_start(runner, pin, start, self.proof)

    def test_exact_active_live_direct_parent_is_accepted(self):
        pin, start = self.publish()
        self.verify(pin, start)

    def test_wrong_attempt_directory_is_rejected_even_with_matching_active(self):
        other = self.root / 'old-control' / 'START.json'
        other.parent.mkdir()
        pin, start = self.publish(path=other, document={**self.start, 'output_root': str(other.parent)})
        with self.assertRaisesRegex(ValueError, 'outside this exact'):
            self.verify(pin, start)

    def test_stale_active_owner_is_rejected(self):
        pin, start = self.publish(active={**self.start, 'pid': 999})
        with self.assertRaisesRegex(ValueError, 'current ACTIVE'):
            self.verify(pin, start)

    def test_dead_or_unrelated_supervisor_is_rejected(self):
        pin, start = self.publish()
        for parent, quiet in [(124, False), (123, True)]:
            with self.assertRaisesRegex(ValueError, 'exact live supervisor'):
                self.verify(pin, start, parent=parent, quiet=quiet)

    def test_wrong_start_status_and_output_are_rejected(self):
        pin, start = self.publish(document={**self.start, 'status': 'STOPPED', 'output_root': str(self.root)})
        with self.assertRaisesRegex(ValueError, 'status/output'):
            self.verify(pin, start)
