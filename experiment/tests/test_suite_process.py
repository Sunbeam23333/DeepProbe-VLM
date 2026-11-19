"""Synthetic nested process ownership; no real child or background process."""
import io
import os
from types import SimpleNamespace
import unittest
from unittest import mock

from deepprobe_vlm import suite_process as process


class SuiteProcessAncestryTest(unittest.TestCase):
    outer, middle, inner = ('1' * 32, '2' * 32, '3' * 32)

    @staticmethod
    def encoded(environment):
        return b'\0'.join((key + '=' + value).encode() for key, value in environment.items()) + b'\0'

    def test_nested_environment_retains_all_ancestors_and_single_marker_api(self):
        initial = {'PYTHONPATH': '/synthetic/source'}
        outer = process._child_environment(initial, self.outer)
        middle = process._child_environment(outer, self.middle)
        inner = process._child_environment(middle, self.inner)
        self.assertEqual(initial, {'PYTHONPATH': '/synthetic/source'})
        self.assertEqual(inner[process.MARKER], self.inner)
        self.assertEqual(process._marker_tokens(inner), (self.outer, self.middle, self.inner))
        self.assertEqual(process._process_marker_tokens(self.encoded(inner)),
                         (self.outer, self.middle, self.inner))
        legacy = process._child_environment({process.MARKER: self.outer}, self.inner)
        self.assertEqual(process._marker_tokens(legacy), (self.outer, self.inner))
        self.assertEqual(process._process_marker_tokens(self.encoded({process.MARKER: self.outer})),
                         (self.outer,))

    def make_tree(self, token):
        leader = mock.Mock(pid=93410)
        leader.poll.return_value = 0
        with mock.patch.object(process, 'process_table', return_value={}):
            tree = process.OwnedTree(leader, token)
        # The inner child detached before the outer scan and its parent died.
        tree.identities = {leader.pid: 'old-leader-start'}
        return tree

    @staticmethod
    def orphan(pid):
        return {'pid': pid, 'ppid': 1, 'pgid': pid, 'sid': str(pid), 'state': 'S',
                'start': 'orphan-start-' + str(pid), 'uid': os.getuid()}

    def test_each_ancestor_finds_reparented_inner_child_without_parent_or_group(self):
        environment = {}
        for token in (self.outer, self.middle, self.inner):
            environment = process._child_environment(environment, token)
        unrelated = process._child_environment({}, '4' * 32)
        rows = {pid: self.orphan(pid) for pid in (93411, 93412)}
        def environ(path, mode):
            self.assertEqual(mode, 'rb')
            values = environment if path == '/proc/93411/environ' else unrelated
            return io.BytesIO(self.encoded(values))
        for token in (self.outer, self.middle, self.inner):
            with self.subTest(supervisor=token[0]):
                tree = self.make_tree(token)
                with mock.patch.object(process, 'LINUX_PROC', True), \
                     mock.patch.object(process, 'process_table', return_value=rows), \
                     mock.patch('builtins.open', side_effect=environ):
                    self.assertEqual(set(tree.refresh()), {93411})
                    self.assertFalse(tree.quiet_after_leader_exit())

    def test_run_group_passes_complete_ancestry_to_mocked_child(self):
        inherited = process._child_environment({}, self.outer)
        child = mock.Mock(pid=93410)
        child.poll.return_value = 0
        tree = mock.Mock()
        tree.refresh.return_value = {}
        tree.quiet_after_leader_exit.return_value = True
        with mock.patch.object(process, 'LINUX_PROC', False), \
             mock.patch.object(process.uuid, 'uuid4', return_value=SimpleNamespace(hex=self.inner)), \
             mock.patch.object(process.signal, 'signal'), \
             mock.patch.object(process.subprocess, 'Popen', return_value=child) as spawn, \
             mock.patch.object(process, 'OwnedTree', return_value=tree):
            result = process.run_group(['/synthetic/python'], stdout=None, stderr=None,
                                       env=inherited, cwd='/synthetic', timeout=1)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.group_cleanup_verified)
        self.assertEqual(process._marker_tokens(spawn.call_args.kwargs['env']), (self.outer, self.inner))

    def test_maximum_depth_preserves_oldest_identity_and_overflow_never_spawns(self):
        tokens = [format(index, '032x') for index in range(process.MAX_MARKER_DEPTH + 1)]
        environment = {}
        for token in tokens[:-1]:
            environment = process._child_environment(environment, token)
        self.assertEqual(process._process_marker_tokens(self.encoded(environment)), tuple(tokens[:-1]))
        with mock.patch.object(process.uuid, 'uuid4', return_value=SimpleNamespace(hex=tokens[-1])), \
             mock.patch.object(process.subprocess, 'Popen') as spawn, \
             mock.patch.object(process, 'PidfdAPI') as pidfds:
            with self.assertRaisesRegex(process.GroupCleanupError, 'ancestry exceeds its bound') as caught:
                process.run_group(['/synthetic/python'], stdout=None, stderr=None,
                                  env=environment, cwd='/synthetic', timeout=1)
        spawn.assert_not_called()
        pidfds.assert_not_called()
        for token in tokens:
            self.assertNotIn(token, str(caught.exception))
        self.assertEqual(process._marker_tokens(environment), tuple(tokens[:-1]))

    def test_invalid_inherited_identity_is_rejected_before_spawn_without_token_disclosure(self):
        cases = [
            {process.MARKER: self.outer + 'x'},
            {process.MARKER: 'G' * 32},
            {process.MARKER_CHAIN: self.outer},
            {process.MARKER: self.outer, process.MARKER_CHAIN: ''},
            {process.MARKER: self.outer, process.MARKER_CHAIN: self.middle},
            {process.MARKER: self.outer, process.MARKER_CHAIN: self.outer + ',' + self.outer},
            {process.MARKER: self.outer, process.MARKER_CHAIN: self.middle + ', ' + self.outer},
            {process.MARKER: self.outer, process.MARKER_CHAIN: (self.outer + ',') * 32},
        ]
        for index, environment in enumerate(cases):
            with self.subTest(case=index), mock.patch.object(process.subprocess, 'Popen') as spawn:
                with self.assertRaises(process.GroupCleanupError) as caught:
                    process.run_group(['/synthetic/python'], stdout=None, stderr=None,
                                      env=environment, cwd='/synthetic', timeout=1)
                spawn.assert_not_called()
                self.assertNotIn(self.outer, str(caught.exception))
                self.assertNotIn(self.middle, str(caught.exception))

    def test_ambiguous_or_malformed_inventory_fails_closed(self):
        marker = (process.MARKER + '=' + self.outer).encode()
        chain = (process.MARKER_CHAIN + '=' + self.outer).encode()
        cases = [marker + b'\0' + marker, marker + b'\0' + chain + b'\0' + chain,
                 marker + b'\0' + chain + b',bad', marker + b'\0' + chain + b'\xff',
                 marker + b'\0' + chain + b',' + self.outer.encode()]
        for index, raw in enumerate(cases):
            tree = self.make_tree(self.outer)
            with self.subTest(case=index), mock.patch.object(process, 'LINUX_PROC', True), \
                 mock.patch.object(process, 'process_table', return_value={93411: self.orphan(93411)}), \
                 mock.patch('builtins.open', return_value=io.BytesIO(raw)):
                with self.assertRaises(process.GroupCleanupError) as caught:
                    tree.refresh()
                self.assertFalse(caught.exception.group_cleanup_verified)
                self.assertNotIn(self.outer, str(caught.exception))


if __name__ == '__main__':
    unittest.main()
