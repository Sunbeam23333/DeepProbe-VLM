"""Authenticate a completed training process tree before post-exit export.

Only the returned intent and exit metadata need durable retention. Their bytes
may be relocated separately; the exit still binds the original intent bytes.
"""
from __future__ import annotations

import os
from pathlib import PurePosixPath
import time

from . import suite_process, suite_runner as runner


ACCEPTED = 'SUITE_TRAINING_PROCESS_EXITED_QUIESCENT'
FAILED = 'SUITE_TRAINING_PROCESS_FAILED'
TIMEOUT_SECONDS = 30 * 24 * 3600


def _pin(pin):
    runner.require(isinstance(pin, dict) and set(pin) == {'path', 'bytes', 'sha256'}
                   and type(pin['bytes']) is int and pin['bytes'] > 0,
                   'Training process evidence requires a complete nonempty pin')
    runner.verify_pin(pin)
    return runner.read(pin['path'])


def _log(path):
    # Empty stdout/stderr are valid and are not regular nonempty receipt pins.
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': runner.sha(path)}


def run_training(argv, *, cwd, env, logs_prefix):
    """Run one command tree; retain failure evidence but never accept an error."""
    from .suite_actions import save
    runner.require(isinstance(argv, (list, tuple)) and argv
                   and all(isinstance(arg, str) and arg and '\x00' not in arg for arg in argv),
                   'Training command needs exact nonempty argv strings')
    source, prefix = runner.safe(cwd), runner.safe(logs_prefix)
    runner.require(source.is_dir(), 'Training source root does not exist')
    runner.require(isinstance(env, dict), 'Training child environment must be explicit')
    paths = {name: prefix.with_name(prefix.name + '.' + name)
             for name in ('intent.json', 'exit.json', 'stdout', 'stderr')}
    runner.require(not any(os.path.lexists(path) for path in paths.values()),
                   'Training logs already exist; command will not restart or overwrite evidence')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    args = list(argv); started = time.monotonic()
    intent = save(paths['intent.json'], {'schema_version': 1, 'status': 'SUITE_TRAINING_PROCESS_INTENT',
        'argv': args, 'argv_sha256': runner.digest(args), 'source_root': str(source), 'automatic_retry': False})
    code, quiet, error = None, False, None
    try:
        with paths['stdout'].open('xb') as stdout, paths['stderr'].open('xb') as stderr:
            result = suite_process.run_group(args, stdout=stdout, stderr=stderr, env=env,
                cwd=str(source), timeout=TIMEOUT_SECONDS)
        code, quiet = result.returncode, result.group_cleanup_verified is True
        runner.require(type(code) is int and code == 0, 'Training command exited unsuccessfully')
        runner.require(quiet, 'Training process tree is not proven quiescent')
    except BaseException as exc:
        if code is None:
            code = getattr(exc, 'group_returncode', None)
        quiet = quiet or getattr(exc, 'group_cleanup_verified', False) is True
        error = type(exc).__name__
        raise
    finally:
        exited = save(paths['exit.json'], {'schema_version': 1,
            'status': ACCEPTED if error is None and type(code) is int and code == 0 and quiet else FAILED,
            'intent': intent, 'argv_sha256': runner.digest(args), 'source_root': str(source),
            'returncode': code, 'group_cleanup_verified': quiet, 'error_type': error,
            'automatic_retry': False, 'elapsed_seconds': time.monotonic() - started,
            'logs': {name: _log(paths[name]) for name in ('stdout', 'stderr') if paths[name].is_file()}})
    proof = {'exit': exited, 'intent': intent}
    verify_training_exit(proof, expected_source_root=source, expected_argv_sha256=runner.digest(args))
    return proof


def verify_training_exit(proof, *, expected_source_root, expected_argv_sha256=None):
    """Read both exact pins; permit only byte-identical intent-path relocation.

    No original intent path is reopened or inferred. The supplied intent must
    equal the exit's recorded intent SHA and size, even after private disk loss.
    Export callers must provide the expected command digest; omitting it checks
    only internal command consistency. Scientific source bytes remain subject
    to the caller's separate frozen-source and history validation.
    """
    runner.require(isinstance(proof, dict) and set(proof) == {'exit', 'intent'},
                   'Training exit proof requires separate exit and intent pins')
    intent, exited = _pin(proof['intent']), _pin(proof['exit'])
    source = str(runner.safe(expected_source_root))
    runner.require(isinstance(intent, dict) and intent.get('schema_version') == 1
        and intent.get('status') == 'SUITE_TRAINING_PROCESS_INTENT'
        and intent.get('source_root') == source and intent.get('automatic_retry') is False,
        'Training intent has another source or execution contract')
    args = intent.get('argv')
    runner.require(isinstance(args, list) and args
        and all(isinstance(arg, str) and arg and '\x00' not in arg for arg in args)
        and intent.get('argv_sha256') == runner.digest(args), 'Training argv identity differs')
    runner.require(isinstance(exited, dict) and exited.get('schema_version') == 1
        and exited.get('status') == ACCEPTED and exited.get('source_root') == source
        and exited.get('argv_sha256') == intent['argv_sha256']
        and type(exited.get('returncode')) is int and exited['returncode'] == 0
        and exited.get('group_cleanup_verified') is True and exited.get('error_type') is None
        and exited.get('automatic_retry') is False, 'Training exit is not successful and quiescent')
    original = exited.get('intent')
    runner.require(isinstance(original, dict) and set(original) == {'path', 'bytes', 'sha256'}
        and isinstance(original['path'], str) and type(original['bytes']) is int
        and all(original[key] == proof['intent'][key] for key in ('bytes', 'sha256')),
        'Training exit does not bind the supplied exact intent bytes')
    original_path = PurePosixPath(original['path'])
    runner.require(original_path.is_absolute() and '..' not in original_path.parts
                   and '\x00' not in original['path'], 'Original intent path is not lexically absolute')
    if expected_argv_sha256 is not None:
        runner.require(isinstance(expected_argv_sha256, str) and runner.SHA.fullmatch(expected_argv_sha256)
            and intent['argv_sha256'] == expected_argv_sha256, 'Training command differs from expected argv')
    return {'status': ACCEPTED, 'intent': dict(proof['intent']), 'exit': dict(proof['exit']),
            'source_root': source, 'argv_sha256': intent['argv_sha256'], 'group_cleanup_verified': True}
