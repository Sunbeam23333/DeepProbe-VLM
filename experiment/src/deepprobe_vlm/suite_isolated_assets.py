"""Route managed benchmark consumers into one isolated worker namespace.

The original READY receipt and cache contract remain immutable inputs. This
helper performs no I/O beyond path inspection and grants no storage authority;
the isolated storage admission validates the coordinator's lifetime hold.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from . import suite_runner as runner
from .experiment_suite import BENCHMARKS


def _directory_path(value, label):
    runner.require(isinstance(value, str) and value and not value.startswith('//')
                   and '\\' not in value and not any(ord(c) < 32 or ord(c) == 127 for c in value),
                   'Invalid canonical ' + label)
    path = Path(value)
    runner.require(path.is_absolute() and str(path) == value and path != Path('/'),
                   'Canonical absolute non-root ' + label + ' required')
    path = runner.safe(path)
    runner.require(all(not p.exists() or p.is_dir() for p in (path, *path.parents)),
                   label + ' must name a directory')
    return path


def isolate_benchmark_asset(environment, task, data):
    """Copy a managed asset with its consumer media root in the new namespace.

    A direct-media asset is only read, so its existing path is retained. Missing
    isolated directories are valid destinations, never a reason to reuse the
    original cache directory. No directories or receipts are created here.
    """
    if 'isolated_storage' not in environment:
        return data
    isolated = environment['isolated_storage']
    runner.require(isinstance(isolated, dict) and type(isolated.get('schema_version')) is int
                   and isolated['schema_version'] == 1, 'Invalid isolated storage namespace')
    namespace = _directory_path(isolated.get('namespace_root'), 'isolated namespace root')
    from .suite_fleet import validate_execution_profile
    profile = validate_execution_profile(environment.get('execution_profile'), environment.get('suite_id'))
    runner.require(len(profile['workers']) == 1, 'Isolated benchmark namespace requires one worker')
    runner.require(isinstance(task, dict) and task.get('benchmark') in BENCHMARKS
                   and type(task.get('frames')) is int and task['frames'] in (4, 8, 16),
                   'Invalid isolated benchmark or frame count')
    runner.require(isinstance(data, dict), 'Invalid isolated benchmark asset')
    if 'cache' not in data:
        return data
    source = _directory_path(data.get('media_root'), 'original benchmark media root')
    runner.require(not namespace.is_relative_to(source) and not source.is_relative_to(namespace),
                   'Isolated namespace must not overlap the original benchmark media root')
    destination = _directory_path(str(namespace / 'media' / task['benchmark'] / ('f' + str(task['frames']))),
                                  'isolated benchmark media root')
    result = deepcopy(data)
    result['media_root'] = str(destination)
    return result
