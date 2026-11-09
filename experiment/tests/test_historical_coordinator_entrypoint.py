"""Missing acquisition plans must not make a casual invocation write remotely."""
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_coordinator_requires_execution_opt_in_before_any_remote_call():
    path = ROOT / 'scripts/operations/benchmark_media_coordinate.py'
    with patch.object(sys, 'argv', [str(path)]), \
            patch('subprocess.run') as run, patch('subprocess.Popen') as popen, \
            patch('urllib.request.urlopen') as urlopen, pytest.raises(SystemExit) as exc:
        runpy.run_path(str(path), run_name='__main__')
    assert exc.value.code == 2
    run.assert_not_called()
    popen.assert_not_called()
    urlopen.assert_not_called()
