"""Real eight-rank Gloo error convergence, using CPU and synthetic uploads."""
import json
import os
from pathlib import Path
import subprocess
import sys

from deepprobe_vlm.distributed_launch import single_node_torchrun, single_node_torchrun_environment
from deepprobe_vlm.suite_process import run_group


def test_eight_cpu_ranks_converge_on_precheck_and_upload_failures_and_flush(tmp_path):
    tmp_path = tmp_path.resolve()
    (tmp_path / 'config.json').write_text(json.dumps({'world_size': 8}))
    fixture = Path(__file__).parent / 'fixtures/suite_export_collectives.py'
    environment = single_node_torchrun_environment(dict(os.environ))
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    environment['OMP_NUM_THREADS'] = '1'
    log = tmp_path / 'stdout.log'
    with log.open('wb') as stream:
        result = run_group(single_node_torchrun(sys.executable) + [str(fixture), '--output', str(tmp_path)],
                           cwd=Path(__file__).resolve().parents[1], env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0 and result.group_cleanup_verified, log.read_text(errors='replace')[-4000:]
    rows = [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(8)]
    assert len({row['pid'] for row in rows}) == 8
    for row in rows:
        assert row['gpu_used'] is False
        assert all(row[name] is True for name in ('rank_precheck_converged', 'background_failure_converged',
                                                  'failure_pending_preserved', 'second_pending_refused', 'final_flush_closed'))
