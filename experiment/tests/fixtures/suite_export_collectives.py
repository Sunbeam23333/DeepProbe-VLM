"""Eight-process CPU-only export control fault fixture, with no model or OSS."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import torch.distributed as distributed

from deepprobe_vlm import suite_checkpoint_export as export, suite_runner as runner


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    distributed.init_process_group('gloo', timeout=timedelta(seconds=60))
    rank = distributed.get_rank()
    output = args.output.resolve()
    config = output / 'config.json'
    os.environ[export.CONFIG_ENV] = str(config)
    os.environ[export.CONFIG_SHA_ENV] = runner.sha(config)
    def trainer():
        return SimpleNamespace(args=SimpleNamespace(output_dir=str(output)), state=SimpleNamespace(global_step=17),
                               is_world_process_zero=lambda: rank == 0)
    results = {'rank': rank, 'pid': os.getpid(), 'gpu_used': False}
    try:
        first = trainer()
        if rank == 3:
            os.environ[export.CONFIG_SHA_ENV] = '0' * 64
        try:
            export.queue_checkpoint_export(first)
        except ValueError as exc:
            results['rank_precheck_converged'] = 'rank precheck failed' in str(exc)
        else:
            raise AssertionError('Invalid rank accepted')
        os.environ[export.CONFIG_SHA_ENV] = runner.sha(config)
        distributed.barrier()
        second = trainer()
        def failing(*args):
            raise OSError('synthetic background transport error')
        export.export_checkpoint = failing
        export.queue_checkpoint_export(second)
        try:
            export.flush_checkpoint_export(second, final=True)
        except ValueError as exc:
            results['background_failure_converged'] = 'before rotation/completion: OSError' in str(exc)
        else:
            raise AssertionError('Failed background export accepted')
        results['failure_pending_preserved'] = second._suite_checkpoint_export_pending is True
        if rank == 0:
            second._suite_checkpoint_export_pool.shutdown(wait=True)
        distributed.barrier()
        third = trainer()
        started, release = threading.Event(), threading.Event()
        def delayed(*args):
            started.set()
            assert release.wait(20)
        export.export_checkpoint = delayed
        export.queue_checkpoint_export(third)
        if rank == 0:
            assert started.wait(5) and not third._suite_checkpoint_export_future.done()
        try:
            export.queue_checkpoint_export(third)
        except ValueError as exc:
            results['second_pending_refused'] = 'Previous savepoint' in str(exc)
        else:
            raise AssertionError('Second pending export accepted')
        release.set()
        export.flush_checkpoint_export(third, final=True)
        results['final_flush_closed'] = not third._suite_checkpoint_export_pending
        assert all(value is True for name, value in results.items() if name not in {'rank', 'pid', 'gpu_used'})
        runner.write_once(output / f'rank-{rank}.json', results)
        distributed.barrier()
    finally:
        distributed.destroy_process_group()


if __name__ == '__main__':
    main()
