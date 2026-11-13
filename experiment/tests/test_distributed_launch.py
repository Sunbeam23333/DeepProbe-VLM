"""Exercise real CPU rendezvous; no GPU, models, scheduler or remote host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from deepprobe_vlm.distributed_launch import single_node_torchrun, single_node_torchrun_environment


WRAPPER = '''import runpy, socket, sys
# Reproduce an unresolvable advertised node name even on a healthy developer Mac.
socket.getfqdn = lambda *args, **kwargs: "dpvlm-unresolvable.invalid"
sys.argv = ["torch.distributed.run", *sys.argv[1:]]
runpy.run_module("torch.distributed.run", run_name="__main__")
'''

WORKER = '''import datetime, json, os, time, sys
from pathlib import Path
import torch
import torch.distributed as dist
root = Path(sys.argv[1])
assert os.environ["MASTER_ADDR"] == "127.0.0.1", os.environ["MASTER_ADDR"]
assert int(os.environ["MASTER_PORT"]) > 0
assert os.environ["WORLD_SIZE"] == "2"
assert os.environ["LOCAL_WORLD_SIZE"] == "2"
assert os.environ["TORCHELASTIC_RESTART_COUNT"] == "0"
dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=20))
try:
    rank = dist.get_rank()
    value = torch.tensor([rank + 1.0])
    dist.all_reduce(value)
    assert value.item() == 3.0
    item = {"rank": rank, "master_addr": os.environ["MASTER_ADDR"],
            "master_port": int(os.environ["MASTER_PORT"]), "sum": value.item(),
            "run_id": os.environ["TORCHELASTIC_RUN_ID"], "torch": torch.__version__}
    ranks = [None] * 2
    dist.all_gather_object(ranks, item)
    if rank == 0:
        pending = root / "ready.pending"
        pending.write_text(json.dumps(ranks))
        pending.replace(root / "READY.json")
    deadline = time.monotonic() + 30
    while not (root / "RELEASE").is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("test did not release both concurrently live groups")
        time.sleep(0.05)
    dist.barrier()
finally:
    dist.destroy_process_group()
'''


def test_single_node_defaults_pin_both_addresses_and_zero_restarts():
    first, second = single_node_torchrun(sys.executable), single_node_torchrun(Path(sys.executable))
    assert first[:4] == [sys.executable, "-B", "-m", "torch.distributed.run"]
    for flag in ("--nnodes=1", "--node-rank=0", "--nproc_per_node=8", "--max_restarts=0",
                 "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0", "--local-addr=127.0.0.1",
                 "--rdzv-conf=is_host=true"):
        assert flag in first
    assert "--standalone" not in first
    assert first[:-1] == second[:-1]
    assert first[-1].startswith("--rdzv-id=dpvlm-") and first[-1] != second[-1]


@pytest.mark.parametrize("count", [0, -1, True, "8"])
def test_invalid_process_count_is_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        single_node_torchrun(sys.executable, nproc_per_node=count)


def test_inherited_flags_cannot_change_the_reviewed_store_path():
    inherited = {"PET_STANDALONE": "1", "TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1",
                 "OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1"}
    observed = single_node_torchrun_environment(inherited)
    assert observed == dict(inherited, PET_STANDALONE="0", TORCH_DISABLE_SHARE_RDZV_TCP_STORE="0")
    assert inherited["PET_STANDALONE"] == inherited["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] == "1"


def test_suite_subprocess_applies_environment_guards_after_overrides(tmp_path, monkeypatch):
    from deepprobe_vlm.suite_actions import Actions
    from deepprobe_vlm import suite_actions
    actions = Actions.__new__(Actions)
    actions.root = tmp_path
    actions.env = {"runtime_environment": {"PET_STANDALONE": "1"}}
    monkeypatch.setenv("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "1")
    captured = {}

    def run(argv, **kwargs):
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(suite_actions.subprocess, "run", run)
    actions.command(single_node_torchrun(sys.executable) + ["worker.py"], tmp_path, "fixture",
                    extra_env={"PET_STANDALONE": "1", "TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1"})
    assert captured["PET_STANDALONE"] == captured["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] == "0"
    assert captured["HF_HUB_OFFLINE"] == captured["PYTHONNOUSERSITE"] == "1"


def test_two_real_gloo_groups_ignore_unresolvable_hostname_and_use_distinct_live_ports(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("the local Torch build has no Gloo distributed backend")
    wrapper, worker = tmp_path / "torchrun_with_bad_hostname.py", tmp_path / "worker.py"
    wrapper.write_text(WRAPPER)
    worker.write_text(WORKER)
    inherited = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PET_STANDALONE="1",
                       TORCH_DISABLE_SHARE_RDZV_TCP_STORE="1",
                       PET_LOCAL_ADDR="dpvlm-unresolvable.invalid", MASTER_ADDR="dpvlm-unresolvable.invalid",
                       MASTER_PORT="1")
    environment = single_node_torchrun_environment(inherited)
    processes, logs, roots = [], [], []
    try:
        for index in range(2):
            root = tmp_path / str(index)
            root.mkdir()
            roots.append(root)
            argv = single_node_torchrun(sys.executable, nproc_per_node=2)
            # Change only the module entry to inject bad hostname resolution;
            # every production rendezvous argument remains unmodified.
            command = argv[:2] + [str(wrapper)] + argv[4:] + [str(worker), str(root)]
            log = (root / "torchrun.log").open("wb")
            logs.append(log)
            processes.append(subprocess.Popen(command, env=environment, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + 45
        while not all((root / "READY.json").is_file() for root in roots):
            failed = [p.returncode for p in processes if p.poll() is not None]
            if failed or time.monotonic() > deadline:
                pytest.fail("CPU rendezvous failed: " + "\n".join(
                    (root / "torchrun.log").read_text() for root in roots))
            time.sleep(0.05)
        reports = [json.loads((root / "READY.json").read_text()) for root in roots]
        ports, run_ids = set(), set()
        for ranks in reports:
            assert [row["rank"] for row in ranks] == [0, 1]
            assert {row["master_addr"] for row in ranks} == {"127.0.0.1"}
            assert {row["sum"] for row in ranks} == {3.0}
            assert len({row["master_port"] for row in ranks}) == 1
            assert len({row["run_id"] for row in ranks}) == 1
            ports.add(ranks[0]["master_port"])
            run_ids.add(ranks[0]["run_id"])
        assert len(ports) == len(run_ids) == 2
        for root in roots:
            (root / "RELEASE").touch()
        for process, root in zip(processes, roots):
            assert process.wait(timeout=20) == 0, (root / "torchrun.log").read_text()
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        for log in logs:
            log.close()
