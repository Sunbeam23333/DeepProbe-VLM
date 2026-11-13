"""Build a single-node torchrun command without depending on the host's DNS."""
from __future__ import annotations

import os
import uuid
from collections.abc import Mapping


def single_node_torchrun(python: str | os.PathLike[str], *, nproc_per_node: int = 8) -> list[str]:
    """Allocate rendezvous ports in TCPStore and pin both addresses to loopback.

    Construct this prefix immediately before a subprocess launch, not while
    generating a persistent plan or binding: each invocation has a fresh ID.
    In PyTorch's c10d backend, the rendezvous endpoint and the advertised worker
    store address are separate. ``local-addr`` controls the latter; setting only
    ``master-addr`` or the endpoint does not prevent an unresolved-hostname hang.
    Port zero lets the live store acquire a port without probe/release races.
    """
    if type(nproc_per_node) is not int or nproc_per_node < 1:
        raise ValueError("nproc_per_node must be a positive integer")
    return [os.fspath(python), "-B", "-m", "torch.distributed.run",
            "--nnodes=1", "--node-rank=0", f"--nproc_per_node={nproc_per_node}",
            "--max_restarts=0", "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0",
            "--local-addr=127.0.0.1", "--rdzv-conf=is_host=true",
            "--rdzv-id=dpvlm-" + uuid.uuid4().hex]


def single_node_torchrun_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Preserve the caller's environment while enforcing the reviewed store path.

    Torch's inherited PET_STANDALONE flag overrides explicit rendezvous options.
    Disabling store sharing selects a probe/release port allocation path instead
    of letting the shared TCPStore acquire a live ephemeral port.
    """
    return dict(environment, PET_STANDALONE="0", TORCH_DISABLE_SHARE_RDZV_TCP_STORE="0")
