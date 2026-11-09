"""The diagnostic leaf hook sees local embedding gradients before Gloo DDP."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from deepprobe_vlm.step_diagnostic import EmbeddingGradientTap, _full_tensor_sha256


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4)

    def get_input_embeddings(self):
        return self.embedding


class TinyEmbeddingBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TinyLanguage()
        self.head = nn.Linear(4, 1, bias=False)

    def prepare_embeddings(self, input_ids):
        lookup = self.language_model.get_input_embeddings()(input_ids)
        tap = getattr(self, "_deepprobe_step_embedding_gradient_tap", None)
        if tap is not None:
            tap.capture_activation("lookup_output", lookup)
        multimodal = lookup.clone()
        multimodal[:, 0] = 0.25
        if tap is not None:
            tap.capture_activation("multimodal_output", multimodal)
        return multimodal

    def forward(self, input_ids, target):
        return (self.head(self.prepare_embeddings(input_ids)).sum() - target).square()


def _worker(rank: int, world_size: int, rendezvous: str):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank,
        world_size=world_size, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(17)
        model = TinyEmbeddingBoundary()
        reference = TinyEmbeddingBoundary()
        reference.load_state_dict(model.state_dict())
        input_ids = torch.tensor([[rank * 3, rank * 3 + 1, rank * 3 + 2]])
        target = torch.tensor(float(rank + 1))
        reference(input_ids, target).backward()
        local_full = reference.language_model.embedding.weight.grad.detach().clone()
        selected_ids = torch.unique(input_ids.reshape(-1))

        ddp = DistributedDataParallel(model)
        tap = EmbeddingGradientTap(ddp.module, input_ids)
        tap.install()
        try:
            ddp(input_ids, target).backward()
            receipt = tap.summary(sync_gradients=True)
        finally:
            tap.close()

        expected_local_rows = local_full.index_select(0, selected_ids)
        torch.distributed.all_reduce(local_full)
        local_full /= world_size
        expected_reduced_rows = local_full.index_select(0, selected_ids)
        assert torch.count_nonzero(expected_local_rows) > 0
        assert receipt["gradients"]["local_embedding_weight_rows"] == _full_tensor_sha256(
            expected_local_rows), "Leaf hook did not observe the local pre-DDP gradient"
        assert receipt["post_backward_embedding_weight_rows"] == _full_tensor_sha256(
            expected_reduced_rows), "Post-backward gradient did not equal the DDP average"
        assert (receipt["gradients"]["local_embedding_weight_rows"]["sha256"]
                != receipt["post_backward_embedding_weight_rows"]["sha256"])
        assert receipt["post_backward_ddp_synchronized"] is True
    finally:
        torch.distributed.destroy_process_group()


class EmbeddingGradientTapDDPTest(unittest.TestCase):
    @unittest.skipUnless(torch.distributed.is_available()
                         and torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_leaf_hook_precedes_ddp_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "ddp-rendezvous")
            torch.multiprocessing.spawn(_worker, args=(2, rendezvous), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
