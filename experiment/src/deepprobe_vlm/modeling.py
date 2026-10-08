"""Native recurrent LM with bounded SigLIP visual prefix and final-loop SFT.

The backbone performs its own shared-weight recurrence. This wrapper never
approximates recurrence with repeated whole-language-model forward calls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F


OURO_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
SIGLIP_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"


@dataclass
class DeepProbeConfig:
    language_model: str = "ByteDance/Ouro-1.4B"
    language_revision: str = OURO_REVISION
    language_backend: str = "ouro"
    vision_model: str = "google/siglip-so400m-patch14-384"
    vision_revision: str = SIGLIP_REVISION
    num_loops: int = 4
    num_frames: int = 8
    pool_grid: int = 4
    freeze_vision: bool = True
    max_text_tokens: int = 512
    attn_implementation: str = "sdpa"

    def __post_init__(self):
        if self.language_backend not in {"ouro", "causal_lm"}:
            raise ValueError("language_backend must be ouro or causal_lm")
        for key in ("num_loops", "num_frames", "pool_grid", "max_text_tokens"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.language_backend == "causal_lm" and self.num_loops != 1:
            raise ValueError("Ordinary causal_lm comparison must use num_loops=1")
        if self.num_frames > 64 or self.pool_grid > 16:
            raise ValueError("Safety bound: num_frames <=64 and pool_grid <=16")
        if not isinstance(self.freeze_vision, bool):
            raise ValueError("freeze_vision must be Boolean")
        for key in ("language_revision", "vision_revision"):
            if not re.fullmatch(r"[0-9a-f]{40}", getattr(self, key)):
                raise ValueError(f"{key} must be an immutable 40-character commit SHA")

    @classmethod
    def from_dict(cls, value):
        value = value.get("model", value)
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model settings: {sorted(unknown)}")
        return cls(**value)

    @property
    def num_visual_tokens(self):
        return self.num_frames * self.pool_grid ** 2


class SpatialPoolProjector(nn.Module):
    """Learned positive within-bin weights; output retains spatial bin order."""
    def __init__(self, vision_dim, language_dim, grid):
        super().__init__()
        self.grid = grid
        self.gate = nn.Linear(vision_dim, 1)
        self.projection = nn.Sequential(nn.Linear(vision_dim, language_dim), nn.GELU(),
                                        nn.Linear(language_dim, language_dim))

    def forward(self, states):
        # SigLIP patch tokens have no CLS token and form a square spatial grid.
        side = math.isqrt(states.shape[1])
        if side * side != states.shape[1] or self.grid > side:
            raise ValueError("Expected square SigLIP patch grid with pool_grid <= patch grid")
        weights = torch.sigmoid(self.gate(states)).float().clamp_min(1e-6)
        values = (states.float() * weights).transpose(1, 2).reshape(states.shape[0], -1, side, side)
        weights = weights.transpose(1, 2).reshape(states.shape[0], 1, side, side)
        pooled = F.adaptive_avg_pool2d(values, self.grid) / F.adaptive_avg_pool2d(weights, self.grid)
        pooled = pooled.flatten(2).transpose(1, 2).to(self.projection[0].weight.dtype)
        return self.projection(pooled)


class DeepProbeVLM(nn.Module):
    def __init__(self, config: DeepProbeConfig, language_model, vision_model):
        super().__init__()
        self.deep_config = config
        self.language_model = language_model
        self.vision_model = vision_model
        self.config = language_model.config  # HF Trainer introspection only.
        self.projector = SpatialPoolProjector(vision_model.config.hidden_size,
                                              language_model.config.hidden_size, config.pool_grid)
        self.stage = "sft"
        if config.language_backend == "ouro":
            core = getattr(language_model, "model", None)
            if core is None or not hasattr(core, "total_ut_steps"):
                raise TypeError("Expected native Ouro with model.total_ut_steps; refusing fake recurrence")
            language_model.config.total_ut_steps = config.num_loops
            core.total_ut_steps = config.num_loops
        self.set_stage("sft")

    def set_stage(self, stage):
        if stage not in {"alignment", "sft"}:
            raise ValueError("stage must be alignment or sft")
        self.stage = stage
        self.language_model.requires_grad_(stage == "sft")
        self.vision_model.requires_grad_(stage == "sft" and not self.deep_config.freeze_vision)
        self.projector.requires_grad_(True)
        # Fixed-depth loss never uses Ouro's adaptive exit gate. Keep these
        # explicitly frozen, avoiding a falsely advertised gate-training path.
        gate = getattr(getattr(self.language_model, "model", None), "early_exit_gate", None)
        if gate is not None:
            gate.requires_grad_(False)
        # SigLIP's contrastive global-pooling head is not used by patch-token
        # SFT. Freeze it explicitly so DDP has no untracked unused parameters.
        vision_head = getattr(getattr(self.vision_model, "vision_model", None), "head", None)
        if vision_head is not None:
            vision_head.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.vision_model.parameters()):
            self.vision_model.eval()
        if self.stage == "alignment":
            self.language_model.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {"use_reentrant": False}
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        if any(p.requires_grad for p in self.vision_model.parameters()):
            self.vision_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def parameter_counts(self):
        return {name: {"total": sum(p.numel() for p in module.parameters()),
                       "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
                for name, module in (("language", self.language_model), ("vision", self.vision_model),
                                     ("pool_projector", self.projector))}

    def encode_visual(self, pixel_values):
        if pixel_values.ndim != 5 or pixel_values.shape[1] != self.deep_config.num_frames:
            raise ValueError("pixel_values must be [batch, configured_frames, C, H, W]")
        batch, frames = pixel_values.shape[:2]
        pixels = pixel_values.flatten(0, 1).to(next(self.vision_model.parameters()).dtype)
        # Frozen vision allows detaching vision only, never the recurrent LM.
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.vision_model.parameters())):
            states = self.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state
        visual = self.projector(states)
        return visual.reshape(batch, frames * self.deep_config.pool_grid ** 2, -1)

    def prepare_embeddings(self, input_ids, visual_mask, pixel_values):
        if visual_mask.shape != input_ids.shape:
            raise ValueError("visual_mask must match input_ids")
        if not torch.all(visual_mask.sum(-1) == self.deep_config.num_visual_tokens):
            raise ValueError("Exactly configured visual tokens are required per example")
        embeds = self.language_model.get_input_embeddings()(input_ids)
        visual = self.encode_visual(pixel_values).to(embeds.dtype)
        # CopySlices preserves gradients into both the projector and text embeddings.
        embeds = embeds.clone()
        embeds[visual_mask] = visual.flatten(0, 1)
        return embeds

    def _language_forward(self, *, inputs_embeds, attention_mask, use_cache=False, past_key_values=None):
        kwargs = dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=None,
                      use_cache=use_cache, past_key_values=past_key_values, return_dict=True)
        if self.deep_config.language_backend == "ouro":
            kwargs.update(exit_at_step=self.deep_config.num_loops - 1, use_weighted_exit=False)
        return self.language_model(**kwargs)

    def forward(self, input_ids, attention_mask, visual_mask, pixel_values, labels=None, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithPast
        embeds = self.prepare_embeddings(input_ids, visual_mask, pixel_values)
        if embeds.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Visual + text sequence exceeds backbone context limit")
        outputs = self._language_forward(inputs_embeds=embeds, attention_mask=attention_mask)
        loss = None
        if labels is not None:
            if not torch.any(labels[:, 1:] != -100):
                raise ValueError("No supervised answer tokens in batch")
            loss = F.cross_entropy(outputs.logits[:, :-1].float().reshape(-1, outputs.logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=outputs.logits)

    @torch.no_grad()
    def greedy_generate(self, input_ids, attention_mask, visual_mask, pixel_values,
                        *, max_new_tokens=64, eos_token_id=None, use_cache=True):
        """Reference single-example decoding with untouched native per-loop KV.

        Cache reuse is across decoded tokens, never across recurrence depths.
        No-cache mode is an explicit correctness/latency comparison.
        """
        if input_ids.shape[0] != 1 or not torch.all(attention_mask == 1):
            raise ValueError("Reference generation supports one unpadded example")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.eval()
        embeds = self.prepare_embeddings(input_ids, visual_mask, pixel_values)
        if embeds.shape[1] + max_new_tokens > self.config.max_position_embeddings:
            raise ValueError("Requested generation exceeds context limit")
        all_embeds, generated, past = embeds, [], None
        current = embeds
        for _ in range(max_new_tokens):
            outputs = self._language_forward(inputs_embeds=current, attention_mask=attention_mask,
                                             use_cache=use_cache, past_key_values=past)
            token = outputs.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
            if eos_token_id is not None and token.item() == eos_token_id:
                break
            embedding = self.language_model.get_input_embeddings()(token)
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=1)
            if use_cache:
                past = outputs.past_key_values
                if past is None:
                    raise RuntimeError("Backbone did not return a native KV cache")
                current = embedding
            else:
                all_embeds = torch.cat((all_embeds, embedding), dim=1)
                current = all_embeds
        return torch.cat(generated, dim=1)

    @classmethod
    def from_backbones(cls, config, *, torch_dtype=torch.float32, checkpoint=None):
        from transformers import AutoModelForCausalLM, SiglipVisionModel
        root = Path(checkpoint) if checkpoint else None
        language = AutoModelForCausalLM.from_pretrained(
            str(root / "language_model") if root else config.language_model,
            revision=None if root else config.language_revision,
            trust_remote_code=config.language_backend == "ouro", torch_dtype=torch_dtype,
            attn_implementation=config.attn_implementation)
        vision = SiglipVisionModel.from_pretrained(
            str(root / "vision_model") if root else config.vision_model,
            revision=None if root else config.vision_revision, torch_dtype=torch_dtype)
        model = cls(config, language, vision)
        model.projector.to(dtype=torch_dtype)
        if root:
            from safetensors.torch import load_file
            model.projector.load_state_dict(load_file(str(root / "projector.safetensors")))
        return model

    def save_pretrained(self, output_dir, **kwargs):
        from safetensors.torch import save_file
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.language_model.save_pretrained(output / "language_model", safe_serialization=True)
        self.vision_model.save_pretrained(output / "vision_model", safe_serialization=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.projector.state_dict().items()},
                  str(output / "projector.safetensors"))
        (output / "deepprobe_config.json").write_text(json.dumps(asdict(self.deep_config), indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, path, *, torch_dtype=torch.float32):
        config = DeepProbeConfig.from_dict(json.loads((Path(path) / "deepprobe_config.json").read_text()))
        return cls.from_backbones(config, checkpoint=path, torch_dtype=torch_dtype)

    def load_checkpoint_weights(self, path):
        restored = type(self).from_pretrained(path, torch_dtype=next(self.parameters()).dtype)
        if restored.deep_config != self.deep_config:
            raise ValueError("Resume configuration differs from saved checkpoint")
        self.load_state_dict(restored.state_dict(), strict=True)
