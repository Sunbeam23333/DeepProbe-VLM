"""CPU-only contracts for the declared core seeds and no-alignment diagnostic.

These declarations freeze training choices; they do not admit a GPU job or
claim equal compute. Old source/checkpoint contracts are deliberately separate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

SEEDS = (17, 29, 43)
LANGUAGE_REVISIONS = {
    "ByteDance/Ouro-1.4B": "574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
}
VISION_ID = "google/siglip-so400m-patch14-384"
VISION_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
TRAIN_EXAMPLES = 91826
TARGET_STEPS = 718


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declaration(method, seed, language_id):
    return {"schema_version": 1, "method": method, "seed": seed,
            "language_id": language_id, "train_examples": TRAIN_EXAMPLES,
            "target_optimizer_steps": TARGET_STEPS, "selection": "final_step",
            "heldout_selection": False, "equal_compute_claim": False,
            "comparison": "same_sft_stage_budget" if method == "no_alignment" else "independent_core_seed"}


def _source_matches(value, identifier, revision):
    if value == identifier:
        return True
    if not isinstance(value, str):
        return False
    path = Path(value)
    return (path.is_absolute() and ".." not in path.parts and
            path.is_relative_to(Path("/home/jinhaopeng")) and path.name == revision and
            path.parent.name == "snapshots" and
            path.parent.parent.name == "models--" + identifier.replace("/", "--"))


def validate_declaration(raw):
    model, training, protocol = raw.get("model", {}), raw.get("training", {}), raw.get("experiment_protocol")
    require(isinstance(protocol, dict), "Explicit experiment_protocol required by the declared runner")
    method, seed, language_id = protocol.get("method"), training.get("seed"), protocol.get("language_id")
    require(method in {"two_stage", "no_alignment"}, "Unsupported training method")
    require(type(seed) is int and seed in SEEDS, "Declared seed must be 17, 29 or 43")
    require(language_id in LANGUAGE_REVISIONS, "Unsupported declared language backbone")
    require(protocol == declaration(method, seed, language_id), "Declared protocol or budget changed")
    stage = training.get("stage")
    require(stage in {"alignment", "sft"}, "Unsupported stage")
    require(_source_matches(model.get("language_model"), language_id, LANGUAGE_REVISIONS[language_id]) and
            model.get("language_revision") == LANGUAGE_REVISIONS[language_id], "Language initialization revision/path changed")
    require(_source_matches(model.get("vision_model"), VISION_ID, VISION_REVISION) and
            model.get("vision_revision") == VISION_REVISION, "Vision initialization revision/path changed")
    require(model.get("language_backend") == ("ouro" if language_id == "ByteDance/Ouro-1.4B" else "causal_lm"),
            "Backbone implementation differs")
    loops = model.get("num_loops")
    require(type(loops) is int and loops in ((1, 2, 4) if model["language_backend"] == "ouro" else (1,)),
            "Unsupported core depth")
    require(all(type(model.get(k)) is int and model[k] == v for k, v in
                {"num_frames": 8, "pool_grid": 4, "max_text_tokens": 512}.items()), "Core F/G/text protocol differs")
    require(model.get("attn_implementation") == "sdpa" and
            model.get("freeze_vision") is (stage == "alignment"), "Stage parameter groups differ")
    expected = {"formal_training": True, "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 16,
                "num_train_epochs": 1, "max_steps": -1, "bf16": True,
                "gradient_checkpointing": True, "warmup_ratio": 0.03,
                "lr_scheduler_type": "cosine", "eval_strategy": "epoch",
                "dataloader_num_workers": 0, "optim": "adamw_torch", "report_to": "none",
                "learning_rate": 0.001 if stage == "alignment" else 0.00001,
                "weight_decay": 0.0 if stage == "alignment" else 0.01}
    optional = {"save_strategy": "steps", "require_final_resume_checkpoint": True,
                "save_only_model": False, "load_best_model_at_end": False, "fp16": False}
    expected.update(logging_steps=10)
    require(set(training) <= set(expected) | set(optional) | {"seed", "stage", "save_steps", "save_total_limit"},
            "Undeclared TrainingArguments override is forbidden")
    retention = 2 if model["language_backend"] == "ouro" and loops in (1, 2) and stage == "alignment" else 1
    require(type(training.get("save_steps")) is int and training["save_steps"] == 50 and
            type(training.get("save_total_limit")) is int and training["save_total_limit"] == retention,
            "Use the matched baseline's declared recovery cadence and retention")
    matches = lambda observed, expected: observed == expected and (
        type(observed) is bool if type(expected) is bool else type(observed) is not bool)
    require(all(k in training and matches(training[k], v) for k, v in expected.items()) and
            all(k not in training or matches(training[k], v) for k, v in optional.items()),
            "Declared one-epoch stage recipe changed")
    if method == "no_alignment":
        require(stage == "sft" and seed == 17 and loops == 4 and model["language_backend"] == "ouro",
                "No-alignment is only the declared R4/F8/G4/seed17 SFT diagnostic")
    return protocol


def initialization_mode(raw, *, init_checkpoint=None, resume=None, no_alignment=False):
    protocol = validate_declaration(raw)
    require(not (init_checkpoint and resume), "Choose alignment initialization OR exact resume")
    require(no_alignment is (protocol["method"] == "no_alignment"), "Explicit no-alignment flag and protocol must agree")
    if no_alignment:
        require(not init_checkpoint, "No-alignment must never load an alignment initializer")
    elif raw["training"]["stage"] == "alignment":
        require(not init_checkpoint, "Fresh alignment must start from pretrained backbones")
    else:
        require(bool(init_checkpoint or resume), "Ordinary SFT requires matching alignment final or exact resume")
    return "exact_resume" if resume else "matching_alignment_final" if init_checkpoint else "pretrained_backbones_random_connector"


def verify_previous_contract(previous, raw, *, resume):
    current = validate_declaration(raw)
    require(isinstance(previous, dict), "Checkpoint lacks a declared training contract")
    if resume:
        require(previous.get("declared_protocol") == current, "Resume method/seed/budget differs")
        require(previous.get("stage") == raw["training"]["stage"], "Resume stage differs")
    else:
        require(current["method"] == "two_stage" and previous.get("stage") == "alignment" and
                previous.get("declared_protocol") == current, "Initializer must be same-seed declared alignment")
    require(previous.get("training", {}).get("seed") == str(current["seed"]), "Checkpoint seed differs")
    validate_stage_initialization(previous, current)


def tree_identity(root):
    """Hash all offline snapshot bytes, retaining Hub blob symlink identities."""
    root = Path(root)
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "Missing regular offline snapshot")
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        require(not any((Path(directory) / n).is_symlink() for n in dirs), "Snapshot directory symlink forbidden")
        for name in names:
            p = Path(directory) / name
            require(p.is_file() and stat.S_ISREG(p.stat().st_mode), "Unsafe offline model file")
            if p.is_symlink():
                require(p.resolve(strict=True).is_relative_to(Path("/home/jinhaopeng")), "Snapshot file alias escapes personal home")
            files[p.relative_to(root).as_posix()] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    require("config.json" in files and any(p.endswith(".safetensors") for p in files), "Offline snapshot lacks model weights")
    return {"path": str(root), "files": files, "files_sha256": canonical(files)}


def pretrained_identity(raw):
    protocol = validate_declaration(raw)
    return {"language_id": protocol["language_id"],
            "language_revision": raw["model"]["language_revision"],
            "vision_id": VISION_ID, "vision_revision": VISION_REVISION,
            "language": tree_identity(raw["model"]["language_model"]),
            "vision": tree_identity(raw["model"]["vision_model"])}


def make_initialization(raw, mode, *, backbones=None, projector_sha256=None, previous=None, checkpoint_provenance_sha256=None):
    protocol = validate_declaration(raw)
    if mode == "exact_resume":
        verify_previous_contract(previous, raw, resume=True)
        initial = previous.get("initialization")
        validate_initialization(initial, protocol)
        return initial
    require(isinstance(projector_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", projector_sha256),
            "Actual initial connector tensor SHA required")
    initial = {"schema_version": 1, "kind": mode, "seed": protocol["seed"], "method": protocol["method"],
               "initial_connector_state_sha256": projector_sha256}
    if mode == "pretrained_backbones_random_connector":
        require(isinstance(backbones, dict) and backbones.get("language_id") == protocol["language_id"],
                "Full fixed-backbone initialization evidence required")
        initial.update(pretrained_backbones=backbones, connector="seeded_fresh_model_constructor")
    else:
        require(mode == "matching_alignment_final", "Unknown initialization mode")
        verify_previous_contract(previous, raw, resume=False)
        require(isinstance(checkpoint_provenance_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", checkpoint_provenance_sha256),
                "Alignment provenance SHA required")
        initial.update(alignment_provenance_sha256=checkpoint_provenance_sha256,
                       alignment_initialization=previous["initialization"], connector="matching_alignment_final")
    validate_initialization(initial, protocol)
    return initial


def validate_initialization(initial, protocol):
    valid_sha = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
    require(isinstance(initial, dict) and initial.get("schema_version") == 1 and
            type(initial.get("seed")) is int and initial["seed"] == protocol["seed"] and
            initial.get("method") == protocol["method"] and
            valid_sha(initial.get("initial_connector_state_sha256")), "Incomplete initialization identity")
    if initial.get("kind") == "matching_alignment_final":
        require(protocol["method"] == "two_stage" and initial.get("connector") == "matching_alignment_final" and
                valid_sha(initial.get("alignment_provenance_sha256")), "Invalid alignment ancestry")
        parent = initial.get("alignment_initialization")
        require(isinstance(parent, dict) and parent.get("kind") == "pretrained_backbones_random_connector",
                "Alignment must itself start from fixed pretrained backbones/random connector")
        validate_initialization(parent, protocol)
        return
    require(initial.get("kind") == "pretrained_backbones_random_connector" and
            initial.get("connector") == "seeded_fresh_model_constructor", "Invalid fresh initialization kind")
    backbones = initial.get("pretrained_backbones", {})
    require(backbones.get("language_id") == protocol["language_id"] and
            backbones.get("language_revision") == LANGUAGE_REVISIONS[protocol["language_id"]] and
            backbones.get("vision_id") == VISION_ID and backbones.get("vision_revision") == VISION_REVISION,
            "Pretrained backbone identity differs")
    for name, identifier, revision in (("language", protocol["language_id"], LANGUAGE_REVISIONS[protocol["language_id"]]),
                                       ("vision", VISION_ID, VISION_REVISION)):
        row = backbones.get(name, {})
        files = row.get("files")
        require(isinstance(files, dict) and "config.json" in files and
                any(key.endswith(".safetensors") for key in files) and
                row.get("files_sha256") == canonical(files) and
                isinstance(row.get("path"), str) and Path(row["path"]).is_absolute() and
                _source_matches(row.get("path"), identifier, revision), "Missing full pretrained-file identity")
        for relative, item in files.items():
            path = Path(relative)
            require(relative and not path.is_absolute() and ".." not in path.parts and
                    isinstance(item, dict) and type(item.get("bytes")) is int and item["bytes"] >= 0 and
                    valid_sha(item.get("sha256")), "Invalid pretrained-file digest")


def completion_identity(contract, state):
    protocol = contract.get("declared_protocol")
    require(isinstance(protocol, dict) and protocol.get("target_optimizer_steps") == TARGET_STEPS,
            "Completion lacks the frozen optimizer-step target")
    require(protocol.get("method") in {"two_stage", "no_alignment"} and
            type(protocol.get("seed")) is int and protocol["seed"] in SEEDS and protocol.get("language_id") in LANGUAGE_REVISIONS and
            protocol == declaration(protocol.get("method"), protocol["seed"], protocol["language_id"]),
            "Completion has an undeclared protocol")
    require(type(state.get("global_step")) is int and state["global_step"] == TARGET_STEPS and
            type(state.get("max_steps")) is int and state["max_steps"] == TARGET_STEPS,
            "Training stopped short of the frozen optimizer-step target")
    require(contract.get("training", {}).get("seed") == str(protocol["seed"]), "Completion seed mismatch")
    initial = contract.get("initialization")
    validate_stage_initialization(contract, protocol)
    return {"declared_protocol": protocol, "seed": protocol["seed"], "method": protocol["method"],
            "initialization": initial, "target_optimizer_steps": TARGET_STEPS,
            "two_stage_training_claim": False}


def validate_stage_initialization(contract, protocol):
    stage = contract.get("stage")
    require(stage in {"alignment", "sft"} and
            (protocol["method"] != "no_alignment" or stage == "sft"), "Undeclared initialization stage")
    validate_initialization(contract.get("initialization"), protocol)
    expected_kind = ("matching_alignment_final" if stage == "sft" and
                     protocol["method"] == "two_stage" else "pretrained_backbones_random_connector")
    require(contract["initialization"]["kind"] == expected_kind,
            "Checkpoint initialization differs from its declared stage")


def verify_alignment_completion(path, contract):
    """Both entry points require a completed native final, never a partial checkpoint."""
    path = Path(path)
    require(path.name == "final" and contract.get("stage") == "alignment",
            "Initializer must be the completed alignment final")
    parent = path.parent
    names = ("FORMAL_TRAIN_COMPLETE.json", "run_manifest.json", "trainer_state.json")
    for name in names:
        require((parent / name).is_file() and not (parent / name).is_symlink(),
                "Alignment final lacks regular completed-stage evidence")
    complete, manifest, state = [json.loads((parent / name).read_text()) for name in names]
    identity = completion_identity(contract, state)
    require(manifest.get("training_contract") == contract and
            complete.get("status") == "formal_training_complete" and
            complete.get("run_manifest_sha256") == sha(parent / "run_manifest.json") and
            complete.get("trainer_state_sha256") == sha(parent / "trainer_state.json") and
            all(complete.get(k) == v for k, v in identity.items()),
            "Alignment final lacks completed declared-stage evidence")
    return identity


def staging_bytes(roots):
    """Account both staging roots; fail closed on broken/external directory links."""
    roots = [Path(p).resolve(strict=True) for p in roots]
    require(len(roots) == 2 and all(a != b and not a.is_relative_to(b)
            for i, a in enumerate(roots) for j, b in enumerate(roots) if i != j), "Two distinct staging roots required")
    totals, regular, referenced = {}, set(), set()
    def fail(exc):
        raise exc
    for root in roots:
        total = 0
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=fail):
            for name in list(dirs):
                p = Path(directory) / name
                if p.is_symlink():
                    target = p.resolve(strict=True)
                    require(target.is_dir() and any(target.is_relative_to(r) for r in roots), "External directory link")
                    dirs.remove(name)
            for name in names:
                p = Path(directory) / name
                target = p.resolve(strict=True)
                require(stat.S_ISREG(target.stat().st_mode), "Nonregular staging payload")
                if p.is_symlink() and any(target.is_relative_to(r) for r in roots):
                    referenced.add(target)
                else:
                    total += target.stat().st_size
                    if not p.is_symlink():
                        regular.add(target)
        totals[str(root)] = total
    require(referenced <= regular, "Internal staging alias target not accounted")
    return totals
