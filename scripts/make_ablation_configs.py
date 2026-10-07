#!/usr/bin/env python3
"""Generate explicit one-factor training configs; does not launch any jobs."""
import argparse
import copy
import json
from pathlib import Path

from deepprobe_vlm.modeling import DeepProbeConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("configs/train/ouro_sft.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty; refusing to overwrite configurations")
    base = json.loads(args.base.read_text())
    DeepProbeConfig.from_dict(base)
    loops = [1, 2, 4] if base["model"].get("language_backend", "ouro") == "ouro" else [1]
    variants = [("loops", "num_loops", v) for v in loops]
    variants += [("frames", "num_frames", v) for v in [4, 8, 16]]
    variants += [("grid", "pool_grid", v) for v in [2, 4, 8]]
    variants += [("freeze_vision", "freeze_vision", v) for v in [True, False]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, key, value in variants:
        for seed in [17, 29, 43]:
            config = copy.deepcopy(base)
            config["model"][key] = value
            config["training"]["seed"] = seed
            DeepProbeConfig.from_dict(config)
            path = args.output_dir / f"{label}_{str(value).lower()}_seed{seed}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(variants) * 3} configs. Duplicate baseline settings across axes are intentional.")
    print("Train each depth separately; changing depth only at inference is a different experiment.")


if __name__ == "__main__":
    main()
