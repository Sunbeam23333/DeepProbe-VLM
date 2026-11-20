"""One entry point for remaining-suite preparation, admission and exact launch.

Default execution only reports a plan. Explicit preparation builds original-video
caches in OSS, imports independently reviewed history, freezes bindings, and runs
CPU admission. GPU launch additionally requires --submit or --dry-run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import uuid

from . import experiment_suite as suite
from . import suite_bindings as bindings
from . import suite_runner as runner


def save(path, value):
    path = runner.safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        runner.require(runner.read(path) == value, "Existing workflow artifact differs; use a new output root")
    else:
        runner.write_once(path, value)
    return runner.artifact(path)


def source_catalog(receipt, store, archive_audit=None):
    """Resolve only originals already SHA-bound by the accepted decode receipt."""
    from . import suite_media as media
    runner.require(receipt.get("complete") is True and receipt.get("entries"), "Original decode receipt incomplete")
    archives, receipt_shas = {}, {}
    for entry in receipt["entries"]:
        key, etag = entry.get("archive_key"), entry.get("archive_etag", entry.get("oss_archive_etag"))
        if not key or not etag:
            candidates = [row for row in (archive_audit or {}).get("checked_archives", [])
                          if row.get("source_sha256") == entry["archive_sha256"]
                          and row.get("key", "").endswith("/" + entry.get("archive_path", ""))]
            runner.require(len(candidates) == 1, "Legacy decode entry needs its exact pinned archive audit mapping")
            match = candidates[0]
            runner.require(not key or key == match["key"], "Legacy archive key changed")
            runner.require(not etag or etag == match["etag"], "Legacy archive ETag changed")
            key, etag = match["key"], match["etag"]
            runner.require(isinstance(match.get("receipt_sha256"), str)
                           and runner.SHA.fullmatch(match["receipt_sha256"]), "Legacy archive audit lacks its receipt SHA")
            receipt_shas[key] = match["receipt_sha256"]
        identity = (entry["archive_sha256"], etag)
        runner.require(key not in archives or archives[key] == identity, "Conflicting original archive identities")
        archives[key] = identity
    result = []
    for key, (sha, etag) in sorted(archives.items()):
        head = store.head(key)
        runner.require(head["etag"] == etag, "Original archive ETag changed")
        raw = store.get(key + ".CACHE_VERIFIED.json")
        runner.require(raw is not None, "Original archive lacks its complete GET/SHA receipt")
        if key in receipt_shas:
            runner.require(media.digest(raw) == receipt_shas[key], "Legacy original archive verification receipt changed")
        row = {"key": key, "sha256": sha, "etag": etag, "size_bytes": head["size_bytes"],
               "receipt": {"key": key + ".CACHE_VERIFIED.json", "size_bytes": len(raw), "sha256": media.digest(raw)}}
        media.verify_archive(store, row)
        result.append(row)
    return result


def bind_legacy_archive_identities(receipt, catalog):
    """Add only archive locations/ETags authenticated by SHA and the source audit."""
    result = deepcopy(receipt)
    for entry in result["entries"]:
        matches = [row for row in catalog if row["sha256"] == entry["archive_sha256"]
                   and (row["key"] == entry["archive_key"] if entry.get("archive_key")
                        else row["key"].endswith("/" + entry.get("archive_path", "")))]
        runner.require(len(matches) == 1, "Original archive mapping is ambiguous")
        row = matches[0]
        observed_etag = entry.get("archive_etag", entry.get("oss_archive_etag"))
        runner.require(not observed_etag or observed_etag == row["etag"], "Original archive ETag differs")
        entry.update(archive_key=row["key"], archive_etag=row["etag"])
    return result


def benchmark_resume_mode(output, prefix, store, *, resume_requested):
    """A suite resume may contain both interrupted caches and untouched shapes."""
    output = runner.safe(output)
    if (output / "BUILD_PLAN.json").is_file():
        runner.require(resume_requested, "Existing cache plan requires explicit asset resume")
        return True
    # The real OSS adapter yields keys lazily. A generator is truthy even when
    # its complete listing is empty; consume it before deciding ownership.
    runner.require(not output.exists() and not list(store.keys(prefix)),
                   "Cache objects or output exist without a local build plan; reconcile exact ownership")
    return False


def prepare_benchmark(environment, request, *, resume=False):
    """Build a real F4/F8/F16 cache and hand its receipt to lazy node hydration."""
    from . import suite_media as media
    from .suite_storage import budget_for
    required = {"benchmark", "frames", "source_manifest", "accepted_decode_receipt", "output_dir", "oss_prefix",
                "media_root", "max_png_bytes", "metadata_reserve_bytes"}
    runner.require(isinstance(request, dict) and required <= set(request) <= required | {"archive_audit"}, "Invalid benchmark preparation request")
    runner.require(request["benchmark"] in suite.BENCHMARKS and request["frames"] in {4, 8, 16}, "Unsupported benchmark shape")
    for key in ("source_manifest", "accepted_decode_receipt"):
        runner.verify_pin(request[key])
    cap = request["metadata_reserve_bytes"]
    runner.require(type(cap) is int and cap >= 64 * 1024 * 1024, "Explicit bounded cache metadata reserve required")
    output = runner.safe(request["output_dir"])
    roots = [runner.safe(path) for path in environment["storage"]["roots"]]
    for path in (output, output.with_name(output.name + "-inputs"), runner.safe(request["media_root"])):
        runner.require(any(path != root and path.is_relative_to(root) for root in roots), "Cache output escapes accounted home roots")
    budget = budget_for(environment)
    owner = {"suite_id": environment["suite_id"], "operation": "frame-cache", "prefix": request["oss_prefix"]}
    token = budget.reserve(owner, home_bytes=2 * cap, oss_bytes=request["max_png_bytes"] + cap)
    try:
        store = media.OSSStore(environment["tools"]["oss_config"], allowed_prefix=request["oss_prefix"])
        cache_resume = benchmark_resume_mode(output, request["oss_prefix"], store, resume_requested=resume)
        original = runner.read(request["accepted_decode_receipt"]["path"])
        audit = None
        if request.get("archive_audit"):
            runner.verify_pin(request["archive_audit"]); audit = runner.read(request["archive_audit"]["path"])
        catalog = source_catalog(original, store, audit)
        normalized = bind_legacy_archive_identities(original, catalog)
        manifest_raw = Path(request["source_manifest"]["path"]).read_bytes()
        source = media.create_source_binding(manifest_raw,
            media.encode(normalized), catalog, store=store)
        inputs = output.with_name(output.name + "-inputs")
        normalized_pin = save(inputs / "ARCHIVE_BOUND_SOURCE_RECEIPT.json", normalized)
        source["normalization_evidence"] = {"original_decode_receipt": request["accepted_decode_receipt"],
            "archive_audit": request.get("archive_audit"), "archive_bound_receipt": normalized_pin}
        source_pin = save(inputs / "SOURCE_BINDING.json", source)
        save(inputs / "ARCHIVE_CATALOG.json", catalog)
        # Metadata is part of both budgets; the PNG cap alone is insufficient.
        class LimitedStore:
            def __init__(self): self.metadata_bytes = 0
            def __getattr__(self, name): return getattr(store, name)
            def put_verified(self, key, raw):
                if not key.endswith(".png"):
                    self.metadata_bytes += len(raw)
                    runner.require(self.metadata_bytes <= cap, "Cache metadata exceeds reserved OSS bytes")
                return store.put_verified(key, raw)
        completed = media.build_frame_cache(request["source_manifest"]["path"], source,
            frames=request["frames"], output_dir_new=output, oss_prefix_new=request["oss_prefix"],
            max_bytes=request["max_png_bytes"], store=LimitedStore(), resume=cache_resume)
        runner.require(sum(p.stat().st_size for root in (inputs, output) for p in root.iterdir() if p.is_file()) <= 2 * cap,
                       "Cache metadata exceeds reserved home bytes")
        asset = media.cache_asset_binding(output, source_manifest=request["source_manifest"]["path"],
            source_binding=source_pin["path"], media_root=request["media_root"], max_bytes=request["max_png_bytes"])
        # This validates the full original→question→frame mapping. Actual PNG
        # node hydration remains just-in-time inside suite_actions.
        media._bound_cache_inventory(asset["cache"])
        return asset, completed
    finally:
        # All OSS calls above are synchronous; no upload worker survives return.
        budget.release(token, quiescent=lambda observed: {"owner": observed, "active_writers": [], "active_uploads": []})


def workflow_report(plan, candidate):
    env = candidate["environment"]
    report = bindings.remaining_summary(plan, env)
    report.update(status="SUITE_WORKFLOW_PLANNED_NOT_EXECUTED", production_admitted=False,
                  benchmark_preparations=[{"benchmark": r["benchmark"], "frames": r["frames"],
                    "source_manifest": r["source_manifest"], "output_prefix": r["oss_prefix"],
                    "maximum_new_oss_bytes": r["max_png_bytes"] + r["metadata_reserve_bytes"],
                    "maximum_local_metadata_bytes": 2 * r["metadata_reserve_bytes"],
                    "local_video_payload_bytes": 0} for r in candidate.get("benchmark_preparations", [])],
                  history_import_tasks=sorted(candidate.get("history_imports", {})),
                  recovery_preparation_tasks=[{"task_id": r["task_id"], "operation": r["operation"],
                    "entry": r["entry"], "review": r["review"]} for r in candidate.get("recovery_preparations", [])],
                  history_preparation_tasks=[{"task_id": r["task_id"], "operation": r["operation"],
                    "entry": r["entry"], "review": r["review"]} for r in candidate.get("history_preparations", [])],
                  observed_recovery_candidates=deepcopy(candidate.get("recovery_candidates", {})),
                  storage=deepcopy(env["storage"]), execution_profile=deepcopy(env.get("execution_profile")))
    for key in ("capacity_snapshot", "capacity_recovery_plan"):
        if candidate.get(key):
            runner.verify_pin(candidate[key])
            report[key] = deepcopy(candidate[key])
    if candidate.get("capacity_snapshot"):
        observed = runner.read(candidate["capacity_snapshot"]["path"])
        report["observed_capacity_not_fresh_admission"] = {
            "home": {k: observed["home"][k] for k in ("bytes", "files") if k in observed["home"]},
            "oss": {k: observed["oss"][k] for k in ("bytes", "objects", "accounting_scope", "legacy_version_bytes_unknown") if k in observed["oss"]},
            "observed_utc": observed.get("finished_utc", observed.get("utc")),
            "fresh_inventory_required_before_preparation": True}
    return report


def preparation_capacity(environment, recovery_plan=None):
    """Produce an actionable fresh gate before any asset transfer starts."""
    from .suite_storage import budget_for, capacity_violations
    snapshot = budget_for(environment).snapshot()
    blocked = capacity_violations(snapshot, environment['storage'])
    return {"status": "SUITE_PREPARATION_CAPACITY_GATED" if blocked else "SUITE_PREPARATION_CAPACITY_AVAILABLE",
            "fresh_snapshot": snapshot, "blocked_domains": blocked,
            "recovery_plan": deepcopy(recovery_plan), "automatic_deletion": False,
            "next_action": "Review and execute only authorized exact retirement/migration plan, then rerun fresh admission" if blocked
                           else "Each subsequent operation must still reserve its own additional peak",
            "gpu_submitted": False}


def prepare_workflow(plan, candidate, output, *, prepare_assets=False, resume_assets=False, cpu_admission=False):
    """Prepare every dependency before scheduling; all current gates remain real."""
    output = runner.safe(output)
    runner.require(candidate.get("schema_version") == 1 and isinstance(candidate.get("environment"), dict), "Invalid workflow candidate")
    env = deepcopy(candidate["environment"])
    report = workflow_report(plan, candidate)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "WORKFLOW_REQUEST.json", candidate)
    if not prepare_assets:
        save(output / "PLAN_REPORT.json", report)
        return report
    # Admission is a fresh observation. Keep its timestamps/evidence append-only
    # while preserving identical candidate/history/source/binding artifacts.
    attempt = output / "attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    if env["storage"].get("schema_version") == 2:
        capacity = preparation_capacity(env, candidate.get("capacity_recovery_plan"))
        capacity_pin = save(attempt / "CAPACITY_GATE.json", capacity)
        if capacity["blocked_domains"]:
            result = {"status": "SUITE_WORKFLOW_CAPACITY_GATED", "capacity_gate": capacity_pin,
                      "capacity_recovery_plan": candidate.get("capacity_recovery_plan"),
                      "cpu_admitted": False, "gpu_submitted": False, "asset_preparation_started": False}
            save(attempt / "WORKFLOW_GATED.json", result)
            return result
    if env.get("modelscope_token_file"):
        bindings.load_modelscope_token_file(env["modelscope_token_file"])
    for request in [*candidate.get("recovery_preparations", []), *candidate.get("history_preparations", [])]:
        from .suite_recovery import prepare_recovery
        tid = request["task_id"]
        runner.require(tid not in env["history"] and tid not in candidate.get("history_imports", {}),
                       "Recovery task already has a different history import")
        env["history"][tid] = prepare_recovery(plan, env, request, resume=resume_assets)
    for tid, spec in candidate.get("history_imports", {}).items():
        runner.verify_pin(spec["entry"]); runner.verify_pin(spec["review"])
        env["history"][tid] = bindings.import_history_entry(plan, tid, runner.read(spec["entry"]["path"]),
            spec["review"], output / "history" / (tid + ".json"))
    remaining = bindings.remaining_summary(plan, env)
    remaining_pin = save(output / "REMAINING.json", remaining)
    if remaining["blocked_history_tasks"]:
        result = {"status": "SUITE_WORKFLOW_HISTORY_GATED", "remaining": remaining_pin,
                  "blocked_history_tasks": remaining["blocked_history_tasks"],
                  "cpu_admitted": False, "gpu_submitted": False, "benchmark_preparation_started": False}
        save(attempt / "WORKFLOW_GATED.json", result)
        return result
    cache_results = []
    for request in candidate.get("benchmark_preparations", []):
        asset, complete = prepare_benchmark(env, request, resume=resume_assets)
        env["assets"]["benchmarks"].setdefault(request["benchmark"], {})[str(request["frames"])] = asset
        cache_results.append(complete)
    # Freeze the source actually used for new work. Resumed historical work
    # keeps its independently pinned old source and recipe in the history entry.
    from .formal_integrity import source_inventory
    source_root = Path(env["experiment_root"])
    files = source_inventory(source_root)
    manifest = save(output / "P1_SOURCE_MANIFEST.json", {"schema_version": 1, "purpose": "DEEPPROBE_FROZEN_P1_SOURCE", "files": files})
    env["source_files"] = [runner.artifact(source_root / name) for name in sorted(files)]
    env["assets"]["training"]["frozen_source_manifest"] = manifest
    development = env["assets"]["development"]
    if "source_metadata" not in development:
        runner.verify_pin(development["manifest"])
        runner.require(development["manifest"]["sha256"] == env["assets"]["training"]["eval_manifest"]["sha256"],
                       "Automatic development selection must use the frozen training-validation manifest")
        development["source_metadata"] = save(output / "DEVELOPMENT_SOURCE.json", {
            "scope": "development", "manifest_sha256": development["manifest"]["sha256"],
            "quality_score_based_selection": False,
            "source_description": "Frozen source-disjoint training validation split; deterministic benchmark_latency.select_records, no held-out selection"})
    plan_pin = save(output / "PLAN.json", plan)
    environment_pin = save(output / "ENVIRONMENT.json", env)
    binding = bindings.build_binding(plan, env, plan_path=plan_pin["path"], plan_sha256=plan_pin["sha256"],
        environment_path=environment_pin["path"], environment_sha256=environment_pin["sha256"], config_root=source_root)
    binding_pin = save(output / "BINDING.json", binding)
    result = {"status": "SUITE_WORKFLOW_BOUND_NOT_ADMITTED", "plan": plan_pin, "environment": environment_pin,
              "binding": binding_pin, "cache_preparations": cache_results, "gpu_submitted": False,
              "state_root": candidate["state_root"], "cpu_admitted": False}
    if cpu_admission:
        from .suite_preflight import inspect
        preflight = inspect(plan, binding, env, plan_sha256=plan_pin["sha256"], binding_sha256=binding_pin["sha256"],
            environment_sha256=environment_pin["sha256"], state_root=candidate["state_root"], prepare_media=False)
        result.update(preflight=save(attempt / "CPU_PREFLIGHT.json", preflight), cpu_admitted=preflight["status"] == "SUITE_CPU_ADMITTED")
    template_pin = candidate.get("job_template")
    if "job_template_document" in candidate:
        runner.require(not template_pin, "Choose one pinned worker template source")
        runner.verify_pin(candidate["job_template_source"])
        template_pin = save(output / "JOB_TEMPLATE.json", candidate["job_template_document"])
        result["template_source"] = candidate["job_template_source"]
    if template_pin:
        runner.verify_pin(template_pin)
        from .suite_fleet_launch import prepare_worker_jobs
        result["jobs"] = prepare_worker_jobs(runner.read(template_pin["path"]), env,
            plan_pin, binding_pin, candidate["state_root"], output / "jobs")
    result["attempt_root"] = str(attempt)
    save(attempt / "WORKFLOW_PREPARED.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--plan", type=Path, default=suite.DEFAULT_PLAN)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--resume-assets", action="store_true")
    parser.add_argument("--cpu-admission", action="store_true")
    parser.add_argument("--modelscope-token-file", type=Path, help="Private 0600 file; only its path is stored, never the token")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--dry-run", action="store_true")
    choice.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    args.candidate, args.plan, args.output = (p.absolute() for p in (args.candidate, args.plan, args.output))
    runner.verify_pin({"path": str(args.candidate), "sha256": args.candidate_sha256})
    runner.require(not args.resume_assets or args.prepare_assets, "Asset resume requires explicit preparation")
    runner.require(not (args.dry_run or args.submit) or (args.prepare_assets and args.cpu_admission),
                   "Launch requires asset preparation and fresh CPU admission in the same workflow")
    plan = suite.read_plan(args.plan)
    candidate = runner.read(args.candidate)
    if args.modelscope_token_file:
        candidate["environment"]["modelscope_token_file"] = str(runner.safe(args.modelscope_token_file))
    result = prepare_workflow(plan, candidate, args.output,
        prepare_assets=args.prepare_assets, resume_assets=args.resume_assets, cpu_admission=args.cpu_admission)
    if args.dry_run or args.submit:
        if not result.get("cpu_admitted") or not result.get("jobs"):
            print(json.dumps(result, indent=2))
            return 2
        from .suite_fleet_launch import FleetLauncher
        launched = FleetLauncher(result["plan"], result["binding"], result["environment"], result["jobs"],
            result["preflight"], state_root=result["state_root"], output=args.output / "launch").run(
                "submit" if args.submit else "dry-run")
        result["launch"] = launched
    print(json.dumps(result, indent=2))
    unsuccessful = result.get("status") in {"SUITE_WORKFLOW_CAPACITY_GATED", "SUITE_WORKFLOW_HISTORY_GATED"}
    unsuccessful |= args.cpu_admission and not result.get("cpu_admitted")
    unsuccessful |= result.get("launch", {}).get("status") in {"SUBMISSION_OUTCOME_UNKNOWN", "FLEET_WORKER_TERMINATED"}
    return 2 if unsuccessful else 0


if __name__ == "__main__":
    raise SystemExit(main())
