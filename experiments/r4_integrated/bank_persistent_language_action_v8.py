"""Bank authenticated complete/partial V8 evidence; never invoke a learner or task."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.r4_integrated.bank_persistent_language_action_v7 import read, sha

DRAFT = True
ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SHA = "66b0bd96be3b9e97c4eb5f9812c2b5977f294fc5900d55109445c0a303e522d0"
PREPARATION = {
    "persistent-language-action-v8-build.launch.json": "20d93cd30dfb0c8a3b2107840b340695674704a42dac562f6e1ac4162d44e1ac",
    "persistent-language-action-v8-build.stdout": "25200273e88c82bf608989adff945a2ef0694f9038bf178215e54788615288e2",
    "persistent-language-action-v8-author.launch.json": "5efbb6c80dbc165298b132984192cac668f5e68f43a858c52ca877b442ce97e3",
    "persistent-language-action-v8-author-check/report.json": "062837fff06dd6065ad2e2da16ccfff15d1bd27255c113ce3595d8c285703ae0",
    "persistent-language-action-v8-author-lf.launch.json": "78736cbc31d2699f5f1e54b5ab919d434e336fc87e134b1715981d7afb9dc3d1",
    "persistent-language-action-v8-author-check-lf/report.json": "c94c586acb8866770b509d850a3d8afcfa42a7928fab1cb1bef92e1407c4e707",
}
SAVED_TOOLS = {
    "experiments/r4_integrated/persistent_language_action_v8_review.py",
    "experiments/r4_integrated/bank_persistent_language_action_v8.py",
}


def preparation(evidence):
    """Preserve the one build and both real author launches, including the failure."""
    directory = ROOT / ".pytest-tmp"
    for name, identity in PREPARATION.items():
        path = directory / name
        if sha(path) != identity:
            raise ValueError("frozen preparation artifact changed: " + name)
        evidence[path.relative_to(ROOT).as_posix()] = identity
    build = read(directory / "persistent-language-action-v8-build.launch.json")
    receipt = read(directory / "persistent-language-action-v8-build.stdout")
    if (build["status"] != "passed" or receipt["sha256"] != MANIFEST_SHA
            or receipt["task_builds"] != 1 or receipt["children"] != 0
            or receipt["model_constructions"] != 0 or receipt["checkpoint_loads"] != 0):
        raise ValueError("pure build provenance changed")
    attempts = []
    for suffix, folder, passed in (("author", "author-check", False),
                                  ("author-lf", "author-check-lf", True)):
        stem = "persistent-language-action-v8-" + suffix
        launch = read(directory / (stem + ".launch.json"))
        report = read(directory / ("persistent-language-action-v8-" + folder + "/report.json"))
        if (launch["status"] != ("passed" if passed else "failed")
                or report["schema"] != "persistent-language-action-author-check-v8"
                or report["manifest_sha256"] != MANIFEST_SHA or report["passed"] is not passed
                or report["children"] != len(report["cases"]) or report["children"] != 27
                or report["canonical"] != 19 or report["broken"] != 8
                or any(report[k] != 0 for k in ("models", "learning_offers", "checkpoint_decodes"))
                or (passed and not all(case["passed"] for case in report["cases"]))):
            raise ValueError("authored validation count/status changed")
        for extension in ("stdout", "stderr"):
            path = directory / (stem + "." + extension)
            evidence[path.relative_to(ROOT).as_posix()] = sha(path)
        attempts.append({"status": launch["status"], "seconds": launch["elapsed_seconds"],
                         "peak_mib": launch["observed_tree_peak_sum_mib"], "children": report["children"],
                         "source_sha256": report["source_sha256"],
                         "case_seconds": sum(c["elapsed_seconds"] for c in report["cases"]),
                         "passed_cases": sum(c["passed"] for c in report["cases"]),
                         "scope": "author-only canonical/broken programs, zero neuronal exposure"})
    return {"pure_builds": 1, "build_seconds": build["elapsed_seconds"],
            "build_peak_mib": build["observed_tree_peak_sum_mib"], "build_children": 0,
            "author_attempts": attempts, "author_children": sum(a["children"] for a in attempts),
            "seconds": build["elapsed_seconds"] + sum(a["seconds"] for a in attempts),
            "peak_mib": max(build["observed_tree_peak_sum_mib"], *(a["peak_mib"] for a in attempts)),
            "failure_scope": "first author harness used platform CRLF stdout; second used the qualified LF bootstrap; both launches retained",
            "budget_scope": "completed preparation cost outside the selected 3600-second runner and 120-second saved audit"}


def bank(run, review_path, run_launch, review_launch, previous_review_launches=()):
    started = time.perf_counter()
    run, review_path, run_launch, review_launch = (
        Path(p).resolve() for p in (run, review_path, run_launch, review_launch))
    review, monitor, audit = read(review_path), read(run_launch), read(review_launch)
    complete = review.get("complete_run") is True
    if (review["schema"] != "persistent-language-action-saved-review-v8"
            or review["status"] != ("passed" if complete else "passed_partial")
            or review["manifest_sha256"] != MANIFEST_SHA or Path(review["run"]).resolve() != run
            or review["run_launch_sha256"] != sha(run_launch) or audit["status"] != "passed"
            or monitor["status"] != ("passed" if complete else "failed")):
        raise ValueError("selected saved run/review status or identity changed")
    command = audit["command"]
    if (command[1:5] != ["-X", "utf8", "-m", "experiments.r4_integrated.persistent_language_action_v8_review"]
            or Path(command[command.index("--run") + 1]).resolve() != run
            or Path(command[command.index("--output") + 1]).resolve() != review_path
            or command[command.index("--manifest-sha") + 1] != MANIFEST_SHA
            or ("--partial" in command) == complete):
        raise ValueError("saved audit monitor command does not bind the review")
    if set(review["saved_tool_sources"]) != SAVED_TOOLS or any(
            sha(ROOT / name) != identity for name, identity in review["saved_tool_sources"].items()):
        raise ValueError("separately frozen saved-only sources changed")
    evidence, previous = {}, []
    for name, identity in review["evidence"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha(path) != identity:
            raise ValueError("reviewed evidence changed: " + name)
        evidence[path.relative_to(ROOT).as_posix()] = identity
    for path in (review_path, run_launch, review_launch, Path(__file__),
                 ROOT / "experiments/r4_integrated/persistent_language_action_v8_review.py"):
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    for previous_path in previous_review_launches:
        previous_path = Path(previous_path).resolve()
        item = read(previous_path)
        argv = item["command"]
        if (item["status"] != "failed" or argv[1:5] != command[1:5]
                or Path(argv[argv.index("--run") + 1]).resolve() != run):
            raise ValueError("previous failed audit belongs to another run")
        artifact = Path(argv[argv.index("--output") + 1]).resolve()
        evidence[previous_path.relative_to(ROOT).as_posix()] = sha(previous_path)
        output_hash = sha(artifact) if artifact.exists() else None
        if output_hash is not None:
            evidence[artifact.relative_to(ROOT).as_posix()] = output_hash
        previous.append({"launch_path": previous_path.relative_to(ROOT).as_posix(),
                         "output_path": artifact.relative_to(ROOT).as_posix(), "output_sha256": output_hash,
                         "seconds": item["elapsed_seconds"], "peak_mib": item["observed_tree_peak_sum_mib"]})
    prepared = preparation(evidence)
    all_audit_seconds = audit["elapsed_seconds"] + sum(r["seconds"] for r in previous)
    run_audit_seconds = monitor["elapsed_seconds"] + all_audit_seconds
    peak = max(monitor["observed_tree_peak_sum_mib"], audit["observed_tree_peak_sum_mib"],
               prepared["peak_mib"], *(r["peak_mib"] for r in previous))
    budget = {"runner_seconds": 3600, "saved_audit_seconds": 120, "peak_mib": 4096,
              "new_artifact_bytes": 26 * 1024**3, "initial_free_bytes": 34 * 1024**3}
    budget_passed = (monitor["elapsed_seconds"] <= budget["runner_seconds"]
                     and all_audit_seconds <= budget["saved_audit_seconds"] and peak <= budget["peak_mib"])
    if complete:
        report = read(run / "report.json")
        if (report["budget"]["runner_seconds"] != 3600 or report["budget"]["audit_seconds"] != 120
                or report["budget"]["peak_mib"] != 4096
                or report["budget"]["new_artifact_bytes"] != budget["new_artifact_bytes"]):
            raise ValueError("reported resource ceiling changed")
    banking_seconds = time.perf_counter() - started
    return {
        "schema": "persistent-language-action-result-bank-v8",
        "status": ("completed_and_independently_reviewed" if budget_passed
                   else "completed_reviewed_over_resource_budget") if complete
                  else "incomplete_resource_or_runtime_failure",
        "complete_run": complete, "source_revision": review["source_revision"],
        "manifest_sha256": MANIFEST_SHA, "metrics": review["metrics"], "exposures": review["exposures"],
        "source_offers": review["source_offers"], "controller_exposures": review["controller_exposures"],
        "saved_costs": review["saved_costs"], "saved_cost_scope": review["cost_scope"],
        "observed_attempts": review["attempt_rows"], "planned_attempts": 192,
        "observed_execution_children": sum(r["executions"] for r in review["exposures"].values()),
        "planned_execution_children": 132, "preparation": prepared,
        "checkpoint": review["checkpoint"], "pending_transaction": review["pending_transaction"],
        "checkpoint_telemetry_seconds": review["checkpoint_telemetry_seconds"],
        "saved_prefix_review": {"passed": True, "scope": review["scope"]},
        "runner": {"status": monitor["status"], "seconds": monitor["elapsed_seconds"],
                   "peak_mib": monitor["observed_tree_peak_sum_mib"], "reason": monitor.get("reason", monitor.get("error"))},
        "saved_review": {"seconds": audit["elapsed_seconds"], "peak_mib": audit["observed_tree_peak_sum_mib"]},
        "previous_failed_reviews": previous, "all_saved_review_seconds": all_audit_seconds,
        "run_and_audit_seconds": run_audit_seconds, "bank_preparation_seconds": banking_seconds,
        "measured_seconds_through_bank_preparation": prepared["seconds"] + run_audit_seconds + banking_seconds,
        "measured_peak_mib": peak, "selected_budget": budget, "complete_resource_budget_passed": complete and budget_passed,
        "saved_tool_sources": review["saved_tool_sources"], "complete_outer_measurement_pending": True,
        "timing_scope": "all build/failed+corrected author/run/failed+corrected audit outer times and this bank's byte-check preparation; banker serialization/process tail requires its outer watchdog",
        "claim": "open authored continuation; valid scalar custody is not a learning success or protected evaluation",
        "checkpoint_decodes_in_bank": 0, "models_or_children_in_bank": 0, "no_scientific_replays": True,
        "evidence": evidence,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "review", "run-launch", "review-launch", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-review-launch", type=Path, action="append", default=[])
    args = parser.parse_args()
    if DRAFT:
        raise SystemExit("DRAFT: root source release required")
    result = bank(args.run, args.review, args.run_launch, args.review_launch, args.previous_review_launch)
    with args.output.open("xb") as handle:
        handle.write((json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "observed_attempts": result["observed_attempts"]}))


if __name__ == "__main__":
    main()
