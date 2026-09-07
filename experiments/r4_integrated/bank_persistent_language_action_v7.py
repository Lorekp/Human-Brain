"""Publish one reviewed complete or partial V7 result, never run/replay a learner."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

DRAFT = False
ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SHA = "c2e1cb6237fdff468c6d413ce62c9dfbf514a607fb1a8af45bdde285c3f4b898"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    with path.open("rb") as handle:
        raw = handle.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("bounded saved JSON exceeded")
    return json.loads(raw)


def bank(run, review_path, run_launch, review_launch, previous_review_launches=()):
    started = time.perf_counter()
    run, review_path, run_launch, review_launch = (
        Path(p).resolve() for p in (run, review_path, run_launch, review_launch)
    )
    review, monitor, audit_monitor = read(review_path), read(run_launch), read(review_launch)
    complete = review.get("complete_run") is True
    if (
        review["schema"] != "persistent-language-action-saved-review-v7"
        or review["status"] != ("passed" if complete else "passed_partial")
        or review["manifest_sha256"] != MANIFEST_SHA
        or Path(review["run"]).resolve() != run
        or review["run_launch_sha256"] != sha(run_launch)
        or audit_monitor["status"] != "passed"
    ):
        raise ValueError("saved review is not the selected complete/partial result")
    command = audit_monitor["command"]
    if (
        command[1:5]
        != ["-X", "utf8", "-m", "experiments.r4_integrated.persistent_language_action_v7_review"]
        or Path(command[command.index("--run") + 1]).resolve() != run
        or Path(command[command.index("--output") + 1]).resolve() != review_path
        or ("--partial" in command) == complete
    ):
        raise ValueError("review monitor command is not bound to this saved artifact")
    if monitor["status"] != ("passed" if complete else "failed"):
        raise ValueError("completion status disagrees with runner watchdog")
    previous_reviews = []
    for previous_path in previous_review_launches:
        previous_path = Path(previous_path).resolve()
        previous = read(previous_path)
        previous_command = previous["command"]
        if (
            previous["status"] != "failed"
            or previous_command[1:5] != command[1:5]
            or Path(previous_command[previous_command.index("--run") + 1]).resolve() != run
        ):
            raise ValueError("previous failed review belongs to another workload")
        previous_output = Path(previous_command[previous_command.index("--output") + 1]).resolve()
        previous_reviews.append(
            {
                "launch_path": previous_path.relative_to(ROOT).as_posix(),
                "launch_sha256": sha(previous_path),
                "output_path": previous_output.relative_to(ROOT).as_posix(),
                "output_sha256": sha(previous_output),
                "seconds": previous["elapsed_seconds"],
                "peak_mib": previous["observed_tree_peak_sum_mib"],
            }
        )
    saved_tools = {
        "experiments/r4_integrated/persistent_language_action_v7_review.py",
        "experiments/r4_integrated/bank_persistent_language_action_v7.py",
    }
    if set(review["saved_tool_sources"]) != saved_tools or any(
        sha(ROOT / name) != identity for name, identity in review["saved_tool_sources"].items()
    ):
        raise ValueError("separately bound saved review/banker source changed")
    evidence = {}
    for name, expected in review["evidence"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha(path) != expected:
            raise ValueError("reviewed evidence changed: " + name)
        evidence[path.relative_to(ROOT).as_posix()] = expected
    for path in (
        review_path,
        run_launch,
        review_launch,
        Path(__file__),
        ROOT / "experiments/r4_integrated/persistent_language_action_v7_review.py",
    ):
        evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    for previous in previous_reviews:
        evidence[previous["launch_path"]] = previous["launch_sha256"]
        evidence[previous["output_path"]] = previous["output_sha256"]
    all_review_seconds = audit_monitor["elapsed_seconds"] + sum(
        x["seconds"] for x in previous_reviews
    )
    run_and_audit_seconds = monitor["elapsed_seconds"] + all_review_seconds
    peak = max(
        monitor["observed_tree_peak_sum_mib"],
        audit_monitor["observed_tree_peak_sum_mib"],
        *(x["peak_mib"] for x in previous_reviews),
    )
    admission = None
    if complete:
        report = read(run / "report.json")
        budget = report["budget"]
        if (
            run_and_audit_seconds > budget["complete_seconds"]
            or peak > budget["peak_mib"]
            or all_review_seconds > budget["reserved_audit_seconds"]
        ):
            raise ValueError("completed run+review exceeds selected complete budget")
        admission = budget
    preparation_seconds = time.perf_counter() - started
    subtotal_seconds = run_and_audit_seconds + preparation_seconds
    if complete and subtotal_seconds > admission["complete_seconds"]:
        raise ValueError("run+review+bank preparation exceeds selected complete budget")
    return {
        "schema": "persistent-language-action-result-bank-v7",
        "status": "completed_and_independently_reviewed"
        if complete
        else "incomplete_resource_or_runtime_failure",
        "source_revision": review["source_revision"],
        "manifest_sha256": MANIFEST_SHA,
        "complete_run": complete,
        "completed_run_audit": {"launched": complete, "passed": complete},
        "saved_prefix_review": {"passed": True, "scope": review["scope"]},
        "metrics": review["metrics"],
        "exposures": review["exposures"],
        "source_offers": review["source_offers"],
        "saved_costs": review.get("saved_costs", {}),
        "saved_cost_scope": review.get("saved_cost_scope"),
        "checkpoint_telemetry_seconds": review.get("checkpoint_telemetry_seconds", {}),
        "observed_attempts": review["attempt_rows"],
        "planned_attempts": 192,
        "checkpoint": review["checkpoint"],
        "pending_transaction": review.get("pending_transaction"),
        "runner": {
            "status": monitor["status"],
            "seconds": monitor["elapsed_seconds"],
            "peak_mib": monitor["observed_tree_peak_sum_mib"],
            "reason": monitor.get("reason"),
        },
        "saved_review": {
            "seconds": audit_monitor["elapsed_seconds"],
            "peak_mib": audit_monitor["observed_tree_peak_sum_mib"],
        },
        "previous_failed_reviews": previous_reviews,
        "all_saved_review_seconds": all_review_seconds,
        "saved_tool_sources": review["saved_tool_sources"],
        "bank_preparation_seconds": preparation_seconds,
        "measured_seconds_through_bank_preparation": subtotal_seconds,
        "complete_outer_measurement_pending": True,
        "timing_scope": "Run and audit outer times plus banker preparation through evidence "
        "rehashing. Bank serialization and process completion require the final bank watchdog; "
        "the final three-process outer total is not yet available inside this artifact.",
        "measured_peak_mib": peak,
        "admitted_budget": admission,
        "claim": "Authored public continuation; valid custody is not a learning success "
        "or protected evaluation.",
        "no_checkpoint_decodes_in_review": True,
        "no_scientific_replays": True,
        "evidence": evidence,
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "review", "run-launch", "review-launch", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-review-launch", type=Path, action="append", default=[])
    args = parser.parse_args()
    if DRAFT:
        raise SystemExit("DRAFT: root source review/release required")
    result = bank(
        args.run, args.review, args.run_launch, args.review_launch, args.previous_review_launch
    )
    with args.output.open("xb") as handle:
        handle.write(
            (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        )
    print(
        json.dumps({"status": result["status"], "observed_attempts": result["observed_attempts"]})
    )


if __name__ == "__main__":
    main()
