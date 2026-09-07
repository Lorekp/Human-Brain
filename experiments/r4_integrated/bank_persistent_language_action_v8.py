"""Bank only the corrected fresh V8 session, retaining the first launch's costs."""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.r4_integrated.bank_persistent_language_action_v7 import read, sha

DRAFT = False  # Root-selected saved-only CLI; no task or checkpoint decoder is invoked.
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
CORRECTION_EVIDENCE = {
    "benchmarks/persistent_language_action_v8_initialization_failure.result.json": "0882b79cad1f89448989e65bf5a029f57878d4a05b28c1b1f5bf368e392a9da8",
    ".pytest-tmp/persistent-language-action-v8.launch.json": "757fea5a26f0094a0f4bfc8fd0b10e73581c3ab6d7fe94d3c7d13176540e6045",
    ".pytest-tmp/persistent-language-action-v8-review.launch.json": "61a3ea7dfea5e69354c6945231d9f0e3b52b7f834602e51e0916f11b8651cc22",
    ".pytest-tmp/persistent-language-action-v8-failure-bank.launch.json": "e42b71b512ce540b52ff5b817861957a3b461d1b87849990c3c237f6ad21ea59",
    ".pytest-tmp/recurrent-v8-serialization.launch.json": "bd60d0590ef65bdd368a4576beec6c79dbd350895140f90afb95a2eaeec82594",
    ".pytest-tmp/recurrent-v8-serialization.junit.xml": "b9a504b862a052772349be5fb45121eb83d8bae37fd8a63eb9aa278bbd13c2c6",
    ".pytest-tmp/recurrent-v8-serialization-tests/test_recurrent_v8_nested_recei0/v8-receipt-serialization-passed.json": "2ba2683c1d7770383df3fb87e404c2d5e52363dc55968d4fcd96effa1af8582f",
}


def correction_history(evidence):
    """Authenticate prior failure and ordinary QA; do not rebank their old sources."""
    for name, identity in CORRECTION_EVIDENCE.items():
        if sha(ROOT / name) != identity:
            raise ValueError("frozen correction evidence changed: " + name)
        evidence[name] = identity
    old = read(ROOT / "benchmarks/persistent_language_action_v8_initialization_failure.result.json")
    launches = {name: read(ROOT / (".pytest-tmp/" + stem + ".launch.json")) for name, stem in (
        ("failed_runner", "persistent-language-action-v8"),
        ("successful_partial_review", "persistent-language-action-v8-review"),
        ("first_failure_bank", "persistent-language-action-v8-failure-bank"),
        ("serialization_qa", "recurrent-v8-serialization"))}
    if (old["schema"] != "persistent-language-action-result-bank-v8" or old["complete_run"] is not False
            or old["manifest_sha256"] != MANIFEST_SHA or old["observed_attempts"] != 0
            or old["checkpoint"]["head"]["sequence"] != 0 or old["checkpoint"]["boundaries"] != 1
            or not old["checkpoint"]["dirty"]
            or any(value for row in old["exposures"].values() for value in row.values())
            or any(old["source_offers"].values())
            or launches["failed_runner"]["status"] != "failed"
            or any(launches[name]["status"] != "passed" for name in launches if name != "failed_runner")
            or old["runner"]["seconds"] != launches["failed_runner"]["elapsed_seconds"]
            or old["saved_review"]["seconds"] != launches["successful_partial_review"]["elapsed_seconds"]):
        raise ValueError("prior failed-launch custody or costs changed")
    failure_name = ".pytest-tmp/persistent-language-action-v8-run/failure.json"
    manifest_name = ".pytest-tmp/persistent-language-action-v8-run/manifest.json"
    for name in (failure_name, manifest_name):
        if sha(ROOT / name) != old["evidence"][name]:
            raise ValueError("original failure/manifest evidence changed")
        evidence[name] = old["evidence"][name]
    failure, manifest = read(ROOT / failure_name), read(ROOT / manifest_name)
    if (failure["type"] != "TypeError" or "RecurrentContentSourceFeedback" not in failure["error"]
            or manifest["episodes"][0]["append_document_ids"] != ["N36"]):
        raise ValueError("first source serialization failure identity changed")
    first = next(d for d in manifest["documents"] if d["document_id"] == "N36")
    inferred_offers = len(bytes.fromhex(first["text_hex"])) + 1
    if inferred_offers != 89:
        raise ValueError("first failed source exposure inference changed")
    qa_path = ROOT / ".pytest-tmp/recurrent-v8-serialization-tests/test_recurrent_v8_nested_recei0/v8-receipt-serialization-passed.json"
    qa = read(qa_path)
    if (qa["status"] != "passed" or qa["serializer_calls"] != 1 or qa["journal_appends"] != 1
            or qa["authored_tensors"] != 4
            or any(qa[k] != 0 for k in ("models", "source_offers", "teacher_offers", "children", "checkpoint_decodes"))):
        raise ValueError("ordinary serialization QA exposure changed")
    suite = ET.parse(ROOT / ".pytest-tmp/recurrent-v8-serialization.junit.xml").getroot().find("testsuite")
    if (suite is None or any(suite.get(k) != v for k, v in (
            ("tests", "1"), ("failures", "0"), ("errors", "0"), ("skipped", "0")))
            or suite.find("testcase").get("name") != "test_recurrent_v8_nested_receipt_serialization"):
        raise ValueError("ordinary serializer JUnit result changed")
    components = {name: {"status": monitor["status"], "seconds": monitor["elapsed_seconds"],
                         "peak_mib": monitor["observed_tree_peak_sum_mib"]} for name, monitor in launches.items()}
    return {"components": components, "seconds": sum(v["seconds"] for v in components.values()),
            "peak_mib": max(v["peak_mib"] for v in components.values()),
            "first_failed_source": {
                "route": "online", "document_id": "N36", "inferred_offers": inferred_offers,
                "saved_source_rows": 0, "saved_ticks": None, "saved_rewards": None,
                "basis": "source/control-flow inference: source call returned before source-feedback journal encoding failed",
                "scope": "not reconstructed scalar exposure; unpublished dirty state is not restored",
            },
            "serialization_qa": qa,
            "double_counting_policy": "add first runner, audit and banker outer times once; never add the prior bank's preparation subtotal (which already includes build/author work)",
            "continuation_policy": "corrected fresh actors restore V7 ONLINE; N36 is deliberately reoffered, no dirty V8 resume"}


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
    correction = correction_history(evidence)
    return {"pure_builds": 1, "build_seconds": build["elapsed_seconds"],
            "build_peak_mib": build["observed_tree_peak_sum_mib"], "build_children": 0,
            "author_attempts": attempts, "author_children": sum(a["children"] for a in attempts),
            "correction_history": correction,
            "seconds": build["elapsed_seconds"] + sum(a["seconds"] for a in attempts) + correction["seconds"],
            "peak_mib": max(build["observed_tree_peak_sum_mib"], correction["peak_mib"], *(a["peak_mib"] for a in attempts)),
            "failure_scope": "first author harness used platform CRLF stdout; second used the qualified LF bootstrap; both launches retained",
            "budget_scope": "completed preparation cost outside the selected 3600-second runner and 120-second saved audit"}


def bank(run, review_path, run_launch, review_launch, previous_review_launches=()):
    started = time.perf_counter()
    run, review_path, run_launch, review_launch = (
        Path(p).resolve() for p in (run, review_path, run_launch, review_launch))
    if (run != (ROOT / ".pytest-tmp/persistent-language-action-v8-corrected-run").resolve()
            or run_launch != (ROOT / ".pytest-tmp/persistent-language-action-v8-corrected.launch.json").resolve()):
        raise ValueError("this banker is selected only for the corrected fresh V8 launch")
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
        "timing_scope": "one build; failed+corrected author QA; first failed runtime, its audit and banker outer; serialization QA; corrected runtime and its audits; current bank byte checks. Current banker serialization/process tail requires its outer watchdog",
        "claim": "open authored continuation; valid scalar custody is not a learning success or protected evaluation",
        "checkpoint_decodes_in_bank": 0, "models_or_children_in_bank": 0,
        "scientific_replays_by_banker": 0, "corrected_session_deliberately_reoffers_N36": True,
        "dirty_failed_state_resumed": False,
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
