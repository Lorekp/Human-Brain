"""Serial source-isolated comparison and saved-only audit under one outer clock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAFT = False
COMPLETE_SECONDS = 10800.0
RUNNER_SECONDS = 10200.0
ARTIFACT_BYTES = 46 * 1024**3
FREE_RESERVE_BYTES = 8 * 1024**3


def save(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    # All coordinator files are direct children of the comparison evidence root.
    # Admit the exact pending bytes, including the final saved-only audit report.
    used = sum(item.stat().st_size for item in path.parent.rglob("*") if item.is_file())
    if (used + len(raw) > ARTIFACT_BYTES
            or shutil.disk_usage(path.parent).free < FREE_RESERVE_BYTES + len(raw)):
        raise RuntimeError("comparison receipt exceeds artifact/free-space budget")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    used = sum(item.stat().st_size for item in path.parent.rglob("*") if item.is_file())
    if used > ARTIFACT_BYTES or shutil.disk_usage(path.parent).free < FREE_RESERVE_BYTES:
        raise RuntimeError("comparison artifact/free-space budget exceeded after publication")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def run(args):
    if DRAFT:
        raise RuntimeError("DRAFT: complete comparison source review is pending")
    start = time.perf_counter()
    outer_start = float(os.environ["HUMANBRAIN_RUN_START_UNIX"])
    outer_end = float(os.environ["HUMANBRAIN_RUN_DEADLINE_UNIX"])
    if (not math.isfinite(outer_start) or not math.isfinite(outer_end)
            or not math.isclose(outer_end - outer_start, COMPLETE_SECONDS, abs_tol=0.01)):
        raise ValueError("comparison requires the declared 10800-second outer watchdog")
    # perf_counter is system-wide on the qualified Windows Python environment.
    # The outer launch includes interpreter startup before this process begins.
    comparison_started = start - (time.time() - outer_start)
    deadline = comparison_started + RUNNER_SECONDS
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError("comparison needs a distinct checkout-local evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    phases = []
    clock = time.get_clock_info("perf_counter")
    receipt = {"schema": "stable-language-action-credit-comparison-launch-v1",
               "source_revision": args.source_revision,
               "manifest_sha256": args.manifest_sha,
               "comparison_started": comparison_started,
               "main_deadline": deadline, "complete_seconds": COMPLETE_SECONDS,
               "clock": {key: getattr(clock, key) for key in
                         ("implementation", "monotonic", "adjustable", "resolution")},
               "phases": phases, "status": "running"}
    save(output / "invocation.json", receipt)
    child = None
    try:
        old_cp0 = None
        for rule in ("old", "causal"):
            command = [sys.executable, "-X", "utf8", "-m",
                       "experiments.r4_integrated.stable_language_action_credit_v1",
                       "--output", str(output / rule), "--comparison-root", str(output),
                       "--comparison-started", repr(comparison_started),
                       "--deadline", repr(deadline),
                       "--manifest", str(args.manifest.resolve()),
                       "--manifest-sha", args.manifest_sha,
                       "--source-revision", args.source_revision,
                       "--session-id", "stable-credit-v1-" + rule,
                       "--credit-rule", rule, "--old-snapshot", str(args.old_snapshot.resolve())]
            if old_cp0 is not None:
                command += ["--old-cp0-path", old_cp0["path"], "--old-cp0-sha", old_cp0["sha256"]]
            tick = time.perf_counter()
            phase = {"phase": rule, "command": command, "started": tick,
                     "status": "running"}
            phases.append(phase)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("main worker has no remaining audit-reserved time")
            with ((output / (rule + ".stdout")).open("xb") as stdout,
                  (output / (rule + ".stderr")).open("xb") as stderr):
                child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                phase["pid"] = child.pid
                phase["returncode"] = child.wait(timeout=remaining)
            phase["elapsed_seconds"] = time.perf_counter() - tick
            phase["status"] = "passed" if phase["returncode"] == 0 else "failed"
            save(output / (rule + "-launch.json"), phase)
            if phase["returncode"] != 0:
                raise RuntimeError(rule + " worker failed; no automatic restart")
            if time.perf_counter() >= deadline:
                raise TimeoutError("main comparison exhausted the complete audit reserve")
            if rule == "old":
                cp0_path = output / "old/probes/cp-0/summary.json"
                cp0_raw = cp0_path.read_bytes()
                old_cp0 = {"path": str(cp0_path), "sha256": hashlib.sha256(cp0_raw).hexdigest()}
        from experiments.r4_integrated.stable_language_action_credit_review_v1 import review

        tick = time.perf_counter()
        audited = review(output, args.manifest_sha)
        audit_file = save(output / "review.json", audited)
        phases.append({"phase": "saved_only_audit", "elapsed_seconds": time.perf_counter() - tick,
                       "artifact": audit_file})
        if audited.get("passed") is not True:
            raise ValueError("saved comparison audit did not pass; findings are retained")
        if time.perf_counter() >= comparison_started + COMPLETE_SECONDS:
            raise TimeoutError("complete audit and publication budget exhausted")
        receipt.update(status="completed", elapsed_seconds=time.perf_counter() - start,
                       outer_accounted_seconds=time.perf_counter() - comparison_started,
                       audit=audit_file)
        save(output / "comparison.json", receipt)
    except BaseException as error:
        # The required outer watchdog retains creation-identity ownership of
        # every observed descendant. Exiting nonzero triggers its tree cleanup;
        # killing only a Windows Python redirector here could orphan the learner.
        # Do not claim that exposure stopped at this parent receipt's timestamp.
        worker_running = child is not None and child.poll() is None
        receipt.update(status="incomplete", error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.perf_counter() - start,
                       outer_accounted_seconds=time.perf_counter() - comparison_started,
                       worker_running_at_parent_failure=worker_running,
                       cleanup_authority="required outer watchdog creation-identity process tree",
                       exposure_stop_receipt="outer launch receipt; parent timestamp is not a stop proof")
        try:
            save(output / "comparison-failure.json", receipt)
        except (OSError, RuntimeError) as receipt_error:
            error.add_note("Parent failure receipt write failed: " + str(receipt_error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--old-snapshot", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
