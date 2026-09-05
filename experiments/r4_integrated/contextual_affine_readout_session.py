"""One saved-checkpoint affine/interpolated comparison; no evidence learning."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.r4_integrated import contextual_binding_diagnostic as saved
from experiments.r4_integrated import contextual_exploration_python_session as exploration
from experiments.r4_integrated.contextual_occurrence_motor import ContextualOccurrenceMotor

session, ROOT = saved.session, saved.ROOT
shared, require = session.shared, saved.require
SOURCE_REVISION, RESULT_REVISION = "6120335", "6c64189"
SOURCE = ROOT / "benchmarks/contextual_exploration_python_development_v1.json"
SOURCE_PATHS = (
    *exploration.SOURCE_PATHS,
    "experiments/r4_integrated/contextual_binding_diagnostic.py",
    "experiments/r4_integrated/contextual_affine_readout_session.py",
    "experiments/r4_integrated/contextual_affine_readout_protocol.md",
)
CHANGED_SOURCES = {
    "src/humanbrain/contextual_evidence_memory.py",
}


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _pinned(path, revision):
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT
    )


def preflight():
    """Byte-only custody checks; safe before any model evaluation."""
    artifact = json.loads(SOURCE.read_text("utf-8"))
    require(SOURCE.read_bytes() == _pinned(SOURCE, RESULT_REVISION), "source artifact changed")
    require(artifact["base_revision"].startswith(SOURCE_REVISION), "wrong original revision")
    require(
        artifact["terminal"] == "completed" and artifact["sources_unchanged_during_run"],
        "incomplete original comparison",
    )
    checkpoint = Path(artifact["synapse_checkpoint"]["path"])
    require(checkpoint.read_bytes() == _pinned(checkpoint, RESULT_REVISION), "NPZ changed")
    require(_hash(checkpoint) == artifact["synapse_checkpoint"]["sha256"], "NPZ hash differs")
    changed = []
    for path, expected in artifact["source_sha256"].items():
        require(
            sha256(_pinned(ROOT / path, SOURCE_REVISION)).hexdigest() == expected,
            f"historical source differs: {path}",
        )
        if _hash(ROOT / path) != expected:
            require(path in CHANGED_SOURCES, f"undeclared current source change: {path}")
            changed.append(path)
    fixtures = session.load_fixture(
        session.HERE / "contextual_exploration_cases.json", exploration.SOURCE_PATHS
    )
    session.validate_fixture(fixtures)
    return artifact, checkpoint, fixtures, changed


def _motor(artifact):
    """Restore only the original certified motor prior, without new executions."""
    motor = ContextualOccurrenceMotor()
    cases = (
        shared.Case("motor-a", "axel", "fresh", "motor_alpha", "cold"),
        shared.Case("motor-b", "bora", "ready", "motor_beta", "warm"),
    )
    for case, receipt in zip(cases, artifact["motor_demonstrations"], strict=True):
        execution = receipt["execution"]
        require(
            receipt["case"] == case.name
            and execution["passed"]
            and execution["returncode"] == 0
            and execution["decoys_unchanged"]
            and execution["materialized_matches_action"]
            and execution["stdout"].strip() == case.value,
            "original motor execution receipt differs",
        )
        motor.learn(f"Configure {case.key}".encode(), case.files(), case.files(corrected=True))
    require(
        {key: getattr(motor, key) for key in artifact["motor"]} == artifact["motor"],
        "restored motor prior differs",
    )
    return motor


def _verify_original(arm, models):
    matched = 0
    parents = [m.checkpoint_sha256 for m in models]
    for policy in session.evaluation_policies(*models, binding=arm["candidate_interactions"]):
        for phase in ("transfer", "retention"):
            for row in arm[phase][policy.name]["rows"]:
                require(
                    shared.decision_record(policy.respond(row["selection"]["query"].encode()))
                    == row["selection"],
                    f"original decision differs: {phase}/{policy.name}/{row['case']}",
                )
                matched += 1
    require(parents == [m.checkpoint_sha256 for m in models], "verification changed state")
    return matched


def _policies(models):
    return tuple(
        session.EvaluationPolicy(name, memory, getattr(memory, method), method)
        for memory, prefix in zip(models, ("", "frozen_"), strict=True)
        for name, method in (
            (prefix + "affine", "respond_affine_interpolated"),
            (prefix + "interpolated", "respond_interpolated"),
        )
    )


def _evaluate(models, motor, fixtures, work, result):
    parents = [m.checkpoint_sha256 for m in models]
    motor_parent = motor.checkpoint_sha256
    for phase, cases in (("retention", fixtures["training"]), ("transfer", fixtures["transfer"])):
        result[phase] = {}
        for policy in _policies(models):
            began, rows = time.perf_counter(), []
            for item in cases:
                row, decision = shared.attempt(
                    policy, "respond", motor, item, work / phase / policy.name / item["name"]
                )
                raw = policy.memory.raw_score_reference(item["request"].encode())
                row.update(
                    policy=policy.name,
                    raw_reference=shared.decision_record(raw),
                    raw_occurrence_agreement=raw.candidate_index == decision.candidate_index,
                    selected_occurrence=None
                    if decision.stopped
                    else saved.occurrence(policy.memory, decision.candidate_index),
                )
                rows.append(row)
            result[phase][policy.name] = {
                "rows": rows,
                "summary": session.prior._summary(rows, cases),
                "raw_occurrence_agreements": sum(r["raw_occurrence_agreement"] for r in rows),
                "checkpoint_sha256": policy.checkpoint_sha256,
                "elapsed_seconds": time.perf_counter() - began,
            }
    require(parents == [m.checkpoint_sha256 for m in models], "execution changed checkpoint")
    require(motor.checkpoint_sha256 == motor_parent, "execution changed motor")


def _exam(models, original):
    # The shared helper ingests the unchanged original passage once per model;
    # repeated ingestion is a verified no-op when the new policies follow.
    policies = session.evaluation_policies(*models, binding=original["candidate_interactions"])
    old = session.prior._exam(
        [(p.name, p.memory, p.method) for p in policies if p.method is not None]
    )
    require(old == original["exam"], "original exam decisions or scoring differ")
    require(
        [m.checkpoint_sha256 for m in models]
        == [original[key]["checkpoint_sha256"] for key in ("end", "frozen_end")],
        "post-exam checkpoint differs",
    )
    parents = [m.checkpoint_sha256 for m in models]
    result = session.prior._exam([(p.name, p.memory, p.method) for p in _policies(models)])
    require(parents == [m.checkpoint_sha256 for m in models], "new exam changed checkpoint")
    return {"original_policies_exactly_matched": len(old), "policies": result}


def run(output, model_dir):
    require(not output.exists(), "refuse to overwrite evidence")
    require(os.environ.get("HUMANBRAIN_DEVICE") == "cpu", "set CPU before Python")
    require(
        Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve() == ROOT / ".native-cache",
        "set checkout native cache before Python",
    )
    artifact, checkpoint, fixtures, changed = preflight()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contextual-affine-", dir=ROOT / ".runtime"))
    began = time.perf_counter()
    shared.tracemalloc.start()
    report = {
        "terminal": "running",
        "kind": "saved exploration checkpoints; affine readout recovery; no evidence learning",
        "source_artifact_sha256": _hash(SOURCE),
        "original_source_revision": artifact["base_revision"],
        "original_result_revision": RESULT_REVISION,
        "original_source_sha256": artifact["source_sha256"],
        "synapse_checkpoint": artifact["synapse_checkpoint"],
        "source_sha256": session._hashes(SOURCE_PATHS),
        "changed_historical_sources": changed,
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "workspace_root": str(work),
        "exposure": "reconstructed original documents, receipt lineage and motor prior; "
        "no evidence updates, receipt replay updates or new teacher executions",
        "controls": "same saved weights; frozen evidence weights; shared learned motor; "
        "evidence-selection comparison, not whole-system core-disabled",
        "arms": {},
    }
    try:
        native = saved.loaded_build_info()
        report["native"] = {
            "source_digest": native.source_digest,
            "binary_sha256": _hash(native.library_path),
            "library": str(native.library_path),
        }
        require(report["native"] == artifact["native"], "native identity differs")
        require(native.library_path.resolve().is_relative_to(ROOT / ".native-cache"), "wrong DLL")
        start = time.perf_counter()
        encoder = saved.FrozenContextEncoder(model_dir)
        require(encoder.metadata == artifact["encoder"], "encoder identity differs")
        report["encoder"], report["encoder_load_seconds"] = (
            encoder.metadata,
            time.perf_counter() - start,
        )
        start = time.perf_counter()
        motor = _motor(artifact)
        report["motor_restoration_seconds"] = time.perf_counter() - start
        report["motor"] = artifact["motor"]
        report["certified_motor_receipts"] = artifact["motor_demonstrations"]
        restored = {}
        with np.load(checkpoint, allow_pickle=False) as archive:
            for name, original in artifact["arms"].items():
                start = time.perf_counter()
                models, seen, pairs, verified = saved.reconstruct(
                    original, name, fixtures, archive, encoder
                )
                for item in fixtures["transfer"]:
                    session.prior._ingest_once(item, models, seen)
                require(
                    models[0].checkpoint_sha256 == original["transfer_start"]["checkpoint_sha256"],
                    "transfer checkpoint differs",
                )
                result = {
                    "reconstruction_seconds": time.perf_counter() - start,
                    "historical_feedback_offers_verified": verified,
                    "before": [session._snapshot(m) for m in models],
                    "retrospective_training_pairs": [
                        {
                            key: pair[key]
                            for key in (
                                "pair_id",
                                "query",
                                "final_margin",
                                "first_gap_after",
                                "last_gap_after",
                                "offer_count",
                            )
                        }
                        for pair in pairs
                    ],
                }
                report["arms"][name] = result
                start = time.perf_counter()
                result["original_decisions_exactly_matched"] = _verify_original(original, models)
                result["original_verification_seconds"] = time.perf_counter() - start
                restored[name] = models
        report["original_decisions_exactly_matched"] = sum(
            arm["original_decisions_exactly_matched"] for arm in report["arms"].values()
        )
        require(report["original_decisions_exactly_matched"] == 416, "old policy coverage differs")
        for name, models in restored.items():
            result, original = report["arms"][name], artifact["arms"][name]
            _evaluate(models, motor, fixtures, work / name, result)
            result["after_execution"] = [session._snapshot(m) for m in models]
            require(result["before"] == result["after_execution"], "read-only state changed")
            start = time.perf_counter()
            result["exam"] = _exam(models, original)
            result["exam_seconds"] = time.perf_counter() - start
            result["end"] = [session._snapshot(m) for m in models]
            print(json.dumps({"arm": name, "terminal": "completed"}), flush=True)
        report["sources_unchanged_during_run"] = (
            session._hashes(SOURCE_PATHS) == report["source_sha256"]
        )
        require(report["sources_unchanged_during_run"], "sources changed during run")
        require(_hash(SOURCE) == report["source_artifact_sha256"], "original artifact changed")
        require(_hash(checkpoint) == artifact["synapse_checkpoint"]["sha256"], "NPZ changed")
        report["terminal"] = "completed"
    except Exception:
        report["terminal"], report["error"] = "error", traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - began
        report["python_traced_peak_bytes"] = shared.tracemalloc.get_traced_memory()[1]
        shared.tracemalloc.stop()
        report["process_memory"] = shared._process_memory() if os.name == "nt" else None
        report["resource_scope"] = (
            "parent/evaluator memory excludes child memory; wall time includes child waits"
        )
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(session._json(report) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    run(args.output.resolve(), args.model_dir.resolve())
