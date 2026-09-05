"""Gain 1 versus fixed gain 16 in the same executed corrective replay loop.

Launch with CPU and checkout-local native-cache environment set before Python.
Teaching is openly reused; the adjacent fixture declares fresh transfer cases.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.r4_integrated import contextual_revisit_python_session as replay
from humanbrain.contextual_evidence_memory import ContextualEvidenceMemory
from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.native import loaded_build_info

shared, prior, ROOT, HERE = replay.shared, replay.prior, replay.ROOT, replay.HERE
SOURCE_PATHS = (
    *replay.SOURCE_PATHS,
    "experiments/r4_integrated/contextual_conditioned_python_session.py",
    "experiments/r4_integrated/contextual_conditioned_cases.json",
    "experiments/r4_integrated/contextual_conditioned_protocol.md",
)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hashes() -> dict[str, str]:
    return {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def _training_summary(rows) -> dict:
    sources = {}
    for source in ("neuronal", "raw_learned_score", "external_only"):
        selected = [row for row in rows if row["negative_source"] == source]
        admitted = [row for row in selected if row["correction_status"] == "applied"]
        applied = sum(row["feedback"][0]["update"]["native_update_applied"] for row in admitted)
        sources[source] = {
            "candidate_attempts": sum(row["candidate_source"] == source for row in rows),
            "executed_negatives": len(selected),
            "admitted_pairs": len(admitted),
            "new_native_updates": applied,
            "new_noops": len(admitted) - applied,
        }
    return {
        "attempts": len(rows),
        "first_executed_success": sum(row["first"]["execution"]["passed"] for row in rows),
        "frozen_first_executed_success": sum(
            row["frozen_first"]["execution"]["passed"] for row in rows
        ),
        "fallback_attempts": sum(row["fallback_kind"] != "none" for row in rows),
        "fallback_executed_success": sum(
            row.get("exploration", {}).get("execution", {}).get("passed", False) for row in rows
        ),
        "teacher_executions": len(rows),
        "score_spike_occurrence_disagreements": sum(
            row["first_score_spike_disagreement"] for row in rows
        ),
        "raw_score_spike_occurrence_disagreements": sum(
            row["first_raw_score_spike_disagreement"] for row in rows
        ),
        "raw_mapped_occurrence_disagreements": sum(
            row["first_raw_mapped_disagreement"] for row in rows
        ),
        "correction_status_counts": dict(Counter(row["correction_status"] for row in rows)),
        "negative_sources": sources,
    }


def _old_query_scores(memory, cases) -> dict:
    """Evaluator-only raw ranking progress; never influences feedback or scheduling."""
    parent, began, rows = memory.checkpoint_sha256, time.perf_counter(), []
    for item in cases:
        decision = memory.raw_score_reference(item["request"].encode())
        rows.append(
            {
                "case": item["name"],
                "selection": shared.decision_record(decision),
                "correct": decision.emitted == item["value"].encode(),
            }
        )
    if memory.checkpoint_sha256 != parent:
        raise RuntimeError("read-only old-query raw ranking changed checkpoint")
    return {
        "scope": "raw learned-drive evidence accuracy only; no execution or feedback",
        "policy": "raw_score_reference",
        "checkpoint_sha256": parent,
        "rows": rows,
        "correct": sum(row["correct"] for row in rows),
        "seconds": time.perf_counter() - began,
    }


def _teach(name, update, frozen, motor, fixtures, work, arm, record):
    cases = {item["name"]: item for item in fixtures["training"]}
    seen, encountered, ledger = {}, {}, []
    replay_cursor = replay_count = 0
    for step, case_name in enumerate(fixtures["schedule"], 1):
        item = cases[case_name]
        row = {
            "step": step,
            "pass": (step - 1) // 16 + 1,
            "case": case_name,
            "document_id": item["document_id"],
            "feedback": [],
            "fallback_kind": "none",
        }
        arm["training"].append(row)
        began = time.perf_counter()
        row["ingested"] = prior._ingest_once(item, (update, frozen), seen)
        row["paired_ingestion_seconds"] = time.perf_counter() - began
        row["after_ingestion"] = shared.snapshot(update)
        folder = work / name / "training" / f"{step:02d}-{case_name}"
        row["frozen_first"], _ = shared.attempt(frozen, "respond", motor, item, folder / "frozen")
        row["first"], candidate = shared.attempt(update, "respond", motor, item, folder / "first")
        began, parent = time.perf_counter(), update.checkpoint_sha256
        conventional = update.conventional(item["request"].encode())
        row["first_score_reference_seconds"] = time.perf_counter() - began
        began = time.perf_counter()
        raw = update.raw_score_reference(item["request"].encode())
        row["first_raw_score_reference_seconds"] = time.perf_counter() - began
        if update.checkpoint_sha256 != parent:
            raise RuntimeError("read-only score reference changed checkpoint")
        row["first_score_reference"] = shared.decision_record(conventional)
        row["first_raw_score_reference"] = shared.decision_record(raw)
        row["first_score_spike_disagreement"] = (
            candidate.candidate_index != conventional.candidate_index
        )
        row["first_raw_score_spike_disagreement"] = candidate.candidate_index != raw.candidate_index
        row["first_raw_mapped_disagreement"] = raw.candidate_index != conventional.candidate_index
        candidate_result = row["first"]
        if candidate.stopped:
            policy = "external_only" if raw.stopped else "raw_score_reference"
            row["fallback_kind"] = "external_only" if raw.stopped else "raw_learned_score"
            row["exploration"], candidate = shared.attempt(
                update, policy, motor, item, folder / "exploration"
            )
            if policy == "raw_score_reference" and candidate.candidate_index != raw.candidate_index:
                raise RuntimeError("executed raw fallback differs from recorded score winner")
            candidate_result = row["exploration"]
        row["candidate_source"] = (
            "neuronal" if row["fallback_kind"] == "none" else row["fallback_kind"]
        )
        row["negative_source"] = (
            row["candidate_source"]
            if candidate_result["edited"] and not candidate_result["execution"]["passed"]
            else None
        )
        case = prior._case(item)
        correction = shared.execute(
            case, case.files(corrected=True), case.files(), folder / "teacher"
        )
        row["teacher_execution"] = correction
        if not correction["passed"]:
            raise RuntimeError("teacher correction did not execute successfully")
        if candidate_result["edited"] and candidate_result["execution"]["passed"]:
            row["correction_status"] = "no_needed_correction"
        elif not candidate_result["edited"]:
            row["correction_status"] = "no_executed_negative"
        elif len(ledger) == replay.PAIR_BUDGET:
            row["correction_status"] = "pair_budget_exhausted"
        else:
            try:
                accepted = update.decision_for_span(
                    item["request"].encode(), item["value"].encode()
                )
            except ValueError as error:
                row["correction_status"], row["teacher_rejection"] = (
                    "ambiguous_teacher_span",
                    str(error),
                )
            else:
                pair = replay._Pair(
                    f"{name}-{len(ledger) + 1:02d}",
                    step,
                    candidate,
                    accepted,
                    _json(candidate_result["execution"]),
                    _json(correction),
                )
                row.update(
                    {
                        "pair_id": pair.pair_id,
                        "rejected_decision": shared.decision_record(candidate),
                        "accepted_decision": shared.decision_record(accepted),
                    }
                )
                row["feedback"].append(replay._feedback(update, frozen, pair, replay=False))
                older = len(ledger)
                ledger.append(pair)
                arm["ledger"].append(pair.record())
                row["correction_status"] = "applied"
                row["old_query_scores_after_new"] = _old_query_scores(update, encountered.values())
                if older:
                    count = min(4, older, replay.REPLAY_BUDGET - replay_count)
                    for offset in range(count):
                        previous = ledger[(replay_cursor + offset) % older]
                        row["feedback"].append(
                            replay._feedback(update, frozen, previous, replay=True)
                        )
                    replay_count += count
                    replay_cursor = (replay_cursor + count) % older
                    if count:
                        row["old_query_scores_after_rehearsal"] = _old_query_scores(
                            update, encountered.values()
                        )
        encountered.setdefault(case_name, item)
        row["after"], row["frozen_after"] = shared.snapshot(update), shared.snapshot(frozen)
        row["new_pair_count"], row["replay_count"] = len(ledger), replay_count
        record(
            "teaching",
            arm=name,
            step=step,
            case=case_name,
            first_success=row["first"]["execution"]["passed"],
            fallback=row["fallback_kind"],
            fallback_success=row.get("exploration", {}).get("execution", {}).get("passed"),
            correction_status=row["correction_status"],
            pairs=len(ledger),
            replays=replay_count,
        )
    arm["training_summary"] = _training_summary(arm["training"])
    arm["pass_summaries"] = [
        {
            "pass": number,
            **_training_summary([row for row in arm["training"] if row["pass"] == number]),
        }
        for number in (1, 2, 3)
    ]
    # Preserve the predecessor's canonical ledger hash convention for diagnostics;
    # final artifact encoding is independently compact and retains every float.
    serialized = replay._json([pair.record() for pair in ledger]).encode()
    events = [event for row in arm["training"] for event in row["feedback"]]
    arm["ledger_summary"] = {
        "new_pairs": len(ledger),
        "replay_events": replay_count,
        "new_native_updates": sum(
            e["update"]["native_update_applied"] for e in events if e["kind"] == "new"
        ),
        "replay_native_updates": sum(
            e["update"]["native_update_applied"] for e in events if e["kind"] == "certified_replay"
        ),
        "serialized_bytes": len(serialized),
        "sha256": sha256(serialized).hexdigest(),
        "compact_serialized_bytes": len(_json(arm["ledger"]).encode()),
        "eligibility_float64_payload_bytes": sum(
            8 * (len(p.rejected.eligibility) + len(p.accepted.eligibility)) for p in ledger
        ),
        "scope": "external immutable correction ledger; serialized size excludes "
        "Python object overhead",
        "fifo_policy": "append in admission order; cap 32; final correction rehearsal "
        "completes before updates stop",
    }
    if len(seen) != 8 or len(ledger) > replay.PAIR_BUDGET or replay_count > replay.REPLAY_BUDGET:
        raise RuntimeError("teaching exposure or feedback budget differs from protocol")
    return seen


def _evaluate(name, update, frozen, motor, fixtures, work, arm, record):
    policies = (
        ("update", update, "respond"),
        ("frozen", frozen, "respond"),
        ("conventional", update, "conventional"),
        ("external", frozen, "external_only"),
        ("raw_score_reference", update, "raw_score_reference"),
    )
    for phase in ("transfer", "retention"):
        began = time.perf_counter()
        cases = fixtures["transfer" if phase == "transfer" else "training"]
        for policy_name, memory, policy in policies:
            rows = [
                shared.attempt(
                    memory, policy, motor, item, work / name / phase / policy_name / item["name"]
                )[0]
                for item in cases
            ]
            summary = prior._summary(rows, cases)
            arm[phase][policy_name] = {"summary": summary, "rows": rows}
            record("evaluation", arm=name, phase=phase, policy=policy_name, summary=summary)
        for metric, left, right in (
            ("score_spike_disagreements", "conventional", "update"),
            ("raw_score_spike_disagreements", "raw_score_reference", "update"),
            ("raw_mapped_disagreements", "raw_score_reference", "conventional"),
        ):
            arm[phase + "_" + metric] = sum(
                a["selection"]["candidate_index"] != b["selection"]["candidate_index"]
                for a, b in zip(arm[phase][left]["rows"], arm[phase][right]["rows"], strict=True)
            )
        arm["phase_seconds"][phase] = time.perf_counter() - began
    return policies


def run(output: Path, model_dir: Path) -> dict:
    output = output.resolve()
    if output.exists() or output.with_suffix(".npz").exists():
        raise FileExistsError("choose a fresh output path; existing evidence is immutable")
    if (
        os.environ.get("HUMANBRAIN_DEVICE") != "cpu"
        or Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve()
        != (ROOT / ".native-cache").resolve()
    ):
        raise RuntimeError("set CPU and checkout native-cache environment before launching Python")
    output.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / ".runtime").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contextual-conditioned-", dir=ROOT / ".runtime"))
    started = time.perf_counter()
    shared.tracemalloc.start()
    report = {
        "kind": "public development; interaction gain 1 versus fixed gain 16; same replay policy",
        "terminal": "running",
        "workspace_root": str(work),
        "arms": {},
        "events": [],
        "native_environment": {
            key: os.environ[key] for key in ("HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        },
        "teaching_exposure": "sixteen public teaching requests and 48 schedule entries openly "
        "reused; fresh transfer fixture declared separately",
        "feedback_contract": "only executed FAIL/teacher PASS pairs; certified replay; after "
        "neuronal STOP prefer raw learned-score winner, then external cosine",
        "retained_source_policy": "pretrained frozen MiniLM plus external raw documents, "
        "candidate vectors, storage ancestry and immutable correction ledger",
        "controls": "same-gain frozen mate receives exact feedback; all references share "
        "learned motor; updating arms may collect different on-policy failures",
    }

    def record(event, **fields):
        row = {"event": event, **fields}
        report["events"].append(row)
        with (work / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(row) + "\n")
        print(_json(row), flush=True)

    try:
        report["source_sha256"] = _hashes()
        report["base_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        fixtures = json.loads((HERE / "contextual_conditioned_cases.json").read_text("utf-8"))
        replay.validate_fixture(fixtures)
        native = loaded_build_info()
        report["native"] = {
            "source_digest": native.source_digest,
            "library": str(native.library_path),
            "binary_sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        }
        if not native.library_path.resolve().is_relative_to((ROOT / ".native-cache").resolve()):
            raise RuntimeError("loaded native library is outside the declared checkout cache")
        began = time.perf_counter()
        motor = prior._motor(work, report)
        report["motor_setup_seconds"] = time.perf_counter() - began
        began = time.perf_counter()
        encoder = FrozenContextEncoder(model_dir)
        report["encoder"], report["encoder_load_seconds"] = (
            encoder.metadata,
            time.perf_counter() - began,
        )
        arrays = {}
        for gain in (1.0, 16.0):
            name = f"GAIN{int(gain)}"
            update = ContextualEvidenceMemory(encoder, learning_rate=1.0, interaction_gain=gain)
            frozen = ContextualEvidenceMemory(
                encoder, learning_rate=1.0, interaction_gain=gain, learning_enabled=False
            )
            if update.checkpoint_sha256 != frozen.checkpoint_sha256:
                raise RuntimeError("paired initial checkpoints differ")
            arm = {
                "interaction_gain": gain,
                "learning_rate": 1.0,
                "seed": shared.snapshot(update),
                "training": [],
                "ledger": [],
                "transfer": {},
                "retention": {},
                "phase_seconds": {},
            }
            report["arms"][name] = arm
            began = time.perf_counter()
            seen = _teach(name, update, frozen, motor, fixtures, work, arm, record)
            arm["phase_seconds"]["teaching"] = time.perf_counter() - began
            arm["trained"], arm["frozen_trained"] = shared.snapshot(update), shared.snapshot(frozen)
            began = time.perf_counter()
            for item in fixtures["transfer"]:
                prior._ingest_once(item, (update, frozen), seen)
            arm["phase_seconds"]["fresh_ingestion"] = time.perf_counter() - began
            if len(seen) != 14:
                raise RuntimeError("fresh source exposure differs from protocol")
            arm["transfer_start"] = shared.snapshot(update)
            began = time.perf_counter()
            policies = _evaluate(name, update, frozen, motor, fixtures, work, arm, record)
            arm["phase_seconds"]["transfer_and_retention"] = time.perf_counter() - began
            if (
                frozen.synapse_sha256 != arm["seed"]["synapse_sha256"]
                or update.synapse_sha256 != arm["trained"]["synapse_sha256"]
            ):
                raise RuntimeError("frozen or evaluation synapses changed")
            if update.feedback_count != frozen.feedback_count:
                raise RuntimeError("paired feedback counts differ")
            arm.update(
                {
                    "feedback_count": update.feedback_count,
                    "native_update_count": update.update_count,
                    "frozen_feedback_count": frozen.feedback_count,
                    "frozen_native_update_count": frozen.update_count,
                }
            )
            for policy_name, memory in (("update", update), ("frozen", frozen)):
                for key, value in memory.checkpoint_arrays().items():
                    arrays[f"{name}_{policy_name}_{key}"] = value
            began = time.perf_counter()
            arm["exam"] = prior._exam(policies)
            arm["phase_seconds"]["exam"] = time.perf_counter() - began
            arm["end"], arm["frozen_end"] = shared.snapshot(update), shared.snapshot(frozen)
            record(
                "exam",
                arm=name,
                summary={key: value["scoring"] for key, value in arm["exam"].items()},
            )
        checkpoint = output.with_suffix(".npz")
        np.savez_compressed(checkpoint, **arrays)
        report["synapse_checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint.read_bytes()).hexdigest(),
            "scope": "trained synapses; ordered fixtures and receipts reconstruct "
            "storage and lineage",
        }
        report["sources_unchanged_during_run"] = _hashes() == report["source_sha256"]
        if not report["sources_unchanged_during_run"]:
            raise RuntimeError("sources changed during comparison")
        report["terminal"] = "completed"
    except Exception:
        report["terminal"], report["error"] = "error", traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["python_traced_peak_bytes"] = shared.tracemalloc.get_traced_memory()[1]
        shared.tracemalloc.stop()
        report["process_memory"] = shared._process_memory() if os.name == "nt" else None
        report["resource_scope"] = (
            "wall time includes child waits; memory covers parent/evaluator/ledger, "
            "excluding child memory"
        )
        output.write_text(_json(report) + "\n", encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.output, args.model_dir)
    print(_json({"terminal": result["terminal"], "output": str(args.output)}))
