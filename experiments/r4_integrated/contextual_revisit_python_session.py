"""Bounded executed revisitation, with or without certified correction replay.

Set HUMANBRAIN_DEVICE=cpu and HUMANBRAIN_NATIVE_CACHE to this checkout's cache
before launching Python: shared imports may initialize native cells. The adjacent
protocol and fresh fixture own the schedule; this module never changes them.
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
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.r4_integrated import contextual_pairwise_python_session as prior
from humanbrain.contextual_evidence_memory import (
    ContextualEvidenceDecision,
    ContextualEvidenceMemory,
)
from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.native import loaded_build_info

shared, ROOT, HERE = prior.shared, prior.ROOT, prior.HERE
SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            *prior.SOURCE_PATHS,
            "experiments/r4_integrated/contextual_revisit_python_session.py",
            "experiments/r4_integrated/contextual_revisit_cases.json",
            "experiments/r4_integrated/contextual_revisit_protocol.md",
            "src/humanbrain/opaque_document_event_tape.py",
        )
    )
)
PAIR_BUDGET, REPLAY_BUDGET = 32, 128


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _hashes() -> dict[str, str]:
    return {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def validate_fixture(fixtures: dict) -> None:
    training, transfer, schedule = (fixtures[key] for key in ("training", "transfer", "schedule"))
    names = [item["name"] for item in training]
    if len(training) != 16 or len(set(names)) != 16 or len(transfer) != 10:
        raise ValueError("requires sixteen distinct teaching requests and ten transfer requests")
    if len(schedule) != 48 or any(
        Counter(schedule[i : i + 16]) != Counter(names) for i in range(0, 48, 16)
    ):
        raise ValueError("schedule must contain three complete sixteen-request passes")
    if sum(item["value"] is not None for item in transfer) != 8:
        raise ValueError("requires eight supported and two unsupported transfer requests")
    documents = {}
    for item in (*training, *transfer):
        identity, content = item["document_id"], item["document"]
        if identity in documents and documents[identity] != content:
            raise ValueError("one document identity has conflicting bytes")
        documents[identity] = content
        if item["value"] is not None and item["value"] == item["stale"]:
            raise ValueError("a baseline already equals its teacher target")
    if len({item["document_id"] for item in training}) != 8 or len(documents) != 14:
        raise ValueError("requires eight teaching documents and six new transfer documents")


@dataclass(frozen=True)
class _Pair:
    pair_id: str
    origin_step: int
    rejected: ContextualEvidenceDecision
    accepted: ContextualEvidenceDecision
    rejected_execution_json: str
    teacher_execution_json: str

    def record(self) -> dict:
        def decision(value):
            return {**shared.decision_record(value), "eligibility": list(value.eligibility)}

        return {
            "pair_id": self.pair_id,
            "origin_step": self.origin_step,
            "rejected_decision": decision(self.rejected),
            "accepted_decision": decision(self.accepted),
            "rejected_execution": json.loads(self.rejected_execution_json),
            "teacher_execution": json.loads(self.teacher_execution_json),
        }


def _feedback(update, frozen, pair, *, replay: bool) -> dict:
    rejected, accepted = pair.rejected, pair.accepted
    if replay:
        rejected, accepted = (update.revisit_decision(value) for value in (rejected, accepted))
        for old, current in ((pair.rejected, rejected), (pair.accepted, accepted)):
            if (old.query, old.candidate_index, old.emitted, old.eligibility) != (
                current.query,
                current.candidate_index,
                current.emitted,
                current.eligibility,
            ):
                raise RuntimeError("rehearsal changed the original certified occurrence")
    frozen_rejected = frozen.replay_decision(rejected)
    frozen_accepted = frozen.replay_decision(accepted)
    receipt = {
        "kind": "certified_replay" if replay else "new",
        "pair_id": pair.pair_id,
        "origin_step": pair.origin_step,
        "rejected_decision": shared.decision_record(rejected),
        "accepted_decision": shared.decision_record(accepted),
        "update": update.observe_comparison(rejected, accepted),
        "frozen": frozen.observe_comparison(frozen_rejected, frozen_accepted),
    }
    if (
        receipt["update"]["eligibility_delta_sha256"]
        != receipt["frozen"]["eligibility_delta_sha256"]
    ):
        raise RuntimeError("frozen comparison received a different eligibility delta")
    return receipt


def _old_query_scores(memory, cases) -> dict:
    """Evaluator-only score probe; targets never enter learning or scheduling."""
    parent, began, rows = memory.checkpoint_sha256, time.perf_counter(), []
    for item in cases:
        decision = memory.conventional(item["request"].encode())
        rows.append(
            {
                "case": item["name"],
                "selection": shared.decision_record(decision),
                "correct": decision.emitted == item["value"].encode(),
            }
        )
    if memory.checkpoint_sha256 != parent:
        raise RuntimeError("read-only old-query scoring changed checkpoint")
    return {
        "scope": "learned-score evidence accuracy only; no execution or feedback",
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
        }
        arm["training"].append(row)
        ingestion_started = time.perf_counter()
        row["ingested"] = prior._ingest_once(item, (update, frozen), seen)
        row["paired_ingestion_seconds"] = time.perf_counter() - ingestion_started
        row["after_ingestion"] = shared.snapshot(update)
        folder = work / name / "training" / f"{step:02d}-{case_name}"
        row["frozen_first"], _ = shared.attempt(frozen, "respond", motor, item, folder / "frozen")
        row["first"], candidate = shared.attempt(update, "respond", motor, item, folder / "first")
        conventional = update.conventional(item["request"].encode())
        row["first_score_reference"] = shared.decision_record(conventional)
        row["first_score_spike_disagreement"] = (
            candidate.candidate_index != conventional.candidate_index
        )
        candidate_result = row["first"]
        if candidate.stopped:
            row["exploration"], candidate = shared.attempt(
                update, "external_only", motor, item, folder / "exploration"
            )
            candidate_result = row["exploration"]
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
        elif len(ledger) == PAIR_BUDGET:
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
                pair = _Pair(
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
                row["feedback"].append(_feedback(update, frozen, pair, replay=False))
                older = len(ledger)
                ledger.append(pair)
                arm["ledger"].append(pair.record())
                row["correction_status"] = "applied"
                row["old_query_scores_after_new"] = _old_query_scores(update, encountered.values())
                if name == "REPLAY" and older:
                    for offset in range(min(4, older, REPLAY_BUDGET - replay_count)):
                        replay_pair = ledger[(replay_cursor + offset) % older]
                        row["feedback"].append(_feedback(update, frozen, replay_pair, replay=True))
                        replay_count += 1
                    replay_cursor = (replay_cursor + min(4, older)) % older
                if len(row["feedback"]) > 1:
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
            correction_status=row["correction_status"],
            pairs=len(ledger),
            replays=replay_count,
        )
    arm["pass_summaries"] = [
        {
            "pass": number,
            "attempts": 16,
            "first_executed_success": sum(row["first"]["execution"]["passed"] for row in selected),
            "frozen_first_executed_success": sum(
                row["frozen_first"]["execution"]["passed"] for row in selected
            ),
            "new_pairs": sum(row["correction_status"] == "applied" for row in selected),
            "replay_events": sum(
                len(row["feedback"]) - (row["correction_status"] == "applied") for row in selected
            ),
            "score_spike_occurrence_disagreements": sum(
                row["first_score_spike_disagreement"] for row in selected
            ),
        }
        for number in (1, 2, 3)
        for selected in [[row for row in arm["training"] if row["pass"] == number]]
    ]
    serialized = _json([pair.record() for pair in ledger]).encode()
    arm["ledger_summary"] = {
        "new_pairs": len(ledger),
        "replay_events": replay_count,
        "new_native_updates": sum(
            event["update"]["native_update_applied"]
            for row in arm["training"]
            for event in row["feedback"]
            if event["kind"] == "new"
        ),
        "replay_native_updates": sum(
            event["update"]["native_update_applied"]
            for row in arm["training"]
            for event in row["feedback"]
            if event["kind"] == "certified_replay"
        ),
        "serialized_bytes": len(serialized),
        "sha256": sha256(serialized).hexdigest(),
        "eligibility_float64_payload_bytes": sum(
            8 * (len(p.rejected.eligibility) + len(p.accepted.eligibility)) for p in ledger
        ),
        "scope": "external immutable correction ledger; serialized size excludes "
        "Python object overhead",
        "fifo_policy": "append in admission order; cap 32 with no eviction or further admission",
    }
    if len(seen) != 8 or len(ledger) > PAIR_BUDGET or replay_count > REPLAY_BUDGET:
        raise RuntimeError("teaching exposure or feedback budget differs from protocol")
    return seen


def _evaluate(name, update, frozen, motor, fixtures, work, arm, record):
    policies = (
        ("update", update, "respond"),
        ("frozen", frozen, "respond"),
        ("conventional", update, "conventional"),
        ("external", frozen, "external_only"),
    )
    for phase in ("transfer", "retention"):
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
        arm[phase + "_score_spike_disagreements"] = sum(
            a["selection"]["candidate_index"] != b["selection"]["candidate_index"]
            for a, b in zip(
                arm[phase]["update"]["rows"], arm[phase]["conventional"]["rows"], strict=True
            )
        )
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
    work = Path(tempfile.mkdtemp(prefix="contextual-revisit-", dir=ROOT / ".runtime"))
    started = time.perf_counter()
    shared.tracemalloc.start()
    report = {
        "kind": "public development; executed revisitation versus certified pair replay",
        "terminal": "running",
        "workspace_root": str(work),
        "arms": {},
        "events": [],
        "native_environment": {
            key: os.environ[key] for key in ("HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        },
        "feedback_contract": "executed FAIL and teacher PASS for new pairs; replay uses "
        "original certified pairs without re-execution",
        "retained_source_policy": "external raw documents, candidate vectors and correction ledger",
        "controls": "each frozen mate receives exact feedback; on-policy arms may collect "
        "different failures; references share the learned motor",
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
        fixtures = json.loads((HERE / "contextual_revisit_cases.json").read_text("utf-8"))
        validate_fixture(fixtures)
        native = loaded_build_info()
        report["native"] = {
            "source_digest": native.source_digest,
            "library": str(native.library_path),
            "binary_sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        }
        if not native.library_path.resolve().is_relative_to((ROOT / ".native-cache").resolve()):
            raise RuntimeError("loaded native library is outside the declared checkout cache")
        motor = prior._motor(work, report)
        began = time.perf_counter()
        encoder = FrozenContextEncoder(model_dir)
        report["encoder"], report["encoder_load_seconds"] = (
            encoder.metadata,
            time.perf_counter() - began,
        )
        arrays = {}
        for name in ("REVISIT", "REPLAY"):
            update = ContextualEvidenceMemory(encoder, learning_rate=1.0)
            frozen = ContextualEvidenceMemory(encoder, learning_rate=1.0, learning_enabled=False)
            if update.checkpoint_sha256 != frozen.checkpoint_sha256:
                raise RuntimeError("paired initial checkpoints differ")
            arm = {
                "learning_rate": 1.0,
                "seed": shared.snapshot(update),
                "training": [],
                "ledger": [],
                "transfer": {},
                "retention": {},
            }
            report["arms"][name] = arm
            seen = _teach(name, update, frozen, motor, fixtures, work, arm, record)
            arm["trained"], arm["frozen_trained"] = shared.snapshot(update), shared.snapshot(frozen)
            for item in fixtures["transfer"]:
                prior._ingest_once(item, (update, frozen), seen)
            if len(seen) != 14:
                raise RuntimeError("fresh source exposure differs from protocol")
            arm["transfer_start"] = shared.snapshot(update)
            policies = _evaluate(name, update, frozen, motor, fixtures, work, arm, record)
            if (
                frozen.synapse_sha256 != arm["seed"]["synapse_sha256"]
                or update.synapse_sha256 != arm["trained"]["synapse_sha256"]
            ):
                raise RuntimeError("frozen or evaluation synapses changed")
            if update.feedback_count != frozen.feedback_count:
                raise RuntimeError("paired feedback counts differ")
            arm["feedback_count"], arm["native_update_count"] = (
                update.feedback_count,
                update.update_count,
            )
            arm["frozen_feedback_count"], arm["frozen_native_update_count"] = (
                frozen.feedback_count,
                frozen.update_count,
            )
            for policy_name, memory in (("update", update), ("frozen", frozen)):
                for key, value in memory.checkpoint_arrays().items():
                    arrays[f"{name}_{policy_name}_{key}"] = value
            arm["exam"] = prior._exam(policies)
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
            "elapsed_seconds includes child execution waits; memory metrics cover the "
            "parent process including evaluator and replay ledger, excluding child memory"
        )
        output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.output, args.model_dir)
    print(_json({"terminal": result["terminal"], "output": str(args.output)}))
