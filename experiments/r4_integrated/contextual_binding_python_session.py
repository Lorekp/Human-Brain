"""Fixed paired-entity binding and separately attributed neuronal readout comparison.

All teaching uses the predecessor's discrete readout. Evaluation policy callables
share trained weights, source storage and the learned motor; none can supply
feedback. Set CPU and checkout native-cache environment before Python starts.
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
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.r4_integrated import contextual_conditioned_python_session as conditioned
from humanbrain.contextual_evidence_memory import (
    ContextualEvidenceDecision,
    ContextualEvidenceMemory,
)
from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.native import loaded_build_info

shared, prior, replay = conditioned.shared, conditioned.prior, conditioned.replay
ROOT, HERE = conditioned.ROOT, conditioned.HERE
_json = conditioned._json
_old_query_scores = conditioned._old_query_scores
SOURCE_PATHS = (
    *conditioned.SOURCE_PATHS,
    "experiments/r4_integrated/contextual_binding_cases.json",
    "experiments/r4_integrated/contextual_binding_python_session.py",
    "experiments/r4_integrated/contextual_binding_protocol.md",
)


def _hashes(source_paths=SOURCE_PATHS) -> dict[str, str]:
    return {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths}


def load_fixture(path: Path, source_paths=SOURCE_PATHS) -> dict:
    if path.resolve().relative_to(ROOT).as_posix() not in source_paths:
        raise ValueError("fixture is absent from the declared source inventory")
    fixtures = json.loads(path.read_text("utf-8"))
    if "training_source" in fixtures:
        source = fixtures["training_source"]
        if source not in source_paths:
            raise ValueError("linked teaching fixture is absent from the source inventory")
        teaching = json.loads((ROOT / source).read_text("utf-8"))
        fixtures = {**fixtures, "training": teaching["training"], "schedule": teaching["schedule"]}
    return fixtures


def _training_summary(rows) -> dict:
    result = conditioned._training_summary(rows)
    if not any(row.get("exploration_policy") == "raw_candidate" for row in rows):
        return result
    source = "raw_candidate_exploration"
    selected = [row for row in rows if row["negative_source"] == source]
    admitted = [row for row in selected if row["correction_status"] == "applied"]
    applied = sum(row["feedback"][0]["update"]["native_update_applied"] for row in admitted)
    result["negative_sources"][source] = {
        "candidate_attempts": sum(row["candidate_source"] == source for row in rows),
        "executed_negatives": len(selected),
        "admitted_pairs": len(admitted),
        "new_native_updates": applied,
        "new_noops": len(admitted) - applied,
    }
    result["below_stop_proposals_used"] = sum(row.get("proposal_below_stop", False) for row in rows)
    return result


def _snapshot(memory) -> dict:
    result = shared.snapshot(memory)
    weights = memory.checkpoint_arrays()["readout_synapses_pa"]
    difference = weights[0] - weights[1]
    result["feature_block_l2"]["candidate_interactions"] = float(np.linalg.norm(difference[1537:]))
    result["representation_policy"] = memory.representation_policy
    result["readout_policy"] = memory.readout_policy
    return result


def validate_fixture(fixtures: dict) -> None:
    training, transfer, schedule = (fixtures[key] for key in ("training", "transfer", "schedule"))
    names = [item["name"] for item in training]
    if len(training) != 16 or len(set(names)) != 16 or len(transfer) != 10:
        raise ValueError("requires sixteen unique teaching and ten transfer requests")
    if len(schedule) != 48 or any(
        Counter(schedule[i : i + 16]) != Counter(names) for i in range(0, 48, 16)
    ):
        raise ValueError("requires three complete sixteen-request schedule passes")
    docs = {}
    for item in (*training, *transfer):
        identity, content = item["document_id"], item["document"]
        if identity in docs and docs[identity] != content:
            raise ValueError("document identity has conflicting bytes")
        docs[identity] = content
        if item["value"] is not None and item["value"] == item["stale"]:
            raise ValueError("baseline already equals teacher value")
    if len({r["document_id"] for r in training}) != 4 or len(docs) != 10:
        raise ValueError("requires four teaching and six fresh documents")
    if len({r["key"] for r in training}) != 8 or len({r["value"] for r in training}) != 16:
        raise ValueError("requires eight teaching entities and sixteen distinct answer spans")
    if sum(r["value"] is not None for r in transfer) != 8:
        raise ValueError("requires eight supported and two unsupported fresh requests")


@dataclass(frozen=True)
class EvaluationPolicy:
    name: str
    memory: ContextualEvidenceMemory
    select: Callable[[bytes], ContextualEvidenceDecision]
    method: str | None

    @property
    def checkpoint_sha256(self) -> str:
        return self.memory.checkpoint_sha256

    def respond(self, query: bytes) -> ContextualEvidenceDecision:
        return self.select(query)


def evaluation_policies(update, frozen, *, binding: bool) -> tuple[EvaluationPolicy, ...]:
    policies = [
        EvaluationPolicy(name, memory, getattr(memory, method), method)
        for name, memory, method in (
            ("update", update, "respond"),
            ("frozen", frozen, "respond"),
            ("interpolated", update, "respond_interpolated"),
            ("frozen_interpolated", frozen, "respond_interpolated"),
            ("conventional", update, "conventional"),
            ("external", frozen, "external_only"),
            ("raw_score_reference", update, "raw_score_reference"),
        )
    ]
    if binding:
        for name, ablation in (
            ("disable_new_block", "disabled"),
            ("collapse_position", "position_collapsed"),
        ):
            policies.append(
                EvaluationPolicy(
                    name,
                    update,
                    partial(update.raw_score_reference, interaction_ablation=ablation),
                    None,
                )
            )
    return tuple(policies)


def _evaluate(name, update, frozen, motor, fixtures, work, arm, record, *, binding: bool):
    policies = evaluation_policies(update, frozen, binding=binding)
    for phase in ("transfer", "retention"):
        began = time.perf_counter()
        parents = {"update": update.checkpoint_sha256, "frozen": frozen.checkpoint_sha256}
        cases = fixtures["transfer" if phase == "transfer" else "training"]
        for policy in policies:
            rows = []
            for item in cases:
                row, _ = shared.attempt(
                    policy, "respond", motor, item, work / name / phase / policy.name / item["name"]
                )
                row["policy"] = policy.name
                rows.append(row)
            summary = prior._summary(rows, cases)
            arm[phase][policy.name] = {
                "summary": summary,
                "rows": rows,
                "checkpoint_sha256": policy.checkpoint_sha256,
            }
            record("evaluation", arm=name, phase=phase, policy=policy.name, summary=summary)
        if parents != {"update": update.checkpoint_sha256, "frozen": frozen.checkpoint_sha256}:
            raise RuntimeError("evaluation changed a trained or frozen checkpoint")
        for metric, left, right in (
            ("score_spike_disagreements", "conventional", "update"),
            ("raw_score_spike_disagreements", "raw_score_reference", "update"),
            ("raw_mapped_disagreements", "raw_score_reference", "conventional"),
            ("raw_interpolated_disagreements", "raw_score_reference", "interpolated"),
            ("mapped_interpolated_disagreements", "conventional", "interpolated"),
            ("discrete_interpolated_disagreements", "update", "interpolated"),
            ("frozen_readout_disagreements", "frozen", "frozen_interpolated"),
        ):
            arm[phase + "_" + metric] = sum(
                a["selection"]["candidate_index"] != b["selection"]["candidate_index"]
                for a, b in zip(arm[phase][left]["rows"], arm[phase][right]["rows"], strict=True)
            )
        arm.setdefault("readout_audit", {})[phase] = {
            "scope": "same trained checkpoint and candidate currents; readout-only policies, "
            "no updates",
            "checkpoint_sha256": parents["update"],
            "raw_correct_discrete_failed": sum(
                a["execution"]["passed"] and not b["execution"]["passed"]
                for a, b in zip(
                    arm[phase]["raw_score_reference"]["rows"],
                    arm[phase]["update"]["rows"],
                    strict=True,
                )
            ),
            "interpolated_correct_discrete_failed": sum(
                a["execution"]["passed"] and not b["execution"]["passed"]
                for a, b in zip(
                    arm[phase]["interpolated"]["rows"], arm[phase]["update"]["rows"], strict=True
                )
            ),
        }
        arm["phase_seconds"][phase] = time.perf_counter() - began
    return policies


def _teach(
    name, update, frozen, motor, fixtures, work, arm, record, *, exploration_policy="thresholded"
):
    if exploration_policy not in ("thresholded", "raw_candidate"):
        raise ValueError("unknown training exploration policy")
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
        row["after_ingestion"] = _snapshot(update)
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
        if exploration_policy == "raw_candidate":
            row["exploration_policy"] = exploration_policy
        if candidate.stopped:
            proposal = raw
            if exploration_policy == "raw_candidate":
                began = time.perf_counter()
                proposal = update.raw_candidate_proposal(item["request"].encode())
                row["candidate_proposal_seconds"] = time.perf_counter() - began
                row["candidate_proposal"] = shared.decision_record(proposal)
                row["proposal_below_stop"] = not proposal.stopped and raw.stopped
                row["candidate_proposal_drive"] = (
                    None
                    if proposal.stopped
                    else float(update._raw_drives(np.asarray([proposal.eligibility]))[0])
                )
                if update.checkpoint_sha256 != parent:
                    raise RuntimeError("training proposal changed checkpoint")
            policy = (
                "external_only"
                if proposal.stopped
                else (
                    "raw_candidate_proposal"
                    if exploration_policy == "raw_candidate"
                    else "raw_score_reference"
                )
            )
            row["fallback_kind"] = (
                "external_only"
                if proposal.stopped
                else (
                    "raw_candidate_exploration"
                    if exploration_policy == "raw_candidate"
                    else "raw_learned_score"
                )
            )
            row["exploration"], candidate = shared.attempt(
                update, policy, motor, item, folder / "exploration"
            )
            if policy != "external_only" and candidate.candidate_index != proposal.candidate_index:
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
                if row["candidate_source"] == "raw_candidate_exploration":
                    weights = update.checkpoint_arrays()["readout_synapses_pa"]
                    delta = np.asarray(accepted.eligibility) - np.asarray(candidate.eligibility)
                    row["actual_wrong_leader_gap_before"] = float((weights[0] - weights[1]) @ delta)
                    if row["actual_wrong_leader_gap_before"] > 1e-10:
                        raise RuntimeError(
                            "executed wrong raw leader did not supply a violated margin"
                        )
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
                if row["candidate_source"] == "raw_candidate_exploration":
                    receipt = row["feedback"][-1]["update"]
                    if not receipt["native_update_applied"] or not np.isclose(
                        receipt["gap_before"],
                        row["actual_wrong_leader_gap_before"],
                        rtol=1e-10,
                        atol=1e-10,
                    ):
                        raise RuntimeError("wrong-leader correction was inert or its gap changed")
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
        row["after"], row["frozen_after"] = _snapshot(update), _snapshot(frozen)
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
    if len(seen) != 4 or len(ledger) > replay.PAIR_BUDGET or replay_count > replay.REPLAY_BUDGET:
        raise RuntimeError("teaching exposure or feedback budget differs from protocol")
    return seen


def rank_diagnostics(memory, fixtures) -> dict:
    """Evaluator-only ranking/STOP separation; never used to select feedback."""
    parent, began = memory.checkpoint_sha256, time.perf_counter()
    threshold = float(np.arctanh((600.1 - 600.0) / 200.0))
    result = {
        "checkpoint_sha256": parent,
        "stop_raw_threshold": threshold,
        "scope": "read-only evaluator ranks over all retained occurrences; no updates",
    }
    for phase, cases in (("retention", fixtures["training"]), ("transfer", fixtures["transfer"])):
        records = []
        for item in cases:
            features, _ = memory._features(item["request"].encode())
            drives = memory._raw_drives(features)
            best = (
                np.flatnonzero(drives == drives.max()) if len(drives) else np.array([], dtype=int)
            )
            index = int(best[0]) if len(best) == 1 else None
            expected = None if item["value"] is None else item["value"].encode()
            correct = np.array([c.emitted == expected for c in memory._candidates], dtype=bool)
            gold = float(drives[correct].max()) if correct.any() else None
            wrong = float(drives[~correct].max()) if (~correct).any() else None
            records.append(
                {
                    "case": item["name"],
                    "unique_top_candidate_index": index,
                    "unique_top_emitted": None
                    if index is None
                    else memory._candidates[index].emitted.decode(),
                    "top_tie_count": len(best),
                    "top_drive": float(drives.max()) if len(drives) else None,
                    "correct_occurrence_count": None if expected is None else int(correct.sum()),
                    "best_correct_rank": None
                    if gold is None
                    else 1 + int(np.count_nonzero(drives > gold)),
                    "best_correct_drive": gold,
                    "best_correct_minus_best_wrong": None
                    if gold is None or wrong is None
                    else gold - wrong,
                    "best_correct_minus_stop": None if gold is None else gold - threshold,
                }
            )
        result[phase] = records
    if memory.checkpoint_sha256 != parent:
        raise RuntimeError("final ranking diagnostic changed checkpoint")
    result["seconds"] = time.perf_counter() - began
    return result


def run(
    output: Path,
    model_dir: Path,
    *,
    fixture_path: Path | None = None,
    exploration_policy="thresholded",
    source_paths=SOURCE_PATHS,
    report_labels: dict[str, str] | None = None,
    include_rank_diagnostics=False,
) -> dict:
    if exploration_policy not in ("thresholded", "raw_candidate"):
        raise ValueError("unknown training exploration policy")
    fixture_path = HERE / "contextual_binding_cases.json" if fixture_path is None else fixture_path
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
    work = Path(tempfile.mkdtemp(prefix="contextual-binding-", dir=ROOT / ".runtime"))
    started = time.perf_counter()
    shared.tracemalloc.start()
    report = {
        "kind": "public development; fixed gain16 baseline versus candidate-relative binding",
        "terminal": "running",
        "workspace_root": str(work),
        "arms": {},
        "events": [],
        "native_environment": {
            key: os.environ[key] for key in ("HUMANBRAIN_DEVICE", "HUMANBRAIN_NATIVE_CACHE")
        },
        "teaching_exposure": "sixteen new requests over four paired-entity documents; "
        "fixed 48-encounter schedule and fresh transfer fixture",
        "feedback_contract": "only executed FAIL/teacher PASS pairs; certified replay; after "
        "neuronal STOP prefer raw learned-score winner, then external cosine",
        "retained_source_policy": "pretrained frozen MiniLM plus external raw documents, "
        "candidate vectors, storage ancestry and immutable correction ledger",
        "controls": "same-gain frozen mate receives exact feedback; all references share "
        "learned motor; updating arms may collect different on-policy failures",
    }
    if report_labels:
        if set(report_labels) - {"kind", "teaching_exposure", "feedback_contract"}:
            raise ValueError("only descriptive report labels can be overridden")
        report.update(report_labels)
    if exploration_policy != "thresholded":
        report["exploration_policy"] = exploration_policy

    def record(event, **fields):
        row = {"event": event, **fields}
        report["events"].append(row)
        with (work / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(row) + "\n")
        print(_json(row), flush=True)

    try:
        report["source_sha256"] = _hashes(source_paths)
        report["base_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        fixtures = load_fixture(fixture_path, source_paths)
        validate_fixture(fixtures)
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
        for name, binding in (("GAIN16", False), ("BINDING", True)):
            gain = 16.0
            update = ContextualEvidenceMemory(
                encoder, learning_rate=1.0, interaction_gain=gain, candidate_interactions=binding
            )
            frozen = ContextualEvidenceMemory(
                encoder,
                learning_rate=1.0,
                interaction_gain=gain,
                learning_enabled=False,
                candidate_interactions=binding,
            )
            if update.checkpoint_sha256 != frozen.checkpoint_sha256:
                raise RuntimeError("paired initial checkpoints differ")
            arm = {
                "interaction_gain": gain,
                "candidate_interactions": binding,
                "learning_rate": 1.0,
                "seed": _snapshot(update),
                "training": [],
                "ledger": [],
                "transfer": {},
                "retention": {},
                "phase_seconds": {},
            }
            report["arms"][name] = arm
            began = time.perf_counter()
            seen = _teach(
                name,
                update,
                frozen,
                motor,
                fixtures,
                work,
                arm,
                record,
                exploration_policy=exploration_policy,
            )
            arm["phase_seconds"]["teaching"] = time.perf_counter() - began
            arm["trained"], arm["frozen_trained"] = _snapshot(update), _snapshot(frozen)
            began = time.perf_counter()
            for item in fixtures["transfer"]:
                prior._ingest_once(item, (update, frozen), seen)
            arm["phase_seconds"]["fresh_ingestion"] = time.perf_counter() - began
            if len(seen) != 10:
                raise RuntimeError("fresh source exposure differs from protocol")
            arm["transfer_start"] = _snapshot(update)
            began = time.perf_counter()
            policies = _evaluate(
                name, update, frozen, motor, fixtures, work, arm, record, binding=binding
            )
            arm["phase_seconds"]["transfer_and_retention"] = time.perf_counter() - began
            if include_rank_diagnostics:
                arm["rank_diagnostics"] = rank_diagnostics(update, fixtures)
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
            arm["exam"] = prior._exam(
                [(p.name, p.memory, p.method) for p in policies if p.method is not None]
            )
            arm["phase_seconds"]["exam"] = time.perf_counter() - began
            arm["end"], arm["frozen_end"] = _snapshot(update), _snapshot(frozen)
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
        report["sources_unchanged_during_run"] = _hashes(source_paths) == report["source_sha256"]
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
