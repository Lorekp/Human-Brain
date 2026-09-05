"""Fixed paired-entity binding and separately attributed neuronal readout comparison.

Default teaching uses the predecessor's discrete readout. Evaluation policy callables
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


def load_teaching_fixtures(fixtures, paths, source_paths, arm_specs) -> dict:
    """Validate arm renditions against common probes before allocating learners."""
    if paths is None:
        return {}
    if arm_specs is None or set(paths) != {spec[0] for spec in arm_specs}:
        raise ValueError("teaching fixtures must name every explicit arm exactly once")
    common = {item["name"]: item for item in fixtures["training"]}
    result = {}
    for name, path in paths.items():
        teaching = load_fixture(path, source_paths)
        candidate = {
            **fixtures,
            "training": teaching["training"],
            "schedule": teaching["schedule"],
        }
        validate_fixture(candidate)
        if candidate["schedule"] != fixtures["schedule"]:
            raise ValueError("teaching arms must preserve the common schedule")
        for item in candidate["training"]:
            reference = common.get(item["name"])
            if reference is None or {
                key: value for key, value in item.items() if key not in ("document", "request")
            } != {
                key: value for key, value in reference.items() if key not in ("document", "request")
            }:
                raise ValueError("teaching arms may change only document and request text")
        result[name] = candidate
    return result


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
    result["feature_block_l2"]["candidate_interactions"] = float(
        np.linalg.norm(difference[1537:1621])
    )
    result["representation_policy"] = memory.representation_policy
    result["readout_policy"] = memory.readout_policy
    if memory.projected_bilinear:
        result["projection"] = memory.projection_summary
        result["feature_block_l2"]["projected_bilinear"] = float(np.linalg.norm(difference[1621:]))
    if memory.comparison_step_policy == "arow":
        result["covariance"] = memory.covariance_summary
    return result


def _score_transitions(before, after) -> dict:
    """Compare old-query probes only within the same retained source context."""
    pairs = list(zip(before["rows"], after["rows"], strict=True))
    if any(
        a["case"] != b["case"]
        or a["selection"]["storage_sha256"] != b["selection"]["storage_sha256"]
        for a, b in pairs
    ):
        raise RuntimeError("old-query probes have incomparable source context")
    return {
        "scope": "evaluator-only thresholded raw selection changes, including STOP; "
        "same queries and retained source context",
        "before_checkpoint_sha256": before["checkpoint_sha256"],
        "after_checkpoint_sha256": after["checkpoint_sha256"],
        "compared_queries": len(pairs),
        "compared_storage_sha256": sorted({a["selection"]["storage_sha256"] for a, _ in pairs}),
        "correct_to_wrong": [a["case"] for a, b in pairs if a["correct"] and not b["correct"]],
        "wrong_to_correct": [a["case"] for a, b in pairs if not a["correct"] and b["correct"]],
        "correct_to_stop": [
            a["case"]
            for a, b in pairs
            if a["correct"] and b["selection"]["candidate_index"] is None
        ],
        "stop_to_correct": [
            a["case"]
            for a, b in pairs
            if a["selection"]["candidate_index"] is None and b["correct"]
        ],
    }


def _pair_diagnostics(memory, arm, fixtures) -> dict:
    """Final saved-pair fit and realized negative coverage; no feedback decisions."""
    parent, began = memory.checkpoint_sha256, time.perf_counter()
    weights = memory.checkpoint_arrays()["readout_synapses_pa"]
    difference = weights[0] - weights[1]
    events = [event for row in arm["training"] for event in row["feedback"]]
    if sha256(replay._json(arm["ledger"]).encode()).hexdigest() != arm["ledger_summary"]["sha256"]:
        raise RuntimeError("final pair diagnosis ledger hash differs")
    pairs = []
    for pair in arm["ledger"]:
        rejected, accepted = pair["rejected_decision"], pair["accepted_decision"]
        delta = np.asarray(accepted["eligibility"]) - np.asarray(rejected["eligibility"])
        offers = [event for event in events if event["pair_id"] == pair["pair_id"]]
        if not offers:
            raise RuntimeError("final pair diagnosis lacks original feedback offer")
        pairs.append(
            {
                "pair_id": pair["pair_id"],
                "query": rejected["query"],
                "rejected_candidate_index": rejected["candidate_index"],
                "accepted_candidate_index": accepted["candidate_index"],
                "final_margin": float(difference @ delta),
                "first_gap_after": offers[0]["update"]["gap_after"],
                "last_gap_after": offers[-1]["update"]["gap_after"],
                "offer_count": len(offers),
            }
        )
    queries = []
    execution = {
        row["case"]: row["execution"]["passed"] for row in arm["retention"]["affine"]["rows"]
    }
    for item, rank in zip(fixtures["training"], arm["rank_diagnostics"]["retention"], strict=True):
        matching = [pair for pair in pairs if pair["query"] == item["request"]]
        negatives = {pair["rejected_candidate_index"] for pair in matching}
        leader = rank["unique_top_candidate_index"]
        executed_steps = []
        for encounter in arm["training"]:
            if encounter["case"] != item["name"]:
                continue
            action = encounter.get("exploration", encounter["first"])
            if (
                action["edited"]
                and action["selection"]["query"] == item["request"]
                and not action["execution"]["passed"]
                and action["selection"]["candidate_index"] == leader
            ):
                executed_steps.append(encounter["step"])
        admitted_leader_pairs = [
            {"pair_id": pair["pair_id"], "final_margin": pair["final_margin"]}
            for pair in matching
            if pair["rejected_candidate_index"] == leader
        ]
        queries.append(
            {
                "case": item["name"],
                "pair_count": len(matching),
                "distinct_executed_negative_occurrences": len(negatives),
                "unique_top_candidate_index": rank["unique_top_candidate_index"],
                "top_was_executed_negative": rank["unique_top_candidate_index"] in negatives,
                "final_leader_was_executed_failure": bool(executed_steps),
                "final_leader_executed_failure_steps": executed_steps,
                "final_leader_was_admitted_negative": bool(admitted_leader_pairs),
                "final_leader_admitted_negative_pairs": admitted_leader_pairs,
                "all_recorded_pairs_fit": None
                if not matching
                else all(pair["final_margin"] >= 1 - 1e-10 for pair in matching),
                "best_correct_rank": rank["best_correct_rank"],
                "affine_executed_success": execution[item["name"]],
                "has_erased_pair": any(
                    pair["first_gap_after"] > 0 and pair["final_margin"] < 0 for pair in matching
                ),
            }
        )
    if memory.checkpoint_sha256 != parent:
        raise RuntimeError("pair diagnosis changed checkpoint")
    return {
        "scope": "evaluator-only final trained-pair margins and realized on-policy coverage",
        "checkpoint_sha256": parent,
        "pairs": pairs,
        "negative_pair_margins": sum(pair["final_margin"] < 0 for pair in pairs),
        "nonnegative_subunit_pair_margins": sum(
            0 <= pair["final_margin"] < 1 - 1e-10 for pair in pairs
        ),
        "satisfied_pair_margins": sum(pair["final_margin"] >= 1 - 1e-10 for pair in pairs),
        "queries": queries,
        "queries_without_pairs": sum(row["pair_count"] == 0 for row in queries),
        "failed_queries_all_pairs_fit": sum(
            not row["affine_executed_success"] and row["all_recorded_pairs_fit"] is True
            for row in queries
        ),
        "failed_queries_with_erased_pair": sum(
            not row["affine_executed_success"] and row["has_erased_pair"] for row in queries
        ),
        "failed_queries_without_pairs": sum(
            not row["affine_executed_success"] and row["pair_count"] == 0 for row in queries
        ),
        "seconds": time.perf_counter() - began,
    }


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


def evaluation_policies(
    update, frozen, *, binding: bool, include_affine=False
) -> tuple[EvaluationPolicy, ...]:
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
    if update.projected_bilinear:
        for name, ablation in (
            ("disable_projected_bilinear", "disabled"),
            ("diagonal_projected_bilinear", "diagonal_only"),
        ):
            policies.append(
                EvaluationPolicy(
                    name,
                    update,
                    partial(update.raw_score_reference, projected_bilinear_ablation=ablation),
                    None,
                )
            )
    if include_affine:
        policies.extend(
            EvaluationPolicy(
                name, memory, memory.respond_affine_interpolated, "respond_affine_interpolated"
            )
            for name, memory in (("affine", update), ("frozen_affine", frozen))
        )
    return tuple(policies)


def _evaluate(
    name,
    update,
    frozen,
    motor,
    fixtures,
    work,
    arm,
    record,
    *,
    binding: bool,
    include_affine=False,
):
    policies = evaluation_policies(update, frozen, binding=binding, include_affine=include_affine)
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
        if include_affine:
            arm[phase + "_raw_affine_disagreements"] = sum(
                a["selection"]["candidate_index"] != b["selection"]["candidate_index"]
                for a, b in zip(
                    arm[phase]["raw_score_reference"]["rows"],
                    arm[phase]["affine"]["rows"],
                    strict=True,
                )
            )
    return policies


def _teach(
    name,
    update,
    frozen,
    motor,
    fixtures,
    work,
    arm,
    record,
    *,
    exploration_policy="thresholded",
    primary_policy="respond",
    track_learning_diagnostics=False,
    allocation_policy="baseline",
):
    if exploration_policy not in ("thresholded", "raw_candidate"):
        raise ValueError("unknown training exploration policy")
    if allocation_policy not in ("baseline", "reserve_future_corrections"):
        raise ValueError("unknown feedback allocation policy")
    pair_limit = replay.PAIR_BUDGET if allocation_policy == "baseline" else 48
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
        row["frozen_first"], _ = shared.attempt(
            frozen, primary_policy, motor, item, folder / "frozen"
        )
        row["first"], candidate = shared.attempt(
            update, primary_policy, motor, item, folder / "first"
        )
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
        elif len(ledger) == pair_limit:
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
                if track_learning_diagnostics:
                    row["old_query_scores_before_new"] = _old_query_scores(
                        update, encountered.values()
                    )
                row["feedback"].append(replay._feedback(update, frozen, pair, replay=False))
                if row["candidate_source"] == "raw_candidate_exploration":
                    receipt = row["feedback"][-1]["update"]
                    protected_noop = (
                        arm.get("comparison_protection_policy") == "orthogonal_certified"
                        and receipt["protection"]["projection_exhausted"]
                    )
                    if (
                        not receipt["native_update_applied"] and not protected_noop
                    ) or not np.isclose(
                        receipt["gap_before"],
                        row["actual_wrong_leader_gap_before"],
                        rtol=1e-10,
                        atol=1e-10,
                    ):
                        raise RuntimeError(
                            "wrong-leader correction was unexpectedly inert or its gap changed"
                        )
                ledger.append(pair)
                arm["ledger"].append(pair.record())
                row["correction_status"] = "applied"
                row["old_query_scores_after_new"] = _old_query_scores(update, encountered.values())
        admitted = row["correction_status"] == "applied"
        older = len(ledger) - int(admitted)
        if allocation_policy == "baseline":
            count = min(4, older, replay.REPLAY_BUDGET - replay_count) if admitted else 0
        else:
            allocation = _reserved_replay_allowance(step, len(ledger), replay_count, admitted)
            count = allocation["replay_offers"]
            allocation["trigger"] = (
                "after_new_correction" if admitted else "after_" + row["correction_status"]
            )
            allocation["timing"] = "after primary/fallback and existing teacher execution"
            row["allocation"] = allocation
        before_rehearsal = row.get("old_query_scores_after_new")
        if count and before_rehearsal is None and track_learning_diagnostics:
            before_rehearsal = _old_query_scores(update, encountered.values())
            row["old_query_scores_before_rehearsal"] = before_rehearsal
        for offset in range(count):
            previous = ledger[(replay_cursor + offset) % older]
            row["feedback"].append(replay._feedback(update, frozen, previous, replay=True))
        if arm.get("comparison_protection_policy") == "orthogonal_certified":
            for event in row["feedback"]:
                for mode in ("update", "frozen"):
                    projection = event[mode]["protection"]
                    # NumPy comparisons return scalar flags; keep report counts
                    # and raw receipts JSON-native without changing their values.
                    projection["projection_exhausted"] = bool(projection["projection_exhausted"])
                    projection["projection_stalled"] = bool(projection["projection_stalled"])
                _verify_protected_feedback(event)
        replay_count += count
        if older and (admitted or count):
            replay_cursor = (replay_cursor + count) % older
        if count:
            row["old_query_scores_after_rehearsal"] = _old_query_scores(
                update, encountered.values()
            )
        if track_learning_diagnostics and (admitted or count):
            transitions = {}
            if admitted:
                transitions["new"] = _score_transitions(
                    row["old_query_scores_before_new"], row["old_query_scores_after_new"]
                )
            transitions["rehearsal"] = _score_transitions(
                before_rehearsal, row.get("old_query_scores_after_rehearsal", before_rehearsal)
            )
            row["old_query_transitions"] = transitions
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
    if arm.get("comparison_protection_policy") == "orthogonal_certified":
        arm["protection_feedback_summary"] = {
            "offers": len(events),
            "projection_exhausted_offers": sum(
                event["update"]["protection"]["projection_exhausted"] for event in events
            ),
            "projection_stalled_offers": sum(
                event["update"]["protection"]["projection_stalled"] for event in events
            ),
            "new_consolidations": sum(
                event["update"]["protection"]["consolidated"] for event in events
            ),
            "new_basis_directions": sum(
                event["update"]["protection"]["independent_direction_added"] for event in events
            ),
            "protected_margin_checks": sum(
                event["update"]["protection"]["protected_margin_checks"] for event in events
            ),
            "protected_margin_violations": sum(
                event["update"]["protection"]["protected_margin_violations"] for event in events
            ),
            "no_op_reasons": dict(
                Counter(
                    event["update"]["step_skipped_reason"]
                    for event in events
                    if not event["update"]["synaptic_update"]
                )
            ),
            "scope": "original competitive gate and normalization; projected native direction; "
            "exhausted projections consume offers without a synaptic update",
        }
    if allocation_policy != "baseline":
        arm["ledger_summary"].update(
            allocation_policy=allocation_policy,
            offer_budget=150,
            offers_used=len(ledger) + replay_count,
            unused_offer_budget=150 - len(ledger) - replay_count,
            max_new_pairs=48,
            fifo_policy="append certified pairs in admission order; at most48; no padding",
        )
    invalid_replays = (
        replay_count > replay.REPLAY_BUDGET
        if allocation_policy == "baseline"
        else len(ledger) + replay_count > 150
    )
    if len(seen) != 4 or len(ledger) > pair_limit or invalid_replays:
        raise RuntimeError("teaching exposure or feedback budget differs from protocol")
    return seen


def _verify_protected_feedback(event) -> None:
    """Validate recorded own-gap arithmetic without reconstructing or choosing an update."""
    for mode in ("update", "frozen"):
        receipt = event[mode]
        protection = receipt["protection"]
        eta = receipt["effective_learning_rate"]
        norm = receipt["eligibility_delta_norm_squared"]
        change = 2 * eta * protection["original_dot_projected"] / np.sqrt(1 + norm)
        if not np.isclose(
            receipt["gap_after"], receipt["gap_before"] + change, rtol=1e-10, atol=1e-10
        ):
            raise RuntimeError("projected update violated the original-margin arithmetic")
        if protection["projection_exhausted"] and (
            receipt["synaptic_update"] or receipt["native_update_applied"] or eta != 0
        ):
            raise RuntimeError("exhausted projection reported a synaptic update")
        if mode == "frozen" and (
            receipt["synaptic_update"] or receipt["native_update_applied"] or eta != 0
        ):
            raise RuntimeError("frozen projection control changed synapses")


def _reserved_replay_allowance(step, pair_count, replay_count, admitted) -> dict:
    """Budget arithmetic only; no query, reward, rank or eligibility input."""
    remaining = 48 - step
    used = pair_count + replay_count
    free = 150 - used - remaining
    older = pair_count - int(admitted)
    if not 1 <= step <= 48 or free < 0 or older < 0:
        raise RuntimeError("feedback reservation invariant failed")
    return {
        "policy": "reserve_future_corrections",
        "offer_budget": 150,
        "used_before_replay": used,
        "remaining_encounters": remaining,
        "reserved_for_future_corrections": remaining,
        "free_replay_offers": free,
        "eligible_old_pairs": older,
        "replay_offers": min(4, older, free),
    }


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
    arm_specs=None,
    primary_policy="respond",
    include_affine=False,
    track_learning_diagnostics=False,
    allocation_policies: dict[str, str] | None = None,
    protection_policies: dict[str, str] | None = None,
    track_source_competition=False,
    teaching_fixture_paths: dict[str, Path] | None = None,
    projected_bilinear_policies: dict[str, bool] | None = None,
) -> dict:
    if exploration_policy not in ("thresholded", "raw_candidate"):
        raise ValueError("unknown training exploration policy")
    if primary_policy not in ("respond", "respond_affine_interpolated"):
        raise ValueError("unknown primary neuronal policy")
    if primary_policy == "respond_affine_interpolated" and not include_affine:
        raise ValueError("affine teaching requires declared affine evaluation")
    if track_learning_diagnostics and not (include_rank_diagnostics and include_affine):
        raise ValueError("pair coverage requires final rank diagnostics and affine evaluation")
    if track_learning_diagnostics and not include_affine:
        raise ValueError("pair coverage requires affine executed outcomes")
    if track_source_competition and not include_rank_diagnostics:
        raise ValueError("source competition requires final raw rank diagnostics")
    if protection_policies is not None:
        if arm_specs is None or set(protection_policies) != {spec[0] for spec in arm_specs}:
            raise ValueError("protection policies must name every explicit arm exactly once")
        if set(protection_policies.values()) - {"none", "orthogonal_certified"}:
            raise ValueError("unknown comparison protection policy")
    if projected_bilinear_policies is not None:
        if arm_specs is None or set(projected_bilinear_policies) != {spec[0] for spec in arm_specs}:
            raise ValueError("projected bilinear policies must name every explicit arm")
        if any(type(value) is not bool for value in projected_bilinear_policies.values()):
            raise ValueError("projected bilinear policies must be boolean")
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
    if primary_policy != "respond":
        report["primary_teaching_policy"] = primary_policy
        report["primary_evaluation_policy"] = "affine"

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
        teaching_fixtures = load_teaching_fixtures(
            fixtures, teaching_fixture_paths, source_paths, arm_specs
        )
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
        specs = (
            (("GAIN16", False, "fixed"), ("BINDING", True, "fixed"))
            if arm_specs is None
            else arm_specs
        )
        for name, binding, step_policy in specs:
            gain = 16.0
            step_options = {} if arm_specs is None else {"comparison_step_policy": step_policy}
            if protection_policies is not None:
                step_options["comparison_protection_policy"] = protection_policies[name]
                if protection_policies[name] != "none":
                    step_options["protection_capacity"] = 48
            if projected_bilinear_policies is not None:
                step_options["projected_bilinear"] = projected_bilinear_policies[name]
            update = ContextualEvidenceMemory(
                encoder,
                learning_rate=1.0,
                interaction_gain=gain,
                candidate_interactions=binding,
                **step_options,
            )
            frozen = ContextualEvidenceMemory(
                encoder,
                learning_rate=1.0,
                interaction_gain=gain,
                learning_enabled=False,
                candidate_interactions=binding,
                **step_options,
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
            arm_teaching = teaching_fixtures.get(name, fixtures)
            if teaching_fixture_paths is not None:
                arm["teaching_fixture"] = {
                    "path": teaching_fixture_paths[name].resolve().relative_to(ROOT).as_posix(),
                    "sha256": sha256(teaching_fixture_paths[name].read_bytes()).hexdigest(),
                    "common_probe_scope": "same common requests for both arms; probe document "
                    "bytes are never ingested in place of the arm's teaching rendition",
                }
            if projected_bilinear_policies is not None:
                arm["projected_bilinear"] = projected_bilinear_policies[name]
            if arm_specs is not None:
                arm["comparison_step_policy"] = step_policy
            if protection_policies is not None:
                arm["comparison_protection_policy"] = protection_policies[name]
            allocation = "baseline" if allocation_policies is None else allocation_policies[name]
            if allocation_policies is not None:
                arm["allocation_policy"] = allocation
            began = time.perf_counter()
            seen = _teach(
                name,
                update,
                frozen,
                motor,
                arm_teaching,
                work,
                arm,
                record,
                exploration_policy=exploration_policy,
                primary_policy=primary_policy,
                track_learning_diagnostics=track_learning_diagnostics,
                allocation_policy=allocation,
            )
            arm["phase_seconds"]["teaching"] = time.perf_counter() - began
            arm["trained"], arm["frozen_trained"] = _snapshot(update), _snapshot(frozen)
            covariance_state = None
            if step_policy == "arow":
                covariance_state = {
                    "update": update.covariance_summary,
                    "frozen": frozen.covariance_summary,
                }
                if covariance_state["frozen"] != arm["seed"]["covariance"]:
                    raise RuntimeError("frozen control changed its initial covariance")
                arm["covariance_after_teaching"] = covariance_state
            if teaching_fixture_paths is not None:
                arm["teaching_probe_after_training"] = {
                    **_old_query_scores(update, arm_teaching["training"]),
                    "scope": "read-only raw selection on this arm's actual TRAIN requests; "
                    "not executed success or feedback",
                }
            protected_state = None
            if protection_policies is not None:
                protected_state = {
                    "update": update.protection_summary,
                    "frozen": frozen.protection_summary,
                }
                if protection_policies[name] != "none" and (
                    protected_state["frozen"]["basis_rank"] != 0
                    or protected_state["frozen"]["protected_count"] != 0
                ):
                    raise RuntimeError("frozen control acquired learned protection state")
                arm["protection_after_teaching"] = protected_state
            if track_source_competition:
                began = time.perf_counter()
                arm["source_competition"] = {
                    "scope": "evaluator-only same-weight retention before/after fresh ingestion; "
                    "no action or feedback; fresh competitors remain visible",
                    "synapse_sha256_before": update.synapse_sha256,
                    "before_fresh": _old_query_scores(update, fixtures["training"]),
                    "ranks_before_fresh": rank_diagnostics(
                        update, {"training": fixtures["training"], "transfer": []}
                    )["retention"],
                }
                arm["phase_seconds"]["source_competition_before_fresh"] = (
                    time.perf_counter() - began
                )
            began = time.perf_counter()
            for item in fixtures["transfer"]:
                prior._ingest_once(item, (update, frozen), seen)
            arm["phase_seconds"]["fresh_ingestion"] = time.perf_counter() - began
            if len(seen) != 10:
                raise RuntimeError("fresh source exposure differs from protocol")
            arm["transfer_start"] = _snapshot(update)
            if covariance_state is not None and covariance_state != {
                "update": update.covariance_summary,
                "frozen": frozen.covariance_summary,
            }:
                raise RuntimeError("fresh ingestion changed learned covariance")
            if protected_state is not None and protected_state != {
                "update": update.protection_summary,
                "frozen": frozen.protection_summary,
            }:
                raise RuntimeError("fresh ingestion changed learned protection state")
            if track_source_competition:
                began = time.perf_counter()
                competition = arm["source_competition"]
                competition["after_fresh"] = _old_query_scores(update, fixtures["training"])
                if projected_bilinear_policies is not None:
                    arm["teaching_probe_after_fresh"] = {
                        **_old_query_scores(update, arm_teaching["training"]),
                        "scope": "read-only actual-TRAIN query selection after fresh ingestion; "
                        "not executed success or feedback",
                    }
                competition["synapse_sha256_after"] = update.synapse_sha256
                if competition["synapse_sha256_before"] != competition["synapse_sha256_after"]:
                    raise RuntimeError("fresh ingestion changed the trained synapses")
                arm["phase_seconds"]["source_competition_after_fresh"] = time.perf_counter() - began
            began = time.perf_counter()
            policies = _evaluate(
                name,
                update,
                frozen,
                motor,
                fixtures,
                work,
                arm,
                record,
                binding=binding,
                include_affine=include_affine,
            )
            arm["phase_seconds"]["transfer_and_retention"] = time.perf_counter() - began
            if include_rank_diagnostics:
                arm["rank_diagnostics"] = rank_diagnostics(update, fixtures)
            if track_source_competition:
                competition["ranks_after_fresh"] = arm["rank_diagnostics"]["retention"]
            if track_learning_diagnostics:
                arm["pair_diagnostics"] = _pair_diagnostics(update, arm, fixtures)
                if teaching_fixture_paths is not None:
                    arm["pair_diagnostics"]["scope"] = (
                        "original actual-TRAIN pair margins; query coverage matches exact "
                        "common-probe wording only, so zero pairs does not mean the fact "
                        "was never taught"
                    )
            if covariance_state is not None and covariance_state != {
                "update": update.covariance_summary,
                "frozen": frozen.covariance_summary,
            }:
                raise RuntimeError("evaluation changed learned covariance")
            if protected_state is not None and protected_state != {
                "update": update.protection_summary,
                "frozen": frozen.protection_summary,
            }:
                raise RuntimeError("evaluation changed learned protection state")
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
                state_options = {"include_learning_state": True} if step_policy == "arow" else {}
                for key, value in memory.checkpoint_arrays(**state_options).items():
                    arrays[f"{name}_{policy_name}_{key}"] = value
            began = time.perf_counter()
            arm["exam"] = prior._exam(
                [(p.name, p.memory, p.method) for p in policies if p.method is not None]
            )
            if covariance_state is not None:
                arm["covariance_after_exam"] = {
                    "update": update.covariance_summary,
                    "frozen": frozen.covariance_summary,
                }
                if covariance_state != arm["covariance_after_exam"]:
                    raise RuntimeError("exam changed learned covariance")
            if protected_state is not None:
                arm["protection_after_exam"] = {
                    "update": update.protection_summary,
                    "frozen": frozen.protection_summary,
                }
                if protected_state != arm["protection_after_exam"]:
                    raise RuntimeError("exam changed learned protection state")
            arm["phase_seconds"]["exam"] = time.perf_counter() - began
            arm["end"], arm["frozen_end"] = _snapshot(update), _snapshot(frozen)
            if update.projected_bilinear and any(
                snapshot["projection"] != arm["seed"]["projection"]
                for snapshot in (
                    arm["trained"], arm["frozen_trained"], arm["end"], arm["frozen_end"]
                )
            ):
                raise RuntimeError("fixed projection identity changed during the session")
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
        if any(spec[2] == "arow" for spec in specs):
            report["synapse_checkpoint"]["scope"] += (
                "; includes full trained and fixed-initial AROW covariance; "
                "covariance is training state, not used for inference"
            )
        if projected_bilinear_policies and any(projected_bilinear_policies.values()):
            report["synapse_checkpoint"]["scope"] += "; includes fixed projection matrices"
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
