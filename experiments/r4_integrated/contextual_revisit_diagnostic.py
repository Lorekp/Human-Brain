"""Receipt-only occurrence, capacity, and forgetting audit for revisitation.

Replay each recorded update once to verify checkpoint reconstruction. Score only
original executed contrasts; repeated feedback is exposure, not new constraints.
HiGHS feasibility references are conventional diagnostics, never capability runs.
No transfer labels, new negatives, fresh tasks, or protected inputs are used.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from graphlib import CycleError, TopologicalSorter
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy
from experiments.r4_integrated.contextual_pairwise_diagnostic import BLOCKS, digest
from experiments.r4_integrated.contextual_pairwise_diagnostic import measure as _measure
from humanbrain.contextual_evidence_memory import ContextualEvidenceMemory
from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.native import loaded_build_info
from humanbrain.opaque_document_event_tape import OpaqueDocument
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "experiments/r4_integrated/contextual_revisit_diagnostic.py",
    "experiments/r4_integrated/contextual_pairwise_diagnostic.py",
    "src/humanbrain/contextual_evidence_memory.py",
    "src/humanbrain/frozen_context_encoder.py",
    "src/humanbrain/opaque_document_event_tape.py",
    "src/humanbrain/native.py",
    "src/humanbrain/resting_glif_race.py",
)
TOLERANCE = 1e-7


def measure(deltas, weights):
    result = _measure(deltas, weights)
    result["margin_violations_below_one"] = int(
        np.count_nonzero(np.asarray(result["margins"]) < 1 - TOLERANCE)
    )
    return result


def occurrence(memory, index):
    candidate = memory._candidates[index]
    document = memory._documents[candidate.document_index]
    return json.dumps(
        [
            document.document_id.hex(),
            document.provenance_id.hex(),
            sha256(document.content).hexdigest(),
            candidate.start,
            candidate.end,
        ],
        separators=(",", ":"),
    )


def graph_diagnosis(pairs):
    predecessors, queries = {}, {}
    for pair in pairs:
        rejected, accepted = pair["rejected_occurrence"], pair["accepted_occurrence"]
        predecessors.setdefault(rejected, set())
        predecessors.setdefault(accepted, set()).add(rejected)
        queries.setdefault((rejected, accepted), set()).add(pair["query"])
    opposites = []
    for (left, right), forward in sorted(queries.items()):
        if left < right and (right, left) in queries:
            reverse = queries[right, left]
            opposites.append(
                {
                    "occurrences": [left, right],
                    "forward_queries": sorted(forward),
                    "reverse_queries": sorted(reverse),
                    "distinct_query_reversal": any(a != b for a in forward for b in reverse),
                    "same_query_conflict": bool(forward & reverse),
                }
            )
    try:
        ordered = {k: sorted(v) for k, v in sorted(predecessors.items())}
        order = list(TopologicalSorter(ordered).static_order())
        cycle = None
    except CycleError as error:
        order, cycle = None, error.args[1]
    return {
        "node_count": len(predecessors),
        "unique_directed_edges": len(queries),
        "acyclic": cycle is None,
        "cycle": cycle,
        "topological_order": order,
        "opposite_edges": opposites,
        "query_dependent_opposite_edge_count": sum(p["distinct_query_reversal"] for p in opposites),
        "query_independent_strict_order_impossible": cycle is not None,
        "scope": "identical document/provenance/content/byte-span occurrences, not emitted strings",
    }


def feasible(deltas, columns):
    """Numerical hard-margin feasibility; solver failures remain inconclusive."""
    began = time.perf_counter()
    selected = deltas[:, columns]
    nonzero = np.any(selected != 0, axis=0)
    selected = selected[:, nonzero]
    if not len(deltas):
        return {"verdict": "no_constraints", "constraint_count": 0}
    if selected.shape[1] == 0:
        return {"verdict": "infeasible_zero_features", "constraint_count": len(deltas)}
    result = linprog(
        np.zeros(selected.shape[1]),
        A_ub=-selected,
        b_ub=-np.ones(len(deltas)),
        bounds=(None, None),
        method="highs",
        options={
            "time_limit": 30.0,
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    report = {
        "solver": "scipy.optimize.linprog/highs",
        "scipy_version": scipy.__version__,
        "time_limit_seconds": 30,
        "status": int(result.status),
        "message": result.message,
        "constraint_count": len(deltas),
        "active_feature_count": selected.shape[1],
        "verdict": "inconclusive",
        "seconds": time.perf_counter() - began,
    }
    if result.status == 2:
        report["verdict"] = "solver_reports_infeasible"
    elif result.status == 0 and result.x is not None:
        margins = selected @ result.x
        if np.all(np.isfinite(result.x)) and np.min(margins) >= 1 - TOLERANCE:
            full = np.zeros(deltas.shape[1])
            full[np.asarray(columns)[nonzero]] = result.x
            report.update(verdict="feasible_verified", **measure(deltas, full))
            report["solution_sha256"] = sha256(full.tobytes()).hexdigest()
    return report


def geometry(pairs, deltas):
    unique = np.unique(deltas, axis=0)
    ranges = {}
    for name, left, right in BLOCKS:
        norms = np.linalg.norm(unique[:, left:right], axis=1)
        ranges[name] = (
            {
                "min": float(norms.min()),
                "median": float(np.median(norms)),
                "max": float(norms.max()),
            }
            if len(norms)
            else None
        )
    reversals = []
    for i, first in enumerate(pairs):
        for j in range(i + 1, len(pairs)):
            second = pairs[j]
            if (
                first["query"] == second["query"]
                or first["rejected_occurrence"] != second["accepted_occurrence"]
                or first["accepted_occurrence"] != second["rejected_occurrence"]
            ):
                continue
            total = deltas[i] + deltas[j]
            norm = float(np.linalg.norm(total))
            reversals.append(
                {
                    "pair_ids": [first["pair_id"], second["pair_id"]],
                    "sum_delta_norm": norm,
                    "sum_conditional_components_norm": float(np.linalg.norm(total[768:])),
                    "sum_block_norms": {
                        name: float(np.linalg.norm(total[left:right]))
                        for name, left, right in BLOCKS
                    },
                    "minimum_weight_difference_norm_for_two_unit_margins": (
                        2 / norm if norm > 0 else None
                    ),
                    "contradictory_zero_sum": norm == 0,
                }
            )
    return {
        "unique_delta_block_norm_ranges": ranges,
        "distinct_query_direct_reversals": reversals,
        "bound_scope": "Cauchy necessary lower bound 2/||d1+d2|| for both unit margins; "
        "not sufficient, not a fitted weight norm or capability result",
    }


def reconstruct_decision(memory, record):
    query = record["query"].encode()
    rows, _ = memory._features(query)
    index = record["candidate_index"]
    if record["stopped"] or index is None:
        raise ValueError("STOP cannot be a comparison member")
    decision = memory._decision(
        query,
        rows,
        index,
        record["mode"],
        record["stop_reason"],
        record["physical_cell_steps"],
        record["spike_step"],
    )
    if decision.sha256 != record["sha256"]:
        raise ValueError("decision custody/checkpoint differs on exact reconstruction")
    if (
        sha256(np.asarray(decision.eligibility).tobytes()).hexdigest()
        != record["eligibility_sha256"]
    ):
        raise ValueError("eligibility hash differs")
    if decision.emitted.decode() != record["emitted"]:
        raise ValueError("selected occurrence bytes differ")
    return decision


def diagnose_arm(arm, training, schedule, archive, name, encoder):
    memory = ContextualEvidenceMemory(encoder, learning_rate=1.0)
    if memory.checkpoint_sha256 != arm["seed"]["checkpoint_sha256"]:
        raise ValueError("seed differs")
    fixtures = {item["name"]: item for item in training}
    ledger = {item["pair_id"]: item for item in arm["ledger"]}
    if len(ledger) != len(arm["ledger"]):
        raise ValueError("duplicate ledger IDs")
    serialized = json.dumps(arm["ledger"], sort_keys=True, allow_nan=False).encode()
    if sha256(serialized).hexdigest() != arm["ledger_summary"]["sha256"]:
        raise ValueError("immutable ledger serialization hash differs")
    if [row["case"] for row in arm["training"]] != schedule:
        raise ValueError("recorded teaching order differs from declared schedule")
    seen, pairs, delta_by_id, last_gap, exposure = {}, {}, {}, {}, []
    for row in arm["training"]:
        item = fixtures[row["case"]]
        identity, content = item["document_id"], item["document"]
        if identity not in seen:
            memory.ingest(
                (
                    OpaqueDocument(
                        f"{identity}/0".encode(),
                        f"provenance/{identity}/0".encode(),
                        content.encode(),
                    ),
                )
            )
            seen[identity] = content
        elif seen[identity] != content:
            raise ValueError("conflicting source document")
        if memory.checkpoint_sha256 != row["after_ingestion"]["checkpoint_sha256"]:
            raise ValueError("post-ingestion checkpoint differs")
        for event in row["feedback"]:
            pair_id = event["pair_id"]
            original = ledger[pair_id]
            if event["origin_step"] != original["origin_step"]:
                raise ValueError("replay origin differs from ledger")
            rejected = reconstruct_decision(memory, event["rejected_decision"])
            accepted = reconstruct_decision(memory, event["accepted_decision"])
            delta = np.asarray(accepted.eligibility) - np.asarray(rejected.eligibility)
            key = sha256(delta.tobytes()).hexdigest()
            if key != event["update"]["eligibility_delta_sha256"]:
                raise ValueError("delta hash differs")
            if event["kind"] == "new":
                if pair_id in pairs:
                    raise ValueError("new event repeats an existing pair ID")
                if original["rejected_decision"]["sha256"] != rejected.sha256:
                    raise ValueError("original rejected decision differs")
                if original["accepted_decision"]["sha256"] != accepted.sha256:
                    raise ValueError("original accepted decision differs")
                for decision, side in ((rejected, "rejected"), (accepted, "accepted")):
                    if not np.array_equal(
                        decision.eligibility, original[side + "_decision"]["eligibility"]
                    ):
                        raise ValueError("immutable ledger eligibility payload differs")
                if not original["teacher_execution"]["passed"]:
                    raise ValueError("missing original teacher PASS")
                failed = (
                    row["exploration"] if row["first"]["selection"]["stopped"] else row["first"]
                )
                if not failed["edited"] or failed["execution"]["passed"]:
                    raise ValueError("missing original executed failure")
                if failed["selection"]["sha256"] != rejected.sha256:
                    raise ValueError("executed failure differs from rejected decision")
                if (
                    original["rejected_execution"] != failed["execution"]
                    or original["teacher_execution"] != row["teacher_execution"]
                    or original["origin_step"] != row["step"]
                ):
                    raise ValueError("immutable ledger execution or origin differs")
                pairs[pair_id] = {
                    "pair_id": pair_id,
                    "origin_step": original["origin_step"],
                    "query": rejected.query.decode(),
                    "delta_sha256": key,
                    "rejected_occurrence": occurrence(memory, rejected.candidate_index),
                    "accepted_occurrence": occurrence(memory, accepted.candidate_index),
                    "rejected_emitted": rejected.emitted.decode(),
                    "accepted_emitted": accepted.emitted.decode(),
                    "initial_gap_after": event["update"]["gap_after"],
                }
                delta_by_id[pair_id] = delta
            elif event["kind"] == "certified_replay":
                if pair_id not in pairs or not np.array_equal(delta, delta_by_id[pair_id]):
                    raise ValueError("replay precedes original or changes eligibility")
                for decision, side in ((rejected, "rejected"), (accepted, "accepted")):
                    if (
                        occurrence(memory, decision.candidate_index)
                        != pairs[pair_id][side + "_occurrence"]
                    ):
                        raise ValueError("replay changes retained occurrence")
            else:
                raise ValueError("unknown feedback kind")
            actual = memory.observe_comparison(rejected, accepted)
            if actual != event["update"]:
                raise ValueError("native comparison receipt differs from original run")
            last_gap[pair_id] = actual["gap_after"]
            exposure.append(
                {
                    "kind": event["kind"],
                    "pair_id": pair_id,
                    "step": row["step"],
                    "gap_after": actual["gap_after"],
                }
            )
        if memory.checkpoint_sha256 != row["after"]["checkpoint_sha256"]:
            raise ValueError("training row checkpoint differs")
    if set(pairs) != set(ledger):
        raise ValueError("some immutable ledger originals were not reconstructed")
    arrays = memory.checkpoint_arrays()
    for key, value in arrays.items():
        if not np.array_equal(value, archive[f"{name}_update_{key}"]):
            raise ValueError("reconstructed final checkpoint array differs")
        if np.any(archive[f"{name}_frozen_{key}"] != 0):
            raise ValueError("frozen checkpoint differs from zero synaptic initialization")
    if memory.checkpoint_sha256 != arm["trained"]["checkpoint_sha256"]:
        raise ValueError("trained checkpoint differs")
    matrix = np.asarray(list(delta_by_id.values())).reshape(-1, 1537)
    weights = arrays["readout_synapses_pa"][0] - arrays["readout_synapses_pa"][1]
    margins = matrix @ weights
    for pair, margin in zip(pairs.values(), margins, strict=True):
        pair.update(
            final_margin=float(margin),
            last_exposure_gap_after=last_gap[pair["pair_id"]],
            margin_change_since_initial=float(margin - pair["initial_gap_after"]),
            margin_change_since_last=float(margin - last_gap[pair["pair_id"]]),
        )
        delta = delta_by_id[pair["pair_id"]]
        pair["saved_margin_block_contributions"] = {
            name: float(delta[left:right] @ weights[left:right]) for name, left, right in BLOCKS
        }
    unique = np.unique(matrix, axis=0)
    return {
        "verified_exact_native_event_receipts_and_final_arrays": True,
        "training_document_count": len(seen),
        "retained_store": memory.storage_summary,
        "original_pair_count": len(pairs),
        "distinct_delta_count": len(unique),
        "feedback_exposure_count": len(exposure),
        "replay_exposure_count": sum(e["kind"] == "certified_replay" for e in exposure),
        "pairs": list(pairs.values()),
        "exposures": exposure,
        "occurrence_graph": graph_diagnosis(list(pairs.values())),
        "constraint_geometry": geometry(list(pairs.values()), matrix),
        "saved_margins": measure(matrix, weights) if len(matrix) else None,
        "forgetting": {
            "initially_satisfied_then_violated": sum(
                p["initial_gap_after"] >= 1 - TOLERANCE and p["final_margin"] < 1 - TOLERANCE
                for p in pairs.values()
            ),
            "last_satisfied_then_violated": sum(
                p["last_exposure_gap_after"] >= 1 - TOLERANCE and p["final_margin"] < 1 - TOLERANCE
                for p in pairs.values()
            ),
        },
        "full_feature_feasibility": feasible(unique, np.arange(1537)),
        "query_independent_span_feasibility": feasible(unique, np.arange(384, 768)),
    }


def run(source, cases, output, model_dir):
    if output.exists():
        raise FileExistsError("existing diagnostic evidence must remain unchanged")
    os.environ["HUMANBRAIN_DEVICE"] = "cpu"
    os.environ["HUMANBRAIN_NATIVE_CACHE"] = str(ROOT / ".native-cache")
    began = time.perf_counter()
    sources = {path: digest(ROOT / path) for path in SOURCES}
    original = json.loads(source.read_text("utf-8"))
    if original["terminal"] != "completed" or not original["sources_unchanged_during_run"]:
        raise ValueError("original run is incomplete or source-unstable")
    for path, expected in original["source_sha256"].items():
        if path in SOURCES and sources[path] != expected:
            raise ValueError(f"original source changed: {path}")
    if digest(cases) != original["source_sha256"][cases.resolve().relative_to(ROOT).as_posix()]:
        raise ValueError("case fixture changed")
    checkpoint = source.with_suffix(".npz")
    if digest(checkpoint) != original["synapse_checkpoint"]["sha256"]:
        raise ValueError("checkpoint archive changed")
    fixtures = json.loads(cases.read_text("utf-8"))
    training, schedule = fixtures["training"], fixtures["schedule"]
    encoder = FrozenContextEncoder(model_dir)
    if encoder.metadata != original["encoder"]:
        raise ValueError("encoder metadata differs")
    report = {
        "kind": "recorded comparison diagnosis; not neuronal capability evidence",
        "source_sha256": sources,
        "original_sha256": digest(source),
        "checkpoint_archive_sha256": digest(checkpoint),
        "case_fixture_sha256": digest(cases),
        "arms": {},
        "new_negatives": 0,
        "transfer_evaluations": 0,
        "protected_evaluations": 0,
        "feasibility_tolerance": TOLERANCE,
    }
    with np.load(checkpoint, allow_pickle=False) as archive:
        for name in ("REVISIT", "REPLAY"):
            report["arms"][name] = diagnose_arm(
                original["arms"][name], training, schedule, archive, name, encoder
            )
    native = loaded_build_info()
    report["native"] = {
        "source_digest": native.source_digest,
        "library": str(native.library_path),
        "binary_sha256": digest(native.library_path),
    }
    if report["native"] != original["native"]:
        raise ValueError("native source/binary differs from original run")
    report["sources_unchanged_during_run"] = sources == {
        path: digest(ROOT / path) for path in SOURCES
    }
    if not report["sources_unchanged_during_run"]:
        raise ValueError("diagnostic source changed during run")
    report.update(terminal="completed", elapsed_seconds=time.perf_counter() - began)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "experiments/r4_integrated/contextual_revisit_cases.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.source, args.cases, args.output, args.model_dir)
    print(
        json.dumps({"terminal": result["terminal"], "elapsed_seconds": result["elapsed_seconds"]})
    )
