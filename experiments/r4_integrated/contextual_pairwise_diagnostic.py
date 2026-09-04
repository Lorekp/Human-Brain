"""Evaluator-only fit/forgetting diagnosis of sixteen recorded executed contrasts.

No new negatives, teacher values, global candidate scoring, transfer, or protected
tasks. The fixed native budget is 25 ordered passes from zero over the original
sixteen deltas. The least-norm linear fit is a conventional capacity reference.
Neither diagnostic result is an integrated learner capability measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from graphlib import CycleError, TopologicalSorter
from hashlib import sha256
from pathlib import Path

import numpy as np
from humanbrain.contextual_evidence_memory import ContextualEvidenceMemory
from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.native import NativeThreeFactorLearner, loaded_build_info
from humanbrain.opaque_document_event_tape import OpaqueDocument

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "experiments/r4_integrated/contextual_pairwise_diagnostic.py",
    "src/humanbrain/contextual_evidence_memory.py",
    "src/humanbrain/frozen_context_encoder.py",
    "src/humanbrain/native.py",
    "src/humanbrain/resting_glif_race.py",
    "src/humanbrain/opaque_document_event_tape.py",
)
BLOCKS = (
    ("query", 0, 384),
    ("span", 384, 768),
    ("query_times_span", 768, 1152),
    ("query_times_window", 1152, 1536),
    ("lexical_overlap", 1536, 1537),
)
TOLERANCE = 1e-10


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def measure(deltas: np.ndarray, weights: np.ndarray) -> dict:
    margins = deltas @ weights
    return {
        "margin_violations_below_one": int(np.count_nonzero(margins < 1 - TOLERANCE)),
        "wrong_or_tied_pairs": int(np.count_nonzero(margins <= 0)),
        "min_margin": float(margins.min()),
        "mean_margin": float(margins.mean()),
        "margins": margins.tolist(),
        "weight_difference_l2": float(np.linalg.norm(weights)),
        "weight_block_l2": {
            name: float(np.linalg.norm(weights[left:right])) for name, left, right in BLOCKS
        },
    }


def value_order(pairs: list[dict]) -> dict:
    predecessors = {}
    for pair in pairs:
        rejected, accepted = pair["rejected_emitted"], pair["accepted_emitted"]
        predecessors.setdefault(rejected, set())
        predecessors.setdefault(accepted, set()).add(rejected)
    try:
        order = list(TopologicalSorter(predecessors).static_order())
    except CycleError as error:
        return {"acyclic": False, "cycle": error.args[1], "topological_order": None}
    rank = {value: i for i, value in enumerate(order)}
    return {
        "acyclic": True,
        "topological_order": order,
        "cycle": None,
        "minimum_value_rank_margin": min(
            rank[p["accepted_emitted"]] - rank[p["rejected_emitted"]] for p in pairs
        ),
        "scope": "emitted-byte ordering ignores query and occurrence; explanatory reference only",
    }


def linear_fit(deltas: np.ndarray, columns: np.ndarray) -> dict:
    began = time.perf_counter()
    fitted, _, rank, singular = np.linalg.lstsq(deltas[:, columns], np.ones(16), rcond=None)
    full = np.zeros(deltas.shape[1])
    full[columns] = fitted
    return {
        **measure(deltas, full),
        "rank": int(rank),
        "singular_values": singular.tolist(),
        "maximum_absolute_target_residual": float(np.max(np.abs(deltas @ full - 1))),
        "elapsed_seconds": time.perf_counter() - began,
    }


def reconstruct(original: dict, training: list[dict], memory: ContextualEvidenceMemory):
    recorded = original["arms"]["PAIRWISE"]["training"]
    if len(recorded) != 16 or len(training) != 16:
        raise ValueError("diagnostic requires exactly sixteen original training events")
    seen, deltas, pairs = {}, [], []
    for event, item in zip(recorded, training, strict=True):
        if event["case"] != item["name"] or event["correction_status"] != "applied":
            raise ValueError("training order or executed comparison differs")
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
            raise ValueError("duplicate document identity has different content")
        rejected, accepted = event["rejected_decision"], event["accepted_decision"]
        execution = (
            event["exploration"] if event["first"]["selection"]["stopped"] else event["first"]
        )
        if not execution["edited"] or execution["execution"]["passed"]:
            raise ValueError("rejected selection lacks an executed failure")
        if execution["selection"]["sha256"] != rejected["sha256"]:
            raise ValueError("rejected selection does not identify the failed execution")
        if not event["teacher_execution"]["passed"]:
            raise ValueError("accepted correction lacks an executed success")
        if rejected["query"] != accepted["query"]:
            raise ValueError("recorded contrast has mismatched queries")
        rows, _ = memory._features(rejected["query"].encode())  # Evaluator-only reconstruction.
        for decision in (rejected, accepted):
            index = decision["candidate_index"]
            if (
                decision["stopped"]
                or memory._candidates[index].emitted.decode() != decision["emitted"]
            ):
                raise ValueError("recorded occurrence bytes differ from retained source")
            if memory._storage_root != decision["storage_sha256"]:
                raise ValueError("reconstructed cumulative source store differs")
            if sha256(rows[index].tobytes()).hexdigest() != decision["eligibility_sha256"]:
                raise ValueError("reconstructed occurrence eligibility hash differs")
        delta = rows[accepted["candidate_index"]] - rows[rejected["candidate_index"]]
        receipt = event["feedback"][0]["update"]
        if len(event["feedback"]) != 1 or receipt["mode"] != "same_query_pairwise":
            raise ValueError("unexpected training feedback objective")
        if sha256(delta.tobytes()).hexdigest() != receipt["eligibility_delta_sha256"]:
            raise ValueError("recorded pair delta hash differs")
        deltas.append(delta)
        pairs.append(
            {
                "case": event["case"],
                "query": rejected["query"],
                "rejected_candidate_index": rejected["candidate_index"],
                "accepted_candidate_index": accepted["candidate_index"],
                "rejected_emitted": rejected["emitted"],
                "accepted_emitted": accepted["emitted"],
                "eligibility_delta_sha256": receipt["eligibility_delta_sha256"],
                "immediate_gap_before": receipt["gap_before"],
                "immediate_gap_after": receipt["gap_after"],
            }
        )
    if len(seen) != 8:
        raise ValueError("diagnostic must reconstruct exactly eight training documents")
    return np.asarray(deltas), pairs


def run(source: Path, cases: Path, output: Path, model_dir: Path) -> dict:
    if output.exists():
        raise FileExistsError("diagnostic evidence already exists; choose a fresh output")
    os.environ["HUMANBRAIN_DEVICE"] = "cpu"
    os.environ["HUMANBRAIN_NATIVE_CACHE"] = str(ROOT / ".native-cache")
    started = time.perf_counter()
    sources = {path: digest(ROOT / path) for path in SOURCE_PATHS}
    original = json.loads(source.read_text("utf-8"))
    if original["terminal"] != "completed" or not original["sources_unchanged_during_run"]:
        raise ValueError("original comparison did not complete against stable source")
    for path in SOURCE_PATHS[1:]:
        if path in original["source_sha256"] and sources[path] != original["source_sha256"][path]:
            raise ValueError(f"original implementation changed: {path}")
    if (
        digest(cases)
        != original["source_sha256"]["experiments/r4_integrated/contextual_pairwise_cases.json"]
    ):
        raise ValueError("original case fixture changed")
    checkpoint = source.with_suffix(".npz")
    if digest(checkpoint) != original["synapse_checkpoint"]["sha256"]:
        raise ValueError("original checkpoint archive changed")
    with np.load(checkpoint, allow_pickle=False) as archive:
        saved = archive["PAIRWISE_update_readout_synapses_pa"]
        saved_bias = archive["PAIRWISE_update_pacemaker_synapses_pa"]
    if (
        sha256(saved.tobytes() + saved_bias.tobytes()).hexdigest()
        != original["arms"]["PAIRWISE"]["trained"]["synapse_sha256"]
    ):
        raise ValueError("recorded trained synapse hash differs")
    encoder = FrozenContextEncoder(model_dir)
    memory = ContextualEvidenceMemory(encoder, learning_enabled=False)
    reconstruction_started = time.perf_counter()
    # Access only the training slice; values/transfer entries are never used.
    deltas, pairs = reconstruct(original, json.loads(cases.read_text("utf-8"))["training"], memory)
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    native = loaded_build_info()
    if native.source_digest != original["native"]["source_digest"]:
        raise ValueError("native build sources differ from original experiment")
    saved_measure = measure(deltas, saved[0] - saved[1])
    for pair, margin in zip(pairs, saved_measure["margins"], strict=True):
        pair["saved_checkpoint_margin"] = margin
        pair["margin_change_since_own_update"] = margin - pair["immediate_gap_after"]
    eligibility = np.zeros(deltas.shape[1], dtype=np.float64)
    weights = np.zeros_like(saved)
    scratch_bias, heterogeneity = np.zeros(2), np.zeros(2)
    learner = NativeThreeFactorLearner(
        eligibility=eligibility,
        output_heterogeneity_pa=heterogeneity,
        readout_synapses_pa=weights,
        pacemaker_synapses_pa=scratch_bias,
    )
    passes = []
    replay_started = time.perf_counter()
    for number in range(1, 26):
        applied = 0
        for delta, pair in zip(deltas, pairs, strict=True):
            eligibility[:] = delta
            scratch_bias.fill(0)
            before = float((weights[0] - weights[1]) @ delta)
            applied += int(learner.update(0, 1.0, 1.0) is not None)
            after = float((weights[0] - weights[1]) @ delta)
            if number == 1 and not np.allclose(
                [before, after],
                [pair["immediate_gap_before"], pair["immediate_gap_after"]],
                rtol=0,
                atol=TOLERANCE,
            ):
                raise ValueError("first native replay does not reproduce recorded immediate gaps")
        if number == 1 and not np.array_equal(weights, saved):
            raise ValueError("first native replay does not exactly reconstruct saved weights")
        passes.append(
            {"pass": number, "native_updates": applied, **measure(deltas, weights[0] - weights[1])}
        )
    replay_seconds = time.perf_counter() - replay_started
    fit = linear_fit(deltas, np.arange(deltas.shape[1]))
    span_fit = linear_fit(deltas, np.arange(384, 768))
    span_lexical_fit = linear_fit(deltas, np.r_[384:768, 1536])
    report = {
        "kind": "evaluator-only recorded contrast fit diagnostic; no capability claim",
        "terminal": "completed",
        "source_sha256": sources,
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "original_artifact": {"path": str(source.resolve()), "sha256": digest(source)},
        "original_checkpoint_sha256": digest(checkpoint),
        "case_fixture_sha256": digest(cases),
        "encoder": encoder.metadata,
        "native": {
            "source_digest": native.source_digest,
            "library": str(native.library_path),
            "binary_sha256": digest(native.library_path),
        },
        "exposure": {
            "documents": 8,
            "recorded_executed_pairs": 16,
            "new_teacher_values": 0,
            "new_negatives": 0,
            "global_candidate_evaluations": 0,
            "transfer_evaluations": 0,
            "protected_evaluations": 0,
        },
        "reconstruction": {
            "all_delta_and_eligibility_hashes_match": True,
            "first_native_pass_exact_saved_weights": True,
            "first_native_pass_matches_immediate_gaps": True,
            "pure_query_delta_max_abs": float(np.abs(deltas[:, :384]).max()),
            "seconds": reconstruction_seconds,
        },
        "saved_checkpoint": saved_measure,
        "forgetting": {
            "immediate_margin_satisfied": sum(
                p["immediate_gap_after"] >= 1 - TOLERANCE for p in pairs
            ),
            "satisfied_then_violated": sum(
                p["immediate_gap_after"] >= 1 - TOLERANCE
                and p["saved_checkpoint_margin"] < 1 - TOLERANCE
                for p in pairs
            ),
            "positive_then_wrong_or_tied": sum(
                p["immediate_gap_after"] > 0 and p["saved_checkpoint_margin"] <= 0 for p in pairs
            ),
        },
        "pairs": pairs,
        "native_replay": {
            "initialization": "zero",
            "passes_budget": 25,
            "offered_updates": 400,
            "learning_rate": 1.0,
            "margin": 1.0,
            "bias": "scratch reset to zero before each pair; persistent bias excluded",
            "seconds": replay_seconds,
            "passes": passes,
        },
        "conventional_least_norm_fit": fit,
        "query_independent_span_only_fit": span_fit,
        "span_plus_lexical_fit": {
            **span_lexical_fit,
            "scope": "no query-vector interactions; lexical overlap remains query-dependent",
        },
        "emitted_value_constraint_graph": value_order(pairs),
        "margin_tolerance": TOLERANCE,
        "claim_boundary": (
            "Fit only to the sixteen executed contrasts. Fitting them cannot establish "
            "global candidate ranking, fresh transfer, retention task success, or GLIF capability."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["sources_unchanged_during_run"] = sources == {
        path: digest(ROOT / path) for path in SOURCE_PATHS
    }
    if not report["sources_unchanged_during_run"]:
        raise RuntimeError("diagnostic sources changed during run")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "benchmarks/contextual_pairwise_python_development_v1.json",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "experiments/r4_integrated/contextual_pairwise_cases.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.source, args.cases, args.output, args.model_dir)
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "forgetting": result["forgetting"],
                "final_replay": result["native_replay"]["passes"][-1],
                "least_norm": result["conventional_least_norm_fit"],
            }
        )
    )
