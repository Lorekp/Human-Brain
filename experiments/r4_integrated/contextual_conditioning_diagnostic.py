"""Preregistered conditioning diagnostic on 32 already executed REVISIT pairs.

Contract fixed before execution: exactly three zero-initialized native variants,
four deterministic admission-order passes (128 offered pairs) each. A: existing
scales, eta=1. B: gain16 on delta[768:1536], eta=1. C: the same gain16, with
eta=(1-gap)*sqrt(1+||d||²)/(2||d||²) when gap<1, otherwise skip. Scratch bias is
zeroed before every pair and discarded. No extra epochs or parameter search.

Gain16 was fixed from the earlier measured span/query-times-span median norm
ratio 1.181/.0623 (~19), choosing a nearby power of two before these outputs.
Only the existing REVISIT ledger is used: no new actions, negatives, teacher
values, global-candidate scoring, transfer, or protected tasks. Results are
finite constraint-learning diagnostics, not integrated capability evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.r4_integrated.contextual_pairwise_diagnostic import BLOCKS, digest, measure
from humanbrain.contextual_evidence_memory import _digest
from humanbrain.native import NativeThreeFactorLearner, loaded_build_info

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "experiments/r4_integrated/contextual_conditioning_diagnostic.py",
    "experiments/r4_integrated/contextual_pairwise_diagnostic.py",
    "src/humanbrain/contextual_evidence_memory.py",
    "src/humanbrain/native.py",
    "src/humanbrain/_native_src/src/lib.rs",
)
VARIANTS = (
    ("A_current", 1.0, False),
    ("B_gain16", 16.0, False),
    ("C_gain16_margin_deficit", 16.0, True),
)
PASSES, PAIRS, TOLERANCE = 4, 32, 1e-10


def eligibility(record):
    values = np.asarray(record["eligibility"], dtype=np.float64)
    if (
        values.shape != (1537,)
        or sha256(values.tobytes()).hexdigest() != record["eligibility_sha256"]
    ):
        raise ValueError("ledger eligibility payload or hash differs")
    serial = {
        key: record[key]
        for key in (
            "candidate_index",
            "checkpoint_sha256",
            "storage_sha256",
            "mode",
            "stop_reason",
            "physical_cell_steps",
            "spike_step",
        )
    }
    serial.update(
        query=record["query"].encode().hex(),
        emitted=record["emitted"].encode().hex(),
        eligibility=values.tolist(),
    )
    if _digest(serial) != record["sha256"]:
        raise ValueError("ledger decision custody checksum differs")
    return values


def load_pairs(source, prior_diagnostic):
    original = json.loads(source.read_text("utf-8"))
    diagnosis = json.loads(prior_diagnostic.read_text("utf-8"))
    if original["terminal"] != "completed" or not original["sources_unchanged_during_run"]:
        raise ValueError("integrated receipt did not complete against stable source")
    if diagnosis["terminal"] != "completed" or diagnosis["original_sha256"] != digest(source):
        raise ValueError("prior exact-occurrence diagnosis is not linked to this receipt")
    if not diagnosis["arms"]["REVISIT"]["verified_exact_native_event_receipts_and_final_arrays"]:
        raise ValueError("prior source/occurrence reconstruction was not verified")
    for path, expected in diagnosis["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"verified reconstruction source changed: {path}")
    checkpoint = source.with_suffix(".npz")
    if digest(checkpoint) != original["synapse_checkpoint"]["sha256"]:
        raise ValueError("completed checkpoint archive changed")
    if digest(checkpoint) != diagnosis["checkpoint_archive_sha256"]:
        raise ValueError("prior diagnosis checkpoint link differs")
    arm = original["arms"]["REVISIT"]
    ledger = arm["ledger"]
    encoded = json.dumps(ledger, sort_keys=True, allow_nan=False).encode()
    if len(ledger) != PAIRS or sha256(encoded).hexdigest() != arm["ledger_summary"]["sha256"]:
        raise ValueError("expected exactly 32 immutable executed pairs")
    events = [e for row in arm["training"] for e in row["feedback"]]
    if len(events) != PAIRS or any(e["kind"] != "new" for e in events):
        raise ValueError("REVISIT must have exactly the 32 original feedback events")
    deltas, records = [], []
    for pair, event in zip(ledger, events, strict=True):
        if pair["pair_id"] != event["pair_id"]:
            raise ValueError("feedback order differs from ledger admission order")
        rejected, accepted = pair["rejected_decision"], pair["accepted_decision"]
        if (rejected["query"], rejected["checkpoint_sha256"]) != (
            accepted["query"],
            accepted["checkpoint_sha256"],
        ):
            raise ValueError("comparison has mismatched query/parent")
        if pair["rejected_execution"]["passed"] or not pair["teacher_execution"]["passed"]:
            raise ValueError("comparison lacks recorded FAIL and PASS")
        delta = eligibility(accepted) - eligibility(rejected)
        if sha256(delta.tobytes()).hexdigest() != event["update"]["eligibility_delta_sha256"]:
            raise ValueError("comparison delta differs from executed feedback")
        deltas.append(delta)
        records.append(
            {
                "pair_id": pair["pair_id"],
                "origin_step": pair["origin_step"],
                "delta_sha256": sha256(delta.tobytes()).hexdigest(),
                "original_gap_before": event["update"]["gap_before"],
                "original_gap_after": event["update"]["gap_after"],
            }
        )
    with np.load(checkpoint, allow_pickle=False) as archive:
        saved = archive["REVISIT_update_readout_synapses_pa"]
        bias = archive["REVISIT_update_pacemaker_synapses_pa"]
    if sha256(saved.tobytes() + bias.tobytes()).hexdigest() != arm["trained"]["synapse_sha256"]:
        raise ValueError("saved synapse payload differs from trained receipt")
    if np.any(bias):
        raise ValueError("pairwise checkpoint unexpectedly retains a nonzero bias")
    return original, np.asarray(deltas), records, saved


def variant(name, gain, adaptive, raw, records, saved):
    started = time.perf_counter()
    scales = np.ones(1537)
    scales[768:1536] = gain
    transformed = raw * scales
    weights, trace, bias, heterogeneity = (
        np.zeros((2, 1537)),
        np.zeros(1537),
        np.zeros(2),
        np.zeros(2),
    )
    learner = NativeThreeFactorLearner(
        eligibility=trace,
        output_heterogeneity_pa=heterogeneity,
        readout_synapses_pa=weights,
        pacemaker_synapses_pa=bias,
    )
    passes = []
    for number in range(1, PASSES + 1):
        pass_started = time.perf_counter()
        updates, events, interference = 0, [], 0
        for index, delta in enumerate(transformed):
            trace[:] = delta
            bias.fill(0)
            before_weights = weights[0] - weights[1]
            before_margins = transformed @ before_weights
            gap = float(before_weights @ delta)
            squared = float(delta @ delta)
            eta = 1.0
            if adaptive:
                eta = (1 - gap) * math.sqrt(1 + squared) / (2 * squared) if gap < 1 else 0.0
            applied = eta > 0 and learner.update(0, eta, 1.0) is not None
            updates += int(applied)
            after_weights = weights[0] - weights[1]
            after_margins = transformed @ after_weights
            after = float(after_weights @ delta)
            predicted = gap + 2 * eta * squared / math.sqrt(1 + squared) if applied else gap
            if abs(after - predicted) > TOLERANCE:
                raise ValueError("native own-gap change differs from declared arithmetic")
            other = np.arange(PAIRS) != index
            forgotten = int(
                np.count_nonzero(
                    other & (before_margins >= 1 - TOLERANCE) & (after_margins < 1 - TOLERANCE)
                )
            )
            interference += forgotten
            events.append(
                {
                    "pair_id": records[index]["pair_id"],
                    "eta": eta,
                    "delta_norm_squared": squared,
                    "gap_before": gap,
                    "predicted_gap_after": predicted,
                    "actual_gap_after": after,
                    "arithmetic_error": after - predicted,
                    "native_update_applied": applied,
                    "other_margin_satisfied_to_violated": forgotten,
                    "minimum_other_margin_change": float(
                        (after_margins - before_margins)[other].min()
                    ),
                }
            )
            if (
                name == "A_current"
                and number == 1
                and not np.allclose(
                    [gap, after],
                    [records[index]["original_gap_before"], records[index]["original_gap_after"]],
                    rtol=0,
                    atol=TOLERANCE,
                )
            ):
                raise ValueError("A first pass does not match recorded immediate gaps")
        if name == "A_current" and number == 1 and not np.array_equal(weights, saved):
            raise ValueError("A first pass does not exactly match the completed NPZ")
        native_weights = weights[0] - weights[1]
        effective = native_weights * scales
        scoring = measure(raw, effective)
        if not np.allclose(raw @ effective, transformed @ native_weights, rtol=0, atol=TOLERANCE):
            raise ValueError("scaled and original-coordinate margins differ")
        scoring["native_weight_difference_l2"] = float(np.linalg.norm(native_weights))
        scoring["native_weight_block_l2"] = {
            block: float(np.linalg.norm(native_weights[left:right]))
            for block, left, right in BLOCKS
        }
        scoring["block_margin_contributions"] = {
            block: (raw[:, left:right] @ effective[left:right]).tolist()
            for block, left, right in BLOCKS
        }
        passes.append(
            {
                "pass": number,
                "offered_pairs": PAIRS,
                "native_updates": updates,
                "interference_events": interference,
                "original_constraints": scoring,
                "step_arithmetic": events,
                "elapsed_seconds": time.perf_counter() - pass_started,
            }
        )
    return {
        "interaction_gain": gain,
        "margin_deficit_eta": adaptive,
        "passes": passes,
        "offered_pairs": PASSES * PAIRS,
        "native_updates": sum(p["native_updates"] for p in passes),
        "native_weight_sha256": sha256(weights.tobytes()).hexdigest(),
        "persistent_bias": "zero; scratch reset before every pair and discarded",
        "elapsed_seconds": time.perf_counter() - started,
    }


def run(source, prior_diagnostic, output):
    if output.exists():
        raise FileExistsError("existing evidence remains immutable")
    if (
        os.environ.get("HUMANBRAIN_DEVICE") != "cpu"
        or Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve() != ROOT / ".native-cache"
    ):
        raise ValueError("set CPU and checkout-local native cache before launching Python")
    began = time.perf_counter()
    sources = {path: digest(ROOT / path) for path in SOURCES}
    original, raw, pairs, saved = load_pairs(source, prior_diagnostic)
    report = {
        "kind": "preregistered finite conditioning diagnostic; no capability evidence",
        "contract": __doc__,
        "source_sha256": sources,
        "original_sha256": digest(source),
        "prior_diagnostic_sha256": digest(prior_diagnostic),
        "original_checkpoint_sha256": digest(source.with_suffix(".npz")),
        "original_native": original["native"],
        "pairs": pairs,
        "variants": {},
        "arithmetic": "delta_w=2*eta*d/sqrt(1+||d||^2); "
        "delta_gap=2*eta*||d||^2/sqrt(1+||d||^2); scratch bias excluded",
        "scoring_coordinates": "For gain G, native scores (G*d)@w equal d@(G*w). "
        "Original constraints use effective weights G*w; inference would require the same gain.",
        "new_actions": 0,
        "new_negatives": 0,
        "global_candidate_scores": 0,
        "transfer_scores": 0,
        "protected_scores": 0,
        "arithmetic_tolerance": TOLERANCE,
    }
    for name, gain, adaptive in VARIANTS:
        report["variants"][name] = variant(name, gain, adaptive, raw, pairs, saved)
    native = loaded_build_info()
    report["native"] = {
        "source_digest": native.source_digest,
        "library": str(native.library_path),
        "binary_sha256": digest(native.library_path),
    }
    if report["native"] != original["native"]:
        raise ValueError("actual native identity differs from completed comparison")
    report["sources_unchanged_during_run"] = sources == {
        path: digest(ROOT / path) for path in SOURCES
    }
    if not report["sources_unchanged_during_run"]:
        raise ValueError("source changed during diagnostic")
    report.update(terminal="completed", elapsed_seconds=time.perf_counter() - began)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "benchmarks/contextual_revisit_python_development_v1.json",
    )
    parser.add_argument(
        "--prior-diagnostic",
        type=Path,
        default=ROOT / "benchmarks/contextual_revisit_diagnostic_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.prior_diagnostic, args.output)
    print(
        json.dumps({"terminal": result["terminal"], "elapsed_seconds": result["elapsed_seconds"]})
    )
