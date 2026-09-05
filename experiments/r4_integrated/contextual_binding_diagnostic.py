"""Read-only saved-score/receipt diagnosis of the completed binding comparison.

No updates, new actions, fitting, or new requests. Reconstruct saved NPZ state,
source ingestion and feedback hashes; reproduce all ordinary evaluation decisions
before interpreting unthresholded ranks. Expected bytes are evaluator-only and
never enter selection. Rank matches are not executed successes or a new policy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

import numpy as np
from experiments.r4_integrated import contextual_binding_python_session as session
from humanbrain.contextual_evidence_memory import (
    ContextualEvidenceDecision,
    ContextualEvidenceMemory,
)
from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.native import loaded_build_info

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = float(np.arctanh((600.1 - 600.0) / 200.0))
BLOCKS = {
    "query": np.arange(384),
    "span": np.arange(384, 768),
    "query_span": np.arange(768, 1152),
    "query_window": np.arange(1152, 1536),
    "lexical": np.asarray([1536]),
    "F_exact": np.arange(1537, 1609, 2),
    "F_semantic": np.arange(1538, 1609, 2),
    "H_exact": np.arange(1609, 1621, 2),
    "H_semantic": np.arange(1610, 1621, 2),
}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contributions(row, weights):
    return {
        name: float(row[indices] @ weights[indices])
        for name, indices in BLOCKS.items()
        if len(indices) and indices[-1] < len(weights)
    }


def occurrence(memory, index):
    candidate = memory._candidates[index]
    source = memory._documents[candidate.document_index]
    return {
        "index": int(index),
        "emitted": candidate.emitted.decode(),
        "document_id": source.document_id.decode(),
        "provenance_id": source.provenance_id.decode(),
        "source_sha256": sha256(source.content).hexdigest(),
        "byte_span": [candidate.start, candidate.end],
    }


def verify_decision(record, eligibility):
    """Check original custody without pretending to possess past weight arrays."""
    require(
        sha256(eligibility.tobytes()).hexdigest() == record["eligibility_sha256"],
        "recorded eligibility hash differs",
    )
    values = {
        f.name: record[f.name]
        for f in fields(ContextualEvidenceDecision)
        if f.name not in ("sha256", "eligibility")
    }
    values["query"] = record["query"].encode().hex()
    values["emitted"] = None if record["emitted"] is None else record["emitted"].encode().hex()
    values["eligibility"] = eligibility.tolist()
    require(digest(values) == record["sha256"], "original decision custody differs")


def reconstruct(arm, name, fixtures, archive, encoder):
    models = [
        ContextualEvidenceMemory(
            encoder,
            interaction_gain=arm["interaction_gain"],
            candidate_interactions=arm["candidate_interactions"],
            learning_enabled=enabled,
        )
        for enabled in (True, False)
    ]
    require(
        all(m.checkpoint_sha256 == arm["seed"]["checkpoint_sha256"] for m in models),
        "seed identity differs",
    )
    require(digest(arm["ledger"]) == arm["ledger_summary"]["sha256"], "ledger hash differs")
    by_name = {item["name"]: item for item in fixtures["training"]}
    require([r["case"] for r in arm["training"]] == fixtures["schedule"], "schedule differs")
    seen, ledger, deltas, verified = {}, {p["pair_id"]: p for p in arm["ledger"]}, {}, 0
    for row in arm["training"]:
        require(
            session.prior._ingest_once(by_name[row["case"]], models, seen) == row["ingested"],
            "ingestion schedule differs",
        )
        for feedback in row["feedback"]:
            pair = ledger[feedback["pair_id"]]
            original = [
                np.asarray(pair[side + "_decision"]["eligibility"], dtype=np.float64)
                for side in ("rejected", "accepted")
            ]
            delta = original[1] - original[0]
            deltas[feedback["pair_id"]] = delta
            for side, eligibility in zip(("rejected", "accepted"), original, strict=True):
                record = feedback[side + "_decision"]
                verify_decision(record, eligibility)
                require(
                    record["storage_sha256"] == models[0]._storage_root, "feedback source differs"
                )
                original_record = pair[side + "_decision"]
                for key in ("query", "candidate_index", "emitted", "eligibility_sha256"):
                    require(record[key] == original_record[key], "replay changed occurrence")
            for model, mode in zip(models, ("update", "frozen"), strict=True):
                receipt = feedback[mode]
                require(
                    receipt["eligibility_delta_sha256"] == sha256(delta.tobytes()).hexdigest(),
                    "feedback delta hash differs",
                )
                # Restore the serialized feedback lineage only; never call a
                # learner/update or infer intermediate weights from final weights.
                for side, reward in (("rejected", -1), ("accepted", 1)):
                    model._feedback_root = digest(
                        [model._feedback_root, receipt[side + "_decision_sha256"], reward]
                    )
                model.feedback_count += 2
                model.update_count += int(receipt["native_update_applied"])
            verified += 1
    for model, mode, snapshot in zip(
        models, ("update", "frozen"), (arm["trained"], arm["frozen_trained"]), strict=True
    ):
        model._weights[:] = archive[f"{name}_{mode}_readout_synapses_pa"]
        model._bias[:] = archive[f"{name}_{mode}_pacemaker_synapses_pa"]
        require(model.synapse_sha256 == snapshot["synapse_sha256"], "NPZ synapse hash differs")
        require(
            model.checkpoint_sha256 == snapshot["checkpoint_sha256"], "trained checkpoint differs"
        )
        require(model.storage_summary == snapshot["storage"], "trained storage/counts differ")
    require(
        not np.any(models[1]._weights) and not np.any(models[1]._bias), "frozen weights changed"
    )
    update = models[0]
    pairs, cache = [], {}
    for pair in arm["ledger"]:
        rejected, accepted = pair["rejected_decision"], pair["accepted_decision"]
        query = rejected["query"]
        require(query == accepted["query"], "pair query differs")
        if query not in cache:
            cache[query] = update._features(query.encode())[0]
        for decision in (rejected, accepted):
            index = decision["candidate_index"]
            require(index < update._storage_ancestry[decision["storage_sha256"]], "late occurrence")
            require(
                update._candidates[index].emitted.decode() == decision["emitted"],
                "source bytes differ",
            )
            eligibility = cache[query][index]
            require(
                np.array_equal(eligibility, decision["eligibility"]), "source eligibility differs"
            )
            verify_decision(decision, eligibility)
        require(
            not pair["rejected_execution"]["passed"] and pair["teacher_execution"]["passed"],
            "pair lacks recorded FAIL/PASS",
        )
        delta = deltas[pair["pair_id"]]
        weights = update._weights[0] - update._weights[1]
        offers = [
            f for r in arm["training"] for f in r["feedback"] if f["pair_id"] == pair["pair_id"]
        ]
        pairs.append(
            {
                "pair_id": pair["pair_id"],
                "query": query,
                "rejected": occurrence(update, rejected["candidate_index"]),
                "accepted": occurrence(update, accepted["candidate_index"]),
                "final_margin": float(delta @ weights),
                "first_gap_after": offers[0]["update"]["gap_after"],
                "last_gap_after": offers[-1]["update"]["gap_after"],
                "offer_count": len(offers),
                "block_margin": contributions(delta, weights),
            }
        )
    return models, seen, pairs, verified


def score_queries(memory, items, pairs):
    weights = memory._weights[0] - memory._weights[1]
    bias = float(memory._bias[0] - memory._bias[1])
    result = []
    for item in items:
        rows, _ = memory._features(item["request"].encode())
        scores = rows @ weights + bias
        best = float(scores.max())
        tops = np.flatnonzero(scores == best)
        expected = None if item["value"] is None else item["value"].encode()
        exact = [i for i, c in enumerate(memory._candidates) if c.emitted == expected]
        correct = [
            i
            for i in exact
            if memory._documents[memory._candidates[i].document_index].document_id
            == (item["document_id"] + "/0").encode()
        ]
        require(expected is None or correct, "fixture has no source-correct occurrence")
        best_correct = None if not correct else max(correct, key=lambda i: scores[i])
        negatives = {p["rejected"]["index"] for p in pairs if p["query"] == item["request"]}

        def candidate(index, scores=scores, rows=rows):
            return {
                **occurrence(memory, index),
                "raw_score": float(scores[index]),
                "score_minus_stop": float(scores[index] - THRESHOLD),
                "block_contributions": contributions(rows[index], weights),
                "bias": bias,
            }

        result.append(
            {
                "case": item["name"],
                "query": item["request"],
                "stratum": item.get("stratum", "taught"),
                "expected_value": item["value"],
                "candidate_count": len(rows),
                "raw_top_tie_count": len(tops),
                "raw_top": [candidate(int(i)) for i in tops],
                "top_five": [candidate(int(i)) for i in np.argsort(-scores, kind="stable")[:5]],
                "correct_occurrence_count": len(correct),
                "same_value_occurrence_count": len(exact),
                "best_correct": None if best_correct is None else candidate(best_correct),
                "correct_rank": None
                if best_correct is None
                else 1 + int(np.count_nonzero(scores > scores[best_correct])),
                "correct_minus_top": None
                if best_correct is None
                else float(scores[best_correct] - best),
                "unique_top_source_correct": len(tops) == 1 and int(tops[0]) in correct,
                "unique_top_value_match": len(tops) == 1 and int(tops[0]) in exact,
                "top_below_stop": best <= THRESHOLD,
                "top_was_same_query_executed_negative": any(int(i) in negatives for i in tops),
            }
        )
    return result


def summarize(rows):
    return {
        stratum: {
            "queries": len(group),
            "unique_unthresholded_source_correct": sum(
                r["unique_top_source_correct"] for r in group
            ),
            "unique_unthresholded_value_match": sum(r["unique_top_value_match"] for r in group),
            "top_below_stop": sum(r["top_below_stop"] for r in group),
            "wrong_top_never_executed_as_same_query_negative": sum(
                r["expected_value"] is not None
                and not r["unique_top_value_match"]
                and not r["top_was_same_query_executed_negative"]
                for r in group
            ),
        }
        for stratum in sorted({r["stratum"] for r in rows})
        if (group := [r for r in rows if r["stratum"] == stratum])
    }


def run(source, output):
    require(not output.exists(), "refuse to overwrite diagnostic evidence")
    require(os.environ.get("HUMANBRAIN_DEVICE") == "cpu", "set CPU before Python")
    require(
        Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve() == ROOT / ".native-cache",
        "set checkout native cache before Python",
    )
    began = time.perf_counter()
    artifact = json.loads(source.read_text())
    require(
        artifact["terminal"] == "completed" and artifact["sources_unchanged_during_run"],
        "incomplete source",
    )
    for path, expected in artifact["source_sha256"].items():
        require(
            sha256((ROOT / path).read_bytes()).hexdigest() == expected,
            f"source hash differs: {path}",
        )
    checkpoint = Path(artifact["synapse_checkpoint"]["path"])
    require(
        sha256(checkpoint.read_bytes()).hexdigest() == artifact["synapse_checkpoint"]["sha256"],
        "NPZ hash differs",
    )
    fixtures = json.loads(
        (ROOT / "experiments/r4_integrated/contextual_binding_cases.json").read_text()
    )
    session.validate_fixture(fixtures)
    encoder = FrozenContextEncoder(ROOT / ".models/all-MiniLM-L6-v2")
    require(encoder.metadata == artifact["encoder"], "encoder metadata differs")
    report = {
        "kind": "read-only saved-checkpoint score diagnosis; no learning or actions",
        "scope": "completed public requests only; labels evaluator-only; "
        "rankings are not executed successes",
        "source_artifact_sha256": sha256(source.read_bytes()).hexdigest(),
        "source_revision": artifact["base_revision"],
        "checkout_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": artifact["source_sha256"],
        "diagnostic_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "npz_sha256": artifact["synapse_checkpoint"]["sha256"],
        "stop_raw_threshold": THRESHOLD,
        "arms": {},
    }
    with np.load(checkpoint, allow_pickle=False) as archive:
        for name, arm in artifact["arms"].items():
            models, seen, pairs, verified = reconstruct(arm, name, fixtures, archive, encoder)
            update, frozen = models
            taught = score_queries(update, fixtures["training"], pairs)
            for item in fixtures["transfer"]:
                session.prior._ingest_once(item, models, seen)
            require(
                update.checkpoint_sha256 == arm["transfer_start"]["checkpoint_sha256"],
                "evaluation parent differs",
            )
            parents = [m.checkpoint_sha256 for m in models]
            matched = 0
            for policy in session.evaluation_policies(
                update, frozen, binding=arm["candidate_interactions"]
            ):
                for phase in ("transfer", "retention"):
                    for row in arm[phase][policy.name]["rows"]:
                        actual = session.shared.decision_record(
                            policy.respond(row["selection"]["query"].encode())
                        )
                        require(
                            actual == row["selection"],
                            f"ordinary decision differs: {name}/{phase}/{policy.name}/"
                            f"{row['case']}",
                        )
                        matched += 1
            scored = score_queries(update, fixtures["transfer"] + fixtures["training"], pairs)
            require(
                parents == [m.checkpoint_sha256 for m in models],
                "read-only diagnosis changed checkpoint",
            )
            margins = [p["final_margin"] for p in pairs]
            first = arm["ledger"][0]
            delta = np.asarray(first["accepted_decision"]["eligibility"]) - np.asarray(
                first["rejected_decision"]["eligibility"]
            )
            first_step = arm["learning_rate"] * delta / np.sqrt(1 + delta @ delta)
            report["arms"][name] = {
                "trained_checkpoint_sha256": arm["trained"]["checkpoint_sha256"],
                "evaluation_checkpoint_sha256": parents[0],
                "ordinary_decisions_exactly_matched": matched,
                "feedback_offers_verified": verified,
                "native_updates": update.update_count,
                "frozen_native_updates": frozen.update_count,
                "feedback_count": update.feedback_count,
                "final_weight_max_absolute_difference_from_first_update_formula": float(
                    np.max(np.abs(update._weights - np.stack((first_step, -first_step))))
                ),
                "ledger_summary": arm["ledger_summary"],
                "training_summary": arm["training_summary"],
                "recorded_pair_margin_summary": {
                    "count": len(pairs),
                    "min": min(margins),
                    "median": float(np.median(margins)),
                    "max": max(margins),
                    "below_unit_margin": sum(m < 1 - 1e-10 for m in margins),
                    "wrong_pairs": sum(m <= 0 for m in margins),
                },
                "pairs": pairs,
                "trained_pool_rank_summary": summarize(taught),
                "trained_pool_rows": taught,
                "evaluation_pool_rank_summary": summarize(scored),
                "evaluation_pool_rows": scored,
            }
            print(
                name, json.dumps(report["arms"][name]["evaluation_pool_rank_summary"]), flush=True
            )
    native = loaded_build_info()
    report["native"] = {
        "source_digest": native.source_digest,
        "binary_sha256": sha256(native.library_path.read_bytes()).hexdigest(),
        "library": str(native.library_path),
    }
    require(report["native"] == artifact["native"], "native identity differs")
    report["elapsed_seconds"] = time.perf_counter() - began
    for path, expected in artifact["source_sha256"].items():
        require(
            sha256((ROOT / path).read_bytes()).hexdigest() == expected,
            "source changed during diagnosis",
        )
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "benchmarks/contextual_binding_python_development_v1.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/contextual_binding_diagnostic_v1.json"
    )
    arguments = parser.parse_args()
    run(arguments.source.resolve(), arguments.output.resolve())
