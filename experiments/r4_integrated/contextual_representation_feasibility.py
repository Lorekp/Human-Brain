"""One TRAIN-only external linear feasibility reference; never learner training.

All taught answers must outrank every other retained candidate by one and clear
the unchanged raw STOP threshold by one. Pure-query weights and persistent bias
are zero. Solver weights are archived, never installed in the neuronal model.
Only --selfcheck may run before the project source/fixture/protocol freeze.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import time
import tracemalloc
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
QUERY_COLUMNS = 384
STOP = float(np.arctanh((600.1 - 600.0) / 200.0))
VERIFY_TOLERANCE = 1e-7
SOLVER_OPTIONS = {
    "time_limit": 300.0,
    "presolve": True,
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
}
SOURCES = (
    "experiments/r4_integrated/contextual_representation_feasibility.py",
    "src/humanbrain/contextual_evidence_memory.py",
    "src/humanbrain/frozen_context_encoder.py",
    "src/humanbrain/opaque_document_event_tape.py",
    "src/humanbrain/native.py",
    "src/humanbrain/resting_glif_race.py",
    "benchmarks/native_scaling.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def constraints(features, teachers):
    """Rows: each query's ascending wrong indices, then its STOP constraint."""
    query_count, candidate_count, width = features.shape
    require(len(teachers) == query_count, "teacher/query count mismatch")
    require(np.isfinite(features).all(), "nonfinite input features")
    free = np.arange(QUERY_COLUMNS, width, dtype=np.int64)
    matrix = np.empty((query_count * candidate_count, len(free)), dtype=np.float64)
    rhs = np.ones(query_count * candidate_count, dtype=np.float64)
    for q, teacher in enumerate(teachers):
        require(0 <= teacher < candidate_count, "teacher index outside candidates")
        wrong = np.delete(np.arange(candidate_count), teacher)
        start = q * candidate_count
        matrix[start : start + candidate_count - 1] = (
            features[q, teacher, free][None, :] - features[q, wrong][:, free]
        )
        matrix[start + candidate_count - 1] = features[q, teacher, free]
        rhs[start + candidate_count - 1] = STOP + 1.0
    return matrix, rhs, free


def validate_witness(features, teachers, witness):
    """Independent score-space check; does not use the LP matrix or its residuals."""
    valid_shape = witness.shape == (features.shape[2],)
    finite = bool(valid_shape and np.isfinite(witness).all())
    query_zero = bool(
        valid_shape and np.array_equal(witness[:QUERY_COLUMNS], np.zeros(QUERY_COLUMNS))
    )
    summary = {
        "tolerance": VERIFY_TOLERANCE,
        "finite_witness": finite,
        "pure_query_weights_exactly_zero": query_zero,
        "persistent_bias": 0.0,
        "verified": False,
    }
    if not finite or not query_zero:
        return summary, {}
    # A different summation path from matrix@w; verify original unrounded scores.
    scores = np.einsum("qcf,f->qc", features, witness, optimize=False)
    if not np.isfinite(scores).all():
        summary["nonfinite_scores"] = True
        return summary, {"witness_raw_scores": scores}
    gaps = []
    query_rows = []
    for q, teacher in enumerate(teachers):
        wrong = np.delete(scores[q], teacher)
        ranking = scores[q, teacher] - wrong
        stop_gap = float(scores[q, teacher] - STOP)
        gaps.extend((*ranking, stop_gap))
        query_rows.append(
            {
                "query_index": q,
                "teacher_index": int(teacher),
                "teacher_score": float(scores[q, teacher]),
                "maximum_wrong_score": float(wrong.max()),
                "minimum_competitor_gap": float(ranking.min()),
                "teacher_minus_stop": stop_gap,
                "competitor_violations": int(np.count_nonzero(ranking < 1 - VERIFY_TOLERANCE)),
                "stop_violation": bool(stop_gap < 1 - VERIFY_TOLERANCE),
            }
        )
    gaps = np.asarray(gaps)
    if not np.isfinite(gaps).all():
        summary["nonfinite_gaps"] = True
        return summary, {"witness_raw_scores": scores, "verified_gaps": gaps}
    violations = int(np.count_nonzero(gaps < 1 - VERIFY_TOLERANCE))
    summary.update(
        {
            "verified": violations == 0,
            "constraint_count": len(gaps),
            "violation_count": violations,
            "minimum_gap": float(gaps.min()),
            "minimum_ranking_gap": min(row["minimum_competitor_gap"] for row in query_rows),
            "minimum_stop_gap": min(row["teacher_minus_stop"] for row in query_rows),
            "maximum_deficit": float(np.maximum(1 - gaps, 0).max()),
            "witness_l2": float(np.linalg.norm(witness)),
            "witness_max_absolute": float(np.abs(witness).max()),
            "raw_score_minimum": float(scores.min()),
            "raw_score_maximum": float(scores.max()),
            "queries": query_rows,
        }
    )
    return summary, {
        "witness_raw_scores": scores,
        "verified_gaps": gaps,
        "verified_residuals": gaps - 1,
    }


def solve(matrix, rhs):
    return linprog(
        np.zeros(matrix.shape[1]),
        A_ub=-matrix,
        b_ub=-rhs,
        bounds=(None, None),
        method="highs",
        options=SOLVER_OPTIONS,
    )


def selfcheck():
    """Inert arrays only; no encoder, model, native library or project fixture."""
    features = np.zeros((2, 2, QUERY_COLUMNS + 2))
    features[0, :, QUERY_COLUMNS:] = [[-2, 0], [0, 1]]
    features[1, :, QUERY_COLUMNS:] = [[0, 1], [-2, 0]]
    teachers = np.asarray([0, 1])
    matrix, rhs, free = constraints(features, teachers)
    np.testing.assert_array_equal(matrix, [[-2, -1], [-2, 0], [-2, -1], [-2, 0]])
    np.testing.assert_array_equal(rhs, [1, STOP + 1, 1, STOP + 1])
    result = solve(matrix, rhs)
    require(result.status == 0, "synthetic feasible solve failed")
    witness = np.zeros(features.shape[2])
    witness[free] = result.x
    require(witness[QUERY_COLUMNS] < 0, "feasible reference requires a negative free weight")
    require(
        validate_witness(features, teachers, witness)[0]["verified"], "witness validation failed"
    )
    require(
        not validate_witness(features, teachers, np.zeros_like(witness))[0]["verified"],
        "zero witness incorrectly accepted",
    )
    bad = witness.copy()
    bad[0] = 1
    require(
        not validate_witness(features, teachers, bad)[0]["verified"], "fixed query weight accepted"
    )
    bad[0] = np.nan
    require(
        not validate_witness(features, teachers, bad)[0]["verified"], "nonfinite witness accepted"
    )
    impossible = features[:1].copy()
    impossible[0, 1] = impossible[0, 0]
    matrix, rhs, _ = constraints(impossible, teachers[:1])
    require(solve(matrix, rhs).status == 2, "synthetic zero-contrast infeasibility was missed")
    return {
        "terminal": "passed",
        "scope": "synthetic feasible/infeasible/witness rejection only",
        "project_encodings": 0,
        "project_solves": 0,
        "scipy_version": scipy.__version__,
    }


def run(fixture, protocol, output, model_dir, expected_revision):
    require(
        not output.exists() and not output.with_suffix(".npz").exists(), "refuse existing outputs"
    )
    began = time.perf_counter()
    report = {
        "kind": "host-linear joint feasibility",
        "terminal": "started",
        "expected_source_revision": expected_revision,
        "verdict": "inconclusive",
    }
    tracemalloc.start()
    try:
        _run(fixture, protocol, output, model_dir, expected_revision, report, began)
        report["terminal"] = "completed"
    except Exception as error:
        report.update(
            terminal="error",
            verdict="inconclusive",
            error_type=type(error).__name__,
            error_message=str(error),
        )
    finally:
        report["elapsed_seconds"] = time.perf_counter() - began
        report["python_traced_peak_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": report["verdict"],
                "terminal": report["terminal"],
                "validation": report.get("validation"),
            }
        ),
        flush=True,
    )
    return report


def _run(fixture, protocol, output, model_dir, expected_revision, report, began):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(
        expected_revision and revision == expected_revision, "pass the frozen full source revision"
    )
    paths = [ROOT / name for name in SOURCES] + [fixture, protocol]
    hashes = {}
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        actual = path.read_bytes()
        committed = subprocess.check_output(
            ["git", "show", f"{expected_revision}:{name}"], cwd=ROOT
        )
        hashes[name] = sha256(actual).hexdigest()
        require(actual == committed, f"source differs from frozen commit: {name}")
    report.update(source_revision=revision, source_sha256=hashes, source_bytes_match_commit=True)
    require(os.environ.get("HUMANBRAIN_DEVICE") == "cpu", "set CPU before Python")
    require(
        Path(os.environ.get("HUMANBRAIN_NATIVE_CACHE", "")).resolve() == ROOT / ".native-cache",
        "set checkout native cache before Python",
    )
    # Runtime imports are deliberately absent from inert selfcheck.
    from benchmarks.native_scaling import _process_memory
    from humanbrain.contextual_evidence_memory import ContextualEvidenceMemory
    from humanbrain.frozen_context_encoder import FrozenContextEncoder
    from humanbrain.native import loaded_build_info
    from humanbrain.opaque_document_event_tape import OpaqueDocument

    roster = json.loads(fixture.read_text(encoding="utf-8"))
    require(
        set(roster) == {"kind", "provenance", "documents", "queries"},
        "fixture must be TRAIN-only projection",
    )
    documents, queries = roster["documents"], roster["queries"]
    require(
        len(documents) == 4 and len(queries) == 16, "expected four TRAIN docs and sixteen queries"
    )
    require(
        all(set(d) == {"document_id", "document"} for d in documents), "unexpected document fields"
    )
    require(all(set(q) == {"name", "request", "value"} for q in queries), "unexpected query fields")
    require(len({d["document_id"] for d in documents}) == 4, "repeated document identity")
    require(len({q["name"] for q in queries}) == 16, "repeated query identity")
    encoder = FrozenContextEncoder(model_dir)
    memory = ContextualEvidenceMemory(
        encoder,
        learning_enabled=False,
        interaction_gain=16,
        candidate_interactions=True,
        comparison_step_policy="fixed",
    )
    for document in documents:
        identity = document["document_id"]
        memory.ingest(
            (
                OpaqueDocument(
                    f"{identity}/0".encode(),
                    f"provenance/{identity}/0".encode(),
                    document["document"].encode(),
                ),
            )
        )
    require(len(memory._candidates) == 285, "unexpected candidate count")
    parent, storage, synapses = (
        memory.checkpoint_sha256,
        memory.storage_summary,
        memory.synapse_sha256,
    )
    features = np.stack([memory._features(q["request"].encode())[0] for q in queries])
    require(features.shape == (16, 285, 1621), "unexpected feature shape")
    teachers = []
    for query in queries:
        matches = [
            i for i, c in enumerate(memory._candidates) if c.emitted == query["value"].encode()
        ]
        require(len(matches) == 1, "teacher value must identify exactly one retained occurrence")
        teachers.append(matches[0])
    teachers = np.asarray(teachers, dtype=np.int64)
    extraction_seconds = time.perf_counter() - began
    print("TRAIN feature extraction complete; one fixed feasibility solve begins", flush=True)
    matrix, rhs, free = constraints(features, teachers)
    matrix_hash, rhs_hash = sha256(matrix.tobytes()).hexdigest(), sha256(rhs.tobytes()).hexdigest()
    report.update(
        {
            "contract": __doc__,
            "fixture_provenance": roster["provenance"],
            "provenance_source_opened": False,
            "queries": queries,
            "feature_shape": list(features.shape),
            "unrestricted_variable_count": len(free),
            "fixed_zero_query_columns": QUERY_COLUMNS,
            "persistent_bias": 0,
            "constraint_count": len(rhs),
            "constraint_matrix_sha256": matrix_hash,
            "constraint_rhs_sha256": rhs_hash,
            "constraints": {
                "order": "query order; ascending nonteacher candidates then STOP",
                "rows_per_query": 285,
                "competitor_rhs": 1.0,
                "stop_score_rhs": STOP + 1,
                "teacher_indices": teachers.tolist(),
                "all_nonteacher_candidates_included": True,
            },
            "solver": {
                "method": "scipy.optimize.linprog/highs",
                "scipy_version": scipy.__version__,
                "objective": "zero",
                "bounds": "all retained variables unrestricted",
                "options": SOLVER_OPTIONS,
            },
            "interpretation": "A verified witness is finite TRAIN representation capacity only; "
            "not neuronal learning, native-update-cone reachability, "
            "action execution or generalization. "
            "Failure/infeasibility/timeout is inconclusive without an independent certificate.",
            "encoder": encoder.metadata,
            "checkpoint_before": parent,
            "storage": storage,
            "extraction_seconds": extraction_seconds,
        }
    )
    scipy_source = Path(inspect.getfile(linprog))
    report["solver"]["scipy_source"] = {
        "path": str(scipy_source),
        "sha256": sha256(scipy_source.read_bytes()).hexdigest(),
    }
    native = loaded_build_info()
    report["native"] = {
        "source_digest": native.source_digest,
        "library": str(native.library_path),
        "binary_sha256": sha256(native.library_path.read_bytes()).hexdigest(),
    }
    require(
        native.library_path.resolve().is_relative_to(ROOT / ".native-cache"), "wrong native cache"
    )
    candidates = []
    for c in memory._candidates:
        d = memory._documents[c.document_index]
        candidates.append(
            {
                "document_id": d.document_id.decode(),
                "provenance_id": d.provenance_id.decode(),
                "source_sha256": sha256(d.content).hexdigest(),
                "byte_span": [c.start, c.end],
                "emitted": c.emitted.decode(),
            }
        )
    report["candidates"] = candidates
    solve_start = time.perf_counter()
    witness = np.empty(0, dtype=np.float64)
    try:
        result = solve(matrix, rhs)
        report["solver"].update(
            status=int(result.status),
            message=result.message,
            iterations=int(result.nit),
            success=bool(result.success),
        )
        if result.x is not None:
            witness = np.zeros(features.shape[2])
            witness[free] = result.x
    except Exception as error:
        report["solver"].update(status=None, error_type=type(error).__name__, message=str(error))
    report["solver_seconds"] = time.perf_counter() - solve_start
    verification_start = time.perf_counter()
    validation, verified_arrays = validate_witness(features, teachers, witness)
    report["validation"] = validation
    report["verdict"] = "feasible_witness_verified" if validation["verified"] else "inconclusive"
    if validation["verified"]:
        reduced_residuals = matrix @ witness[free] - rhs
        report["validation"]["lp_matrix_minimum_residual"] = float(reduced_residuals.min())
        report["validation"]["max_independent_residual_difference"] = float(
            np.max(np.abs(reduced_residuals - verified_arrays["verified_residuals"]))
        )
        require(
            reduced_residuals.min() >= -VERIFY_TOLERANCE,
            "LP residual and independent validation disagree",
        )
    report["verification_seconds"] = time.perf_counter() - verification_start
    require(
        memory.checkpoint_sha256 == parent and memory.synapse_sha256 == synapses,
        "feature extraction or solving changed learner state",
    )
    require(
        memory.storage_summary == storage and memory.feedback_count == memory.update_count == 0,
        "unexpected learner update or storage change",
    )
    report["checkpoint_after"] = memory.checkpoint_sha256
    report["learner_unchanged"] = True
    for path in paths:
        require(
            sha256(path.read_bytes()).hexdigest() == hashes[path.relative_to(ROOT).as_posix()],
            "source changed during run",
        )
    archive = output.with_suffix(".npz")
    with archive.open("xb") as stream:
        np.savez_compressed(
            stream,
            features=features,
            teacher_indices=teachers,
            free_columns=free,
            witness_weight_difference=witness,
            **verified_arrays,
        )
    report["archive"] = {
        "path": str(archive),
        "sha256": sha256(archive.read_bytes()).hexdigest(),
        "witness_assigned_to_model": False,
        "feature_array_bytes": features.nbytes,
        "witness_bytes": witness.nbytes,
    }
    report["process_memory"] = _process_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument(
        "--fixture", type=Path, default=HERE / "contextual_representation_train.json"
    )
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "contextual_representation_feasibility_protocol.md"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/contextual_representation_feasibility_v1.json",
    )
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models/all-MiniLM-L6-v2")
    parser.add_argument("--expected-source-revision")
    args = parser.parse_args()
    if args.selfcheck:
        print(json.dumps(selfcheck(), allow_nan=False))
    else:
        result = run(
            args.fixture.resolve(),
            args.protocol.resolve(),
            args.output.resolve(),
            args.model_dir.resolve(),
            args.expected_source_revision,
        )
        raise SystemExit(0 if result["terminal"] == "completed" else 1)
