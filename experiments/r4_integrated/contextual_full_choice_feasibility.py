"""Saved TRAIN-only host LP diagnostic. Witnesses never enter a learner."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.optimize._highspy import _core as highs_identity
from scipy.sparse import csr_matrix, eye, hstack, vstack

ROOT = Path(__file__).resolve().parents[2]
SOLVE_MARGIN = 1.000001
OPTIONS = {
    "presolve": True,
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
}


def _process_memory() -> dict[str, float]:
    if os.name != "nt":
        import resource

        peak_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return {
            "working_set_mib": peak_kib / 1024.0,
            "peak_working_set_mib": peak_kib / 1024.0,
            "private_memory_mib": peak_kib / 1024.0,
        }

    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    process = get_current_process()
    ok = get_process_memory_info(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    scale = 1024.0 * 1024.0
    return {
        "working_set_mib": counters.WorkingSetSize / scale,
        "peak_working_set_mib": counters.PeakWorkingSetSize / scale,
        "private_memory_mib": counters.PrivateUsage / scale,
    }


def sha(value):
    return hashlib.sha256(value).hexdigest()


def choice_rows(features):
    """Candidate [F,0] rows and one independent learned STOP coordinate."""
    result = np.zeros((len(features) + 1, features.shape[1] + 1))
    result[:-1, :-1] = features
    result[-1, -1] = 1.0
    return result


def verify_witness(provider, gold, witness, *, deadline=math.inf):
    """Independent scalar score summation, checking every non-gold choice."""
    minimum, count, violations, queries = math.inf, 0, 0, []
    max_score_error = max_absolute_product_sum = 0.0
    score_agreement = True
    if not np.isfinite(witness).all():
        return {"verified": False, "reason": "nonfinite_witness"}
    for query, target in enumerate(gold):
        if time.perf_counter() >= deadline:
            return {"verified": False, "reason": "verification_timeout", "checked": count}
        rows = choice_rows(provider(query))
        scores = []
        for row in rows:
            products = [float(x) * float(w) for x, w in zip(row, witness, strict=True)]
            scores.append(math.fsum(products))
            max_absolute_product_sum = max(
                max_absolute_product_sum, math.fsum(abs(x) for x in products)
            )
        ordinary = rows @ witness
        errors = np.abs(ordinary - scores)
        max_score_error = max(max_score_error, float(errors.max()))
        score_agreement = score_agreement and bool(np.all(errors <= 1e-10 + 1e-10 * np.abs(scores)))
        margins = [scores[target] - score for j, score in enumerate(scores) if j != target]
        if not all(math.isfinite(x) for x in margins):
            return {"verified": False, "reason": "nonfinite_margin"}
        local = min(margins)
        minimum = min(minimum, local)
        count += len(margins)
        violations += sum(x < 1.0 for x in margins)
        queries.append({"query_index": query, "gold_index": target, "minimum_margin": local})
    return {
        "verified": violations == 0 and score_agreement,
        "matrix_scores_agree": score_agreement,
        "maximum_score_disagreement": max_score_error,
        "maximum_absolute_product_sum": max_absolute_product_sum,
        "constraint_count": count,
        "minimum_margin": minimum,
        "violations": violations,
        "queries": queries,
    }


def solve_cutting_plane(
    provider, gold, dimension, *, seconds=900.0, max_iterations=32, progress_state=None
):
    """Minimize L-infinity norm, adding each query's worst unseen violated row."""
    started = time.perf_counter()
    deadline = started + seconds
    witness = np.zeros(dimension)
    active, directions, seen, history = [], [], set(), []
    result = {"status": "inconclusive", "reason": "iteration_limit", "iterations": history}
    if progress_state is not None:
        progress_state.update(
            witness=witness, active_constraints=active, active_directions=directions
        )
    for iteration in range(max_iterations):
        if time.perf_counter() >= deadline:
            result["reason"] = "time_limit"
            break
        additions, worst, checked = [], math.inf, 0
        for query, target in enumerate(gold):
            rows = choice_rows(provider(query))
            if rows.shape[1] != dimension or not np.isfinite(rows).all():
                raise ValueError("invalid feature dimensions or values")
            if not 0 <= target < len(rows):
                raise ValueError("gold outside full choice inventory")
            scores = rows @ witness
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite separation scores")
            margins = scores[target] - scores
            margins[target] = math.inf
            worst = min(worst, float(margins.min()))
            checked += len(rows) - 1
            # argmin selects the first occurrence in an exact tie; STOP is last.
            index = int(np.argmin(margins))
            key = (query, index)
            if margins[index] < 1.0 and key not in seen:
                additions.append((key, rows[target] - rows[index]))
        if worst >= 1.0:
            verification = verify_witness(provider, gold, witness, deadline=deadline)
            result["verification"] = verification
            if verification["verified"]:
                result.update(status="feasible_verified", reason="all_original_margins_verified")
                break
        if not additions:
            result["reason"] = "no_new_cut_with_unverified_witness"
            break
        for key, direction in additions:
            seen.add(key)
            active.append(key)
            directions.append(direction)
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            result["reason"] = "time_limit"
            break
        cuts = csr_matrix(-np.asarray(directions))
        cut_matrix = hstack((cuts, csr_matrix((len(active), 1))), format="csr")
        identity = eye(dimension, format="csr")
        minus_t = csr_matrix(-np.ones((dimension, 1)))
        matrix = vstack(
            (cut_matrix, hstack((identity, minus_t)), hstack((-identity, minus_t))), format="csr"
        )
        rhs = np.r_[-np.full(len(active), SOLVE_MARGIN), np.zeros(2 * dimension)]
        objective = np.r_[np.zeros(dimension), 1.0]
        solved = linprog(
            objective,
            A_ub=matrix,
            b_ub=rhs,
            bounds=[(None, None)] * dimension + [(0, None)],
            method="highs-ds",
            options={**OPTIONS, "time_limit": remaining},
        )
        history.append(
            {
                "iteration": iteration,
                "active_count": len(active),
                "separated_count": checked,
                "minimum_margin_before": worst,
                "solver_status": int(solved.status),
                "objective_t": None if solved.fun is None else float(solved.fun),
                "message": solved.message,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if solved.x is not None and np.isfinite(solved.x).all():
            witness = solved.x[:-1].copy()
            if progress_state is not None:
                progress_state["witness"] = witness
        if solved.status != 0:
            result["reason"] = "solver_status_without_verified_certificate"
            break
    # The final iteration's solution also receives complete independent checking.
    if result["status"] != "feasible_verified" and time.perf_counter() < deadline:
        result["verification"] = verify_witness(provider, gold, witness, deadline=deadline)
        if result["verification"]["verified"]:
            result.update(status="feasible_verified", reason="all_original_margins_verified")
    result.update(
        elapsed_seconds=time.perf_counter() - started,
        active_count=len(active),
        witness_l2=float(np.linalg.norm(witness)),
        witness_linf=float(np.max(np.abs(witness))),
        witness_sha256=sha(witness.tobytes()),
    )
    return result, {
        "witness": witness,
        "active_constraints": np.asarray(active, dtype=np.int64),
        "active_directions": np.asarray(directions, dtype=np.float64),
    }


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def candidate_interactions(exact, semantic, intervals):
    """Independent loop ordering for published 72 joint +12 direct features."""
    exact = np.asarray(exact, dtype=np.float64)
    semantic = np.asarray(semantic, dtype=np.float64)
    intervals = np.asarray(intervals, dtype=np.int64)
    output = np.zeros((len(intervals), 84), dtype=np.float64)
    query_count, source_count = exact.shape
    if not query_count:
        return output
    positions = np.arange(source_count)
    kernels = [(side, scale) for side in (-1, 1) for scale in (2.0, 8.0, 32.0)]
    pair_distance = positions[:, None] - positions[None, :]
    anchor = []
    for side, scale in kernels:
        kernel = (pair_distance * side >= 0) * np.exp(-np.abs(pair_distance) / scale)
        anchor.append((exact[:, :, None] * kernel[None, :, :]).max(axis=1))
    matches = (exact, semantic)
    others = [(value.sum(axis=0)[None, :] - value) / max(1, query_count - 1) for value in matches]
    for candidate, (left, right) in enumerate(intervals):
        distance = np.where(
            positions < left, positions - left, np.where(positions > right, positions - right, 0)
        )
        for b, (side, scale) in enumerate(kernels):
            proximity = (distance * side >= 0) * np.exp(-np.abs(distance) / scale)
            for kind, value in enumerate(matches):
                output[candidate, 72 + b * 2 + kind] = (value * proximity).max(axis=1).mean()
                for a in range(6):
                    output[candidate, (a * 6 + b) * 2 + kind] = (
                        (anchor[a] * others[kind] * proximity).max(axis=1).mean()
                    )
    return output


def candidate_interactions_ordered(exact, semantic, intervals):
    """Published array fold ordering for hash identity, checked against loops."""
    exact = np.asarray(exact, dtype=np.float64)
    semantic = np.asarray(semantic, dtype=np.float64)
    intervals = np.asarray(intervals, dtype=np.int64)
    if not len(exact):
        return np.zeros((len(intervals), 84))
    positions = np.arange(exact.shape[1])
    distance = positions[:, None] - positions
    kernels = np.asarray(
        [
            (distance * side >= 0) * np.exp(-np.abs(distance) / scale)
            for side in (-1, 1)
            for scale in (2.0, 8.0, 32.0)
        ]
    )
    left, right = intervals[:, 0, None], intervals[:, 1, None]
    distance = np.where(
        positions < left, positions - left, np.where(positions > right, positions - right, 0)
    )
    proximity = np.asarray(
        [
            (distance * side >= 0) * np.exp(-np.abs(distance) / scale)
            for side in (-1, 1)
            for scale in (2.0, 8.0, 32.0)
        ]
    )
    anchor = (exact[None, :, :, None] * kernels[:, None, :, :]).max(axis=2)
    matches = np.stack((exact, semantic))
    other = (matches.sum(axis=1, keepdims=True) - matches) / max(1, len(exact) - 1)
    joint = np.empty((len(intervals), 6, 6, 2))
    direct = np.empty((len(intervals), 6, 2))
    for b in range(6):
        direct[:, b, :] = (matches[:, :, None, :] * proximity[b]).max(axis=-1).mean(axis=1).T
        for a in range(6):
            supported = other * anchor[a][None, :, :]
            joint[:, a, b, :] = (
                (supported[:, :, None, :] * proximity[b]).max(axis=-1).mean(axis=1).T
            )
    return np.concatenate(
        (joint.reshape(len(intervals), 72), direct.reshape(len(intervals), 12)), axis=1
    )


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def close(actual, reference, label):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape, (label, actual.shape, reference.shape)
    assert np.isfinite(actual).all() and np.isfinite(reference).all(), label
    error = np.abs(actual - reference)
    assert np.all(error <= 1e-10 + 1e-10 * np.abs(reference)), (label, float(error.max(initial=0)))
    return float(error.max(initial=0))


class FixedBasis:
    def __init__(self, metadata, arrays):
        self.meta, self.arrays = metadata, arrays
        self.candidates = metadata["candidates"]
        self.vectors = arrays["candidate_vectors"]
        self.contexts = arrays["context_vectors"]
        self.window_indices = arrays["candidate_window_indices"]
        self.queries = {row["query_hex"]: (i, row) for i, row in enumerate(metadata["queries"])}
        assert self.vectors.shape == (len(self.candidates), 384)
        assert self.contexts.ndim == 2 and self.contexts.shape[1] == 384
        assert self.window_indices.shape == (len(self.candidates),)
        assert len(self.queries) == 128
        for value in arrays.values():
            assert np.isfinite(value).all()
        self._validate_custody()

    def _validate_custody(self):
        root = digest({"representation": self.meta["representation"]})
        checked_candidates = 0
        for document_index, document in enumerate(self.meta["documents"]):
            raw = bytes.fromhex(document["content_hex"])
            indices = [
                i for i, c in enumerate(self.candidates) if c["document_index"] == document_index
            ]
            assert indices == list(range(checked_candidates, checked_candidates + len(indices)))
            for i in indices:
                c = self.candidates[i]
                assert c["document_id_hex"] == document["document_id_hex"]
                assert c["provenance_id_hex"] == document["provenance_id_hex"]
                assert c["content_sha256"] == hashlib.sha256(raw).hexdigest()
                assert raw[c["byte_start"] : c["byte_end"]].hex() == c["emitted_hex"]
            checked_candidates += len(indices)
            windows = [
                w for w in self.meta["interaction_windows"] if w["candidate_start"] in indices
            ]
            context_indices = sorted(set(int(self.window_indices[i]) for i in indices))
            assert len(windows) == len(context_indices)
            storage = {
                "parent": root,
                "documents": [
                    (
                        document["document_id_hex"],
                        document["provenance_id_hex"],
                        hashlib.sha256(raw).hexdigest(),
                    )
                ],
                "vectors": array_sha(self.vectors[:checked_candidates]),
                "contexts": hashlib.sha256(
                    b"".join(self.contexts[i].tobytes() for i in context_indices)
                ).hexdigest(),
                "interaction_windows": [
                    (
                        w["candidate_start"],
                        w["words"],
                        array_sha(np.asarray(w["word_candidate_indices"], dtype=np.int64)),
                        array_sha(np.asarray(w["candidate_word_spans"], dtype=np.int64)),
                    )
                    for w in windows
                ],
            }
            root = digest(storage)
        assert checked_candidates == len(self.candidates)
        assert root == self.meta["source_pool"]["sealed_sha256"]
        assert root == self.meta["source_pool"]["current_sha256"]
        self.storage_sha256 = root

    def features(self, query_hex):
        i, query = self.queries[query_hex]
        raw = bytes.fromhex(query_hex)
        assert hashlib.sha256(raw).hexdigest() == query["query_sha256"]
        words = re.findall(r"\w+", raw.decode("utf-8"))
        assert [word.casefold() for word in words] == query["words"]
        q = self.arrays["query_sentence_vectors"][i]
        start, count = query["word_offset"], query["word_count"]
        word_vectors = self.arrays["query_word_vectors"][start : start + count]
        assert count == len(words)
        overlap_words = set(re.findall(r"\w+", raw.decode("utf-8").casefold()))
        result = np.empty((len(self.candidates), 1621), dtype=np.float64)
        result[:, :384] = q
        result[:, 384:768] = self.vectors
        result[:, 768:1152] = 16.0 * (q * self.vectors)
        result[:, 1152:1536] = 16.0 * (q * self.contexts[self.window_indices])
        result[:, 1536] = [
            float(bytes.fromhex(c["emitted_hex"]).decode("utf-8").casefold() in overlap_words)
            for c in self.candidates
        ]
        for window in self.meta["interaction_windows"]:
            exact = np.asarray(query["words"])[:, None] == np.asarray(window["words"])[None, :]
            semantic = np.maximum(
                0, word_vectors @ self.vectors[window["word_candidate_indices"]].T
            )
            block = candidate_interactions_ordered(exact, semantic, window["candidate_word_spans"])
            independent = candidate_interactions(exact, semantic, window["candidate_word_spans"])
            close(block, independent, "independent candidate interaction blocks")
            left = window["candidate_start"]
            result[left : left + len(block), 1537:] = 16.0 * block
        assert np.isfinite(result).all()
        return result

    def gold(self, answer_hex):
        return (
            [len(self.candidates)]
            if answer_hex is None
            else [i for i, c in enumerate(self.candidates) if c["emitted_hex"] == answer_hex]
        )


def run(artifact_path, fixture_path, output, *, seconds=900.0, max_iterations=32):
    paths = [
        output,
        output.with_suffix(".npz"),
        output.with_suffix(".failure.json"),
        ROOT / ".pytest-tmp" / (output.stem + ".features.npy"),
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError("diagnostic output or scratch already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hard_deadline = datetime(2026, 9, 5, 9, 5, tzinfo=UTC).timestamp()
    seconds = min(seconds, hard_deadline - time.time())
    if seconds <= 0:
        raise TimeoutError("predeclared absolute deadline passed")
    report = {
        "scope": "TRAIN-only host feasibility; never deployed or learner training",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "numpy": np.__version__,
        "highs": ".".join(
            str(getattr(highs_identity, "HIGHS_VERSION_" + x)) for x in ("MAJOR", "MINOR", "PATCH")
        ),
        "scipy": scipy.__version__,
        "solve_margin": SOLVE_MARGIN,
        "verification_margin": 1.0,
        "solver_options": OPTIONS,
        "seconds": seconds,
        "max_iterations": max_iterations,
    }
    progress_state = {}
    try:
        sources = [
            Path(__file__),
            fixture_path,
            ROOT / "experiments/r4_integrated/contextual_full_choice_feasibility_protocol.md",
        ]
        report["source_sha256"] = {
            p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in sources
        }
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        archive_path = artifact_path.with_suffix(".npz")
        report["input_sha256"] = {
            "artifact": sha(artifact_path.read_bytes()),
            "npz": sha(archive_path.read_bytes()),
        }
        if report["input_sha256"] != {
            "artifact": "c25cfd867a203cc8ce6ba5f5b2af11cf880fd18ccab055fd5243c452b4671c08",
            "npz": "ff7f61a8b6134a78d578316faf24df0863128aab90891f18205c10374de1983b",
        }:
            raise ValueError("frozen input identity mismatch")
        if artifact["source_revision"] != "dd871c936893628f2deb37e79fa3c25da0f2be50":
            raise ValueError("frozen source revision mismatch")
        if artifact["terminal"] != "completed":
            raise ValueError("incomplete source artifact")
        if report["input_sha256"]["npz"] != artifact["checkpoint"]["sha256"]:
            raise ValueError("source archive hash mismatch")
        if (
            sha(fixture_path.read_bytes())
            != artifact["source_sha256"][fixture_path.relative_to(ROOT).as_posix()]
        ):
            raise ValueError("TRAIN fixture hash mismatch")
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {
                name: archive[name]
                for name in (
                    "candidate_vectors",
                    "context_vectors",
                    "candidate_window_indices",
                    "query_sentence_vectors",
                    "query_word_vectors",
                )
            }
        basis = FixedBasis(artifact["audit_basis"], arrays)
        tasks = fixture["tasks"]
        if len(tasks) != 64 or len(basis.candidates) != 1458:
            raise ValueError("unexpected fixed TRAIN inventory")
        features = np.lib.format.open_memmap(
            paths[3], mode="w+", dtype=np.float64, shape=(64, 1458, 1621)
        )
        gold, manifest = [], []
        prior_features = {}
        for event in artifact["training"]:
            prior_features.setdefault(event["task"], event["new"]["update"]["features_sha256"])
        for i, task in enumerate(tasks):
            if time.perf_counter() - started >= seconds:
                raise TimeoutError("feature reconstruction reached declared deadline")
            query_hex = task["query"].encode().hex()
            answer_hex = None if task["answer"] is None else task["answer"].encode().hex()
            targets = basis.gold(answer_hex)
            if len(targets) != 1:
                raise ValueError("joint linear diagnostic requires unique authorized gold")
            gold.append(targets[0])
            features[i] = basis.features(query_hex)
            if sha(features[i].tobytes()) != prior_features[task["id"]]:
                raise ValueError("reconstructed feature hash differs from frozen offer")
            manifest.append(
                {
                    "task": task["id"],
                    "query_hex": query_hex,
                    "answer_hex": answer_hex,
                    "gold_index": targets[0],
                    "features_sha256": sha(features[i].tobytes()),
                }
            )
        features.flush()
        report.update(
            queries=manifest,
            source_pool_sha256=basis.storage_sha256,
            scratch_feature_bytes=int(features.nbytes),
            scratch_feature_path=str(paths[3]),
            construction_seconds=time.perf_counter() - started,
        )
        remaining = seconds - (time.perf_counter() - started)
        result, saved = solve_cutting_plane(
            features.__getitem__,
            gold,
            1622,
            seconds=max(0, remaining),
            max_iterations=max_iterations,
            progress_state=progress_state,
        )
        report["result"] = result
        saved["gold_indices"] = np.asarray(gold, dtype=np.int64)
        report["array_sha256"] = {key: sha(value.tobytes()) for key, value in saved.items()}
        with paths[1].open("xb") as stream:
            np.savez_compressed(stream, **saved)
        report["checkpoint"] = {"path": str(paths[1]), "sha256": sha(paths[1].read_bytes())}
        report["elapsed_seconds"] = time.perf_counter() - started
        report["parent_memory"] = _process_memory()
        report["memory_scope"] = (
            "parent process including mapped pages and solver; scratch is also on disk"
        )
        if any(
            sha((ROOT / name).read_bytes()) != value
            for name, value in report["source_sha256"].items()
        ):
            raise ValueError("source changed during diagnostic")
        if sha(artifact_path.read_bytes()) != report["input_sha256"]["artifact"] or (
            sha(archive_path.read_bytes()) != report["input_sha256"]["npz"]
        ):
            raise ValueError("input artifact changed during diagnostic")
        report["terminal"] = "completed"
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
    except BaseException:
        report["terminal"] = "error"
        report["traceback"] = traceback.format_exc()
        if progress_state and not paths[1].exists():
            try:
                with paths[1].open("xb") as stream:
                    np.savez_compressed(
                        stream, **{key: np.asarray(value) for key, value in progress_state.items()}
                    )
                report["partial_checkpoint"] = {
                    "path": str(paths[1]),
                    "sha256": sha(paths[1].read_bytes()),
                }
            except Exception:
                pass
        try:
            with paths[2].open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(report, stream, allow_nan=False)
                stream.write("\n")
        except Exception:
            pass
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "benchmarks/contextual_full_choice_acquisition_v1.json",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "experiments/r4_integrated/contextual_full_choice_train.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/contextual_full_choice_feasibility_v1.json",
    )
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--max-iterations", type=int, default=32)
    args = parser.parse_args()
    completed = run(
        args.artifact.resolve(),
        args.fixture.resolve(),
        args.output.resolve(),
        seconds=args.seconds,
        max_iterations=args.max_iterations,
    )
    print(
        json.dumps(
            {
                "status": completed["result"]["status"],
                "elapsed_seconds": completed["elapsed_seconds"],
            }
        )
    )
