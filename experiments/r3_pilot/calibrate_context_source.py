"""TRAIN-only calibration of candidate sparse context assemblies for R3 C1.

The fast ``--check`` path validates the committed artifact without importing
NumPy or constructing a circuit.  ``--replay-check`` is the explicitly
requested, approximately one-minute physical replay.  ``--write`` regenerates
the artifact from that same replay; no held-out book is ever decoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = Path(__file__).with_name("context_source_calibration.json")

ARTIFACT_ID = "r3-context-source-calibration-v1"
SCHEMA_VERSION = 1
CORPUS_SHA256 = "d585e5644505b34ca516446f9b4fa2e95ee8ffa860d27858a7ca6450ee9ad3aa"
TOKENIZER_SHA256 = "130f42fc92e5c1f42c53d8904aeefbc4db21c189dc5b9fae98157dc06c9218a2"
PREFIX_200_SHA256 = "de5d527b12968762d6bd614c3a77c705e053e272361a72d20f3c0a21b272ef3c"
REJECTION = "REJECT_CURRENT_GLIF5_CONTEXT_SOURCE"
KS = (20, 32, 64)
SELECTOR_STEPS = 1_500
NONFINITE_PROBE_STEPS = 1_600
LANE_CELLS = 8_192
TOKEN_CELLS = 1_536
INPUT_CURRENT_PA = 250_000.0
SEED = 17
MAX_K = max(KS)
CHECKPOINT_STEPS = (512, 1_024, 1_500, 1_536)
SELECTOR_NAMES = (
    "earliest_refractory",
    "active_low_gid",
    "count_desc",
    "eligibility_short",
    "eligibility_long",
    "voltage_high",
)


class CalibrationError(RuntimeError):
    """The artifact or physical replay violates the frozen calibration."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the fast, source-controlled contract without replaying cells."""
    _require(artifact.get("artifact_id") == ARTIFACT_ID, "artifact id mismatch")
    _require(artifact.get("schema_version") == SCHEMA_VERSION, "schema mismatch")
    scope = artifact.get("scope")
    _require(isinstance(scope, dict), "scope must be an object")
    _require(scope.get("corpus_sha256") == CORPUS_SHA256, "corpus hash mismatch")
    _require(
        scope.get("tokenizer_sha256") == TOKENIZER_SHA256,
        "tokenizer hash mismatch",
    )
    _require(
        scope.get("first_200_token_ids_sha256") == PREFIX_200_SHA256,
        "TRAIN prefix hash mismatch",
    )
    _require(scope.get("train_books_decoded") == 49, "TRAIN book count mismatch")
    _require(
        scope.get("held_out_books_decoded") == 0,
        "held-out books must never be decoded",
    )
    _require(scope.get("held_out_books_skipped") == 17, "held-out split mismatch")
    _require(scope.get("selector_steps") == 1_500, "selector horizon mismatch")

    simulation = artifact.get("simulation")
    _require(isinstance(simulation, dict), "simulation must be an object")
    _require(simulation.get("lane_cells") == 8_192, "lane size mismatch")
    _require(simulation.get("input_current_pa") == 250_000, "drive mismatch")
    _require(simulation.get("seed") == 17, "seed mismatch")

    selectors = artifact.get("selectors")
    _require(isinstance(selectors, dict), "selectors must be an object")
    _require("earliest_refractory" in selectors, "first-spike proxy is missing")
    earliest = selectors["earliest_refractory"]
    _require(isinstance(earliest, dict), "first-spike selector must be an object")
    _require(
        set(earliest.get("by_k", {})) == {str(k) for k in KS},
        "first-spike K grid mismatch",
    )

    deep_state = artifact.get("deep_state")
    _require(isinstance(deep_state, dict), "deep-state evidence is missing")
    first_bad = deep_state.get("first_nonfinite")
    _require(isinstance(first_bad, dict), "first non-finite record is missing")
    _require(first_bad.get("step_index_zero_based") == 1_578, "non-finite step drift")

    verdict = artifact.get("verdict")
    _require(isinstance(verdict, dict), "verdict must be an object")
    _require(verdict.get("status") == REJECTION, "rejection verdict mismatch")
    _require(verdict.get("selected_k") is None, "no K may be selected")
    _require(
        verdict.get("implement_native_first_k") is False,
        "native first-K is not justified",
    )


def load_and_validate(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(artifact, dict), "artifact root must be an object")
    validate_artifact(artifact)
    return artifact


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rounded(value: float, digits: int = 6) -> float:
    result = round(float(value), digits)
    return 0.0 if result == 0.0 else result


def _order10(maximum_absolute_value: float) -> int:
    _require(
        math.isfinite(maximum_absolute_value) and maximum_absolute_value > 0.0,
        "order-10 diagnostic requires one positive finite value",
    )
    return math.floor(math.log10(maximum_absolute_value))


def _load_train_prefix(limit: int) -> tuple[Any, dict[str, Any]]:
    """Reconstruct canonical ids while decoding TRAIN spans only."""
    import numpy as np

    src = REPO_ROOT / "src"
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(src))
    from experiments.r3_pilot.run_c1_cohort import (  # noqa: PLC0415
        build_concept_stack,
        segment_verses,
    )

    freeze_dir = Path(
        os.environ.get("TEMP", os.environ.get("LOCALAPPDATA", ""))
    ) / "hb_bible_freeze"
    freeze = json.loads(
        (freeze_dir / "FREEZE-CANDIDATE-v3-BIBLE.json").read_text(encoding="utf-8")
    )
    spans = json.loads(
        (freeze_dir / "evidence" / "pg10_book_spans_v3.json").read_text(
            encoding="utf-8"
        )
    )
    trimmed = (freeze_dir / "evidence_pg10.trimmed.bin").read_bytes()
    corpus_sha = _sha256(trimmed)
    _require(corpus_sha == CORPUS_SHA256, "canonical corpus bytes drifted")
    _require(
        corpus_sha == freeze["corpus"]["sha256_trimmed"],
        "freeze manifest disagrees with corpus bytes",
    )

    train_books: list[tuple[str, str]] = []
    decoded_held_out = 0
    skipped_held_out = 0
    for book in spans["books"]:
        # This branch precedes decode deliberately: held-out bytes are never
        # converted to text or passed to the tokenizer in this diagnostic.
        if book["role"] != "train":
            skipped_held_out += 1
            continue
        start, end = book["span_offsets"]
        train_books.append(
            (book["short_name"], trimmed[start:end].decode("utf-8"))
        )
    _require(len(train_books) == 49, "TRAIN split must contain 49 books")
    _require(skipped_held_out == 17, "held-out split must contain 17 books")
    _require(decoded_held_out == 0, "held-out decode counter must stay zero")

    stack, stack_info = build_concept_stack([text for _, text in train_books])
    tokenizer_payload = {
        "concepts_hex": [payload.hex() for payload in stack.concepts],
        "tag": stack_info["tokenizer"],
    }
    tokenizer_sha = _sha256(_canonical_json_bytes(tokenizer_payload))
    _require(tokenizer_sha == TOKENIZER_SHA256, "canonical tokenizer drifted")
    _require(stack_info == {
        "base_vocab": 256,
        "concepts": 512,
        "tokenizer": (
            "BytePhraseTokenizer(max_phrases=0,min_frequency=3)+"
            "ConceptTokenizer(freq>=50,MI>=5,cap512)"
        ),
    }, "tokenizer construction metadata drifted")

    ids: list[int] = []
    for _, text in train_books:
        for verse in segment_verses(text):
            ids.extend(int(token) for token in stack.encode(verse))
            if len(ids) >= limit:
                break
        if len(ids) >= limit:
            break
    _require(len(ids) >= limit, "TRAIN prefix is shorter than the probe horizon")
    prefix = np.asarray(ids[:limit], dtype=np.int32)
    prefix_200_sha = _sha256(",".join(map(str, prefix[:200])).encode("utf-8"))
    _require(prefix_200_sha == PREFIX_200_SHA256, "canonical TRAIN prefix drifted")
    return prefix, {
        "corpus_sha256": corpus_sha,
        "first_200_token_ids_sha256": prefix_200_sha,
        "held_out_books_decoded": decoded_held_out,
        "held_out_books_skipped": skipped_held_out,
        "selector_prefix_sha256": _sha256(prefix[:SELECTOR_STEPS].tobytes()),
        "tokenizer_sha256": tokenizer_sha,
        "train_books_decoded": len(train_books),
    }


def _top_descending(values: Any, count: int) -> Any:
    """Highest values with an exact ascending-GID tie break."""
    import numpy as np

    candidate = np.argpartition(values, values.size - count)[values.size - count :]
    cutoff = float(values[candidate].min())
    above = np.flatnonzero(values > cutoff)
    tied = np.flatnonzero(values == cutoff)
    selected = np.concatenate((above, tied[: count - above.size])).astype(np.int32)
    _require(selected.size == count, "top-K selector encountered non-finite state")
    return selected[np.lexsort((selected, -values[selected]))]


def _nearest_other(positions: list[int], anchor: int) -> int | None:
    offset = bisect_left(positions, anchor)
    candidates: list[int] = []
    if offset > 0:
        candidates.append(positions[offset - 1])
    if offset < len(positions) and positions[offset] != anchor:
        candidates.append(positions[offset])
    if offset + 1 < len(positions):
        candidates.append(positions[offset + 1])
    if not candidates:
        return None
    return min(candidates, key=lambda value: (abs(value - anchor), value))


def _paired_occurrences(ids: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Pair each repeated bigram anchor with hard and unrelated negatives."""
    import numpy as np

    by_current: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in range(1, SELECTOR_STEPS):
        by_current[int(ids[index])][int(ids[index - 1])].append(index)

    anchors: list[int] = []
    same: list[tuple[int, int]] = []
    different: list[tuple[int, int]] = []
    for groups in by_current.values():
        if len(groups) < 2:
            continue
        for previous, positions in groups.items():
            positions = sorted(positions)
            other = sorted(
                position
                for other_previous, other_positions in groups.items()
                if other_previous != previous
                for position in other_positions
            )
            for anchor in positions:
                same_partner = _nearest_other(positions, anchor)
                different_partner = _nearest_other(other, anchor)
                if same_partner is None or different_partner is None:
                    continue
                anchors.append(anchor)
                same.append((anchor, same_partner))
                different.append((anchor, different_partner))

    rng = np.random.default_rng(SEED)
    unrelated: list[tuple[int, int]] = []
    for anchor in anchors:
        while True:
            candidate = int(rng.integers(1, SELECTOR_STEPS))
            if (
                ids[candidate] != ids[anchor]
                and ids[candidate - 1] != ids[anchor - 1]
            ):
                unrelated.append((anchor, candidate))
                break

    same_array = np.asarray(same, dtype=np.int32)
    different_array = np.asarray(different, dtype=np.int32)
    unrelated_array = np.asarray(unrelated, dtype=np.int32)
    _require(
        len(same_array) == len(different_array) == len(unrelated_array) == 1_132,
        "paired occurrence population drifted",
    )
    return same_array, different_array, unrelated_array, {
        "anchors": len(same_array),
        "different_previous_lag_median": _rounded(
            np.quantile(abs(different_array[:, 0] - different_array[:, 1]), 0.5)
        ),
        "different_previous_lag_p90": _rounded(
            np.quantile(abs(different_array[:, 0] - different_array[:, 1]), 0.9)
        ),
        "same_previous_lag_median": _rounded(
            np.quantile(abs(same_array[:, 0] - same_array[:, 1]), 0.5)
        ),
        "same_previous_lag_p90": _rounded(
            np.quantile(abs(same_array[:, 0] - same_array[:, 1]), 0.9)
        ),
    }


def _overlaps(rank: Any, k: int, pairs: Any) -> Any:
    import numpy as np

    dense = np.zeros((SELECTOR_STEPS, LANE_CELLS), dtype=np.uint8)
    selected = rank[:, :k]
    rows = np.repeat(np.arange(SELECTOR_STEPS, dtype=np.int32), k)
    flattened = selected.reshape(-1)
    valid = flattened >= 0
    dense[rows[valid], flattened[valid]] = 1
    packed = np.packbits(dense, axis=1, bitorder="little")
    del dense
    popcount = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
    output = np.empty(len(pairs), dtype=np.int16)
    for start in range(0, len(pairs), 512):
        block = pairs[start : start + 512]
        output[start : start + len(block)] = popcount[
            np.bitwise_and(packed[block[:, 0]], packed[block[:, 1]])
        ].sum(axis=1, dtype=np.int16)
    return output


def _hypergeometric_theta10(k: int, overlap: int) -> float:
    low = max(10, 20 - (k - overlap))
    high = min(20, overlap)
    if low > high:
        return 0.0
    numerator = sum(
        math.comb(overlap, hits) * math.comb(k - overlap, 20 - hits)
        for hits in range(low, high + 1)
    )
    return numerator / math.comb(k, 20)


def _auc(positive: Any, negative: Any) -> float:
    import numpy as np

    ordered = np.sort(negative)
    left = np.searchsorted(ordered, positive, side="left")
    right = np.searchsorted(ordered, positive, side="right")
    return float(np.mean((left + 0.5 * (right - left)) / len(ordered)))


def _selector_metric(same: Any, different: Any, unrelated: Any, k: int) -> dict[str, Any]:
    import numpy as np

    _require(
        same.size == different.size == unrelated.size,
        "selector comparison populations must be paired",
    )
    same_probabilities = np.asarray(
        [_hypergeometric_theta10(k, int(value)) for value in same]
    )
    unrelated_probabilities = np.asarray(
        [_hypergeometric_theta10(k, int(value)) for value in unrelated]
    )
    return {
        "auc_same_vs_different_previous": _rounded(_auc(same, different)),
        "different_previous_mean_overlap": _rounded(np.mean(different)),
        "paired_occurrences": int(same.size),
        "same_previous_mean_overlap": _rounded(np.mean(same)),
        "same_previous_median_overlap": _rounded(np.quantile(same, 0.5)),
        "same_previous_theta10_probability": _rounded(
            np.mean(same_probabilities)
        ),
        "unrelated_theta10_probability": _rounded(
            np.mean(unrelated_probabilities)
        ),
    }


def run_replay() -> dict[str, Any]:
    """Run the frozen physical diagnostic and return deterministic evidence."""
    import numpy as np

    os.environ["HUMANBRAIN_DEVICE"] = "cpu"
    os.environ.setdefault(
        "HUMANBRAIN_NATIVE_CACHE", str((REPO_ROOT / ".native-cache").resolve())
    )
    src = REPO_ROOT / "src"
    sys.path.insert(0, str(src))
    from humanbrain.recurrent import (  # noqa: PLC0415
        RecurrentCellCircuit,
        RecurrentCircuitConfig,
    )

    ids, scope = _load_train_prefix(NONFINITE_PROBE_STEPS)
    ranks = {
        name: np.full((SELECTOR_STEPS, MAX_K), -1, dtype=np.int32)
        for name in SELECTOR_NAMES
    }
    lengths = {
        name: np.full(SELECTOR_STEPS, MAX_K, dtype=np.int16)
        for name in SELECTOR_NAMES
    }
    for name in ("earliest_refractory", "active_low_gid", "count_desc"):
        lengths[name].fill(0)
    active_sizes = np.empty(SELECTOR_STEPS, dtype=np.int32)
    spike_totals = np.empty(SELECTOR_STEPS, dtype=np.int64)
    checkpoints: list[dict[str, Any]] = []
    first_nonfinite: dict[str, Any] | None = None
    max_token_spike_count = 0

    circuit = RecurrentCellCircuit(
        TOKEN_CELLS,
        config=RecurrentCircuitConfig(
            reservoir_cells=LANE_CELLS,
            input_current_pa=INPUT_CURRENT_PA,
            random_seed=SEED,
        ),
    )
    for index, token_id in enumerate(ids):
        step = circuit.observe(int(token_id))
        state = circuit._reservoir.state
        arrays = {
            "after": state.after_spike_currents_pa,
            "eligibility": circuit._eligibility_traces,
            "synaptic_current": circuit._synaptic_current_pa,
            "threshold_spike": state.threshold_spike_mv,
            "threshold_voltage": state.threshold_voltage_mv,
            "voltage": state.voltage_relative_mv,
        }
        bad_counts = {
            name: int(array.size - np.count_nonzero(np.isfinite(array)))
            for name, array in arrays.items()
            if not np.all(np.isfinite(array))
        }
        step_count = index + 1
        if step_count in CHECKPOINT_STEPS:
            _require(not bad_counts, f"state became non-finite by step {step_count}")
            checkpoints.append(
                {
                    "all_finite": True,
                    "step_count": step_count,
                    "threshold_voltage_max_abs_order10": _order10(
                        float(np.max(np.abs(state.threshold_voltage_mv)))
                    ),
                    "voltage_max_abs_order10": _order10(
                        float(np.max(np.abs(state.voltage_relative_mv)))
                    ),
                }
            )
        if bad_counts:
            first_nonfinite = {
                "bad_value_counts": bad_counts,
                "current_token_active_cells": int(
                    np.count_nonzero(circuit._token_spike_counts)
                ),
                "observe_spike_events": int(step.reservoir_spikes),
                "step_count": step_count,
                "step_index_zero_based": index,
                "token_id": int(token_id),
            }
            break

        if index >= SELECTOR_STEPS:
            continue
        counts = circuit._token_spike_counts
        active = np.flatnonzero(counts).astype(np.int32, copy=False)
        active_sizes[index] = active.size
        spike_totals[index] = int(step.reservoir_spikes)
        max_token_spike_count = max(max_token_spike_count, int(counts.max()))
        selected_count = min(MAX_K, active.size)
        refractory = state.refractory_steps[active]
        ranks["earliest_refractory"][index, :selected_count] = active[
            np.lexsort((active, refractory))
        ][:selected_count]
        ranks["active_low_gid"][index, :selected_count] = active[:selected_count]
        ranks["count_desc"][index, :selected_count] = active[
            np.lexsort((active, -counts[active].astype(np.int64)))
        ][:selected_count]
        for name in ("earliest_refractory", "active_low_gid", "count_desc"):
            lengths[name][index] = selected_count
        ranks["eligibility_short"][index] = _top_descending(
            circuit._eligibility_traces[0], MAX_K
        )
        ranks["eligibility_long"][index] = _top_descending(
            circuit._eligibility_traces[-1], MAX_K
        )
        ranks["voltage_high"][index] = _top_descending(
            state.voltage_relative_mv, MAX_K
        )

    _require(first_nonfinite is not None, "expected non-finite state was not observed")
    _require(
        first_nonfinite["step_index_zero_based"] == 1_578,
        "first non-finite step drifted",
    )
    _require(max_token_spike_count == 1, "first-spike proxy lost single-spike proof")
    _require(
        np.array_equal(ranks["count_desc"], ranks["active_low_gid"]),
        "count-desc must equal low-GID when every count is one",
    )

    same_pairs, different_pairs, unrelated_pairs, pairing = _paired_occurrences(ids)
    selector_output: dict[str, Any] = {}
    validity = {
        "active_low_gid": "diagnostic_only_global_gid_bias",
        "eligibility_long": "diagnostic_only_persistent_trace",
        "eligibility_short": "diagnostic_only_short_trace",
        "earliest_refractory": "exact_first_spike_proxy_for_this_single_spike_run",
        "voltage_high": "diagnostic_only_pathological_voltage_state",
    }
    for name in (
        "earliest_refractory",
        "active_low_gid",
        "eligibility_short",
        "eligibility_long",
        "voltage_high",
    ):
        by_k: dict[str, Any] = {}
        for k in KS:
            valid = (
                (lengths[name][same_pairs[:, 0]] >= k)
                & (lengths[name][same_pairs[:, 1]] >= k)
                & (lengths[name][different_pairs[:, 1]] >= k)
                & (lengths[name][unrelated_pairs[:, 1]] >= k)
            )
            same = _overlaps(ranks[name], k, same_pairs[valid])
            different = _overlaps(ranks[name], k, different_pairs[valid])
            unrelated = _overlaps(ranks[name], k, unrelated_pairs[valid])
            by_k[str(k)] = _selector_metric(same, different, unrelated, k)
        selector_output[name] = {
            "assembly_rank_sha256": _sha256(ranks[name].tobytes()),
            "by_k": by_k,
            "validity": validity[name],
        }
    selector_output["count_desc"] = {
        "alias_of": "active_low_gid",
        "assembly_rank_sha256": _sha256(ranks["count_desc"].tobytes()),
        "identical": True,
        "validity": "diagnostic_only_all_counts_tie_at_one",
    }

    return {
        "artifact_id": ARTIFACT_ID,
        "deep_state": {
            "checkpoints": checkpoints,
            "first_nonfinite": first_nonfinite,
        },
        "pairing": pairing,
        "rejection_reasons": [
            "first-spike assemblies have no same-bigram advantage",
            "alternative selectors are context-independent stable cores or diagnostics",
            "dense GLIF5 voltage state diverges before the requested horizon",
        ],
        "schema_version": SCHEMA_VERSION,
        "scope": {
            **scope,
            "nonfinite_probe_steps": NONFINITE_PROBE_STEPS,
            "selector_steps": SELECTOR_STEPS,
        },
        "selectors": selector_output,
        "simulation": {
            "active_cells_max": int(active_sizes.max()),
            "active_cells_mean": _rounded(active_sizes.mean()),
            "active_cells_median": _rounded(np.quantile(active_sizes, 0.5)),
            "active_cells_min": int(active_sizes.min()),
            "active_size_sha256": _sha256(active_sizes.tobytes()),
            "input_current_pa": int(INPUT_CURRENT_PA),
            "lane_cells": LANE_CELLS,
            "max_token_spike_count": max_token_spike_count,
            "seed": SEED,
            "spike_cut_steps": int(circuit.cell_parameters.spike_cut_steps),
            "spike_total_sha256": _sha256(spike_totals.tobytes()),
            "token_steps": int(circuit._token_steps),
            "total_spike_events": int(spike_totals.sum()),
            "trace_taus_ms": [_rounded(value, 12) for value in circuit.config.eligibility_taus_ms],
        },
        "verdict": {
            "implement_native_first_k": False,
            "selected_k": None,
            "status": REJECTION,
        },
    }


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true", help="fast artifact check")
    actions.add_argument(
        "--replay-check",
        action="store_true",
        help="rerun the TRAIN-only physical probe and compare the artifact",
    )
    actions.add_argument(
        "--write",
        action="store_true",
        help="rerun the TRAIN-only physical probe and write the artifact",
    )
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.check:
            load_and_validate(args.artifact)
            print("context-source artifact check: PASS")
            return 0
        started = time.perf_counter()
        observed = run_replay()
        validate_artifact(observed)
        elapsed = time.perf_counter() - started
        if args.write:
            _write_artifact(args.artifact, observed)
            print(
                f"context-source artifact written: {args.artifact} "
                f"({elapsed:.1f}s)"
            )
            return 0
        expected = load_and_validate(args.artifact)
        _require(
            _canonical_json_bytes(observed) == _canonical_json_bytes(expected),
            "physical replay differs from committed artifact",
        )
        print(f"context-source replay check: PASS ({elapsed:.1f}s)")
        return 0
    except (CalibrationError, OSError, json.JSONDecodeError) as error:
        print(f"context-source calibration: FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
