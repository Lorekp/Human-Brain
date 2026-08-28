"""Compact completed canonical TRAIN-prefix cost artifacts into one curve."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any


OUTPUT_PATH = Path(__file__).with_name("r3_native_train_prefix_scaling.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        value = value[key]
    return value


def _same(payloads: list[dict[str, Any]], *keys: str) -> Any:
    values = [_nested(payload, *keys) for payload in payloads]
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError(f"scaling inputs disagree at {'.'.join(keys)}")
    return values[0]


def _log_log_fit(points: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float]:
    x = [math.log(float(point["prefix_token_count"])) for point in points]
    y = [math.log(float(_nested(point, *path))) for point in points]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    x_var = sum((value - x_mean) ** 2 for value in x)
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y, strict=True)
    ) / x_var
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * value for value in x]
    residual = sum(
        (actual - estimate) ** 2
        for actual, estimate in zip(y, predicted, strict=True)
    )
    total = sum((value - y_mean) ** 2 for value in y)
    return {
        "coefficient": math.exp(intercept),
        "prefix_exponent": slope,
        "r_squared_log_space": 1.0 - residual / total if total else 1.0,
    }


def _point(path: Path, raw: bytes, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("outcome") == "SCIENTIFIC_STOP":
        raise RuntimeError(
            f"{path} contains a SCIENTIFIC_STOP; preserve it directly instead of "
            "folding it into a completed curve"
        )
    measurement = payload["measurement"]
    projection = payload["cap_projection"]
    return {
        "prefix_token_count": payload["train_input"]["prefix_token_count"],
        "prefix_token_sha256_le_i32": payload["train_input"][
            "prefix_token_sha256_le_i32"
        ],
        "prefix_unique_token_ids": payload["train_input"]["prefix_unique_token_ids"],
        "source_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "source_generated_at_utc": payload["generated_at_utc"],
        "bounded_definition_identity_sha256": payload["definition"][
            "bounded_definition_identity_sha256"
        ],
        "bounded_segment_capacity": payload["definition"]["bounded_segment_capacity"],
        "transition_steps": measurement["transition_steps"],
        "latency_ns": measurement["latency_ns"],
        "recognition": measurement["recognition"],
        "mechanism": {
            "live_segments_final": measurement["live_segments_final"],
            "allocated_segments_total": measurement["allocated_segments_total"],
            "punished_segments_total": measurement["punished_segments_total"],
            "predicted_steps": measurement["predicted_steps"],
            "winner_spikes": measurement["winner_spikes"],
            "basket_spikes": measurement["basket_spikes"],
            "all_winner_to_basket_ticks": measurement[
                "all_winner_to_basket_ticks"
            ],
            "all_basket_to_gaba_ticks": measurement["all_basket_to_gaba_ticks"],
        },
        "replay_wall_ns": measurement["replay_wall_ns"],
        "per_step_replay_wall_ns": measurement["replay_wall_ns"]
        / payload["train_input"]["prefix_token_count"],
        "cap_projection": {
            "abi_call_steps_per_second": projection["abi_call_steps_per_second"],
            "instrumented_loop_steps_per_second": projection[
                "instrumented_loop_steps_per_second"
            ],
            "instrumented_loop_multiple_of_required": projection[
                "instrumented_loop_multiple_of_required"
            ],
            "instrumented_loop_projected_hours": projection[
                "instrumented_loop_projected_hours"
            ],
        },
    }


def main() -> int:
    args = _parse_args()
    loaded: list[tuple[Path, bytes, dict[str, Any]]] = []
    for path in args.inputs:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("benchmark") != "r3-native-train-prefix-cost":
            raise RuntimeError(f"{path} is not a native TRAIN-prefix cost artifact")
        if payload["train_input"]["held_out_books_materialized"] != 0:
            raise RuntimeError(f"{path} materialized held-out data")
        loaded.append((path, raw, payload))
    loaded.sort(key=lambda item: item[2]["train_input"]["prefix_token_count"])
    payloads = [payload for _path, _raw, payload in loaded]
    prefixes = [payload["train_input"]["prefix_token_count"] for payload in payloads]
    if len(prefixes) < 2 or len(set(prefixes)) != len(prefixes):
        raise RuntimeError("scaling curve needs at least two unique prefix lengths")

    points = [_point(path, raw, payload) for path, raw, payload in loaded]
    first = points[0]
    last = points[-1]
    ratio_paths = {
        "prefix_tokens": ("prefix_token_count",),
        "live_segments": ("mechanism", "live_segments_final"),
        "mean_candidate_segments": ("recognition", "candidate_segments", "mean"),
        "mean_postings_visited": ("recognition", "postings_visited", "mean"),
        "per_step_replay_wall_ns": ("per_step_replay_wall_ns",),
        "projected_hours": (
            "cap_projection",
            "instrumented_loop_projected_hours",
        ),
    }
    first_to_last_ratios = {
        name: float(_nested(last, *path)) / float(_nested(first, *path))
        for name, path in ratio_paths.items()
    }

    output = {
        "schema_version": 1,
        "benchmark": "r3-native-train-prefix-scaling",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            f"one completed run each over canonical first-book TRAIN prefixes "
            f"{prefixes}; synthetic pre-outcome authority, prefix-bounded segment "
            "capacity, no held-out/C1/capability/cohort evidence; full-run growth, "
            "checkpoints, I/O, and arm topology remain excluded"
        ),
        "train_input": {
            "candidate_id": _same(payloads, "train_input", "candidate_id"),
            "train_input_authority_sha256": _same(
                payloads, "train_input", "train_input_authority_sha256"
            ),
            "train_stream_sha256": _same(
                payloads, "train_input", "train_stream_sha256"
            ),
            "tokenizer_sha256": _same(payloads, "train_input", "tokenizer_sha256"),
            "prefix_book": _same(payloads, "train_input", "prefix_book"),
            "held_out_books_materialized": 0,
        },
        "definition": {
            "arm": _same(payloads, "definition", "arm"),
            "construction_seed": _same(payloads, "definition", "construction_seed"),
            "protocol_sha256": _same(payloads, "definition", "protocol_sha256"),
            "protocol_runnable": _same(payloads, "definition", "protocol_runnable"),
            "production_definition_identity_sha256": _same(
                payloads,
                "definition",
                "production_definition_identity_sha256",
            ),
            "production_segment_capacity": _same(
                payloads, "definition", "production_segment_capacity"
            ),
            "bounded_capacity_rule": _same(
                payloads, "definition", "bounded_capacity_rule"
            ),
            "column_count": _same(payloads, "definition", "column_count"),
            "max_pyramidal_cells": _same(
                payloads, "definition", "max_pyramidal_cells"
            ),
            "physical_cell_count": _same(
                payloads, "definition", "physical_cell_count"
            ),
            "token_code_count": _same(payloads, "definition", "token_code_count"),
            "threshold_offsets_sha256_le_f64": _same(
                payloads,
                "definition",
                "threshold_offsets_sha256_le_f64",
            ),
        },
        "measurement": {
            "one_run_per_prefix": True,
            "mode": _same(payloads, "measurement", "mode"),
            "timer": _same(payloads, "measurement", "timer"),
            "gc_during_replay": _same(
                payloads, "measurement", "gc_during_replay"
            ),
            "points": points,
        },
        "scaling": {
            "fit_form": "metric = coefficient * prefix_token_count ** prefix_exponent",
            "log_log_fits": {
                "live_segments": _log_log_fit(
                    points, ("mechanism", "live_segments_final")
                ),
                "mean_candidate_segments": _log_log_fit(
                    points, ("recognition", "candidate_segments", "mean")
                ),
                "mean_postings_visited": _log_log_fit(
                    points, ("recognition", "postings_visited", "mean")
                ),
                "per_step_replay_wall_ns": _log_log_fit(
                    points, ("per_step_replay_wall_ns",)
                ),
            },
            "first_to_last_ratios": first_to_last_ratios,
            "max_prefix_observation": {
                "prefix_token_count": last["prefix_token_count"],
                "live_segments_final": last["mechanism"]["live_segments_final"],
                "mean_candidate_segments": last["recognition"][
                    "candidate_segments"
                ]["mean"],
                "mean_postings_visited": last["recognition"]["postings_visited"][
                    "mean"
                ],
                "instrumented_loop_steps_per_second": last["cap_projection"][
                    "instrumented_loop_steps_per_second"
                ],
                "instrumented_loop_multiple_of_required": last["cap_projection"][
                    "instrumented_loop_multiple_of_required"
                ],
                "instrumented_loop_projected_hours": last["cap_projection"][
                    "instrumented_loop_projected_hours"
                ],
            },
        },
        "environment": {
            "python": _same(payloads, "environment", "python"),
            "platform": _same(payloads, "environment", "platform"),
            "processor": _same(payloads, "environment", "processor"),
            "native_source_digest": _same(
                payloads, "environment", "native_source_digest"
            ),
            "native_binary_size": _same(
                payloads, "environment", "native_binary_size"
            ),
            "native_binary_sha256": _same(
                payloads, "environment", "native_binary_sha256"
            ),
        },
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
