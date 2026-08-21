"""Frozen validation gate for genuine developmental recurrent-cell growth.

All schedules and ablations come from the protocol before the named validation
file is opened. There is no test partition and no post-validation adaptation.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import tracemalloc

import numpy as np

from humanbrain.development import (
    DevelopmentalGrowthConfig,
    DevelopmentalTraining,
    grow_developmentally,
)
from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig


ROOT = Path(__file__).resolve().parents[1]


def _content_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _bounded_training(paths: list[Path], maximum_characters: int) -> list[str]:
    per_file = max(2, maximum_characters // len(paths))
    texts = [path.read_text(encoding="utf-8")[:per_file] for path in paths]
    if any(len(text) < 2 for text in texts):
        raise ValueError("a registered training file is too short")
    return texts


def _common_prefix(left: bytes, right: bytes) -> int:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return min(len(left), len(right))


def _persistent_array_mebibytes(model: RecurrentCellLanguageModel) -> float:
    circuit = model._circuit
    arrays: list[np.ndarray] = []
    arrays.extend(value for value in vars(circuit).values() if isinstance(value, np.ndarray))
    arrays.extend(
        value for value in vars(circuit._reservoir).values() if isinstance(value, np.ndarray)
    )
    arrays.extend(
        value
        for value in vars(circuit._reservoir.state).values()
        if isinstance(value, np.ndarray)
    )
    unique = {id(array): array for array in arrays}
    return sum(array.nbytes for array in unique.values()) / (1024 * 1024)


def _state_diagnostics(
    model: RecurrentCellLanguageModel,
    text: str,
    maximum_tokens: int,
    seed: int,
) -> dict[str, float | int | str]:
    token_ids = model._tokenizer.encode(text)[:maximum_tokens]
    circuit = model._circuit
    circuit.reset_sequence()
    states: list[np.ndarray] = []
    reservoir_spikes = 0
    biological_time_ms = 0.0
    for token_id in token_ids:
        step = circuit.observe(token_id)
        reservoir_spikes += step.reservoir_spikes
        biological_time_ms += step.biological_time_ms
        states.append(circuit.eligibility)
    matrix = np.asarray(states, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values**2
    effective_rank = (
        float(variance.sum() ** 2 / np.square(variance).sum())
        if np.any(variance)
        else 0.0
    )
    standard_deviation = centered.std(axis=0)
    active_columns = np.flatnonzero(standard_deviation > 1e-12)
    pair_count = min(4_096, len(active_columns) * max(0, len(active_columns) - 1) // 2)
    if pair_count:
        rng = np.random.default_rng(seed)
        left = rng.choice(active_columns, pair_count, replace=True)
        right = rng.choice(active_columns, pair_count, replace=True)
        same = left == right
        while np.any(same):
            right[same] = rng.choice(active_columns, int(np.count_nonzero(same)), replace=True)
            same = left == right
        numerator = np.sum(centered[:, left] * centered[:, right], axis=0)
        denominator = np.sqrt(
            np.sum(np.square(centered[:, left]), axis=0)
            * np.sum(np.square(centered[:, right]), axis=0)
        )
        mean_correlation = float(np.mean(np.abs(numerator / denominator)))
    else:
        mean_correlation = 0.0
    return {
        "tokens": len(token_ids),
        "active_state_columns": int(len(active_columns)),
        "effective_state_rank": effective_rank,
        "mean_absolute_cell_correlation": mean_correlation,
        "correlation_measurement": f"deterministic sample of {pair_count} cell pairs",
        "reservoir_spikes": reservoir_spikes,
        "mean_spikes_per_cell_per_token": reservoir_spikes
        / (model.active_reservoir_cells * len(token_ids)),
        "mean_firing_hz_per_cell": reservoir_spikes
        / model.active_reservoir_cells
        / (biological_time_ms / 1_000.0),
    }


def _model_configuration(
    cells: int,
    seed: int,
    protocol: dict[str, object],
) -> RecurrentModelConfig:
    model = protocol["model"]
    return RecurrentModelConfig(
        growth_epochs=int(model["growth_epochs"]),
        circuit=RecurrentCircuitConfig(
            reservoir_cells=cells,
            input_fanout=min(int(model["input_fanout"]), cells),
            recurrent_fanout=int(model["overgrowth_recurrent_fanout"]),
            token_interval_ms=float(model["token_interval_ms"]),
            input_pulse_ms=float(model["input_pulse_ms"]),
            input_current_pa=float(model["input_current_pa"]),
            recurrent_current_pa=float(model["recurrent_current_pa"]),
            eligibility_tau_ms=float(model["eligibility_tau_ms"]),
            output_base_current_pa=float(model["output_base_current_pa"]),
            output_race_ms=float(model["output_race_ms"]),
            recurrent_topology="uniform_no_self",
            readout_learning_rule="three_factor",
            local_learning_rate_pa=float(model["local_learning_rate_pa"]),
            local_competitive_margin_pa=float(model["local_competitive_margin_pa"]),
            random_seed=seed,
        ),
    )


def _development_configuration(
    condition: str,
    cells: int,
    seed: int,
    protocol: dict[str, object],
) -> DevelopmentalGrowthConfig:
    developmental = protocol["developmental_schedule"]
    fractions = tuple(float(value) for value in developmental["active_cell_fractions"])
    staged = tuple(max(8, round(cells * fraction)) for fraction in fractions[:-1]) + (cells,)
    if condition == "full_size_from_start":
        staged = tuple(cells for _ in fractions)
    ordering = "random" if condition == "staged_random_order" else "layered_prefix"
    pruning = "random" if condition == "staged_random_pruning" else "activity"
    homeostasis = condition != "staged_no_homeostasis"
    return DevelopmentalGrowthConfig(
        stage_active_cells=staged,
        cell_ordering=ordering,
        initial_recurrent_fanout=int(developmental["initial_recurrent_fanout"]),
        final_recurrent_fanout=int(developmental["final_recurrent_fanout"]),
        pruning_rule=pruning,
        homeostatic_scaling=homeostasis,
        structural_seed=seed + int(developmental["structural_seed_offset"]),
        turnover_edges_per_source=int(developmental["turnover_edges_per_source"]),
        homeostatic_min_scale=float(developmental["homeostatic_scale_bounds"][0]),
        homeostatic_max_scale=float(developmental["homeostatic_scale_bounds"][1]),
    )


def _evaluate(
    training: DevelopmentalTraining,
    validation_text: str,
    protocol: dict[str, object],
    seed: int,
) -> dict[str, object]:
    model = training.model
    evaluation = protocol["evaluation"]
    teacher = model.evaluate(
        validation_text,
        max_predictions=int(evaluation["teacher_forced_predictions"]),
    )
    prompt_characters = int(evaluation["generation_prompt_characters"])
    generated_tokens = int(evaluation["generation_target_bytes"])
    prompt = validation_text[:prompt_characters]
    target = validation_text[
        prompt_characters : prompt_characters + generated_tokens
    ].encode("utf-8")
    started = perf_counter()
    generated_text = model.complete(prompt, max_new_tokens=generated_tokens)
    generation_seconds = perf_counter() - started
    generated = generated_text.encode("utf-8")
    compared = min(len(generated), len(target))
    correct = sum(generated[index] == target[index] for index in range(compared))
    diagnostics = _state_diagnostics(
        model,
        validation_text,
        int(evaluation["diagnostic_tokens"]),
        seed + 50_000,
    )
    return {
        "teacher_forced": {
            "predictions": teacher.predictions,
            "correct": teacher.correct,
            "accuracy": teacher.accuracy,
            "tokens_per_second": teacher.tokens_per_second,
        },
        "free_running": {
            "target_bytes": len(target),
            "generated_bytes": len(generated),
            "correct": correct,
            "accuracy": correct / len(target),
            "exact_prefix_bytes": _common_prefix(generated, target),
            "generated_preview": generated_text[:120],
        },
        "state_diagnostics": diagnostics,
        "validation_wall_seconds": {
            "teacher_forced": teacher.wall_seconds,
            "free_running": generation_seconds,
        },
    }


def _summary(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    groups = sorted({(int(run["reservoir_cells"]), str(run["condition"])) for run in runs})
    for cells, condition in groups:
        matches = [
            run
            for run in runs
            if int(run["reservoir_cells"]) == cells and run["condition"] == condition
        ]
        summaries.append(
            {
                "reservoir_cells": cells,
                "condition": condition,
                "runs": len(matches),
                "teacher_forced_median_accuracy": median(
                    float(run["teacher_forced"]["accuracy"]) for run in matches
                ),
                "free_running_median_accuracy": median(
                    float(run["free_running"]["accuracy"]) for run in matches
                ),
                "effective_state_rank_median": median(
                    float(run["state_diagnostics"]["effective_state_rank"])
                    for run in matches
                ),
                "mean_firing_hz_per_cell_median": median(
                    float(run["state_diagnostics"]["mean_firing_hz_per_cell"])
                    for run in matches
                ),
                "training_wall_seconds_median": median(
                    float(run["wall_seconds"]["training"]) for run in matches
                ),
            }
        )
    return summaries


def run(protocol: dict[str, object], protocol_hash: str, output: Path) -> dict[str, object]:
    data = protocol["data"]
    training_paths = [ROOT / path for path in data["training_files"]]
    validation_path = ROOT / str(data["validation_file"])
    training_texts = _bounded_training(training_paths, int(data["training_characters"]))
    if output.is_file():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["protocol_sha256"] != protocol_hash:
            raise ValueError("existing output was produced by a different protocol")
    else:
        report = {
            "protocol_version": protocol["version"],
            "protocol_sha256": protocol_hash,
            "selection_partition": "named validation file",
            "test_partition_accessed": False,
            "data": {
                "training": [_content_record(path) for path in training_paths],
                "validation_path": validation_path.relative_to(ROOT).as_posix(),
            },
            "runs": [],
            "complete": False,
        }
    completed = {str(measurement["key"]) for measurement in report["runs"]}
    conditions = [str(value) for value in protocol["conditions"]]
    sizes = [int(value) for value in protocol["model"]["reservoir_cells"]]
    seeds = [int(value) for value in protocol["model"]["random_seeds"]]
    for cells in sizes:
        for seed in seeds:
            for condition in conditions:
                key = f"cells={cells}:seed={seed}:condition={condition}"
                if key in completed:
                    continue
                gc.collect()
                tracemalloc.start()
                started = perf_counter()
                training = grow_developmentally(
                    training_texts,
                    _model_configuration(cells, seed, protocol),
                    _development_configuration(condition, cells, seed, protocol),
                )
                training_seconds = perf_counter() - started
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if not training.model._circuit.learning_frozen:
                    raise RuntimeError("validation opened before cell learning was frozen")
                # No model choice below this line: the schedule is fully frozen.
                validation_bytes = validation_path.read_bytes()
                report["data"]["validation"] = {
                    "path": validation_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(validation_bytes).hexdigest(),
                    "bytes": len(validation_bytes),
                }
                validation_text = validation_bytes.decode("utf-8")[
                    : int(data["validation_characters"])
                ]
                result = _evaluate(training, validation_text, protocol, seed)
                circuit = training.model._circuit
                result.update(
                    {
                        "key": key,
                        "condition": condition,
                        "reservoir_cells": cells,
                        "seed": seed,
                        "attempted_updates": training.attempted_updates,
                        "local_weight_updates": circuit.local_update_count,
                        "active_cells_by_stage": [asdict(stage) for stage in training.stages],
                        "learned_readout_synapses": training.model.learned_synapses,
                        "trained_structural_synapses": int(
                            np.count_nonzero(circuit._structural_exposure)
                        ),
                        "final_pruned_recurrent_synapses": training.final_pruned_synapses,
                        "final_recurrent_synapses": training.final_recurrent_synapses,
                        "ridge_normal_matrix_allocated": circuit._normal_matrix is not None,
                        "wall_seconds": {
                            "training": training_seconds,
                            **result.pop("validation_wall_seconds"),
                        },
                        "memory_mib": {
                            "training_peak_traced": peak_bytes / (1024 * 1024),
                            "persistent_arrays": _persistent_array_mebibytes(training.model),
                        },
                    }
                )
                report["runs"].append(result)
                report["summary"] = _summary(report["runs"])
                report["complete"] = len(report["runs"]) == len(conditions) * len(sizes) * len(
                    seeds
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            "key": key,
                            "teacher": result["teacher_forced"]["accuracy"],
                            "free": result["free_running"]["accuracy"],
                            "rank": result["state_diagnostics"]["effective_state_rank"],
                            "train_seconds": training_seconds,
                        }
                    ),
                    flush=True,
                )
                del training
                gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks" / "development_protocol_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "development_validation_v2.json",
    )
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    report = run(json.loads(protocol_bytes), sha256(protocol_bytes).hexdigest(), args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "runs": len(report["runs"]),
                "complete": report["complete"],
            }
        )
    )


if __name__ == "__main__":
    main()
