"""Run the frozen large-cell scaling and organization capability protocol.

The command is plan-only by default. Validation starts only when the caller
passes both ``--execute`` and the exact protocol SHA-256 printed by plan mode.
The output is incremental and resumes in the protocol's fixed stage order.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import tracemalloc
from dataclasses import asdict
from hashlib import sha256
from itertools import product
from math import comb, sqrt
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any

import numpy as np
from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig
from humanbrain.topology import RECURRENT_TOPOLOGIES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "benchmarks" / "large_scale_capability_protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "large_scale_capability_validation_v1.json"


def _canonical_lf_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _source_text(record: dict[str, object]) -> str:
    path = ROOT / str(record["path"])
    canonical = _canonical_lf_bytes(path)
    expected_size = int(record["canonical_lf_bytes"])
    expected_hash = str(record["canonical_lf_sha256"])
    actual_hash = sha256(canonical).hexdigest()
    if len(canonical) != expected_size or actual_hash != expected_hash:
        raise ValueError(
            f"frozen corpus content changed for {record['path']}: "
            f"expected {expected_size} bytes/{expected_hash}, "
            f"found {len(canonical)} bytes/{actual_hash}"
        )
    return canonical.decode("utf-8")


def _specifications(protocol: dict[str, Any]) -> list[dict[str, object]]:
    sizes = [int(value) for value in protocol["model"]["active_reservoir_cells"]]
    specifications: list[dict[str, object]] = []
    seen: set[str] = set()
    for stage_index, stage in enumerate(protocol["execution"]["stages"]):
        for topology in stage["topologies"]:
            for seed in stage["seeds"]:
                for cells in sizes:
                    key = (
                        f"stage={stage['name']}:track={stage['data_track']}:"
                        f"topology={topology}:seed={seed}:cells={cells}"
                    )
                    if key in seen:
                        raise ValueError(f"duplicate frozen run key: {key}")
                    seen.add(key)
                    specifications.append(
                        {
                            "key": key,
                            "stage_index": stage_index,
                            "stage": stage["name"],
                            "data_track": stage["data_track"],
                            "topology": topology,
                            "seed": int(seed),
                            "cells": cells,
                        }
                    )
    return specifications


def _assert_fresh_validation_history(
    protocol: dict[str, Any], protocol_path: Path
) -> None:
    validation_paths = [
        str(record["path"]) for record in protocol["data"]["validation"]["files"]
    ]
    candidates = {
        path
        for path in (ROOT / "artifacts").rglob("*")
        if path.is_file()
    }
    candidates.update((ROOT / "benchmarks").glob("*protocol*.json"))
    violations: list[str] = []
    for path in sorted(candidates):
        if path.resolve() == protocol_path.resolve():
            continue
        content = path.read_bytes()
        if path.suffix == ".json":
            try:
                payload = json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            if (
                isinstance(payload, dict)
                and payload.get("protocol_version") == protocol["version"]
            ):
                # A resumable output from this protocol necessarily names its
                # validation folds; it is later than registration, not prior
                # artifact history.
                continue
        for validation_path in validation_paths:
            if validation_path.encode("utf-8") in content:
                relative = path.relative_to(ROOT).as_posix()
                violations.append(f"{validation_path} in {relative}")
    if violations:
        raise ValueError(
            "fresh validation path appeared in pre-registration history: "
            + "; ".join(violations)
        )


def _validate_protocol(protocol: dict[str, Any], protocol_path: Path) -> None:
    if protocol["status"] != "preregistered-before-first-validation-run":
        raise ValueError("protocol is not marked as preregistered")
    data = protocol["data"]
    if data["test_partition_defined"] or data["test_partition_accessed"]:
        raise ValueError("large-scale validation protocol must not define or access test")
    if data["adaptation_on_validation"]:
        raise ValueError("validation adaptation is forbidden")
    model = protocol["model"]
    if model["readout_learning_rule"] != "three_factor" or model["ridge_allowed"]:
        raise ValueError("only the scalable cell-local readout is permitted")
    if not model["all_configured_cells_are_active"]:
        raise ValueError("dormant padding is forbidden")
    sizes = [int(value) for value in model["active_reservoir_cells"]]
    if sizes != [6_144, 24_576, 98_304, 131_072]:
        raise ValueError("the registered four-size scale curve changed")
    required = {
        "uniform_no_self",
        "random",
        "single_scale_groups",
        "nested_groups",
        "mixed_membership",
        "ring_1d",
        "sheet_2d",
        "layered_sheet",
        "volume_3d",
        "hypercube_4d",
    }
    registered = {
        topology
        for stage in protocol["execution"]["stages"]
        for topology in stage["topologies"]
    }
    if registered != required or not registered <= RECURRENT_TOPOLOGIES:
        raise ValueError("the frozen topology matrix is incomplete or unsupported")
    primary_seeds = next(
        stage["seeds"]
        for stage in protocol["execution"]["stages"]
        if stage["name"] == "uniform_sqrt_data"
    )
    if list(primary_seeds) != [17, 29, 43, 59, 71]:
        raise ValueError("the primary scale claim requires five frozen paired seeds")
    specifications = _specifications(protocol)
    if len(specifications) != int(protocol["execution"]["expected_runs"]):
        raise ValueError("expected run count does not match the frozen stages")
    train_paths = {record["path"] for record in data["training"]["files"]}
    validation_paths = {record["path"] for record in data["validation"]["files"]}
    if train_paths & validation_paths:
        raise ValueError("training and validation files overlap")
    for record in [*data["training"]["files"], *data["validation"]["files"]]:
        _source_text(record)
    _assert_fresh_validation_history(protocol, protocol_path)


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    _validate_protocol(protocol, path)
    return protocol, sha256(raw).hexdigest()


def _training_texts(
    protocol: dict[str, Any], data_track: str, cells: int
) -> list[str]:
    training = protocol["data"]["training"]
    per_file = int(
        training["tracks"][data_track]["characters_per_file_by_cells"][str(cells)]
    )
    texts = []
    for record in training["files"]:
        text = _source_text(record)
        if len(text) < per_file:
            raise ValueError(f"training file {record['path']} is shorter than its budget")
        texts.append(text[:per_file])
    return texts


def _configuration(
    protocol: dict[str, Any], cells: int, topology: str, seed: int
) -> RecurrentModelConfig:
    model = protocol["model"]
    input_fanout = max(4, round(cells / 12))
    return RecurrentModelConfig(
        max_phrases=0,
        growth_epochs=int(protocol["data"]["training"]["growth_epochs"]),
        circuit=RecurrentCircuitConfig(
            reservoir_cells=cells,
            input_fanout=input_fanout,
            recurrent_fanout=int(model["recurrent_fanout"]),
            token_interval_ms=float(model["token_interval_ms"]),
            input_pulse_ms=float(model["input_pulse_ms"]),
            input_current_pa=float(model["input_current_pa"]),
            recurrent_current_pa=float(model["recurrent_current_pa"]),
            eligibility_tau_ms=float(model["eligibility_tau_ms"]),
            output_base_current_pa=float(model["output_base_current_pa"]),
            output_race_ms=float(model["output_race_ms"]),
            readout_learning_rule="three_factor",
            local_learning_rate_pa=float(model["local_learning_rate_pa"]),
            local_competitive_margin_pa=float(
                model["local_competitive_margin_pa"]
            ),
            inhibitory_fraction=float(model["inhibitory_fraction"]),
            random_seed=seed,
            recurrent_topology=topology,
        ),
    )


def _all_host_arrays(model: RecurrentCellLanguageModel) -> list[np.ndarray]:
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
    return list({id(array): array for array in arrays}.values())


def _memory_report(model: RecurrentCellLanguageModel) -> dict[str, float]:
    host_bytes = sum(array.nbytes for array in _all_host_arrays(model))
    device_bytes = int(getattr(model._circuit._native_kernel, "device_bytes", 0))
    return {
        "host_persistent_array_mib": host_bytes / (1024 * 1024),
        "device_persistent_array_mib": device_bytes / (1024 * 1024),
    }


def _learning_audit(model: RecurrentCellLanguageModel) -> dict[str, object]:
    circuit = model._circuit
    readout = circuit._readout_synapses_pa
    nonzero = readout != 0.0
    attempted = int(circuit._growth_examples)
    applied = int(circuit.local_update_count)
    audit = {
        "configured_active_cells": int(model.active_reservoir_cells),
        "attempted_local_updates": attempted,
        "applied_local_updates": applied,
        "applied_update_fraction": applied / attempted if attempted else 0.0,
        "allocated_readout_synapses": int(readout.size),
        "nonzero_readout_synapses": int(np.count_nonzero(nonzero)),
        "readout_source_cells": int(np.count_nonzero(np.any(nonzero, axis=0))),
        "readout_source_cell_fraction": float(np.mean(np.any(nonzero, axis=0))),
        "readout_output_cells": int(np.count_nonzero(np.any(nonzero, axis=1))),
        "nonzero_pacemaker_synapses": int(
            np.count_nonzero(circuit._pacemaker_synapses_pa)
        ),
        "afferent_synapses": int(circuit._afferent_targets.size),
        "recurrent_synapses": int(
            circuit.active_reservoir_cells * circuit.config.recurrent_fanout
        ),
        "nonzero_recurrent_synapses": int(
            np.count_nonzero(
                circuit._recurrent_weights_pa[: circuit.active_reservoir_cells]
            )
        ),
        "ridge_normal_matrix_absent": circuit._normal_matrix is None,
        "ridge_target_moments_absent": circuit._target_moments is None,
        "ridge_feature_batch_absent": circuit._feature_batch is None,
    }
    del nonzero
    if not all(
        audit[name]
        for name in (
            "ridge_normal_matrix_absent",
            "ridge_target_moments_absent",
            "ridge_feature_batch_absent",
        )
    ):
        raise RuntimeError("forbidden quadratic ridge state was allocated")
    return audit


def _effective_rank_from_states(states: np.ndarray) -> float:
    centered = states - states.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues = np.linalg.eigvalsh(gram.astype(np.float64, copy=False))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    squared = float(np.square(eigenvalues).sum())
    return total * total / squared if squared else 0.0


def _sampled_correlation(states: np.ndarray, sample_cells: int) -> float:
    count = min(sample_cells, states.shape[1])
    indices = np.linspace(0, states.shape[1] - 1, count, dtype=np.int64)
    sampled = states[:, indices]
    variable = sampled.std(axis=0) > 1e-9
    sampled = sampled[:, variable]
    if sampled.shape[1] < 2:
        return 0.0
    correlation = np.corrcoef(sampled, rowvar=False)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    finite = upper[np.isfinite(upper)]
    return float(np.mean(np.abs(finite))) if finite.size else 0.0


def _state_diagnostics(
    model: RecurrentCellLanguageModel,
    text: str,
    maximum_tokens: int,
    sample_cells: int,
    active_epsilon: float,
    responsive_epsilon: float,
) -> dict[str, float | int]:
    token_ids = model._tokenizer.encode(text)[:maximum_tokens]
    if len(token_ids) < maximum_tokens:
        raise ValueError("state diagnostic validation file is shorter than its budget")
    cells = model.active_reservoir_cells
    states = np.empty((len(token_ids), cells), dtype=np.float32)
    active_counts = np.empty(len(token_ids), dtype=np.int64)
    model._circuit.reset_sequence()
    reservoir_spikes = 0
    biological_time_ms = 0.0
    spiking_tokens = 0
    for index, token_id in enumerate(token_ids):
        step = model._circuit.observe(token_id)
        reservoir_spikes += step.reservoir_spikes
        biological_time_ms += step.biological_time_ms
        spiking_tokens += int(step.reservoir_spikes > 0)
        eligibility = model._circuit.eligibility
        states[index] = eligibility
        active_counts[index] = np.count_nonzero(np.abs(eligibility) > active_epsilon)
    responsive = states.std(axis=0) > responsive_epsilon
    union_active = np.any(np.abs(states) > active_epsilon, axis=0)
    result = {
        "tokens": len(token_ids),
        "reservoir_spikes": reservoir_spikes,
        "spiking_token_fraction": spiking_tokens / len(token_ids),
        "reservoir_spikes_per_cell_per_token": reservoir_spikes
        / (cells * len(token_ids)),
        "mean_firing_hz_per_cell": reservoir_spikes
        / cells
        / (biological_time_ms / 1_000.0),
        "mean_active_eligibility_fraction": float(np.mean(active_counts / cells)),
        "maximum_active_eligibility_fraction": float(np.max(active_counts / cells)),
        "union_active_eligibility_fraction": float(np.mean(union_active)),
        "responsive_state_cells": int(np.count_nonzero(responsive)),
        "responsive_state_cell_fraction": float(np.mean(responsive)),
        "effective_state_rank_from_token_gram": _effective_rank_from_states(states),
        "fixed_sample_cells": min(sample_cells, cells),
        "fixed_sample_mean_absolute_cell_correlation": _sampled_correlation(
            states, sample_cells
        ),
    }
    return result


def _common_prefix(left: bytes, right: bytes) -> int:
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if left_value != right_value:
            return index
    return min(len(left), len(right))


def _capability_evaluation(
    model: RecurrentCellLanguageModel,
    records: list[dict[str, object]],
    evaluation: dict[str, Any],
) -> tuple[dict[str, object], dict[str, float]]:
    validation = [(str(record["path"]), _source_text(record)) for record in records]
    teacher_folds: list[dict[str, object]] = []
    teacher_started = perf_counter()
    for path, text in validation:
        result = model.evaluate(
            text,
            max_predictions=int(evaluation["teacher_forced_predictions_per_file"]),
        )
        teacher_folds.append({"file": path, **asdict(result)})
    teacher_seconds = perf_counter() - teacher_started

    free_folds: list[dict[str, object]] = []
    prompt_characters = int(evaluation["free_running_prompt_characters_per_file"])
    target_bytes = int(evaluation["free_running_target_bytes_per_file"])
    free_started = perf_counter()
    for path, text in validation:
        prompt = text[:prompt_characters]
        target = text[prompt_characters:].encode("utf-8")[:target_bytes]
        if len(target) != target_bytes:
            raise ValueError(f"validation file {path} is shorter than free-run budget")
        generated_text = model.complete(prompt, max_new_tokens=target_bytes)
        generated = generated_text.encode("utf-8")
        compared = min(len(generated), len(target))
        correct = sum(
            generated[index] == target[index] for index in range(compared)
        )
        free_folds.append(
            {
                "file": path,
                "target_bytes": len(target),
                "generated_bytes": len(generated),
                "correct": correct,
                "accuracy": correct / len(target),
                "exact_prefix_bytes": _common_prefix(generated, target),
                "generated_preview": generated_text[:120],
            }
        )
    free_seconds = perf_counter() - free_started

    teacher_predictions = sum(int(fold["predictions"]) for fold in teacher_folds)
    teacher_correct = sum(int(fold["correct"]) for fold in teacher_folds)
    free_predictions = sum(int(fold["target_bytes"]) for fold in free_folds)
    free_correct = sum(int(fold["correct"]) for fold in free_folds)
    capability = {
        "teacher_forced": {
            "predictions": teacher_predictions,
            "correct": teacher_correct,
            "accuracy": teacher_correct / teacher_predictions,
            "folds": teacher_folds,
        },
        "free_running": {
            "predictions": free_predictions,
            "correct": free_correct,
            "accuracy": free_correct / free_predictions,
            "mean_exact_prefix_bytes": mean(
                int(fold["exact_prefix_bytes"]) for fold in free_folds
            ),
            "folds": free_folds,
        },
    }
    return capability, {
        "teacher_forced_wall_seconds": teacher_seconds,
        "free_running_wall_seconds": free_seconds,
    }


def _run_one(protocol: dict[str, Any], specification: dict[str, object]) -> dict[str, object]:
    cells = int(specification["cells"])
    topology = str(specification["topology"])
    seed = int(specification["seed"])
    data_track = str(specification["data_track"])
    training_texts = _training_texts(protocol, data_track, cells)
    configuration = _configuration(protocol, cells, topology, seed)

    gc.collect()
    tracemalloc.start()
    training_started = perf_counter()
    model = RecurrentCellLanguageModel.grow(training_texts, configuration)
    training_seconds = perf_counter() - training_started
    _, training_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not model._circuit.learning_frozen:
        raise RuntimeError("validation reached a model before plasticity was frozen")
    if model.active_reservoir_cells != cells or model.reservoir_cells != cells:
        raise RuntimeError("configured cells were not all active")
    expected_dt = float(protocol["model"]["integration_step_ms"])
    if model._circuit.cell_parameters.dt_ms != expected_dt:
        raise RuntimeError("integration step differs from the frozen biological timestep")

    learning = _learning_audit(model)
    validation_records = protocol["data"]["validation"]["files"]
    diagnostic_text = _source_text(validation_records[0])
    evaluation = protocol["evaluation"]
    diagnostic_started = perf_counter()
    dynamics = _state_diagnostics(
        model,
        diagnostic_text,
        int(evaluation["state_diagnostic_tokens"]),
        int(evaluation["state_correlation_sample_cells"]),
        float(evaluation["eligibility_active_epsilon"]),
        float(evaluation["responsive_state_std_epsilon"]),
    )
    diagnostic_seconds = perf_counter() - diagnostic_started
    capability, evaluation_times = _capability_evaluation(
        model, validation_records, evaluation
    )
    memory = _memory_report(model)
    result = {
        **specification,
        "training_characters": sum(len(text) for text in training_texts),
        "growth_epochs": configuration.growth_epochs,
        "token_cells": model.vocabulary_size,
        "input_fanout": configuration.circuit.input_fanout,
        "recurrent_fanout": configuration.circuit.recurrent_fanout,
        "capability": capability,
        "dynamics": dynamics,
        "learning": learning,
        "performance": {
            "model_build_and_training_wall_seconds": training_seconds,
            "state_diagnostics_wall_seconds": diagnostic_seconds,
            **evaluation_times,
            **memory,
            "training_peak_traced_mib": training_peak / (1024 * 1024),
            "simulation_backend": model._circuit.simulation_backend,
            "simulation_device": model._circuit.simulation_device,
        },
    }
    del model
    gc.collect()
    return result


def _nested_value(item: dict[str, Any], path: str) -> float:
    value: Any = item
    for part in path.split("."):
        value = value[part]
    return float(value)


def _exact_sign_p(values: list[float]) -> float:
    nonzero = [value for value in values if value != 0.0]
    if not nonzero:
        return 1.0
    positives = sum(value > 0.0 for value in nonzero)
    return sum(comb(len(nonzero), count) for count in range(positives, len(nonzero) + 1)) / (
        2 ** len(nonzero)
    )


def _exact_sign_flip_p(values: list[float]) -> float:
    if not values:
        return 1.0
    observed = mean(values)
    if observed <= 0.0:
        return 1.0
    permuted = [
        mean(sign * value for sign, value in zip(signs, values, strict=True))
        for signs in product((-1.0, 1.0), repeat=len(values))
    ]
    return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)


def _paired_statistics(seed_deltas: dict[int, float]) -> dict[str, object]:
    ordered = sorted(seed_deltas.items())
    values = [value for _, value in ordered]
    count = len(values)
    average = mean(values)
    critical = {2: 12.7062047364, 3: 4.3026527299, 4: 3.1824463053, 5: 2.7764451052}
    if count >= 2:
        half_width = critical[count] * stdev(values) / sqrt(count)
        confidence: list[float] | None = [average - half_width, average + half_width]
    else:
        confidence = None
    return {
        "paired_seeds": [seed for seed, _ in ordered],
        "per_seed_deltas": {str(seed): value for seed, value in ordered},
        "count": count,
        "mean_delta": average,
        "median_delta": median(values),
        "minimum_delta": min(values),
        "maximum_delta": max(values),
        "positive_directions": sum(value > 0.0 for value in values),
        "nonnegative_directions": sum(value >= 0.0 for value in values),
        "paired_t_95_interval": confidence,
        "exact_one_sided_sign_p": _exact_sign_p(values),
        "exact_one_sided_sign_flip_permutation_p": _exact_sign_flip_p(values),
    }


def _matches(
    runs: list[dict[str, Any]],
    *,
    data_track: str,
    topology: str,
    cells: int,
) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if run["data_track"] == data_track
        and run["topology"] == topology
        and int(run["cells"]) == cells
    ]


def _paired_effect(
    runs: list[dict[str, Any]],
    *,
    metric: str,
    candidate_track: str,
    candidate_topology: str,
    candidate_cells: int,
    baseline_track: str,
    baseline_topology: str,
    baseline_cells: int,
) -> dict[str, object] | None:
    candidate = {
        int(run["seed"]): run
        for run in _matches(
            runs,
            data_track=candidate_track,
            topology=candidate_topology,
            cells=candidate_cells,
        )
    }
    baseline = {
        int(run["seed"]): run
        for run in _matches(
            runs,
            data_track=baseline_track,
            topology=baseline_topology,
            cells=baseline_cells,
        )
    }
    seeds = sorted(candidate.keys() & baseline.keys())
    if not seeds:
        return None
    deltas = {
        seed: _nested_value(candidate[seed], metric) - _nested_value(baseline[seed], metric)
        for seed in seeds
    }
    return {
        "metric": metric,
        "candidate": {
            "data_track": candidate_track,
            "topology": candidate_topology,
            "cells": candidate_cells,
        },
        "baseline": {
            "data_track": baseline_track,
            "topology": baseline_topology,
            "cells": baseline_cells,
        },
        **_paired_statistics(deltas),
    }


def _condition_summary(
    runs: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, object]]:
    expected: dict[tuple[str, str, int], set[int]] = {}
    for specification in _specifications(protocol):
        key = (
            str(specification["data_track"]),
            str(specification["topology"]),
            int(specification["cells"]),
        )
        expected.setdefault(key, set()).add(int(specification["seed"]))
    rows: list[dict[str, object]] = []
    for (track, topology, cells), seeds in expected.items():
        matches = _matches(runs, data_track=track, topology=topology, cells=cells)
        if not matches:
            continue
        rows.append(
            {
                "data_track": track,
                "topology": topology,
                "cells": cells,
                "completed_seeds": sorted(int(run["seed"]) for run in matches),
                "expected_seeds": sorted(seeds),
                "complete": {int(run["seed"]) for run in matches} == seeds,
                "teacher_forced_median_accuracy": median(
                    _nested_value(run, "capability.teacher_forced.accuracy")
                    for run in matches
                ),
                "free_running_median_accuracy": median(
                    _nested_value(run, "capability.free_running.accuracy")
                    for run in matches
                ),
                "responsive_state_fraction_median": median(
                    _nested_value(run, "dynamics.responsive_state_cell_fraction")
                    for run in matches
                ),
                "readout_source_cell_fraction_median": median(
                    _nested_value(run, "learning.readout_source_cell_fraction")
                    for run in matches
                ),
                "mean_firing_hz_per_cell_median": median(
                    _nested_value(run, "dynamics.mean_firing_hz_per_cell")
                    for run in matches
                ),
                "attempted_local_updates_median": median(
                    _nested_value(run, "learning.attempted_local_updates")
                    for run in matches
                ),
                "applied_local_updates_median": median(
                    _nested_value(run, "learning.applied_local_updates")
                    for run in matches
                ),
                "host_persistent_array_mib_median": median(
                    _nested_value(run, "performance.host_persistent_array_mib")
                    for run in matches
                ),
            }
        )
    return rows


def _scale_criterion(
    runs: list[dict[str, Any]], protocol: dict[str, Any], track: str
) -> dict[str, object] | None:
    teacher = _paired_effect(
        runs,
        metric="capability.teacher_forced.accuracy",
        candidate_track=track,
        candidate_topology="uniform_no_self",
        candidate_cells=131_072,
        baseline_track=track,
        baseline_topology="uniform_no_self",
        baseline_cells=6_144,
    )
    free = _paired_effect(
        runs,
        metric="capability.free_running.accuracy",
        candidate_track=track,
        candidate_topology="uniform_no_self",
        candidate_cells=131_072,
        baseline_track=track,
        baseline_topology="uniform_no_self",
        baseline_cells=6_144,
    )
    large = _matches(
        runs, data_track=track, topology="uniform_no_self", cells=131_072
    )
    if teacher is None or free is None or len(large) != 5:
        return None
    thresholds = protocol["analysis"]["primary_numeric_criterion"]
    responsive = median(
        _nested_value(run, "dynamics.responsive_state_cell_fraction") for run in large
    )
    readout_source = median(
        _nested_value(run, "learning.readout_source_cell_fraction") for run in large
    )
    firing = median(
        _nested_value(run, "dynamics.mean_firing_hz_per_cell") for run in large
    )
    checks = {
        "teacher_median_delta": teacher["median_delta"]
        >= float(thresholds["teacher_forced_median_absolute_delta_minimum"]),
        "teacher_seed_directions": teacher["positive_directions"]
        >= int(thresholds["teacher_forced_positive_seed_directions_minimum"]),
        "free_median_delta": free["median_delta"]
        >= float(thresholds["free_running_median_absolute_delta_minimum"]),
        "free_seed_directions": free["nonnegative_directions"]
        >= int(thresholds["free_running_nonnegative_seed_directions_minimum"]),
        "responsive_state_density": responsive
        >= float(thresholds["large_model_responsive_state_fraction_minimum"]),
        "readout_source_density": readout_source
        >= float(thresholds["large_model_readout_source_cell_fraction_minimum"]),
        "firing_lower_bound": firing
        >= float(thresholds["large_model_median_firing_hz_per_cell_minimum"]),
        "firing_upper_bound": firing
        <= float(thresholds["large_model_median_firing_hz_per_cell_maximum"]),
    }
    return {
        "data_track": track,
        "teacher_forced_effect": teacher,
        "free_running_effect": free,
        "large_model_medians": {
            "responsive_state_cell_fraction": responsive,
            "readout_source_cell_fraction": readout_source,
            "mean_firing_hz_per_cell": firing,
        },
        "checks": checks,
        "passes": all(checks.values()),
    }


def _condition_median(
    runs: list[dict[str, Any]],
    track: str,
    topology: str,
    cells: int,
    metric: str,
    seeds: set[int] | None = None,
) -> float | None:
    matches = _matches(runs, data_track=track, topology=topology, cells=cells)
    if seeds is not None:
        matches = [run for run in matches if int(run["seed"]) in seeds]
    if len(matches) < 3:
        return None
    return median(_nested_value(run, metric) for run in matches)


def _organization_criterion(
    runs: list[dict[str, Any]], candidate: str, lower_dimensional: str
) -> dict[str, object] | None:
    architecture_seeds = {17, 29, 43}
    teacher = _paired_effect(
        runs,
        metric="capability.teacher_forced.accuracy",
        candidate_track="fixed_1024",
        candidate_topology=candidate,
        candidate_cells=131_072,
        baseline_track="fixed_1024",
        baseline_topology="uniform_no_self",
        baseline_cells=131_072,
    )
    free = _paired_effect(
        runs,
        metric="capability.free_running.accuracy",
        candidate_track="fixed_1024",
        candidate_topology=candidate,
        candidate_cells=131_072,
        baseline_track="fixed_1024",
        baseline_topology="uniform_no_self",
        baseline_cells=131_072,
    )
    if teacher is None or free is None or int(teacher["count"]) != 3:
        return None
    metric = "capability.teacher_forced.accuracy"
    candidate_teacher = _condition_median(
        runs, "fixed_1024", candidate, 131_072, metric, architecture_seeds
    )
    comparison_teacher = _condition_median(
        runs,
        "fixed_1024",
        lower_dimensional,
        131_072,
        metric,
        architecture_seeds,
    )
    candidate_responsive = _condition_median(
        runs,
        "fixed_1024",
        candidate,
        131_072,
        "dynamics.responsive_state_cell_fraction",
        architecture_seeds,
    )
    baseline_responsive = _condition_median(
        runs,
        "fixed_1024",
        "uniform_no_self",
        131_072,
        "dynamics.responsive_state_cell_fraction",
        architecture_seeds,
    )
    candidate_readout = _condition_median(
        runs,
        "fixed_1024",
        candidate,
        131_072,
        "learning.readout_source_cell_fraction",
        architecture_seeds,
    )
    baseline_readout = _condition_median(
        runs,
        "fixed_1024",
        "uniform_no_self",
        131_072,
        "learning.readout_source_cell_fraction",
        architecture_seeds,
    )
    firing = _condition_median(
        runs,
        "fixed_1024",
        candidate,
        131_072,
        "dynamics.mean_firing_hz_per_cell",
        architecture_seeds,
    )
    values = (
        candidate_teacher,
        comparison_teacher,
        candidate_responsive,
        baseline_responsive,
        candidate_readout,
        baseline_readout,
        firing,
    )
    if any(value is None for value in values):
        return None
    checks = {
        "teacher_median_delta": float(teacher["median_delta"]) >= 0.005,
        "free_median_delta": float(free["median_delta"]) >= 0.0,
        "teacher_seed_directions": int(teacher["positive_directions"]) >= 2,
        "not_below_single_or_lower_dimensional_control": candidate_teacher
        >= comparison_teacher,
        "responsive_density_guard": candidate_responsive
        >= 0.5 * baseline_responsive,
        "readout_density_guard": candidate_readout >= 0.5 * baseline_readout,
        "firing_range_guard": 0.5 <= firing <= 80.0,
    }
    return {
        "candidate": candidate,
        "control": lower_dimensional,
        "teacher_forced_effect_vs_uniform": teacher,
        "free_running_effect_vs_uniform": free,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _scale_slopes(runs: list[dict[str, Any]], track: str) -> dict[str, object] | None:
    sizes = [6_144, 24_576, 98_304, 131_072]
    seeds = [17, 29, 43, 59, 71]
    teacher_slopes: dict[int, float] = {}
    free_slopes: dict[int, float] = {}
    x = np.log2(np.asarray(sizes, dtype=np.float64))
    for seed in seeds:
        by_size = {
            int(run["cells"]): run
            for run in runs
            if run["data_track"] == track
            and run["topology"] == "uniform_no_self"
            and int(run["seed"]) == seed
        }
        if set(by_size) != set(sizes):
            continue
        teacher = [
            _nested_value(by_size[size], "capability.teacher_forced.accuracy")
            for size in sizes
        ]
        free = [
            _nested_value(by_size[size], "capability.free_running.accuracy")
            for size in sizes
        ]
        teacher_slopes[seed] = float(np.polyfit(x, teacher, 1)[0])
        free_slopes[seed] = float(np.polyfit(x, free, 1)[0])
    if len(teacher_slopes) != len(seeds):
        return None
    return {
        "units": "absolute accuracy per doubling of active cells",
        "teacher_forced": _paired_statistics(teacher_slopes),
        "free_running": _paired_statistics(free_slopes),
    }


def _summary(runs: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, object]:
    fixed = _scale_criterion(runs, protocol, "fixed_1024")
    scaled = _scale_criterion(runs, protocol, "sqrt_scaled")
    if scaled is None:
        interpretation = "incomplete"
    elif not scaled["passes"]:
        interpretation = "no_registered_scale_effect"
    elif fixed is not None and fixed["passes"]:
        interpretation = "robust_capacity_scaling"
    else:
        interpretation = "data_starved_capacity"

    topology_effects = []
    for topology in (
        "random",
        "single_scale_groups",
        "nested_groups",
        "mixed_membership",
        "ring_1d",
        "sheet_2d",
        "layered_sheet",
        "volume_3d",
        "hypercube_4d",
    ):
        for cells in (6_144, 24_576, 98_304, 131_072):
            for metric in (
                "capability.teacher_forced.accuracy",
                "capability.free_running.accuracy",
                "dynamics.responsive_state_cell_fraction",
                "learning.readout_source_cell_fraction",
            ):
                effect = _paired_effect(
                    runs,
                    metric=metric,
                    candidate_track="fixed_1024",
                    candidate_topology=topology,
                    candidate_cells=cells,
                    baseline_track="fixed_1024",
                    baseline_topology="uniform_no_self",
                    baseline_cells=cells,
                )
                if effect is not None and int(effect["count"]) == 3:
                    topology_effects.append(effect)

    dimensional_4d_vs_3d = []
    for cells in (6_144, 24_576, 98_304, 131_072):
        for metric in (
            "capability.teacher_forced.accuracy",
            "capability.free_running.accuracy",
            "dynamics.effective_state_rank_from_token_gram",
        ):
            effect = _paired_effect(
                runs,
                metric=metric,
                candidate_track="fixed_1024",
                candidate_topology="hypercube_4d",
                candidate_cells=cells,
                baseline_track="fixed_1024",
                baseline_topology="volume_3d",
                baseline_cells=cells,
            )
            if effect is not None and int(effect["count"]) == 3:
                dimensional_4d_vs_3d.append(effect)

    return {
        "completed_runs": len(runs),
        "expected_runs": int(protocol["execution"]["expected_runs"]),
        "complete": len(runs) == int(protocol["execution"]["expected_runs"]),
        "conditions": _condition_summary(runs, protocol),
        "scale": {
            "fixed_1024_criterion": fixed,
            "sqrt_scaled_primary_criterion": scaled,
            "fixed_1024_slopes": _scale_slopes(runs, "fixed_1024"),
            "sqrt_scaled_slopes": _scale_slopes(runs, "sqrt_scaled"),
            "registered_interpretation": interpretation,
        },
        "organization": {
            "multiscale_groups": _organization_criterion(
                runs, "nested_groups", "single_scale_groups"
            ),
            "layered_2d": _organization_criterion(
                runs, "layered_sheet", "sheet_2d"
            ),
            "paired_effects_vs_uniform": topology_effects,
            "abstract_4d_effects_vs_3d": dimensional_4d_vs_3d,
        },
    }


def _atomic_write_json(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _initial_report(protocol: dict[str, Any], protocol_hash: str) -> dict[str, object]:
    return {
        "protocol_version": protocol["version"],
        "protocol_sha256": protocol_hash,
        "selection_partition": "validation",
        "test_partition_defined": False,
        "test_partition_accessed": False,
        "adaptation_on_validation": False,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "humanbrain_device_policy": os.environ.get("HUMANBRAIN_DEVICE", "auto"),
        },
        "runs": [],
        "complete": False,
    }


def _load_report(
    output: Path, protocol: dict[str, Any], protocol_hash: str
) -> dict[str, Any]:
    if not output.is_file():
        return _initial_report(protocol, protocol_hash)
    report = json.loads(output.read_text(encoding="utf-8"))
    if report["protocol_sha256"] != protocol_hash:
        raise ValueError("existing output belongs to a different frozen protocol")
    if report["test_partition_defined"] or report["test_partition_accessed"]:
        raise ValueError("existing report records forbidden test access")
    return report


def _plan(protocol: dict[str, Any], protocol_hash: str) -> dict[str, object]:
    specifications = _specifications(protocol)
    stages = []
    for stage in protocol["execution"]["stages"]:
        stage_runs = [spec for spec in specifications if spec["stage"] == stage["name"]]
        stages.append(
            {
                "name": stage["name"],
                "runs": len(stage_runs),
                "data_track": stage["data_track"],
                "topologies": stage["topologies"],
                "seeds": stage["seeds"],
            }
        )
    return {
        "mode": "plan_only_no_validation_executed",
        "protocol": str(DEFAULT_PROTOCOL),
        "protocol_sha256": protocol_hash,
        "expected_runs": len(specifications),
        "stages": stages,
        "sizes": protocol["model"]["active_reservoir_cells"],
        "runtime_estimate": protocol["runtime_estimate"],
        "execution_command": (
            "python benchmarks/large_scale_capability.py --execute "
            f"--confirm-protocol-sha {protocol_hash}"
        ),
    }


def run(
    protocol: dict[str, Any],
    protocol_hash: str,
    output: Path,
    limit_runs: int | None,
) -> dict[str, Any]:
    report = _load_report(output, protocol, protocol_hash)
    completed = {str(item["key"]) for item in report["runs"]}
    specifications = _specifications(protocol)
    added = 0
    for specification in specifications:
        if specification["key"] in completed:
            continue
        if limit_runs is not None and added >= limit_runs:
            break
        result = _run_one(protocol, specification)
        report["runs"].append(result)
        report["summary"] = _summary(report["runs"], protocol)
        report["complete"] = len(report["runs"]) == len(specifications)
        _atomic_write_json(output, report)
        added += 1
        print(
            json.dumps(
                {
                    "key": result["key"],
                    "teacher_forced_accuracy": result["capability"][
                        "teacher_forced"
                    ]["accuracy"],
                    "free_running_accuracy": result["capability"]["free_running"][
                        "accuracy"
                    ],
                    "responsive_state_fraction": result["dynamics"][
                        "responsive_state_cell_fraction"
                    ],
                    "applied_local_updates": result["learning"][
                        "applied_local_updates"
                    ],
                }
            ),
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-protocol-sha")
    parser.add_argument("--limit-runs", type=int)
    args = parser.parse_args()
    if args.limit_runs is not None and args.limit_runs < 1:
        raise ValueError("--limit-runs must be at least one")

    protocol, protocol_hash = _load_protocol(args.protocol)
    if not args.execute:
        print(json.dumps(_plan(protocol, protocol_hash), indent=2, sort_keys=True))
        return
    if args.confirm_protocol_sha != protocol_hash:
        raise ValueError(
            "execution requires --confirm-protocol-sha with the exact plan hash"
        )
    policy = str(protocol["execution"]["device_policy"])
    configured_policy = os.environ.get("HUMANBRAIN_DEVICE", "auto").lower()
    if configured_policy != policy:
        raise ValueError(
            f"frozen device policy is {policy!r}, found {configured_policy!r}"
        )
    report = run(protocol, protocol_hash, args.output, args.limit_runs)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "protocol_sha256": protocol_hash,
                "runs": len(report["runs"]),
                "complete": report["complete"],
            }
        )
    )


if __name__ == "__main__":
    main()
