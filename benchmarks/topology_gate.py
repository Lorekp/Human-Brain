from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from cell_scaling import _bounded_corpus, _circuit_config, _common_prefix
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig
from language_gate import partition_files

ROOT = Path(__file__).resolve().parents[1]


def _state_diagnostics(
    model: RecurrentCellLanguageModel, text: str, maximum_tokens: int
) -> dict[str, float | int]:
    token_ids = model._tokenizer.encode(text)[:maximum_tokens]
    model._circuit.reset_sequence()
    states = []
    reservoir_spikes = 0
    biological_time_ms = 0.0
    for token_id in token_ids:
        step = model._circuit.observe(token_id)
        reservoir_spikes += step.reservoir_spikes
        biological_time_ms += step.biological_time_ms
        states.append(model._circuit.eligibility)
    matrix = np.asarray(states, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values**2
    effective_rank = (
        float(variance.sum() ** 2 / np.square(variance).sum())
        if np.any(variance)
        else 0.0
    )
    active_columns = centered[:, centered.std(axis=0) > 1e-12]
    if active_columns.shape[1] > 1:
        correlation = np.corrcoef(active_columns, rowvar=False)
        upper = correlation[np.triu_indices_from(correlation, k=1)]
        mean_absolute_correlation = float(np.mean(np.abs(upper)))
    else:
        mean_absolute_correlation = 0.0
    return {
        "tokens": len(token_ids),
        "active_state_columns": int(active_columns.shape[1]),
        "effective_state_rank": effective_rank,
        "mean_absolute_cell_correlation": mean_absolute_correlation,
        "reservoir_spikes": reservoir_spikes,
        "mean_spikes_per_cell_per_token": reservoir_spikes
        / (model.active_reservoir_cells * len(token_ids)),
        "mean_firing_hz_per_cell": reservoir_spikes
        / model.active_reservoir_cells
        / (biological_time_ms / 1_000.0),
    }


def _run_one(
    *,
    topology: str,
    seed: int,
    cell_count: int,
    training_characters: int,
    protocol: dict[str, object],
    train_paths: list[Path],
    validation_paths: list[Path],
    source_dir: Path,
) -> dict[str, object]:
    evaluation_protocol = protocol["evaluation"]
    dynamics = protocol["model"]["dynamics"]
    train_texts = _bounded_corpus(train_paths, training_characters)
    recurrent_fanout = min(int(dynamics["recurrent_fanout"]), cell_count - 1)
    input_fanout = max(
        int(dynamics["minimum_input_fanout"]),
        round(cell_count / float(dynamics["input_fanout_divisor"])),
    )
    circuit = replace(
        _circuit_config(cell_count, cell_count, seed),
        input_fanout=input_fanout,
        recurrent_fanout=recurrent_fanout,
        token_interval_ms=float(dynamics["token_interval_ms"]),
        input_pulse_ms=float(dynamics["input_pulse_ms"]),
        input_current_pa=float(dynamics["input_current_pa"]),
        recurrent_current_pa=float(dynamics["recurrent_current_pa"]),
        eligibility_tau_ms=float(dynamics["eligibility_tau_ms"]),
        readout_ridge=training_characters / 12_000,
        recurrent_topology=topology,
    )
    started = perf_counter()
    model = RecurrentCellLanguageModel.grow(
        train_texts, RecurrentModelConfig(circuit=circuit)
    )
    training_seconds = perf_counter() - started
    diagnostics_text = validation_paths[0].read_text(encoding="utf-8")
    state_diagnostics = _state_diagnostics(
        model,
        diagnostics_text,
        int(evaluation_protocol["state_diagnostic_tokens"]),
    )

    teacher_folds: list[dict[str, object]] = []
    generation_folds: list[dict[str, object]] = []
    evaluation_started = perf_counter()
    for path in validation_paths:
        text = path.read_text(encoding="utf-8")
        result = model.evaluate(
            text,
            max_predictions=int(
                evaluation_protocol["teacher_forced_predictions_per_validation_file"]
            ),
        )
        teacher_folds.append(
            {"file": path.relative_to(source_dir).as_posix(), **asdict(result)}
        )

        prompt_characters = int(evaluation_protocol["generation_prompt_characters"])
        target_characters = int(evaluation_protocol["generation_target_characters"])
        prompt = text[:prompt_characters]
        target = text[
            prompt_characters : prompt_characters + target_characters
        ].encode("utf-8")
        if len(target) < target_characters:
            continue
        generated = model.complete(prompt, max_new_tokens=target_characters).encode(
            "utf-8"
        )
        compared = min(len(generated), len(target))
        correct = sum(generated[index] == target[index] for index in range(compared))
        generation_folds.append(
            {
                "file": path.relative_to(source_dir).as_posix(),
                "target_bytes": len(target),
                "generated_bytes": len(generated),
                "correct": correct,
                "accuracy": correct / len(target),
                "exact_prefix_bytes": _common_prefix(generated, target),
            }
        )
    evaluation_seconds = perf_counter() - evaluation_started
    teacher_predictions = sum(int(fold["predictions"]) for fold in teacher_folds)
    teacher_correct = sum(int(fold["correct"]) for fold in teacher_folds)
    generation_predictions = sum(int(fold["target_bytes"]) for fold in generation_folds)
    generation_correct = sum(int(fold["correct"]) for fold in generation_folds)
    return {
        "key": (
            f"cells={cell_count}:data={training_characters}:"
            f"topology={topology}:seed={seed}"
        ),
        "topology": topology,
        "seed": seed,
        "active_cells": cell_count,
        "training_characters": training_characters,
        "recurrent_edges": cell_count * circuit.recurrent_fanout,
        "input_fanout": circuit.input_fanout,
        "recurrent_fanout": circuit.recurrent_fanout,
        "learned_synapses": model.learned_synapses,
        "training_wall_seconds": training_seconds,
        "evaluation_wall_seconds": evaluation_seconds,
        "state_diagnostics": state_diagnostics,
        "teacher_forced": {
            "predictions": teacher_predictions,
            "correct": teacher_correct,
            "accuracy": teacher_correct / teacher_predictions,
            "folds": teacher_folds,
        },
        "free_running": {
            "predictions": generation_predictions,
            "correct": generation_correct,
            "accuracy": generation_correct / generation_predictions,
            "mean_exact_prefix_bytes": sum(
                int(fold["exact_prefix_bytes"]) for fold in generation_folds
            )
            / len(generation_folds),
            "folds": generation_folds,
        },
    }


def _summary(runs: list[dict[str, object]], protocol: dict[str, object]) -> dict[str, object]:
    seeds = protocol["model"]["random_seeds"]
    rows = []
    for case in protocol["model"]["architecture_cases"]:
        cell_count = int(case["active_reservoir_cells"])
        for topology in case["topologies"]:
            matches = [
                run
                for run in runs
                if run["topology"] == topology
                and run["active_cells"] == cell_count
            ]
            if len(matches) != len(seeds):
                continue
            rows.append(
                {
                    "active_cells": cell_count,
                    "training_characters": case["training_characters"],
                    "topology": topology,
                    "teacher_forced_median_accuracy": median(
                        float(run["teacher_forced"]["accuracy"])
                        for run in matches
                    ),
                    "free_running_median_accuracy": median(
                        float(run["free_running"]["accuracy"])
                        for run in matches
                    ),
                    "mean_exact_prefix_median_bytes": median(
                        float(run["free_running"]["mean_exact_prefix_bytes"])
                        for run in matches
                    ),
                    "effective_state_rank_median": median(
                        float(run["state_diagnostics"]["effective_state_rank"])
                        for run in matches
                    ),
                    "mean_absolute_cell_correlation_median": median(
                        float(
                            run["state_diagnostics"][
                                "mean_absolute_cell_correlation"
                            ]
                        )
                        for run in matches
                    ),
                }
            )
    for cell_count in {row["active_cells"] for row in rows}:
        baseline = next(
            (
                row
                for row in rows
                if row["active_cells"] == cell_count
                and row["topology"] == protocol["model"]["comparison_baseline"]
            ),
            None,
        )
        if baseline is not None:
            for row in rows:
                if row["active_cells"] != cell_count:
                    continue
                row["meets_primary_criterion"] = (
                    row["free_running_median_accuracy"]
                    > baseline["free_running_median_accuracy"]
                    and row["teacher_forced_median_accuracy"]
                    >= baseline["teacher_forced_median_accuracy"] - 0.01
                )
    return {"topologies": rows}


def run(
    protocol: dict[str, object], protocol_hash: str, output: Path, limit_runs: int | None
) -> dict[str, object]:
    source_dir = ROOT.parent / "src" / "deepsim"
    split_protocol = json.loads(
        (ROOT / "benchmarks" / "code_language_protocol_v4.json").read_text(
            encoding="utf-8"
        )
    )
    split = split_protocol["split"]
    partitions = partition_files(
        source_dir,
        bucket_count=int(split["bucket_count"]),
        train_buckets=split["train_buckets"],
        validation_buckets=split["validation_buckets"],
        test_buckets=split["test_buckets"],
    )
    if output.is_file():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["protocol_sha256"] != protocol_hash:
            raise ValueError("existing output was produced by a different protocol")
    else:
        report = {
            "protocol_version": protocol["version"],
            "protocol_sha256": protocol_hash,
            "selection_partition": "validation",
            "test_partition_accessed": False,
            "runs": [],
            "complete": False,
        }
    completed = {run["key"] for run in report["runs"]}
    specifications = [
        (
            int(case["active_reservoir_cells"]),
            int(case["training_characters"]),
            topology,
            seed,
        )
        for case in protocol["model"]["architecture_cases"]
        for topology in case["topologies"]
        for seed in protocol["model"]["random_seeds"]
    ]
    added = 0
    for cell_count, training_characters, topology, seed in specifications:
        key = (
            f"cells={cell_count}:data={training_characters}:"
            f"topology={topology}:seed={seed}"
        )
        if key in completed:
            continue
        if limit_runs is not None and added >= limit_runs:
            break
        result = _run_one(
            topology=topology,
            seed=seed,
            cell_count=cell_count,
            training_characters=training_characters,
            protocol=protocol,
            train_paths=partitions["train"],
            validation_paths=partitions["validation"],
            source_dir=source_dir,
        )
        report["runs"].append(result)
        report["summary"] = _summary(report["runs"], protocol)
        report["complete"] = len(report["runs"]) == len(specifications)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        added += 1
        print(
            json.dumps(
                {
                    "key": key,
                    "teacher_accuracy": result["teacher_forced"]["accuracy"],
                    "generation_accuracy": result["free_running"]["accuracy"],
                }
            ),
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks" / "topology_protocol_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "topology_validation_v2.json",
    )
    parser.add_argument("--limit-runs", type=int)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    report = run(
        json.loads(protocol_bytes),
        sha256(protocol_bytes).hexdigest(),
        args.output,
        args.limit_runs,
    )
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
