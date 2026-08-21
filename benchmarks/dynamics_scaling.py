from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from cell_scaling import _bounded_corpus, _circuit_config
from development_gate import _evaluate
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig
from language_gate import partition_files
from topology_gate import _state_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def _regime_config(
    active_cells: int,
    training_characters: int,
    seed: int,
    regime: str,
    topology: str,
):
    if regime == "fixed_degree":
        recurrent_fanout = min(6, active_cells - 1)
        recurrent_current_pa = 650.0
    elif regime == "density_current_normalized":
        recurrent_fanout = max(2, round(active_cells / 16))
        recurrent_current_pa = 650.0 * 6 / recurrent_fanout
    elif regime == "calibrated_sparse":
        recurrent_fanout = min(6, active_cells - 1)
        recurrent_current_pa = 325.0
    else:
        raise ValueError(f"unknown dynamical regime: {regime}")
    configuration = replace(
        _circuit_config(active_cells, active_cells, seed),
        recurrent_fanout=recurrent_fanout,
        recurrent_current_pa=recurrent_current_pa,
        readout_ridge=training_characters / 12_000,
        recurrent_topology=topology,
    )
    if regime == "calibrated_sparse":
        configuration = replace(
            configuration,
            input_fanout=max(4, round(active_cells / 12)),
            token_interval_ms=5.0,
            input_pulse_ms=1.5,
            input_current_pa=4_000.0,
            eligibility_tau_ms=40.0,
        )
    return configuration


def _summary(runs: list[dict[str, object]], protocol: dict[str, object]) -> dict[str, object]:
    rows = []
    seeds = protocol["model"]["random_seeds"]
    for regime in protocol["model"]["dynamical_regimes"]:
        for pair in protocol["model"]["cell_data_pairs"]:
            active_cells = int(pair["active_cells"])
            matches = [
                run
                for run in runs
                if run["regime"] == regime and run["active_cells"] == active_cells
            ]
            if len(matches) != len(seeds):
                continue
            rows.append(
                {
                    "regime": regime,
                    "active_cells": active_cells,
                    "training_characters": pair["training_characters"],
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
    return {"curves": rows}


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
    model_protocol = protocol["model"]
    specifications = [
        (regime, pair, seed)
        for regime in model_protocol["dynamical_regimes"]
        for pair in model_protocol["cell_data_pairs"]
        for seed in model_protocol["random_seeds"]
    ]
    added = 0
    for regime, pair, seed in specifications:
        active_cells = int(pair["active_cells"])
        training_characters = int(pair["training_characters"])
        key = (
            f"regime={regime}:cells={active_cells}:"
            f"data={training_characters}:seed={seed}"
        )
        if key in completed:
            continue
        if limit_runs is not None and added >= limit_runs:
            break
        train_texts = _bounded_corpus(partitions["train"], training_characters)
        circuit = _regime_config(
            active_cells,
            training_characters,
            int(seed),
            regime,
            model_protocol["topology"],
        )
        started = perf_counter()
        model = RecurrentCellLanguageModel.grow(
            train_texts, RecurrentModelConfig(circuit=circuit)
        )
        training_seconds = perf_counter() - started
        diagnostics_text = partitions["validation"][0].read_text(encoding="utf-8")
        diagnostics = _state_diagnostics(
            model,
            diagnostics_text,
            int(protocol["evaluation"]["state_diagnostic_tokens"]),
        )
        evaluation = _evaluate(
            model, partitions["validation"], source_dir, protocol
        )
        regularized = model._circuit._normal_matrix.copy()
        diagonal = np.diag_indices_from(regularized)
        regularized[diagonal] += circuit.readout_ridge
        regularized[-1, -1] -= circuit.readout_ridge * 0.99
        result = {
            "key": key,
            "regime": regime,
            "active_cells": active_cells,
            "training_characters_budget": training_characters,
            "training_characters_used": sum(len(text) for text in train_texts),
            "seed": seed,
            "input_fanout": circuit.input_fanout,
            "recurrent_fanout": circuit.recurrent_fanout,
            "recurrent_current_pa": circuit.recurrent_current_pa,
            "readout_ridge": circuit.readout_ridge,
            "learned_synapses": model.learned_synapses,
            "readout_condition_number": float(np.linalg.cond(regularized)),
            "training_wall_seconds": training_seconds,
            "state_diagnostics": diagnostics,
            **evaluation,
        }
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
                    "state_rank": diagnostics["effective_state_rank"],
                    "firing_hz": diagnostics["mean_firing_hz_per_cell"],
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
        default=ROOT / "benchmarks" / "dynamics_scaling_protocol_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "dynamics_scaling_validation_v2.json",
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
