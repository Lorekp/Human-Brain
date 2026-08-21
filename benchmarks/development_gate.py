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


def _apply_condition(
    model: RecurrentCellLanguageModel,
    baseline: np.ndarray,
    condition: str,
    pruning_fraction: float,
    seed: int,
) -> None:
    weights = baseline.copy()
    if condition == "unpruned":
        model._circuit._readout_synapses_pa[:] = weights
        return
    rng = np.random.default_rng(seed)
    use_magnitude = condition.startswith("magnitude")
    use_homeostasis = condition.endswith("homeostatic")
    for row_index, row in enumerate(weights):
        nonzero = np.flatnonzero(row)
        prune_count = round(pruning_fraction * len(nonzero))
        if use_magnitude:
            ordering = np.lexsort((nonzero, np.abs(row[nonzero])))
            pruned = nonzero[ordering[:prune_count]]
        else:
            pruned = rng.choice(nonzero, size=prune_count, replace=False)
        original_strength = float(np.abs(row).sum())
        row[pruned] = 0.0
        if use_homeostasis:
            surviving_strength = float(np.abs(row).sum())
            if surviving_strength > 0.0:
                row *= original_strength / surviving_strength
        weights[row_index] = row
    model._circuit._readout_synapses_pa[:] = weights


def _evaluate(
    model: RecurrentCellLanguageModel,
    validation_paths: list[Path],
    source_dir: Path,
    protocol: dict[str, object],
) -> dict[str, object]:
    evaluation = protocol["evaluation"]
    teacher_folds: list[dict[str, object]] = []
    generation_folds: list[dict[str, object]] = []
    started = perf_counter()
    for path in validation_paths:
        text = path.read_text(encoding="utf-8")
        result = model.evaluate(
            text,
            max_predictions=int(
                evaluation["teacher_forced_predictions_per_validation_file"]
            ),
        )
        teacher_folds.append(
            {"file": path.relative_to(source_dir).as_posix(), **asdict(result)}
        )
        prompt_characters = int(evaluation["generation_prompt_characters"])
        target_characters = int(evaluation["generation_target_characters"])
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
    teacher_predictions = sum(int(fold["predictions"]) for fold in teacher_folds)
    teacher_correct = sum(int(fold["correct"]) for fold in teacher_folds)
    generation_predictions = sum(int(fold["target_bytes"]) for fold in generation_folds)
    generation_correct = sum(int(fold["correct"]) for fold in generation_folds)
    return {
        "wall_seconds": perf_counter() - started,
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
    rows = []
    seeds = protocol["model"]["random_seeds"]
    for condition in protocol["developmental_ablation"]["conditions"]:
        matches = [run for run in runs if run["condition"] == condition]
        if len(matches) != len(seeds):
            continue
        rows.append(
            {
                "condition": condition,
                "learned_synapses_median": median(
                    int(run["learned_synapses"]) for run in matches
                ),
                "teacher_forced_median_accuracy": median(
                    float(run["teacher_forced"]["accuracy"]) for run in matches
                ),
                "free_running_median_accuracy": median(
                    float(run["free_running"]["accuracy"]) for run in matches
                ),
                "mean_exact_prefix_median_bytes": median(
                    float(run["free_running"]["mean_exact_prefix_bytes"])
                    for run in matches
                ),
            }
        )
    return {"conditions": rows}


def run(
    protocol: dict[str, object], protocol_hash: str, output: Path, limit_seeds: int | None
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
    train_texts = _bounded_corpus(
        partitions["train"], int(model_protocol["training_characters"])
    )
    processed_seeds = 0
    for seed in model_protocol["random_seeds"]:
        if limit_seeds is not None and processed_seeds >= limit_seeds:
            break
        conditions = protocol["developmental_ablation"]["conditions"]
        if all(f"condition={condition}:seed={seed}" in completed for condition in conditions):
            continue
        cell_count = int(model_protocol["active_reservoir_cells"])
        circuit = replace(
            _circuit_config(cell_count, cell_count, int(seed)),
            recurrent_topology=model_protocol["recurrent_topology"],
        )
        training_started = perf_counter()
        model = RecurrentCellLanguageModel.grow(
            train_texts, RecurrentModelConfig(circuit=circuit)
        )
        training_seconds = perf_counter() - training_started
        baseline = model._circuit._readout_synapses_pa.copy()
        for condition in conditions:
            key = f"condition={condition}:seed={seed}"
            if key in completed:
                continue
            _apply_condition(
                model,
                baseline,
                condition,
                float(protocol["developmental_ablation"]["pruning_fraction"]),
                int(seed) + 10_000,
            )
            result = _evaluate(
                model, partitions["validation"], source_dir, protocol
            )
            result.update(
                {
                    "key": key,
                    "condition": condition,
                    "seed": seed,
                    "training_wall_seconds": training_seconds,
                    "learned_synapses": int(
                        np.count_nonzero(model._circuit._readout_synapses_pa)
                    ),
                }
            )
            report["runs"].append(result)
            report["summary"] = _summary(report["runs"], protocol)
            report["complete"] = len(report["runs"]) == len(conditions) * len(
                model_protocol["random_seeds"]
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
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
        processed_seeds += 1
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks" / "development_protocol_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "development_validation_v1.json",
    )
    parser.add_argument("--limit-seeds", type=int)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    report = run(
        json.loads(protocol_bytes),
        sha256(protocol_bytes).hexdigest(),
        args.output,
        args.limit_seeds,
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
