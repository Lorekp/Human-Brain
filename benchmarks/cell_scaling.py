from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter

from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig
from language_gate import partition_files

ROOT = Path(__file__).resolve().parents[1]


def _bounded_corpus(paths: list[Path], maximum_characters: int) -> list[str]:
    if maximum_characters < len(paths):
        raise ValueError("training character budget is too small for the file count")
    per_file = maximum_characters // len(paths)
    return [path.read_text(encoding="utf-8")[:per_file] for path in paths]


def _common_prefix(left: bytes, right: bytes) -> int:
    return next(
        (
            index
            for index, values in enumerate(zip(left, right, strict=False))
            if values[0] != values[1]
        ),
        min(len(left), len(right)),
    )


def _circuit_config(
    allocated_cells: int, active_cells: int, random_seed: int
) -> RecurrentCircuitConfig:
    return RecurrentCircuitConfig(
        reservoir_cells=allocated_cells,
        active_reservoir_cells=(
            active_cells if active_cells != allocated_cells else None
        ),
        input_fanout=max(4, round(active_cells / 6)),
        recurrent_fanout=max(2, round(active_cells / 16)),
        token_interval_ms=2.0,
        input_pulse_ms=1.2,
        input_current_pa=8_000.0,
        output_base_current_pa=700.0,
        output_race_ms=20.0,
        random_seed=random_seed,
    )


def _run_one(
    *,
    train_paths: list[Path],
    validation_paths: list[Path],
    source_dir: Path,
    allocated_cells: int,
    active_cells: int,
    training_characters: int,
    random_seed: int,
    predictions_per_file: int,
    prompt_characters: int,
    target_characters: int,
    control: str,
) -> dict[str, object]:
    train_texts = _bounded_corpus(train_paths, training_characters)
    started = perf_counter()
    model = RecurrentCellLanguageModel.grow(
        train_texts,
        RecurrentModelConfig(
            circuit=_circuit_config(allocated_cells, active_cells, random_seed)
        ),
    )
    training_wall_seconds = perf_counter() - started

    teacher_folds: list[dict[str, object]] = []
    generation_folds: list[dict[str, object]] = []
    evaluation_started = perf_counter()
    for path in validation_paths:
        text = path.read_text(encoding="utf-8")
        evaluation = model.evaluate(text, max_predictions=predictions_per_file)
        teacher_folds.append(
            {
                "file": path.relative_to(source_dir).as_posix(),
                **asdict(evaluation),
            }
        )

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
    evaluation_wall_seconds = perf_counter() - evaluation_started

    teacher_predictions = sum(int(fold["predictions"]) for fold in teacher_folds)
    teacher_correct = sum(int(fold["correct"]) for fold in teacher_folds)
    generated_predictions = sum(int(fold["target_bytes"]) for fold in generation_folds)
    generated_correct = sum(int(fold["correct"]) for fold in generation_folds)
    return {
        "key": (
            f"{control}:allocated={allocated_cells}:active={active_cells}:"
            f"data={training_characters}:seed={random_seed}"
        ),
        "control": control,
        "allocated_cells": allocated_cells,
        "active_cells": active_cells,
        "training_characters_budget": training_characters,
        "training_characters_used": sum(len(text) for text in train_texts),
        "random_seed": random_seed,
        "input_fanout": max(4, round(active_cells / 6)),
        "recurrent_fanout": max(2, round(active_cells / 16)),
        "learned_synapses": model.learned_synapses,
        "training_wall_seconds": training_wall_seconds,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "teacher_forced": {
            "predictions": teacher_predictions,
            "correct": teacher_correct,
            "accuracy": teacher_correct / teacher_predictions,
            "folds": teacher_folds,
        },
        "free_running": {
            "predictions": generated_predictions,
            "correct": generated_correct,
            "accuracy": generated_correct / generated_predictions,
            "mean_exact_prefix_bytes": sum(
                int(fold["exact_prefix_bytes"]) for fold in generation_folds
            )
            / len(generation_folds),
            "folds": generation_folds,
        },
    }


def _summarize(runs: list[dict[str, object]], protocol: dict[str, object]) -> dict[str, object]:
    model_protocol = protocol["model"]
    curves: list[dict[str, object]] = []
    for data_budget in model_protocol["training_characters"]:
        points = []
        for active_cells in model_protocol["active_reservoir_cells"]:
            matches = [
                run
                for run in runs
                if run["control"] == "fully_active"
                and run["active_cells"] == active_cells
                and run["training_characters_budget"] == data_budget
            ]
            if len(matches) != len(model_protocol["random_seeds"]):
                continue
            points.append(
                {
                    "active_cells": active_cells,
                    "teacher_forced_median_accuracy": median(
                        float(run["teacher_forced"]["accuracy"]) for run in matches
                    ),
                    "free_running_median_accuracy": median(
                        float(run["free_running"]["accuracy"]) for run in matches
                    ),
                }
            )
        adjacent_improvements = sum(
            points[index]["teacher_forced_median_accuracy"]
            > points[index - 1]["teacher_forced_median_accuracy"]
            for index in range(1, len(points))
        )
        supported = (
            len(points) == len(model_protocol["active_reservoir_cells"])
            and points[-1]["teacher_forced_median_accuracy"]
            > points[0]["teacher_forced_median_accuracy"]
            and adjacent_improvements >= 2
        )
        curves.append(
            {
                "training_characters": data_budget,
                "points": points,
                "positive_adjacent_changes": adjacent_improvements,
                "exploratory_scaling_supported": supported,
            }
        )
    return {"cell_scaling_curves": curves}


def run(
    protocol: dict[str, object], protocol_sha256: str, output: Path, limit_runs: int | None
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
        if report["protocol_sha256"] != protocol_sha256:
            raise ValueError("existing output was produced by a different protocol")
    else:
        report = {
            "protocol_version": protocol["version"],
            "protocol_sha256": protocol_sha256,
            "selection_partition": "validation",
            "test_partition_accessed": False,
            "train_files": [
                path.relative_to(source_dir).as_posix() for path in partitions["train"]
            ],
            "validation_files": [
                path.relative_to(source_dir).as_posix()
                for path in partitions["validation"]
            ],
            "runs": [],
            "complete": False,
        }
    completed = {run["key"] for run in report["runs"]}
    evaluation = protocol["evaluation"]
    specifications: list[dict[str, object]] = []
    model_protocol = protocol["model"]
    for training_characters in model_protocol["training_characters"]:
        for active_cells in model_protocol["active_reservoir_cells"]:
            for random_seed in model_protocol["random_seeds"]:
                specifications.append(
                    {
                        "allocated_cells": active_cells,
                        "active_cells": active_cells,
                        "training_characters": training_characters,
                        "random_seed": random_seed,
                        "control": "fully_active",
                    }
                )
    dormant = protocol["dormant_cell_control"]
    for allocated_cells in dormant["allocated_reservoir_cells"]:
        for random_seed in dormant["random_seeds"]:
            specifications.append(
                {
                    "allocated_cells": allocated_cells,
                    "active_cells": dormant["active_reservoir_cells"],
                    "training_characters": dormant["training_characters"],
                    "random_seed": random_seed,
                    "control": "dormant_cells",
                }
            )

    pending = 0
    for specification in specifications:
        key = (
            f"{specification['control']}:allocated={specification['allocated_cells']}:"
            f"active={specification['active_cells']}:"
            f"data={specification['training_characters']}:"
            f"seed={specification['random_seed']}"
        )
        if key in completed:
            continue
        if limit_runs is not None and pending >= limit_runs:
            break
        result = _run_one(
            train_paths=partitions["train"],
            validation_paths=partitions["validation"],
            source_dir=source_dir,
            predictions_per_file=int(
                evaluation["teacher_forced_predictions_per_validation_file"]
            ),
            prompt_characters=int(evaluation["generation_prompt_characters"]),
            target_characters=int(evaluation["generation_target_characters"]),
            **specification,
        )
        report["runs"].append(result)
        pending += 1
        report["summary"] = _summarize(report["runs"], protocol)
        report["complete"] = len(report["runs"]) == len(specifications)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(
            json.dumps(
                {
                    "key": result["key"],
                    "teacher_accuracy": result["teacher_forced"]["accuracy"],
                    "generation_accuracy": result["free_running"]["accuracy"],
                    "training_seconds": result["training_wall_seconds"],
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
        default=ROOT / "benchmarks" / "cell_scaling_protocol_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "cell_scaling_validation_v1.json",
    )
    parser.add_argument("--limit-runs", type=int)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    report = run(protocol, sha256(protocol_bytes).hexdigest(), args.output, args.limit_runs)
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
