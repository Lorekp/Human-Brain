"""Train-only calibrated ridge/local readout scaling comparison.

The protocol deliberately has no test partition. A named training file is used
only to choose the three-factor learning rate. All choices are then frozen
before either learner opens the named validation file.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import tracemalloc

import numpy as np

from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "humanbrain"
FIT_FILES = (PACKAGE / "cells.py", PACKAGE / "topology.py", PACKAGE / "tokenizer.py")
CALIBRATION_FILE = PACKAGE / "native.py"
VALIDATION_FILE = PACKAGE / "recurrent_language.py"


def _bounded_text(path: Path, maximum_characters: int) -> str:
    text = path.read_text(encoding="utf-8")[:maximum_characters]
    if len(text) < 2:
        raise ValueError(f"not enough text in {path}")
    return text


def _content_record(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _config(
    cells: int,
    rule: str,
    learning_rate_pa: float,
    epochs: int,
) -> RecurrentModelConfig:
    return RecurrentModelConfig(
        growth_epochs=epochs,
        circuit=RecurrentCircuitConfig(
            reservoir_cells=cells,
            input_fanout=min(32, cells),
            recurrent_fanout=6,
            token_interval_ms=2.0,
            input_pulse_ms=1.2,
            input_current_pa=8_000.0,
            recurrent_current_pa=325.0,
            eligibility_tau_ms=40.0,
            output_base_current_pa=700.0,
            output_race_ms=20.0,
            recurrent_topology="uniform_no_self",
            readout_learning_rule=rule,
            local_learning_rate_pa=learning_rate_pa,
        ),
    )


def _persistent_array_mebibytes(model: RecurrentCellLanguageModel) -> float:
    circuit = model._circuit
    arrays: list[np.ndarray] = []
    arrays.extend(value for value in vars(circuit).values() if isinstance(value, np.ndarray))
    arrays.extend(
        value for value in vars(circuit._reservoir).values() if isinstance(value, np.ndarray)
    )
    arrays.extend(
        value for value in vars(circuit._reservoir.state).values() if isinstance(value, np.ndarray)
    )
    unique = {id(array): array for array in arrays}
    return sum(array.nbytes for array in unique.values()) / (1024 * 1024)


def _common_prefix(left: bytes, right: bytes) -> int:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return min(len(left), len(right))


def _calibrate_local_rate(
    fit_texts: list[str],
    calibration_text: str,
    candidates: tuple[float, ...],
    cells: int,
    epochs: int,
    predictions: int,
) -> tuple[float, list[dict[str, float]]]:
    trials: list[dict[str, float]] = []
    for learning_rate in candidates:
        started = perf_counter()
        model = RecurrentCellLanguageModel.grow(
            fit_texts,
            _config(cells, "three_factor", learning_rate, epochs),
        )
        train_seconds = perf_counter() - started
        if not model._circuit.learning_frozen:
            raise RuntimeError("calibration model reached evaluation before freeze")
        result = model.evaluate(calibration_text, max_predictions=predictions)
        trials.append(
            {
                "learning_rate_pa": learning_rate,
                "accuracy": result.accuracy,
                "train_seconds": train_seconds,
                "evaluation_seconds": result.wall_seconds,
            }
        )
        del model
        gc.collect()
    winner = max(trials, key=lambda item: (item["accuracy"], -item["learning_rate_pa"]))
    return winner["learning_rate_pa"], trials


def _run_one(
    training_texts: list[str],
    validation_text: str,
    cells: int,
    rule: str,
    learning_rate_pa: float,
    epochs: int,
    predictions: int,
    generated_tokens: int,
    prompt_characters: int,
) -> dict[str, object]:
    gc.collect()
    tracemalloc.start()
    started = perf_counter()
    model = RecurrentCellLanguageModel.grow(
        training_texts,
        _config(cells, rule, learning_rate_pa, epochs),
    )
    train_seconds = perf_counter() - started
    _, train_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not model._circuit.learning_frozen:
        raise RuntimeError("validation model reached evaluation before freeze")

    teacher_forced = model.evaluate(validation_text, max_predictions=predictions)
    prompt = validation_text[:prompt_characters]
    target = validation_text[
        prompt_characters : prompt_characters + generated_tokens
    ].encode("utf-8")
    generation_started = perf_counter()
    generated_text = model.complete(prompt, max_new_tokens=generated_tokens)
    generation_seconds = perf_counter() - generation_started
    generated = generated_text.encode("utf-8")
    compared = min(len(generated), len(target))
    free_correct = sum(generated[index] == target[index] for index in range(compared))
    circuit = model._circuit
    result = {
        "rule": rule,
        "reservoir_cells": cells,
        "learning_rate_pa": learning_rate_pa if rule == "three_factor" else None,
        "growth_examples": circuit._growth_examples,
        "local_weight_updates": circuit.local_update_count,
        "learned_readout_synapses": model.learned_synapses,
        "ridge_normal_matrix_allocated": circuit._normal_matrix is not None,
        "wall_seconds": {
            "train": train_seconds,
            "teacher_forced_validation": teacher_forced.wall_seconds,
            "free_running_validation": generation_seconds,
        },
        "memory_mib": {
            "training_peak_traced": train_peak / (1024 * 1024),
            "persistent_arrays": _persistent_array_mebibytes(model),
        },
        "teacher_forced": {
            "predictions": teacher_forced.predictions,
            "correct": teacher_forced.correct,
            "accuracy": teacher_forced.accuracy,
            "tokens_per_second": teacher_forced.tokens_per_second,
        },
        "free_running": {
            "target_bytes": len(target),
            "generated_bytes": len(generated),
            "correct": free_correct,
            "accuracy": free_correct / len(target),
            "exact_prefix_bytes": _common_prefix(generated, target),
            "generated_preview": generated_text[:120],
        },
    }
    del model
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="96,384,1536,6144")
    parser.add_argument("--ridge-max-size", type=int, default=1536)
    parser.add_argument("--fit-characters", type=int, default=768)
    parser.add_argument("--calibration-characters", type=int, default=256)
    parser.add_argument("--validation-characters", type=int, default=512)
    parser.add_argument("--predictions", type=int, default=128)
    parser.add_argument("--generated-tokens", type=int, default=64)
    parser.add_argument("--prompt-characters", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rates", default="16,32,64,128")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(","))
    rates = tuple(float(value) for value in args.learning_rates.split(","))
    if not sizes or any(size < 8 for size in sizes):
        raise ValueError("sizes must contain cell counts of eight or more")
    if not rates or any(rate <= 0.0 for rate in rates):
        raise ValueError("learning rates must be positive")

    per_fit_file = max(2, args.fit_characters // len(FIT_FILES))
    fit_texts = [_bounded_text(path, per_fit_file) for path in FIT_FILES]
    calibration_text = _bounded_text(CALIBRATION_FILE, args.calibration_characters)
    calibrated_rate, calibration_trials = _calibrate_local_rate(
        fit_texts,
        calibration_text,
        rates,
        min(sizes),
        args.epochs,
        args.predictions,
    )
    # The rate is now frozen. Only from this point onward is validation opened.
    validation_text = _bounded_text(VALIDATION_FILE, args.validation_characters)
    training_texts = [*fit_texts, calibration_text]

    measurements: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for cells in sizes:
        measurements.append(
            _run_one(
                training_texts,
                validation_text,
                cells,
                "three_factor",
                calibrated_rate,
                args.epochs,
                args.predictions,
                args.generated_tokens,
                args.prompt_characters,
            )
        )
        if cells <= args.ridge_max_size:
            measurements.append(
                _run_one(
                    training_texts,
                    validation_text,
                    cells,
                    "ridge",
                    calibrated_rate,
                    args.epochs,
                    args.predictions,
                    args.generated_tokens,
                    args.prompt_characters,
                )
            )
        else:
            normal_bytes = (cells + 1) ** 2 * np.dtype(np.float64).itemsize
            skipped.append(
                {
                    "rule": "ridge",
                    "reservoir_cells": cells,
                    "reason": "above explicit ridge-max-size safety limit",
                    "normal_matrix_mib": normal_bytes / (1024 * 1024),
                }
            )

    report = {
        "protocol": {
            "name": "local-readout-scaling-v1",
            "split": (
                "fit and learning-rate calibration use named training files only; "
                "the learning rate is frozen before named validation is opened; "
                "there is no test partition and no test access"
            ),
            "fit_files": [_content_record(path) for path in FIT_FILES],
            "calibration_file": _content_record(CALIBRATION_FILE),
            "validation_file": _content_record(VALIDATION_FILE),
            "sizes": sizes,
            "ridge_max_size": args.ridge_max_size,
            "epochs": args.epochs,
            "fit_characters": sum(len(text) for text in fit_texts),
            "calibration_characters": len(calibration_text),
            "validation_characters": len(validation_text),
            "predictions": args.predictions,
            "generated_tokens": args.generated_tokens,
            "prompt_characters": args.prompt_characters,
            "peak_memory": "Python and NumPy allocations traced by tracemalloc",
        },
        "calibration": {
            "reservoir_cells": min(sizes),
            "candidates": calibration_trials,
            "selected_learning_rate_pa": calibrated_rate,
        },
        "measurements": measurements,
        "skipped": skipped,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
