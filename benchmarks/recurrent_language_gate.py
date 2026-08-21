"""Held-out code gate for the recurrent adaptive human-cell circuit."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from humanbrain.recurrent import RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig


ROOT = Path(__file__).resolve().parents[1]


def _python_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*.py") if path.is_file())


def _bounded_corpus(paths: list[Path], maximum_characters: int) -> list[str]:
    if maximum_characters < len(paths):
        raise ValueError("training character budget is too small for the file count")
    per_file = maximum_characters // len(paths)
    return [path.read_text(encoding="utf-8")[:per_file] for path in paths]


def run(
    source_dir: Path,
    held_out: Path,
    max_predictions: int,
    max_training_characters: int,
) -> dict[str, object]:
    files = _python_files(source_dir)
    train_files = [path for path in files if path.resolve() != held_out.resolve()]
    if not train_files or not held_out.is_file():
        raise ValueError("need train files and one held-out Python file")
    train_texts = _bounded_corpus(train_files, max_training_characters)
    test_text = held_out.read_text(encoding="utf-8")
    model = RecurrentCellLanguageModel.grow(
        train_texts,
        RecurrentModelConfig(
            circuit=RecurrentCircuitConfig(
                reservoir_cells=96,
                input_fanout=16,
                recurrent_fanout=6,
                token_interval_ms=2.0,
                input_pulse_ms=1.2,
                input_current_pa=8_000.0,
                output_base_current_pa=700.0,
                output_race_ms=20.0,
            )
        ),
    )
    cell = model.evaluate(test_text, max_predictions=max_predictions)

    train_bytes = b"".join(text.encode("utf-8") for text in train_texts)
    most_common, _ = Counter(train_bytes).most_common(1)[0]
    targets = test_text.encode("utf-8")[1 : 1 + cell.predictions]
    unigram_correct = sum(value == most_common for value in targets)
    unigram_accuracy = unigram_correct / len(targets)
    return {
        "protocol": (
            "bounded prefixes from every package Python file except the named held-out file; "
            "byte tokens; teacher forcing; persistent adaptive-cell state"
        ),
        "source_dir": str(source_dir.resolve()),
        "held_out": str(held_out.resolve()),
        "train_files": [str(path.relative_to(source_dir)) for path in train_files],
        "training_characters": sum(len(text) for text in train_texts),
        "model": {
            "reservoir_cells": model.reservoir_cells,
            "learned_synapses": model.learned_synapses,
        },
        "cell": asdict(cell),
        "unigram": {
            "token": most_common,
            "correct": unigram_correct,
            "accuracy": unigram_accuracy,
        },
        "pass": cell.accuracy > unigram_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=ROOT / "src" / "humanbrain")
    parser.add_argument("--held-out", type=Path, default=ROOT / "src" / "humanbrain" / "cli.py")
    parser.add_argument("--max-predictions", type=int, default=1_000)
    parser.add_argument("--max-training-characters", type=int, default=24_000)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source_dir,
        args.held_out,
        args.max_predictions,
        args.max_training_characters,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
