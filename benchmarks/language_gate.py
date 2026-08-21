"""Pre-registered, file-disjoint byte prediction gate for the human-cell model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanbrain import CellLanguageModel, ModelConfig  # noqa: E402


def _python_files(path: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in path.rglob("*.py")
        if candidate.name != "__init__.py" and "__pycache__" not in candidate.parts
    )


def _file_bucket(path: Path, source_dir: Path, bucket_count: int) -> int:
    relative = path.relative_to(source_dir).as_posix()
    digest = sha256(relative.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % bucket_count


def partition_files(
    source_dir: Path,
    *,
    bucket_count: int,
    train_buckets: Iterable[int],
    validation_buckets: Iterable[int],
    test_buckets: Iterable[int],
) -> dict[str, list[Path]]:
    bucket_sets = {
        "train": set(train_buckets),
        "validation": set(validation_buckets),
        "test": set(test_buckets),
    }
    flattened = [bucket for buckets in bucket_sets.values() for bucket in buckets]
    if len(flattened) != len(set(flattened)):
        raise ValueError("protocol buckets must not overlap")
    if set(flattened) != set(range(bucket_count)):
        raise ValueError("protocol buckets must cover every bucket exactly once")
    partitions = {name: [] for name in bucket_sets}
    for path in _python_files(source_dir):
        bucket = _file_bucket(path, source_dir, bucket_count)
        split = next(name for name, buckets in bucket_sets.items() if bucket in buckets)
        partitions[split].append(path)
    if any(not paths for paths in partitions.values()):
        raise ValueError("protocol produced an empty data split")
    return partitions


def _bounded_text(path: Path, max_characters: int) -> str:
    if max_characters < 2:
        raise ValueError("maximum characters per file must be at least two")
    return path.read_text(encoding="utf-8")[:max_characters]


def _counter_winner(counts: Counter[int]) -> int:
    return min(counts, key=lambda token: (-counts[token], token))


def _train_baselines(
    train_texts: Iterable[str], max_order: int
) -> tuple[Counter[int], dict[bytes, Counter[int]]]:
    unigram: Counter[int] = Counter()
    contexts: dict[bytes, Counter[int]] = defaultdict(Counter)
    for text in train_texts:
        values = text.encode("utf-8")
        unigram.update(values)
        for target_index in range(1, len(values)):
            for order in range(1, min(max_order, target_index) + 1):
                contexts[values[target_index - order : target_index]][values[target_index]] += 1
    if not unigram:
        raise ValueError("training corpus contains no bytes")
    return unigram, contexts


def _baseline_counts(
    baselines: tuple[Counter[int], dict[bytes, Counter[int]]],
    test_text: str,
    predictions: int,
    max_order: int,
) -> dict[str, dict[str, float | int]]:
    unigram, contexts = baselines
    unigram_winner = _counter_winner(unigram)
    values = test_text.encode("utf-8")
    targets = values[1 : predictions + 1]
    unigram_correct = sum(target == unigram_winner for target in targets)
    bigram_correct = 0
    variable_order_correct = 0
    witten_bell_correct = 0
    for target_index, target in enumerate(targets, start=1):
        bigram_counts = contexts.get(values[target_index - 1 : target_index])
        bigram_prediction = (
            _counter_winner(bigram_counts) if bigram_counts else unigram_winner
        )
        bigram_correct += target == bigram_prediction
        variable_order_prediction = unigram_winner
        for order in range(min(max_order, target_index), 0, -1):
            counts = contexts.get(values[target_index - order : target_index])
            if counts:
                variable_order_prediction = _counter_winner(counts)
                break
        variable_order_correct += target == variable_order_prediction
        # Chen & Goodman, eq. 8: recursively interpolate each context's MLE
        # distribution with the lower-order distribution using its number of
        # distinct continuations as the escape mass.
        probabilities = [1.0 / 256.0] * 256
        for order in range(1, min(max_order, target_index) + 1):
            counts = contexts.get(values[target_index - order : target_index])
            if not counts:
                continue
            total = sum(counts.values())
            distinct = len(counts)
            denominator = total + distinct
            probabilities = [
                (counts.get(token, 0) + distinct * probabilities[token]) / denominator
                for token in range(256)
            ]
        witten_bell_prediction = min(
            range(256), key=lambda token: (-probabilities[token], token)
        )
        witten_bell_correct += target == witten_bell_prediction
    return {
        "unigram": {
            "token": unigram_winner,
            "correct": unigram_correct,
            "accuracy": unigram_correct / predictions,
        },
        "bigram": {
            "correct": bigram_correct,
            "accuracy": bigram_correct / predictions,
        },
        "variable_order": {
            "max_order": max_order,
            "correct": variable_order_correct,
            "accuracy": variable_order_correct / predictions,
        },
        "witten_bell": {
            "max_order": max_order,
            "correct": witten_bell_correct,
            "accuracy": witten_bell_correct / predictions,
        },
    }


def _aggregate(folds: list[dict[str, object]], name: str) -> dict[str, float | int]:
    predictions = sum(int(fold["cell"]["predictions"]) for fold in folds)  # type: ignore[index]
    correct = sum(int(fold[name]["correct"]) for fold in folds)  # type: ignore[index]
    accuracies = [float(fold[name]["accuracy"]) for fold in folds]  # type: ignore[index]
    return {
        "predictions": predictions,
        "correct": correct,
        "micro_accuracy": correct / predictions,
        "macro_accuracy": sum(accuracies) / len(accuracies),
    }


def run(
    source_dir: Path,
    protocol: dict[str, object],
    split: str,
) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    split_config = protocol["split"]
    partitions = partition_files(
        source_dir,
        bucket_count=int(split_config["bucket_count"]),  # type: ignore[index]
        train_buckets=split_config["train_buckets"],  # type: ignore[index]
        validation_buckets=split_config["validation_buckets"],  # type: ignore[index]
        test_buckets=split_config["test_buckets"],  # type: ignore[index]
    )
    max_training_characters = int(protocol["max_training_characters_per_file"])
    max_predictions = int(protocol["max_predictions_per_file"])
    train_texts = [
        _bounded_text(path, max_training_characters) for path in partitions["train"]
    ]
    model_config = protocol["model"]
    max_order = int(model_config["max_order"])  # type: ignore[index]
    model = CellLanguageModel.grow(
        train_texts,
        ModelConfig(
            max_order=max_order,
            max_phrases=int(model_config["max_phrases"]),  # type: ignore[index]
            random_seed=int(model_config["random_seed"]),  # type: ignore[index]
        ),
    )
    baselines = _train_baselines(train_texts, max_order)

    folds: list[dict[str, object]] = []
    for path in partitions[split]:
        text = _bounded_text(path, max_predictions + 1)
        cell = asdict(model.evaluate(text, max_predictions=max_predictions))
        baseline_scores = _baseline_counts(
            baselines, text, int(cell["predictions"]), max_order
        )
        fold = {
            "file": path.relative_to(source_dir).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "cell": cell,
            **baseline_scores,
        }
        fold["beats_unigram"] = cell["accuracy"] > baseline_scores["unigram"]["accuracy"]
        fold["beats_bigram"] = cell["accuracy"] > baseline_scores["bigram"]["accuracy"]
        folds.append(fold)

    aggregate = {
        "cell": _aggregate(folds, "cell"),
        "unigram": _aggregate(folds, "unigram"),
        "bigram": _aggregate(folds, "bigram"),
        "variable_order": _aggregate(folds, "variable_order"),
        "witten_bell": _aggregate(folds, "witten_bell"),
    }
    passed = (
        aggregate["cell"]["micro_accuracy"] > aggregate["bigram"]["micro_accuracy"]
        and aggregate["cell"]["micro_accuracy"]
        >= 0.95 * aggregate["variable_order"]["micro_accuracy"]
        and aggregate["cell"]["micro_accuracy"]
        >= 0.95 * aggregate["witten_bell"]["micro_accuracy"]
        and all(bool(fold["beats_unigram"]) for fold in folds)
    )
    return {
        "protocol_version": protocol["version"],
        "protocol": (
            "path-hash split; bounded train-only file prefixes; frozen model; "
            "file-disjoint teacher-forced byte prediction; no test-time growth"
        ),
        "split": split,
        "source_dir": str(source_dir.resolve()),
        "model_training_digest": model.training_digest,
        "training": {
            "files": [path.relative_to(source_dir).as_posix() for path in partitions["train"]],
            "characters": sum(len(text) for text in train_texts),
            "vocabulary_size": model.vocabulary_size,
            "grown_branches": model.grown_branches,
        },
        "partition_counts": {name: len(paths) for name, paths in partitions.items()},
        "evaluated_files": folds,
        "aggregate": aggregate,
        "gate": {
            "criteria": (
                "cell micro accuracy > bigram, at least 95% of same-order conventional "
                "backoff and Witten-Bell accuracy, and every fold cell accuracy > unigram"
            ),
            "pass": passed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT.parent / "src" / "deepsim")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks" / "code_language_protocol_v4.json",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--confirm-frozen", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if args.split == "test" and protocol.get("test_locked"):
        parser.error("this post-test protocol is validation-only")
    if args.split == "test" and not args.confirm_frozen:
        parser.error("test split requires --confirm-frozen after validation choices are frozen")
    report = run(args.source_dir, protocol, args.split)
    output = args.output or ROOT / "artifacts" / f"code_language_{args.split}_v4.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["gate"]["pass"] else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
