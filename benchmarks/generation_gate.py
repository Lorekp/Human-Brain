"""Validation-only free-running continuation diagnostic for the cell model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from language_gate import _bounded_text, _counter_winner, partition_files
from humanbrain import CellLanguageModel, ModelConfig


ROOT = Path(__file__).resolve().parents[1]


def _common_prefix(left: bytes, right: bytes) -> int:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return min(len(left), len(right))


def run(protocol: dict[str, object]) -> dict[str, object]:
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
    training_limit = int(protocol["max_training_characters_per_file"])
    train_texts = [_bounded_text(path, training_limit) for path in partitions["train"]]
    model_protocol = protocol["model"]
    model = CellLanguageModel.grow(
        train_texts,
        ModelConfig(
            max_order=int(model_protocol["max_order"]),
            max_phrases=int(model_protocol["max_phrases"]),
            random_seed=int(model_protocol["random_seed"]),
        ),
    )
    train_bytes = b"".join(text.encode("utf-8") for text in train_texts)
    unigram_token = _counter_winner(Counter(train_bytes))
    prompt_characters = int(protocol["prompt_characters"])
    target_characters = int(protocol["target_characters"])

    folds = []
    for path in partitions["validation"]:
        text = path.read_text(encoding="utf-8")
        prompt = text[:prompt_characters]
        target = text[prompt_characters : prompt_characters + target_characters].encode(
            "utf-8"
        )
        if len(target) < target_characters:
            continue
        generated_text = model.complete(prompt, max_new_tokens=target_characters)
        generated = generated_text.encode("utf-8")
        compared = min(len(generated), len(target))
        correct = sum(
            generated[index] == target[index] for index in range(compared)
        )
        unigram_correct = sum(value == unigram_token for value in target)
        folds.append(
            {
                "file": path.relative_to(source_dir).as_posix(),
                "target_bytes": len(target),
                "generated_bytes": len(generated),
                "correct": correct,
                "accuracy": correct / len(target),
                "exact_prefix_bytes": _common_prefix(generated, target),
                "unigram_correct": unigram_correct,
                "unigram_accuracy": unigram_correct / len(target),
                "generated_preview": generated_text[:160],
            }
        )
    predictions = sum(int(fold["target_bytes"]) for fold in folds)
    correct = sum(int(fold["correct"]) for fold in folds)
    unigram_correct = sum(int(fold["unigram_correct"]) for fold in folds)
    report = {
        "protocol_version": protocol["version"],
        "protocol": "train-only growth; validation prompt; 256 free-running cell tokens; no teacher forcing or adaptation",
        "model_training_digest": model.training_digest,
        "evaluated_files": folds,
        "aggregate": {
            "predictions": predictions,
            "correct": correct,
            "accuracy": correct / predictions,
            "unigram_correct": unigram_correct,
            "unigram_accuracy": unigram_correct / predictions,
            "mean_exact_prefix_bytes": sum(
                int(fold["exact_prefix_bytes"]) for fold in folds
            )
            / len(folds),
        },
    }
    report["gate"] = {
        "criteria": protocol["gate"],
        "pass": report["aggregate"]["accuracy"]
        > report["aggregate"]["unigram_accuracy"],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks" / "generation_protocol_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "generation_validation_v1.json",
    )
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = run(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
