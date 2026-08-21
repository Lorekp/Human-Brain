"""One-shot domain-shift gate on pre-registered Python standard-library files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sysconfig

from language_gate import (
    _aggregate,
    _baseline_counts,
    _bounded_text,
    _train_baselines,
    partition_files,
)
from humanbrain import CellLanguageModel, ModelConfig


ROOT = Path(__file__).resolve().parents[1]


def run(protocol: dict[str, object]) -> dict[str, object]:
    train_source = ROOT.parent / "src" / "deepsim"
    split_protocol = json.loads(
        (ROOT / "benchmarks" / "code_language_protocol_v4.json").read_text(
            encoding="utf-8"
        )
    )
    split = split_protocol["split"]
    partitions = partition_files(
        train_source,
        bucket_count=int(split["bucket_count"]),
        train_buckets=split["train_buckets"],
        validation_buckets=split["validation_buckets"],
        test_buckets=split["test_buckets"],
    )
    max_training_characters = int(protocol["max_training_characters_per_file"])
    max_predictions = int(protocol["max_predictions_per_file"])
    train_texts = [
        _bounded_text(path, max_training_characters) for path in partitions["train"]
    ]
    model_protocol = protocol["model"]
    max_order = int(model_protocol["max_order"])
    model = CellLanguageModel.grow(
        train_texts,
        ModelConfig(
            max_order=max_order,
            max_phrases=int(model_protocol["max_phrases"]),
            random_seed=int(model_protocol["random_seed"]),
        ),
    )
    baselines = _train_baselines(train_texts, max_order)
    stdlib = Path(sysconfig.get_path("stdlib"))
    external_files = [stdlib / str(relative) for relative in protocol["external_files"]]
    if any(not path.is_file() for path in external_files):
        raise FileNotFoundError("pre-registered standard-library source file is missing")

    folds: list[dict[str, object]] = []
    for path in external_files:
        text = _bounded_text(path, max_predictions + 1)
        cell = asdict(model.evaluate(text, max_predictions=max_predictions))
        scores = _baseline_counts(baselines, text, int(cell["predictions"]), max_order)
        fold = {
            "file": path.relative_to(stdlib).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "cell": cell,
            **scores,
        }
        fold["beats_unigram"] = cell["accuracy"] > scores["unigram"]["accuracy"]
        fold["beats_bigram"] = cell["accuracy"] > scores["bigram"]["accuracy"]
        folds.append(fold)

    aggregate = {
        name: _aggregate(folds, name)
        for name in ("cell", "unigram", "bigram", "variable_order", "witten_bell")
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
        "protocol": "train on DeepSim only; frozen one-shot evaluation on Python standard-library files; no test-time growth",
        "python_stdlib": str(stdlib.resolve()),
        "model_training_digest": model.training_digest,
        "training": {
            "files": [path.relative_to(train_source).as_posix() for path in partitions["train"]],
            "characters": sum(len(text) for text in train_texts),
            "grown_branches": model.grown_branches,
        },
        "evaluated_files": folds,
        "aggregate": aggregate,
        "gate": {"criteria": protocol["gate"], "pass": passed},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks" / "external_code_protocol_v1.json",
    )
    parser.add_argument("--confirm-frozen", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "external_code_test_v1.json",
    )
    args = parser.parse_args(argv)
    if not args.confirm_frozen:
        parser.error("one-shot external test requires --confirm-frozen")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = run(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
