from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from statistics import median

from cell_scaling import _bounded_corpus
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.recurrent_language import RecurrentCellLanguageModel
from humanbrain.tokenizer import BytePhraseTokenizer
from language_gate import partition_files
from topology_gate import _state_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def _summary(runs: list[dict[str, object]], protocol: dict[str, object]) -> dict[str, object]:
    rows = []
    seed_count = len(protocol["random_seeds"])
    for configuration in protocol["configurations"]:
        matches = [run for run in runs if run["configuration"] == configuration["name"]]
        if len(matches) != seed_count:
            continue
        rows.append(
            {
                "configuration": configuration["name"],
                "effective_state_rank_median": median(
                    float(run["effective_state_rank"]) for run in matches
                ),
                "mean_absolute_cell_correlation_median": median(
                    float(run["mean_absolute_cell_correlation"]) for run in matches
                ),
                "mean_firing_hz_per_cell_median": median(
                    float(run["mean_firing_hz_per_cell"]) for run in matches
                ),
                "active_state_columns_median": median(
                    int(run["active_state_columns"]) for run in matches
                ),
            }
        )
    baseline = next(
        (row for row in rows if row["configuration"] == "v1_baseline"), None
    )
    for row in rows:
        row["stable_firing"] = 1.0 < row["mean_firing_hz_per_cell_median"] <= 80.0
        row["rank_exceeds_baseline"] = (
            baseline is not None
            and row["effective_state_rank_median"]
            > baseline["effective_state_rank_median"]
        )
    candidates = [
        row for row in rows if row["stable_firing"] and row["rank_exceeds_baseline"]
    ]
    selected = (
        min(
            candidates,
            key=lambda row: (
                -row["effective_state_rank_median"],
                row["mean_absolute_cell_correlation_median"],
            ),
        )["configuration"]
        if candidates
        else None
    )
    return {"configurations": rows, "selected_configuration": selected}


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
    corpus = "\n".join(_bounded_corpus(partitions["train"], 4096))
    tokenizer = BytePhraseTokenizer()
    if output.is_file():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report["protocol_sha256"] != protocol_hash:
            raise ValueError("existing output was produced by a different protocol")
    else:
        report = {
            "protocol_version": protocol["version"],
            "protocol_sha256": protocol_hash,
            "selection_data": "training",
            "validation_partition_accessed": False,
            "test_partition_accessed": False,
            "runs": [],
            "complete": False,
        }
    completed = {run["key"] for run in report["runs"]}
    specifications = [
        (configuration, seed)
        for configuration in protocol["configurations"]
        for seed in protocol["random_seeds"]
    ]
    added = 0
    for configuration, seed in specifications:
        key = f"configuration={configuration['name']}:seed={seed}"
        if key in completed:
            continue
        if limit_runs is not None and added >= limit_runs:
            break
        circuit = RecurrentCellCircuit(
            token_cells=tokenizer.vocab_size,
            config=RecurrentCircuitConfig(
                reservoir_cells=int(protocol["cell_count"]),
                input_fanout=int(configuration["input_fanout"]),
                recurrent_fanout=6,
                token_interval_ms=float(configuration["token_interval_ms"]),
                input_pulse_ms=float(configuration["input_pulse_ms"]),
                input_current_pa=float(configuration["input_current_pa"]),
                recurrent_current_pa=float(configuration["recurrent_current_pa"]),
                eligibility_tau_ms=float(configuration["eligibility_tau_ms"]),
                inhibitory_fraction=float(configuration["inhibitory_fraction"]),
                random_seed=int(seed),
                recurrent_topology=protocol["topology"],
            ),
        )
        model = RecurrentCellLanguageModel(tokenizer, circuit, training_digest="diagnostic")
        diagnostics = _state_diagnostics(
            model, corpus, int(protocol["diagnostic_tokens"])
        )
        result = {
            "key": key,
            "configuration": configuration["name"],
            "seed": seed,
            **diagnostics,
        }
        report["runs"].append(result)
        report["summary"] = _summary(report["runs"], protocol)
        report["complete"] = len(report["runs"]) == len(specifications)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        added += 1
        print(json.dumps(result), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "benchmarks" / "dynamics_calibration_protocol_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "dynamics_calibration_train_v1.json",
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
