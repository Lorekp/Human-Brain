"""Million-scale CUDA LM smoke: text grow + read throughput at N=1,000,000.

Engineering instrument (never artifacts/): prices a real scripture-text
training run on THIS box by measuring, not projecting:

- circuit construction wall time at 1e6 reservoir cells (CUDA full-touch);
- GROW phase tokens/second (observe+strengthen over verse-context texts,
  sparse_three_factor readout — the dense ridge normal matrix cannot exist
  at this scale on any single box);
- READ phase tokens/second (teacher-forced race()+observe streaming);
- VRAM headroom (nvidia-smi deltas + kernel.device_bytes) and stability
  (sustained second pass; any error fails loudly).

Usage:
    python benchmarks/lm_million_smoke.py --cells 1000000 --grow-tokens 4000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))


def _gpu_used_mib() -> float | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return float(completed.stdout.strip().splitlines()[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=1_000_000)
    parser.add_argument("--grow-tokens", type=int, default=4_000)
    parser.add_argument("--read-tokens", type=int, default=2_000)
    parser.add_argument("--input-fanout", type=int, default=24)
    parser.add_argument("--recurrent-fanout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-phrases", type=int, default=384)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "lm_scale_bridge" / "_lm_smoke_results.json",
    )
    args = parser.parse_args()

    # Bind the CUDA policy BEFORE any circuit exists (recurrent_kernel reads
    # HUMANBRAIN_DEVICE at construction), mirroring scale_opt_gpu_bench.py.
    os.environ["HUMANBRAIN_DEVICE"] = "cuda"

    from humanbrain.qa_corpus import build_curriculum, growth_text
    from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
    from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

    curriculum = build_curriculum(args.seed)
    texts = [growth_text(task) for task in curriculum["train_tasks"]]  # type: ignore[arg-type]
    eval_text = " ".join(texts)

    gpu_before = _gpu_used_mib()
    t0 = time.perf_counter()
    tokenizer = BytePhraseTokenizer.fit(
        texts, TokenizerConfig(max_phrases=args.max_phrases, min_frequency=2)
    )
    fit_seconds = time.perf_counter() - t0

    config = RecurrentCircuitConfig(
        reservoir_cells=args.cells,
        active_reservoir_cells=args.cells,
        input_fanout=args.input_fanout,
        recurrent_fanout=args.recurrent_fanout,
        random_seed=args.seed,
        readout_learning_rule="sparse_three_factor",
    )
    t0 = time.perf_counter()
    circuit = RecurrentCellCircuit(token_cells=tokenizer.vocab_size, config=config)
    construct_seconds = time.perf_counter() - t0
    kernel = circuit._native_kernel
    if kernel.backend != "cuda":
        raise RuntimeError(f"expected CUDA backend, got {kernel.backend}")

    # --- GROW phase: observe+strengthen over repeated verse texts ----------
    growth_tokens_target = args.grow_tokens
    grown_tokens = 0
    t0 = time.perf_counter()
    repetition = 0
    while grown_tokens < growth_tokens_target:
        for text in texts:
            ids = tokenizer.encode(text)
            if not ids:
                continue
            circuit.grow_sequence(ids, epochs=1)
            grown_tokens += len(ids)
            if grown_tokens >= growth_tokens_target:
                break
        repetition += 1
        if repetition > 1000:
            raise RuntimeError("corpus too small for grow-token target")
    grow_seconds = time.perf_counter() - t0

    circuit.freeze_learning()
    freeze_seconds = time.perf_counter() - t0 - grow_seconds

    # --- READ phase: teacher-forced race()+observe streaming ---------------
    read_ids = tokenizer.encode(eval_text)[: args.read_tokens]
    while len(read_ids) < args.read_tokens:
        read_ids = read_ids + read_ids
    read_ids = read_ids[: args.read_tokens]
    circuit.reset_sequence()
    circuit.observe(read_ids[0])
    t0 = time.perf_counter()
    races = 0
    winners = 0
    kernel_ms = 0.0
    winners_sequence: list[int] = []
    for token_id in read_ids[1:]:
        race = circuit.race()
        races += 1
        winners += int(race.winner == token_id)
        winners_sequence.append(int(race.winner))
        circuit.observe(token_id)
        kernel_ms += float(kernel.last_kernel_milliseconds)
    read_seconds = time.perf_counter() - t0

    # --- stability probe: bit-exact inference determinism (learning frozen) --
    # Replay the identical read stream a second time and require an identical
    # winner sequence — sustained-operation + determinism evidence without
    # violating the frozen-learning contract.
    circuit.reset_sequence()
    circuit.observe(read_ids[0])
    replay_winners: list[int] = []
    t0 = time.perf_counter()
    for token_id in read_ids[1:]:
        race = circuit.race()
        replay_winners.append(int(race.winner))
        circuit.observe(token_id)
    stability_seconds = time.perf_counter() - t0
    first_pass_winners = winners_sequence
    stable = replay_winners == first_pass_winners
    if not stable:
        mismatch = next(
            (
                index
                for index, (a, b) in enumerate(
                    zip(first_pass_winners, replay_winners)
                )
                if a != b
            ),
            -1,
        )
        print(f"stability check FAILED: winner sequences diverge at {mismatch}")

    result = {
        "cells": args.cells,
        "backend": kernel.backend,
        "device": kernel.device_name,
        "vocab_size": tokenizer.vocab_size,
        "token_cells": circuit.token_cells,
        "construction_seconds": round(construct_seconds, 3),
        "tokenizer_fit_seconds": round(fit_seconds, 3),
        "freeze_seconds": round(freeze_seconds, 3),
        "growth": {
            "tokens": grown_tokens,
            "seconds": round(grow_seconds, 3),
            "tokens_per_second": round(grown_tokens / grow_seconds, 2),
            "strengthen_calls": grown_tokens,
            "learned_synapses": circuit.learned_synapse_count,
            # Sparse readout context: active contacts vs the structural
            # ceiling (token_cells x readout_fan_in) — sparsity is by design.
            "sparse_readout_fan_in": config.readout_fan_in,
            "synapse_ceiling": circuit.token_cells * config.readout_fan_in,
            "structural_recruitments": circuit.structural_recruitment_count,
            "structural_deletions": circuit.structural_deletion_count,
        },
        "read": {
            "tokens": len(read_ids),
            "races": races,
            "winner_match_rate": round(winners / max(1, races), 4),
            "chance_rate": round(1.0 / tokenizer.vocab_size, 4),
            "seconds": round(read_seconds, 3),
            "tokens_per_second": round(len(read_ids) / read_seconds, 2),
            "mean_kernel_ms_per_token": round(kernel_ms / max(1, len(read_ids)), 4),
        },
        "stability_inference_determinism": {
            "stable_bit_exact": stable,
            "replay_tokens": len(replay_winners),
            "replay_seconds": round(stability_seconds, 3),
            "replay_tokens_per_second": round(
                len(replay_winners) / stability_seconds, 2
            )
            if stability_seconds > 0
            else None,
        },
        "device_persistent_array_mib": round(kernel.device_bytes / (1024 * 1024), 1),
        "device_to_host_bytes_per_token": kernel.device_to_host_bytes_per_token,
        "host_mirror_mode": getattr(kernel, "_host_mirror", "unknown"),
        "gpu_memory_used_mib_before": gpu_before,
        "gpu_memory_used_mib_after": _gpu_used_mib(),
        "learned_synapses": circuit.learned_synapse_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    kernel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
