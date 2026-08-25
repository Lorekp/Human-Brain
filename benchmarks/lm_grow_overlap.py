"""Quantify and eliminate host-device idle gaps in the CUDA LM grow epoch.

Engineering instrument (never artifacts/). Three modes:

- ``instrument``: per-token timeline of one grow epoch at N cells —
  python prep | kernel (CUDA event clock) | D2H mirror | strengthen —
  and the resulting GPU-idle fraction of wall time.
- ``gate``: byte-identical trajectory check. Grows a fixed token budget on
  seed 17, freezes learning, and hashes every persistent cell/synapse/readout
  array plus a teacher-forced winner replay. Run once with
  ``HUMANBRAIN_CUDA_ASYNC=0`` and once with ``=1``; hashes must match.
- ``bench``: end-to-end grow tok/s for one pipeline mode at one scale.

Usage:
    python benchmarks/lm_grow_overlap.py instrument --cells 131072 --tokens 400
    python benchmarks/lm_grow_overlap.py gate --label sync --tokens 600
    python benchmarks/lm_grow_overlap.py bench --cells 131072 --tokens 400 --mode async
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))


def _build_circuit(cells: int, seed: int):
    os.environ.setdefault("HUMANBRAIN_DEVICE", "cuda")
    from humanbrain.qa_corpus import build_curriculum, growth_text
    from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
    from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

    curriculum = build_curriculum(seed)
    texts = [growth_text(task) for task in curriculum["train_tasks"]]  # type: ignore[arg-type]
    tokenizer = BytePhraseTokenizer.fit(
        texts, TokenizerConfig(max_phrases=384, min_frequency=2)
    )
    config = RecurrentCircuitConfig(
        reservoir_cells=cells,
        active_reservoir_cells=cells,
        input_fanout=24,
        recurrent_fanout=8,
        random_seed=seed,
        readout_learning_rule="sparse_three_factor",
    )
    circuit = RecurrentCellCircuit(token_cells=tokenizer.vocab_size, config=config)
    kernel = circuit._native_kernel
    if kernel.backend != "cuda":
        raise RuntimeError(f"expected CUDA backend, got {kernel.backend}")
    return circuit, kernel, tokenizer, texts


def _grow_ids(tokenizer, texts, budget: int) -> list[int]:
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
        if len(ids) >= budget + 1:
            break
    if len(ids) < budget + 1:
        raise RuntimeError("corpus too small for token budget")
    return ids[: budget + 1]


def _t0() -> float:
    return time.perf_counter()


def mode_instrument(args: argparse.Namespace) -> int:
    import numpy as np

    from humanbrain.cuda import CudaRecurrentKernel

    circuit, kernel, tokenizer, texts = _build_circuit(args.cells, args.seed)
    ids = _grow_ids(tokenizer, texts, args.tokens)

    d2h = {"seconds": 0.0, "bytes": 0}
    original_copy_from = CudaRecurrentKernel._copy_from_device

    def timed_copy_from(self, name, array):  # type: ignore[no-untyped-def]
        start = _t0()
        original_copy_from(self, name, array)
        d2h["seconds"] += _t0() - start
        d2h["bytes"] += array.nbytes

    CudaRecurrentKernel._copy_from_device = timed_copy_from  # type: ignore[method-assign]
    try:
        t_reset = _t0()
        circuit.reset_sequence()
        reset_seconds = _t0() - t_reset

        rows: list[tuple[float, float, float, float]] = []
        t_encode = _t0()
        encoded = [int(token_id) for token_id in ids]
        encode_seconds = _t0() - t_encode

        t_epoch = _t0()
        for index in range(len(encoded) - 1):
            before = d2h["seconds"]
            start = _t0()
            circuit.observe(encoded[index])
            observe_seconds = _t0() - start
            copy_seconds = d2h["seconds"] - before
            start = _t0()
            circuit.strengthen(encoded[index + 1])
            strengthen_seconds = _t0() - start
            rows.append(
                (
                    observe_seconds,
                    float(kernel.last_kernel_milliseconds) * 1e-3,
                    copy_seconds,
                    strengthen_seconds,
                )
            )
        epoch_seconds = _t0() - t_epoch
    finally:
        CudaRecurrentKernel._copy_from_device = original_copy_from  # type: ignore[method-assign]

    observe = np.array([r[0] for r in rows])
    kernel_s = np.array([r[1] for r in rows])
    copy_s = np.array([r[2] for r in rows])
    strengthen = np.array([r[3] for r in rows])
    tokens = len(rows)
    kernel_total = float(kernel_s.sum())
    copy_total = float(copy_s.sum())
    host_total = float((observe + strengthen).sum()) - kernel_total - copy_total
    idle_incl_copies = epoch_seconds - kernel_total
    idle_excl_copies = epoch_seconds - kernel_total - copy_total

    result = {
        "mode": "instrument",
        "cells": args.cells,
        "seed": args.seed,
        "tokens": tokens,
        "vocab": tokenizer.vocab_size,
        "epoch_seconds": round(epoch_seconds, 4),
        "reset_seconds": round(reset_seconds, 4),
        "encode_prep_seconds": round(encode_seconds, 4),
        "ms_per_token": {
            "wall": round(epoch_seconds / tokens * 1e3, 3),
            "kernel_mean": round(float(kernel_s.mean()) * 1e3, 3),
            "kernel_p50": round(float(np.percentile(kernel_s, 50)) * 1e3, 3),
            "kernel_max": round(float(kernel_s.max()) * 1e3, 3),
            "d2h_mean": round(float(copy_s.mean()) * 1e3, 3),
            "d2h_max": round(float(copy_s.max()) * 1e3, 3),
            "strengthen_mean": round(float(strengthen.mean()) * 1e3, 3),
            "strengthen_max": round(float(strengthen.max()) * 1e3, 3),
            "host_python_mean": round(host_total / tokens * 1e3, 3),
        },
        "tokens_per_second": round(tokens / epoch_seconds, 1),
        "gpu_idle_fraction_of_wall": round(idle_incl_copies / epoch_seconds, 4),
        "gpu_idle_excluding_dma_busy": round(idle_excl_copies / epoch_seconds, 4),
        "d2h_mb_per_token": round(d2h["bytes"] / max(1, tokens) / 1e6, 3),
        "decision_threshold": "overlap profitable only if idle >= 5% of wall",
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    kernel.close()
    return 0


def mode_gate(args: argparse.Namespace) -> int:
    circuit, kernel, tokenizer, texts = _build_circuit(args.cells, args.seed)
    ids = _grow_ids(tokenizer, texts, args.tokens)

    t0 = _t0()
    circuit.grow_sequence(ids[: args.tokens], epochs=args.epochs)
    grow_seconds = _t0() - t0

    circuit.freeze_learning()

    # Teacher-forced winner replay on the frozen circuit (bit-exact expected).
    read_ids = ids[: min(256, args.tokens)]
    circuit.reset_sequence()
    circuit.observe(read_ids[0])
    winners: list[int] = []
    for token_id in read_ids[1:]:
        winners.append(int(circuit.race().winner))
        circuit.observe(token_id)

    state_sha = circuit.cell_synapse_state_sha256
    winners_sha = hashlib.sha256(
        ",".join(str(w) for w in winners).encode("ascii")
    ).hexdigest()

    result = {
        "mode": "gate",
        "label": args.label,
        "cells": args.cells,
        "seed": args.seed,
        "grow_tokens": args.tokens,
        "epochs": args.epochs,
        "async_env": os.environ.get("HUMANBRAIN_CUDA_ASYNC", "<unset>"),
        "checkpoint_sha256": state_sha,
        "winner_replay_sha256": winners_sha,
        "learned_synapses": circuit.learned_synapse_count,
        "local_updates": circuit.local_update_count,
        "structural_recruitments": circuit.structural_recruitment_count,
        "grow_tokens_per_second": round(args.tokens * args.epochs / grow_seconds, 1),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    kernel.close()
    return 0


def mode_bench(args: argparse.Namespace) -> int:
    os.environ["HUMANBRAIN_CUDA_ASYNC"] = "1" if args.mode == "async" else "0"
    circuit, kernel, tokenizer, texts = _build_circuit(args.cells, args.seed)
    ids = _grow_ids(tokenizer, texts, args.tokens)

    # Unmeasured warmup pass (lazy allocations, cubin load); identical for
    # both modes so the measured pass compares like with like.
    circuit.grow_sequence(ids[:64], epochs=1)
    t0 = _t0()
    circuit.grow_sequence(ids[: args.tokens], epochs=args.epochs)
    grow_seconds = _t0() - t0

    result = {
        "mode": "bench",
        "pipeline": args.mode,
        "cells": args.cells,
        "seed": args.seed,
        "tokens": args.tokens,
        "epochs": args.epochs,
        "grow_seconds": round(grow_seconds, 3),
        "grow_tokens_per_second": round(
            args.tokens * args.epochs / grow_seconds, 1
        ),
        "ms_per_token": round(grow_seconds / (args.tokens * args.epochs) * 1e3, 3),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    kernel.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--cells", type=int, default=131_072)
        sub.add_argument("--tokens", type=int, default=400)
        sub.add_argument("--seed", type=int, default=17)
        sub.add_argument("--epochs", type=int, default=1)
        sub.add_argument("--output", type=Path, default=None)

    instrument = subparsers.add_parser("instrument", help="per-token idle-gap timeline")
    add_common(instrument)
    gate = subparsers.add_parser("gate", help="byte-identical trajectory hash")
    add_common(gate)
    gate.add_argument("--label", type=str, required=True)
    bench = subparsers.add_parser("bench", help="end-to-end grow tok/s")
    add_common(bench)
    bench.add_argument("--pipeline", dest="mode", choices=["sync", "async"], required=True)

    args = parser.parse_args()
    if args.mode == "instrument":
        return mode_instrument(args)
    if args.mode == "gate":
        return mode_gate(args)
    return mode_bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
