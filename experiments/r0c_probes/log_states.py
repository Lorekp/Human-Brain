#!/usr/bin/env python3
"""R0-c Thread-1 D1-2 state-logging harness.

Replays the v3-freeze-candidate Bible QA corpus (TRAIN 200 + VAL 60 items)
through ladder-config growth (ridge readout, seed 7) at N=1024 and N=96,
logging via the ``observation_sink`` hook on ``RecurrentCellCircuit.observe``:

- per-token feature_state_vector() rows for the three strategy families
  {traces, spike_counts, assembly_overlap-rand}: float16 at a fixed stride,
  with every decision-point token additionally captured at float32;
- raw voltage / spike-count snapshots of the reservoir at each DECISION POINT
  (the answer:-boundary state: after observing the serialized prompt through
  "A:" inclusive, before any answer token).

Two arms per N (they differ in afferent wiring, so they cannot share a
circuit):

- arm ``explicit_spike``: feature_family="spike_counts" keeps the explicit
  ladder afferent wiring AND makes both the trace bank ("traces" family) and
  the spike-count windows host-visible per token -- one circuit logs two
  families;
- arm ``asm_rand``: feature_family="assembly_overlap" with the banked random
  codebook (k=16, seed-keyed streams, ladder _CODEBOOK_RNG_SALT) installed
  over the afferent table; logs the assembly_overlap-rand family.

Growth mirrors RecurrentCellLanguageModel.grow exactly (same tokenizer fit,
same length-prefixed sha256 training digest, grow_sequence epochs=1,
freeze_learning), with ridge statistics routed through
ladder_experiments.install_scipy_ridge (scipy OpenBLAS dsyrk/cho_factor --
required for F=8193 at N=1024).

Outputs (under experiments/r0c_probes/):
- logged_states.npz            keyed arrays (consumed by probes.py)
- logged_states_manifest.json  curriculum digests + config + array hashes

Deterministic: fixed seeds everywhere; the npz carries no wall-clock data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

SCHEMA_VERSION = "humanbrain/r0c-logged-states/v1"
DEFAULT_FREEZE_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Temp"
    / "hb_bible_freeze"
)
CODEBOOK_RNG_SALT = 0xC0DEB005  # ladder_experiments._CODEBOOK_RNG_SALT (documentation)
DEFAULT_ASSEMBLY_K = 16  # ladder_experiments._DEFAULT_ASSEMBLY_K


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LADDER = _load_module(
    "hb_ladder_experiments", REPO / "benchmarks" / "ladder_experiments.py"
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


# ------------------------------------------------------------------- corpus


def split_prompt(serialized: str) -> tuple[str, str]:
    """Prompt through 'A:' inclusive, and the remainder (answer terminator)."""
    marker = "\nA:"
    cut = serialized.rindex(marker) + len(marker)
    return serialized[:cut], serialized[cut:]


def training_digest(texts: list[str]) -> str:
    """Same construction as RecurrentCellLanguageModel.grow."""
    digest = hashlib.sha256()
    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


# ------------------------------------------------------------------ logging


class ReplayLogger:
    """observation_sink consumer: per-token features + boundary snapshots."""

    def __init__(self, circuit: RecurrentCellCircuit, stride: int):
        self.circuit = circuit
        self.stride = max(1, int(stride))
        self.token_offset = 0
        self.current_boundary: int | None = None
        # per-token family stores
        self.family_stride_rows: dict[str, list[np.ndarray]] = {}
        self.family_stride_idx: dict[str, list[int]] = {}
        self.family_boundary_rows: dict[str, list[np.ndarray]] = {}
        self.family_boundary_idx: dict[str, list[int]] = {}
        # decision-point snapshots
        self.boundary_voltage: list[np.ndarray] = []
        self.boundary_spikes: list[np.ndarray] = []
        self.boundary_feature: list[np.ndarray] = []
        self.boundary_traces: list[np.ndarray] = []
        self.boundary_global_index: list[int] = []
        self.all_token_ids: list[int] = []

    def begin_item(self, boundary_global: int) -> None:
        if self.current_boundary is not None:
            raise RuntimeError("previous item never ended")
        self.current_boundary = boundary_global

    def end_item(self) -> None:
        self.current_boundary = None

    def mark_boundary(self) -> None:
        """Capture raw snapshots; call right after the boundary observe()."""
        circuit = self.circuit
        self.boundary_voltage.append(
            np.asarray(
                circuit._reservoir.state.voltage_relative_mv, dtype=np.float64
            ).copy()
        )
        self.boundary_spikes.append(
            np.asarray(circuit._token_spike_counts, dtype=np.uint32).copy()
        )
        self.boundary_feature.append(
            np.asarray(circuit.feature_state_vector(), dtype=np.float32).copy()
        )
        self.boundary_traces.append(
            np.asarray(circuit._feature_traces, dtype=np.float32).copy()
        )
        self.boundary_global_index.append(self.token_offset - 1)

    def _sink(self, circuit: RecurrentCellCircuit, token_id: int) -> None:
        if circuit is not self.circuit:
            raise RuntimeError("observation_sink fired on a foreign circuit")
        t = self.token_offset
        self.all_token_ids.append(int(token_id))
        at_boundary = t == self.current_boundary
        vec = np.asarray(circuit.feature_state_vector())
        name = circuit.config.feature_family
        # The eligibility-trace bank is maintained in EVERY arm (frozen R0-b
        # protocol); log it as its own "traces" family stream so per-token
        # coverage of {traces} does not depend on the active readout family.
        trace_bank = np.asarray(circuit._feature_traces)
        if at_boundary:
            self.family_boundary_rows.setdefault(name, []).append(
                vec.astype(np.float32)
            )
            self.family_boundary_idx.setdefault(name, []).append(t)
            if name != "traces":
                self.family_boundary_rows.setdefault("traces", []).append(
                    trace_bank.astype(np.float32)
                )
                self.family_boundary_idx.setdefault("traces", []).append(t)
        elif t % self.stride == 0:
            self.family_stride_rows.setdefault(name, []).append(vec.astype(np.float16))
            self.family_stride_idx.setdefault(name, []).append(t)
            if name != "traces":
                self.family_stride_rows.setdefault("traces", []).append(
                    trace_bank.astype(np.float16)
                )
                self.family_stride_idx.setdefault("traces", []).append(t)
        self.token_offset += 1

    def attach(self) -> None:
        self.circuit.observation_sink = self._sink

    def detach(self) -> None:
        self.circuit.observation_sink = None


# -------------------------------------------------------------------- grow


def grow_arm(
    *,
    n_cells: int,
    seed: int,
    family: str,
    train_texts: list[str],
    growth_epochs: int,
    assembly_k: int | None,
    scipy_ridge: bool,
) -> tuple[RecurrentCellCircuit, BytePhraseTokenizer, str, str]:
    """Mirror RecurrentCellLanguageModel.grow with an explicit circuit."""
    tokenizer = BytePhraseTokenizer.fit(
        train_texts, TokenizerConfig(max_phrases=0, min_frequency=3)
    )
    config = RecurrentCircuitConfig(
        reservoir_cells=n_cells,
        input_fanout=LADDER._INPUT_FANOUT,
        recurrent_fanout=LADDER._RECURRENT_FANOUT,
        random_seed=seed,
        readout_learning_rule="ridge",
        inhibitory_fraction=LADDER.DEFAULT_INHIBITORY_FRACTION,
        feature_family=family,
    )
    circuit = RecurrentCellCircuit(token_cells=tokenizer.vocab_size, config=config)
    ridge_backend = "numpy"
    if scipy_ridge:
        ridge_backend = LADDER.install_scipy_ridge(circuit)
    codebook_sha = ""
    if assembly_k is not None:
        codebook = LADDER.build_codebook(
            "random",
            token_count=tokenizer.vocab_size,
            n_e=int(circuit.active_reservoir_cells),
            k=assembly_k,
            seed=seed,
        )
        LADDER.install_codebook(circuit, codebook)
        codebook_sha = codebook.sha256
    for text in train_texts:
        circuit.grow_sequence(tokenizer.encode(text), epochs=growth_epochs)
    circuit.freeze_learning()
    return circuit, tokenizer, ridge_backend, codebook_sha


def replay_items(
    logger: ReplayLogger,
    tokenizer: BytePhraseTokenizer,
    items: list[dict],
) -> tuple[list[dict], int]:
    """Teacher-force every item's serialized text through the logger."""
    rows: list[dict] = []
    for position, item in enumerate(items):
        prompt_text, _rest = split_prompt(item["serialized_prompt_answer"])
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = tokenizer.encode(item["serialized_prompt_answer"])
        if tuple(full_ids[: len(prompt_ids)]) != tuple(prompt_ids):
            raise ValueError(f"item {position}: prompt is not a byte-token prefix")
        boundary_global = logger.token_offset + len(prompt_ids) - 1
        logger.begin_item(boundary_global)
        circuit = logger.circuit
        circuit.reset_sequence()
        for index, token_id in enumerate(full_ids):
            circuit.observe(int(token_id))
            if index == len(prompt_ids) - 1:
                logger.mark_boundary()
        logger.end_item()
        meta = item.get("meta") or {}
        rows.append(
            {
                "qa_id": item["qa_id"],
                "family": item["family"],
                "cohort_position": position,
                "prompt_tokens": len(prompt_ids),
                "total_tokens": len(full_ids),
                "boundary_global_index": boundary_global,
                "answer_text": item["answer"],
                "chapter_distance": meta.get("chapter_distance"),
                "speaker_token": meta.get("speaker_token"),
            }
        )
    return rows, logger.token_offset


# --------------------------------------------------------------------- run


def run_arm(args_ns, items: dict, n_cells: int, arm: str) -> dict:
    train_items = items["train"][: args_ns.max_items] if args_ns.max_items else items["train"]
    val_items = items["val"][: args_ns.max_items] if args_ns.max_items else items["val"]
    train_texts = [it["serialized_prompt_answer"] for it in train_items]
    family = "spike_counts" if arm == "explicit_spike" else "assembly_overlap"
    started = time.perf_counter()
    circuit, tokenizer, ridge_backend, codebook_sha = grow_arm(
        n_cells=n_cells,
        seed=args_ns.seed,
        family=family,
        train_texts=train_texts,
        growth_epochs=args_ns.growth_epochs,
        assembly_k=args_ns.assembly_k if arm == "asm_rand" else None,
        scipy_ridge=n_cells >= 1024,
    )
    growth_wall = time.perf_counter() - started
    log(
        f"arm={arm} N={n_cells}: grown+consolidated in {growth_wall:.1f}s "
        f"(vocab={tokenizer.vocab_size}, features+1={circuit._growth_feature_count}, "
        f"ridge_backend={ridge_backend})"
    )

    stride = args_ns.stride_small if n_cells <= 128 else args_ns.stride_large
    lg = ReplayLogger(circuit, stride=stride)
    lg.attach()
    started = time.perf_counter()
    rows_by_split: dict[str, list[dict]] = {}
    tokens_by_split: dict[str, int] = {}
    start_token = 0
    for split_name, split_items in (("train", train_items), ("val", val_items)):
        rows, end_token = replay_items(lg, tokenizer, split_items)
        rows_by_split[split_name] = rows
        tokens_by_split[split_name] = end_token - start_token
        start_token = end_token
        log(f"arm={arm} N={n_cells}: replayed {split_name} ({len(rows)} items)")
    replay_wall = time.perf_counter() - started

    # Trajectory determinism spot-check: full reset + replay up to the
    # decision point must reproduce the captured boundary snapshot bit-for-bit.
    probe_row = rows_by_split["val"][0]
    probe_item = val_items[0]
    full_ids = tokenizer.encode(probe_item["serialized_prompt_answer"])
    circuit.reset_sequence()
    for token_id in full_ids[: probe_row["prompt_tokens"]]:
        circuit.observe(int(token_id))
    voltage_again = np.asarray(
        circuit._reservoir.state.voltage_relative_mv, dtype=np.float64
    )
    feature_again = np.asarray(circuit.feature_state_vector(), dtype=np.float32)
    voltage_first = lg.boundary_voltage[-len(val_items)]
    feature_first = lg.boundary_feature[-len(val_items)]
    determinism = bool(
        np.array_equal(voltage_again, voltage_first)
        and np.array_equal(feature_again, feature_first)
    )
    if not determinism:
        raise RuntimeError("replay determinism spot-check FAILED")
    log(f"arm={arm} N={n_cells}: replay determinism spot-check PASS")
    lg.detach()

    prefix = f"arm_{arm}_n{n_cells}"
    arrays: dict[str, np.ndarray] = {
        f"{prefix}_token_ids": np.asarray(lg.all_token_ids, dtype=np.int32),
        f"{prefix}_boundary_voltage": np.asarray(lg.boundary_voltage, dtype=np.float64),
        f"{prefix}_boundary_spikes": np.asarray(lg.boundary_spikes, dtype=np.uint32),
        f"{prefix}_boundary_feature": np.asarray(lg.boundary_feature, dtype=np.float32),
        f"{prefix}_boundary_traces": np.asarray(lg.boundary_traces, dtype=np.float32),
    }
    families_meta: dict[str, dict] = {}
    for name in sorted(set(lg.family_stride_rows) | set(lg.family_boundary_rows)):
        stride_idx = np.asarray(lg.family_stride_idx.get(name, []), dtype=np.int64)
        stride_rows = (
            np.stack(lg.family_stride_rows[name])
            if lg.family_stride_rows.get(name)
            else np.zeros((0, arrays[f"{prefix}_boundary_feature"].shape[1]), np.float16)
        )
        bnd_idx = np.asarray(lg.family_boundary_idx.get(name, []), dtype=np.int64)
        bnd_rows = np.stack(lg.family_boundary_rows[name])
        base = f"{prefix}_tokfeat__{name}"
        arrays[f"{base}_stride_idx"] = stride_idx
        arrays[f"{base}_stride_rows"] = stride_rows
        arrays[f"{base}_boundary_idx"] = bnd_idx
        arrays[f"{base}_boundary_rows"] = bnd_rows
        families_meta[name] = {
            "stride": stride,
            "stride_rows_shape": list(stride_rows.shape),
            "boundary_rows_shape": list(bnd_rows.shape),
            "dtype_stride": "float16",
            "dtype_boundary": "float32",
        }

    return {
        "prefix": prefix,
        "arrays": arrays,
        "meta": {
            "arm": arm,
            "family": family,
            "n_cells": n_cells,
            "seed": args_ns.seed,
            "growth_epochs": args_ns.growth_epochs,
            "assembly_k": args_ns.assembly_k if arm == "asm_rand" else None,
            "codebook_sha256": codebook_sha,
            "ridge_backend": ridge_backend,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "training_digest": training_digest(train_texts),
            "curriculum_digest": sha256_bytes(
                canonical_json(
                    [
                        [
                            {
                                key: row[key]
                                for key in (
                                    "qa_id",
                                    "family",
                                    "prompt_tokens",
                                    "total_tokens",
                                )
                            }
                            for row in rows_by_split[split]
                        ]
                        for split in ("train", "val")
                    ]
                )
            ),
            "tokens_train": tokens_by_split["train"],
            "tokens_val": tokens_by_split["val"],
            "items_train": len(train_items),
            "items_val": len(val_items),
            "rows": rows_by_split,
            "families": families_meta,
            "wall_growth_s": round(growth_wall, 3),
            "wall_replay_s": round(replay_wall, 3),
            "simulation_backend": circuit.simulation_backend,
            "simulation_device": circuit.simulation_device,
            "config": asdict(circuit.config),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    parser.add_argument("--out-dir", default=str(REPO / "experiments" / "r0c_probes"))
    parser.add_argument("--cells", type=int, nargs="+", default=[1024, 96])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--growth-epochs", type=int, default=1)
    parser.add_argument("--stride-large", type=int, default=16)
    parser.add_argument("--stride-small", type=int, default=1)
    parser.add_argument("--assembly-k", type=int, default=DEFAULT_ASSEMBLY_K)
    parser.add_argument("--arms", nargs="+", default=["explicit_spike", "asm_rand"])
    parser.add_argument("--max-items", type=int, default=None, help="debug cap per split")
    args = parser.parse_args()

    try:
        git_head = (
            __import__("subprocess")
            .run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=15,
            )
            .stdout.strip()
        )
    except Exception:
        git_head = "unknown"

    freeze_dir = Path(args.freeze_dir)
    ev = freeze_dir / "evidence"
    train_man = json.loads((ev / "qa_manifest_train.json").read_text(encoding="utf-8"))
    val_man = json.loads((ev / "qa_manifest_val.json").read_text(encoding="utf-8"))
    spans = json.loads((ev / "pg10_book_spans_v3.json").read_text(encoding="utf-8"))
    trimmed = (ev / "pg10.trimmed.bin").read_bytes()
    train_chunks = []
    for book in spans["books"]:
        if book["role"] == "train":
            lo, hi = book["span_offsets"]
            train_chunks.append(trimmed[lo:hi])
    train_stream = b"\n".join(train_chunks)

    items = {
        "train": train_man["items"],
        "val": val_man["items"],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "logged_states.npz"

    all_arrays: dict[str, np.ndarray] = {}
    arm_metas = []
    curriculum_digests = set()
    for n_cells in args.cells:
        for arm in args.arms:
            result = run_arm(args, items, n_cells=n_cells, arm=arm)
            all_arrays.update(result["arrays"])
            meta = result["meta"]
            meta.pop("rows")
            curriculum_digests.add(meta["curriculum_digest"])
            arm_metas.append(meta)
            del result
    if len(curriculum_digests) != 1:
        raise RuntimeError(
            f"curriculum digest mismatch across arms: {sorted(curriculum_digests)}"
        )

    import scipy

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_head": git_head,
        "seed": args.seed,
        "growth_epochs": args.growth_epochs,
        "stride_policy": {"small_n_le_128": args.stride_small, "large_n": args.stride_large},
        "corpus": {
            "source": "bible-kjv-pg10:v3-freeze-candidate",
            "candidate_sha256": sha256_bytes(
                (freeze_dir / "FREEZE-CANDIDATE-v3-BIBLE.json").read_bytes()
            ),
            "train_manifest_sha256": train_man["sha256_canonical"],
            "val_manifest_sha256": val_man["sha256_canonical"],
            "train_stream_sha256": sha256_bytes(train_stream),
            "train_count": len(items["train"]),
            "val_count": len(items["val"]),
            "max_items_cap": args.max_items,
        },
        "curriculum_digest": sorted(curriculum_digests)[0],
        "decision_point_rule": (
            "state after observing the serialized prompt through 'A:' inclusive "
            "(byte-token index len(prompt_ids)-1); teacher-forced replay"
        ),
        "arms": arm_metas,
        "npz_arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": sha256_bytes(np.ascontiguousarray(value).tobytes()),
            }
            for name, value in sorted(all_arrays.items())
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }

    tmp = npz_path.with_name("logged_states_tmp.npz")
    np.savez_compressed(tmp, **all_arrays)
    tmp.replace(npz_path)
    manifest_path = out_dir / "logged_states_manifest.json"
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    log(f"wrote {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    log(f"wrote {manifest_path}")
    print(json.dumps({"curriculum_digest": manifest["curriculum_digest"]}, indent=1))


if __name__ == "__main__":
    main()
