#!/usr/bin/env python3
"""U-STATE D3 state-logging harness (mirror of r0c_probes/log_states.py).

Regenerates the r0c logged-states artifact layout with SDR union states
(docs/research/ustate-preregistration.md §3) replacing reservoir families over
the SAME corpus: Bible v3-freeze TRAIN 200 / VAL 60, byte tokenizer fitted
identically, same "\\nA:" decision point, same teacher-forced replay pattern
(UnionReplayLogger mirrors ReplayLogger: begin_item/mark_boundary/end_item,
per-token stride rows, boundary snapshots, reset-and-replay determinism spot
check). There is no reservoir: the union state IS the network state.

Arms = frozen lambda sweep {lam090, lam097, lam099} at n=2048/w=32/seed 7.
Slot mapping into the UNCHANGED probes.py layout (preregistration §3):

- boundary_voltage (f64) = U_T          - boundary_spikes (u32) = s_T indicator
- boundary_feature (f32) = U_T          - boundary_traces (f32) = C_T count-union

Gate 0 (reproduction): recomputed capacity statistics must match the frozen
precondition table within 2x. Gate G-U1 evaluated here per arm/split from
boundary unions (states only). Both results land in the manifest; mismatch or
failure STOPS before probes (stop-on-failure, preregistration §6).

Outputs (under experiments/ustate_probes/):
- logged_states.npz            keyed arrays (consumed by UNCHANGED probes.py)
- logged_states_manifest.json  corpus digests + config + array hashes + G-U1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.concept_tokens import ConceptConfig, ConceptTokenizer
from humanbrain.sdr_union import (
    DEFAULT_N_BITS,
    DEFAULT_SEED,
    DEFAULT_W_ONES,
    LAMBDA_SWEEP,
    SDRProjector,
    UnionEncoder,
    cosine_separation,
)
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

SCHEMA_VERSION = "humanbrain/ustate-logged-states/v1"
DEFAULT_OUT_DIR_BYTE = REPO / "experiments" / "ustate_probes"
DEFAULT_OUT_DIR_CONCEPT = REPO / "experiments" / "ustate_probes_concept"
DEFAULT_FREEZE_DIR = (
    Path(__import__("os").environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Temp"
    / "hb_bible_freeze"
)
R0C_CURRICULUM_DIGEST = (
    "c2cc7357ab073d0c8447c203e4482eedb751a464dce6df3d0aa134b66c65ee5c"
)
STRIDE_LARGE = 16


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%MZ', time.gmtime())}] {msg}", flush=True)


def split_prompt(serialized: str) -> tuple[str, str]:
    marker = "\nA:"
    cut = serialized.rindex(marker) + len(marker)
    return serialized[:cut], serialized[cut:]


def training_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


# ------------------------------------------------------------------- logging


class UnionReplayLogger:
    """Mirror of the r0c ReplayLogger around a UnionEncoder arm."""

    def __init__(self, encoder: UnionEncoder, stride: int):
        self.encoder = encoder
        self.stride = max(1, int(stride))
        self.token_offset = 0
        self.current_boundary: int | None = None
        self.family_stride_rows: dict[str, list[np.ndarray]] = {}
        self.family_stride_idx: dict[str, list[int]] = {}
        self.family_boundary_rows: dict[str, list[np.ndarray]] = {}
        self.family_boundary_idx: dict[str, list[int]] = {}
        self.boundary_voltage: list[np.ndarray] = []
        self.boundary_spikes: list[np.ndarray] = []
        self.boundary_feature: list[np.ndarray] = []
        self.boundary_traces: list[np.ndarray] = []
        self.all_token_ids: list[int] = []

    def begin_item(self, boundary_global: int) -> None:
        if self.current_boundary is not None:
            raise RuntimeError("previous item never ended")
        self.current_boundary = boundary_global

    def end_item(self) -> None:
        self.current_boundary = None

    def mark_boundary(self, state) -> None:
        """Capture the answer:-boundary snapshot (call after the last prompt observe)."""
        self.boundary_voltage.append(np.array(state.value, dtype=np.float64))
        self.boundary_spikes.append((state.last_sdr > 0).astype(np.uint32).copy())
        self.boundary_feature.append(state.value.astype(np.float32).copy())
        self.boundary_traces.append(state.count_value.astype(np.float32).copy())
        self.family_boundary_rows.setdefault("union", []).append(
            state.value.astype(np.float32)
        )
        self.family_boundary_rows.setdefault("count_union", []).append(
            state.count_value.astype(np.float32)
        )
        self.family_boundary_idx.setdefault("union", []).append(self.token_offset - 1)
        self.family_boundary_idx.setdefault("count_union", []).append(self.token_offset - 1)

    def observe(self, token_id: int) -> None:
        """One teacher-forced token advance (the sink fires after the update)."""
        t = self.token_offset
        self.encoder.state.step(int(token_id))
        self.all_token_ids.append(int(token_id))
        state = self.encoder.state
        at_boundary = t == self.current_boundary
        if at_boundary:
            self.mark_boundary(state)
        elif t % self.stride == 0:
            self.family_stride_rows.setdefault("union", []).append(
                state.value.astype(np.float16)
            )
            self.family_stride_rows.setdefault("count_union", []).append(
                state.count_value.astype(np.float16)
            )
            self.family_stride_idx.setdefault("union", []).append(t)
            self.family_stride_idx.setdefault("count_union", []).append(t)
        self.token_offset += 1


def replay_items(
    encoder: UnionEncoder,
    logger: UnionReplayLogger,
    encode_text,
    token_bytes_of,
    items: list[dict],
    encoding: str,
) -> tuple[list[dict], int]:
    """Teacher-forced replay; boundary = last token fully inside the prompt.

    Byte encoding keeps the r0c prefix assertion verbatim. Concept encoding
    (Change Note 1 §5) cuts the boundary by cumulative ``token_bytes`` length
    so a mined phrase straddling "\\nA:" is never half-observed, and asserts a
    lossless round-trip of the cut.
    """
    rows: list[dict] = []
    for position, item in enumerate(items):
        prompt_text, _rest = split_prompt(item["serialized_prompt_answer"])
        full_ids = encode_text(item["serialized_prompt_answer"])
        if encoding == "byte":
            prompt_ids = encode_text(prompt_text)
            if tuple(full_ids[: len(prompt_ids)]) != tuple(prompt_ids):
                raise ValueError(f"item {position}: prompt is not a byte-token prefix")
            prompt_count = len(prompt_ids)
        else:
            prompt_byte_len = len(prompt_text.encode("utf-8"))
            acc = 0
            prompt_count = 0
            for tid in full_ids:
                blen = len(token_bytes_of(int(tid)))
                if acc + blen <= prompt_byte_len:
                    acc += blen
                    prompt_count += 1
                else:
                    break
            if not 0 < prompt_count < len(full_ids):
                raise ValueError(f"item {position}: degenerate boundary cut")
            round_trip = b"".join(
                token_bytes_of(int(t)) for t in full_ids[:prompt_count]
            )
            # Prefix-consistency (Change Note 1 §5 erratum): mined concepts
            # legally straddle '\nA:' (18 of them contain the marker), so the
            # cut stops BEFORE the straddling token and the final 4-8 cue
            # bytes are uniformly unobserved in concept arms. Observing the
            # straddler instead would leak answer-region bytes into the
            # decision state. Record the omission per item.
            missed_bytes = prompt_byte_len - len(round_trip)
            if round_trip != prompt_text.encode("utf-8")[: len(round_trip)]:
                raise ValueError(f"item {position}: boundary cut not a lossless prefix")
        boundary_global = logger.token_offset + prompt_count - 1
        logger.begin_item(boundary_global)
        encoder.state.reset()
        for index, token_id in enumerate(full_ids):
            logger.observe(int(token_id))
        logger.end_item()
        meta = item.get("meta") or {}
        row = {
            "qa_id": item["qa_id"],
            "family": item["family"],
            "cohort_position": position,
            "prompt_tokens": prompt_count,
            "total_tokens": len(full_ids),
            "boundary_global_index": boundary_global,
            "answer_text": item["answer"],
            "chapter_distance": meta.get("chapter_distance"),
            "speaker_token": meta.get("speaker_token"),
        }
        if encoding != "byte":
            row["boundary_missed_prompt_bytes"] = missed_bytes
        rows.append(row)
    return rows, logger.token_offset


# ------------------------------------------------------------- gates & checks


def reproduction_check(train_unions: np.ndarray, precondition: dict) -> dict:
    """Gate 0: logged-state stats must match the frozen table within 2x."""
    supports = (train_unions > 0).sum(axis=1)
    effective = (train_unions >= 1e-2).sum(axis=1)
    sep = cosine_separation(train_unions)
    proj = precondition["projection"]
    rng = np.random.default_rng(proj["seed"])
    sample = rng.choice(100000, size=600, replace=False)
    p = SDRProjector(n_bits=proj["n_bits"], w_ones=proj["w_ones"], seed=proj["seed"])
    vecs = np.stack([p.vector(int(t)) for t in sample])
    gram = vecs @ vecs.T
    iu = np.triu_indices(len(sample), k=1)
    pair_overlap = float(gram[iu].mean())
    checks = {
        "pairwise_sdr_overlap_bits": (
            pair_overlap,
            precondition["analytic"]["pairwise_sdr_overlap_bits_w2_over_n"],
        ),
        "support_bits_mean_any_touch": (
            float(supports.mean()),
            float(
                np.mean(
                    [
                        precondition["lambdas"][k]["support_bits_mean_any_touch"]
                        for k in sorted(precondition["lambdas"])
                    ]
                )
            ),
        ),
        "effective_support_bits_mean": (
            float(effective.mean()),
            float(
                np.mean(
                    [
                        precondition["lambdas"][k][
                            "effective_support_bits_mean_1e-2_floor"
                        ]
                        for k in sorted(precondition["lambdas"])
                    ]
                )
            ),
        ),
        "pooled_median_one_minus_cosine": (
            sep["pooled_median_one_minus_cosine"],
            float(
                np.mean(
                    [
                        precondition["lambdas"][k]["pooled_median_one_minus_cosine"]
                        for k in sorted(precondition["lambdas"])
                    ]
                )
            ),
        ),
    }
    rows = {}
    ok = True
    for name, (measured, frozen) in checks.items():
        ratio = measured / frozen if frozen else float("inf")
        within = bool(0.5 <= ratio <= 2.0)
        ok &= within
        rows[name] = {"measured": measured, "frozen": frozen, "ratio": round(ratio, 4), "within_2x": within}
    return {"pass": bool(ok), "metrics": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stride", type=int, default=STRIDE_LARGE)
    parser.add_argument(
        "--encoding",
        choices=("byte", "concept"),
        default="byte",
        help="byte = frozen week-1 arms; concept = Change Note 1 primary arms",
    )
    parser.add_argument("--max-items", type=int, default=None, help="debug cap per split")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = str(
            DEFAULT_OUT_DIR_BYTE if args.encoding == "byte" else DEFAULT_OUT_DIR_CONCEPT
        )

    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
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

    train_items = train_man["items"]
    val_items = val_man["items"]
    if args.max_items:
        train_items = train_items[: args.max_items]
        val_items = val_items[: args.max_items]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identical fit to r0c log_states.py grow_arm; concept mode stacks the
    # Change Note 1 miner on the SAME base tokenizer.
    train_texts = [it["serialized_prompt_answer"] for it in train_items]
    tokenizer = BytePhraseTokenizer.fit(
        train_texts, TokenizerConfig(max_phrases=0, min_frequency=3)
    )
    assert tokenizer.vocab_size == 256, tokenizer.vocab_size
    if args.encoding == "concept":
        stack = ConceptTokenizer.fit(
            train_texts, base=tokenizer, config=ConceptConfig()
        )
        encode_text = stack.encode
        token_bytes_of = stack.token_bytes
        vocab_size = stack.vocab_size
        n_concepts = len(stack.concepts)
        log(f"encoding=concept: {n_concepts} concepts mined, band [1024,{1024 + n_concepts})")
    else:
        encode_text = tokenizer.encode
        token_bytes_of = None
        vocab_size = tokenizer.vocab_size
        n_concepts = 0

    all_arrays: dict[str, np.ndarray] = {}
    arm_metas: list[dict] = []
    curriculum_digests: set[str] = set()
    repro_report: dict | None = None

    for lam in LAMBDA_SWEEP:
        arm = f"lam{int(round(lam * 100)):03d}"
        if args.encoding == "concept":
            arm = f"{arm}_concept"
        started = time.perf_counter()
        encoder = UnionEncoder(n_bits=DEFAULT_N_BITS, w_ones=DEFAULT_W_ONES, seed=args.seed, lam=float(lam))
        lg = UnionReplayLogger(encoder, stride=args.stride)
        rows_by_split: dict[str, list[dict]] = {}
        tokens_by_split: dict[str, int] = {}
        start_token = 0
        for split_name, split_items in (("train", train_items), ("val", val_items)):
            rows, end_token = replay_items(
                encoder, lg, encode_text, token_bytes_of, split_items, args.encoding
            )
            rows_by_split[split_name] = rows
            tokens_by_split[split_name] = end_token - start_token
            start_token = end_token
            log(f"arm={arm}: replayed {split_name} ({len(rows)} items)")

        # Determinism spot-check: reset + replay of val item 0 to the boundary.
        probe_row = rows_by_split["val"][0]
        probe_item = val_items[0]
        full_ids = encode_text(probe_item["serialized_prompt_answer"])
        encoder.state.reset()
        for token_id in full_ids[: probe_row["prompt_tokens"]]:
            encoder.state.step(int(token_id))
        if not np.array_equal(encoder.state.value, lg.boundary_voltage[-len(val_items)]):
            raise RuntimeError(f"arm={arm}: replay determinism spot-check FAILED")
        log(f"arm={arm}: replay determinism spot-check PASS")

        prefix = f"arm_{arm}_n{DEFAULT_N_BITS}"
        arrays = {
            f"{prefix}_token_ids": np.asarray(lg.all_token_ids, dtype=np.int32),
            f"{prefix}_boundary_voltage": np.asarray(lg.boundary_voltage, dtype=np.float64),
            f"{prefix}_boundary_spikes": np.asarray(lg.boundary_spikes, dtype=np.uint32),
            f"{prefix}_boundary_feature": np.asarray(lg.boundary_feature, dtype=np.float32),
            f"{prefix}_boundary_traces": np.asarray(lg.boundary_traces, dtype=np.float32),
        }
        families_meta: dict[str, dict] = {}
        for name in sorted(set(lg.family_stride_rows) | {"union", "count_union"}):
            stride_idx = np.asarray(lg.family_stride_idx.get(name, []), dtype=np.int64)
            stride_rows = (
                np.stack(lg.family_stride_rows[name])
                if lg.family_stride_rows.get(name)
                else np.zeros((0, DEFAULT_N_BITS), np.float16)
            )
            bnd_idx = np.asarray(lg.family_boundary_idx.get(name, []), dtype=np.int64)
            bnd_rows = np.stack(lg.family_boundary_rows[name])
            base = f"{prefix}_tokfeat__{name}"
            arrays[f"{base}_stride_idx"] = stride_idx
            arrays[f"{base}_stride_rows"] = stride_rows
            arrays[f"{base}_boundary_idx"] = bnd_idx
            arrays[f"{base}_boundary_rows"] = bnd_rows
            families_meta[name] = {
                "stride": args.stride,
                "stride_rows_shape": list(stride_rows.shape),
                "boundary_rows_shape": list(bnd_rows.shape),
                "dtype_stride": "float16",
                "dtype_boundary": "float32",
            }
        all_arrays.update(arrays)

        # Gate 0 reproduction check (byte arms only; Change Note 1 §4).
        if args.encoding == "byte" and repro_report is None:
            precond_path = Path(args.out_dir) / "capacity_precondition.json"
            precondition = json.loads(precond_path.read_text(encoding="utf-8"))
            repro_report = reproduction_check(
                arrays[f"{prefix}_boundary_voltage"], precondition
            )
            log(f"gate0 reproduction: {'PASS' if repro_report['pass'] else 'FAIL'}")
            if not repro_report["pass"]:
                raise RuntimeError("capacity precondition NOT reproduced within 2x")

        # G-U1 per split from boundary unions (states only).
        n_train = len(train_items)
        gu1 = {}
        for split, sl in (("train", slice(0, n_train)), ("val", slice(n_train, None))):
            sep = cosine_separation(arrays[f"{prefix}_boundary_voltage"][sl])
            sep["pass_ge_0p10"] = bool(sep["pooled_median_one_minus_cosine"] >= 0.10)
            gu1[split] = sep
        gu1["pass"] = bool(gu1["train"]["pass_ge_0p10"] and gu1["val"]["pass_ge_0p10"])

        curriculum_digest = sha256_bytes(
            canonical_json(
                [
                    [
                        {key: row[key] for key in ("qa_id", "family", "prompt_tokens", "total_tokens")}
                        for row in rows_by_split[split]
                    ]
                    for split in ("train", "val")
                ]
            )
        )
        curriculum_digests.add(curriculum_digest)
        arm_metas.append(
            {
                "arm": arm,
                "family": "sdr_union",
                "encoding": args.encoding,
                "n_bits": DEFAULT_N_BITS,
                "w_ones": DEFAULT_W_ONES,
                "seed": args.seed,
                "lam": float(lam),
                "tokenizer_vocab_size": vocab_size,
                "n_concepts": n_concepts,
                "training_digest": training_digest(train_texts),
                "curriculum_digest": curriculum_digest,
                "tokens_train": tokens_by_split["train"],
                "tokens_val": tokens_by_split["val"],
                "items_train": len(train_items),
                "items_val": len(val_items),
                "rows": rows_by_split,
                "families": families_meta,
                "gu1_separation": gu1,
                "wall_replay_s": round(time.perf_counter() - started, 3),
            }
        )
        log(
            f"arm={arm}: G-U1 train={gu1['train']['pooled_median_one_minus_cosine']:.4f} "
            f"val={gu1['val']['pooled_median_one_minus_cosine']:.4f} "
            f"({'PASS' if gu1['pass'] else 'FAIL'})"
        )

    if args.encoding == "byte":
        if curriculum_digests != {R0C_CURRICULUM_DIGEST}:
            raise RuntimeError(
                f"curriculum digest mismatch vs r0c anchor: {sorted(curriculum_digests)}"
            )
        log(f"curriculum digest matches r0c anchor: {R0C_CURRICULUM_DIGEST}")
    else:
        log(f"concept curriculum digest recorded: {sorted(curriculum_digests)}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_head": git_head,
        "encoding": args.encoding,
        "seed": args.seed,
        "projection_rng": "numpy default_rng((seed, n, w, token_id)).choice(n, w, replace=False)",
        "lambda_sweep": list(LAMBDA_SWEEP),
        "corpus": {
            "source": "bible-kjv-pg10:v3-freeze-candidate",
            "candidate_sha256": sha256_bytes(
                (freeze_dir / "FREEZE-CANDIDATE-v3-BIBLE.json").read_bytes()
            ),
            "train_manifest_sha256": train_man["sha256_canonical"],
            "val_manifest_sha256": val_man["sha256_canonical"],
            "train_stream_sha256": sha256_bytes(train_stream),
            "train_count": len(train_items),
            "val_count": len(val_items),
            "max_items_cap": args.max_items,
        },
        "curriculum_digest": sorted(curriculum_digests)[0],
        "decision_point_rule": (
            "union state after observing the serialized prompt through 'A:' "
            "inclusive (byte-token index len(prompt_ids)-1); teacher-forced replay"
        ),
        "gate0_reproduction": repro_report,
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
        },
    }

    npz_path = out_dir / "logged_states.npz"
    tmp = npz_path.with_name("logged_states_tmp.npz")
    np.savez_compressed(tmp, **all_arrays)
    tmp.replace(npz_path)
    for arm in manifest["arms"]:
        arm.pop("rows")
    manifest_path = out_dir / "logged_states_manifest.json"
    manifest_path.write_bytes(canonical_json(manifest) + b"\n")

    log(f"wrote {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    log(f"wrote {manifest_path}")
    print(json.dumps({"curriculum_digest": manifest["curriculum_digest"], "gu1": {a["arm"]: a["gu1_separation"]["pass"] for a in manifest["arms"]}}, indent=1))


if __name__ == "__main__":
    main()
