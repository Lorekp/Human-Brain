#!/usr/bin/env python3
"""U-STATE capacity study: union-overlap growth vs length across the grid.

Empirical validation of the analytic w^2/n capacity claims BEFORE the D3 probe
suite spends compute (precondition study declared by
docs/research/ustate-preregistration.md §2; runs on the frozen
FREEZE-CANDIDATE-v3-BIBLE evidence corpus, TRAIN books only).

Grid: lambda in {0.90, 0.97, 0.99} x dims {(n=2048, w=32), (n=4096, w=64)}.
Projection: the pinned seeded rule of sdr_union.py
(``default_rng((seed, n, w, token_id)).choice(n, w, replace=False)``), imported
from ``src/humanbrain.sdr_union`` — no reimplementation.

Measurements per config:
1. Token-SDR pairwise overlap over ALL byte-token pairs vs analytic w^2/n.
2. 200 real Bible verses (TRAIN split, seeded sample) encoded at incremental
   prefix lengths: pooled pairwise union cosine + active-bit counts vs token
   position (overlap-vs-length curves; checkpoint L uses only verses with
   length >= L), plus the full-verse boundary state (the direct G-U1 analog).
3. 200 disjoint contiguous 800-token windows of the TRAIN growth stream:
   same statistics at window checkpoints; end-of-window distinctness and
   retained sparsity (untouched-bit fraction under a 1e-2 floor).
4. G-U1 precondition check at realistic verse lengths: does any config keep
   pooled median cross-item cosine <= 0.90 on full-verse boundary states?

Pure numpy, CPU-only. Writes ustate_capacity_results.json and
overlap_vs_length_curves.csv beside this script.
"""

from __future__ import annotations

import csv
import json
import hashlib
import os
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from humanbrain.sdr_union import SDRProjector, cosine_separation  # noqa: E402

FREEZE_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Temp"
    / "hb_bible_freeze"
)

SEED = 7
LAMBDAS = (0.90, 0.97, 0.99)
DIMS = ((2048, 32), (4096, 64))  # (n_bits, w_ones)
N_ITEMS = 200
WINDOW = 800
N_WINDOWS = 200
FLOOR = 1e-2  # effective-support floor, matching the D1 precondition convention
PREFIX_CHECKPOINTS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384)
WINDOW_CHECKPOINTS = (50, 100, 200, 400, 600, 800)


VERSE_MARKER_RE = re.compile(r"\b(\d+):(\d+)\s")


def load_train_verses() -> tuple[list[str], list[str], str]:
    """Parse verses from TRAIN-role books of the frozen PG10 trimmed text.

    Mirrors tools/build_bible_qa_corpus.py::parse_verses: inline ``C:V``
    markers parsed BY POSITION (PG10 wraps multiple markers onto one physical
    line); continuation bytes append to the open verse until the next marker.
    """
    spans = json.loads(
        (FREEZE_DIR / "evidence" / "pg10_book_spans_v3.json").read_text(encoding="utf-8")
    )
    raw = (FREEZE_DIR / "evidence" / "pg10.trimmed.bin").read_bytes()
    texts: list[str] = []
    refs: list[str] = []
    for book in spans["books"]:
        if book["role"] != "train":
            continue
        lo, hi = book["span_offsets"]
        chunk = raw[lo:hi].decode("utf-8")
        markers = [
            (m.end(), int(m.group(1)), int(m.group(2)), m.start())
            for m in VERSE_MARKER_RE.finditer(chunk)
        ]
        parsed = []
        for i, (body_start, ch, ve, _mstart) in enumerate(markers):
            body_end = markers[i + 1][3] if i + 1 < len(markers) else len(chunk)
            joined = re.sub(r"\s+", " ", chunk[body_start:body_end]).strip()
            parsed.append((f'{book["short_name"]} {ch}:{ve}', joined))
        assert len(parsed) == book["verses"], (
            f'{book["short_name"]}: parsed {len(parsed)} verses, '
            f'span says {book["verses"]}'
        )
        for ref, text in parsed:
            texts.append(text)
            refs.append(ref)
    return texts, refs, spans["derivation_rule"]


def pair_cosine_stats(mat: np.ndarray) -> dict[str, float]:
    stats = cosine_separation(mat)
    return {
        k: stats[k]
        for k in (
            "n_items",
            "n_pairs",
            "pooled_median_one_minus_cosine",
            "pooled_median_cosine",
            "max_cross_item_cosine",
            "frac_pairs_cosine_above_090",
        )
    }


def active_bit_stats(mat: np.ndarray) -> dict[str, float]:
    any_touch = (mat > 0).sum(axis=1).astype(np.float64)
    effective = (mat >= FLOOR).sum(axis=1).astype(np.float64)
    return {
        "active_bits_any_touch_mean": float(any_touch.mean()),
        "active_bits_any_touch_median": float(np.median(any_touch)),
        "active_bits_effective_mean": float(effective.mean()),
        "active_bits_effective_median": float(np.median(effective)),
        "untouched_fraction_effective_mean": float(
            1.0 - effective.mean() / mat.shape[1]
        ),
    }


def stats_at(U: np.ndarray, rows: np.ndarray | None) -> dict[str, dict[str, float]]:
    mat = U if rows is None else U[rows]
    return {"cosine": pair_cosine_stats(mat), "active_bits": active_bit_stats(mat)}


def make_vecs(projector: SDRProjector) -> dict[int, np.ndarray]:
    return {t: projector.vector(t) for t in range(256)}


def step_column(
    U: np.ndarray, col: np.ndarray, lam: float, vecs: dict[int, np.ndarray]
) -> None:
    """One union step for every row whose token id >= 0 (padding skips)."""
    for tid in np.unique(col):
        t = int(tid)
        if t < 0:
            continue
        rows = col == tid
        U[rows] = np.maximum(lam * U[rows], vecs[t])


def encode_padded(
    tokens: np.ndarray,
    lens: np.ndarray,
    lam: float,
    vecs: dict[int, np.ndarray],
    checkpoints: tuple[int, ...],
) -> tuple[dict[int, dict], dict]:
    """Batch-encode padded streams; snapshot stats at checkpoints.

    Checkpoint L uses only rows with length >= L (true prefix unions);
    ``boundary`` is every row at its own final length (frozen after end).
    """
    n_bits = vecs[0].shape[0]
    U = np.zeros((tokens.shape[0], n_bits), dtype=np.float64)
    cps = sorted(c for c in checkpoints if int((lens >= c).sum()) >= 2)
    out: dict[int, dict] = {}
    cp_set = set(cps)
    for t in range(int(lens.max())):
        step_column(U, tokens[:, t], lam, vecs)
        if (t + 1) in cp_set:
            out[t + 1] = stats_at(U, lens >= (t + 1))
    boundary = stats_at(U, None)
    return out, boundary


def main() -> None:
    t0 = time.time()
    freeze_sha = hashlib.sha256(
        (FREEZE_DIR / "FREEZE-CANDIDATE-v3-BIBLE.json").read_bytes()
    ).hexdigest()

    verse_texts, verse_refs, derivation_rule = load_train_verses()
    rng = np.random.default_rng(SEED)
    picked = np.sort(rng.choice(len(verse_texts), size=N_ITEMS, replace=False))
    verses = [verse_texts[i] for i in picked]
    refs = [verse_refs[i] for i in picked]
    streams = [
        np.frombuffer(v.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        for v in verses
    ]
    lengths = np.array([len(s) for s in streams])
    max_len = int(lengths.max())
    verse_tokens = np.full((N_ITEMS, max_len), -1, dtype=np.int64)
    for i, s in enumerate(streams):
        verse_tokens[i, : len(s)] = s

    spans = json.loads(
        (FREEZE_DIR / "evidence" / "pg10_book_spans_v3.json").read_text(encoding="utf-8")
    )
    raw = (FREEZE_DIR / "evidence" / "pg10.trimmed.bin").read_bytes()

    report: dict = {
        "schema_version": "humanbrain/ustate-capacity-study/v1",
        "purpose": (
            "empirical validation of the w^2/n union-capacity claim across "
            "lambda x n x w before the U-STATE D3 probe suite spends compute"
        ),
        "projection_rule": (
            "humanbrain.sdr_union.SDRProjector: default_rng((seed, n, w, "
            "token_id)).choice(n, w, replace=False), sorted"
        ),
        "imported_module_sha256": hashlib.sha256(
            (REPO / "src" / "humanbrain" / "sdr_union.py").read_bytes()
        ).hexdigest(),
        "seed": SEED,
        "corpus": {
            "source": "bible-kjv-pg10:v3-freeze-candidate, TRAIN-role books only",
            "freeze_json_sha256": freeze_sha,
            "train_verses_available": len(verse_texts),
            "verses_sampled": N_ITEMS,
            "sample_rule": f"default_rng({SEED}).choice without replacement, sorted",
            "verse_refs_first10": refs[:10],
            "verse_token_lengths_mean": float(lengths.mean()),
            "verse_token_lengths_median": float(np.median(lengths)),
            "verse_token_lengths_min": int(lengths.min()),
            "verse_token_lengths_max": max_len,
            "tokenization": (
                "byte ids 0..255 (utf-8); preregistration section 1 tokenizer "
                "with max_phrases=0"
            ),
            "book_span_derivation": derivation_rule,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "analytic": {},
        "token_sdr_overlap": {},
        "configs": {},
    }

    csv_rows: list[dict] = []

    def emit_rows(key: str, series: str, curve: dict) -> None:
        for pos, val in sorted(curve.items()):
            csv_rows.append(
                {
                    "config": key,
                    "series": series,
                    "position_tokens": pos,
                    "n_items": val["cosine"]["n_items"],
                    "pooled_median_cosine": val["cosine"]["pooled_median_cosine"],
                    "median_one_minus_cosine": val["cosine"][
                        "pooled_median_one_minus_cosine"
                    ],
                    "max_cross_item_cosine": val["cosine"]["max_cross_item_cosine"],
                    "frac_pairs_above_090": val["cosine"]["frac_pairs_cosine_above_090"],
                    "active_bits_any_touch_mean": val["active_bits"][
                        "active_bits_any_touch_mean"
                    ],
                    "active_bits_effective_mean": val["active_bits"][
                        "active_bits_effective_mean"
                    ],
                    "untouched_fraction_effective_mean": val["active_bits"][
                        "untouched_fraction_effective_mean"
                    ],
                }
            )

    for n_bits, w_ones in DIMS:
        key_dim = f"n{n_bits}_w{w_ones}"
        projector = SDRProjector(n_bits=n_bits, w_ones=w_ones, seed=SEED)
        p = w_ones / n_bits
        vecs = make_vecs(projector)
        report["analytic"][key_dim] = {
            "pairwise_sdr_overlap_bits_w2_over_n": w_ones * w_ones / n_bits,
            "fresh_union_support_L200_bits": float(n_bits * (1 - (1 - p) ** 200)),
            "recency_window_tokens_by_lambda": {
                f"{lam:.2f}": float(np.log(FLOOR) / np.log(lam)) for lam in LAMBDAS
            },
        }

        # --- token-SDR overlap over ALL byte-token pairs (exact, 256 ids) ----
        ind = np.stack([vecs[t] for t in range(256)])
        gram = ind @ ind.T
        iu = np.triu_indices(256, k=1)
        overlaps = gram[iu]
        report["token_sdr_overlap"][key_dim] = {
            "n_pairs": int(overlaps.size),
            "mean_bits": float(overlaps.mean()),
            "median_bits": float(np.median(overlaps)),
            "p95_bits": float(np.percentile(overlaps, 95)),
            "max_bits": float(overlaps.max()),
            "analytic_w2_over_n": w_ones * w_ones / n_bits,
        }

        # --- 800-token windows from the TRAIN growth stream -------------------
        train_chunks = [
            raw[b["span_offsets"][0] : b["span_offsets"][1]].decode(
                "utf-8", errors="replace"
            )
            for b in spans["books"]
            if b["role"] == "train"
        ]
        growth = ("\n".join(train_chunks)).encode("utf-8")
        need = N_WINDOWS * WINDOW
        assert len(growth) >= need, f"TRAIN stream shorter than {need} bytes"
        windows = (
            np.frombuffer(growth[:need], dtype=np.uint8)
            .reshape(N_WINDOWS, WINDOW)
            .astype(np.int64)
        )

        for lam in LAMBDAS:
            key = f"{key_dim}_lam{lam:.2f}"
            print(f"[{time.time()-t0:6.1f}s] encoding config {key} ...", flush=True)

            curve, boundary = encode_padded(
                verse_tokens, lengths, lam, vecs, PREFIX_CHECKPOINTS
            )
            win_curve, _ = encode_padded(
                windows, np.full(N_WINDOWS, WINDOW), lam, vecs, WINDOW_CHECKPOINTS
            )

            report["configs"][key] = {
                "lambda": lam,
                "n_bits": n_bits,
                "w_ones": w_ones,
                "sparsity_w_over_n": p,
                "verse_boundary_full_lengths": boundary,
                "window800_curve": {str(pos): val for pos, val in win_curve.items()},
                "verse_prefix_curves": {str(pos): val for pos, val in curve.items()},
            }
            emit_rows(key, "verse_prefix", curve)
            # position_tokens = -1 marks the full-verse boundary series
            # (every verse at its own final length).
            emit_rows(key, "verse_boundary", {-1: boundary})
            emit_rows(key, "window800", win_curve)

    # ---------------------------------------------------------------- summary
    summary = {}
    for key, cfg in report["configs"].items():
        end = cfg["window800_curve"][str(WINDOW)]
        vb = cfg["verse_boundary_full_lengths"]
        summary[key] = {
            "window800_median_cosine": end["cosine"]["pooled_median_cosine"],
            "window800_median_one_minus_cosine": end["cosine"][
                "pooled_median_one_minus_cosine"
            ],
            "window800_frac_pairs_above_090": end["cosine"]["frac_pairs_cosine_above_090"],
            "window800_active_bits_effective_mean": end["active_bits"][
                "active_bits_effective_mean"
            ],
            "window800_retained_sparsity_fraction": end["active_bits"][
                "untouched_fraction_effective_mean"
            ],
            "verse_boundary_median_cosine": vb["cosine"]["pooled_median_cosine"],
            "verse_boundary_gu1_pass_median_cosine_le_090": (
                vb["cosine"]["pooled_median_cosine"] <= 0.90
            ),
        }
    report["summary"] = summary

    eligible = {
        k: v
        for k, v in summary.items()
        if v["window800_retained_sparsity_fraction"] >= 0.01
    }
    pool = eligible or summary
    best = min(
        pool.items(),
        key=lambda kv: (
            kv[1]["window800_median_cosine"],
            kv[1]["window800_frac_pairs_above_090"],
        ),
    )
    report["recommendation"] = {
        "rule": (
            "among configs retaining >=1% sparsity (untouched-bit fraction under "
            "a 1e-2 floor) at 800-token windows, minimize pooled median "
            "window-end cosine; tie-break on share of pairs above 0.90"
        ),
        "sparsity_eligible_configs": sorted(eligible.keys()),
        "best_config": best[0],
        **best[1],
    }

    out_dir = REPO / "experiments" / "ustate_capacity"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ustate_capacity_results.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (out_dir / "overlap_vs_length_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[{time.time()-t0:.1f}s] wrote {out_dir/'ustate_capacity_results.json'}")
    print(json.dumps(report["summary"], indent=1))
    print(json.dumps(report["recommendation"], indent=1))


if __name__ == "__main__":
    main()
