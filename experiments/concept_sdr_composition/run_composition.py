"""Concept-token vs byte-token SDR union composition study (U-STATE x concept tokens).

Question (a013 composition brief): do concept tokens produce MORE SEPARABLE
SDR unions than byte tokens?  Concepts are multi-token units ('the LORD '),
so their SDRs are activated by whole phrases — unions may carry longer
effective context per bit-set and avoid the common-byte collapse tail that
ustate-preregistration.md §2 measured (universal-bit L2 mass share 0.60-0.82).

Arms (identical projector n=2048 / w=32 / seed=7, identical union rule
U_t = max(lam*U_{t-1}, s_t), lam in frozen sweep {0.90, 0.97, 0.99}):

- byte_full   : BytePhraseTokenizer(max_phrases=0, min_frequency=3) -> ids
                [0,256); every byte its own SDR. This is exactly the week-1
                U-STATE tokenizer.
- concept     : ConceptTokenizer over the SAME base; mined concepts map to
                their own ids in [1024,1536) and thus to their own SDRs;
                residual bytes stay in the base band.
- byte_matched : byte stream TRUNCATED to the concept stream's length for the
                same verse — isolates "which bits get set" from "how many SDR
                writes happened" (concept streams are ~30% shorter, so with
                lam<1 they suffer less decay dilution; this control spends the
                same number of writes on a verse prefix).

Corpus: Bible TRAIN books from the frozen FREEZE-CANDIDATE-v3-BIBLE.json split
(same loader as experiments/concept_tokens/run_study.py). Sample = first 4
verses of each of the 49 train books + 4 top-up verses (books 1-4 take a 5th)
= 200 verses, deterministic.

Metrics per arm x lambda:
- pooled pairwise union cosine stats across the 200 unions
  (cosine_separation, the G-U1 metric family);
- support diagnostics: active bits (>0 and >=1e-2 floor), universal bits
  (active in >=95% of items) and their L2 mass share (prereg §2 collapse tail);
- active-bit trajectories at lam=0.97: mean active-bit count vs normalized
  within-verse position (20 bins);
- decode-ability proxy at every lambda: 49-way logistic probe (book id from
  the union), stratified 4-fold CV accuracy vs chance 1/49, plus a
  shuffled-label control at lam=0.97.

Outputs results_v1.json beside this script. Stdlib + numpy + sklearn only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from humanbrain.concept_tokens import (
    ConceptConfig,
    ConceptTokenizer,
    mine_concepts_detailed,
)
from humanbrain.sdr_union import (
    DEFAULT_LAMBDA,
    LAMBDA_SWEEP,
    cosine_separation,
)
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

FREEZE_DIR = Path(os.environ.get("TEMP", os.environ.get("LOCALAPPDATA", ""))) / "hb_bible_freeze"
N_VERSES = 200
VERSES_PER_BOOK = 4
TOPUP_BOOKS = N_VERSES - VERSES_PER_BOOK * 49  # 200 - 196
TRAJECTORY_BINS = 20
SUPPORT_FLOOR = 1e-2
UNIVERSAL_ACTIVE_FRAC = 0.95
CV_SEED = 42

VERSE_RE = re.compile(r"^(\d+):(\d+)\s")


def load_bible_train() -> tuple[list[tuple[str, str]], dict]:
    """(book short_name, book text) for every TRAIN book, canonical order."""
    freeze = json.loads((FREEZE_DIR / "FREEZE-CANDIDATE-v3-BIBLE.json").read_text(encoding="utf-8"))
    spans = json.loads(
        (FREEZE_DIR / "evidence" / "pg10_book_spans_v3.json").read_text(encoding="utf-8")
    )
    trimmed = (FREEZE_DIR / "evidence_pg10.trimmed.bin").read_bytes()

    digest = hashlib.sha256(trimmed).hexdigest()
    expected = freeze["corpus"]["sha256_trimmed"]
    if digest != expected:
        raise SystemExit(f"corpus drift: sha256 {digest} != frozen {expected} => STOP")

    books = []
    for book in spans["books"]:
        if book["role"] != "train":
            continue
        start, end = book["span_offsets"]
        books.append((book["short_name"], trimmed[start:end].decode("utf-8")))
    provenance = {
        "candidate_id": freeze["candidate_id"],
        "trimmed_sha256": digest,
        "sha256_match": True,
        "train_books": len(books),
    }
    return books, provenance


def segment_verses(book_text: str) -> list[str]:
    """Split one book into verse paragraphs ('c:v ...', blank-line separated)."""
    paragraphs = re.split(r"\r?\n\s*\r?\n", book_text)
    return [p.strip() for p in paragraphs if VERSE_RE.match(p.strip())]


def select_sample(books: list[tuple[str, str]]) -> tuple[list[dict], dict]:
    sample: list[dict] = []
    per_book_counts: list[int] = []
    for idx, (name, text) in enumerate(books):
        verses = segment_verses(text)
        per_book_counts.append(len(verses))
        take = VERSES_PER_BOOK + (1 if idx < TOPUP_BOOKS else 0)
        for j, verse in enumerate(verses[:take]):
            sample.append({"book": name, "book_index": idx, "verse_index": j, "text": verse})
    if len(sample) != N_VERSES:
        raise SystemExit(f"sample selection produced {len(sample)} verses, expected {N_VERSES}")
    stats = {
        "total_verses_train": int(sum(per_book_counts)),
        "min_verses_per_book": int(min(per_book_counts)),
        "max_verses_per_book": int(max(per_book_counts)),
        "n_classes": len({s["book"] for s in sample}),
    }
    return sample, stats


def encode_union(token_ids: list[int], projector, lam: float) -> np.ndarray:
    """Union state via the pinned rule; local copy so we can also trace steps."""
    n_bits = projector.n_bits
    union = np.zeros(n_bits, dtype=np.float64)
    trace: list[np.ndarray] = []
    for token in token_ids:
        union *= lam
        np.maximum(union, projector.vector(int(token)), out=union)
        trace.append(union.copy())
    return union, trace


def trajectory_summary(traces: list[list[np.ndarray]]) -> dict:
    """Mean active bits vs normalized within-verse position (20 bins)."""
    strict_bins = [[] for _ in range(TRAJECTORY_BINS)]
    floor_bins = [[] for _ in range(TRAJECTORY_BINS)]
    lengths = [len(t) for t in traces]
    for trace in traces:
        m = len(trace)
        if m == 0:
            continue
        for i, u in enumerate(trace):
            bin_idx = min((i * TRAJECTORY_BINS) // m, TRAJECTORY_BINS - 1)
            strict_bins[bin_idx].append(int(np.count_nonzero(u > 0)))
            floor_bins[bin_idx].append(int(np.count_nonzero(u >= SUPPORT_FLOOR)))
    return {
        "bin_centers_norm": [round((b + 0.5) / TRAJECTORY_BINS, 4) for b in range(TRAJECTORY_BINS)],
        "mean_active_bits_gt0": [round(float(np.mean(b)), 2) for b in strict_bins],
        "mean_active_bits_ge_floor": [round(float(np.mean(b)), 2) for b in floor_bins],
        "mean_verse_steps": round(float(np.mean(lengths)), 2),
    }


def support_stats(matrix: np.ndarray) -> dict:
    active_frac = (matrix > 0).mean(axis=0)
    universal_mask = active_frac >= UNIVERSAL_ACTIVE_FRAC
    l2_sq = (matrix**2).sum(axis=1)
    universal_mass = (matrix[:, universal_mask] ** 2).sum(axis=1) if universal_mask.any() else np.zeros(matrix.shape[0])
    return {
        "mean_support_gt0": round(float(np.count_nonzero(matrix > 0, axis=1).mean()), 2),
        "mean_support_ge_floor": round(float(np.count_nonzero(matrix >= SUPPORT_FLOOR, axis=1).mean()), 2),
        "universal_bits": int(universal_mask.sum()),
        "universal_bit_l2_mass_share": (
            round(float((universal_mass / l2_sq).mean()), 6) if universal_mask.any() else 0.0
        ),
    }


def probe_accuracy(unions: np.ndarray, labels: list[str], seed: int) -> dict:
    """49-way book-ID logistic probe, stratified 4-fold CV."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, random_state=seed),
    )
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(unions, labels):
        clf.fit(unions[tr], np.array(labels)[tr])
        accs.append(float((clf.predict(unions[te]) == np.array(labels)[te]).mean()))
    return {
        "fold_accuracies": [round(a, 4) for a in accs],
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "chance": round(1.0 / len(set(labels)), 4),
    }


def main() -> None:
    print("loading frozen Bible TRAIN books...", flush=True)
    books, prov = load_bible_train()
    sample, sample_stats = select_sample(books)
    print(f"{len(books)} train books, {sample_stats['total_verses_train']} verses total; "
          f"sampled {len(sample)} verses", flush=True)

    train_texts = [t for _, t in books]

    print("fitting base BytePhraseTokenizer (max_phrases=0, U-STATE spec)...", flush=True)
    base = BytePhraseTokenizer.fit(train_texts, TokenizerConfig(max_phrases=0, min_frequency=3))
    assert base.vocab_size == 256, f"base vocab {base.vocab_size} != 256"

    print("mining concepts (freq>=50, MI>=5bits, cap 512)...", flush=True)
    candidates = mine_concepts_detailed(train_texts, base, ConceptConfig())
    stack = ConceptTokenizer(base, tuple(c.payload for c in candidates))
    print(f"{len(candidates)} concepts, band [1024,{1024 + len(candidates)})", flush=True)

    # Tokenize all verses once per arm.
    arm_streams: dict[str, list[list[int]]] = {"byte_full": [], "concept": [], "byte_matched": []}
    concept_token_count = 0
    total_concept_stream_len = 0
    total_byte_stream_len = 0
    for row in sample:
        byte_ids = base.encode(row["text"])
        concept_ids = stack.encode(row["text"])
        assert stack._base.decode_bytes(byte_ids).decode("utf-8", "replace") == row["text"]
        arm_streams["byte_full"].append(byte_ids)
        arm_streams["concept"].append(concept_ids)
        arm_streams["byte_matched"].append(byte_ids[: len(concept_ids)])
        concept_token_count += sum(1 for t in concept_ids if t >= 1024)
        total_concept_stream_len += len(concept_ids)
        total_byte_stream_len += len(byte_ids)

    token_stats = {
        "verse_bytes_total": sum(len(r["text"].encode("utf-8")) for r in sample),
        "byte_full_total_steps": total_byte_stream_len,
        "concept_total_steps": total_concept_stream_len,
        "steps_ratio_concept_over_byte": round(total_concept_stream_len / total_byte_stream_len, 4),
        "concept_token_share_of_concept_stream": round(concept_token_count / total_concept_stream_len, 4),
        "distinct_ids_used": {
            "byte_full": len({t for s in arm_streams["byte_full"] for t in s}),
            "concept": len({t for s in arm_streams["concept"] for t in s}),
            "concept_band_ids_used": len({t for s in arm_streams["concept"] for t in s if t >= 1024}),
        },
    }
    print(f"token stats: {token_stats}", flush=True)

    labels = [r["book"] for r in sample]
    from humanbrain.sdr_union import SDRProjector

    projector = SDRProjector()  # n=2048 w=32 seed=7, prereg defaults

    results: dict = {
        "study": "concept-sdr-composition-v1",
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "provenance": prov,
        "protocol": {
            "projector": "SDRProjector defaults n=2048 w=32 seed=7 (ustate-preregistration.md section 1)",
            "union_rule": "U_t = max(lam*U_{t-1}, s_t)",
            "lambda_sweep": list(LAMBDA_SWEEP),
            "base_tokenizer": "BytePhraseTokenizer.fit(train, max_phrases=0, min_frequency=3)",
            "concept_config": {
                "min_frequency": 50, "min_mutual_information_bits": 5.0,
                "max_concepts": 512, "min_tokens": 2, "max_tokens": 12,
            },
            "sample": f"first {VERSES_PER_BOOK} verses of each of 49 train books (+1 top-up for books 1-{TOPUP_BOOKS})",
            "n_verses": N_VERSES,
            "arms": ["byte_full", "concept", "byte_matched"],
        },
        "sample_stats": sample_stats,
        "token_stats": token_stats,
        "top20_concepts": [
            {"text": c.payload.decode("utf-8"), "count": c.count,
             "mi_bits": round(c.mutual_information_bits, 2)}
            for c in candidates[:20]
        ],
        "arms": {},
    }

    for lam in LAMBDA_SWEEP:
        lam_key = f"lam{int(round(lam * 100)):03d}"
        print(f"=== encoding 200 verses x 3 arms at lam={lam} ===", flush=True)
        entry: dict = {}
        for arm, streams in arm_streams.items():
            unions = np.zeros((len(streams), projector.n_bits), dtype=np.float64)
            traces_store: list[list[np.ndarray]] = []
            for i, ids in enumerate(streams):
                u, trace = encode_union(ids, projector, lam)
                unions[i] = u
                traces_store.append(trace)
            sep = cosine_separation(unions)
            sup = support_stats(unions)
            row = {"separation": sep, "support": sup}
            if abs(lam - DEFAULT_LAMBDA) < 1e-12:
                row["trajectory"] = trajectory_summary(traces_store)
            entry[arm] = row
            print(f"  {arm:>13}: med_cos={sep['pooled_median_cosine']:.4f} "
                  f"max={sep['max_cross_item_cosine']:.4f} "
                  f">0.9:{sep['frac_pairs_cosine_above_090']:.4f} "
                  f"sup>0={sup['mean_support_gt0']:.1f}", flush=True)

        print(f"  book-ID logistic probe (49-way, 4-fold CV)...", flush=True)
        for arm, streams in arm_streams.items():
            unions = np.stack([
                encode_union(ids, projector, lam)[0] for ids in streams
            ])
            probe = probe_accuracy(unions, labels, CV_SEED)
            entry[arm]["probe_book_id"] = probe
            extra = ""
            if abs(lam - DEFAULT_LAMBDA) < 1e-12 and arm == "concept":
                rng = np.random.default_rng(CV_SEED)
                shuffled = rng.permutation(labels)
                entry[arm]["probe_shuffled_label_control"] = probe_accuracy(unions, list(shuffled), CV_SEED)
                extra = f" shuffled_ctrl={entry[arm]['probe_shuffled_label_control']['mean_accuracy']:.4f}"
            print(f"    {arm:>13}: acc={probe['mean_accuracy']:.4f} chance={probe['chance']:.4f}{extra}",
                  flush=True)
        results["arms"][lam_key] = entry

    out_path = Path(__file__).resolve().parent / "results_v1.json"
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
