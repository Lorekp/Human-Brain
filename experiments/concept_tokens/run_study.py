"""Concept-token study runner: Bible TRAIN books vs repo Python code.

Fits per corpus: BytePhraseTokenizer (base), ConceptTokenizer stack (mined
concepts, freq>=50 / MI>=5 bits / cap 512), plain BPE-4096 and matched-budget
BPE-1280 (stack vocab = 768 base + 512 concepts). Evaluates every fitted
tokenizer on both corpora and reports bytes/token compression.

Outputs results_v1.json next to this script. Stdlib-only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpe_baseline import BpeTokenizer  # noqa: E402
from humanbrain.concept_tokens import (  # noqa: E402
    ConceptConfig,
    ConceptTokenizer,
    mine_concepts_detailed,
)
from humanbrain.tokenizer import BytePhraseTokenizer  # noqa: E402

FREEZE_DIR = Path(os.environ.get("TEMP", os.environ.get("LOCALAPPDATA", ""))) / "hb_bible_freeze"
CODE_TARGET_BYTES = 204_800


def load_bible_train() -> tuple[list[str], dict]:
    freeze = json.loads((FREEZE_DIR / "FREEZE-CANDIDATE-v3-BIBLE.json").read_text(encoding="utf-8"))
    spans = json.loads(
        (FREEZE_DIR / "evidence" / "pg10_book_spans_v3.json").read_text(encoding="utf-8")
    )
    trimmed = (FREEZE_DIR / "evidence_pg10.trimmed.bin").read_bytes()
    texts = []
    for book in spans["books"]:
        if book["role"] != "train":
            continue
        start, end = book["span_offsets"]
        texts.append(trimmed[start:end].decode("utf-8"))
    provenance = {
        "candidate_id": freeze["candidate_id"],
        "trimmed_sha256": freeze["corpus"]["sha256_trimmed"],
        "train_books": len(texts),
    }
    return texts, provenance


def load_code_sample() -> tuple[list[str], dict]:
    files: list[Path] = sorted((SRC / "humanbrain").rglob("*.py"))
    selected: list[Path] = []
    total = 0
    for path in files:
        selected.append(path)
        total += path.stat().st_size
        if total >= CODE_TARGET_BYTES:
            break
    texts = [path.read_text(encoding="utf-8") for path in selected]
    provenance = {
        "root": "src/humanbrain",
        "files_used": len(selected),
        "files_available": len(files),
        "total_bytes": sum(len(t.encode("utf-8")) for t in texts),
        "paths": [str(p.relative_to(REPO_ROOT)) for p in selected],
    }
    return texts, provenance


def evaluate(name: str, tokenizer, eval_text: str) -> dict:
    raw = eval_text.encode("utf-8")
    start = time.perf_counter()
    ids = tokenizer.encode(eval_text)
    elapsed = time.perf_counter() - start
    assert tokenizer.decode_bytes(ids) == raw, f"{name} failed losslessness on eval text"
    return {
        "tokenizer": name,
        "tokens": len(ids),
        "bytes": len(raw),
        "bytes_per_token": round(len(raw) / len(ids), 3),
        "encode_seconds": round(elapsed, 2),
    }


def fit_all(corpus_name: str, train_texts: list[str]) -> dict:
    print(f"[{corpus_name}] fitting base BytePhraseTokenizer...", flush=True)
    t0 = time.perf_counter()
    base = BytePhraseTokenizer.fit(train_texts)
    print(f"[{corpus_name}] base vocab={base.vocab_size} in {time.perf_counter()-t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    candidates = mine_concepts_detailed(train_texts, base)
    print(
        f"[{corpus_name}] mined {len(candidates)} concepts "
        f"(top count {candidates[0].count}) in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )
    stack = ConceptTokenizer(base, tuple(c.payload for c in candidates))

    t0 = time.perf_counter()
    bpe4096 = BpeTokenizer.fit(train_texts, vocab_size=4096)
    print(f"[{corpus_name}] BPE-4096 vocab={bpe4096.vocab_size} in {time.perf_counter()-t0:.1f}s", flush=True)
    bpe1280 = BpeTokenizer.fit(train_texts, vocab_size=1280)

    return {
        "base": base,
        "stack": stack,
        "bpe4096": bpe4096,
        "bpe1280": bpe1280,
        "candidates": candidates,
    }


def main() -> None:
    bible_train, bible_prov = load_bible_train()
    code_sample, code_prov = load_code_sample()
    bible_text = "\n".join(bible_train)
    code_text = "\n".join(code_sample)

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()

    fits = {}
    fits["bible"] = fit_all("bible", bible_train)
    fits["code"] = fit_all("code", code_sample)

    corpora = {"bible_train": bible_text, "code_sample": code_text}
    tokenizers_per_fit = {
        "bible": [
            ("BPE-4096", "bpe4096"),
            ("BPE-1280", "bpe1280"),
            ("BytePhrase-base", "base"),
            ("ConceptStack", "stack"),
        ],
        "code": [
            ("BPE-4096", "bpe4096"),
            ("BPE-1280", "bpe1280"),
            ("BytePhrase-base", "base"),
            ("ConceptStack", "stack"),
        ],
    }

    grid = []
    for fit_domain, entries in tokenizers_per_fit.items():
        for eval_domain, eval_text in corpora.items():
            for label, key in entries:
                row = evaluate(f"{fit_domain}/{label}", fits[fit_domain][key], eval_text)
                row.update({"fit_domain": fit_domain, "eval_domain": eval_domain})
                grid.append(row)
                print(f"  {fit_domain:>5} -> {eval_domain:<11} {label:<16} {row['tokens']:>8} tok  {row['bytes_per_token']:>7} B/tok", flush=True)

    def concept_rows(fit_domain: str, limit: int = 50) -> list[dict]:
        rows = []
        for candidate in fits[fit_domain]["candidates"][:limit]:
            payload = candidate.payload
            rows.append(
                {
                    "text": payload.decode("utf-8"),
                    "count": candidate.count,
                    "mi_bits": round(candidate.mutual_information_bits, 2),
                    "bytes": len(payload),
                }
            )
        return rows

    sensitivity_code = mine_concepts_detailed(
        code_sample, fits["code"]["base"], ConceptConfig(min_frequency=20)
    )

    results = {
        "study": "concept-tokens-v1",
        "git_head": git_head,
        "python": sys.version.split()[0],
        "protocol": {
            "concept_config": {
                "min_frequency": 50,
                "min_mutual_information_bits": 5.0,
                "max_concepts": 512,
                "min_tokens": 2,
                "max_tokens": 12,
            },
            "base_tokenizer": "BytePhraseTokenizer defaults (max_phrases=512); ids [0,768)",
            "concept_band": "[1024,1536)",
        },
        "provenance": {"bible": bible_prov, "code": code_prov},
        "corpora_bytes": {
            "bible_train": len(bible_text.encode("utf-8")),
            "code_sample": len(code_text.encode("utf-8")),
        },
        "compression_grid": grid,
        "concept_counts": {
            domain: len(fits[domain]["candidates"]) for domain in fits
        },
        "top50_concepts_bible": concept_rows("bible"),
        "top50_concepts_code": concept_rows("code"),
        "sensitivity": {
            "code_min_frequency_20": {
                "num_concepts": len(sensitivity_code),
                "top10": [c.payload.decode("utf-8") for c in sensitivity_code[:10]],
            }
        },
    }

    out_path = Path(__file__).resolve().parent / "results_v1.json"
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
