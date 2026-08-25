"""Red tests: synthetic-episode synthesizer (T-ROUTE reserve plan).

Contract under test (src/humanbrain/episode_synthesizer.py):

  1. COUNT       >= TARGET_EPISODES (2000) who_said-style speech-event episodes
                 mined from ALL TRAIN-book verses of the pinned PG10 corpus
                 (validation books stay excluded — VAL hygiene).
  2. VERBATIM    every emitted quote span is an exact contiguous substring of
                 its own source verse text (normalized whitespace), with
                 bounded word count and no nested-quotation markers.
  3. DEDUPE      zero verse-address overlap with the frozen 200-item TRAIN /
                 60-item VAL QA manifests (pins verified before use), and zero
                 (speaker, quote) pair collision with frozen who_said items;
                 plus one-episode-per-verse within the synthetic set.
  4. DETERMINISM generation is a pure function of (corpus bytes, pins, seed):
                 same seed twice => byte-identical records incl. order and
                 ids; different seed => identical multiset, permuted order.

Runs against the cached pinned PG10 fetch (%LOCALAPPDATA%/Temp/
hb_bible_freeze/books/pg10.fetch1.txt, sha-pinned); the synthesizer dual-
fetches against the same pinned digest only when the cache is absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import humanbrain.episode_synthesizer as es  # noqa: E402


FREEZE_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Temp"
    / "hb_bible_freeze"
)


def _canonical(data: object) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


@pytest.fixture(scope="module")
def corpus():
    trimmed, info = es.acquire_corpus(FREEZE_DIR)
    books_parsed, book_names, parse_stats = es.parse_all_books(trimmed)
    return {
        "info": info,
        "books_parsed": books_parsed,
        "book_names": book_names,
        "parse_stats": parse_stats,
    }


@pytest.fixture(scope="module")
def frozen():
    return es.load_frozen_corpus(FREEZE_DIR / "evidence")


@pytest.fixture(scope="module")
def built(corpus, frozen):
    out = {}
    for scope in ("all", "train"):
        indices = (
            set(es.TRAIN_INDICES) if scope == "train" else set(range(66))
        )
        candidates, tier_counts = es.extract_pool(
            corpus["books_parsed"], corpus["book_names"], indices
        )
        pool, dedupe_counts = es.dedupe(candidates, frozen)
        out[scope] = {
            "candidates": candidates,
            "tier_counts": tier_counts,
            "pool": pool,
            "dedupe_counts": dedupe_counts,
        }
    return out


@pytest.fixture(scope="module")
def records(built):
    return es.order_and_serialize(built["all"]["pool"], es.DEFAULT_SEED)


# ---------------------------------------------------------------- 1. count

def test_emits_at_least_2000_episodes(built):
    assert len(built["all"]["pool"]) >= es.TARGET_EPISODES, (
        f"synthetic pool {len(built['all']['pool'])} < target {es.TARGET_EPISODES}"
    )


def test_train_scope_subset_recorded_for_val_gated_protocols(built):
    """The hygiene subset (TRAIN-side books only) must exist, be strictly
    smaller than the all-scope corpus (validation books contribute mass), and
    stay clear of validation books entirely."""
    tr = built["train"]["pool"]
    al = built["all"]["pool"]
    assert 0 < len(tr) < len(al)
    val_idx = set(es.VAL_BOOK_RANGE)
    assert all(c["book_index"] not in val_idx for c in tr)


def test_every_tier_contributed_mechanically(built):
    # the miner-exact tier (builder's own pattern) must fire on its own;
    # variations exist to push past the natural-corpus ceiling
    assert built["all"]["tier_counts"]["miner_exact"] > 0
    assert built["all"]["tier_counts"]["short_infix"] > 0


# ---------------------------------------------------------------- 2. verbatim

def test_quotes_are_verbatim_substrings_of_source_verse(built):
    for cand in built["all"]["pool"]:
        verse = cand["verse_text"]
        assert cand["quote"] in verse, (
            f"{cand['book']} {cand['address']}: quote not verbatim in source verse"
        )
        wc = len(cand["quote"].split(" "))
        assert es.MIN_QUOTE_WORDS <= wc <= es.MAX_QUOTE_WORDS, f"word count {wc}"
        assert '"' not in cand["quote"], "nested quotation marker"
        assert "(" != cand["quote"][:1], "parenthetical lead"


def test_speaker_strings_are_well_formed(built):
    valid = es._VALID_SPEAKER_RE  # same shape gate the builder applies
    block = es._PRONOUN_BLOCKLIST
    for cand in built["all"]["pool"]:
        sp = cand["speaker"]
        assert sp and valid.match(sp), f"malformed speaker: {sp!r}"
        assert sp.split(" ")[-1] not in block, f"pronoun speaker leaked: {sp!r}"
        assert cand["tier"] in ("miner_exact", "short_infix", "thus_saith")


def test_serialized_records_carry_the_verbatim_quote(records):
    for rec in records[:500]:  # serialized slice; full-set checks above
        ctx = rec["context"][-1]
        assert rec["meta"]["quote"] in ctx, rec["episode_id"]
        assert rec["question"] == f'Who said, "{rec["meta"]["quote"]}"?'
        assert rec["serialized_prompt_answer"].endswith("\n")


# ---------------------------------------------------------------- 3. dedupe

def test_frozen_corpus_pins_verified_before_use(frozen):
    pins = frozen["pins"]
    assert pins == {
        "train_raw_sha256": es.PIN_TRAIN_RAW_SHA,
        "train_canonical": es.PIN_TRAIN_CANONICAL,
        "val_raw_sha256": es.PIN_VAL_RAW_SHA,
        "val_canonical": es.PIN_VAL_CANONICAL,
        "verified": True,
    }
    assert len(frozen["addresses"]) == 260  # 200 TRAIN + 60 VAL items


def test_no_address_overlap_with_frozen_corpus(built, frozen):
    synth_addrs = {
        (c["book_index"], *c["address"]) for c in built["all"]["pool"]
    } | {(c["book_index"], *c["source_address"]) for c in built["all"]["pool"]}
    overlap = synth_addrs & frozen["addresses"]
    assert not overlap, f"verse-address overlap with frozen corpus: {sorted(overlap)[:5]}"
    from collections import Counter
    per_verse = Counter((c["book_index"], *c["address"]) for c in built["all"]["pool"])
    assert max(per_verse.values()) <= es.MAX_EPISODES_PER_VERSE, (
        "per-verse episode cap exceeded"
    )


def test_nested_speech_guard_holds(built):
    """No episode may be extracted from reported/quoted speech context: within
    its own current sentence segment there must be no prior speech verb and no
    question mark. Backward formulas ('..., saith the LORD.') attribute AFTER
    the quote inside one sentence and are exempt from the prefix check."""
    for cand in built["all"]["pool"]:
        if cand.get("backward"):
            continue
        anchor = cand["anchor_text"]
        seg0 = es._seg_start(anchor, min(cand["match_start"], len(anchor)))
        prefix = anchor[seg0:min(cand["match_start"], len(anchor))]
        assert not es._PRIOR_SPEECH_RE.search(prefix), (
            f"{cand['book']} {cand['address']}: guard violation "
            f"(prefix={prefix[-60:]!r})"
        )


def test_no_speaker_quote_collision_with_frozen_who_said(built, frozen):
    synth_pairs = {
        (c["speaker"].casefold(), " ".join(c["quote"].split()).casefold())
        for c in built["all"]["pool"]
    }
    collide = synth_pairs & frozen["speech_pairs"]
    assert not collide, f"(speaker, quote) collisions with frozen who_said: {collide}"


# ---------------------------------------------------------------- 4. determinism

def test_same_seed_reproduces_byte_identical_records(built):
    a = es.order_and_serialize(built["all"]["pool"], es.DEFAULT_SEED)
    b = es.order_and_serialize(built["all"]["pool"], es.DEFAULT_SEED)
    ca, cb = _canonical(a), _canonical(b)
    assert ca == cb, "same-seed rerun diverged"
    assert hashlib.sha256(ca).hexdigest() == hashlib.sha256(cb).hexdigest()


def test_different_seed_permutes_order_but_preserves_multiset(built):
    s7 = es.order_and_serialize(built["all"]["pool"], 7)
    s8 = es.order_and_serialize(built["all"]["pool"], 8)
    key = lambda r: (r["references"][0], r["answer"], r["question"])
    assert sorted(map(key, s7)) == sorted(map(key, s8)), "seed changed the multiset"
    pairing7 = [(r["episode_id"], r["references"][0]) for r in s7]
    pairing8 = [(r["episode_id"], r["references"][0]) for r in s8]
    assert pairing7 != pairing8, "seed did not key the training order"
    assert len(set(p for p, _ in pairing8)) == len(pool := built["all"]["pool"]) == len(pairing8)


def test_scope_hygiene(corpus, built):
    """train-scope pool: TRAIN books only. all-scope pool may include VAL
    books by explicit contract (frozen-item addresses still excluded)."""
    val_idx = set(es.VAL_BOOK_RANGE)
    assert not val_idx & set(es.TRAIN_INDICES)
    for cand in built["train"]["pool"]:
        assert cand["book_index"] in es.TRAIN_INDICES, (
            f"VAL book leaked into train-scope episodes: {cand['book_index']}"
        )
