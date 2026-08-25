"""Synthetic who-said episode synthesizer (T-ROUTE reserve plan).

E1/E2 verdicts localized the routing failure to the N=200 who_said sample
(experiments/t_route/FINDINGS.md); the preregistered escalation ladder puts
synthetic-episode scaling FIRST if E3 passes. This module makes
verdict-to-campaign latency zero by pre-building the scaled training corpus
mechanically from the corpus builder's OWN narrative-pattern miner
(tools/build_bible_qa_corpus.py ``gen_who_said_candidates`` machinery), so the
episodes share the exact extraction semantics of the natural corpus.

What it does
------------
1. Acquires the pinned PG10 KJV source exactly like the builder: existing
   stable fetch pair reused when SHA-256 matches the pinned digest, otherwise
   dual sequential GET with equality enforcement (mismatch BLOCKS).
2. Parses ALL books with the builder's own TOC-anchored spans + inline C:V
   verse parser (identical normalized verse text as the frozen manifests).
3. Scans every TRAIN-book verse (validation books stay excluded - VAL hygiene;
   the frozen split rule is reused verbatim) with mechanical pattern tiers,
   all under nested-speech guards scoped to the CURRENT SENTENCE SEGMENT (any
   speech verb or question mark between the last sentence terminator and the
   event kills it - reported/quoted context), plus the builder's speaker-shape
   gate extended with a sentence-initial-junk blocklist:
     - ``miner_exact``  the builder's own SPEAKER_VERB_RE ("and X said,")
     - ``short_infix``  "<Name> answered and said," / "<Name> said unto X[,
       saying]," (also "to") / "<Name> answered <obj>," narrative idioms with
       bounded all-lowercase filler clauses (sentence punctuation cannot
       appear inside a match, so bindings never cross a sentence)
     - ``thus_saith``   closed formulas: "Thus saith the LORD[,;]", post-posed
       "saith the LORD[,;]", backward ", ..., saith the LORD." sentences, and
       "the word of the LORD came unto X, saying," - whitelisted speakers
   At most MAX_EPISODES_PER_VERSE (2) guarded events per verse.
4. Emits (verse_ref, speaker, quote_span) triples whose quote span is, by
   construction, a contiguous verbatim substring of its SOURCE verse text
   (edge trims and fixed filler-strips only; long speeches cut deterministically
   at [.?!;], then ':', then hard-truncated to MAX_QUOTE_WORDS), MIN..MAX words
   (two-word spans must carry a content word), no nested-quotation
   markers, no parenthetical leads. Verse-FINAL anchors
   ("... spake unto Moses, saying," at verse end) take their quote from the
   FIRST span of the following verse - the anchor address stays the reference;
   the source verse is recorded in meta.
5. Dedupes against the FROZEN corpora: BOTH anchor and source verse-addresses
   vs the 200-item TRAIN + 60-item VAL QA manifests (digest pins verified
   before use); identical (speaker, quote) at the same address collapsed;
   identical utterances at DIFFERENT addresses are kept - they are separate
   narrative speech acts with different passages.
6. Orders episodes with a seed-keyed shuffle (training order reproducible;
   multiset seed-independent) and serializes prompt-last with the builder's
   delimiters.

Determinism: generation is a pure function of (corpus bytes, pins, seed);
same seed twice reproduces byte-identical records including ids and order.

stdlib-only. No gates, no probes - pure corpus engineering.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------- pins

PIN_PG10_RAW_SHA = (
    "0204adaed1f25700aa854218cae63c7172228c41088f335e99167a071eed83c0"
)
PIN_TRAIN_RAW_SHA = (
    "d82fe615dced145d4a6d10d19e90583d1eca4bb6be79e638a96bebc511b0a0e8"
)
PIN_TRAIN_CANONICAL = (
    "d256c27dacec0cb2a169005611338c3408bbb252ba619dcde86fb01e3f8c9a7b"
)
PIN_VAL_RAW_SHA = (
    "4f9be9b725c630cbc0e94f8cea4a96413bc026895a3d1b5312882e1980e37dec"
)
PIN_VAL_CANONICAL = (
    "2771de48a90d9e694cc4260e9392d83c8e39f2aca7d57fe06dd51c7ce9a55bec"
)

SCHEMA_VERSION = "humanbrain/synthetic-episodes/v1"
DEFAULT_SEED = 7
TARGET_EPISODES = 2000
MIN_QUOTE_WORDS = 2
MAX_QUOTE_WORDS = 20

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO / "experiments" / "t_route" / "synthetic_episodes"


def default_freeze_dir() -> Path:
    local = os.environ.get(
        "LOCALAPPDATA", str(Path.home() / "AppData" / "Local")
    )
    return Path(local) / "Temp" / "hb_bible_freeze"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(value: object) -> bytes:
    """The builder's own canonicalization."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


_BUILDER_CACHE: list = []


def load_builder():
    """Import tools/build_bible_qa_corpus.py as the single source of truth for
    the corpus pins, the split rule, the verse parser, and the speech miner."""
    if _BUILDER_CACHE:
        return _BUILDER_CACHE[0]
    path = REPO / "tools" / "build_bible_qa_corpus.py"
    if not path.exists():
        raise RuntimeError(f"corpus builder not found at {path}")
    spec = importlib.util.spec_from_file_location(
        "hb_build_bible_qa_corpus", path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _BUILDER_CACHE.append(mod)
    return mod


BUILDER = load_builder()

TRAIN_INDICES = frozenset(set(range(66)) - set(BUILDER.VAL_BOOK_RANGE))
VAL_BOOK_RANGE = BUILDER.VAL_BOOK_RANGE
_VALID_SPEAKER_RE = BUILDER.VALID_SPEAKER_RE
_PRONOUN_BLOCKLIST = BUILDER.PRONOUN_SPEAKER_BLOCKLIST
_CANONICAL_KJV_VERSES = 31_102


# ------------------------------------------------------- corpus acquisition

def acquire_corpus(freeze_dir: Path | None = None) -> tuple[bytes, dict]:
    """Reuse the builder's cached stable fetch pair when pinned, else dual-fetch.

    Returns (trimmed_bytes, info) where info carries fetch_mode + source sha.
    """
    root = Path(freeze_dir) if freeze_dir else default_freeze_dir()
    books_dir, evidence_dir = root / "books", root / "evidence"
    books_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    f1 = books_dir / f"pg{BUILDER.GUTENBERG_ID}.fetch1.txt"
    f2 = books_dir / f"pg{BUILDER.GUTENBERG_ID}.fetch2.txt"
    if f1.exists() and f2.exists() and (
        sha256_bytes(f1.read_bytes())
        == sha256_bytes(f2.read_bytes())
        == PIN_PG10_RAW_SHA
    ):
        mode = "existing-stable-pair-reused"
    else:
        info = BUILDER.stage_fetch(
            books_dir, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "synthetic-episode-synthesizer"
        )
        mode = info.get("dual_fetch_mode", "fresh-dual-fetch")
    raw = f1.read_bytes()
    if sha256_bytes(raw) != PIN_PG10_RAW_SHA:
        raise RuntimeError(
            "PG10 raw bytes do not match the pinned SHA-256 - BLOCKS "
            f"(got {sha256_bytes(raw)}, want {PIN_PG10_RAW_SHA})"
        )
    trimmed = BUILDER.stage_trim(raw, evidence_dir)
    return trimmed, {
        "fetch_mode": mode,
        "sha256_raw": PIN_PG10_RAW_SHA,
        "size_trimmed_bytes": len(trimmed),
    }


def parse_all_books(
    trimmed: bytes,
) -> tuple[dict[int, list[dict]], list[str], dict]:
    """Builder-identical book spans + verse parsing for the whole Bible."""
    books, span_info = BUILDER.derive_book_spans(trimmed)
    books_parsed: dict[int, list[dict]] = {}
    anomaly_total: Counter = Counter()
    for b in books:
        vs, anom = BUILDER.parse_verses(trimmed[b["span"][0]: b["span"][1]])
        books_parsed[b["index"]] = vs
        for k, v in anom.items():
            anomaly_total[k] += v
    verse_total = sum(len(v) for v in books_parsed.values())
    train_verses = sum(len(books_parsed[i]) for i in sorted(TRAIN_INDICES))
    stats = {
        "verse_total_parsed": verse_total,
        "canonical_kjv_verses": _CANONICAL_KJV_VERSES,
        "delta_vs_canonical": verse_total - _CANONICAL_KJV_VERSES,
        "train_books_scanned": len(TRAIN_INDICES),
        "train_verses_scanned": train_verses,
        "val_book_indices_excluded": sorted(BUILDER.VAL_BOOK_RANGE),
        "anomalies": dict(anomaly_total),
    }
    return books_parsed, list(BUILDER.BOOK_SHORT_NAMES), stats


# ---------------------------------------------------------------- extraction

# Nested-speech guard, scoped to the current sentence segment: any prior
# speech verb or question mark means reported/quoted context.
_PRIOR_SPEECH_RE = re.compile(r"\b(?:said|answered|spake|saith|cried|replied)\b|\?")
_SENTENCE_END_RE = re.compile(r"[.?!;]")
_LEAD_CONJ_RE = re.compile(r"^(?:and|then|but|now|again|also|for)\s+", re.IGNORECASE)

_SPEAKER_ALT = (
    r"(?:The?\s+(?:Lord\s+GOD|Lord\s+JEHOVAH|Lord\s+God)"
    r"|Jesus\s+Christ|LORD|GOD|[A-Z][A-Za-z']*)"
)

# Tier B sub-patterns. Lowercase-only fillers make sentence-punctuation
# crossings impossible, so bindings never cross a sentence boundary.
_SHORT_INFIX_RE = re.compile(
    r"\b(?P<speaker>" + _SPEAKER_ALT + r")"
    r"(?P<infix>(?:\s+[a-z']+){0,4})"
    r"\s+(?P<verb>said|answered|spake|saith|cried|replied)(?=,)"
)
_UNTO_OBJ = r"(?:(?:him|her|them|me|thee|it|us|all)|(?:the\s+)?[A-Za-z']+(?:\s+[A-Za-z']+){0,5})"
_SAYING_UNTO_RE = re.compile(
    r"\b(?P<speaker>" + _SPEAKER_ALT + r")"
    r"\s+(?:said|spake|answered|testified|commanded|proclaimed|cried)"
    r"(?:\s+(?:moreover|again|also|yet)){0,2}"
    r"\s+(?:unto|to)\s+" + _UNTO_OBJ +
    r"(?:,\s+(?:(?:even|again)\s+)?saying)?(?=,)"
)
_ANSWERED_AND_SAID_RE = re.compile(
    r"\b(?P<speaker>" + _SPEAKER_ALT + r")"
    r"\s+answered"
    r"(?:\s+(?:unto\s+)?(?:him|her|them|me|thee|it|us"
    r"|[A-Z][A-Za-z']*|(?:the\s+)?(?:LORD|GOD)))?"
    r"(?:\s+[a-z']+){0,2}"
    r"\s+and\s+said(?=,)"
)
_COMMA_SAID_RE = re.compile(
    r"\b(?P<speaker>" + _SPEAKER_ALT + r")"
    r"(?:(?:\s+[a-z']+|\s*,\s*[a-z']+){0,5})"
    r",?\s+(?:and\s+|then\s+|now\s+|but\s+|also\s+|yet\s+)?"
    r"(?:said|cried|answered|spake|replied)(?=,)"
)
_ANSWERED_OBJ_RE = re.compile(
    r"\b(?P<speaker>" + _SPEAKER_ALT + r")\s+answered\s+"
    r"(?:him|her|them|me|thee|it|[A-Z][A-Za-z']*)"
    r"(?:,\s+saying)?(?=,)"
)

# Tier C sub-patterns: closed prophetic formulas, whitelisted speakers only.
_THUS_SAITH_RE = re.compile(
    r"\bThus\s+saith\s+(?P<speaker>the\s+Lord\s+(?:GOD|JEHOVAH|God)|the\s+LORD)"
    r"(?:\s+of\s+hosts)?(?:,\s*the\s+God\s+of\s+[^,]{1,25})?(?P<punct>[,;])"
)
_SAITH_POST_RE = re.compile(
    r"\bsaith\s+(?P<speaker>the\s+(?:LORD|Lord\s+GOD|Lord\s+JEHOVAH))"
    r"(?:\s+of\s+hosts)?(?=[,;])"
)
_BACKWARD_SAITH_RE = re.compile(
    r"(?P<pre>[^.;?!]{15,400}?),\s*saith\s+"
    r"(?:the\s+(?:LORD|Lord\s+GOD|Lord\s+JEHOVAH))(?:\s+of\s+hosts)?(?=[.;])"
)
_WORD_CAME_RE = re.compile(
    r"\b(?:Again\s+)?the\s+word\s+of\s+(?P<speaker>the\s+(?:LORD|God))\s+came\s+"
    r"(?:unto|to)\s+[A-Za-z']+(?:\s+[A-Za-z']+){0,2},\s*saying,"
)

_UNTO_STRIP = re.compile(r"^unto\s[^,]{1,40},\s*")
_SAYING_STRIP = re.compile(r"^(?:even\s+)?saying,\s*")

# Dialogue turns / duplicate suppression bounds.
MAX_EPISODES_PER_VERSE = 2

# Two-word quotes must carry at least one content word ("Fear not" in,
# "and said" out).
_TINY_STOPWORDS = frozenset(
    {"and", "but", "or", "then", "now", "for", "also", "yet", "unto",
     "saying", "behold", "from", "with", "that", "this", "again"}
)

# Sentence-initial tokens that can never BE a speaker when captured alone.
_JUNK_FIRST_TOKENS = frozenset(
    {"and", "but", "then", "now", "for", "also", "yet", "again", "behold",
     "when", "while", "after", "before", "if", "as", "so", "or", "nor",
     "thus", "lo", "hath", "is", "was", "it", "there", "these", "those",
     "this", "that", "he", "she", "they", "we", "ye", "thou", "thee",
     "him", "her", "them", "i"}
)


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _first_valid_speaker(raw_speaker: str) -> str | None:
    """Builder speaker-shape gate + sentence-initial junk purge."""
    raw = raw_speaker.strip()
    if not raw:
        return None
    if _VALID_SPEAKER_RE.match(raw):
        sp = raw  # wholesale fixed pair ("the LORD", "Lord GOD", single name)
    else:
        parts = raw.split(" ")
        if len(parts) > 1 and parts[0].lower() == "the":
            core = " ".join(parts[1:])
            if not _VALID_SPEAKER_RE.match(core):
                return None
            sp = f"the {core}"
        else:
            return None
    # Junk purge: a speaker made ONLY of sentence words ("And", "The",
    # "Thus") is never a person; proper names contain a non-junk token.
    if all(t.lower() in _JUNK_FIRST_TOKENS for t in sp.split(" ")):
        return None
    if not _VALID_SPEAKER_RE.match(sp):  # final assembled shape
        return None
    first_word = sp.split(" ")[-1]
    if (
        first_word in _PRONOUN_BLOCKLIST
        or len(first_word) < 2
        or _LEAD_CONJ_RE.match(sp)
    ):
        return None
    return sp


def _seg_start(text: str, pos: int) -> int:
    """Start index of the sentence segment containing pos."""
    seg0 = 0
    for m in _SENTENCE_END_RE.finditer(text, 0, pos):
        seg0 = m.end()
    return seg0


def _extract_quote(verse_text: str, tail_start: int) -> str | None:
    """Verbatim-guaranteed quote span after a speech event.

    Edge trims and fixed filler strips ('unto X,', '(even) saying,') only -
    the result is always a contiguous substring of verse_text. Long speeches
    fall back deterministically: primary cut at [.?!;]; if still >MAX words,
    cut at the first ':'; if still too long, hard-truncate to MAX words.
    """
    rest = verse_text[tail_start:].lstrip()
    if rest.startswith(","):
        rest = rest[1:].lstrip()
    for _ in range(2):
        st = _UNTO_STRIP.match(rest)
        if st:
            rest = rest[st.end():]
            continue
        st = _SAYING_STRIP.match(rest)
        if st:
            rest = rest[st.end():]
            continue
        break

    def _valid(excerpt: str) -> str | None:
        words = excerpt.split(" ")
        if not (MIN_QUOTE_WORDS <= len(words) <= MAX_QUOTE_WORDS):
            return None
        if len(words) == 2 and all(
            w.lower() in _TINY_STOPWORDS for w in words
        ):
            return None
        if any(w.startswith("(") for w in words):
            return None
        if '"' in excerpt:
            return None
        return excerpt

    cut = len(rest)
    for ch in ".?!;":
        p = rest.find(ch)
        if p != -1:
            cut = min(cut, p)
    excerpt = rest[:cut].strip().rstrip(",").strip()
    got = _valid(excerpt)
    if got is not None:
        return got
    if len(excerpt.split(" ")) > MAX_QUOTE_WORDS:
        p = rest.find(":")
        if p != -1 and p < cut:
            alt = rest[:p].strip().rstrip(",").strip()
            got = _valid(alt)
            if got is not None:
                return got
        truncated = " ".join(excerpt.split(" ")[:MAX_QUOTE_WORDS])
        return _valid(truncated.rstrip(","))
    return None


def _candidate_from_match(
    tier: str,
    verse_text: str,
    m: re.Match,
    speaker: str,
    book_index: int,
    short: str,
    verses: list[dict],
    vi: int,
    source_offset: int = 0,
) -> dict | None:
    quote = _extract_quote(verse_text, m.end())
    if quote is None:
        return None
    token = speaker.split(" ")[-1]
    dist = BUILDER._speaker_chapter_distance(verses, vi, token)
    src = verses[vi + source_offset] if source_offset else verses[vi]
    return {
        "family": "who_said",
        "book_index": book_index,
        "book": short,
        "address": (verses[vi]["chapter"], verses[vi]["verse"]),
        "source_address": (src["chapter"], src["verse"]),
        "speaker": speaker,
        "quote": quote,
        "verse_text": src["text"],
        "anchor_text": verses[vi]["text"],
        "tier": tier,
        "match_start": m.start(),
        "chapter_distance": dist,
        "speaker_token": token,
    }


def extract_candidates_for_book(
    book_index: int, short: str, verses: list[dict]
) -> list[dict]:
    """Guarded speech-event episodes for one book (<=2 per verse).

    Events are processed left to right; every event must independently pass
    the same-sentence nested-speech guard, the speaker-shape gate, and the
    verbatim-quote extraction. Verse-final attribution anchors take their
    quote from the opening span of the following verse.
    """
    out: list[dict] = []
    n = len(verses)
    for vi, v in enumerate(verses):
        text = v["text"]
        events = []
        for m in BUILDER.SPEAKER_VERB_RE.finditer(text):
            events.append(("miner_exact", m))
        for rx in (_SHORT_INFIX_RE, _SAYING_UNTO_RE, _ANSWERED_AND_SAID_RE,
                   _COMMA_SAID_RE, _ANSWERED_OBJ_RE):
            for m in rx.finditer(text):
                events.append(("short_infix", m))
        for rx in (_THUS_SAITH_RE, _SAITH_POST_RE, _WORD_CAME_RE):
            for m in rx.finditer(text):
                events.append(("thus_saith", m))
        if not events:
            continue
        made = 0
        for tier, m in sorted(events, key=lambda e: e[1].start()):
            if made >= MAX_EPISODES_PER_VERSE:
                break
            pre = text[_seg_start(text, m.start()):m.start()]
            if _PRIOR_SPEECH_RE.search(pre):
                continue  # reported/quoted speech context
            speaker = _first_valid_speaker(m.group("speaker"))
            if speaker is None:
                continue
            cand = _candidate_from_match(
                tier, text, m, speaker, book_index, short, verses, vi
            )
            if cand is None:
                continue
            out.append(cand)
            made += 1

        # Verse-final attribution anchor: quote lives in the next verse.
        if made < MAX_EPISODES_PER_VERSE and vi + 1 < n:
            nxt = verses[vi + 1]["text"]
            am = _VERSE_FINAL_ANCHOR_RE.search(text.rstrip())
            if am:
                pre = text[_seg_start(text, am.start()):am.start()]
                if not _PRIOR_SPEECH_RE.search(pre):
                    # reject if the NEXT verse opens its own attribution turn
                    if not re.match(
                            r"^\s*(?:And\s+|Then\s+|But\s+|Now\s+)?"
                            r"[A-Z][A-Za-z']*\s+"
                            r"(?:said|spake|answered)[^a-z]", nxt):
                        sp_raw = am.group("sp1") or am.group("sp2")
                        speaker = _first_valid_speaker(sp_raw)
                        if speaker is not None:
                            q = _extract_quote(nxt, 0)
                            if q is not None and '"' not in nxt[:40]:
                                cand = {
                                    "family": "who_said",
                                    "book_index": book_index,
                                    "book": short,
                                    "address": (v["chapter"], v["verse"]),
                                    "source_address": (
                                        verses[vi + 1]["chapter"],
                                        verses[vi + 1]["verse"],
                                    ),
                                    "speaker": speaker,
                                    "quote": q,
                                    "verse_text": nxt,
                                    "anchor_text": text,
                                    "tier": "thus_saith"
                                    if am.group("sp2") else "short_infix",
                                    "match_start": am.start(),
                                    "chapter_distance": None,
                                    "speaker_token": speaker.split(" ")[-1],
                                }
                                out.append(cand)
    return out


# A verse-final anchor: any validated attribution formula reaching the end of
# the verse text (its object/saying tail may consume the remainder).
_VERSE_FINAL_ANCHOR_RE = re.compile(
    r"(?:\b(?P<sp1>" + _SPEAKER_ALT + r")"
    r"(?:(?:\s+[a-z']+|\s*,\s*[a-z']+){0,4})?"
    r"\s+(?:said|spake|answered|cried|testified|commanded|proclaimed)"
    r"(?:\s+(?:moreover|again|also|yet))?"
    r"(?:\s+(?:unto|to)\s+(?:him|her|them|me|thee|it|us|all"
    r"|(?:the\s+)?[A-Za-z']+(?:\s+[A-Za-z']+){0,5}))?"
    r"(?:,\s+(?:(?:even\s+)?saying|thus))?"
    r"|(?P<sp2>the\s+(?:LORD|Lord\s+GOD|Lord\s+JEHOVAH)))"
    r"\s*,?\s*(?:even\s+)?\s*$"
)


def _backward_saith_candidates(
    book_index: int, short: str, verses: list[dict], limit_per_verse: int
) -> list[dict]:
    """"..., <sentence>, saith the LORD[.]" - the quoted sentence precedes."""
    out: list[dict] = []
    for vi, v in enumerate(verses):
        text = v["text"]
        taken = 0
        for m in _BACKWARD_SAITH_RE.finditer(text):
            if taken >= limit_per_verse:
                break
            excerpt = m.group("pre").strip().lstrip(",").strip()
            words = excerpt.split(" ")
            if not (MIN_QUOTE_WORDS <= len(words) <= MAX_QUOTE_WORDS):
                continue
            if '"' in excerpt or any(w.startswith("(") for w in words):
                continue
            speaker = _first_valid_speaker(m.groupdict().get("speaker") or "")
            if speaker is None:
                speaker = "the LORD"  # closed-formula whitelist
            out.append({
                "family": "who_said",
                "book_index": book_index,
                "book": short,
                "address": (v["chapter"], v["verse"]),
                "source_address": (v["chapter"], v["verse"]),
                "speaker": speaker,
                "quote": excerpt,
                "verse_text": text,
                "anchor_text": text,
                "tier": "thus_saith",
                "backward": True,
                "match_start": m.start("pre"),
                "chapter_distance": None,
                "speaker_token": "LORD",
            })
            taken += 1
    return out


def extract_pool(
    books_parsed: dict[int, list[dict]],
    book_names: list[str],
    train_indices: frozenset[int] | set[int] = TRAIN_INDICES,
) -> tuple[list[dict], dict[str, int]]:
    """Scan every TRAIN-book verse; canonically ordered candidate pool."""
    pool: list[dict] = []
    tier_counts: Counter = Counter()
    for bi in sorted(train_indices):
        cands = extract_candidates_for_book(bi, book_names[bi], books_parsed[bi])
        cands.extend(_backward_saith_candidates(
            bi, book_names[bi], books_parsed[bi], MAX_EPISODES_PER_VERSE))
        for c in cands:
            tier_counts[c["tier"]] += 1
        pool.extend(cands)
    return pool, dict(tier_counts)


# ------------------------------------------------------------ frozen corpus

_FROZEN_PINS = {
    "train_raw_sha256": PIN_TRAIN_RAW_SHA,
    "train_canonical": PIN_TRAIN_CANONICAL,
    "val_raw_sha256": PIN_VAL_RAW_SHA,
    "val_canonical": PIN_VAL_CANONICAL,
}

_FROZEN_QUESTION_RE = re.compile(r'^Who said, "(.+)"\?$')


def _verify_frozen_manifest(path: Path, pin_raw: str, pin_canonical: str) -> dict:
    data = path.read_bytes()
    if sha256_bytes(data) != pin_raw:
        raise RuntimeError(
            f"{path.name}: raw SHA-256 mismatch vs frozen pin - BLOCKS "
            f"(got {sha256_bytes(data)}, want {pin_raw})"
        )
    man = json.loads(data)
    body = {k: v for k, v in man.items() if k != "sha256_canonical"}
    recomputed = hashlib.sha256(compact_json(body)).hexdigest()
    if (
        man.get("sha256_canonical") != pin_canonical
        or recomputed != pin_canonical
    ):
        raise RuntimeError(
            f"{path.name}: embedded/recomputed canonical mismatch vs frozen "
            "pin - BLOCKS"
        )
    return man


def load_frozen_corpus(evidence_dir: Path | None = None) -> dict:
    """Load + pin-verify the frozen TRAIN/VAL QA manifests (dedupe contract)."""
    ev = Path(evidence_dir) if evidence_dir else default_freeze_dir() / "evidence"
    train = _verify_frozen_manifest(
        ev / "qa_manifest_train.json", PIN_TRAIN_RAW_SHA, PIN_TRAIN_CANONICAL
    )
    val = _verify_frozen_manifest(
        ev / "qa_manifest_val.json", PIN_VAL_RAW_SHA, PIN_VAL_CANONICAL
    )
    addresses: set[tuple[int, int, int]] = set()
    speech_pairs: set[tuple[str, str]] = set()
    for man in (train, val):
        for item in man["items"]:
            ref = item["references"][0]
            c, v = ref.rsplit(" ", 1)[1].split(":")
            addresses.add((item["meta"]["book_index"], int(c), int(v)))
            if item["family"] == "who_said":
                q = _FROZEN_QUESTION_RE.match(item["question"])
                if q:
                    speech_pairs.add(
                        (item["answer"].casefold(), _norm(q.group(1)))
                    )
    return {
        "addresses": addresses,
        "speech_pairs": speech_pairs,
        "pins": {**_FROZEN_PINS, "verified": True},
        "counts": {"train": train["count"], "val": val["count"]},
    }


# ------------------------------------------------------------------ dedupe

def dedupe(
    candidates: list[dict], frozen: dict
) -> tuple[list[dict], dict[str, int]]:
    """Per-verse cap, frozen-address (anchor AND source), and same-address
    pair dedupe. Identical (speaker, quote) at DIFFERENT addresses is kept:
    those are separate narrative speech acts with different passages.
    """
    counts = {
        "candidates": len(candidates),
        "per_verse_cap_enforced_removed": 0,
        "frozen_address_overlap_removed": 0,
        "same_address_duplicate_removed": 0,
    }
    pool: list[dict] = []
    seen_addr: Counter = Counter()
    seen_addr_pairs: set[tuple[tuple[int, int, int], tuple[str, str]]] = set()
    for c in candidates:
        addr = (c["book_index"], *c["address"])
        src_addr = (c["book_index"], *c["source_address"])
        pair = (c["speaker"].casefold(), _norm(c["quote"]))
        if seen_addr[addr] >= MAX_EPISODES_PER_VERSE:
            counts["per_verse_cap_enforced_removed"] += 1
            continue
        if addr in frozen["addresses"] or src_addr in frozen["addresses"]:
            counts["frozen_address_overlap_removed"] += 1
            continue
        key = (addr, pair)
        if key in seen_addr_pairs:
            counts["same_address_duplicate_removed"] += 1
            continue
        seen_addr[addr] += 1
        seen_addr_pairs.add(key)
        pool.append(c)
    counts["emitted"] = len(pool)
    return pool, counts


# ------------------------------------------------------- ordering + records

def order_and_serialize(pool: list[dict], seed: int = DEFAULT_SEED) -> list[dict]:
    """Seed-keyed training order + prompt-last serialization (builder rules).

    Same seed => byte-identical records (ids and order included); the
    underlying multiset is identical for every seed.
    """
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    records: list[dict] = []
    for n, i in enumerate(order):
        c = pool[i]
        ref = f"{c['book']} {c['address'][0]}:{c['address'][1]}"
        context = [f"{ref} {c['anchor_text']}"]
        if c["source_address"] != c["address"]:
            src_ref = (
                f"{c['book']} {c['source_address'][0]}:{c['source_address'][1]}"
            )
            context.append(f"{src_ref} {c['verse_text']}")
        question = f'Who said, "{c["quote"]}"?'
        item = {
            "family": "who_said",
            "context": context,
            "question": question,
            "answer": c["speaker"],
        }
        records.append({
            "episode_id": f"syn-{n:06d}",
            "family": "who_said",
            "cohort": "SYNTHETIC",
            "references": [ref],
            "context": context,
            "question": question,
            "answer": c["speaker"],
            "serialized_prompt_answer": BUILDER.serialize_item(item),
            "meta": {
                "book": c["book"],
                "book_index": c["book_index"],
                "address": [c["address"][0], c["address"][1]],
                "source_address": [
                    c["source_address"][0], c["source_address"][1]
                ],
                "pattern_tier": c["tier"],
                "quote": c["quote"],
                "quote_words": len(c["quote"].split(" ")),
                "speaker_token": c["speaker_token"],
                "chapter_distance": c["chapter_distance"],
                "seed": seed,
            },
        })
    return records


# ------------------------------------------------------------- full builds

def generate_episodes(
    freeze_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    scope: str = "all",
) -> tuple[list[dict], dict]:
    """Full in-memory pipeline: acquire -> parse -> scan -> dedupe -> order.

    scope="all" scans every parsed verse of all 66 books (minus frozen-item
    addresses); scope="train" restricts mining to TRAIN-side books for
    protocols whose evaluation gates hold out the validation books. Returns
    (records, report).
    """
    trimmed, src_info = acquire_corpus(freeze_dir)
    books_parsed, book_names, parse_stats = parse_all_books(trimmed)
    if scope == "train":
        scan_indices: set[int] = set(TRAIN_INDICES)
    elif scope == "all":
        scan_indices = set(range(66))
    else:
        raise ValueError(f"unknown scope: {scope}")
    candidates, tier_counts = extract_pool(books_parsed, book_names, scan_indices)
    frozen = load_frozen_corpus((freeze_dir or default_freeze_dir()) / "evidence")
    pool, dedupe_counts = dedupe(candidates, frozen)
    records = order_and_serialize(pool, seed)

    speakers = Counter(_norm(r["answer"]) for r in records)
    qlens = Counter(r["meta"]["quote_words"] for r in records)
    val_idx = set(BUILDER.VAL_BOOK_RANGE)
    checks = {
        "count_ge_target": len(pool) >= TARGET_EPISODES,
        "quotes_verbatim_by_construction": all(
            r["meta"]["quote"] in r["context"][-1] for r in records
        ),
        "max_two_episodes_per_anchor_verse": all(
            n <= MAX_EPISODES_PER_VERSE
            for n in Counter(
                (r["meta"]["book_index"], *r["meta"]["address"])
                for r in records
            ).values()
        ),
        "frozen_address_disjoint": not (
            {
                (r["meta"]["book_index"], *r["meta"]["address"])
                for r in records
            }
            | {
                (r["meta"]["book_index"], *r["meta"]["source_address"])
                for r in records
            }
        ) & frozen["addresses"],
        "frozen_pins_verified": frozen["pins"]["verified"],
        "scope_respected": (
            all(r["meta"]["book_index"] not in val_idx for r in records)
            if scope == "train"
            else True
        ),
    }
    report = {
        "scope": scope,
        "source": src_info,
        "parse_stats": parse_stats,
        "tier_counts": tier_counts,
        "dedupe_counts": dedupe_counts,
        "frozen_counts": frozen["counts"],
        "speakers_top20": speakers.most_common(20),
        "unique_speakers": len(speakers),
        "quote_words_histogram": {
            str(k): qlens[k] for k in sorted(qlens)
        },
        "mean_quote_words": round(
            sum(k * v for k, v in qlens.items()) / max(1, sum(qlens.values())), 3
        ),
        "checks": checks,
        "emitted": len(records),
    }
    return records, report


def materialize(
    out_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
    allow_shortfall: bool = False,
    freeze_dir: Path | None = None,
) -> dict:
    """Run the pipeline and write episodes files + manifest.json atomically.

    Writes BOTH scopes:
      - ``episodes.jsonl``       scope="all"  (the task brief: all 31,102
        verses, frozen-item addresses excluded) - the >=2000 corpus;
      - ``episodes_train.jsonl`` scope="train" (TRAIN-side books only) -
        the hygiene subset for any protocol that gates on the held-out
        validation books.
    """
    started = utcnow()
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"synthesizer starting: seed={seed} out={out_dir}")

    recs_by_scope: dict[str, list[dict]] = {}
    reports: dict[str, dict] = {}
    for scope in ("all", "train"):
        records, report = generate_episodes(freeze_dir, seed, scope=scope)
        recs_by_scope[scope] = records
        reports[scope] = report
        log(f"scope={scope}: {report['emitted']} episodes")
    report = reports["all"]
    records = recs_by_scope["all"]
    if not report["checks"]["count_ge_target"] and not allow_shortfall:
        raise RuntimeError(
            f"emitted {report['emitted']} < target {TARGET_EPISODES}; "
            "rerun with --allow-shortfall to accept a shortfall"
        )

    def _write(name: str, recs: list[dict]) -> tuple[str, bytes]:
        data = "".join(
            compact_json(rec).decode("utf-8") + "\n" for rec in recs
        ).encode("utf-8")
        path = out_dir / name
        tmp = out_dir / (name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return name, data

    all_name, all_bytes = _write("episodes.jsonl", records)
    tr_name, tr_bytes = _write("episodes_train.jsonl", recs_by_scope["train"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": started,
        "purpose": (
            "T-ROUTE reserve plan: mechanically scale who_said training "
            "episodes past the N=200 sample limit; ready if E3 passes"
        ),
        "builder_tool": "tools/build_bible_qa_corpus.py (imported verbatim)",
        "synthesizer": "src/humanbrain/episode_synthesizer.py",
        "seed": seed,
        "source": {
            **report["source"],
            "gutenberg_id": BUILDER.GUTENBERG_ID,
        },
        "parse_stats": report["parse_stats"],
        "scopes": {
            "all": {
                "file": all_name,
                "sha256": sha256_bytes(all_bytes),
                "emitted": reports["all"]["emitted"],
                "note": (
                    "all 66 books minus frozen-item addresses; CONTAMINATES "
                    "any evaluation that gates on held-out validation books"
                ),
            },
            "train": {
                "file": tr_name,
                "sha256": sha256_bytes(tr_bytes),
                "emitted": reports["train"]["emitted"],
                "note": (
                    "TRAIN-side books only (builder split rule); use this "
                    "subset when E3-style gates score the validation cohort"
                ),
            },
        },
        "hygiene_warning": (
            "episodes.jsonl includes validation-book verses; if E3's "
            "preregistered gates include VAL-cohort scoring, train from "
            "episodes_train.jsonl instead"
        ),
        "frozen_corpus": {
            "counts": report["frozen_counts"],
            "digest_pins": _FROZEN_PINS,
            "dedupe": (
                "anchor+source verse-address exclusion vs TRAIN and VAL "
                "items; same-address pair collapse; cross-address repeats "
                "kept (separate speech acts)"
            ),
        },
        "extraction": {
            "tiers": ["miner_exact", "short_infix", "thus_saith"],
            "tier_candidates_all_scope": report["tier_counts"],
            "max_episodes_per_anchor_verse": MAX_EPISODES_PER_VERSE,
            "sub_patterns": [
                "miner_exact: builder SPEAKER_VERB_RE verbatim",
                "short_infix: '<Name> answered and said,' / '<Name> said "
                "unto|to <X>[, saying],' / '<Name> [lowercase clause][, and] "
                "said,' / '<Name> answered <obj>,'",
                "thus_saith: 'Thus saith the LORD[,;]' / post-posed 'saith "
                "the LORD[,;]' / backward ', ..., saith the LORD.' / 'the "
                "word of the LORD came unto X, saying,'",
                "verse-final anchors: attribution formula at verse end takes "
                "its quote from the opening span of the following verse",
            ],
            "guards": [
                "nested-speech guard scoped to the current sentence segment: "
                "prior speech verb / question mark => event skipped",
                f"at most {MAX_EPISODES_PER_VERSE} guarded events per verse",
                "pronoun/conjunction-speaker purge + builder speaker-shape gate",
                f"quote {MIN_QUOTE_WORDS}..{MAX_QUOTE_WORDS} words, no nested "
                "quotes, no parenthetical leads",
            ],
            "verbatim_rule": (
                "quote is a contiguous substring of its source verse text; "
                "edge trims and 'unto X,'/'saying,' filler strips only; long "
                "speeches cut at [.?!;], then ':', then hard-truncated to 20 "
                "words"
            ),
        },
        "counts": {
            **report["dedupe_counts"],
            "frozen_train_items": report["frozen_counts"]["train"],
            "frozen_val_items": report["frozen_counts"]["val"],
            "emitted_all_scope": reports["all"]["emitted"],
            "emitted_train_scope": reports["train"]["emitted"],
            "target": TARGET_EPISODES,
        },
        "stats": {
            "unique_speakers": report["unique_speakers"],
            "speakers_top20": report["speakers_top20"],
            "quote_words_histogram": report["quote_words_histogram"],
            "mean_quote_words": report["mean_quote_words"],
        },
        "checks": {
            **{f"all/{k}": v for k, v in reports["all"]["checks"].items()},
            **{
                f"train/{k}": v
                for k, v in reports["train"]["checks"].items()
                if k != "count_ge_target"
            },
        },
        "episodes_file": all_name,
        "episodes_sha256": sha256_bytes(all_bytes),
    }
    manifest["sha256_content"] = hashlib.sha256(
        compact_json(
            {k: v for k, v in manifest.items() if k != "sha256_content"}
        )
    ).hexdigest()
    manifest_path = out_dir / "manifest.json"
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_bytes(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False).encode()
    )
    tmp.replace(manifest_path)

    log(
        f"done: all={reports['all']['emitted']} train={reports['train']['emitted']}"
        f" ({report['unique_speakers']} unique speakers in all-scope); "
        f"checks={'PASS' if all(report['checks'].values()) else 'FAIL'}"
    )
    return manifest


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
            capture_output=True, text=True, timeout=15,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    global TARGET_EPISODES
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--target", type=int, default=TARGET_EPISODES,
                    help="documented floor; changing it edits the contract")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--freeze-dir", type=Path, default=None)
    ap.add_argument("--allow-shortfall", action="store_true")
    args = ap.parse_args(argv)
    TARGET_EPISODES = args.target
    manifest = materialize(
        out_dir=args.out_dir, seed=args.seed,
        allow_shortfall=args.allow_shortfall, freeze_dir=args.freeze_dir,
    )
    print(json.dumps({"emitted": manifest["counts"]["emitted"],
                      "manifest_sha256_content": manifest["sha256_content"]}))
    return 0 if all(manifest["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
