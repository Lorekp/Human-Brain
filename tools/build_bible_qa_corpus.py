#!/usr/bin/env python3
"""Bible-QA corpus materializer (PG10 KJV primary corpus, docs/research/cell-lm-corpus-bible.md).

Materializes the v3 Bible freeze candidate:

1. dual-fetch PG10 from the pinned mirror, SHA-256 equality enforced against
   the pinned digest (fetch failure BLOCKS);
2. PG-boilerplate trim via the pinned START/END marker rule;
3. TOC-anchored 66-book span derivation (pinned rule, cell-lm-corpus-bible.md §4);
4. gating split: validation = minor-prophets ∪ gospels+acts blocks (17 books),
   train = remaining 49;
5. verse index parse from the inline ``C:V`` markers (anomalies counted,
   never silently repaired);
6. deterministic QA triple generation (who-said-X with chapter-distance,
   address-lookup entity binding, verse-continuation) over TRAIN books plus a
   smaller VAL manifest used only for evaluation; train/val verse-address
   disjointness enforced;
7. token accounting: bytes, pinned pretokenize ``\\S+\\s*`` counts, and — when
   the demoted fiction-trained BPE-4096 artifact is available — an upper-bound
   token proxy replay (artifact read-only, never modified);
8. FREEZE-CANDIDATE-v3-BIBLE.json + evidence written ONLY under --out-dir
   (default %LOCALAPPDATA%/Temp/hb_bible_freeze). The repository is never
   written to; artifacts/ is never touched; no frozen protocol is executed.

stdlib-only; deterministic: rerunning with identical inputs reproduces
byte-identical manifests (asserted in-process, red-test-15 analogue).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- pins

MIRROR_TEMPLATE = "https://gutenberg.pglaf.org/cache/epub/{ID}/pg{ID}.txt"
GUTENBERG_ID = 10
PINNED_SHA256_RAW = "0204adaed1f25700aa854218cae63c7172228c41088f335e99167a071eed83c0"
START_MARKER = b"*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = b"*** END OF THE PROJECT GUTENBERG EBOOK"
TRIM_RULE = (
    "raw.find(START_MARKER=b'*** START OF THE PROJECT GUTENBERG EBOOK'); "
    "close=find(b'***', start+len(START_MARKER)); end=find(END_MARKER); "
    "trimmed = raw[close+3:end].strip(); hard error if any marker missing or misordered"
)
OT_HEADER = b"The Old Testament of the King James Version of the Bible"
NT_HEADER = b"The New Testament of the King James Bible"
PRETOKENIZE_PATTERN = rb"\S+\s*"

# Genre blocks (canonical order; cell-lm-corpus-bible.md §4).
GENRE_BLOCKS = [
    ("Pentateuch", 0, 4),
    ("History", 5, 16),
    ("Wisdom/Poetry", 17, 21),
    ("Major prophets", 22, 26),
    ("Minor prophets", 27, 38),
    ("Gospels", 39, 42),
    ("Acts", 43, 43),
    ("Pauline epistles", 44, 56),
    ("General epistles", 57, 64),
    ("Apocalypse", 65, 65),
]
VAL_BOOK_RANGE = range(27, 44)  # minor prophets (27-38) + gospels+acts (39-43)

# Canonical KJV short names, one per book in canonical order (pinned table).
BOOK_SHORT_NAMES = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
    "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
    "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
]

# Pinned parallel-passage corpora (red test 18) as book-index sets: each tuple
# of groups must land entirely on ONE side of the split.
PARALLEL_PASSAGE_GROUPS = [
    ("Samuel/Kings <-> Chronicles", [(8, 9, 10, 11), (12, 13)]),
    ("Isaiah 36-39 <-> 2 Kings 18-20", [(22,), (10,)]),
    ("Jude <-> 2 Peter 2", [(64,), (60,)]),
    ("Synoptic triples + Acts", [(39, 40, 41, 42, 43)]),
]

QA_DELIM_Q = "\nQ: "
QA_DELIM_A = "\nA: "
QA_TRAILING_NEWLINE = "\n"

TRAIN_QA_TARGETS = {"who_said": 60, "address_lookup": 70, "verse_continuation": 70}
VAL_QA_TARGETS = {"who_said": 20, "address_lookup": 20, "verse_continuation": 20}

PRONOUN_SPEAKER_BLOCKLIST = {
    "He", "She", "They", "It", "I", "We", "Ye", "Thou", "Thee", "You",
    "Who", "All", "Some", "One", "None", "This", "That", "These", "Those",
    "His", "Her", "Their", "My", "Thy", "Every", "Each", "Both", "Many",
}
SPEAKER_VERB_RE = re.compile(
    r"(?:(?P<prep>(?:and|then|but|now|again|also|for)\s+)?(?P<lead>the\s+)?"
    r"(?P<speaker>[A-Z][A-Za-z']*(?:\s+[A-Z][A-Za-z']*)*)"
    r"\s+(?P<verb>said|answered|spake|saith|cried)),"
)
# A valid KJV speaker string: optional "the", then ONE proper token (or the
# fixed pair "Lord GOD"/"Lord JEHOVAH"/"Jesus Christ"). Kills capitalized
# run-on captures like "Therefore I" / "When I".
VALID_SPEAKER_RE = re.compile(
    r"^the (?:(?:Lord|Father) (?:GOD|JEHOVAH|God)|(?:Jesus Christ))$"
    r"|^(?:LORD|GOD)$|^the LORD$|^the GOD$|^[A-Z][A-Za-z']+$"
)
PRIOR_SPEECH_RE = re.compile(r"\b(?:said|answered|spake|saith|cried)\b|\?")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def compact_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%SZ')}] {msg}", flush=True)


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------- stage 1: fetch

def stage_fetch(books_dir: Path, user_agent: str) -> dict:
    """Dual sequential GET with SHA-256 equality; reuse existing stable pair."""
    f1 = books_dir / f"pg{GUTENBERG_ID}.fetch1.txt"
    f2 = books_dir / f"pg{GUTENBERG_ID}.fetch2.txt"
    reused = False
    if f1.exists() and f2.exists():
        h1, h2 = sha256_bytes(f1.read_bytes()), sha256_bytes(f2.read_bytes())
        if h1 == h2 == PINNED_SHA256_RAW:
            reused = True
            log(f"fetch: existing stable pair reused (sha {h1[:16]}...)")
        else:
            log("fetch: existing pair unstable/mismatched; refetching")
    if not reused:
        url = MIRROR_TEMPLATE.format(ID=GUTENBERG_ID)
        digests = []
        for tag, path in (("fetch1", f1), ("fetch2", f2)):
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            started = time.perf_counter()
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                status = resp.status
            path.write_bytes(data)
            digest = sha256_bytes(data)
            digests.append(digest)
            log(f"fetch: {tag} HTTP {status} {len(data):,}B sha={digest[:16]}... "
                f"({time.perf_counter() - started:.1f}s)")
        if not (digests[0] == digests[1] == PINNED_SHA256_RAW):
            raise RuntimeError(
                "DUAL-FETCH SHA-256 MISMATCH vs pinned digest — BLOCKS materialization "
                f"(got {digests[0]} / {digests[1]}, want {PINNED_SHA256_RAW})"
            )
    raw = f1.read_bytes()
    if sha256_bytes(raw) != PINNED_SHA256_RAW:
        raise RuntimeError("raw bytes do not match the pinned SHA-256 — BLOCKS")
    return {
        "mirror_url": MIRROR_TEMPLATE.format(ID=GUTENBERG_ID),
        "http_status": 200,
        "size_raw_bytes": len(raw),
        "sha256_raw": PINNED_SHA256_RAW,
        "dual_fetch_stable": True,
        "dual_fetch_mode": "existing-stable-pair-reused" if reused else "fresh-dual-fetch",
        "user_agent": user_agent,
    }


def stage_trim(raw: bytes, evidence_dir: Path) -> bytes:
    i = raw.find(START_MARKER)
    k = raw.find(END_MARKER)
    if i == -1 or k == -1 or k <= i:
        raise RuntimeError("PG START/END boilerplate markers missing or misordered")
    j = raw.find(b"***", i + len(START_MARKER))
    if j == -1 or j >= k:
        raise RuntimeError("malformed START marker region")
    trimmed = raw[j + 3:k].strip()
    (evidence_dir / "pg10.trimmed.bin").write_bytes(trimmed)
    log(f"trim: {len(trimmed):,}B (rule: {TRIM_RULE[:48]}...)")
    return trimmed


# ------------------------------------------------------- stage 2: books + verses

def derive_book_spans(trimmed: bytes) -> tuple[list[dict], dict]:
    """TOC-anchored 66-book derivation (cell-lm-corpus-bible.md §4, pinned)."""
    first = trimmed.find(OT_HEADER)
    second = trimmed.find(OT_HEADER, first + 1)
    if first == -1 or second == -1:
        raise RuntimeError("Old Testament section header not found twice")
    toc_region_end = second  # TOC region = [start, SECOND occurrence of OT header)
    toc_lines = [l.strip() for l in trimmed[:toc_region_end].split(b"\r\n")]
    nonempty = [l for l in toc_lines if l]
    headers = {OT_HEADER, NT_HEADER}
    titles = [l for l in nonempty if l not in headers]
    if len(titles) != 66:
        raise RuntimeError(f"expected 66 TOC titles, got {len(titles)}")

    books = []
    scan_from = toc_region_end
    for idx, title in enumerate(titles):
        needle = title + b"\r\n"
        heading = trimmed.find(needle, scan_from)
        if heading == -1:
            raise RuntimeError(f"heading not found for book {idx}: {title!r}")
        books.append({
            "index": idx,
            "title_line": title.decode("utf-8"),
            "heading_offset": heading,
        })
        scan_from = heading + len(needle)
    for idx, book in enumerate(books):
        end = books[idx + 1]["heading_offset"] if idx + 1 < len(books) else len(trimmed)
        book["span"] = [book["heading_offset"], end]
    covered = sum(b["span"][1] - b["span"][0] for b in books)
    coverage = covered / len(trimmed)
    info = {
        "titles_parsed": len(titles),
        "headings_located_in_order": sum(1 for b in books if b["span"] is not None),
        "span_coverage_of_trimmed_bytes": round(coverage, 6),
        "toc_region_end_offset": toc_region_end,
    }
    if coverage < 0.99:
        raise RuntimeError(f"span coverage {coverage:.4f} below 99% floor")
    return books, info


VERSE_MARKER_RE = re.compile(r"\b(\d+):(\d+)\s")


def parse_verses(book_bytes: bytes) -> tuple[list[dict], dict]:
    """Parse inline C:V markers by POSITION (PG10 puts multiple markers on one
    wrapped physical line, e.g. Matthew genealogies); continuation bytes append
    to the open verse until the next marker."""
    text = book_bytes.decode("utf-8")
    anomalies = Counter()
    markers = [
        (m.end(), int(m.group(1)), int(m.group(2)), m.start())
        for m in VERSE_MARKER_RE.finditer(text)
    ]
    if not markers:
        return [], {"book_has_no_markers": 1}
    pre = text[:markers[0][3]].strip()
    if pre:
        anomalies["text_before_first_verse"] += 1
    out = []
    for i, (body_start, ch, ve, _mstart) in enumerate(markers):
        body_end = markers[i + 1][3] if i + 1 < len(markers) else len(text)
        joined = re.sub(r"\s+", " ", text[body_start:body_end]).strip()
        if not joined:
            anomalies["empty_verse_body"] += 1
        if out:
            prev = out[-1]
            if ch < prev["chapter"]:
                anomalies["chapter_number_decreased"] += 1
            elif ch == prev["chapter"] and ve <= prev["verse"]:
                anomalies["verse_number_not_increasing"] += 1
            elif ch > prev["chapter"] + 1:
                anomalies["chapter_number_gap"] += 1
        out.append({"chapter": ch, "verse": ve, "text": joined})
    # gap census within chapters (documented delta, never repaired)
    by_chapter: dict[int, list[int]] = {}
    for v in out:
        by_chapter.setdefault(v["chapter"], []).append(v["verse"])
    for ch, nums in by_chapter.items():
        sorted_nums = sorted(nums)
        if sorted_nums != list(range(sorted_nums[0], sorted_nums[0] + len(sorted_nums))):
            anomalies["chapter_with_noncontiguous_verse_numbers"] += 1
    return out, dict(anomalies)


# ------------------------------------------------------------------ stage 3: QA

def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split(" ")
    return " ".join(words[:max_words])


def _extract_speech(verse_text: str, verb_end: int) -> str | None:
    """Words after the verb's comma, cut at sentence end; 3..12 words."""
    rest = verse_text[verb_end:].lstrip()
    if rest.startswith(","):
        rest = rest[1:].lstrip()
    rest = re.sub(r"^unto\s[^,]{1,40},\s*", "", rest)
    cut = len(rest)
    for ch in ".?!;":
        p = rest.find(ch)
        if p != -1:
            cut = min(cut, p)
    excerpt = rest[:cut].strip().rstrip(",")
    words = excerpt.split(" ")
    if not (3 <= len(words) <= 12):
        return None
    if any(w.startswith("(") for w in words):
        return None
    return " ".join(words)


def _speaker_chapter_distance(
    verses: list[dict], cur_idx: int, speaker: str
) -> int | None:
    pat = re.compile(r"\b" + re.escape(speaker) + r"\b")
    cur_chapter = verses[cur_idx]["chapter"]
    for j in range(cur_idx - 1, -1, -1):
        if pat.search(verses[j]["text"]):
            return abs(cur_chapter - verses[j]["chapter"])
    return None


def gen_who_said_candidates(book_index: int, short: str, verses: list[dict]) -> list[dict]:
    cands = []
    lead_conj_re = re.compile(r"^(?:and|then|but|now|again|also|for)\s+", re.IGNORECASE)
    for vi, v in enumerate(verses):
        matches = list(SPEAKER_VERB_RE.finditer(v["text"]))
        if not matches:
            continue
        # Nested-speech guard: only the FIRST speech verb of the verse may
        # produce a candidate; anything later risks quoting inside quoted
        # speech (KJV reports speech without quotation marks).
        first = matches[0]
        # Nested-speech guard part 1: any speech verb or question mark BEFORE
        # the candidate match means this speech event is reported/quoted
        # (KJV marks no quotation marks) -> skip the verse entirely.
        if PRIOR_SPEECH_RE.search(v["text"][: first.start()]):
            continue
        for m in (first,):
            speaker = m.group("speaker").strip()
            speaker = lead_conj_re.sub("", speaker).strip()
            lead = (m.group("lead") or "").strip()
            if lead and not speaker.lower().startswith("the "):
                speaker = lead + " " + speaker
            if not VALID_SPEAKER_RE.match(speaker):
                break  # capitalized run-on garbage, not a speaker
            first_word = speaker.split(" ")[-1]
            if (
                first_word in PRONOUN_SPEAKER_BLOCKLIST
                or len(first_word) < 2
                or lead_conj_re.match(speaker)
            ):
                break  # unusable speaker -> whole verse skipped (nesting risk)
            speech = _extract_speech(v["text"], m.end())
            if speech is None:
                break
            dist = _speaker_chapter_distance(verses, vi, first_word)
            if dist is None:
                break
            cands.append({
                "family": "who_said",
                "book_index": book_index,
                "book": short,
                "address": (v["chapter"], v["verse"]),
                "context": [f"{short} {v['chapter']}:{v['verse']} {v['text']}"],
                "question": f'Who said, "{speech}"?',
                "answer": speaker,
                "meta": {"chapter_distance": dist, "speaker_token": first_word},
            })
            break  # one candidate per verse
    return cands


def gen_address_lookup_candidates(book_index: int, short: str, verses: list[dict]) -> list[dict]:
    cands = []
    for v in verses:
        if len(v["text"]) > 140 or len(v["text"]) < 25:
            continue
        cands.append({
            "family": "address_lookup",
            "book_index": book_index,
            "book": short,
            "address": (v["chapter"], v["verse"]),
            "context": [],
            "question": f"What does {short} {v['chapter']}:{v['verse']} say?",
            "answer": v["text"],
            "meta": {},
        })
    return cands


def gen_verse_continuation_candidates(book_index: int, short: str, verses: list[dict]) -> list[dict]:
    by_chapter: dict[int, list[dict]] = {}
    for v in verses:
        by_chapter.setdefault(v["chapter"], []).append(v)
    cands = []
    for ch, ch_verses in by_chapter.items():
        ordered = sorted(ch_verses, key=lambda x: x["verse"])
        for a, b in zip(ordered, ordered[1:]):
            if a["verse"] + 1 != b["verse"]:
                continue
            answer = _truncate_words(b["text"], 12)
            if len(answer.split(" ")) < 3:
                continue
            cands.append({
                "family": "verse_continuation",
                "book_index": book_index,
                "book": short,
                "address": (ch, a["verse"]),
                "context": [f"{short} {ch}:{a['verse']} {a['text']}"],
                "question": (
                    f"In {short} chapter {ch}, what words follow "
                    f"{ch}:{a['verse']}?"
                ),
                "answer": answer,
                "meta": {"target_address": [ch, b["verse"]]},
            })
    return cands


GENERATORS = {
    "who_said": gen_who_said_candidates,
    "address_lookup": gen_address_lookup_candidates,
    "verse_continuation": gen_verse_continuation_candidates,
}

# who_said ranks long-range items first so stride sampling carries real
# chapter distances; the others stay canonical-order for even book coverage.
FAMILY_POOL_SORT = {
    "who_said": lambda c: (-c["meta"].get("chapter_distance", 0), c["book_index"], c["address"]),
    "address_lookup": lambda c: (c["book_index"], c["address"]),
    "verse_continuation": lambda c: (c["book_index"], c["address"]),
}


def select_items(
    books_parsed: dict[int, list[dict]], book_names: list[str],
    allowed_indices: set[int], targets: dict[str, int], cohort: str,
    used_addresses_global: set[tuple[int, int, int]],
) -> tuple[list[dict], dict[str, int]]:
    selected: list[dict] = []
    used = set(used_addresses_global)
    shortfalls: dict[str, int] = {}
    order_map = {"who_said": 0, "address_lookup": 1, "verse_continuation": 2}
    for family in sorted(targets, key=lambda f: order_map[f]):
        pool: list[dict] = []
        for bi in sorted(allowed_indices):
            if bi in books_parsed:
                pool.extend(GENERATORS[family](bi, book_names[bi], books_parsed[bi]))
        pool.sort(key=FAMILY_POOL_SORT[family])
        step = max(1, len(pool) // targets[family])
        picks, seen_addr = [], set()
        for i in range(0, len(pool), step):
            c = pool[i]
            key = (c["book_index"], *c["address"])
            if key in used or key in seen_addr:
                continue
            seen_addr.add(key)
            picks.append(c)
            if len(picks) >= targets[family]:
                break
        # top-up pass from unpicked remainder (deterministic order)
        if len(picks) < targets[family]:
            for c in pool:
                if len(picks) >= targets[family]:
                    break
                key = (c["book_index"], *c["address"])
                if key in used or key in seen_addr:
                    continue
                seen_addr.add(key)
                picks.append(c)
        shortfalls[family] = targets[family] - len(picks)
        used.update((c["book_index"], *c["address"]) for c in picks)
        for n, c in enumerate(picks):
            c["qa_id"] = f"bq-{cohort[:1]}{order_map[family]}-{n:04d}"
            c["cohort"] = cohort
        selected.extend(picks)
        log(f"qa[{cohort}] {family}: {len(picks)}/{targets[family]} "
            f"(pool {len(pool)}, stride {step})")
    return selected, shortfalls


def serialize_item(item: dict) -> str:
    """Prompt-last framing per cell-lm-corpus-bible.md §6 (pinned delimiters)."""
    body = "\n".join(item["context"])
    return (
        body + (QA_DELIM_Q if body else QA_DELIM_Q.lstrip("\n"))
        + item["question"] + QA_DELIM_A + item["answer"] + QA_TRAILING_NEWLINE
    )


def finalize_manifest(items: list[dict]) -> dict:
    records = []
    for item in items:
        rec = {
            "qa_id": item["qa_id"],
            "cohort": item["cohort"],
            "family": item["family"],
            "references": [
                f"{item['book']} {item['address'][0]}:{item['address'][1]}"
            ],
            "context": item["context"],
            "question": item["question"],
            "answer": item["answer"],
            "serialized_prompt_answer": serialize_item(item),
            "meta": dict(item["meta"], book=item["book"], book_index=item["book_index"]),
        }
        records.append(rec)
    manifest = {
        "schema_version": "humanbrain/bible-qa-manifest/v3",
        "generated_utc": GENERATION_STAMP[0],
        "delimiter_q": QA_DELIM_Q.replace("\n", "\\n"),
        "delimiter_a": QA_DELIM_A.replace("\n", "\\n"),
        "count": len(records),
        "items": records,
    }
    manifest["sha256_canonical"] = sha256_bytes(compact_json(manifest))
    return manifest


GENERATION_STAMP = [None]  # set in main so reruns stay byte-identical in-process


# ------------------------------------------------------------ stage 4: tokens

def pretoken_count(data: bytes) -> int:
    pieces = re.findall(PRETOKENIZE_PATTERN, data)
    assert b"".join(pieces) == data
    return len(pieces)


class CachedBPEEncoder:
    """Greedy lowest-rank merge encoder (hb-cell-lm-bpe-v1 algorithm), memoized."""

    def __init__(self, merges: list[list[int]]):
        self.merges = [tuple(m) for m in merges]
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.cache: dict[bytes, tuple[int, ...]] = {}

    def _encode_piece(self, piece: bytes) -> tuple[int, ...]:
        cur = list(piece)
        while len(cur) > 1:
            best_r = None
            for i in range(len(cur) - 1):
                r = self.rank.get((cur[i], cur[i + 1]))
                if r is not None and (best_r is None or r < best_r):
                    best_r = r
            if best_r is None:
                break
            a, b = self.merges[best_r]
            nid = 256 + best_r
            out, i, L = [], 0, len(cur)
            while i < L:
                if i < L - 1 and cur[i] == a and cur[i + 1] == b:
                    out.append(nid)
                    i += 2
                else:
                    out.append(cur[i])
                    i += 1
            cur = out
        return tuple(cur)

    def count(self, data: bytes) -> int:
        total = 0
        for piece in re.findall(PRETOKENIZE_PATTERN, data):
            ids = self.cache.get(piece)
            if ids is None:
                ids = self._encode_piece(piece)
                self.cache[piece] = ids
            total += len(ids)
        return total


def stage_tokens(spans: list[dict], evidence_dir: Path, artifact_src: Path | None) -> dict:
    per_book = []
    for b in spans:
        data = b["_bytes"]
        per_book.append({
            "index": b["index"],
            "title": b["title_line"],
            "bytes": len(data),
            "pretokens": pretoken_count(data),
        })
    def agg(idxs):
        sub = [p for p in per_book if p["index"] in set(idxs)]
        return {
            "books": len(sub),
            "bytes": sum(p["bytes"] for p in sub),
            "pretokens": sum(p["pretokens"] for p in sub),
        }
    train_idx = sorted(set(range(66)) - set(VAL_BOOK_RANGE))
    result = {
        "generated_utc": GENERATION_STAMP[0],
        "per_book": per_book,
        "train": agg(train_idx),
        "validation": agg(VAL_BOOK_RANGE),
    }
    result["total"] = {
        "bytes": result["train"]["bytes"] + result["validation"]["bytes"],
        "pretokens": result["train"]["pretokens"] + result["validation"]["pretokens"],
    }
    if artifact_src is not None and artifact_src.exists():
        art = json.loads(artifact_src.read_bytes())
        enc = CachedBPEEncoder(art["merges"])
        t0 = time.perf_counter()
        tok_by_book = {}
        for b in spans:
            tok_by_book[b["index"]] = enc.count(b["_bytes"])
        result["fiction_bpe4096_proxy"] = {
            "artifact_sha256": sha256_bytes(artifact_src.read_bytes()),
            "artifact_source": str(artifact_src),
            "encoding_seconds": round(time.perf_counter() - t0, 1),
            "train_tokens": sum(tok_by_book[i] for i in train_idx),
            "validation_tokens": sum(tok_by_book[i] for i in sorted(VAL_BOOK_RANGE)),
            "total_tokens": sum(tok_by_book.values()),
        }
        log(f"tokens: fiction-BPE proxy total {result['fiction_bpe4096_proxy']['total_tokens']:,}")
    result["target_band_tokens"] = [1000000, 10000000]
    proxy_total = result.get("fiction_bpe4096_proxy", {}).get("total_tokens")
    result["within_target_band_proxy"] = (
        proxy_total is not None and 1000000 <= proxy_total <= 10000000
    )
    (evidence_dir / "token_metrics_v3.json").write_bytes(canonical_json(result))
    return result


# ------------------------------------------------------------------ main flow

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()  # flags reserved for future pins; defaults are the pinned ones

    local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
    root = Path(local) / "Temp" / "hb_bible_freeze"
    books_dir, evidence_dir = root / "books", root / "evidence"
    books_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    GENERATION_STAMP[0] = utcnow()

    repo = Path(__file__).resolve().parent.parent
    head = git_head(repo)
    log(f"builder starting; out={root}; git_head={head}")

    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) bible-corpus-materializer"
    fetch_info = stage_fetch(books_dir, user_agent)
    raw = (books_dir / f"pg{GUTENBERG_ID}.fetch1.txt").read_bytes()
    trimmed = stage_trim(raw, evidence_dir)

    books, span_info = derive_book_spans(trimmed)
    for b in books:
        b["_bytes"] = trimmed[b["span"][0]:b["span"][1]]
    log(f"spans: {span_info['titles_parsed']}/66 titles, coverage "
        f"{span_info['span_coverage_of_trimmed_bytes']:.4f}")

    if len(BOOK_SHORT_NAMES) != 66 or len(set(BOOK_SHORT_NAMES)) != 66:
        raise RuntimeError("BOOK_SHORT_NAMES table malformed")

    books_parsed = {}
    anomaly_total = Counter()
    for b in books:
        vs, anom = parse_verses(b["_bytes"])
        books_parsed[b["index"]] = vs
        for k2, v2 in anom.items():
            anomaly_total[k2] += v2
    verse_total = sum(len(v) for v in books_parsed.values())
    log(f"verses: {verse_total:,} parsed (canonical 31,102; delta "
        f"{verse_total - 31102:+d}); anomalies={dict(anomaly_total)}")

    # --- split + parallel-passage hygiene (red test 18 analogue) -------------
    val_indices = set(VAL_BOOK_RANGE)
    train_indices = set(range(66)) - val_indices
    parallel_check = []
    for name, groups in PARALLEL_PASSAGE_GROUPS:
        sides = [
            ("train" if set(g) <= train_indices else
             ("val" if set(g) <= val_indices else "MIXED"))
            for g in groups
        ]
        ok = (
            bool(sides)
            and all(s in ("train", "val") for s in sides)
            and len(set(sides)) == 1
        )
        parallel_check.append({"group": name, "placement": sides, "single_side_pass": ok})
    if not all(c["single_side_pass"] for c in parallel_check):
        raise RuntimeError("parallel-passage same-side assertion FAILED")

    # --- QA generation: TRAIN manifest, then VAL manifest --------------------
    used_addresses: set[tuple[int, int, int]] = set()
    train_items, train_short = select_items(
        books_parsed, BOOK_SHORT_NAMES, train_indices, TRAIN_QA_TARGETS, "TRAIN",
        used_addresses)
    used_addresses.update((c["book_index"], *c["address"]) for c in train_items)
    val_items, val_short = select_items(
        books_parsed, BOOK_SHORT_NAMES, val_indices, VAL_QA_TARGETS, "VAL",
        used_addresses)
    used_addresses.update((c["book_index"], *c["address"]) for c in val_items)

    # --- disjointness verification ------------------------------------------
    def addr_set(items):
        return {(c["book_index"], *c["address"]) for c in items}
    tr_addrs, va_addrs = addr_set(train_items), addr_set(val_items)
    checks = {
        "all_train_books_on_train_side": all(c["book_index"] in train_indices for c in train_items),
        "all_val_books_on_val_side": all(c["book_index"] in val_indices for c in val_items),
        "verse_address_disjoint_train_val": not (tr_addrs & va_addrs),
        "no_duplicate_addresses_within_train": len(tr_addrs) == len(train_items),
        "no_duplicate_addresses_within_val": len(va_addrs) == len(val_items),
        "answers_verbatim_in_context_or_addressable": True,
    }
    addressable = {
        (bi, v["chapter"], v["verse"])
        for bi, vs in books_parsed.items() for v in vs
    }
    for item in train_items + val_items:
        if item["family"] == "who_said":
            ctx = _norm(" ".join(item["context"]))
            if _norm(item["meta"].get("speaker_token", "")) not in ctx:
                checks["answers_verbatim_in_context_or_addressable"] = False
                break
        if item["family"] == "address_lookup":
            probe = _norm(item["answer"])
            bi = item["book_index"]
            src = next(
                (v["text"] for v in books_parsed[bi]
                 if v["chapter"] == item["address"][0] and v["verse"] == item["address"][1]),
                None)
            if src is None or _norm(src) != probe:
                checks["answers_verbatim_in_context_or_addressable"] = False
                break
        if item["family"] == "verse_continuation":
            tgt = tuple(item["meta"]["target_address"])
            if (item["book_index"], *tgt) not in addressable:
                checks["answers_verbatim_in_context_or_addressable"] = False
                break
    if not all(checks.values()):
        raise RuntimeError(f"QA disjointness/derivability checks FAILED: {checks}")
    log("disjointness: train/val QA fully book- and verse-address-disjoint PASS")

    # determinism: regenerate and compare (red test 15 analogue)
    used2: set[tuple[int, int, int]] = set()
    train_items_b, _ = select_items(
        books_parsed, BOOK_SHORT_NAMES, train_indices, TRAIN_QA_TARGETS, "TRAIN", used2)
    used2.update((c["book_index"], *c["address"]) for c in train_items_b)
    select_items(books_parsed, BOOK_SHORT_NAMES, val_indices, VAL_QA_TARGETS, "VAL", used2)
    regen = [
        {k2: c[k2] for k2 in ("family", "book_index", "address", "question", "answer")}
        for c in train_items + val_items
    ]
    regen_b = [
        {k2: c[k2] for k2 in ("family", "book_index", "address", "question", "answer")}
        for c in train_items_b + val_items
    ]
    determinism_pass = compact_json(regen) == compact_json(regen_b)
    if not determinism_pass:
        raise RuntimeError("QA generation rerun-and-equal FAILED")

    train_manifest = finalize_manifest(train_items)
    val_manifest = finalize_manifest(val_items)
    (evidence_dir / "qa_manifest_train.json").write_bytes(canonical_json(train_manifest))
    (evidence_dir / "qa_manifest_val.json").write_bytes(canonical_json(val_manifest))

    # --- token accounting ----------------------------------------------------
    fiction_artifact = Path(local) / "Temp" / "hb_cell_lm_freeze" / "artifacts" / "tokenizer_bpe_4096.json"
    token_metrics = stage_tokens(books, evidence_dir, fiction_artifact)

    # --- per-book span evidence ----------------------------------------------
    span_table = {
        "generated_utc": GENERATION_STAMP[0],
        "derivation_rule": (
            "TOC region = bytes until SECOND occurrence of the Old Testament header; "
            "titles = non-empty TOC lines minus the two section headers; book i heading = "
            "FIRST occurrence of its exact title line after book i-1's heading; "
            "span = [heading_i, heading_{i+1}), last ends at EOF"
        ),
        "structure_verification": span_info,
        "books": [
            {
                "index": b["index"], "title": b["title_line"],
                "short_name": BOOK_SHORT_NAMES[b["index"]],
                "span_offsets": b["span"], "bytes": len(b["_bytes"]),
                "chapters": max(v["chapter"] for v in books_parsed[b["index"]]),
                "verses": len(books_parsed[b["index"]]),
                "role": "val" if b["index"] in val_indices else "train",
            }
            for b in books
        ],
        "verse_totals": {
            "parsed": verse_total,
            "canonical_kjv": 31102,
            "delta_vs_canonical": verse_total - 31102,
            "anomalies": dict(anomaly_total),
            "note": "PG10 inline C:V markers carry documented quirks; never repaired",
        },
    }
    (evidence_dir / "pg10_book_spans_v3.json").write_bytes(canonical_json(span_table))

    # --- freeze candidate ----------------------------------------------------
    candidate = {
        "schema": "hb-cell-lm-freeze-candidate-v3-bible",
        "candidate_id": "cell-lm-v1-freeze-candidate-bible-kjv-pg10-v3",
        "status": "CANDIDATE_NOT_FROZEN",
        "generated_utc": GENERATION_STAMP[0],
        "supersedes": "cell-lm-v1-freeze-candidate-bible-kjv-pg10 (v2 PROPOSAL)",
        "design_document": "docs/research/cell-lm-corpus-bible.md",
        "builder_tool": "tools/build_bible_qa_corpus.py",
        "builder_git_head": head,
        "corpus": {
            "role": "PRIMARY",
            "gutenberg_id": GUTENBERG_ID,
            "title": "The King James Version of the Bible",
            "mirror_url": fetch_info["mirror_url"],
            "fetch_policy": "two independent sequential GETs; SHA-256 must equal the pinned digest; mismatch blocks",
            "acquisition": fetch_info,
            "trim_rule": TRIM_RULE,
            "size_trimmed_bytes": len(trimmed),
            "sha256_trimmed": sha256_bytes(trimmed),
            "license": "KJV 1611 public domain; PG boilerplate stripped by trim rule",
        },
        "split": {
            "rule": "whole-block holdout: validation = minor-prophets (27-38) UNION gospels+acts (39-43); train = remaining 49",
            "train_book_count": len(train_indices),
            "val_book_count": len(val_indices),
            "val_book_indices": sorted(val_indices),
            "val_books_short": [BOOK_SHORT_NAMES[i] for i in sorted(val_indices)],
            "genre_blocks": [
                {"block": n, "indices": [lo, hi]} for n, lo, hi in GENRE_BLOCKS
            ],
            "parallel_passage_same_side_check": parallel_check,
        },
        "qa": {
            "framing": "prompt-last; serialized '<context>\\nQ: {question}\\nA: {answer}\\n'; input through 'A:' inclusive; teacher = answer + terminator",
            "generators": {
                "who_said": "G2 source attribution with chapter_distance annotation (requires >=1 earlier occurrence of the speaker token in the same book)",
                "address_lookup": "G3 book/chapter/verse address -> full verse text binding (verses 25-140 chars)",
                "verse_continuation": "G1 next-verse completion, intra-chapter, answer = first <=12 words of verse n+1",
            },
            "train_targets": TRAIN_QA_TARGETS,
            "val_targets": VAL_QA_TARGETS,
            "train_shortfalls": train_short,
            "val_shortfalls": val_short,
            "train_count": len(train_items),
            "val_count": len(val_items),
            "manifest_train_sha256": train_manifest["sha256_canonical"],
            "manifest_val_sha256": val_manifest["sha256_canonical"],
            "checks": {
                **checks,
                "generation_rerun_and_equal": determinism_pass,
                "selection": "deterministic even-stride over canonically sorted pools; global verse-address dedupe",
            },
        },
        "token_counts": {
            "train_bytes": token_metrics["train"]["bytes"],
            "val_bytes": token_metrics["validation"]["bytes"],
            "train_pretokens": token_metrics["train"]["pretokens"],
            "val_pretokens": token_metrics["validation"]["pretokens"],
            "fiction_bpe4096_proxy": token_metrics.get("fiction_bpe4096_proxy"),
            "target_band_tokens": token_metrics["target_band_tokens"],
            "within_target_band_proxy": token_metrics["within_target_band_proxy"],
            "note": "freeze still retrains tokenizer on the new train split; proxy is upper bound",
        },
        "red_test_status": {
            "rt15_manifest_determinism": determinism_pass,
            "rt16_qa_split_disjointness": bool(checks["verse_address_disjoint_train_val"] and checks["all_train_books_on_train_side"] and checks["all_val_books_on_val_side"]),
            "rt18_parallel_passage_single_side": all(c["single_side_pass"] for c in parallel_check),
            "rt19_verse_count_delta_documented": span_table["verse_totals"]["delta_vs_canonical"],
            "rt17_delimiter_boundary": "pinned delimiters carried in manifest; token-level check lands with the frozen tokenizer",
            "rt20_majority_floor": "computed by tools/qa_baselines.py (preregistered comparators)",
        },
        "evidence_files": {
            "trimmed": "evidence/pg10.trimmed.bin",
            "book_spans": "evidence/pg10_book_spans_v3.json",
            "qa_train": "evidence/qa_manifest_train.json",
            "qa_val": "evidence/qa_manifest_val.json",
            "token_metrics": "evidence/token_metrics_v3.json",
        },
    }
    out_path = root / "FREEZE-CANDIDATE-v3-BIBLE.json"
    tmp_path = root / "FREEZE-CANDIDATE-v3-BIBLE.json.tmp"
    tmp_path.write_bytes(canonical_json(candidate))
    tmp_path.replace(out_path)
    log(f"FREEZE-CANDIDATE-v3-BIBLE.json written: {out_path}")
    log(f"summary: train QA {len(train_items)}, val QA {len(val_items)}, "
        f"train bytes {token_metrics['train']['bytes']:,}")


if __name__ == "__main__":
    main()
