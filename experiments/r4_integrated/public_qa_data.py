"""Deterministic public SQuAD TRAIN preparation; tokenizer only, never inference.

Original labels belong to the training/evaluator view. Callers must pass only
raw question/context inputs to a predictor. No DEV/test download or model import.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
LANDING_URL = "https://rajpurkar.github.io/SQuAD-explorer/"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
TOKENIZER = ROOT / ".models/all-MiniLM-L6-v2/tokenizer.json"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
SCHEMA = "public_qa_squad2_train_selection_v1"
QUOTAS = {"train": (768, 256), "probe": (192, 64)}
POLICY = {
    "article_split": (
        "int(SHA256(original UTF-8 article title),16) modulo 5; zero probe, else train"
    ),
    "selection": "ascending SHA256(original UTF-8 QA id), then id; within split/label",
    "interleave": "three supported then one unsupported, repeated",
    "max_total_tokens_each_input": 256,
    "word_pattern": r"\w+",
    "word_pattern_flags": "Python Unicode defaults",
    "window_words": 48,
    "windows": "nonoverlapping complete regex-word windows; unchanged rate-cell inventory",
    "pooling": "each regex word overlaps at least one whole non-special subtoken, none split",
    "supported_annotations": (
        "every original answer matches original characters/bytes and exact word endpoints "
        "within one window"
    ),
    "cross_article_duplicates": (
        "exclude every context SHA appearing under multiple original article titles"
    ),
    "within_article_duplicates": (
        "same context/question bytes: keep smallest QA-id SHA only when answerability "
        "and annotation (text,start) sets agree; otherwise exclude all"
    ),
    "duplicate_ids": (
        "identical source/question/label records collapse to first original position; "
        "conflicting records stop preparation"
    ),
    "retention": (
        "first/last 256 train rows; independently select 24 supported and 8 STOP "
        "by QA-id SHA, then 3:1 interleave"
    ),
    "normalization": "none; tokenizer normalization does not modify retained original text",
}
ATTRIBUTION = {
    "dataset": "Stanford Question Answering Dataset 2.0 (SQuAD 2.0), official training set",
    "creators": "Pranav Rajpurkar, Robin Jia and Percy Liang; Stanford NLP Group",
    "source": LANDING_URL,
    "paper": "https://aclanthology.org/P18-2124/",
    "license": "CC BY-SA 4.0",
    "license_url": LICENSE_URL,
    "changes": (
        "Deterministic subset, article partition and added byte/token/custody metadata; "
        "original question/context/answer text unchanged."
    ),
    "derived_subset_license": "CC BY-SA 4.0",
    "exposure": (
        "Public development; model-pretraining overlap unknown; not a reserved or "
        "contamination-free evaluation."
    ),
}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")


def download_training(cache: Path) -> dict:
    """Download only the official TRAIN link once, preserving bytes and custody."""
    from urllib.request import Request, urlopen

    cache.mkdir(parents=True, exist_ok=True)
    raw_path, receipt_path = cache / "train-v2.0.json", cache / "source-custody.json"
    if raw_path.exists() or receipt_path.exists():
        if not (raw_path.exists() and receipt_path.exists()):
            raise FileExistsError("incomplete existing custody; preserve it rather than redownload")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["sha256"] != file_hash(raw_path) or receipt["bytes"] != raw_path.stat().st_size:
            raise ValueError("cached official TRAIN bytes changed")
        return receipt
    request = Request(SOURCE_URL, headers={"User-Agent": "HumanBrain-public-TRAIN-custody/1"})
    started = datetime.now(UTC).isoformat()
    with urlopen(request, timeout=120) as response:
        if response.status != 200 or response.url != SOURCE_URL:
            raise ValueError("official TRAIN response status or final URL differs")
        headers = dict(response.headers.items())
        with raw_path.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    if "Content-Length" in headers and raw_path.stat().st_size != int(headers["Content-Length"]):
        raise ValueError("official TRAIN download length differs; partial bytes preserved")
    receipt = {
        "source_url": SOURCE_URL,
        "final_url": SOURCE_URL,
        "status": 200,
        "started_utc": started,
        "completed_utc": datetime.now(UTC).isoformat(),
        "http_headers": headers,
        "bytes": raw_path.stat().st_size,
        "sha256": file_hash(raw_path),
        "attribution": ATTRIBUTION,
        "authority": "Authors' landing page, Training Set v2.0 link; no DEV/test access",
    }
    _write_json(receipt_path, receipt)
    return receipt


def _word_layout(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + len(char.encode("utf-8")))
    words = [(offsets[m.start()], offsets[m.end()]) for m in re.finditer(r"\w+", text)]
    return offsets, words


def _token_info(tokenizer, text: str) -> dict:
    chars, words = _word_layout(text)
    if not text:
        return {"reasons": ["empty_input"], "words": words, "chars": chars, "token_count": 0}
    encoded = tokenizer.encode(text)
    reasons = []
    if len(encoded.ids) > 256:
        reasons.append("token_overflow")
    spans = [
        (chars[a], chars[b])
        for (a, b), special in zip(encoded.offsets, encoded.special_tokens_mask, strict=True)
        if not special and a != b
    ]
    for a, b in words:
        overlap = [(left, right) for left, right in spans if left < b and right > a]
        if not overlap:
            reasons.append("word_without_subtoken")
        elif any(left < a or right > b for left, right in overlap):
            reasons.append("word_splits_subtoken")
    return {
        "reasons": sorted(set(reasons)),
        "chars": chars,
        "words": words,
        "token_count": len(encoded.ids),
        "non_special_tokens": len(spans),
    }


def _label_signature(qa: dict) -> str:
    annotations = sorted({(a["text"], a["answer_start"]) for a in qa["answers"]})
    return json.dumps([qa["is_impossible"], annotations], ensure_ascii=False)


def _annotated_answers(context: str, qa: dict, info: dict) -> tuple[list[dict], list[str]]:
    reasons, answers = [], []
    if qa["is_impossible"]:
        return ([], [] if not qa["answers"] else ["unsupported_has_accepted_answers"])
    if not qa["answers"]:
        return [], ["supported_without_annotation"]
    starts = {a: i for i, (a, _) in enumerate(info["words"])}
    ends = {b: i + 1 for i, (_, b) in enumerate(info["words"])}
    raw = context.encode("utf-8")
    for annotation in qa["answers"]:
        a, text = annotation["answer_start"], annotation["text"]
        b = a + len(text)
        if type(a) is not int or not text or not 0 <= a < b <= len(context) or context[a:b] != text:
            reasons.append("answer_character_mismatch")
            continue
        left, right = info["chars"][a], info["chars"][b]
        if raw[left:right] != text.encode("utf-8"):
            reasons.append("answer_byte_mismatch")
            continue
        if left not in starts or right not in ends:
            reasons.append("answer_not_exact_word_boundaries")
            continue
        first, last = starts[left], ends[right]
        if first >= last or first // 48 != (last - 1) // 48:
            reasons.append("answer_crosses_48_word_window")
            continue
        answers.append(
            {
                **annotation,
                "byte_start": left,
                "byte_end": right,
                "word_start": first,
                "word_end": last,
                "window_index": first // 48,
            }
        )
    return answers, sorted(set(reasons))


def _rank(record: dict) -> tuple[str, str]:
    return text_hash(record["id"]), record["id"]


def _interleave(supported: list[dict], unsupported: list[dict]) -> list[dict]:
    if len(supported) != 3 * len(unsupported):
        raise ValueError("the frozen roster requires a 3:1 label ratio")
    return [
        row for i, stop in enumerate(unsupported) for row in (*supported[3 * i : 3 * i + 3], stop)
    ]


def _retention(rows: list[dict]) -> list[str]:
    yes = sorted((r for r in rows if not r["is_impossible"]), key=_rank)[:24]
    no = sorted((r for r in rows if r["is_impossible"]), key=_rank)[:8]
    if len(yes) != 24 or len(no) != 8:
        raise ValueError("insufficient fixed retention roster")
    return [r["id"] for r in _interleave(yes, no)]


def prepare_selection(cache: Path, output: Path, tokenizer_path: Path = TOKENIZER) -> dict:
    """Select from authenticated original TRAIN using only pinned tokenization."""
    from tokenizers import Tokenizer

    exclusions_path = output.with_suffix(".exclusions.json")
    if output.exists() or exclusions_path.exists():
        raise FileExistsError("selection/custody outputs cannot be overwritten")
    raw_path, custody_path = cache / "train-v2.0.json", cache / "source-custody.json"
    custody = json.loads(custody_path.read_text(encoding="utf-8"))
    source_hash = file_hash(raw_path)
    if (
        source_hash != custody["sha256"]
        or raw_path.stat().st_size != custody["bytes"]
        or custody["source_url"] != SOURCE_URL
        or custody["final_url"] != SOURCE_URL
    ):
        raise ValueError("original TRAIN custody mismatch")
    if file_hash(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("pinned MiniLM tokenizer changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    dataset = json.loads(raw_path.read_text(encoding="utf-8"))
    if dataset["version"] != "v2.0":
        raise ValueError("official source is not SQuAD 2.0")
    refs, paragraphs, contexts, duplicates, ids = [], {}, defaultdict(set), defaultdict(list), {}
    excluded = {}
    for ai, article in enumerate(dataset["data"]):
        title = article["title"]
        for pi, paragraph in enumerate(article["paragraphs"]):
            pid = f"squad2-train-a{ai}-p{pi}"
            context = paragraph["context"]
            chash = text_hash(context)
            contexts[chash].add(title)
            paragraphs[pid] = (title, ai, pi, context, chash)
            for qi, qa in enumerate(paragraph["qas"]):
                if type(qa["is_impossible"]) is not bool or not isinstance(qa["id"], str):
                    raise ValueError("invalid original QA identity/answerability")
                signature = (title, context, qa["question"], _label_signature(qa))
                n = len(refs)
                refs.append((pid, qi, qa))
                if qa["id"] in ids:
                    if signature != ids[qa["id"]]:
                        raise ValueError(f"conflicting globally repeated QA id: {qa['id']}")
                    excluded[n] = ["duplicate_qa_id_identical"]
                else:
                    ids[qa["id"]] = signature
                duplicates[(title, chash, qa["question"])].append(n)
    for n, (pid, _, _) in enumerate(refs):
        if len(contexts[paragraphs[pid][4]]) > 1:
            excluded[n] = ["context_repeated_across_article_titles"]
    for indices in duplicates.values():
        active = [n for n in indices if n not in excluded]
        if len(active) < 2:
            continue
        if len({_label_signature(refs[n][2]) for n in active}) > 1:
            for n in active:
                excluded[n] = ["conflicting_same_article_context_question"]
        else:
            winner = min(active, key=lambda n: _rank(refs[n][2]))
            for n in active:
                if n != winner:
                    excluded[n] = ["duplicate_same_article_context_question"]
    candidates = {split: {False: [], True: []} for split in QUOTAS}
    by_paragraph = defaultdict(list)
    for n, (pid, _, _) in enumerate(refs):
        by_paragraph[pid].append(n)
    for pid, indices in by_paragraph.items():
        title, _, _, context, _ = paragraphs[pid]
        if all(n in excluded for n in indices):
            continue
        context_info = _token_info(tokenizer, context)
        split = "probe" if int(text_hash(title), 16) % 5 == 0 else "train"
        for n in indices:
            if n in excluded:
                continue
            qa = refs[n][2]
            reasons = ["context_" + r for r in context_info["reasons"]]
            if not reasons:
                question_info = _token_info(tokenizer, qa["question"])
                reasons.extend("question_" + r for r in question_info["reasons"])
                _, annotation_reasons = _annotated_answers(context, qa, context_info)
                reasons.extend(annotation_reasons)
            if reasons:
                excluded[n] = sorted(set(reasons))
            else:
                candidates[split][qa["is_impossible"]].append({"id": qa["id"], "ref": n})
    selected = {}
    for split, (yes_count, no_count) in QUOTAS.items():
        yes = sorted(candidates[split][False], key=_rank)
        no = sorted(candidates[split][True], key=_rank)
        if len(yes) < yes_count or len(no) < no_count:
            raise ValueError(
                f"insufficient eligible {split} roster: {len(yes)} supported, {len(no)} STOP"
            )
        selected[split] = _interleave(yes[:yes_count], no[:no_count])
    documents, records = {}, {}
    for split, rows in selected.items():
        records[split] = []
        for row in rows:
            pid, qi, qa = refs[row["ref"]]
            title, ai, pi, context, chash = paragraphs[pid]
            cinfo = _token_info(tokenizer, context)
            qinfo = _token_info(tokenizer, qa["question"])
            answers, reasons = _annotated_answers(context, qa, cinfo)
            if reasons:
                raise AssertionError("selected answer eligibility changed")
            documents[pid] = {
                "article_title": title,
                "article_sha256": text_hash(title),
                "article_index": ai,
                "paragraph_index": pi,
                "context": context,
                "context_sha256": chash,
                "word_byte_spans": cinfo["words"],
                "word_count": len(cinfo["words"]),
                "token_count": cinfo["token_count"],
                "non_special_tokens": cinfo["non_special_tokens"],
            }
            records[split].append(
                {
                    "id": qa["id"],
                    "id_sha256": text_hash(qa["id"]),
                    "paragraph_id": pid,
                    "qa_index": qi,
                    "question": qa["question"],
                    "question_sha256": text_hash(qa["question"]),
                    "question_word_byte_spans": qinfo["words"],
                    "question_word_count": len(qinfo["words"]),
                    "question_token_count": qinfo["token_count"],
                    "is_impossible": qa["is_impossible"],
                    "answers": answers,
                    "accepted_text_hex": list(
                        dict.fromkeys(a["text"].encode().hex() for a in answers)
                    ),
                }
            )
    summary = {
        "original_articles": len(dataset["data"]),
        "original_paragraphs": len(paragraphs),
        "original_questions": len(refs),
        "unique_original_qa_ids": len(ids),
        "excluded_questions": len(excluded),
        "exclusion_reason_counts": dict(
            sorted(Counter(r for rs in excluded.values() for r in rs).items())
        ),
        "eligible": {
            s: {"supported": len(v[False]), "unsupported": len(v[True])}
            for s, v in candidates.items()
        },
        "selected": {
            s: {
                "supported": sum(not r["is_impossible"] for r in v),
                "unsupported": sum(r["is_impossible"] for r in v),
            }
            for s, v in records.items()
        },
        "selected_paragraphs": len(documents),
        "unselected_eligible_questions": sum(
            len(v[False]) + len(v[True]) for v in candidates.values()
        )
        - 1280,
    }
    exclusions = {
        "source_sha256": source_hash,
        "policy": POLICY,
        "summary": summary,
        "records": [
            {
                "id": refs[n][2]["id"],
                "paragraph_id": refs[n][0],
                "qa_index": refs[n][1],
                "reasons": reasons,
            }
            for n, reasons in sorted(excluded.items())
        ],
    }
    payload = {
        "schema": SCHEMA,
        "preparation_source_sha256": file_hash(Path(__file__)),
        "policy": POLICY,
        "attribution": ATTRIBUTION,
        "source": custody,
        "source_custody_sha256": file_hash(custody_path),
        "tokenizer": {
            "sha256": TOKENIZER_SHA256,
            "runtime": version("tokenizers"),
            "special_tokens_included_in_limit": True,
            "truncation": "disabled",
        },
        "documents": documents,
        **records,
        "early_train_ids": _retention(records["train"][:256]),
        "late_train_ids": _retention(records["train"][-256:]),
        "summary": summary,
    }
    payload["overlap_checks"] = _overlap_checks(payload)
    payload["cache_estimates"] = _cache_estimates(payload)
    if file_hash(raw_path) != source_hash or file_hash(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("preparation inputs changed")
    _write_json(exclusions_path, exclusions)
    payload["exclusions"] = {"path": str(exclusions_path), "sha256": file_hash(exclusions_path)}
    _write_json(output, payload)
    return {
        "selection_path": str(output),
        "selection_sha256": file_hash(output),
        "summary": summary,
        "overlap_checks": payload["overlap_checks"],
        "cache_estimates": payload["cache_estimates"],
    }


def _overlap_checks(data: dict) -> dict:
    ids, articles, paragraphs, contexts, pairs = {}, {}, {}, {}, {}
    for split in QUOTAS:
        rows = data[split]
        ids[split] = {r["id"] for r in rows}
        paragraphs[split] = {r["paragraph_id"] for r in rows}
        articles[split] = {data["documents"][p]["article_title"] for p in paragraphs[split]}
        contexts[split] = {data["documents"][p]["context_sha256"] for p in paragraphs[split]}
        pairs[split] = {
            (data["documents"][r["paragraph_id"]]["context"], r["question"]) for r in rows
        }
        if len(ids[split]) != len(rows) or len(pairs[split]) != len(rows):
            raise ValueError("selected duplicate QA id or context/question bytes")
    result = {
        "qa_ids": len(ids["train"] & ids["probe"]),
        "article_titles": len(articles["train"] & articles["probe"]),
        "paragraph_ids": len(paragraphs["train"] & paragraphs["probe"]),
        "context_sha256": len(contexts["train"] & contexts["probe"]),
        "train_articles": len(articles["train"]),
        "probe_articles": len(articles["probe"]),
        "train_paragraphs": len(paragraphs["train"]),
        "probe_paragraphs": len(paragraphs["probe"]),
    }
    if any(result[k] for k in ("qa_ids", "article_titles", "paragraph_ids", "context_sha256")):
        raise ValueError("TRAIN/probe source leakage")
    return result


def _cache_estimates(data: dict) -> dict:
    texts = {
        d["context"]: (d["word_count"], d["non_special_tokens"]) for d in data["documents"].values()
    }
    for split in QUOTAS:
        for r in data[split]:
            texts.setdefault(
                r["question"], (r["question_word_count"], r["question_token_count"] - 2)
            )
    words = sum(v[0] for v in texts.values())
    candidates = sum(
        sum(
            min(48, d["word_count"] - start) * (min(48, d["word_count"] - start) + 1) // 2
            for start in range(0, d["word_count"], 48)
        )
        for d in data["documents"].values()
    )
    return {
        "unique_raw_inputs": len(texts),
        "unique_input_regex_words": words,
        "raw_input_bytes": sum(len(t.encode()) for t in texts),
        "word_vectors_float64_bytes": words * 384 * 8,
        "word_offsets_int64_bytes": words * 2 * 8,
        "maximum_temporary_subtoken_float32_bytes": max(v[1] for v in texts.values()) * 384 * 4,
        "all_selected_document_candidate_count_if_materialized": candidates,
        "scope": (
            "Array/raw-byte estimates only; Python objects, models, optimizer, receipts and "
            "live allocations require separate runtime accounting. No candidate objects "
            "or vectors created here."
        ),
    }


def load_selection(path: Path, expected_sha256: str) -> dict:
    """Authenticate reviewed derived bytes; no tokenizer/model/network execution."""
    if file_hash(path) != expected_sha256:
        raise ValueError("public QA selection hash differs")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data["schema"] != SCHEMA
        or data["policy"] != POLICY
        or data["tokenizer"]["sha256"] != TOKENIZER_SHA256
    ):
        raise ValueError("public QA selection policy differs")
    if data["source"]["source_url"] != SOURCE_URL:
        raise ValueError("selection is not from official TRAIN")
    for pid, d in data["documents"].items():
        if d["context_sha256"] != text_hash(d["context"]) or d["article_sha256"] != text_hash(
            d["article_title"]
        ):
            raise ValueError(f"document text identity differs: {pid}")
        if d["word_byte_spans"] != [list(s) for s in _word_layout(d["context"])[1]]:
            raise ValueError("document word boundaries differ")
    for split, (yes, no) in QUOTAS.items():
        rows = data[split]
        if (
            len(rows) != yes + no
            or [r["is_impossible"] for r in rows] != [False, False, False, True] * no
        ):
            raise ValueError("public QA quota/order differs")
        for label in (False, True):
            subset = [r for r in rows if r["is_impossible"] == label]
            if subset != sorted(subset, key=_rank):
                raise ValueError("public QA within-label rank order differs")
        for r in rows:
            d = data["documents"][r["paragraph_id"]]
            actual_split = "probe" if int(d["article_sha256"], 16) % 5 == 0 else "train"
            if (
                actual_split != split
                or r["id_sha256"] != text_hash(r["id"])
                or r["question_sha256"] != text_hash(r["question"])
            ):
                raise ValueError("question/source split identity differs")
            if r["question_word_byte_spans"] != [list(s) for s in _word_layout(r["question"])[1]]:
                raise ValueError("question word boundaries differ")
            info = {"chars": _word_layout(d["context"])[0], "words": d["word_byte_spans"]}
            answers, reasons = _annotated_answers(d["context"], r, info)
            if (
                reasons
                or answers != r["answers"]
                or r["accepted_text_hex"]
                != list(dict.fromkeys(a["text"].encode().hex() for a in answers))
            ):
                raise ValueError("accepted answer custody differs")
    if data["early_train_ids"] != _retention(data["train"][:256]) or data[
        "late_train_ids"
    ] != _retention(data["train"][-256:]):
        raise ValueError("retention selection differs")
    if (
        _overlap_checks(data) != data["overlap_checks"]
        or _cache_estimates(data) != data["cache_estimates"]
    ):
        raise ValueError("selection overlap/cost summary differs")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--cache", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--cache", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--selection", type=Path, required=True)
    verify.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "download":
        result = download_training(args.cache)
    elif args.command == "prepare":
        result = prepare_selection(args.cache, args.output)
    else:
        data = load_selection(args.selection, args.sha256)
        result = {"status": "verified", "summary": data["summary"]}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
