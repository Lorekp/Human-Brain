"""Pinned public TRAIN selection and bounded float32 word-input storage.

Selection uses tokenization only. Encoding is an explicit, separately admitted
callback; importing/loading this module never starts an encoder or learner.
Answer annotations belong to the external teaching/evaluator view.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from experiments.r4_integrated import public_qa_data as prior

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "experiments/r4_integrated/token_pointer_selection_v1.json"
RAW = ROOT / ".knowledge-cache/public_qa/squad2/train-v2.0.json"
OLD = ROOT / "experiments/r4_integrated/real_input_qualification_cases_v1.json"
INTEGRATION = ROOT / "experiments/r4_integrated/fresh_source_cases_v1.json"
PUBLIC = ROOT / "experiments/r4_integrated/public_qa_selection_v1.json"
REPORT = ROOT / "benchmarks/public_qa_acquisition_v1.json"
ARCHIVE = ROOT / "benchmarks/public_qa_acquisition_v1.npz"
PINS = {
    RAW: "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002",
    OLD: "aa988986d8f04d7e796e485ccea33f24c96cadad09f03779de844905ee5ad41b",
    INTEGRATION: "6b78781c78c1d7b6d2474cbd8185123f9349e682e2ecde900ed2a852fbf17faf",
    PUBLIC: "72fa7ac873643be36701c486a66c24ae67631fcd95c6fe18c2021539dd3cfe76",
    REPORT: "9e0f5a0a20ca1726777b5747e86bf69a151622e3911b608c043e24af6a850708",
    ARCHIVE: "ef1abd594613cebf8b5a138efd6c8ab49d9c44817c5e47a90c31a1692993a3d6",
    ROOT / "experiments/r4_integrated/public_qa_data.py": (
        "36403b86f462201d529f4ec6a24fc297e75ba3641c10f1168586f157005e0bd6"
    ),
}
SCHEMA = "token_pointer_public_train_selection_v1"
QUOTAS = {"train": (6144, 2048), "probe": (192, 64)}
VECTOR_POLICY = "complete_word_float32_subtoken_mean;float64_L2_max1e-12;cast_float32"
ENCODER_SHA256 = "3e0ae5f0bdefe2267dc0f1c6c9db57d20a7abcfdd8341dadef2132a9b86e6c27"
POLICY = {
    **prior.POLICY,
    "window_words": None,
    "windows": "complete original paragraph; all within-paragraph word intervals",
    "supported_annotations": "every annotation matches original bytes and exact word endpoints",
    "retention": "128 TRAIN IDs: first 96 supported and 32 STOP by QA-id SHA, interleaved 3:1",
    "required_train": "old sixteen included within quotas; fill remaining slots by QA-id SHA",
    "probe_exclusion": "all 256 previously selected public probe IDs; no new-ID exposure claim",
    "integration_exclusion": "all twelve E/L questions excluded from broad teaching",
}


def _hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _pin(path: Path) -> None:
    if prior.file_hash(path) != PINS[path]:
        raise ValueError(f"pinned source changed: {path}")


def _answers(context: str, qa: dict, info: dict) -> tuple[list[dict], list[str]]:
    """Original annotation rules, except the obsolete 48-word partition."""
    if qa["is_impossible"]:
        return [], [] if not qa["answers"] else ["unsupported_has_accepted_answers"]
    if not qa["answers"]:
        return [], ["supported_without_annotation"]
    starts = {a: i for i, (a, _) in enumerate(info["words"])}
    ends = {b: i + 1 for i, (_, b) in enumerate(info["words"])}
    answers, reasons = [], []
    for annotation in qa["answers"]:
        a, text = annotation["answer_start"], annotation["text"]
        b = a + len(text)
        if type(a) is not int or not text or not 0 <= a < b <= len(context) or context[a:b] != text:
            reasons.append("answer_character_mismatch")
            continue
        left, right = info["chars"][a], info["chars"][b]
        if left not in starts or right not in ends or starts[left] >= ends[right]:
            reasons.append("answer_not_exact_word_boundaries")
            continue
        answers.append(
            {
                **annotation,
                "byte_start": left,
                "byte_end": right,
                "word_start": starts[left],
                "word_end": ends[right],
            }
        )
    return answers, sorted(set(reasons))


def _source_rows(dataset: dict) -> tuple[list, dict, dict]:
    """Keep prior duplicate policy without storing token or feature matrices."""
    refs, paragraphs, contexts, duplicates, ids = [], {}, defaultdict(set), defaultdict(list), {}
    excluded = {}
    for ai, article in enumerate(dataset["data"]):
        title = article["title"]
        for pi, paragraph in enumerate(article["paragraphs"]):
            pid = f"squad2-train-a{ai}-p{pi}"
            context = paragraph["context"]
            chash = prior.text_hash(context)
            contexts[chash].add(title)
            paragraphs[pid] = (title, ai, pi, context, chash)
            for qi, qa in enumerate(paragraph["qas"]):
                n = len(refs)
                refs.append((pid, qi, qa))
                signature = (title, context, qa["question"], prior._label_signature(qa))
                if qa["id"] in ids:
                    if ids[qa["id"]] != signature:
                        raise ValueError("conflicting globally repeated QA ID")
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
        if len({prior._label_signature(refs[n][2]) for n in active}) > 1:
            for n in active:
                excluded[n] = ["conflicting_same_article_context_question"]
        else:
            winner = min(active, key=lambda n: prior._rank(refs[n][2]))
            for n in active:
                if n != winner:
                    excluded[n] = ["duplicate_same_article_context_question"]
    return refs, paragraphs, excluded


def prepare_selection(output: Path = SELECTION) -> dict:
    """Create the one deterministic roster; tokenization allowed, no encoding."""
    from tokenizers import Tokenizer

    exclusions_path = ROOT / ".pytest-tmp/token-pointer-data-exclusions.json"
    if output.exists() or exclusions_path.exists():
        raise FileExistsError("selection outputs already exist; preserve original preparation")
    for path in (
        RAW,
        OLD,
        INTEGRATION,
        PUBLIC,
        ROOT / "experiments/r4_integrated/public_qa_data.py",
    ):
        _pin(path)
    if prior.file_hash(prior.TOKENIZER) != prior.TOKENIZER_SHA256:
        raise ValueError("pinned tokenizer changed")
    old = json.loads(OLD.read_bytes())
    public = prior.load_selection(PUBLIC, PINS[PUBLIC])
    integration = json.loads(INTEGRATION.read_bytes())
    required = {r["id"] for r in old["cases"]}
    banned_probe = {r["id"] for r in public["probe"]}
    banned_train = set(integration["blocks"]["E"] + integration["blocks"]["L"])
    tokenizer = Tokenizer.from_file(str(prior.TOKENIZER))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    dataset = json.loads(RAW.read_bytes())
    refs, paragraphs, excluded = _source_rows(dataset)
    candidates = {s: {False: [], True: []} for s in QUOTAS}
    by_paragraph = defaultdict(list)
    for n, (pid, _, _) in enumerate(refs):
        by_paragraph[pid].append(n)
    for pid, indices in by_paragraph.items():
        title, _, _, context, _ = paragraphs[pid]
        info = prior._token_info(tokenizer, context)
        split = "probe" if int(prior.text_hash(title), 16) % 5 == 0 else "train"
        for n in indices:
            if n in excluded:
                continue
            qa = refs[n][2]
            reasons = ["context_" + r for r in info["reasons"]]
            if not reasons:
                qinfo = prior._token_info(tokenizer, qa["question"])
                reasons += ["question_" + r for r in qinfo["reasons"]]
                reasons += _answers(context, qa, info)[1]
            if split == "probe" and qa["id"] in banned_probe:
                reasons.append("prior_project_selected_probe_id")
            if split == "train" and qa["id"] in banned_train:
                reasons.append("integration_episode_or_late_id")
            if reasons:
                excluded[n] = sorted(set(reasons))
            else:
                candidates[split][qa["is_impossible"]].append({"id": qa["id"], "ref": n})
    chosen = {}
    for split, quotas in QUOTAS.items():
        lists = []
        for label, quota in zip((False, True), quotas, strict=True):
            ranked = sorted(candidates[split][label], key=prior._rank)
            forced = [r for r in ranked if split == "train" and r["id"] in required]
            picked = forced + [r for r in ranked if r not in forced][: quota - len(forced)]
            if len(picked) != quota:
                raise ValueError(f"insufficient {split}/{label} roster; no smaller replacement")
            lists.append(sorted(picked, key=prior._rank))
        chosen[split] = prior._interleave(*lists)
    if not required <= {r["id"] for r in chosen["train"]}:
        raise ValueError("an original small-fit question is ineligible")
    documents, records = {}, {}
    for split, rows in chosen.items():
        records[split] = []
        for row in rows:
            pid, qi, qa = refs[row["ref"]]
            title, ai, pi, context, chash = paragraphs[pid]
            cinfo, qinfo = (
                prior._token_info(tokenizer, context),
                prior._token_info(tokenizer, qa["question"]),
            )
            answers, reasons = _answers(context, qa, cinfo)
            if reasons:
                raise ValueError("selected annotation changed")
            documents[pid] = {
                "article_title": title,
                "article_sha256": prior.text_hash(title),
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
                    "id_sha256": prior.text_hash(qa["id"]),
                    "paragraph_id": pid,
                    "qa_index": qi,
                    "question": qa["question"],
                    "question_sha256": prior.text_hash(qa["question"]),
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
    probe_train = prior._interleave(
        [r for r in records["train"] if not r["is_impossible"]][:96],
        [r for r in records["train"] if r["is_impossible"]][:32],
    )
    data = {
        "schema": SCHEMA,
        "policy": POLICY,
        "attribution": prior.ATTRIBUTION,
        "preparation_source_sha256": prior.file_hash(Path(__file__)),
        "source_pins": {
            p.relative_to(ROOT).as_posix(): PINS[p] for p in (RAW, OLD, INTEGRATION, PUBLIC)
        },
        "tokenizer_sha256": prior.TOKENIZER_SHA256,
        "documents": documents,
        **records,
        "small_fit_ids": [r["id"] for r in old["cases"]],
        "train_probe_ids": [r["id"] for r in probe_train],
        "excluded_prior_probe_ids": sorted(banned_probe),
        "excluded_integration_train_ids": sorted(banned_train),
        "summary": {
            "original_questions": len(refs),
            "excluded_questions": len(excluded),
            "exclusion_reason_counts": dict(
                sorted(Counter(r for rs in excluded.values() for r in rs).items())
            ),
            "eligible": {
                s: {"supported": len(v[False]), "STOP": len(v[True])} for s, v in candidates.items()
            },
        },
    }
    data["overlap_checks"] = prior._overlap_checks(data)
    data["cache_plan"] = cache_plan(data)
    _validate(data)
    prior._write_json(
        exclusions_path,
        {
            "source_sha256": PINS[RAW],
            "records": [
                {"id": refs[n][2]["id"], "paragraph_id": refs[n][0], "reasons": rs}
                for n, rs in sorted(excluded.items())
            ],
        },
    )
    data["exclusions"] = {
        "path": exclusions_path.relative_to(ROOT).as_posix(),
        "sha256": prior.file_hash(exclusions_path),
    }
    prior._write_json(output, data)
    return data


def _validate(data: dict) -> None:
    if (
        data["schema"] != SCHEMA
        or data["policy"] != POLICY
        or data["tokenizer_sha256"] != prior.TOKENIZER_SHA256
    ):
        raise ValueError("selection policy identity differs")
    for split, (_, no) in QUOTAS.items():
        rows = data[split]
        if [r["is_impossible"] for r in rows] != [False, False, False, True] * no:
            raise ValueError("selection quotas/order differ")
        for label in (False, True):
            lane = [r for r in rows if r["is_impossible"] == label]
            if lane != sorted(lane, key=prior._rank):
                raise ValueError("within-label QA hash order differs")
        for r in rows:
            d = data["documents"][r["paragraph_id"]]
            chars, spans = prior._word_layout(d["context"])
            if (
                d["context_sha256"] != prior.text_hash(d["context"])
                or d["article_sha256"] != prior.text_hash(d["article_title"])
                or r["question_sha256"] != prior.text_hash(r["question"])
                or r["id_sha256"] != prior.text_hash(r["id"])
                or [list(v) for v in spans] != [list(v) for v in d["word_byte_spans"]]
                or [list(v) for v in prior._word_layout(r["question"])[1]]
                != [list(v) for v in r["question_word_byte_spans"]]
                or (int(d["article_sha256"], 16) % 5 == 0) != (split == "probe")
                or not 0 < d["token_count"] <= 256
                or not 0 < r["question_token_count"] <= 256
            ):
                raise ValueError("original source/question custody differs")
            answers, reasons = _answers(d["context"], r, {"chars": chars, "words": spans})
            if (
                reasons
                or answers != r["answers"]
                or r["accepted_text_hex"]
                != list(dict.fromkeys(a["text"].encode().hex() for a in answers))
            ):
                raise ValueError("annotation custody differs")
    prior._overlap_checks(data)
    train_ids, probe_ids = {r["id"] for r in data["train"]}, {r["id"] for r in data["probe"]}
    if (
        not set(data["small_fit_ids"]) <= train_ids
        or len(data["small_fit_ids"]) != 16
        or set(data["excluded_prior_probe_ids"]) & probe_ids
        or set(data["excluded_integration_train_ids"]) & train_ids
    ):
        raise ValueError("exposed/regression roster boundary differs")
    expected_probe = prior._interleave(
        [r for r in data["train"] if not r["is_impossible"]][:96],
        [r for r in data["train"] if r["is_impossible"]][:32],
    )
    if data["train_probe_ids"] != [r["id"] for r in expected_probe]:
        raise ValueError("fixed TRAIN probe roster differs")


def load_selection(path: Path = SELECTION, expected_sha256: str | None = None) -> dict:
    """Read authenticated fixed data, without tokenization or model calls."""
    if expected_sha256 is None or prior.file_hash(path) != expected_sha256:
        raise ValueError("an exact published selection SHA256 is required")
    data = json.loads(path.read_bytes())
    _validate(data)
    return data


def input_inventory(data: dict, *, include_integration: bool = True) -> dict[str, bytes]:
    """Deduplicate raw inputs only; this view contains no answer annotations."""
    values = [d["context"] for d in data["documents"].values()]
    values += [r["question"] for split in QUOTAS for r in data[split]]
    if include_integration:
        _pin(INTEGRATION)
        integration = json.loads(INTEGRATION.read_bytes())
        values += [d["context"] for d in integration["documents"].values()]
        values += [r["question"] for r in integration["cases"]]
    return dict(sorted({prior.text_hash(s): s.encode("utf-8") for s in values}.items()))


def cache_plan(data: dict) -> dict:
    inputs = input_inventory(data)
    rows = sum(len(prior._word_layout(raw.decode())[1]) for raw in inputs.values())
    _pin(REPORT)
    records = json.loads(REPORT.read_bytes())["input_records"]
    reused = {sha256(bytes.fromhex(r["raw_hex"])).hexdigest(): r for r in records}
    matching = {k: r for k, r in reused.items() if k in inputs}
    reused_rows = sum(r["vector_shape"][0] for r in matching.values())
    plan = {
        "unique_inputs": len(inputs),
        "word_rows": rows,
        "float32_vector_bytes": rows * 384 * 4,
        "offset_int64_equivalent_bytes": rows * 16,
        "raw_bytes": sum(map(len, inputs.values())),
        "reusable_old_inputs": len(matching),
        "reusable_old_word_rows": reused_rows,
        "reused_float32_ram_limit_bytes": reused_rows * 384 * 4,
        "new_memmap_word_rows": rows - reused_rows,
        "new_memmap_bytes": (rows - reused_rows) * 384 * 4,
        "new_encoding_inputs": len(inputs) - len(matching),
        "dtype": "float32",
        "scope": (
            "TRAIN + untaught probes + unchanged integration inputs; vector file excludes "
            "index/JSON/checkpoints; no candidate features"
        ),
        "old_archive_sha256": PINS[ARCHIVE],
        "old_report_sha256": PINS[REPORT],
    }
    index, offset, journal_bytes = {}, 0, 0
    for key, raw in inputs.items():
        record = _cache_record(raw, matching.get(key), offset, "0" * 64)
        if key not in matching:
            offset = record["row_end"]
        index[key] = record
        journal_bytes += len(_journal_line(key, record).encode("utf-8"))
    template = _cache_index(plan, index, _hash(data), "0" * 64)
    plan["serialized_index_bytes"] = len(_json_bytes(template))
    plan["serialized_journal_bytes"] = journal_bytes
    plan["new_disk_bytes"] = (
        plan["new_memmap_bytes"] + plan["serialized_index_bytes"] + journal_bytes
    )
    plan["metadata_byte_plan"] = (
        "exact UTF-8 serialization; future vector hashes replaced by 64 zeroes"
    )
    return plan


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _cache_record(raw: bytes, old: dict | None, offset: int, vector_hash: str) -> dict:
    spans = prior._word_layout(raw.decode())[1]
    custody = {"encoder_sha256": ENCODER_SHA256, "policy": VECTOR_POLICY}
    if old is None:
        custody["origin"] = "single_frozen_encoder_call"
    else:
        custody.update(
            origin="authenticated_saved_float64_cast",
            original_word_input_sha256=old["sha256"],
            original_vector_sha256=old["vectors_sha256"],
        )
    record = {
        "source": raw.decode(),
        "words": [raw[a:b].decode() for a, b in spans],
        "word_starts": [a for a, _ in spans],
        "word_ends": [b for _, b in spans],
        "vectors_sha256": vector_hash,
        "custody": custody,
        "storage": "old_npz" if old is not None else "new_memmap",
    }
    if old is None:
        record.update(row_start=offset, row_end=offset + len(spans))
    else:
        record["old_record"] = {
            k: old[k]
            for k in (
                "array",
                "sha256",
                "vectors_sha256",
                "encoder_sha256",
                "vector_shape",
                "vector_bytes",
            )
        }
    return record


def _journal_line(key: str, record: dict) -> str:
    return (
        json.dumps(
            {
                "input_sha256": key,
                "storage": record["storage"],
                "vectors_sha256": record["vectors_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _cache_index(plan: dict, inputs: dict, selection_hash: str, vector_hash: str) -> dict:
    return {
        "status": "completed",
        "vector_policy": VECTOR_POLICY,
        "word_rows": plan["new_memmap_word_rows"],
        "vector_sha256": vector_hash,
        "vector_bytes": plan["new_memmap_bytes"],
        "inputs": inputs,
        "reused_float32_ram_limit_bytes": plan["reused_float32_ram_limit_bytes"],
        "new_encoder_calls": plan["new_encoding_inputs"],
        "reused_inputs": plan["reusable_old_inputs"],
        "selection_sha256": selection_hash,
        "source_pins": {"report": PINS[REPORT], "archive": PINS[ARCHIVE]},
    }


@dataclass(frozen=True, slots=True)
class WordInput:
    source: bytes
    words: tuple[str, ...]
    word_starts: tuple[int, ...]
    word_ends: tuple[int, ...]
    vectors: np.ndarray
    custody: dict

    @property
    def raw(self) -> bytes:
        return self.source

    @property
    def byte_spans(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.word_starts, self.word_ends, strict=True))

    def __post_init__(self) -> None:
        spans = tuple(prior._word_layout(self.source.decode("utf-8"))[1])
        if (
            tuple(zip(self.word_starts, self.word_ends, strict=True)) != spans
            or self.words != tuple(self.source[a:b].decode() for a, b in spans)
            or self.vectors.dtype != np.float32
            or self.vectors.shape != (len(spans), 384)
            or not np.isfinite(self.vectors).all()
        ):
            raise ValueError(
                "complete original word boundaries and finite float32 vectors required"
            )
        self.vectors.flags.writeable = False

    @property
    def sha256(self) -> str:
        return _hash(
            {
                "input": sha256(self.source).hexdigest(),
                "vectors": sha256(self.vectors.tobytes()).hexdigest(),
                "policy": VECTOR_POLICY,
                "custody": self.custody,
            }
        )

    @classmethod
    def from_vectors(cls, raw: bytes, vectors: np.ndarray, custody: dict) -> WordInput:
        spans = prior._word_layout(raw.decode())[1]
        # bytes backing enforces immutability without retaining a float64 copy.
        array = np.frombuffer(
            np.asarray(vectors, dtype=np.float32).tobytes(), dtype=np.float32
        ).reshape(-1, 384)
        return cls(
            raw,
            tuple(raw[a:b].decode() for a, b in spans),
            tuple(a for a, _ in spans),
            tuple(b for _, b in spans),
            array,
            custody,
        )


@dataclass(frozen=True, slots=True)
class WordDocument:
    """Action-facing source wrapper; no candidate list or float64 conversion."""

    document_id: bytes
    provenance_id: bytes
    tokens: WordInput


def _saved_word(record: dict, array: np.ndarray) -> WordInput:
    raw = bytes.fromhex(record["raw_hex"])
    if (
        array.dtype != np.float64
        or record["encoder_sha256"] != ENCODER_SHA256
        or list(array.shape) != record["vector_shape"]
        or sha256(array.tobytes()).hexdigest() != record["vectors_sha256"]
        or record["sha256"]
        != _hash(
            {
                "raw": raw.hex(),
                "spans": record["byte_spans"],
                "vectors": record["vectors_sha256"],
                "encoder": record["encoder_sha256"],
                "policy": "unicode_regex_words;float32_subtoken_mean;float64_L2_max1e-12",
            }
        )
    ):
        raise ValueError("saved word-input identity differs")
    if [list(v) for v in record["byte_spans"]] != [
        list(v) for v in prior._word_layout(raw.decode())[1]
    ]:
        raise ValueError("saved word boundaries differ")
    return WordInput.from_vectors(
        raw,
        array,
        {
            "encoder_sha256": record["encoder_sha256"],
            "origin": "authenticated_saved_float64_cast",
            "original_word_input_sha256": record["sha256"],
            "original_vector_sha256": record["vectors_sha256"],
            "policy": VECTOR_POLICY,
        },
    )


def load_small_fit() -> tuple[dict, dict[str, WordInput]]:
    """Only the original 32 saved arrays; no encoder or learned state loaded."""
    _pin(OLD)
    fixture = json.loads(OLD.read_bytes())
    path = ROOT / fixture["encoded_inputs"]["path"]
    if prior.file_hash(path) != fixture["encoded_inputs"]["sha256"]:
        raise ValueError("original small-fit input archive differs")
    inputs = {}
    with np.load(path, allow_pickle=False) as archive:
        for record in fixture["saved_inputs"]:
            word = _saved_word(record, archive[record["array"]])
            inputs[sha256(word.source).hexdigest()] = word
    if len(inputs) != 32:
        raise ValueError("small-fit input roster differs")
    return fixture, inputs


class WordCache:
    """New-input memmap plus explicitly bounded lazy old-NPZ float32 copies."""

    def __init__(self, directory: Path, expected_index_sha256: str):
        path = directory / "index.json"
        if prior.file_hash(path) != expected_index_sha256:
            raise ValueError("word-cache index differs")
        self.index = json.loads(path.read_bytes())
        if self.index["status"] != "completed" or self.index["vector_policy"] != VECTOR_POLICY:
            raise ValueError("word cache is incomplete/incompatible")
        vector_path = directory / "vectors.f32"
        if prior.file_hash(vector_path) != self.index["vector_sha256"]:
            raise ValueError("word-cache payload differs")
        self.vectors = np.memmap(
            vector_path, dtype=np.float32, mode="r", shape=(self.index["word_rows"], 384)
        )
        _pin(ARCHIVE)
        self._old_archive = np.load(ARCHIVE, allow_pickle=False)
        self._reused = {}
        self._reused_bytes = 0

    def close(self) -> None:
        self._old_archive.close()
        self._reused.clear()
        self._reused_bytes = 0
        # Existing WordInputs may still reference the read-only mapping.
        self.vectors = None

    def get(self, input_sha256: str) -> WordInput:
        r = self.index["inputs"][input_sha256]
        raw = r["source"].encode()
        if sha256(raw).hexdigest() != input_sha256:
            raise ValueError("cache raw input hash differs")
        if r["storage"] == "old_npz":
            if input_sha256 not in self._reused:
                record = {
                    **r["old_record"],
                    "raw_hex": raw.hex(),
                    "byte_spans": list(zip(r["word_starts"], r["word_ends"], strict=True)),
                }
                word = _saved_word(record, self._old_archive[record["array"]])
                if (
                    self._reused_bytes + word.vectors.nbytes
                    > self.index["reused_float32_ram_limit_bytes"]
                ):
                    raise MemoryError("declared reused-input float32 RAM cache limit exceeded")
                if sha256(word.vectors.tobytes()).hexdigest() != r["vectors_sha256"]:
                    raise ValueError("referenced old input cast differs")
                self._reused[input_sha256] = word
                self._reused_bytes += word.vectors.nbytes
            return self._reused[input_sha256]
        if r["storage"] != "new_memmap":
            raise ValueError("unknown word-cache storage")
        a, b = r["row_start"], r["row_end"]
        v = self.vectors[a:b]
        if sha256(v.tobytes()).hexdigest() != r["vectors_sha256"]:
            raise ValueError("cache input vector hash differs")
        return WordInput(
            raw, tuple(r["words"]), tuple(r["word_starts"]), tuple(r["word_ends"]), v, r["custody"]
        )


def build_cache(
    data: dict,
    directory: Path,
    *,
    encode,
    byte_budget: int,
    deadline_unix: float,
    minimum_free_bytes: int = 2**30,
) -> dict:
    """Explicitly admitted encoding callback, at most once per new raw input.

    The caller owns the encoder and its runtime budget. Partial payload/journal
    remain on failure; only a completed index permits subsequent loading.
    """
    plan = cache_plan(data)
    inputs = input_inventory(data)
    if (
        directory.exists()
        or plan["new_disk_bytes"] > byte_budget
        or shutil.disk_usage(ROOT).free - plan["new_disk_bytes"] < minimum_free_bytes
    ):
        raise ValueError("cache output exists or admitted disk/byte budget is insufficient")
    if time.time() >= deadline_unix:
        raise TimeoutError("cache encoding budget expired before allocation")
    _pin(REPORT)
    _pin(ARCHIVE)
    records = json.loads(REPORT.read_bytes())["input_records"]
    old = {sha256(bytes.fromhex(r["raw_hex"])).hexdigest(): r for r in records}
    directory.mkdir(parents=True)
    vector_path = directory / "vectors.f32"
    vectors = np.memmap(
        vector_path, dtype=np.float32, mode="w+", shape=(plan["new_memmap_word_rows"], 384)
    )
    index, offset, encoded_count, reused_count = {}, 0, 0, 0
    try:
        with (
            np.load(ARCHIVE, allow_pickle=False) as archive,
            (directory / "inputs.jsonl").open("x", encoding="utf-8", newline="\n") as journal,
        ):
            for key, raw in inputs.items():
                if time.time() >= deadline_unix:
                    raise TimeoutError("cache encoding budget expired")
                if key in old:
                    r = old[key]
                    word = _saved_word(r, archive[r["array"]])
                    reused_count += 1
                else:
                    encoding = encode(raw)
                    if (
                        encoding.input_sha256 != key
                        or encoding.encoder_sha256 != ENCODER_SHA256
                        or encoding.input_byte_count != len(raw)
                        or encoding.truncated
                        or encoding.token_count > 256
                    ):
                        raise ValueError("encoder did not preserve complete pinned input")
                    spans = tuple(prior._word_layout(raw.decode())[1])
                    pooled = encoding.pool_spans(spans).astype(np.float64)
                    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
                    word = WordInput.from_vectors(
                        raw,
                        pooled,
                        {
                            "encoder_sha256": encoding.encoder_sha256,
                            "origin": "single_frozen_encoder_call",
                            "policy": VECTOR_POLICY,
                        },
                    )
                    encoded_count += 1
                record = _cache_record(
                    raw, old.get(key), offset, sha256(word.vectors.tobytes()).hexdigest()
                )
                if key not in old:
                    end = offset + len(word.words)
                    vectors[offset:end] = word.vectors
                    offset = end
                index[key] = record
                journal.write(_journal_line(key, record))
                journal.flush()
        vectors.flush()
        if (
            offset != plan["new_memmap_word_rows"]
            or encoded_count != plan["new_encoding_inputs"]
            or reused_count != plan["reusable_old_inputs"]
        ):
            raise ValueError("cache counts differ from admitted plan")
        vector_hash = prior.file_hash(vector_path)
        if time.time() >= deadline_unix:
            raise TimeoutError("cache budget expired before final publication")
        result = _cache_index(plan, index, _hash(data), vector_hash)
        if (
            len(_json_bytes(result)) != plan["serialized_index_bytes"]
            or (directory / "inputs.jsonl").stat().st_size != plan["serialized_journal_bytes"]
        ):
            raise ValueError("serialized cache metadata differs from admitted byte plan")
        prior._write_json(directory / "index.json", result)
        return {
            "index_sha256": prior.file_hash(directory / "index.json"),
            "plan": plan,
            "new_encoder_calls": encoded_count,
            "reused_inputs": reused_count,
        }
    finally:
        vectors.flush()
        del vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.prepare:
        parser.error("only explicit --prepare selection is available; encoding is a separate API")
    result = prepare_selection()
    print(
        json.dumps(
            {
                "selection": str(SELECTION),
                "sha256": prior.file_hash(SELECTION),
                "summary": result["summary"],
                "cache_plan": result["cache_plan"],
            }
        )
    )
