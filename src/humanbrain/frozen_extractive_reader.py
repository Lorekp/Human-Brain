"""One pinned external QA reader with source-byte custody and fixed decoding.

This is pretrained language understanding, not simulated-neuron capability.
No task labels, expected answers, downloads or learning enter this adapter.
Bare content is tokenized once, context-only token windows preserve all offsets,
and the pinned RoBERTa pair processor runs once per window. This avoids double
offset trimming and makes context coverage independent of paired overflow logic.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from humanbrain.opaque_document_event_tape import OpaqueDocument

MODEL_ID = "optimum/roberta-base-squad2"
MODEL_REVISION = "db07770d7fa024df076e60947b445b0b3290f8eb"
FILE_HASHES = {
    "model.onnx": "204a84036a76ec7dafad3a4613f9425316b0308d77ad88326795fef6eb64b718",
    "tokenizer.json": "fb35154b5c214e4235de078bfd0890777e06bce973689112c1d06a9157ed0288",
    "config.json": "963a7cc92d3d661821cfbeb87a6e1be54ec9081ef9f9780306275fbe57605a0e",
    "tokenizer_config.json": "03c9ebd44d77bfa63470847561d39826b6bb153f3878f5d3d1c399fd31ddcdbc",
    "special_tokens_map.json": "c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00",
    "README.md": "af371578a7688e2c831ff05624725c2b62c132abd6b9953d7d80214153f0ccde",
}
DECODING = {
    "paired_tokens": 386,
    "query_content_tokens": 64,
    "context_overlap": 128,
    "windows": "bare content encoding; context token windows; pinned pair processor once",
    "answer_context_tokens": 30,
    "arithmetic": "float64 promoted logits; exact comparisons",
    "selection": "global maximum per-window span-minus-CLS margin; <=0 STOP",
    "ties": "different answer bytes STOP; identical bytes first occurrence provenance",
    "answer": "original UTF-8 source slice; no normalization",
}


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExtractiveWindow:
    document_index: int
    document_id_hex: str
    provenance_id_hex: str
    content_sha256: str
    window_index: int
    paired_tokens: int
    context_tokens: int
    context_byte_start: int | None
    context_byte_end: int | None
    legal_span_count: int
    null_score: float
    best_margin: float | None
    span_score: float | None
    start_token: int | None
    end_token: int | None
    byte_start: int | None
    byte_end: int | None
    emitted: bytes | None
    different_answer_tie: bool


@dataclass(frozen=True, slots=True)
class ExtractiveReading:
    emitted: bytes | None
    stopped: bool
    stop_reason: str | None
    best_margin: float | None
    selected: ExtractiveWindow | None
    windows: tuple[ExtractiveWindow, ...]
    window_count: int
    paired_tokens: int
    query_content_tokens: int
    query_sha256: str
    source_sha256: str
    source_bytes: int
    reader_sha256: str
    elapsed_seconds: float


class FrozenExtractiveReader:
    """One sequential CPU session; supplied documents participate without filtering."""

    def __init__(
        self, model_dir: Path = Path(".models/roberta-base-squad2-optimum-db07770")
    ) -> None:
        started = perf_counter()
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(model_dir).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
            raise ValueError("reader manifest does not identify the pinned revision")
        if set(manifest.get("files", {})) != set(FILE_HASHES):
            raise ValueError("reader manifest must contain exactly the six pinned files")
        files = {}
        for name, expected in FILE_HASHES.items():
            path = root / name
            with path.open("rb") as stream:
                actual = file_digest(stream, "sha256").hexdigest()
            size = path.stat().st_size
            declared = manifest["files"][name]
            if (
                actual != expected
                or declared.get("sha256") != actual
                or declared.get("bytes") != size
            ):
                raise ValueError(f"pinned reader file hash/size mismatch: {name}")
            files[name] = {"sha256": actual, "bytes": size}
        self._tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        if (
            self._tokenizer.token_to_id("<s>") != 0
            or self._tokenizer.num_special_tokens_to_add(True) != 4
        ):
            raise ValueError("pinned reader special-token layout changed")
        # Preserve the pinned processor separately. Encode content without
        # processing, truncate only context, then process each pair exactly once.
        self._pair_processor = self._tokenizer.post_processor
        self._tokenizer.post_processor = None
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(root / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if (
            {x.name for x in inputs} != {"input_ids", "attention_mask"}
            or any(x.type != "tensor(int64)" or len(x.shape) != 2 for x in inputs)
            or {x.name for x in outputs} != {"start_logits", "end_logits"}
            or any(x.type != "tensor(float)" or len(x.shape) != 2 for x in outputs)
            or self._session.get_providers() != ["CPUExecutionProvider"]
        ):
            raise ValueError("pinned reader runtime signature/provider changed")
        self._metadata = {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "cc-by-4.0",
            "upstream_tensor_equality_verified": False,
            "files": files,
            "total_file_bytes": sum(x["bytes"] for x in files.values()),
            "runtime": {name: version(name) for name in ("onnxruntime", "tokenizers", "numpy")},
            "provider": "CPUExecutionProvider",
            "threads": 1,
            "graph_inputs": [{"name": x.name, "type": x.type, "shape": x.shape} for x in inputs],
            "graph_outputs": [{"name": x.name, "type": x.type, "shape": x.shape} for x in outputs],
            "decoding": DECODING,
        }
        self.sha256 = _digest(self._metadata)
        self.load_seconds = perf_counter() - started

    @property
    def metadata(self) -> dict:
        return json.loads(json.dumps(self._metadata))

    def read(self, query: bytes, documents: Sequence[OpaqueDocument]) -> ExtractiveReading:
        started = perf_counter()
        if not isinstance(query, bytes):
            raise TypeError("query must be original UTF-8 bytes")
        query_text = query.decode("utf-8", errors="strict")
        encoded_query = self._tokenizer.encode(query_text, add_special_tokens=False)
        query_count = len(encoded_query.ids)
        if query_count > 64:
            raise ValueError("query exceeds 64 content tokens; no truncation permitted")
        documents = tuple(documents)
        if any(not isinstance(doc, OpaqueDocument) for doc in documents):
            raise TypeError("reader evidence must be OpaqueDocument objects")
        sources = [
            (d.document_id.hex(), d.provenance_id.hex(), sha256(d.content).hexdigest())
            for d in documents
        ]
        rows = []
        best = None
        different_tie = False
        for doc_index, doc in enumerate(documents):
            text = doc.content.decode("utf-8", errors="strict")
            byte_offsets = [0]
            for char in text:
                byte_offsets.append(byte_offsets[-1] + len(char.encode("utf-8")))
            context = self._tokenizer.encode(text, add_special_tokens=False)
            original_ids = tuple(context.ids)
            original_offsets = tuple(context.offsets)
            capacity = 386 - query_count - 4
            context.truncate(capacity, stride=128, direction="right")
            chunks = (context, *context.overflowing)
            covered = 0
            for window_index, chunk in enumerate(chunks):
                offset = window_index * (capacity - 128)
                if (
                    tuple(chunk.ids) != original_ids[offset : offset + capacity]
                    or tuple(chunk.offsets) != original_offsets[offset : offset + capacity]
                ):
                    raise ValueError("context windows do not preserve original token coverage")
                covered = offset + len(chunk.ids)
                pair = self._pair_processor.process(encoded_query, chunk, add_special_tokens=True)
                row = self._window(pair, doc, doc_index, window_index, byte_offsets)
                rows.append(row)
                if row.best_margin is None:
                    continue
                if best is None or row.best_margin > best.best_margin:
                    best, different_tie = row, row.different_answer_tie
                elif row.best_margin == best.best_margin:
                    different_tie |= row.different_answer_tie or row.emitted != best.emitted
            if covered != len(original_ids):
                raise ValueError("context windows silently dropped source tokens")
        reason = None
        if best is None:
            reason = "empty_evidence"
        elif best.best_margin <= 0:
            reason = "null_margin"
        elif different_tie:
            reason = "different_answer_tie"
        return ExtractiveReading(
            emitted=None if reason else best.emitted,
            stopped=reason is not None,
            stop_reason=reason,
            best_margin=None if best is None else best.best_margin,
            selected=None if reason else best,
            windows=tuple(rows),
            window_count=len(rows),
            paired_tokens=sum(row.paired_tokens for row in rows),
            query_content_tokens=query_count,
            query_sha256=sha256(query).hexdigest(),
            source_sha256=_digest(sources),
            source_bytes=sum(len(d.content) for d in documents),
            reader_sha256=self.sha256,
            elapsed_seconds=perf_counter() - started,
        )

    def _window(self, pair, doc, doc_index, window_index, byte_offsets) -> ExtractiveWindow:
        count = len(pair.ids)
        if count > 386 or not count or pair.ids[0] != 0 or not pair.special_tokens_mask[0]:
            raise ValueError("invalid paired length or CLS position")
        inputs = {
            "input_ids": np.asarray([pair.ids], dtype=np.int64),
            "attention_mask": np.asarray([pair.attention_mask], dtype=np.int64),
        }
        output = self._session.run(["start_logits", "end_logits"], inputs)
        if len(output) != 2:
            raise ValueError("reader must return two logit arrays")
        logits = [np.asarray(x, dtype=np.float64) for x in output]
        if any(x.shape != (1, count) or not np.isfinite(x).all() for x in logits):
            raise ValueError("reader returned malformed or nonfinite logits")
        starts, ends = (x[0] for x in logits)
        null = float(starts[0]) + float(ends[0])
        if not math.isfinite(null):
            raise ValueError("nonfinite null score")
        context = []
        for i, (sequence, special, mask, offsets) in enumerate(
            zip(
                pair.sequence_ids,
                pair.special_tokens_mask,
                pair.attention_mask,
                pair.offsets,
                strict=True,
            )
        ):
            if sequence != 1 or special or not mask:
                continue
            a, b = offsets
            if not 0 <= a <= b < len(byte_offsets):
                raise ValueError("context token offset outside original source")
            context.append((i, byte_offsets[a], byte_offsets[b]))
        best_margin = score = None
        selected = None
        tied = False
        legal = 0
        for pos, (start, a, first_end) in enumerate(context):
            if first_end <= a:
                continue
            for end, last_start, b in context[pos : pos + 30]:
                if b <= last_start or b <= a:
                    continue
                margin_score = float(starts[start]) + float(ends[end])
                margin = margin_score - null
                if not math.isfinite(margin_score) or not math.isfinite(margin):
                    raise ValueError("nonfinite span score or margin")
                legal += 1
                emitted = doc.content[a:b]
                if best_margin is None or margin > best_margin:
                    best_margin, score = margin, margin_score
                    selected, tied = (start, end, a, b, emitted), False
                elif margin == best_margin:
                    tied |= emitted != selected[4]
        start, end, a, b, emitted = selected or (None, None, None, None, None)
        return ExtractiveWindow(
            doc_index,
            doc.document_id.hex(),
            doc.provenance_id.hex(),
            sha256(doc.content).hexdigest(),
            window_index,
            count,
            len(context),
            min((x[1] for x in context), default=None),
            max((x[2] for x in context), default=None),
            legal,
            null,
            best_margin,
            score,
            start,
            end,
            a,
            b,
            emitted,
            tied,
        )
