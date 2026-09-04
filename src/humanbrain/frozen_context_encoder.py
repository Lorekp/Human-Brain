"""Pinned, non-generative MiniLM token features for disclosed hybrid experiments.

This adapter performs only tokenization, frozen transformer inference and span
pooling. It has no answer, relation, document-selection, or action-selection head.
The external model's semantic and QA-pair training prior is part of every result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILE_HASHES = {
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    "README.md": "dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0",
}


def _readonly(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class ContextEncoding:
    token_features: np.ndarray
    byte_spans: tuple[tuple[int, int], ...]
    word_ids: tuple[int | None, ...]
    sentence_vector: np.ndarray
    input_sha256: str
    input_byte_count: int
    encoder_sha256: str
    token_count: int
    truncated: bool = False

    def pool_spans(self, spans: tuple[tuple[int, int], ...]) -> np.ndarray:
        """Mean-pool all overlapping subtokens of caller-specified byte spans.

        This preserves caller span ownership; it does not propose lexical roles.
        A span may cover multiple words, but cannot split a subtoken.
        """
        pooled = []
        for start, end in spans:
            if not 0 <= start < end <= self.input_byte_count:
                raise ValueError("span lies outside the encoded input")
            indices = [
                i for i, (left, right) in enumerate(self.byte_spans) if left < end and right > start
            ]
            if not indices:
                raise ValueError("span contains no contextual token")
            if any(self.byte_spans[i][0] < start or self.byte_spans[i][1] > end for i in indices):
                raise ValueError("span splits a contextual subtoken")
            pooled.append(self.token_features[indices].mean(axis=0))
        array = np.asarray(pooled, dtype=np.float32).reshape(len(pooled), 384)
        return _readonly(array)


class FrozenContextEncoder:
    """Load only the audited local CPU encoder; never download during inference."""

    def __init__(
        self,
        model_dir: Path = Path(".models/all-MiniLM-L6-v2"),
        *,
        max_tokens: int = 256,
        threads: int = 1,
    ) -> None:
        if not 2 <= max_tokens <= 512 or threads < 1:
            raise ValueError("max_tokens must be 2..512 and threads must be positive")
        import onnxruntime as ort
        from tokenizers import Tokenizer

        root = Path(model_dir).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
            raise ValueError("encoder manifest does not identify the pinned model revision")
        files = {}
        for name, expected in FILE_HASHES.items():
            path = root / name
            actual = sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"pinned encoder file hash mismatch: {name}")
            files[name] = {"sha256": actual, "bytes": path.stat().st_size}
        self._tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._max_tokens = max_tokens
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(root / "onnx/model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = {item.name: item for item in self._session.get_inputs()}
        if set(inputs) != {"input_ids", "attention_mask", "token_type_ids"} or any(
            item.type != "tensor(int64)" for item in inputs.values()
        ):
            raise ValueError("pinned encoder input signature changed")
        outputs = self._session.get_outputs()
        if (
            len(outputs) != 1
            or outputs[0].name != "last_hidden_state"
            or outputs[0].shape[-1] != 384
        ):
            raise ValueError("pinned encoder output signature changed")
        self._metadata = {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
            "model_card": f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/README.md",
            "files": files,
            "runtime": {name: version(name) for name in ("onnxruntime", "tokenizers", "numpy")},
            "provider": "CPUExecutionProvider",
            "threads": threads,
            "max_tokens": max_tokens,
            "truncation": "reject; caller must declare any chunking",
            "token_dimension": 384,
            "sentence_pooling": "attention-mask mean including special tokens; L2 normalization",
            "span_pooling": "mean overlapping complete non-special subtokens; no normalization",
            "prior": (
                "contrastive sentence training includes QA and code pairs; contamination unknown"
            ),
        }
        self.sha256 = sha256(json.dumps(self._metadata, sort_keys=True).encode()).hexdigest()

    @property
    def metadata(self) -> dict:
        return json.loads(json.dumps(self._metadata))

    def encode(self, text: bytes | str) -> ContextEncoding:
        if isinstance(text, bytes):
            raw = text
            decoded = raw.decode("utf-8", errors="strict")
        elif isinstance(text, str):
            decoded = text
            raw = text.encode("utf-8")
        else:
            raise TypeError("encoder input must be UTF-8 bytes or text")
        if not raw:
            raise ValueError("encoder input must not be empty")
        encoded = self._tokenizer.encode(decoded)
        if len(encoded.ids) > self._max_tokens:
            raise ValueError(
                f"input has {len(encoded.ids)} tokens; declared limit is {self._max_tokens}"
            )
        inputs = {
            "input_ids": np.asarray([encoded.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
            "token_type_ids": np.asarray([encoded.type_ids], dtype=np.int64),
        }
        hidden = self._session.run(["last_hidden_state"], inputs)[0][0]
        if hidden.shape != (len(encoded.ids), 384) or not np.isfinite(hidden).all():
            raise RuntimeError("encoder returned invalid token features")
        mask = np.asarray(encoded.attention_mask, dtype=np.float32)
        sentence = (hidden * mask[:, None]).sum(axis=0) / mask.sum()
        sentence /= max(float(np.linalg.norm(sentence)), 1e-12)
        # Tokenizers' Python string offsets are character positions. Convert
        # through the original text, never through lowercased decoded tokens.
        char_to_byte = [0]
        for character in decoded:
            char_to_byte.append(char_to_byte[-1] + len(character.encode("utf-8")))
        indices, spans, words = [], [], []
        for i, ((left, right), special, word) in enumerate(
            zip(
                encoded.offsets,
                encoded.special_tokens_mask,
                encoded.word_ids,
                strict=True,
            )
        ):
            if special or left == right:
                continue
            indices.append(i)
            spans.append((char_to_byte[left], char_to_byte[right]))
            words.append(word)
        return ContextEncoding(
            _readonly(hidden[indices].astype(np.float32)),
            tuple(spans),
            tuple(words),
            _readonly(sentence.astype(np.float32)),
            sha256(raw).hexdigest(),
            len(raw),
            self.sha256,
            len(encoded.ids),
        )
