"""Small rate-cell span reader with explicit hierarchical abstention.

NumPy manual BPTT and Adam are conventional supervised training of simulated
rate cells. This module owns numerical state, not teacher or execution custody.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from humanbrain.frozen_context_encoder import ContextEncoding

POLICY = "rate_cell_word_span_hierarchical_stop_v1"
PARAMETER_NAMES = (
    "input_weights",
    "recurrent_weights",
    "cell_bias",
    "span_heads",
    "stop_head",
    "stop_bias",
)
PRODUCER_NAMES = PARAMETER_NAMES[:3]


def _hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _readonly(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _word_spans(raw: bytes) -> tuple[tuple[int, int], ...]:
    text = raw.decode("utf-8", errors="strict")
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + len(char.encode()))
    return tuple((offsets[m.start()], offsets[m.end()]) for m in re.finditer(r"\w+", text))


@dataclass(frozen=True, slots=True)
class TokenInput:
    raw: bytes
    byte_spans: tuple[tuple[int, int], ...]
    vectors: np.ndarray
    encoder_sha256: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.raw, bytes) or self.byte_spans != _word_spans(self.raw):
            raise ValueError("word spans must preserve original UTF-8 regex-word boundaries")
        array = np.asarray(self.vectors, dtype=np.float64)
        if array.shape != (len(self.byte_spans), 384) or not np.isfinite(array).all():
            raise ValueError("finite aligned 384-dimensional word vectors required")
        norms = np.linalg.norm(array, axis=1)
        if np.any((norms > 1e-12) & (np.abs(norms - 1) > 1e-10)):
            raise ValueError("word vectors must be unit normalized or zero")
        array = _readonly(array)
        object.__setattr__(self, "vectors", array)
        object.__setattr__(
            self,
            "sha256",
            _hash(
                {
                    "raw": self.raw.hex(),
                    "spans": self.byte_spans,
                    "vectors": _array_hash(array),
                    "encoder": self.encoder_sha256,
                    "policy": "unicode_regex_words;float32_subtoken_mean;float64_L2_max1e-12",
                }
            ),
        )

    @classmethod
    def from_encoding(cls, raw: bytes, encoding: ContextEncoding) -> TokenInput:
        if (
            encoding.input_sha256 != sha256(raw).hexdigest()
            or encoding.input_byte_count != len(raw)
            or encoding.truncated
        ):
            raise ValueError("encoding does not authenticate the complete original input")
        spans = _word_spans(raw)
        vectors = encoding.pool_spans(spans).astype(np.float64)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return cls(raw, spans, vectors, encoding.encoder_sha256)

    def record(self) -> dict:
        return {
            "raw_hex": self.raw.hex(),
            "byte_spans": self.byte_spans,
            "encoder_sha256": self.encoder_sha256,
            "sha256": self.sha256,
            "vectors_sha256": _array_hash(self.vectors),
            "vector_shape": list(self.vectors.shape),
            "vector_bytes": self.vectors.nbytes,
        }


@dataclass(frozen=True, slots=True)
class TokenDocument:
    document_id: bytes
    provenance_id: bytes
    tokens: TokenInput


@dataclass(frozen=True, slots=True)
class TokenWindow:
    document_index: int
    word_start: int
    word_end: int
    vectors: np.ndarray


@dataclass(frozen=True, slots=True)
class SpanCandidate:
    window_index: int
    document_index: int
    word_start: int
    word_end: int
    byte_start: int
    byte_end: int
    emitted: bytes


@dataclass(frozen=True, slots=True, init=False)
class SpanInventory:
    """Immutable inputs and deterministic window/start/end occurrence ordering."""

    documents: tuple[TokenDocument, ...]
    window_words: int
    windows: tuple[TokenWindow, ...]
    candidates: tuple[SpanCandidate, ...]
    window_candidate_ranges: tuple[tuple[int, int], ...]
    local_spans: tuple[np.ndarray, ...]
    sha256: str

    def __init__(self, documents: tuple[TokenDocument, ...], *, window_words: int = 48) -> None:
        if type(window_words) is not int or window_words < 1:
            raise ValueError("positive integer window size required")
        object.__setattr__(self, "documents", tuple(documents))
        object.__setattr__(self, "window_words", window_words)
        windows, candidates, ranges, spans = [], [], [], []
        for doc_index, doc in enumerate(self.documents):
            if not doc.document_id or not doc.provenance_id:
                raise ValueError("opaque document and provenance identities required")
            tokens = doc.tokens
            for left in range(0, len(tokens.vectors), window_words):
                right = min(left + window_words, len(tokens.vectors))
                window_index = len(windows)
                windows.append(TokenWindow(doc_index, left, right, tokens.vectors[left:right]))
                begin = len(candidates)
                local = []
                for start in range(left, right):
                    for end in range(start + 1, right + 1):
                        a, b = tokens.byte_spans[start][0], tokens.byte_spans[end - 1][1]
                        candidates.append(
                            SpanCandidate(
                                window_index, doc_index, start, end, a, b, tokens.raw[a:b]
                            )
                        )
                        local.append((start - left, end - left))
                ranges.append((begin, len(candidates)))
                spans.append(_readonly(np.asarray(local, dtype=np.int64).reshape(-1, 2)))
        object.__setattr__(self, "windows", tuple(windows))
        object.__setattr__(self, "candidates", tuple(candidates))
        object.__setattr__(self, "window_candidate_ranges", tuple(ranges))
        object.__setattr__(self, "local_spans", tuple(spans))
        object.__setattr__(
            self,
            "sha256",
            _hash(
                {
                    "policy": POLICY,
                    "window_words": window_words,
                    "documents": [
                        (d.document_id.hex(), d.provenance_id.hex(), d.tokens.sha256)
                        for d in self.documents
                    ],
                }
            ),
        )

    def gold_indices(self, answer: bytes) -> tuple[int, ...]:
        return tuple(i for i, c in enumerate(self.candidates) if c.emitted == answer)

    def candidate_manifest(self) -> list[dict]:
        return [
            {
                "window_index": c.window_index,
                "document_index": c.document_index,
                "document_id_hex": self.documents[c.document_index].document_id.hex(),
                "provenance_id_hex": self.documents[c.document_index].provenance_id.hex(),
                "content_sha256": sha256(self.documents[c.document_index].tokens.raw).hexdigest(),
                "word_start": c.word_start,
                "word_end": c.word_end,
                "byte_start": c.byte_start,
                "byte_end": c.byte_end,
                "emitted_hex": c.emitted.hex(),
            }
            for c in self.candidates
        ]


@dataclass(frozen=True, slots=True)
class RateCellPrediction:
    candidate_index: int | None
    stopped: bool
    p_stop: float
    stop_logit: float
    span_scores: np.ndarray
    checkpoint_sha256: str
    query_sha256: str
    inventory_sha256: str
    stop_reason: str | None


class RateCellReader:
    """One recurrent numerical learner; external adapter authorizes each offer."""

    def __init__(
        self,
        *,
        mode: str = "trainable",
        seed: int = 0,
        learning_rate: float = 0.001,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        if mode not in ("trainable", "frozen_producer", "bypass", "frozen"):
            raise ValueError("unknown learning control")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("positive finite learning rate required")
        if not np.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
            raise ValueError("positive finite gradient clip required")
        self.mode = mode
        self.learning_rate, self.gradient_clip_norm = learning_rate, gradient_clip_norm
        rng = np.random.default_rng(seed)
        self._weights = {
            "input_weights": rng.standard_normal((64, 386)) / np.sqrt(386),
            "recurrent_weights": 0.9 * np.eye(64),
            "cell_bias": np.zeros(64),
            "span_heads": rng.standard_normal((3, 64)) * (0.01 / np.sqrt(64)),
            "stop_head": rng.standard_normal(64) * (0.01 / np.sqrt(64)),
            "stop_bias": np.zeros(1),
        }
        self._first = {k: np.zeros_like(v) for k, v in self._weights.items()}
        self._second = {k: np.zeros_like(v) for k, v in self._weights.items()}
        self._steps = {k: 0 for k in self._weights}
        self.offer_count = self.update_count = 0
        self.parameters = {
            "policy": POLICY,
            "mode": mode,
            "seed": seed,
            "cells": 64,
            "leak": 0.5,
            "learning_rate": learning_rate,
            "gradient_clip_norm": gradient_clip_norm,
            "optimizer": "Adam;beta1=.9;beta2=.999;epsilon=1e-8;no_decay;per_parameter_steps",
            "mask": "STOP_skips_span_parameters_moments_steps;clip_active_coordinates_only",
            "gate_pool": "mean_all_document_word_states_token_weighted;exclude_query_padding",
            "inference": "p_stop>=.5_STOP;otherwise_unique_span_max;tie_STOP",
            "bypass": "tanh(W0[doc,0,1]+W0[mean(query),1,0]+b0)",
        }

    @property
    def checkpoint_sha256(self) -> str:
        return _hash(
            {
                "parameters": self.parameters,
                "arrays": {k: _array_hash(v) for k, v in self.checkpoint_arrays().items()},
                "offers": self.offer_count,
                "updates": self.update_count,
            }
        )

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        arrays = {"weight_" + k: v.copy() for k, v in self._weights.items()}
        arrays.update({"adam_first_" + k: v.copy() for k, v in self._first.items()})
        arrays.update({"adam_second_" + k: v.copy() for k, v in self._second.items()})
        arrays["adam_steps"] = np.asarray([self._steps[k] for k in PARAMETER_NAMES], dtype=np.int64)
        return arrays

    def snapshot(self) -> dict:
        arrays = self.checkpoint_arrays()
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "parameters": dict(self.parameters),
            "offer_count": self.offer_count,
            "update_count": self.update_count,
            "adam_steps": dict(self._steps),
            "allocated_parameter_bytes": sum(v.nbytes for v in self._weights.values()),
            "allocated_optimizer_bytes": sum(v.nbytes for v in self._first.values()) * 2 + 48,
            "array_sha256": {k: _array_hash(v) for k, v in arrays.items()},
        }

    def _forward(self, query: TokenInput, inventory: SpanInventory) -> dict:
        windows = inventory.windows
        width = max((len(w.vectors) for w in windows), default=0)
        query_count, batch = len(query.vectors), len(windows)
        time_count = query_count + width
        x = np.zeros((batch, time_count, 386))
        mask = np.zeros((batch, time_count, 1))
        for i, window in enumerate(windows):
            count = len(window.vectors)
            x[i, :query_count, :384] = query.vectors
            x[i, :query_count, 384] = 1
            x[i, query_count : query_count + count, :384] = window.vectors
            x[i, query_count : query_count + count, 385] = 1
            mask[i, : query_count + count] = 1
        w = self._weights
        states = np.zeros((batch, time_count + 1, 64))
        activation = np.zeros((batch, time_count, 64))
        drive = x @ w["input_weights"].T + w["cell_bias"]
        if self.mode == "bypass":
            query_mean = query.vectors.mean(axis=0) if query_count else np.zeros(384)
            q = np.r_[query_mean, 1.0, 0.0]
            activation = np.tanh(drive + q @ w["input_weights"].T)
            states[:, 1:] = activation * mask
        else:
            for t in range(time_count):
                activation[:, t] = np.tanh(drive[:, t] + states[:, t] @ w["recurrent_weights"].T)
                states[:, t + 1] = states[:, t] + mask[:, t] * 0.5 * (
                    activation[:, t] - states[:, t]
                )
        scores = np.empty(len(inventory.candidates))
        pooled = np.zeros(64)
        document_count = sum(len(window.vectors) for window in windows)
        for i, window in enumerate(windows):
            h = states[i, query_count + 1 : query_count + 1 + len(window.vectors)]
            pooled += h.sum(axis=0)
            local = inventory.local_spans[i]
            start, end = local[:, 0], local[:, 1]
            heads = h @ w["span_heads"].T
            prefix = np.r_[0.0, np.cumsum(heads[:, 2])]
            begin, finish = inventory.window_candidate_ranges[i]
            scores[begin:finish] = (
                heads[start, 0] + heads[end - 1, 1] + (prefix[end] - prefix[start]) / (end - start)
            )
        pooled /= max(1, document_count)
        stop_logit = float(pooled @ w["stop_head"] + w["stop_bias"][0])
        p_stop = float(np.exp(-np.logaddexp(0.0, -stop_logit)))
        if not np.isfinite(scores).all() or not np.isfinite(stop_logit):
            raise ValueError("nonfinite forward state")
        return {
            "x": x,
            "mask": mask,
            "states": states,
            "activation": activation,
            "scores": scores,
            "pooled": pooled,
            "stop_logit": stop_logit,
            "p_stop": p_stop,
            "query_count": query_count,
            "document_count": document_count,
        }

    def predict(self, query: TokenInput, inventory: SpanInventory) -> RateCellPrediction:
        before = self.checkpoint_sha256
        forward = self._forward(query, inventory)
        scores = forward["scores"]
        index, reason = None, None
        if not len(scores):
            reason = "empty_inventory"
        elif forward["p_stop"] >= 0.5:
            reason = "answerability_STOP"
        else:
            winners = np.flatnonzero(scores == scores.max())
            if len(winners) == 1:
                index = int(winners[0])
            else:
                reason = "span_tie"
        if self.checkpoint_sha256 != before:
            raise AssertionError("prediction changed learning state")
        return RateCellPrediction(
            index,
            index is None,
            forward["p_stop"],
            forward["stop_logit"],
            _readonly(scores),
            before,
            query.sha256,
            inventory.sha256,
            reason,
        )

    def external_only(self, query: TokenInput, inventory: SpanInventory) -> RateCellPrediction:
        q = query.vectors.mean(axis=0) if len(query.vectors) else np.zeros(384)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = np.empty(len(inventory.candidates))
        for i, window in enumerate(inventory.windows):
            prefix = np.vstack((np.zeros((1, 384)), np.cumsum(window.vectors, axis=0)))
            start, end = inventory.local_spans[i].T
            means = (prefix[end] - prefix[start]) / (end - start)[:, None]
            means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
            begin, finish = inventory.window_candidate_ranges[i]
            scores[begin:finish] = means @ q
        winners = np.flatnonzero(scores == scores.max()) if len(scores) else []
        index = int(winners[0]) if len(winners) == 1 else None
        return RateCellPrediction(
            index,
            index is None,
            0.0,
            0.0,
            _readonly(scores),
            self.checkpoint_sha256,
            query.sha256,
            inventory.sha256,
            None if index is not None else "empty_or_cosine_tie",
        )

    @staticmethod
    def _objective(forward: dict, gold_indices: tuple[int, ...] | None) -> tuple:
        scores = forward["scores"]
        stop = gold_indices is None
        gate_loss = float(
            np.logaddexp(0.0, forward["stop_logit"]) - float(stop) * forward["stop_logit"]
        )
        coefficients = np.zeros_like(scores)
        span_loss = 0.0
        if not stop:
            if (
                not gold_indices
                or len(set(gold_indices)) != len(gold_indices)
                or any(type(i) is not int or not 0 <= i < len(scores) for i in gold_indices)
            ):
                raise ValueError("supported gold must be a nonempty unique occurrence subset")
            chosen = scores[list(gold_indices)]
            all_exp, gold_exp = np.exp(scores - scores.max()), np.exp(chosen - chosen.max())
            coefficients = all_exp / all_exp.sum()
            coefficients[list(gold_indices)] -= gold_exp / gold_exp.sum()
            span_loss = float(
                scores.max() - chosen.max() + np.log(all_exp.sum()) - np.log(gold_exp.sum())
            )
        return gate_loss + span_loss, gate_loss, span_loss, coefficients, forward["p_stop"] - stop

    def loss_and_gradients(
        self, query: TokenInput, inventory: SpanInventory, gold_indices: tuple[int, ...] | None
    ) -> tuple[float, dict, dict]:
        f = self._forward(query, inventory)
        loss, gate_loss, span_loss, coefficients, gate_derivative = self._objective(f, gold_indices)
        w = self._weights
        gradient = {k: np.zeros_like(v) for k, v in w.items()}
        gradient["stop_head"] = gate_derivative * f["pooled"]
        gradient["stop_bias"][0] = gate_derivative
        dh = np.zeros_like(f["states"])
        q = f["query_count"]
        for i, window in enumerate(inventory.windows):
            count = len(window.vectors)
            h = f["states"][i, q + 1 : q + 1 + count]
            local = inventory.local_spans[i]
            start, end = local[:, 0], local[:, 1]
            begin, finish = inventory.window_candidate_ranges[i]
            coef = coefficients[begin:finish]
            start_weight, end_weight = np.zeros(count), np.zeros(count)
            np.add.at(start_weight, start, coef)
            np.add.at(end_weight, end - 1, coef)
            difference = np.zeros(count + 1)
            middle = coef / (end - start)
            np.add.at(difference, start, middle)
            np.add.at(difference, end, -middle)
            mid_weight = np.cumsum(difference)[:-1]
            weights = np.stack((start_weight, end_weight, mid_weight))
            gradient["span_heads"] += weights @ h
            dh[i, q + 1 : q + 1 + count] = weights.T @ w["span_heads"] + gate_derivative * w[
                "stop_head"
            ] / max(1, f["document_count"])
        if self.mode != "bypass":
            delta = np.zeros_like(f["activation"])
            for t in range(f["activation"].shape[1] - 1, -1, -1):
                delta[:, t] = (
                    0.5 * f["mask"][:, t] * dh[:, t + 1] * (1 - f["activation"][:, t] ** 2)
                )
                dh[:, t] += (1 - 0.5 * f["mask"][:, t]) * dh[:, t + 1] + delta[:, t] @ w[
                    "recurrent_weights"
                ]
            flat = delta.reshape(-1, 64)
            gradient["input_weights"] = flat.T @ f["x"].reshape(-1, 386)
            gradient["recurrent_weights"] = flat.T @ f["states"][:, :-1].reshape(-1, 64)
            gradient["cell_bias"] = flat.sum(axis=0)
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradient.values()):
            raise ValueError("nonfinite objective or gradient")
        audit = {
            "loss": loss,
            "gate_loss": gate_loss,
            "span_loss": span_loss,
            "stop_logit": f["stop_logit"],
            "p_stop": f["p_stop"],
            "scores_sha256": _array_hash(f["scores"]),
            "document_word_count": f["document_count"],
            "recurrent_cell_steps": 0 if self.mode == "bypass" else int(f["mask"].sum()) * 64,
            "trajectory_array_bytes": f["states"].nbytes + f["activation"].nbytes,
            "input_array_bytes": f["x"].nbytes,
            "span_score_array_bytes": f["scores"].nbytes,
        }
        return float(loss), gradient, audit

    def observe(
        self, query: TokenInput, inventory: SpanInventory, gold_indices: tuple[int, ...] | None
    ) -> dict:
        before = self.checkpoint_sha256
        _, gradients, audit = self.loss_and_gradients(query, inventory, gold_indices)
        active = (
            []
            if self.mode == "frozen"
            else [
                k
                for k in PARAMETER_NAMES
                if not (
                    (self.mode in ("frozen_producer", "bypass") and k in PRODUCER_NAMES)
                    or (gold_indices is None and k == "span_heads")
                )
            ]
        )
        norm = float(np.sqrt(sum(float(np.sum(gradients[k] ** 2)) for k in active)))
        if not np.isfinite(norm):
            raise ValueError("nonfinite active gradient norm; no state published")
        scale = min(1.0, self.gradient_clip_norm / norm) if norm else 1.0
        steps_before = dict(self._steps)
        proposed = {}
        for k in active:
            g = gradients[k] * scale
            step = self._steps[k] + 1
            first = 0.9 * self._first[k] + 0.1 * g
            second = 0.999 * self._second[k] + 0.001 * (g * g)
            corrected_first = first / (1 - 0.9**step)
            corrected_second = second / (1 - 0.999**step)
            weight = self._weights[k] - self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
            if not all(np.isfinite(a).all() for a in (first, second, weight)):
                raise ValueError("nonfinite Adam proposal; no state published")
            proposed[k] = (weight, first, second, step)
        changed = any(
            not np.array_equal(self._weights[k], value[0]) for k, value in proposed.items()
        )
        for k, (weight, first, second, step) in proposed.items():
            self._weights[k], self._first[k], self._second[k], self._steps[k] = (
                weight,
                first,
                second,
                step,
            )
        self.offer_count += 1
        self.update_count += int(changed)
        after_forward = self._forward(query, inventory)
        after_loss, after_gate, after_span, _, _ = self._objective(after_forward, gold_indices)
        return {
            "mode": self.mode,
            "before": before,
            "after": self.checkpoint_sha256,
            "query_sha256": query.sha256,
            "inventory_sha256": inventory.sha256,
            "gold_indices": None if gold_indices is None else list(gold_indices),
            "before_objective": audit,
            "loss_after": after_loss,
            "gate_loss_after": after_gate,
            "span_loss_after": after_span,
            "gradient_sha256": {k: _array_hash(v) for k, v in gradients.items()},
            "active_parameters": active,
            "gradient_norm": norm,
            "clip_scale": scale,
            "adam_steps_before": steps_before,
            "adam_steps_after": dict(self._steps),
            "parameter_update_applied": changed,
            "native_update_applied": False,
            "offer_count": self.offer_count,
            "update_count": self.update_count,
        }
