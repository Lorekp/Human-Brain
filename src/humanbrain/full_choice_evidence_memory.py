"""Supervised full-inventory acquisition on the existing fixed BINDING features.

This is SGD teacher imitation, not executed-negative three-factor or AROW
learning. Only the action readout uses the compiled GLIF race. The inherited
source store, features and opaque decision format retain their old contracts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from humanbrain.contextual_evidence_memory import (
    ContextualEvidenceDecision,
    ContextualEvidenceMemory,
    _digest,
    _unit,
    _words,
)

_POLICY = "full_choice_exact_answer_marginal_sgd_v1"
_AFFINE = (
    "full_choice_learned_stop_affine_v1;delta=candidate_logit-STOP_logit;"
    "R=max(1,max(abs(delta)));I=600.1+200*delta/R;STOP=600.1;"
    "linear_glif_endpoint_interpolation_v1;equal_or_adjacent_times_STOP"
)


def _array_hash(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def full_choice_loss_gradient(
    features: np.ndarray, weights: np.ndarray, stop: float, gold: tuple[int, ...]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return loss, effective-weight-plus-STOP gradient, and all logits.

    Gold indices include every authorized output occurrence, or the final STOP
    index. Competition gradients are surrogate credit, not execution labels.
    """
    if (
        features.ndim != 2 or weights.shape != (features.shape[1],)
        or not np.isfinite(features).all() or not np.isfinite(weights).all()
        or not np.isfinite(stop)
    ):
        raise ValueError("full-choice features and parameters must be finite and aligned")
    if (
        not gold or len(set(gold)) != len(gold)
        or any(type(i) is not int or not 0 <= i <= len(features) for i in gold)
    ):
        raise ValueError("gold indices must be a nonempty unique subset of the inventory")
    logits = np.append(features @ weights, stop)
    if not np.isfinite(logits).all():
        raise ValueError("full-choice logits must be finite")
    shifted = np.exp(logits - logits.max())
    probabilities = shifted / shifted.sum()
    chosen = logits[list(gold)]
    gold_shifted = np.exp(chosen - chosen.max())
    coefficients = probabilities.copy()
    coefficients[list(gold)] -= gold_shifted / gold_shifted.sum()
    loss = float(
        (logits.max() - chosen.max()) + np.log(shifted.sum()) - np.log(gold_shifted.sum())
    )
    gradient = np.append(features.T @ coefficients[:-1], coefficients[-1])
    if not np.isfinite(gradient).all() or not np.isfinite(loss):
        raise ValueError("full-choice objective is nonfinite")
    return loss, gradient, logits


@dataclass(frozen=True, slots=True)
class FullChoiceLesson:
    """External immutable teacher evidence; never read by inference."""

    original_decision: ContextualEvidenceDecision
    answer: bytes | None
    teacher_id: str
    source_pool_sha256: str
    candidate_count: int
    attempt_receipt_json: bytes
    teacher_receipt_json: bytes
    sha256: str

    def record(self) -> dict:
        """JSON-safe original evidence, storing each receipt/vector only once."""
        decision = asdict(self.original_decision)
        decision["query_hex"] = decision.pop("query").hex()
        emitted = decision.pop("emitted")
        decision["emitted_hex"] = None if emitted is None else emitted.hex()
        return {
            "original_decision": decision,
            "answer_hex": None if self.answer is None else self.answer.hex(),
            "teacher_id": self.teacher_id, "source_pool_sha256": self.source_pool_sha256,
            "candidate_count": self.candidate_count,
            "attempt_receipt": json.loads(self.attempt_receipt_json),
            "teacher_receipt": json.loads(self.teacher_receipt_json), "sha256": self.sha256,
        }


def _lesson_digest(lesson: FullChoiceLesson) -> str:
    return _digest({
        "policy": _POLICY,
        "original_decision_sha256": lesson.original_decision.sha256,
        "answer": None if lesson.answer is None else lesson.answer.hex(),
        "teacher_id": lesson.teacher_id,
        "source_pool_sha256": lesson.source_pool_sha256,
        "candidate_count": lesson.candidate_count,
        "attempt_receipt_sha256": sha256(lesson.attempt_receipt_json).hexdigest(),
        "teacher_receipt_sha256": sha256(lesson.teacher_receipt_json).hexdigest(),
    })


class FullChoiceEvidenceMemory(ContextualEvidenceMemory):
    """Fixed BINDING source memory with a separate authenticated SGD contract.

    Reuses private feature/decision seams within the package; no previous
    learner, native learning rule, or frozen protocol is changed.
    """

    def __init__(
        self, encoder, *, learning_rate: float = 1 / 256,
        gradient_clip_norm: float = 1.0, learning_enabled: bool = True,
    ) -> None:
        if not np.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if type(learning_enabled) is not bool:
            raise ValueError("learning_enabled must be a bool")
        super().__init__(
            encoder, learning_rate=learning_rate, learning_enabled=learning_enabled,
            interaction_gain=16.0, candidate_interactions=True,
        )
        self.gradient_clip_norm = float(gradient_clip_norm)
        self._stop_score = np.zeros(1, dtype=np.float64)
        self._training_pool: str | None = None
        self._training_count = 0

    @property
    def parameters(self) -> dict:
        return {
            "learning_rule": _POLICY, "learning_rate": self.learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_coordinates": "effective_GO_minus_NO_and_STOP",
            "feature_dimension": 1621, "initial_state": "zero_synapses_and_STOP",
            "feature_norm_squared_bound": 22019.0,
            "teacher_scope": "exact_output_imitation; competitors_not_observed_failures",
        }

    @property
    def readout_policy(self) -> dict:
        return {"default": _AFFINE, "raw_reference": "unique_max_including_learned_STOP"}

    @property
    def source_pool(self) -> dict:
        return {
            "sealed_sha256": self._training_pool,
            "candidate_count": self._training_count,
            "current_sha256": self._storage_root,
            "learning_allowed": self._training_pool == self._storage_root,
        }

    def seal_training_pool(self) -> str:
        if self._training_pool is not None and self._training_pool != self._storage_root:
            raise ValueError("the sealed TRAIN source pool cannot be replaced")
        self._training_pool = self._storage_root
        self._training_count = len(self._candidates)
        return self._training_pool

    @property
    def synapse_sha256(self) -> str:
        return sha256(
            self._weights.tobytes() + self._bias.tobytes() + self._stop_score.tobytes()
        ).hexdigest()

    @property
    def checkpoint_sha256(self) -> str:
        return _digest({
            "parent_contract_state": super().checkpoint_sha256,
            "parameters": self.parameters, "source_pool": self.source_pool,
        })

    @property
    def storage_summary(self) -> dict:
        summary = super().storage_summary
        summary["synapse_array_bytes"] += self._stop_score.nbytes
        summary["retained_array_and_byte_payload_bytes"] += self._stop_score.nbytes
        summary["plastic_synapse_count"] += 1
        summary["nonzero_plastic_synapses"] += int(np.count_nonzero(self._stop_score))
        summary["supervised_offer_count"] = self.feedback_count
        summary["sgd_update_count"] = self.update_count
        summary["native_learning_update_count"] = 0
        summary["learning_rule"] = _POLICY
        summary["sealed_pool_identity_bytes"] = 0 if self._training_pool is None else 72
        summary["payload_bytes"] += summary["sealed_pool_identity_bytes"]
        summary["retained_array_and_byte_payload_bytes"] += summary["sealed_pool_identity_bytes"]
        return summary

    def checkpoint_arrays(self, *, include_learning_state: bool = False) -> dict:
        return {**super().checkpoint_arrays(), "stop_score": self._stop_score.copy()}

    def snapshot(self) -> dict:
        return {
            "checkpoint_sha256": self.checkpoint_sha256,
            "synapse_sha256": self.synapse_sha256, "stop_score": float(self._stop_score[0]),
            "parameters": self.parameters, "source_pool": self.source_pool,
            "storage": self.storage_summary,
        }

    def feature_rows(self, query: bytes) -> np.ndarray:
        """Read-only full-inventory features for independent numerical audit."""
        rows, _ = self._features(query)
        return np.frombuffer(rows.tobytes(), dtype=np.float64).reshape(rows.shape)

    def candidate_manifest(self) -> tuple[dict, ...]:
        """All occurrence custody, without semantic selection or teacher labels."""
        return tuple({
            "document_index": c.document_index, "byte_start": c.start, "byte_end": c.end,
            "emitted_hex": c.emitted.hex(),
            "document_id_hex": self._documents[c.document_index].document_id.hex(),
            "provenance_id_hex": self._documents[c.document_index].provenance_id.hex(),
            "content_sha256": sha256(self._documents[c.document_index].content).hexdigest(),
        } for c in self._candidates)

    def export_audit_basis(self, queries: tuple[bytes, ...]) -> dict:
        """Compact immutable inputs for offline full-feature reconstruction.

        No labels and no N-by-1621 per-query matrices are retained. The caller
        saves the returned arrays once alongside JSON-native metadata.
        """
        parent = self.checkpoint_sha256
        sentences, word_vectors, query_metadata = [], [], []
        for query in queries:
            encoded = self.encoder.encode(query)
            spans = _words(query)
            vectors = [_unit(v) for v in encoded.pool_spans(spans)]
            query_metadata.append({
                "query_hex": query.hex(), "query_sha256": sha256(query).hexdigest(),
                "words": [query[a:b].decode("utf-8").casefold() for a, b in spans],
                "word_offset": len(word_vectors), "word_count": len(vectors),
            })
            sentences.append(_unit(encoded.sentence_vector))
            word_vectors.extend(vectors)
        result = {
            "arrays": {
                "candidate_vectors": self._vectors.copy(),
                "context_vectors": np.asarray(self._contexts, dtype=np.float64).reshape(-1, 384),
                "candidate_window_indices": np.asarray(
                    [c.window_index for c in self._candidates], dtype=np.int64
                ),
                "query_sentence_vectors": np.asarray(sentences).reshape(-1, 384),
                "query_word_vectors": np.asarray(word_vectors).reshape(-1, 384),
            },
            "metadata": {
                "checkpoint_sha256": parent, "encoder_sha256": self.encoder.sha256,
                "representation": self.representation_policy, "source_pool": self.source_pool,
                "readout_policy": self.readout_policy, "parameters": self.parameters,
                "queries": query_metadata, "candidates": self.candidate_manifest(),
                "documents": [{
                    "document_id_hex": d.document_id.hex(),
                    "provenance_id_hex": d.provenance_id.hex(), "content_hex": d.content.hex(),
                } for d in self._documents],
                "interaction_windows": [{
                    "candidate_start": w.candidate_start, "words": list(w.words),
                    "word_candidate_indices": w.word_candidate_indices.tolist(),
                    "candidate_word_spans": w.candidate_word_spans.tolist(),
                } for w in self._interaction_windows],
            },
        }
        if parent != self.checkpoint_sha256:
            raise RuntimeError("audit export changed learner state")
        return result

    def _logits(self, rows: np.ndarray) -> np.ndarray:
        logits = np.append(self._raw_drives(rows), self._stop_score[0])
        if not np.isfinite(logits).all():
            raise ValueError("full-choice logits must be finite")
        return logits

    def score_inventory(self, query: bytes) -> dict:
        rows, _ = self._features(query)
        return {
            "checkpoint_sha256": self.checkpoint_sha256, "storage_sha256": self._storage_root,
            "logits": self._logits(rows).tolist(), "stop_index": len(rows),
            "features_sha256": _array_hash(rows),
        }

    def raw_score_reference(self, query: bytes) -> ContextualEvidenceDecision:
        rows, _ = self._features(query)
        scores = self._logits(rows)
        best = np.flatnonzero(scores == scores.max())
        index = int(best[0]) if len(best) == 1 and best[0] < len(rows) else None
        return self._decision(
            query, rows, index, "full_choice_raw", "learned_stop_or_tie" if index is None else None
        )

    def respond_affine_interpolated(self, query: bytes) -> ContextualEvidenceDecision:
        rows, _ = self._features(query)
        logits = self._logits(rows)
        delta = logits[:-1] - logits[-1]
        if not np.isfinite(delta).all():
            raise ValueError("full-choice relative drives must be finite")
        scale = max(1.0, float(np.max(np.abs(delta)))) if len(delta) else 1.0
        currents = np.append(600.1 + 200.0 * (delta / scale), 600.1)
        return self._interpolated_decision(query, rows, currents, "full_choice_affine", _AFFINE)

    respond = respond_affine_interpolated
    conventional = raw_score_reference

    def _require_pool(self) -> None:
        if self._training_pool is None or self._storage_root != self._training_pool:
            raise ValueError("teaching requires the unchanged sealed TRAIN source pool")

    def _validate_original(self, decision: ContextualEvidenceDecision, rows: np.ndarray) -> None:
        values = asdict(decision)
        expected = values.pop("sha256")
        values["query"] = decision.query.hex()
        values["emitted"] = None if decision.emitted is None else decision.emitted.hex()
        if expected != _digest(values) or decision.storage_sha256 != self._training_pool:
            raise ValueError("original decision custody or TRAIN source pool changed")
        if decision.mode not in ("full_choice_raw", "full_choice_affine", "external_only"):
            raise ValueError("teacher requires an ordinary full-choice or external action decision")
        index = decision.candidate_index
        if index is None:
            valid = decision.emitted is None and decision.eligibility == ()
        else:
            valid = (
                type(index) is int and 0 <= index < len(rows)
                and decision.emitted == self._candidates[index].emitted
                and decision.eligibility == tuple(rows[index])
            )
        if not valid:
            raise ValueError("original selected occurrence or eligibility changed")

    @staticmethod
    def _validate_teacher(answer, attempt: dict, teacher: dict) -> None:
        if answer is not None and (not isinstance(answer, bytes) or not answer):
            raise ValueError("teacher answer must be nonempty bytes or explicit STOP")
        if type(attempt.get("passed")) is not bool or teacher.get("passed") is not True:
            raise ValueError("teacher requires genuine attempt and successful execution receipts")
        expected = None if answer is None else answer.hex()
        if "answer_hex" not in teacher or teacher["answer_hex"] != expected:
            raise ValueError("teacher execution authorization does not match exact answer")
        if answer is None:
            valid = (
                teacher.get("teacher_kind") == "authorized_absent_evidence_stop"
                and teacher.get("workspace_unchanged") is True
            )
        else:
            valid = teacher.get("teacher_kind") == "authorized_exact_answer"
        if not valid:
            raise ValueError("teacher must explicitly authorize exact answer or no-action STOP")

    def teacher_lesson(
        self, original_decision: ContextualEvidenceDecision, answer: bytes | None, *,
        teacher_id: str, attempt_receipt: dict, teacher_receipt: dict,
    ) -> FullChoiceLesson:
        self._require_pool()
        if original_decision.checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError("new teacher lesson requires the current original action checkpoint")
        if not isinstance(teacher_id, str) or not teacher_id:
            raise ValueError("teacher identity must be nonempty")
        rows, _ = self._features(original_decision.query)
        self._validate_original(original_decision, rows)
        self._validate_teacher(answer, attempt_receipt, teacher_receipt)
        self._gold(answer)
        fields = dict(
            original_decision=original_decision, answer=answer, teacher_id=teacher_id,
            source_pool_sha256=self._training_pool, candidate_count=len(rows),
            attempt_receipt_json=_canonical(attempt_receipt),
            teacher_receipt_json=_canonical(teacher_receipt),
        )
        unsigned = FullChoiceLesson(**fields, sha256="")
        return FullChoiceLesson(**fields, sha256=_lesson_digest(unsigned))

    def _gold(self, answer: bytes | None) -> tuple[int, ...]:
        gold = (len(self._candidates),) if answer is None else tuple(
            i for i, candidate in enumerate(self._candidates) if candidate.emitted == answer
        )
        if not gold:
            raise ValueError("authorized exact answer has no occurrence in the TRAIN pool")
        return gold

    def observe_teacher(self, lesson: FullChoiceLesson) -> dict:
        """One supervised offer or explicit replay; no new execution is claimed."""
        self._require_pool()
        if (
            lesson.sha256 != _lesson_digest(lesson)
            or lesson.source_pool_sha256 != self._training_pool
            or lesson.candidate_count != len(self._candidates)
        ):
            raise ValueError("teacher lesson custody or fixed source universe changed")
        attempt, teacher = (
            json.loads(raw) for raw in (lesson.attempt_receipt_json, lesson.teacher_receipt_json)
        )
        self._validate_teacher(lesson.answer, attempt, teacher)
        rows, _ = self._features(lesson.original_decision.query)
        self._validate_original(lesson.original_decision, rows)
        gold = self._gold(lesson.answer)
        before, synapses_before = self.checkpoint_sha256, self.synapse_sha256
        stop_before = float(self._stop_score[0])
        loss, gradient, logits = full_choice_loss_gradient(
            rows, self._weights[0] - self._weights[1], stop_before, gold
        )
        norm = float(np.linalg.norm(gradient))
        if not np.isfinite(norm):
            raise ValueError("gradient norm is nonfinite; no state was published")
        clip_scale = min(1.0, self.gradient_clip_norm / norm) if norm else 1.0
        applied = gradient * clip_scale
        proposed = self.learning_rate
        candidate_weights = self._weights.copy()
        candidate_stop = self._stop_score.copy()
        if self.learning_enabled:
            change = (0.5 * proposed) * applied[:-1]
            candidate_weights[0] -= change
            candidate_weights[1] += change
            candidate_stop[0] -= proposed * applied[-1]
        if not np.isfinite(candidate_weights).all() or not np.isfinite(candidate_stop).all():
            raise ValueError("SGD proposal is nonfinite; no state was published")
        updated = not (
            np.array_equal(candidate_weights, self._weights)
            and np.array_equal(candidate_stop, self._stop_score)
        )
        loss_after, _, logits_after = full_choice_loss_gradient(
            rows, candidate_weights[0] - candidate_weights[1], float(candidate_stop[0]), gold
        )
        receipt = {
            "mode": _POLICY, "lesson_sha256": lesson.sha256,
            "original_decision_sha256": lesson.original_decision.sha256,
            "source_pool_sha256": self._training_pool, "before": before,
            "features_sha256": _array_hash(rows), "gold_indices": list(gold),
            "candidate_count": len(rows), "choice_count": len(rows) + 1,
            "loss_before": loss, "loss_after": loss_after,
            "logits_before_sha256": _array_hash(logits),
            "logits_after_sha256": _array_hash(logits_after),
            "gradient": gradient.tolist(), "gradient_sha256": _array_hash(gradient),
            "gradient_norm": norm, "clip_scale": clip_scale,
            "applied_gradient_sha256": _array_hash(applied),
            "proposed_learning_rate": proposed,
            "effective_learning_rate": proposed if updated else 0.0,
            "learning_enabled": self.learning_enabled, "sgd_update_applied": updated,
            "native_update_applied": False,
            "synapse_sha256_before": synapses_before,
            "stop_before": stop_before, "stop_after": float(candidate_stop[0]),
            "supervised_offer": 1, "new_execution_credit": False,
            "feature_matrix_bytes": rows.nbytes, "gradient_array_bytes": gradient.nbytes,
            "score_dot_entries": 2 * rows.size, "gradient_dot_entries": 2 * rows.size,
        }
        self._weights[:] = candidate_weights
        self._stop_score[:] = candidate_stop
        self.feedback_count += 1
        self.update_count += int(updated)
        self._feedback_root = _digest({"parent": self._feedback_root, "offer": receipt})
        receipt["synapse_sha256_after"] = self.synapse_sha256
        receipt["after"] = self.checkpoint_sha256
        receipt["offer_count"] = self.feedback_count
        receipt["update_count"] = self.update_count
        return receipt

    def _legacy_feedback_disabled(self, *args, **kwargs):
        raise ValueError(
            "full-choice learning requires authenticated teacher_lesson/observe_teacher"
        )

    observe_feedback = _legacy_feedback_disabled
    observe_comparison = _legacy_feedback_disabled
    replay_decision = _legacy_feedback_disabled
    revisit_decision = _legacy_feedback_disabled
