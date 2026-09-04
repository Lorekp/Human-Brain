"""Disclosed frozen-encoder memory with shared reward-trained GLIF span selection.

All word spans of lengths 1..3 within consecutive 48-word source windows are
retained, without semantic filtering. Raw source bytes remain external episodic
memory. Learned GO/NO_GO dendrites, rather than an encoder answer head, drive
the neuronal candidate race. External cosine proposals and TRAIN teacher spans
are separately labeled and never count as neuronal selections.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from humanbrain.frozen_context_encoder import FrozenContextEncoder
from humanbrain.native import NativeThreeFactorLearner
from humanbrain.opaque_document_event_tape import OpaqueDocument
from humanbrain.resting_glif_race import CompiledRestingGlifRace, RestingGlifRaceFailure

_DIMENSION = 384
_FEATURES = 4 * _DIMENSION + 1
_RACE_STEPS = 600


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / max(float(np.linalg.norm(array)), 1e-12)


def _words(raw: bytes) -> tuple[tuple[int, int], ...]:
    text = raw.decode("utf-8")
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + len(character.encode("utf-8")))
    return tuple((offsets[m.start()], offsets[m.end()]) for m in re.finditer(r"\w+", text))


@dataclass(frozen=True, slots=True)
class _Candidate:
    document_index: int
    start: int
    end: int
    window_index: int
    emitted: bytes


@dataclass(frozen=True, slots=True)
class ContextualEvidenceDecision:
    candidate_index: int | None
    emitted: bytes | None
    checkpoint_sha256: str
    storage_sha256: str
    query: bytes
    eligibility: tuple[float, ...]
    mode: str
    stop_reason: str | None
    physical_cell_steps: int
    spike_step: int | None
    sha256: str

    @property
    def stopped(self) -> bool:
        return self.candidate_index is None


class ContextualEvidenceMemory:
    """One shared two-row plastic synapse bank across source-derived candidates."""

    def __init__(
        self,
        encoder: FrozenContextEncoder,
        *,
        learning_rate: float = 1.0,
        learning_enabled: bool = True,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.encoder = encoder
        self.learning_rate = float(learning_rate)
        self.learning_enabled = learning_enabled
        self._documents: list[OpaqueDocument] = []
        self._candidates: list[_Candidate] = []
        self._vectors = np.empty((0, _DIMENSION), dtype=np.float64)
        self._contexts: list[np.ndarray] = []
        self._storage_root = _digest([])
        self._weights = np.zeros((2, _FEATURES), dtype=np.float64)
        self._bias = np.zeros(2, dtype=np.float64)
        self._heterogeneity = np.zeros(2, dtype=np.float64)
        self._eligibility = np.zeros(_FEATURES, dtype=np.float64)
        self._native = NativeThreeFactorLearner(
            eligibility=self._eligibility,
            output_heterogeneity_pa=self._heterogeneity,
            readout_synapses_pa=self._weights,
            pacemaker_synapses_pa=self._bias,
        )
        self._race = CompiledRestingGlifRace.compile(step_count=_RACE_STEPS)
        self._feedback_root = _digest([])
        self.feedback_count = 0
        self.update_count = 0

    @property
    def checkpoint_sha256(self) -> str:
        return _digest(
            {
                "encoder": self.encoder.sha256,
                "storage": self._storage_root,
                "synapses": self.synapse_sha256,
                "feedback": self._feedback_root,
            }
        )

    @property
    def synapse_sha256(self) -> str:
        return sha256(self._weights.tobytes() + self._bias.tobytes()).hexdigest()

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        """Return independent copies for recording the learned synaptic state."""
        return {
            "readout_synapses_pa": self._weights.copy(),
            "pacemaker_synapses_pa": self._bias.copy(),
        }

    @property
    def storage_summary(self) -> dict:
        raw = sum(len(d.content) for d in self._documents)
        lexical = sum(len(c.emitted) for c in self._candidates)
        vectors = self._vectors.nbytes + sum(v.nbytes for v in self._contexts)
        metadata = sum(len(d.document_id) + len(d.provenance_id) for d in self._documents)
        neural = sum(
            a.nbytes
            for a in (
                self._weights,
                self._bias,
                self._heterogeneity,
                self._eligibility,
            )
        )
        compiled_race = self._race._unit_voltage.nbytes + self._race._unit_voltage_threshold.nbytes
        return {
            "raw_source_bytes": raw,
            "candidate_lexical_bytes": lexical,
            "context_vector_bytes": vectors,
            "identity_bytes": metadata,
            "synapse_array_bytes": self._weights.nbytes + self._bias.nbytes,
            "payload_bytes": raw + lexical + vectors + metadata,
            "compiled_race_array_bytes": compiled_race,
            "retained_array_and_byte_payload_bytes": (
                raw + lexical + vectors + metadata + neural + compiled_race
            ),
            "plastic_synapse_count": self._weights.size,
            "nonzero_plastic_synapses": int(np.count_nonzero(self._weights)),
            "python_object_overhead_included": False,
            "candidate_count": len(self._candidates),
            "document_count": len(self._documents),
            "window_count": len(self._contexts),
            "feedback_count": self.feedback_count,
            "update_count": self.update_count,
        }

    def ingest(self, documents: tuple[OpaqueDocument, ...]) -> None:
        """Retain every word candidate; fail explicitly instead of truncating encoding."""
        candidates, vectors, contexts = [], [], []
        for offset, document in enumerate(documents):
            words = _words(document.content)
            for begin in range(0, len(words), 48):
                group = words[begin : begin + 48]
                left, right = group[0][0], group[-1][1]
                encoded = self.encoder.encode(document.content[left:right])
                spans = tuple(
                    (group[i][0] - left, group[j][1] - left)
                    for i in range(len(group))
                    for j in range(i, min(i + 3, len(group)))
                )
                pooled = encoded.pool_spans(spans)
                context_index = len(self._contexts) + len(contexts)
                contexts.append(_unit(encoded.sentence_vector))
                for (start, end), vector in zip(spans, pooled, strict=True):
                    candidates.append(
                        _Candidate(
                            len(self._documents) + offset,
                            left + start,
                            left + end,
                            context_index,
                            document.content[left + start : left + end],
                        )
                    )
                    vectors.append(_unit(vector))
        # Publish together only after every window is encoded successfully.
        self._documents.extend(documents)
        self._candidates.extend(candidates)
        self._contexts.extend(contexts)
        if vectors:
            self._vectors = np.vstack((self._vectors, vectors))
        self._storage_root = _digest(
            {
                "parent": self._storage_root,
                "documents": [
                    (d.document_id.hex(), d.provenance_id.hex(), sha256(d.content).hexdigest())
                    for d in documents
                ],
                "vectors": sha256(self._vectors.tobytes()).hexdigest(),
                "contexts": sha256(b"".join(v.tobytes() for v in contexts)).hexdigest(),
            }
        )

    def _features(self, query: bytes) -> tuple[np.ndarray, np.ndarray]:
        q = _unit(self.encoder.encode(query).sentence_vector)
        rows = []
        query_words = set(re.findall(r"\w+", query.decode("utf-8").casefold()))
        for candidate, vector in zip(self._candidates, self._vectors, strict=True):
            overlap = float(candidate.emitted.decode("utf-8").casefold() in query_words)
            rows.append(
                np.concatenate(
                    (q, vector, q * vector, q * self._contexts[candidate.window_index], [overlap])
                )
            )
        return np.asarray(rows, dtype=np.float64).reshape(-1, _FEATURES), self._vectors @ q

    def _decision(
        self,
        query: bytes,
        rows: np.ndarray,
        index: int | None,
        mode: str,
        reason: str | None = None,
        steps: int = 0,
        spike_step: int | None = None,
    ) -> ContextualEvidenceDecision:
        values = dict(
            candidate_index=index,
            emitted=None if index is None else self._candidates[index].emitted,
            checkpoint_sha256=self.checkpoint_sha256,
            storage_sha256=self._storage_root,
            query=query,
            eligibility=() if index is None else tuple(float(x) for x in rows[index]),
            mode=mode,
            stop_reason=reason,
            physical_cell_steps=steps,
            spike_step=spike_step,
        )
        serial = {
            **values,
            "emitted": None if index is None else values["emitted"].hex(),
            "query": query.hex(),
        }
        return ContextualEvidenceDecision(**values, sha256=_digest(serial))

    def _currents(self, rows: np.ndarray) -> np.ndarray:
        drive = rows @ (self._weights[0] - self._weights[1]) + self._bias[0] - self._bias[1]
        return np.append(600.0 + 200.0 * np.tanh(drive), 600.1)

    def respond(self, query: bytes) -> ContextualEvidenceDecision:
        rows, _ = self._features(query)
        currents = self._currents(rows)
        try:
            result = self._race.race_many(currents[None, :], require_unique=True)[0]
        except RestingGlifRaceFailure as error:
            return self._decision(
                query, rows, None, "neuronal", str(error), _RACE_STEPS * len(currents)
            )
        index = result.winner_cell if result.winner_cell < len(self._candidates) else None
        return self._decision(
            query,
            rows,
            index,
            "neuronal",
            "tonic_stop" if index is None else None,
            (result.spike_step + 1) * len(currents),
            result.spike_step,
        )

    def conventional(self, query: bytes) -> ContextualEvidenceDecision:
        """Learned linear reference: identical currents, host unique argmax and STOP."""
        rows, _ = self._features(query)
        currents = self._currents(rows)
        best = np.flatnonzero(currents == currents.max())
        index = int(best[0]) if len(best) == 1 and best[0] < len(self._candidates) else None
        return self._decision(
            query,
            rows,
            index,
            "conventional_learned",
            "tonic_stop_or_tie" if index is None else None,
        )

    def external_only(self, query: bytes) -> ContextualEvidenceDecision:
        """Disclosed cosine proposal, independent of neuronal weights."""
        rows, similarity = self._features(query)
        best = np.flatnonzero(similarity == similarity.max()) if len(similarity) else []
        index = int(best[0]) if len(best) == 1 else None
        return self._decision(
            query, rows, index, "external_only", "empty_or_tied_cosine" if index is None else None
        )

    def decision_for_span(self, query: bytes, emitted: bytes) -> ContextualEvidenceDecision:
        """TRAIN-only teacher credit, after execution; ambiguous occurrences reject."""
        matches = [
            i for i, candidate in enumerate(self._candidates) if candidate.emitted == emitted
        ]
        if len(matches) != 1:
            raise ValueError("teacher span must identify exactly one retained occurrence")
        rows, _ = self._features(query)
        return self._decision(query, rows, matches[0], "teacher_executed_span")

    def _validated_rows(self, decision: ContextualEvidenceDecision) -> np.ndarray:
        record = asdict(decision)
        expected = record.pop("sha256")
        record["query"] = decision.query.hex()
        record["emitted"] = None if decision.emitted is None else decision.emitted.hex()
        if expected != _digest(record):
            raise ValueError("feedback decision custody changed")
        if decision.storage_sha256 != self._storage_root:
            raise ValueError("feedback decision has a different retained source store")
        if decision.candidate_index is None:
            raise ValueError("STOP has no selected candidate to credit")
        rows, _ = self._features(decision.query)
        index = decision.candidate_index
        if (
            not 0 <= index < len(self._candidates)
            or self._candidates[index].emitted != decision.emitted
            or tuple(rows[index]) != decision.eligibility
        ):
            raise ValueError("feedback eligibility does not match the retained candidate")
        return rows

    def replay_decision(self, decision: ContextualEvidenceDecision) -> ContextualEvidenceDecision:
        """Rebind identical executed training exposure to a frozen control's parent."""
        if self.learning_enabled:
            raise ValueError("matched-control replay requires frozen synapses")
        rows = self._validated_rows(decision)
        return self._decision(
            query=decision.query,
            rows=rows,
            index=decision.candidate_index,
            mode="yoked_" + decision.mode,
        )

    def observe_feedback(self, decision: ContextualEvidenceDecision, reward: int) -> dict:
        """Apply exactly one consequence at this parent; wrapper authenticates execution."""
        if reward not in (-1, 1) or type(reward) is not int:
            raise ValueError("reward must be exactly -1 or +1")
        if decision.checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError("feedback decision has a stale parent checkpoint")
        rows = self._validated_rows(decision)
        index = decision.candidate_index
        before = self.checkpoint_sha256
        changed = False
        if self.learning_enabled:
            self._eligibility[:] = rows[index]
            changed = (
                self._native.update(
                    0 if reward > 0 else 1,
                    self.learning_rate,
                    1.0,
                )
                is not None
            )
            self._eligibility.fill(0.0)
        self.feedback_count += 1
        self.update_count += int(changed)
        self._feedback_root = _digest([self._feedback_root, decision.sha256, reward])
        return {
            "before": before,
            "after": self.checkpoint_sha256,
            "decision_sha256": decision.sha256,
            "reward": reward,
            "synaptic_update": changed,
            "mode": decision.mode,
        }

    def observe_comparison(
        self, rejected: ContextualEvidenceDecision, accepted: ContextualEvidenceDecision
    ) -> dict:
        """Learn one same-query ranking margin after caller-verified FAIL and PASS."""
        before = self.checkpoint_sha256
        if any(row.checkpoint_sha256 != before for row in (rejected, accepted)):
            raise ValueError("comparison decision has a stale parent checkpoint")
        if rejected.query != accepted.query:
            raise ValueError("comparison decisions must have the same query")
        self._validated_rows(rejected)
        self._validated_rows(accepted)
        if rejected.candidate_index == accepted.candidate_index:
            raise ValueError("comparison requires two different candidate occurrences")
        delta = np.asarray(accepted.eligibility) - np.asarray(rejected.eligibility)
        gap_before = float((self._weights[0] - self._weights[1]) @ delta)
        synapses_before = self.synapse_sha256
        applied = False
        if self.learning_enabled:
            # The native learner also updates its bias. A fresh scratch bias
            # excludes the persistent bias from the ranking margin and discards
            # that non-ranking update; only shared dendritic weights survive.
            pair_bias = np.zeros_like(self._bias)
            pair_learner = NativeThreeFactorLearner(
                eligibility=self._eligibility,
                output_heterogeneity_pa=self._heterogeneity,
                readout_synapses_pa=self._weights,
                pacemaker_synapses_pa=pair_bias,
            )
            self._eligibility[:] = delta
            try:
                applied = pair_learner.update(0, self.learning_rate, 1.0) is not None
            finally:
                self._eligibility.fill(0.0)
        self.feedback_count += 2
        self.update_count += int(applied)
        for decision, reward in ((rejected, -1), (accepted, 1)):
            self._feedback_root = _digest([self._feedback_root, decision.sha256, reward])
        return {
            "before": before,
            "after": self.checkpoint_sha256,
            "rejected_decision_sha256": rejected.sha256,
            "accepted_decision_sha256": accepted.sha256,
            "eligibility_delta_sha256": sha256(delta.tobytes()).hexdigest(),
            "gap_before": gap_before,
            "gap_after": float((self._weights[0] - self._weights[1]) @ delta),
            "native_update_applied": applied,
            "synaptic_update": self.synapse_sha256 != synapses_before,
            "bias_policy": "zero scratch pair bias; persistent bias excluded and unchanged",
            "feedback_events": 2,
            "mode": "same_query_pairwise",
        }
