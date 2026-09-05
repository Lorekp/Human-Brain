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
_CANDIDATE_FEATURES = 84
_CANDIDATE_GAIN = 16.0
_POSITION_SCALES = (2.0, 8.0, 32.0)
_AFFINE_READOUT_POLICY = (
    "affine_stop_centered_v1;"
    "tau=atanh((600.1-600.0)/200.0);d=raw-tau;R=max(1,max(abs(d)));"
    "I=600.1+200*(d/R);STOP=600.1;"
    "race=linear_glif_endpoint_interpolation_v1;"
    "ties=equal_or_adjacent_float64_times_stop"
)


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / max(float(np.linalg.norm(array)), 1e-12)


def _affine_currents(drive: np.ndarray) -> np.ndarray:
    """Common positive scaling; finite-precision aliases are not broken here."""
    if not np.isfinite(drive).all():
        raise ValueError("affine readout requires finite raw drives")
    delta = drive - np.arctanh((600.1 - 600.0) / 200.0)
    scale = max(1.0, float(np.max(np.abs(delta)))) if len(delta) else 1.0
    # Divide first: multiplying a finite raw drive by 200 could overflow.
    return np.append(600.1 + 200.0 * (delta / scale), 600.1)


def _words(raw: bytes) -> tuple[tuple[int, int], ...]:
    text = raw.decode("utf-8")
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + len(character.encode("utf-8")))
    return tuple((offsets[m.start()], offsets[m.end()]) for m in re.finditer(r"\w+", text))


def _position_kernels(distance: np.ndarray, *, collapsed: bool = False) -> np.ndarray:
    if collapsed:
        return np.ones((6, *distance.shape), dtype=np.float64)
    return np.asarray(
        [
            (distance * side >= 0) * np.exp(-np.abs(distance) / scale)
            for side in (-1, 1)
            for scale in _POSITION_SCALES
        ]
    )


def _candidate_match_features(
    exact: np.ndarray, semantic: np.ndarray, word_spans: np.ndarray, *, collapsed: bool = False
) -> np.ndarray:
    """Unscaled 72 F[a,b,Y] then 12 H[b,Y], Y ordered exact, semantic.

    A_i^a(l)=max_j E_ij K_a(j-l); R_i,Y(l)=sum_{k!=i}Y_kl/max(1,Q-1).
    F=mean_i max_l A_i^a(l) R_i,Y(l) K_b(l-c); H=mean_i max_j Y_ij K_b(j-c).
    All query anchors, source words and candidate intervals participate.
    """
    count, query_count = len(word_spans), len(exact)
    if query_count == 0:
        return np.zeros((count, _CANDIDATE_FEATURES))
    positions = np.arange(exact.shape[1])
    pair_kernels = _position_kernels(positions[:, None] - positions, collapsed=collapsed)
    left, right = word_spans[:, 0, None], word_spans[:, 1, None]
    distance = np.where(
        positions < left, positions - left, np.where(positions > right, positions - right, 0)
    )
    candidate_kernels = _position_kernels(distance, collapsed=collapsed)
    anchors = (exact[None, :, :, None] * pair_kernels[:, None, :, :]).max(axis=2)
    matches = np.stack((exact, semantic))
    other = (matches.sum(axis=1, keepdims=True) - matches) / max(1, query_count - 1)
    joint = np.empty((count, 6, 6, 2))
    direct = np.empty((count, 6, 2))
    # Vectorize over query words and every candidate; only six fixed kernels loop.
    for b, kernel in enumerate(candidate_kernels):
        direct[:, b, :] = (matches[:, :, None, :] * kernel).max(axis=-1).mean(axis=1).T
        for a, anchor in enumerate(anchors):
            supported = other * anchor[None, :, :]
            joint[:, a, b, :] = (supported[:, :, None, :] * kernel).max(axis=-1).mean(axis=1).T
    return np.concatenate((joint.reshape(count, 72), direct.reshape(count, 12)), axis=1)


@dataclass(frozen=True, slots=True)
class _InteractionWindow:
    candidate_start: int
    word_candidate_indices: np.ndarray
    candidate_word_spans: np.ndarray
    words: tuple[str, ...]

    @property
    def payload_bytes(self) -> int:
        return (
            8
            + self.word_candidate_indices.nbytes
            + self.candidate_word_spans.nbytes
            + sum(len(word.encode("utf-8")) for word in self.words)
        )


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
    readout_policy: str | None
    estimated_crossing_time_ms: float | None
    coarse_spike_candidates: int | None
    simultaneous_spike_candidates: int | None
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
        interaction_gain: float = 1.0,
        candidate_interactions: bool = False,
        comparison_step_policy: str = "fixed",
        comparison_protection_policy: str = "none",
        protection_capacity: int = 48,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(interaction_gain) or interaction_gain <= 0:
            raise ValueError("interaction_gain must be finite and positive")
        if type(candidate_interactions) is not bool:
            raise ValueError("candidate_interactions must be a bool")
        if comparison_step_policy not in ("fixed", "unit_margin_capped"):
            raise ValueError("unknown comparison_step_policy")
        if comparison_protection_policy not in ("none", "orthogonal_certified"):
            raise ValueError("unknown comparison_protection_policy")
        if type(protection_capacity) is not int or protection_capacity <= 0:
            raise ValueError("protection_capacity must be a positive integer")
        if comparison_protection_policy != "none" and comparison_step_policy != "fixed":
            raise ValueError("orthogonal protection requires fixed comparison steps")
        self.encoder = encoder
        self.learning_rate = float(learning_rate)
        self.learning_enabled = learning_enabled
        self._interaction_gain = float(interaction_gain)
        self._candidate_interactions = candidate_interactions
        self._comparison_step_policy = comparison_step_policy
        self._comparison_protection_policy = comparison_protection_policy
        self._protection_capacity = protection_capacity
        self._interaction_windows: list[_InteractionWindow] = []
        self._documents: list[OpaqueDocument] = []
        self._candidates: list[_Candidate] = []
        self._vectors = np.empty((0, _DIMENSION), dtype=np.float64)
        self._contexts: list[np.ndarray] = []
        storage_identity = {"representation": self.representation_policy}
        if self.comparison_step_policy != "fixed":
            storage_identity["comparison_step_policy"] = self.comparison_step_policy
            storage_identity["maximum_learning_rate"] = self.learning_rate
        if self.comparison_protection_policy != "none":
            storage_identity["comparison_protection"] = self.protection_policy
        self._storage_root = _digest(storage_identity)
        self._storage_ancestry = {self._storage_root: 0}
        feature_count = _FEATURES + _CANDIDATE_FEATURES * candidate_interactions
        self._weights = np.zeros((2, feature_count), dtype=np.float64)
        self._bias = np.zeros(2, dtype=np.float64)
        self._heterogeneity = np.zeros(2, dtype=np.float64)
        self._eligibility = np.zeros(feature_count, dtype=np.float64)
        if self.comparison_protection_policy != "none":
            self._protection_basis = np.empty((0, feature_count), dtype=np.float64)
            self._protected_deltas = np.empty((0, feature_count), dtype=np.float64)
            self._protection_records: list[dict] = []
            self._projected_eligibility = np.zeros(feature_count, dtype=np.float64)
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
    def interaction_gain(self) -> float:
        return self._interaction_gain

    @property
    def candidate_interactions(self) -> bool:
        return self._candidate_interactions

    @property
    def comparison_step_policy(self) -> str:
        return self._comparison_step_policy

    @property
    def comparison_protection_policy(self) -> str:
        return self._comparison_protection_policy

    @property
    def protection_capacity(self) -> int:
        return self._protection_capacity

    @property
    def protection_policy(self) -> dict:
        return {
            "kind": self.comparison_protection_policy,
            "capacity": self.protection_capacity,
            "learning_rate": self.learning_rate,
            "projection": "two_native_modified_gram_schmidt_sweeps_v1",
            "gate_and_normalizer": "original_delta; unit_margin; sqrt(1+d_dot_d)",
            "certification": "own_final_margin_ge_1; first_identity_only; no_eviction",
            "exhaustion": "64*eps*D*max(1,norm_d)",
            "raw_residual": "64*eps*D*max(1,norm_di*norm_p)",
            "margin_tolerance": "128*eps*D*max(1,abs(before),norm_v_before*norm_di,"
            "norm_v_after*norm_di)",
        }

    def _protection_record_bytes(self) -> bytes:
        if not self._protection_records:
            return b""
        return json.dumps(self._protection_records, sort_keys=True, allow_nan=False).encode()

    @property
    def protection_summary(self) -> dict:
        """Disclosed learned training state; never consulted to select a candidate."""
        if self.comparison_protection_policy == "none":
            return {"policy": "none"}
        basis_hash = sha256(self._protection_basis.tobytes()).hexdigest()
        deltas_hash = sha256(self._protected_deltas.tobytes()).hexdigest()
        records = self._protection_record_bytes()
        state = {
            "policy": self.protection_policy,
            "basis_sha256": basis_hash,
            "protected_deltas_sha256": deltas_hash,
            "certification_sha256": sha256(records).hexdigest(),
            "basis_rank": len(self._protection_basis),
            "protected_count": len(self._protected_deltas),
        }
        return {
            **state,
            "state_sha256": _digest(state),
            "basis_array_bytes": self._protection_basis.nbytes,
            "protected_delta_array_bytes": self._protected_deltas.nbytes,
            "certification_payload_bytes": len(records),
            "learned_payload_bytes": self._protection_basis.nbytes
            + self._protected_deltas.nbytes
            + len(records),
            "scratch_array_bytes": self._projected_eligibility.nbytes,
        }

    @property
    def representation_policy(self) -> dict:
        policy = {
            "kind": "unit_query_span_window_with_scaled_hadamard_products_v1",
            "interaction_gain": self.interaction_gain,
        }
        if self.candidate_interactions:
            policy["candidate_interactions"] = {
                "kind": "relative_word_anchor_other_query_match_v1",
                "scales_words": _POSITION_SCALES,
                "gain": _CANDIDATE_GAIN,
                "direction_order": "left_then_right; zero_in_both",
                "layout": "72_F[a,b,Y]_then_12_H[b,Y]; Y=exact,semantic",
                "matching": "casefold_exact_and_nonnegative_unit_word_cosine",
                "query_pooling": "mean_over_all_word_occurrences; other_excludes_same_index",
                "source_scope": "existing_48_word_window",
            }
        return policy

    @property
    def readout_policy(self) -> dict:
        return {
            "default": "discrete_glif_steps_v1",
            "comparison": "linear_glif_endpoint_interpolation_v1",
            "interpolated_ties": "equal_or_adjacent_float64_times_stop",
            "dt_ms": self._race.parameters.dt_ms,
            "step_count": _RACE_STEPS,
        }

    @property
    def checkpoint_sha256(self) -> str:
        state = {
            "encoder": self.encoder.sha256,
            "representation": self.representation_policy,
            "readouts": self.readout_policy,
            "storage": self._storage_root,
            "synapses": self.synapse_sha256,
            "feedback": self._feedback_root,
        }
        if self.comparison_step_policy != "fixed":
            state["comparison_step_policy"] = self.comparison_step_policy
            state["maximum_learning_rate"] = self.learning_rate
        if self.comparison_protection_policy != "none":
            state["comparison_protection"] = self.protection_summary["state_sha256"]
        return _digest(state)

    @property
    def synapse_sha256(self) -> str:
        return sha256(self._weights.tobytes() + self._bias.tobytes()).hexdigest()

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        """Return independent copies for recording the learned synaptic state."""
        arrays = {
            "readout_synapses_pa": self._weights.copy(),
            "pacemaker_synapses_pa": self._bias.copy(),
        }
        if self.comparison_protection_policy != "none":
            arrays.update(
                protection_basis=self._protection_basis.copy(),
                protected_contrast_deltas=self._protected_deltas.copy(),
                protection_certification=np.frombuffer(
                    self._protection_record_bytes(), dtype=np.uint8
                ).copy(),
            )
        return arrays

    @property
    def storage_summary(self) -> dict:
        raw = sum(len(d.content) for d in self._documents)
        lexical = sum(len(c.emitted) for c in self._candidates)
        vectors = self._vectors.nbytes + sum(v.nbytes for v in self._contexts)
        metadata = sum(len(d.document_id) + len(d.provenance_id) for d in self._documents)
        # ASCII SHA-256 plus an eight-byte candidate count per ingestion root;
        # Python container/object overhead remains excluded as elsewhere here.
        ancestry = sum(len(root) + 8 for root in self._storage_ancestry)
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
        interactions = sum(window.payload_bytes for window in self._interaction_windows)
        summary = {
            "interaction_gain": self.interaction_gain,
            "candidate_interactions": self.candidate_interactions,
            "candidate_interaction_payload_bytes": interactions,
            "candidate_interaction_array_bytes": sum(
                window.word_candidate_indices.nbytes + window.candidate_word_spans.nbytes
                for window in self._interaction_windows
            ),
            "raw_source_bytes": raw,
            "candidate_lexical_bytes": lexical,
            "context_vector_bytes": vectors,
            "identity_bytes": metadata,
            "storage_ancestry_payload_bytes": ancestry,
            "storage_ancestry_root_count": len(self._storage_ancestry),
            "synapse_array_bytes": self._weights.nbytes + self._bias.nbytes,
            "payload_bytes": raw + lexical + vectors + metadata + ancestry + interactions,
            "compiled_race_array_bytes": compiled_race,
            "retained_array_and_byte_payload_bytes": (
                raw
                + lexical
                + vectors
                + metadata
                + ancestry
                + interactions
                + neural
                + compiled_race
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
        if self.comparison_protection_policy != "none":
            protection = self.protection_summary
            summary["comparison_protection"] = protection
            summary["payload_bytes"] += protection["learned_payload_bytes"]
            summary["retained_array_and_byte_payload_bytes"] += (
                protection["learned_payload_bytes"] + protection["scratch_array_bytes"]
            )
        return summary

    def ingest(self, documents: tuple[OpaqueDocument, ...]) -> None:
        """Retain every word candidate; fail explicitly instead of truncating encoding."""
        candidates, vectors, contexts, interaction_windows = [], [], [], []
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
                if self.candidate_interactions:
                    intervals = np.asarray(
                        [
                            (i, j)
                            for i in range(len(group))
                            for j in range(i, min(i + 3, len(group)))
                        ],
                        dtype=np.int64,
                    )
                    candidate_start = len(self._candidates) + len(candidates)
                    word_indices = candidate_start + np.flatnonzero(
                        intervals[:, 0] == intervals[:, 1]
                    )
                    intervals.setflags(write=False)
                    word_indices.setflags(write=False)
                    interaction_windows.append(
                        _InteractionWindow(
                            candidate_start,
                            word_indices,
                            intervals,
                            tuple(
                                document.content[a:b].decode("utf-8").casefold() for a, b in group
                            ),
                        )
                    )
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
        self._interaction_windows.extend(interaction_windows)
        if vectors:
            self._vectors = np.vstack((self._vectors, vectors))
        storage = {
            "parent": self._storage_root,
            "documents": [
                (d.document_id.hex(), d.provenance_id.hex(), sha256(d.content).hexdigest())
                for d in documents
            ],
            "vectors": sha256(self._vectors.tobytes()).hexdigest(),
            "contexts": sha256(b"".join(v.tobytes() for v in contexts)).hexdigest(),
        }
        if self.candidate_interactions:
            storage["interaction_windows"] = [
                (
                    window.candidate_start,
                    window.words,
                    sha256(window.word_candidate_indices.tobytes()).hexdigest(),
                    sha256(window.candidate_word_spans.tobytes()).hexdigest(),
                )
                for window in interaction_windows
            ]
        self._storage_root = _digest(storage)
        self._storage_ancestry[self._storage_root] = len(self._candidates)

    def _features(
        self, query: bytes, *, interaction_ablation: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if interaction_ablation not in (None, "disabled", "position_collapsed"):
            raise ValueError("unknown interaction ablation")
        if interaction_ablation is not None and not self.candidate_interactions:
            raise ValueError("interaction ablation requires candidate_interactions")
        encoded = self.encoder.encode(query)
        q = _unit(encoded.sentence_vector)
        rows = []
        query_words = set(re.findall(r"\w+", query.decode("utf-8").casefold()))
        for candidate, vector in zip(self._candidates, self._vectors, strict=True):
            overlap = float(candidate.emitted.decode("utf-8").casefold() in query_words)
            rows.append(
                np.concatenate(
                    (
                        q,
                        vector,
                        self.interaction_gain * (q * vector),
                        self.interaction_gain * (q * self._contexts[candidate.window_index]),
                        [overlap],
                    )
                )
            )
        result = np.asarray(rows, dtype=np.float64).reshape(-1, _FEATURES)
        if self.candidate_interactions:
            extra = np.zeros((len(rows), _CANDIDATE_FEATURES))
            if interaction_ablation != "disabled":
                spans = _words(query)
                words = tuple(query[a:b].decode("utf-8").casefold() for a, b in spans)
                vectors = np.asarray([_unit(v) for v in encoded.pool_spans(spans)]).reshape(-1, 384)
                for window in self._interaction_windows:
                    exact = np.asarray(words)[:, None] == np.asarray(window.words)[None, :]
                    semantic = np.maximum(
                        0, vectors @ self._vectors[window.word_candidate_indices].T
                    )
                    start = window.candidate_start
                    stop = start + len(window.candidate_word_spans)
                    extra[start:stop] = _candidate_match_features(
                        exact.astype(np.float64),
                        semantic,
                        window.candidate_word_spans,
                        collapsed=interaction_ablation == "position_collapsed",
                    )
            result = np.concatenate((result, _CANDIDATE_GAIN * extra), axis=1)
        return result, self._vectors @ q

    def _decision(
        self,
        query: bytes,
        rows: np.ndarray,
        index: int | None,
        mode: str,
        reason: str | None = None,
        steps: int = 0,
        spike_step: int | None = None,
        *,
        readout_policy: str | None = None,
        estimated_crossing_time_ms: float | None = None,
        coarse_spike_candidates: int | None = None,
        simultaneous_spike_candidates: int | None = None,
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
            readout_policy=readout_policy,
            estimated_crossing_time_ms=estimated_crossing_time_ms,
            coarse_spike_candidates=coarse_spike_candidates,
            simultaneous_spike_candidates=simultaneous_spike_candidates,
        )
        serial = {
            **values,
            "emitted": None if index is None else values["emitted"].hex(),
            "query": query.hex(),
        }
        return ContextualEvidenceDecision(**values, sha256=_digest(serial))

    def _raw_drives(self, rows: np.ndarray) -> np.ndarray:
        return rows @ (self._weights[0] - self._weights[1]) + self._bias[0] - self._bias[1]

    def _currents(self, rows: np.ndarray) -> np.ndarray:
        return np.append(600.0 + 200.0 * np.tanh(self._raw_drives(rows)), 600.1)

    def respond(self, query: bytes) -> ContextualEvidenceDecision:
        rows, _ = self._features(query)
        currents = self._currents(rows)
        try:
            result = self._race.race_many(currents[None, :], require_unique=True)[0]
        except RestingGlifRaceFailure as error:
            return self._decision(
                query,
                rows,
                None,
                "neuronal",
                str(error),
                _RACE_STEPS * len(currents),
                readout_policy=self.readout_policy["default"],
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
            readout_policy=self.readout_policy["default"],
            coarse_spike_candidates=result.simultaneous_spike_candidates,
            simultaneous_spike_candidates=result.simultaneous_spike_candidates,
        )

    def respond_interpolated(self, query: bytes) -> ContextualEvidenceDecision:
        """Separate neuronal timing approximation at the same learned checkpoint."""
        rows, _ = self._features(query)
        currents = self._currents(rows)
        return self._interpolated_decision(
            query, rows, currents, "neuronal_interpolated", self.readout_policy["comparison"]
        )

    def respond_affine_interpolated(self, query: bytes) -> ContextualEvidenceDecision:
        """Opt-in bounded affine currents through the unchanged interpolated race.

        The map preserves mathematical ordering including the tonic STOP. Raw
        ties and numerical aliases remain STOPs; this is not universally
        injective in float64. Its identity belongs only to this decision, leaving
        existing checkpoint and readout identities unchanged.
        """
        rows, _ = self._features(query)
        currents = _affine_currents(self._raw_drives(rows))
        return self._interpolated_decision(
            query, rows, currents, "neuronal_affine_interpolated", _AFFINE_READOUT_POLICY
        )

    def _interpolated_decision(
        self, query: bytes, rows: np.ndarray, currents: np.ndarray, mode: str, policy: str
    ) -> ContextualEvidenceDecision:
        try:
            result = self._race.race_many_interpolated(currents[None, :])[0]
        except RestingGlifRaceFailure as error:
            return self._decision(
                query,
                rows,
                None,
                mode,
                str(error),
                _RACE_STEPS * len(currents),
                readout_policy=policy,
            )
        index = result.winner_cell if result.winner_cell < len(self._candidates) else None
        return self._decision(
            query,
            rows,
            index,
            mode,
            "tonic_stop" if index is None else None,
            (result.spike_step + 1) * len(currents),
            result.spike_step,
            readout_policy=policy,
            estimated_crossing_time_ms=result.estimated_crossing_time_ms,
            coarse_spike_candidates=result.coarse_spike_candidates,
            simultaneous_spike_candidates=result.simultaneous_spike_candidates,
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

    def raw_score_reference(
        self, query: bytes, *, interaction_ablation: str | None = None
    ) -> ContextualEvidenceDecision:
        """Disclosed linear reference before current saturation; no neuronal race."""
        rows, _ = self._features(query, interaction_ablation=interaction_ablation)
        drive = self._raw_drives(rows)
        # Invert the unchanged current map at the tonic STOP current. Require a
        # unique strict win, including against STOP, before float tanh saturation.
        scores = np.append(drive, np.arctanh((600.1 - 600.0) / 200.0))
        best = np.flatnonzero(scores == scores.max())
        index = int(best[0]) if len(best) == 1 and best[0] < len(self._candidates) else None
        mode = "raw_learned_score"
        if interaction_ablation is not None:
            mode += "_ablation_" + interaction_ablation
        return self._decision(
            query, rows, index, mode, "tonic_stop_or_tie" if index is None else None
        )

    def raw_candidate_proposal(self, query: bytes) -> ContextualEvidenceDecision:
        """TRAIN exploration: unique raw candidate leader, without the tonic STOP.

        This read-only proposal is not ordinary inference success. The caller
        must execute it before authenticating corrective feedback, as for other
        exploration decisions. All retained candidate occurrences participate.
        """
        rows, _ = self._features(query)
        drive = self._raw_drives(rows)
        best = np.flatnonzero(drive == drive.max()) if len(drive) else []
        index = int(best[0]) if len(best) == 1 else None
        return self._decision(
            query,
            rows,
            index,
            "raw_candidate_exploration",
            "empty_or_tied_candidates" if index is None else None,
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

    def _validated_rows(
        self, decision: ContextualEvidenceDecision, *, allow_ancestor: bool = False
    ) -> np.ndarray:
        if "_ablation_" in decision.mode:
            raise ValueError(
                "ablation decisions are read-only and cannot receive feedback or replay"
            )
        record = asdict(decision)
        expected = record.pop("sha256")
        record["query"] = decision.query.hex()
        record["emitted"] = None if decision.emitted is None else decision.emitted.hex()
        if expected != _digest(record):
            raise ValueError("feedback decision custody changed")
        candidate_count = (
            len(self._candidates) if decision.storage_sha256 == self._storage_root else None
        )
        if allow_ancestor:
            candidate_count = self._storage_ancestry.get(decision.storage_sha256)
        if candidate_count is None:
            raise ValueError("feedback decision has a different retained source store")
        if decision.candidate_index is None:
            raise ValueError("STOP has no selected candidate to credit")
        rows, _ = self._features(decision.query)
        index = decision.candidate_index
        if (
            type(index) is not int
            or not 0 <= index < candidate_count
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

    def revisit_decision(self, decision: ContextualEvidenceDecision) -> ContextualEvidenceDecision:
        """Rebind a retained executed occurrence after updates or append-only ingestion.

        The caller retains the original decision and authenticates its execution
        receipts. This is explicit experience replay, not autonomous retrieval or
        a fresh selection. The original storage root binds source identities;
        its candidate count and exact eligibility preserve the old occurrence.
        """
        rows = self._validated_rows(decision, allow_ancestor=True)
        return self._decision(
            query=decision.query,
            rows=rows,
            index=decision.candidate_index,
            mode="revisited_" + decision.mode,
        )

    def observe_feedback(self, decision: ContextualEvidenceDecision, reward: int) -> dict:
        """Apply exactly one consequence at this parent; wrapper authenticates execution."""
        if self.comparison_protection_policy != "none":
            raise ValueError("orthogonal protection requires pairwise comparison feedback")
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
        if self.comparison_protection_policy != "none":
            return self._observe_protected_comparison(rejected, accepted, delta, before)
        gap_before = float((self._weights[0] - self._weights[1]) @ delta)
        norm_squared = float(delta @ delta)
        proposed_rate = self.learning_rate
        skip_reason = None
        cap_active = False
        if self.comparison_step_policy == "unit_margin_capped":
            if norm_squared == 0.0 or gap_before >= 1.0:
                proposed_rate = 0.0
                skip_reason = "zero_delta" if norm_squared == 0.0 else "satisfied_margin"
            else:
                proposed_rate = min(
                    self.learning_rate,
                    (1.0 - gap_before) * np.sqrt(1.0 + norm_squared) / (2.0 * norm_squared),
                )
                cap_active = bool(proposed_rate < self.learning_rate)
        synapses_before = self.synapse_sha256
        applied = False
        if self.learning_enabled and proposed_rate > 0.0:
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
                applied = pair_learner.update(0, proposed_rate, 1.0) is not None
            finally:
                self._eligibility.fill(0.0)
        self.feedback_count += 2
        self.update_count += int(applied)
        for decision, reward in ((rejected, -1), (accepted, 1)):
            self._feedback_root = _digest([self._feedback_root, decision.sha256, reward])
        changed = self.synapse_sha256 != synapses_before
        if not changed and skip_reason is None:
            skip_reason = "learning_disabled" if not self.learning_enabled else "native_no_mutation"
        return {
            "before": before,
            "after": self.checkpoint_sha256,
            "rejected_decision_sha256": rejected.sha256,
            "accepted_decision_sha256": accepted.sha256,
            "eligibility_delta_sha256": sha256(delta.tobytes()).hexdigest(),
            "gap_before": gap_before,
            "gap_after": float((self._weights[0] - self._weights[1]) @ delta),
            "comparison_step_policy": self.comparison_step_policy,
            "eligibility_delta_norm_squared": norm_squared,
            "proposed_learning_rate": float(proposed_rate),
            "effective_learning_rate": float(proposed_rate) if changed else 0.0,
            "cap_active": cap_active,
            "step_skipped_reason": skip_reason,
            "learning_enabled": self.learning_enabled,
            "native_update_applied": applied,
            "synaptic_update": changed,
            "bias_policy": "zero scratch pair bias; persistent bias excluded and unchanged",
            "feedback_events": 2,
            "mode": "same_query_pairwise",
        }

    def _contrast_identity(self, rejected, accepted, delta_hash: str) -> str:
        occurrences = []
        for decision in (rejected, accepted):
            candidate = self._candidates[decision.candidate_index]
            source = self._documents[candidate.document_index]
            occurrences.append(
                [
                    candidate.document_index,
                    source.document_id.hex(),
                    source.provenance_id.hex(),
                    sha256(source.content).hexdigest(),
                    candidate.start,
                    candidate.end,
                ]
            )
        return _digest([rejected.query.hex(), occurrences, delta_hash])

    def _observe_protected_comparison(self, rejected, accepted, delta, before) -> dict:
        """Conventional native projection, followed by bounded own-offer consolidation.

        Execution certification remains the caller's responsibility, as for the
        original pair API. Only these two authenticated decisions may add state;
        no other ledger entry, task label or candidate score enters consolidation.
        """
        state_before = self.protection_summary
        v_before = self._weights[0] - self._weights[1]
        norm_squared = float(delta @ delta)
        norm = float(np.sqrt(norm_squared))
        gap_before = float(v_before @ delta)
        if not np.isfinite(v_before).all() or not np.isfinite([norm, gap_before]).all():
            raise ValueError("protected comparison requires finite weights and eligibility")
        dimension = len(delta)
        numerical_scale = np.finfo(np.float64).eps * dimension
        exhausted_tolerance = 64 * numerical_scale * max(1.0, norm)
        delta_hash = sha256(delta.tobytes()).hexdigest()
        identity = self._contrast_identity(rejected, accepted, delta_hash)
        protected_before = self._protected_deltas @ v_before
        synapses_before = self.synapse_sha256
        applied = False
        p = self._projected_eligibility
        try:
            if self.learning_enabled:
                pair_bias = np.zeros_like(self._bias)
                pair_learner = NativeThreeFactorLearner(
                    eligibility=self._eligibility,
                    output_heterogeneity_pa=self._heterogeneity,
                    readout_synapses_pa=self._weights,
                    pacemaker_synapses_pa=pair_bias,
                )
                self._eligibility[:] = delta
                applied = (
                    pair_learner.update_projected(
                        target=0,
                        learning_rate=self.learning_rate,
                        margin=1.0,
                        protected_basis=self._protection_basis,
                        projected_eligibility=p,
                        residual_tolerance=exhausted_tolerance,
                    )
                    is not None
                )
            else:
                # Frozen mates have no learned directions and never call a
                # mutating native entry point, even with temporary weights.
                if len(self._protection_records) or len(self._protection_basis):
                    raise RuntimeError("frozen comparison has learned protection state")
                p[:] = delta
                if norm <= exhausted_tolerance:
                    p.fill(0.0)
            v_after = self._weights[0] - self._weights[1]
            projected_norm = float(np.linalg.norm(p))
            if not np.isfinite(p).all() or not np.isfinite(v_after).all():
                raise RuntimeError("protected comparison produced nonfinite state")
            exhausted = projected_norm <= exhausted_tolerance
            residuals = self._protected_deltas @ p
            protected_after = self._protected_deltas @ v_after
            protected_norms = np.linalg.norm(self._protected_deltas, axis=1)
            residual_limits = (
                64 * numerical_scale * np.maximum(1.0, protected_norms * projected_norm)
            )
            preservation_limits = (
                128
                * numerical_scale
                * np.maximum.reduce(
                    [
                        np.ones(len(protected_before)),
                        np.abs(protected_before),
                        np.linalg.norm(v_before) * protected_norms,
                        np.linalg.norm(v_after) * protected_norms,
                    ]
                )
            )
            margin_changes = np.abs(protected_after - protected_before)
            if (
                not np.isfinite(residuals).all()
                or not np.isfinite(protected_after).all()
                or not np.isfinite(residual_limits).all()
                or not np.isfinite(preservation_limits).all()
                or np.any(np.abs(residuals) > residual_limits)
                or np.any(margin_changes > preservation_limits)
            ):
                raise RuntimeError("protected raw contrast residual or margin preservation failed")
            gap_after = float(v_after @ delta)
            if not np.isfinite(gap_after):
                raise RuntimeError("protected comparison produced a nonfinite margin")
            changed = self.synapse_sha256 != synapses_before
            consolidated = False
            independent = False
            basis_orthogonality_error = 0.0
            basis_orthogonality_tolerance = 64 * numerical_scale
            known = any(record["contrast_id"] == identity for record in self._protection_records)
            if self.learning_enabled and gap_after >= 1.0 and not known:
                if len(self._protection_records) >= self.protection_capacity:
                    raise RuntimeError("protected contrast capacity exceeded; eviction is disabled")
                independent = not exhausted
                basis_row = len(self._protection_basis) if independent else None
                if independent:
                    direction = p / projected_norm
                    basis_orthogonality_error = max(
                        abs(float(direction @ direction) - 1.0),
                        float(np.max(np.abs(self._protection_basis @ direction), initial=0.0)),
                    )
                    if basis_orthogonality_error > basis_orthogonality_tolerance:
                        raise RuntimeError("protected normalized basis orthogonality failed")
                    # Existing rows were checked at insertion and never change;
                    # checking this new row against them keeps work O(KD).
                    self._protection_basis = np.ascontiguousarray(
                        np.vstack((self._protection_basis, direction))
                    )
                self._protected_deltas = np.ascontiguousarray(
                    np.vstack((self._protected_deltas, delta))
                )
                self._protection_records.append(
                    {
                        "contrast_id": identity,
                        "eligibility_delta_sha256": delta_hash,
                        "rejected_decision_sha256": rejected.sha256,
                        "accepted_decision_sha256": accepted.sha256,
                        "certification_parent_sha256": before,
                        "certification_margin": gap_after,
                        "basis_row": basis_row,
                    }
                )
                consolidated = True
            projection = {
                "policy": self.comparison_protection_policy,
                "contrast_id": identity,
                "original_delta_sha256": delta_hash,
                "original_delta_norm": norm,
                "projected_delta_sha256": sha256(p.tobytes()).hexdigest(),
                "projected_delta_norm": projected_norm,
                "original_dot_projected": float(delta @ p),
                "projection_exhausted": exhausted,
                "projection_stalled": self.learning_enabled and gap_before < 1.0 and exhausted,
                "exhaustion_tolerance": exhausted_tolerance,
                "protected_residual_max": float(np.max(np.abs(residuals), initial=0.0)),
                "protected_margin_change_max": float(np.max(margin_changes, initial=0.0)),
                "protected_margin_checks": len(protected_before),
                "protected_margin_violations": 0,
                "protected_margins_before": protected_before.tolist(),
                "protected_margins_after": protected_after.tolist(),
                "protected_residual_tolerances": residual_limits.tolist(),
                "protected_margin_tolerances": preservation_limits.tolist(),
                "consolidated": consolidated,
                "independent_direction_added": independent,
                "basis_insertion_orthogonality_error": basis_orthogonality_error,
                "basis_orthogonality_tolerance": basis_orthogonality_tolerance,
                "already_certified": known,
                "basis_before_sha256": state_before["basis_sha256"],
                "basis_before_rank": state_before["basis_rank"],
                "protected_before_count": state_before["protected_count"],
                "state_before_sha256": state_before["state_sha256"],
            }
        finally:
            self._eligibility.fill(0.0)
            self._projected_eligibility.fill(0.0)
        self.feedback_count += 2
        self.update_count += int(applied)
        for decision, reward in ((rejected, -1), (accepted, 1)):
            self._feedback_root = _digest([self._feedback_root, decision.sha256, reward])
        state_after = self.protection_summary
        projection.update(
            basis_after_sha256=state_after["basis_sha256"],
            basis_after_rank=state_after["basis_rank"],
            protected_after_count=state_after["protected_count"],
            state_after_sha256=state_after["state_sha256"],
        )
        skip_reason = None
        if not self.learning_enabled:
            skip_reason = "learning_disabled"
        elif gap_before >= 1.0:
            skip_reason = "satisfied_margin"
        elif exhausted:
            skip_reason = "projection_exhausted"
        elif not changed:
            skip_reason = "native_no_mutation"
        return {
            "before": before,
            "after": self.checkpoint_sha256,
            "rejected_decision_sha256": rejected.sha256,
            "accepted_decision_sha256": accepted.sha256,
            "eligibility_delta_sha256": delta_hash,
            "gap_before": gap_before,
            "gap_after": gap_after,
            "comparison_step_policy": self.comparison_step_policy,
            "eligibility_delta_norm_squared": norm_squared,
            "proposed_learning_rate": self.learning_rate,
            "effective_learning_rate": self.learning_rate if changed else 0.0,
            "cap_active": False,
            "step_skipped_reason": skip_reason,
            "learning_enabled": self.learning_enabled,
            "native_update_applied": applied,
            "synaptic_update": changed,
            "bias_policy": "zero scratch pair bias; persistent bias excluded and unchanged",
            "feedback_events": 2,
            "mode": "same_query_pairwise",
            "protection": projection,
        }
