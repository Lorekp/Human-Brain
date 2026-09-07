"""One acting reader with paired dendrites and an unchanged neuronal prior.

Source admission and corrective teaching stage only the new branch bank. The
inherited address, codec, event, routing and recognition tensors remain held.
The existing action transaction still stages its own local heads before reader
publication. No journal, task identity or future byte supplies a forward input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_EOS, ByteGeneration
from humanbrain.dendritic_request_binding import DendriticRequestBinding, encode_request
from humanbrain.factorized_sequence_memory_learner import FactorizedSequenceStep
from humanbrain.neuronal_splice_learner import _digest
from humanbrain.recognition_sequence_memory_learner import _recognition_record
from humanbrain.sequence_assembly_memory_learner import (
    SequenceAssemblyFeedback,
    SequenceAssemblyPrediction,
    _add,
    _trace,
)
from humanbrain.sequence_assembly_observation import MAX_REPLACEMENT_BYTES
from humanbrain.sequence_assembly_prefix import RollingPrefixCoincidence2048
from humanbrain.tournament_recognition_learner import (
    TournamentRecognitionMemoryLearner,
    TournamentRecognitionSpliceLearner,
)


@dataclass(frozen=True)
class DendriticSequenceStep(FactorizedSequenceStep):
    binding: dict


@dataclass(frozen=True)
class DendriticSequenceFeedback(SequenceAssemblyFeedback):
    binding_updated: bool = False
    recognition_updated: bool = False
    routing_updated: bool = False


@dataclass(frozen=True)
class DendriticSourceFeedback:
    document_count: int
    raw_bytes: int
    symbol_count: int
    learning_enabled: bool
    writes_enabled: bool
    binding_updated: bool
    source_steps: tuple[dict, ...]
    model_before_sha256: str
    model_after_sha256: str
    memory_before_sha256: str | None
    memory_after_sha256: str
    work: dict


class DendriticSequenceMemoryLearner(TournamentRecognitionMemoryLearner):
    """Same response custody, actual causal prior, and observed-symbol dendrites."""

    def __init__(self, *, mode="learned", seed=0):
        super().__init__(mode=mode, seed=seed)
        self.binding = DendriticRequestBinding(seed=seed)
        self._binding_pending_owned = None
        self._binding_pending_request = None
        self._binding_pending_request_hash = None

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update(
            {
                "schema": "dendritic-sequence-memory-learner-v1",
                "routing": "retained tournament/recognition prior at every actual causal prefix",
                "routing_credit": "none; retained theta, recognition gains and event bank held",
                "recognition_credit": "none; retained recognition gains held",
                "codec_credit": "none; retained codec/U and event outgoing currents held",
                "codec_updated_semantics": "output_updated denotes only new branch-bank changes",
                "free_belief": "all257 class/prior/lower-byte tournament; no posterior recurrence",
                "loss": "prior-current NLL diagnostic; not a probability of class arbitration",
                "teacher_phase": "causal prior and branch activity then current external symbol",
                "publication": "whole correction/source branch stage validated before copies",
                "source_learning": "new raw bytes and EOS; absent request; zero prior",
                "source_disabled": (
                    "same tape admission; branch proposals discarded without advancement"
                ),
                "request_binding": "raw request<=256 held once; no workspace/feedback in new Q",
                "initialization": "retain prior; empty branch bank; no source replay on load",
                "binding_revision": (
                    "ephemeral stage custody; strict load invalidates stages; not experience"
                ),
                "trace_scope": (
                    "actual classes/prior/tournament and selected updates; contact scans counted"
                ),
            }
        )
        return metadata

    def _check_conditioned_parameters(self, work=None):
        super()._check_conditioned_parameters(work)
        if hasattr(self, "binding"):
            self.binding.check_parameters(work)

    def prepare_state_views(self, state, prefix="", work=None):
        self.binding.validate_state(state, prefix + "binding.", work)
        return super().prepare_state_views(state, prefix, work)

    def install_validated_views(self, prepared):
        super().install_validated_views(prepared)
        # Recursive Module loading bypasses the child's custom loader. Invalidate
        # any externally retained branch stage without replaying source symbols.
        self.binding.invalidate_stages_after_load()

    def _owned_values(self, state):
        """References/version guards inside one synchronous call, never a cache."""
        objects = (*self.modules(), self.memory, self._candidate_views, self._self_wirings)
        tensors = (
            *self.parameters(),
            *self.buffers(),
            *(
                tensor
                for view in self._candidate_views
                for tensor in (view.indices, view.values, view.flags)
            ),
            *(
                tensor
                for wiring in self._self_wirings
                for tensor in (wiring.indices, wiring.values)
            ),
            *(() if state is None else vars(state).values()),
        )
        return objects, tensors, tuple(tensor._version for tensor in tensors)

    def _check_owned(self, owned, state):
        objects, tensors, versions = owned
        current_objects, current_tensors, current_versions = self._owned_values(state)
        if (
            len(objects) != len(current_objects)
            or len(tensors) != len(current_tensors)
            or any(a is not b for a, b in zip(objects, current_objects, strict=True))
            or any(a is not b for a, b in zip(tensors, current_tensors, strict=True))
            or versions != current_versions
        ):
            raise RuntimeError("dendritic call rebound or mutated owned storage")

    def _prior_forward(self, prepared, prefix, cache, work, *, owned_free=None):
        inputs, fallback, address, scores = self._context_forward(prepared, prefix, cache, work)
        _, recognition, prior = self._recognize(
            inputs,
            fallback,
            self._candidate_views,
            self._self_wirings,
            self.assemblies.outgoing,
            work,
            owned_free=owned_free,
        )
        _add(work, "dendritic_prior_decisions", 1)
        return address, scores, recognition, prior

    @torch.no_grad()
    def predict_with_receipt(self, observation, state, max_bytes):
        self._require_idle()
        if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_REPLACEMENT_BYTES:
            raise ValueError("dendritic response max_bytes must be0..128")
        work = {}
        prepared = self._prepare_observation(observation, state, work)
        request = encode_request(observation.request, work)
        model_hash, memory_hash = _digest(self.state_dict(), work), _digest(vars(state), work)
        owned = self._owned_values(state)
        recognition_owner = self.recognition._begin_validated_free_response(self.router.theta, work)
        cache = RollingPrefixCoincidence2048(state.keys[: state.used], work=work)
        output, steps = bytearray(), []
        for position in range(max_bytes + 1):
            prefix = bytes(output)
            address, scores, recognition, prior = self._prior_forward(
                prepared, prefix, cache, work, owned_free=recognition_owner
            )
            decision = self.binding.evaluate(prefix, request, prior, work=work)
            token = decision.winner
            nll = float(-F.log_softmax(prior, dim=0)[token].item())
            steps.append(
                DendriticSequenceStep(
                    position,
                    prefix,
                    token,
                    address,
                    scores,
                    recognition.selected_index,
                    recognition.scores,
                    None,
                    prior,
                    nll,
                    _recognition_record(recognition),
                    decision.record(work),
                )
            )
            del decision
            _add(work, "dendritic_prior_log_softmaxes", 1)
            _add(work, "output_decisions", 1)
            if token == BYTE_EOS or position == max_bytes:
                break
            output.append(token)
            cache.advance(token)
        self._check_owned(owned, state)
        self.recognition._check_owned_free_response(recognition_owner)
        self._check_conditioned_parameters(work)
        if (
            _digest(self.state_dict(), work) != model_hash
            or _digest(vars(state), work) != memory_hash
        ):
            raise RuntimeError("dendritic response changed model or tape bytes")
        prediction = SequenceAssemblyPrediction(
            ByteGeneration(bytes(output), "eos" if token == BYTE_EOS else "limit"),
            tuple(steps),
            work,
            observation,
            max_bytes,
            f"{id(self)}:{self._active_experience + 1}",
            model_hash,
            memory_hash,
            "",
        )
        request_hash = _digest(
            {
                "raw": request.raw_request.hex(),
                "population": request.population,
                "seal": request.seal,
            },
            work,
        )
        trace_hash = _digest(_trace(prediction), work)
        object.__setattr__(prediction, "trace_sha256", trace_hash)
        self._sequence_pending = (
            prediction,
            observation.sha256,
            model_hash,
            memory_hash,
            trace_hash,
            _digest(work),
        )
        self._binding_pending_owned = owned
        self._binding_pending_request = request
        self._binding_pending_request_hash = request_hash
        self._active_experience += 1
        return prediction

    def validate_pending(self, prediction, state, *, observation=None, work=None):
        super().validate_pending(prediction, state, observation=observation, work=work)
        if self._binding_pending_owned is None:
            raise ValueError("dendritic response lacks its owned-storage seal")
        self._check_owned(self._binding_pending_owned, state)
        request = self._binding_pending_request
        if (
            request is None
            or request.raw_request != prediction.observation.request
            or _digest(
                {
                    "raw": request.raw_request.hex(),
                    "population": request.population,
                    "seal": request.seal,
                },
                work,
            )
            != self._binding_pending_request_hash
        ):
            raise ValueError("pending dendritic request activity changed")

    def acknowledge(self, prediction, state):
        result = super().acknowledge(prediction, state)
        self._binding_pending_owned = None
        self._binding_pending_request = None
        self._binding_pending_request_hash = None
        return result

    @torch.no_grad()
    def teach(
        self, observation, teacher, state, learning_rate=0.01, *, prediction, learning_enabled=True
    ):
        if (
            type(teacher) is not bytes
            or len(teacher) > MAX_REPLACEMENT_BYTES
            or type(learning_rate) not in (float, int)
            or learning_rate != 0.01
            or type(learning_enabled) is not bool
        ):
            raise ValueError("dendritic teacher needs bounded bytes, eta.01 and Boolean gate")
        work = {}
        self.validate_pending(prediction, state, observation=observation, work=work)
        if len(teacher) > prediction.max_bytes:
            raise ValueError("teacher exceeds the actual response capacity")
        prepared = self._prepare_observation(observation, state, work)
        request = self._binding_pending_request
        cache = RollingPrefixCoincidence2048(state.keys[: state.used], work=work)
        stage = self.binding.begin_stage(work)
        enabled = learning_enabled and self.mode != "slow_frozen"
        records, loss, changed = [], 0.0, False
        for position in range(len(teacher) + 1):
            prefix = teacher[:position]
            address, scores, recognition, prior = self._prior_forward(prepared, prefix, cache, work)
            decision = self.binding.evaluate(prefix, request, prior, stage=stage, work=work)
            before = decision.record(work)
            target = teacher[position] if position < len(teacher) else BYTE_EOS
            update = self.binding.prepare_observation(
                stage, decision, target, apical_learning=True, enabled=enabled, work=work
            )
            nll = float(-F.log_softmax(prior, dim=0)[target].item())
            loss += nll
            if not math.isfinite(loss):
                raise ValueError("dendritic prior-current teacher diagnostic must be finite")
            changed |= update.staged and bool(
                update.changed_contacts or update.recruited or update.attached_apical
            )
            records.append(
                {
                    "output_step": position,
                    "prefix_hex": prefix.hex(),
                    "observed_token": target,
                    "selected_index": address,
                    "address_scores": scores,
                    "routing": _recognition_record(recognition),
                    "prior_logits": prior,
                    "prior_nll": nll,
                    "binding": before,
                    "update": update.record(),
                }
            )
            del decision, update
            _add(work, "dendritic_prior_log_softmaxes", 1)
            _add(work, "teacher_offers", 1)
            _add(work, "output_target_offers", 1)
            if target != BYTE_EOS:
                cache.advance(target)
        self.binding.validate_stage(stage, work)
        self.validate_pending(prediction, state, observation=observation, work=work)
        feedback = DendriticSequenceFeedback(
            state,
            prediction.attempt_id,
            len(teacher) + 1,
            loss,
            loss / (len(teacher) + 1),
            changed,
            tuple(records),
            work,
            binding_updated=changed,
        )
        if enabled:
            self.binding.commit_validated(stage, work)
        self._sequence_pending = None
        self._binding_pending_owned = None
        self._binding_pending_request = None
        self._binding_pending_request_hash = None
        return feedback

    @torch.no_grad()
    def ingest_with_receipt(
        self, documents, state=None, *, learning_enabled=True, writes_enabled=True, work=None
    ):
        self._require_idle()
        if (
            not isinstance(documents, Sequence)
            or isinstance(documents, (bytes, str))
            or any(type(document) is not bytes for document in documents)
            or type(learning_enabled) is not bool
            or type(writes_enabled) is not bool
        ):
            raise ValueError("source admission needs raw documents and independent Boolean gates")
        documents = tuple(documents)
        if work is None:
            work = {}
        elif type(work) is not dict:
            raise ValueError("source work ledger must be a dictionary")
        self._check_conditioned_parameters(work)
        if state is not None:
            self._check_state(state, populated=False)
        model_hash = _digest(self.state_dict(), work)
        memory_hash = None if state is None else _digest(vars(state), work)
        owned = self._owned_values(state)
        writes = writes_enabled and self.mode != "writes_disabled"
        enabled = learning_enabled and self.mode != "slow_frozen"
        # The memory implementation validates capacity before any source offers
        # and returns a private TapeState. Never install that state mid-stream.
        result = self.memory.ingest(documents, state, writes_enabled=writes, work=work)
        _add(
            work,
            "dendritic_source_tape_allocated_bytes",
            sum(tensor.numel() * tensor.element_size() for tensor in vars(result).values()),
        )
        _add(
            work,
            "dendritic_source_tape_copy_bytes",
            0
            if state is None
            else sum(tensor.numel() * tensor.element_size() for tensor in vars(state).values()),
        )
        stage = self.binding.begin_stage(work)
        prior = torch.zeros(257, dtype=torch.float64)
        _add(work, "dendritic_source_zero_prior_values", 257)
        records, changed = [], False
        for document_index, document in enumerate(documents):
            for position in range(len(document) + 1):
                prefix = document[:position]
                decision = self.binding.evaluate(prefix, None, prior, stage=stage, work=work)
                before = decision.record(work)
                target = document[position] if position < len(document) else BYTE_EOS
                update = self.binding.prepare_observation(
                    stage, decision, target, apical_learning=False, enabled=enabled, work=work
                )
                changed |= update.staged and bool(update.changed_contacts or update.recruited)
                records.append(
                    {
                        "document_index": document_index,
                        "output_step": position,
                        "observed_token": target,
                        "binding": before,
                        "update": update.record(),
                    }
                )
                del decision, update
        self.binding.validate_stage(stage, work)
        self._check_state(result, populated=False)
        self._check_owned(owned, state)
        self._check_conditioned_parameters(work)
        if _digest(self.state_dict(), work) != model_hash or (
            state is not None and _digest(vars(state), work) != memory_hash
        ):
            raise RuntimeError("source admission changed live model or input tape")
        proposed = dict(self.state_dict())
        if enabled and records:
            proposed.update({"binding." + name: tensor for name, tensor in stage.bank.items()})
        feedback = DendriticSourceFeedback(
            len(documents),
            sum(map(len, documents)),
            len(records),
            enabled,
            writes,
            changed,
            tuple(records),
            model_hash,
            _digest(proposed, work),
            memory_hash,
            _digest(vars(result), work),
            work,
        )
        if enabled and records:
            self.binding.commit_validated(stage, work)
        return result, feedback

    def ingest(self, documents, state=None, *, learning_enabled=True, writes_enabled=True):
        result, _ = self.ingest_with_receipt(
            documents, state, learning_enabled=learning_enabled, writes_enabled=writes_enabled
        )
        return result

    def accounting(self):
        result = super().accounting()
        result.update(
            {
                "schema": "dendritic-sequence-accounting-v1",
                "binding_tensor_bytes": self.binding.tensor_bytes,
                "binding_stage_tensor_bytes": self.binding.tensor_bytes,
                "binding_source_prior": "one zero257-vector; no inherited source-prefix queries",
                "binding_correction_prior": "actual inherited prior once per offered causal prefix",
                "binding_trace_scope": (
                    "compact actual arbitration/selected contacts; omitted scans counted"
                ),
            }
        )
        return result


class DendriticSequenceSpliceLearner(TournamentRecognitionSpliceLearner):
    """The same reader generates payloads, admits sources and receives corrections."""

    def __init__(self, *, mode="learned", seed=733, reader=None):
        if mode not in self.MODES:
            raise ValueError("invalid dendritic action mode")
        if reader is None:
            reader = DendriticSequenceMemoryLearner(mode=self.MODES[mode], seed=seed)
        if not isinstance(reader, DendriticSequenceMemoryLearner):
            raise ValueError("dendritic actions require the same dendritic reader")
        super().__init__(mode=mode, seed=seed, reader=reader)

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update(
            {
                "schema": "dendritic-sequence-splice-learner-v1",
                "reader": "dendritic-sequence-memory-learner-v1",
                "publication": "all action proposals then dendritic stage then action copies",
                "source": "explicit ingest_with_receipt; no source replay during strict loading",
            }
        )
        return metadata

    def ingest_with_receipt(
        self, documents, state=None, *, learning_enabled=True, writes_enabled=True
    ):
        self._require_idle()
        work = {}
        self._check(work)
        return self.reader.ingest_with_receipt(
            documents,
            state,
            learning_enabled=learning_enabled,
            writes_enabled=writes_enabled,
            work=work,
        )

    def ingest(self, documents, state=None, *, learning_enabled=True, writes_enabled=True):
        state, _ = self.ingest_with_receipt(
            documents, state, learning_enabled=learning_enabled, writes_enabled=writes_enabled
        )
        return state
