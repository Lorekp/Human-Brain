"""One causal controller inside the retained reading and executed-action actor.

The old address, codec, event, recognition and dendritic tensors are numerical
priors. Only the new controller learns from scans and public consequences;
action heads retain their existing separately staged local corrections.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_EOS, ByteGeneration
from humanbrain.dendritic_request_binding import encode_request
from humanbrain.dendritic_sequence_learner import (
    DendriticSequenceMemoryLearner,
    DendriticSequenceSpliceLearner,
    DendriticSequenceStep,
)
from humanbrain.neuronal_splice_learner import _digest
from humanbrain.recognition_sequence_memory_learner import _recognition_record
from humanbrain.recurrent_content_controller import RecurrentContentController
from humanbrain.recurrent_content_frame import (
    advance_output,
    apply_gate,
    context_activity,
    emission_activity,
    enter_field,
    initial_position,
    output_position,
    public_frame,
    read_registers,
    receive,
    source_frame,
)
from humanbrain.sequence_assembly_memory_learner import (
    SequenceAssemblyFeedback,
    SequenceAssemblyPrediction,
    _add,
    _trace,
)
from humanbrain.sequence_assembly_observation import MAX_REPLACEMENT_BYTES
from humanbrain.sequence_assembly_prefix import RollingPrefixCoincidence2048


@dataclass(frozen=True)
class RecurrentContentStep(DendriticSequenceStep):
    controller: dict


@dataclass(frozen=True)
class RecurrentContentPrediction(SequenceAssemblyPrediction):
    scan_steps: tuple[dict, ...]
    controller: dict


@dataclass(frozen=True)
class RecurrentContentFeedback(SequenceAssemblyFeedback):
    controller: dict | None = None


@dataclass(frozen=True)
class RecurrentContentSourceFeedback:
    document_count: int
    raw_bytes: int
    symbol_count: int
    learning_enabled: bool
    writes_enabled: bool
    fact_admitted: bool
    source_steps: tuple[dict, ...]
    controller: dict
    model_before_sha256: str
    model_after_sha256: str
    memory_before_sha256: str | None
    memory_after_sha256: str
    work: dict


def _custody_tree(value):
    # Frozen _digest understands mappings/tensors, but treats a sequence as JSON.
    # Draw sequences contain tensors, so expose those leaves without changing
    # any historical hashing utility or conflating tuples, lists and mappings.
    if isinstance(value, Mapping):
        return {key: _custody_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return {"sequence_type": type(value).__name__,
                "items": {str(i): _custody_tree(item) for i, item in enumerate(value)}}
    return value


def _controller_trace(prediction):
    return _custody_tree({
        **_trace(prediction),
        "scan_steps": prediction.scan_steps,
        "controller": prediction.controller,
    })


def _frame_content(frame, work=None):
    return _digest({
        "fields": tuple((f.field, f.tokens, f.addresses, f.supervised, f.status)
                        for f in frame.fields),
        "symbols": frame.symbols, "successors": frame.successors, "reveal": frame.reveal,
        "fact_used": frame.fact_used, "source": frame.source, "seal": frame.seal,
    }, work)


class RecurrentContentMemoryLearner(DendriticSequenceMemoryLearner):
    """Held V7 prior, causal public scan, and one unresolved stochastic response."""

    def __init__(self, *, mode="learned", seed=0):
        super().__init__(mode=mode, seed=seed)
        self.controller = RecurrentContentController(seed=733)
        self._initial_controller_state_sha256 = _digest(self.controller.state_dict())
        self._content_pending = None
        self._resolution_context = None
        self.last_controller_resolution = None

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "recurrent-content-memory-learner-v2",
            "routing": "held complete V7 prior once per actual free/teacher prefix",
            "codec_updated_semantics": "output_updated denotes new controller weight changes",
            "free_belief": "causal context/gate/mode/optional-literal stochastic populations",
            "loss": "held prior-current NLL diagnostic, not controller trajectory likelihood",
            "teacher_phase": "pre-output neural fork; fresh RNG after actual free stopping time",
            "publication": "held scan/free/teacher eligibility and causal credit; joint validated copies",
            "local_credit": "causal-observation-credit-v1; direct literal CE; upstream own eligibility",
            "source_learning": "one causal document per transaction; old tape; no output prior",
            "source_disabled": "RNG acknowledged; weights/baselines held; tape gate separate",
            "request_binding": "prefix-visible Q during scan; complete Q only after request EOS",
            "initialization": "explicit retained lineage; metadata adaptation preserves numeric state",
            "trace_scope": "all actual controller draws/scans and held prior arbitration",
            "transient_source": "observe_source preserves tape object and every factual byte",
            "controller_seed": 733,
        })
        return metadata

    def _require_idle(self):
        super()._require_idle()
        if hasattr(self, "controller") and self.controller.pending_stage is not None:
            raise ValueError("resolve the active recurrent controller transaction first")

    def _check_conditioned_parameters(self, work=None):
        super()._check_conditioned_parameters(work)
        if hasattr(self, "controller"):
            self.controller.check_parameters(work)

    def prepare_state_views(self, state, prefix="", work=None):
        self.controller.validate_state(state, prefix + "controller.", work)
        return super().prepare_state_views(state, prefix, work)

    def install_validated_views(self, prepared):
        super().install_validated_views(prepared)
        self.controller.invalidate_stages_after_load()

    def _v7_mapping(self, state, prefix, old_metadata, work):
        """Mechanical schema extension; the ordinary strict loader validates all tensors.

        No lesson, pending response or experience counter is inferred from a
        tensor checkpoint. The external initializer binds those idle counters.
        Returned tensors are borrowed until the caller's strict copy-only load.
        """
        if not isinstance(state, Mapping):
            raise ValueError("V7 numerical adaptation requires a complete state mapping")
        if _digest(self.controller.state_dict(), work) != self._initial_controller_state_sha256:
            raise ValueError("V7 adaptation requires the unchanged newly initialized controller")
        result = dict(state)
        for name, expected in old_metadata.items():
            if result.get(name) != expected:
                raise ValueError("V7 adapter metadata differs from the exact retained actor")
        controller_prefix = prefix + "controller."
        if any(name.startswith(controller_prefix) for name in result):
            raise ValueError("V7 adaptation cannot replace an existing controller lineage")
        result.update({controller_prefix + name: value
                       for name, value in self.controller.state_dict().items()})
        result[prefix + "_extra_state"] = self.get_extra_state()
        _add(work, "content_v7_adapter_calls", 1)
        _add(work, "content_v7_adapter_retained_tensor_bytes", sum(
            value.numel() * value.element_size() for value in state.values()
            if isinstance(value, torch.Tensor)))
        return result

    def load_v7_state_dict(self, state, *, work=None):
        """Explicit one-time numeric continuation; no decoder, replay or shadow actor."""
        self._require_idle()
        if work is None:
            work = {}
        expected = {name for name in self.state_dict() if not name.startswith("controller.")}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("V7 reader adaptation requires every retained state key exactly")
        mapped = self._v7_mapping(state, "", {
            "_extra_state": DendriticSequenceMemoryLearner.get_extra_state(self)}, work)
        return self.load_state_dict(mapped, strict=True, assign=False, work=work)

    def _tick(self, stage, frame, position, h, cursors, category, work,
              *, prior_token=None, address=None):
        """No offered symbol enters until this entire stochastic tick completes."""
        before_h, before_cursors = h, cursors
        activity = context_activity(frame, position, h, cursors,
                                    prior_token=prior_token, work=work)
        context = self.controller.draw(stage, "context", activity, category=category, work=work)
        h = context.index
        activity = context_activity(frame, position, h, cursors,
                                    prior_token=prior_token, work=work)
        gate = self.controller.draw(stage, "gate", activity, category=category, work=work)
        cursors = apply_gate(frame, position, cursors, gate.index,
                             retrieved_index=address, work=work)
        registers = read_registers(frame, position, cursors, work)
        activity = emission_activity(frame, position, h, cursors,
                                     prior_token=prior_token, work=work)
        mode = self.controller.draw(stage, "mode", activity, category=category, work=work)
        draws = [context.record(work), gate.record(work), mode.record(work)]
        if mode.index == 3:
            literal = self.controller.draw(stage, "literal", activity,
                                           category=category, work=work)
            draws.append(literal.record(work))
        token = self.controller.complete_tick(stage, prior_token=prior_token,
                                              register_tokens=registers, work=work)
        record = {
            "schema": "recurrent-content-tick-v1", "category": category,
            "frame_sha256": frame.seal, "position_sha256": position.seal,
            "field": frame.fields[position.field_index].field,
            "offset": position.offset, "arrived": position.arrived,
            "h_before": before_h, "h_after": h,
            "cursors_before": before_cursors, "cursors_after": cursors,
            "register_tokens": registers, "prior_token": prior_token,
            "retrieved_index": address, "emitted_token": token, "draws": tuple(draws),
        }
        return token, h, cursors, record

    def _scan(self, stage, frame, work):
        position, h, cursors, records = initial_position(frame), 0, (None, None), []
        for index, item in enumerate(frame.fields):
            position = enter_field(frame, position, index, work)
            for symbol in item.tokens:
                if item.supervised:
                    _, h, cursors, record = self._tick(
                        stage, frame, position, h, cursors, "scan", work)
                    reward = self.controller.observe(stage, symbol, category="scan", work=work)
                    record.update({"observed_token": symbol, "reward": reward,
                                   "credit": self.controller.observation_record(stage, work),
                                   "status_offer": item.status})
                    records.append(record)
                position = receive(frame, position, symbol, work)
        self.controller.seal_scan(stage, work)
        return position, h, cursors, tuple(records)

    def _prior(self, prepared, prefix, cache, request, work, owner=None):
        address, scores, recognition, prior = self._prior_forward(
            prepared, prefix, cache, work, owned_free=owner)
        decision = self.binding.evaluate(prefix, request, prior, work=work)
        _add(work, "content_held_prior_calls", 1)
        return address, scores, recognition, prior, decision

    @torch.no_grad()
    def predict_with_receipt(self, observation, state, max_bytes, *, interaction_kind=None):
        self._require_idle()
        if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_REPLACEMENT_BYTES:
            raise ValueError("recurrent response max_bytes must be0..128")
        kind = interaction_kind or ("reading" if observation.workspace is None else "edit")
        if kind not in ("reading", "edit") or (kind == "edit" and observation.workspace is None):
            raise ValueError("interaction kind must describe the actual public action interface")
        work = {}
        prepared = self._prepare_observation(observation, state, work)
        request = encode_request(observation.request, work)
        frame = public_frame(observation, state, work)
        model_hash, memory_hash = _digest(self.state_dict(), work), _digest(vars(state), work)
        owned = self._owned_values(state)
        owner = self.recognition._begin_validated_free_response(self.router.theta, work)
        stage = self.controller.begin_stage(kind=kind, scan_offers=frame.offer_count,
                                            max_bytes=max_bytes, work=work)
        try:
            position, h, cursors, scan = self._scan(stage, frame, work)
            position = output_position(frame, position)
            preoutput = position, h, cursors
            cache = RollingPrefixCoincidence2048(state.keys[:state.used], work=work)
            output, steps = bytearray(), []
            for index in range(max_bytes + 1):
                prefix = bytes(output)
                address, scores, recognition, prior, decision = self._prior(
                    prepared, prefix, cache, request, work, owner)
                token, h, cursors, record = self._tick(
                    stage, frame, position, h, cursors, "free", work,
                    prior_token=decision.winner, address=address)
                nll = float(-F.log_softmax(prior, dim=0)[token].item())
                steps.append(RecurrentContentStep(
                    index, prefix, token, address, scores, recognition.selected_index,
                    recognition.scores, None, prior, nll, _recognition_record(recognition),
                    decision.record(work), record))
                _add(work, "output_decisions", 1)
                _add(work, "content_prior_log_softmaxes", 1)
                if token == BYTE_EOS or index == max_bytes:
                    break
                output.append(token)
                cache.advance(token)
                position = advance_output(position, token, work)
            self.controller.seal_free(stage, work)
            self._check_owned(owned, state)
            self.recognition._check_owned_free_response(owner)
            if (_digest(self.state_dict(), work) != model_hash
                    or _digest(vars(state), work) != memory_hash):
                raise RuntimeError("recurrent free trajectory changed live model or tape")
            stage_hash = self.controller.stage_digest(stage, work)
            request_hash = _digest({"raw": request.raw_request.hex(),
                                    "population": request.population, "seal": request.seal}, work)
            controller_record = {
                "schema": "recurrent-content-response-v1", "kind": kind,
                "frame_sha256": frame.seal, "scan_offers": frame.offer_count,
                "scan_status_offers": sum(len(f.tokens) for f in frame.fields if f.status),
                "preoutput_position_sha256": preoutput[0].seal,
                "preoutput_h": preoutput[1], "preoutput_cursors": preoutput[2],
                "pending_stage_sha256": stage_hash,
                "frame_content_sha256": _frame_content(frame, work),
                "prior_request_sha256": request_hash,
            }
            prediction = RecurrentContentPrediction(
                ByteGeneration(bytes(output), "eos" if token == BYTE_EOS else "limit"),
                tuple(steps), work, observation, max_bytes,
                f"{id(self)}:{self._active_experience + 1}", model_hash, memory_hash, "",
                scan, controller_record)
            trace_hash = _digest(_controller_trace(prediction), work)
            object.__setattr__(prediction, "trace_sha256", trace_hash)
            self._content_pending = (stage, stage_hash, frame, preoutput, owned, request,
                                     request_hash, kind)
            self._sequence_pending = (prediction, observation.sha256, model_hash, memory_hash,
                                      trace_hash, _digest(work))
            self._active_experience += 1
            return prediction
        except BaseException:
            self.controller.discard_stage(stage)
            raise

    def validate_pending(self, prediction, state, *, observation=None, work=None):
        if work is None:
            work = {}
        self._check_conditioned_parameters(work)
        self._check_state(state)
        if (not isinstance(prediction, RecurrentContentPrediction)
                or not self.is_pending(prediction) or self._content_pending is None):
            raise ValueError("response belongs to another actor or has already resolved")
        _, obs_hash, model_hash, memory_hash, trace_hash, work_hash = self._sequence_pending
        if (prediction.observation.sha256 != obs_hash
                or (observation is not None and observation.sha256 != obs_hash)
                or prediction.model_sha256 != model_hash or prediction.memory_sha256 != memory_hash
                or _digest(self.state_dict(), work) != model_hash
                or _digest(vars(state), work) != memory_hash
                or prediction.trace_sha256 != trace_hash
                or _digest(_controller_trace(prediction), work) != trace_hash
                or _digest(prediction.work) != work_hash):
            raise ValueError("pending recurrent observation, model, memory or trace changed")
        (stage, stage_hash, frame, preoutput, owned, request,
         request_hash, kind) = self._content_pending
        self._check_owned(owned, state)
        if (self.controller.stage_digest(stage, work) != stage_hash
                or stage_hash != prediction.controller["pending_stage_sha256"]
                or request_hash != prediction.controller["prior_request_sha256"]
                or _frame_content(frame, work) != prediction.controller["frame_content_sha256"]
                or frame.seal != prediction.controller["frame_sha256"]
                or preoutput[0].seal != prediction.controller["preoutput_position_sha256"]
                or preoutput[1] != prediction.controller["preoutput_h"]
                or preoutput[2] != prediction.controller["preoutput_cursors"]
                or kind != prediction.controller["kind"]
                or request.raw_request != prediction.observation.request
                or _digest({"raw": request.raw_request.hex(), "population": request.population,
                            "seal": request.seal}, work) != request_hash):
            raise ValueError("pending causal frame, eligibility, RNG or prior request changed")
        _add(work, "sequence_trace_validations", 1)

    def _resolution(self, prediction, learning_enabled):
        if type(learning_enabled) is not bool:
            raise ValueError("controller publication gate must be Boolean")
        if self._content_pending[-1] == "reading":
            if self._resolution_context is not None:
                raise ValueError("reading cannot consume an action reward")
            return None, learning_enabled and self.mode != "slow_frozen"
        context = self._resolution_context
        if context is None or context[0] is not prediction:
            raise ValueError("edit resolution requires its validated actual execution")
        _, _, public_reward, enabled = context
        if enabled != learning_enabled:
            raise ValueError("action and reader learning gates disagree")
        return public_reward, enabled and self.mode != "slow_frozen"

    def _publish(self, proposal, record, work):
        self.controller.commit_validated(proposal, work)
        self._sequence_pending = None
        self._content_pending = None
        self.last_controller_resolution = record

    @torch.no_grad()
    def teach(self, observation, teacher, state, learning_rate=0.01, *, prediction,
              learning_enabled=True):
        if (type(teacher) is not bytes or len(teacher) > MAX_REPLACEMENT_BYTES
                or type(learning_rate) not in (float, int) or learning_rate != 0.01
                or type(learning_enabled) is not bool):
            raise ValueError("teacher needs bounded raw bytes, eta.01 and a Boolean gate")
        work = {}
        self.validate_pending(prediction, state, observation=observation, work=work)
        if len(teacher) > prediction.max_bytes:
            raise ValueError("teacher exceeds actual response capacity")
        reward, enabled = self._resolution(prediction, learning_enabled)
        root_stage, _, frame, preoutput, _, request, _, _ = self._content_pending
        stage = self.controller.fork_stage(root_stage, work)
        try:
            self.controller.declare_teacher(stage, len(teacher) + 1, work)
            position, h, cursors = preoutput
            prepared = self._prepare_observation(observation, state, work)
            cache = RollingPrefixCoincidence2048(state.keys[:state.used], work=work)
            records, loss = [], 0.0
            for index in range(len(teacher) + 1):
                prefix = teacher[:index]
                address, scores, recognition, prior, decision = self._prior(
                    prepared, prefix, cache, request, work)
                _, h, cursors, record = self._tick(
                    stage, frame, position, h, cursors, "teacher", work,
                    prior_token=decision.winner, address=address)
                target = teacher[index] if index < len(teacher) else BYTE_EOS
                outcome = self.controller.observe(stage, target, category="teacher", work=work)
                record["credit"] = self.controller.observation_record(stage, work)
                nll = float(-F.log_softmax(prior, dim=0)[target].item())
                loss += nll
                records.append({"output_step": index, "prefix_hex": prefix.hex(),
                                "observed_token": target, "selected_index": address,
                                "address_scores": scores,
                                "routing": _recognition_record(recognition),
                                "prior_logits": prior, "prior_nll": nll,
                                "binding": decision.record(work), "controller": record,
                                "reward": outcome})
                _add(work, "teacher_offers", 1)
                _add(work, "output_target_offers", 1)
                _add(work, "content_prior_log_softmaxes", 1)
                if target != BYTE_EOS:
                    cache.advance(target)
                    position = advance_output(position, target, work)
            proposal = self.controller.finish(stage, learning_enabled=enabled,
                                               public_reward=reward, work=work)
            update = proposal.record()
            changed = enabled and proposal.delta_norm != 0
            feedback = RecurrentContentFeedback(
                state, prediction.attempt_id, len(teacher) + 1, loss,
                loss / (len(teacher) + 1), changed, tuple(records), work, controller=update)
            self.validate_pending(prediction, state, observation=observation, work=work)
            self.controller.validate_stage(proposal, work)
            self._publish(proposal, update, work)
            return feedback
        except BaseException:
            self.controller.discard_stage(stage)
            raise

    @torch.no_grad()
    def acknowledge(self, prediction, state, *, learning_enabled=True):
        work = {}
        self.validate_pending(prediction, state, work=work)
        if self._resolution_context is not None:
            learning_enabled = self._resolution_context[3]
        reward, enabled = self._resolution(prediction, learning_enabled)
        stage = self.controller.fork_stage(self._content_pending[0], work)
        try:
            proposal = self.controller.finish(stage, learning_enabled=enabled,
                                               public_reward=reward, work=work)
            record = proposal.record()
            _add(work, "sequence_acknowledgements", 1)
            self.validate_pending(prediction, state, work=work)
            self.controller.validate_stage(proposal, work)
            self._publish(proposal, record, work)
            return work
        except BaseException:
            self.controller.discard_stage(stage)
            raise

    @torch.no_grad()
    def _source(self, document, state, *, admitted, learning_enabled, writes_enabled, work):
        self._require_idle()
        if (type(document) is not bytes or type(learning_enabled) is not bool
                or type(writes_enabled) is not bool or (not admitted and state is None)):
            raise ValueError("one raw source and explicit gates require an unchanged valid tape")
        if work is None:
            work = {}
        if type(work) is not dict:
            raise ValueError("source work ledger must be a dictionary")
        self._check_conditioned_parameters(work)
        if state is not None:
            self._check_state(state, populated=False)
        model_hash = _digest(self.state_dict(), work)
        memory_hash = None if state is None else _digest(vars(state), work)
        owned = self._owned_values(state)
        writes = admitted and writes_enabled and self.mode != "writes_disabled"
        enabled = learning_enabled and self.mode != "slow_frozen"
        result = self.memory.ingest((document,), state, writes_enabled=writes, work=work) \
            if admitted else state
        if admitted:
            _add(work, "content_source_tape_allocated_bytes",
                 sum(t.numel() * t.element_size() for t in vars(result).values()))
            _add(work, "content_source_tape_copy_bytes", 0 if state is None else
                 sum(t.numel() * t.element_size() for t in vars(state).values()))
        frame = source_frame(document, state, work)
        stage = self.controller.begin_stage(kind="source", scan_offers=frame.offer_count, work=work)
        try:
            _, _, _, records = self._scan(stage, frame, work)
            proposal = self.controller.finish(stage, learning_enabled=enabled, work=work)
            self._check_state(result, populated=False)
            self._check_owned(owned, state)
            if (_digest(self.state_dict(), work) != model_hash
                    or (state is not None and _digest(vars(state), work) != memory_hash)):
                raise RuntimeError("source scan changed live prior or original tape")
            proposed = dict(self.state_dict())
            proposed.update({"controller." + key: value for key, value in proposal.state.items()})
            update = proposal.record()
            _add(work, "content_admitted_source_symbols" if admitted else
                 "content_transient_source_symbols", len(document) + 1)
            feedback = RecurrentContentSourceFeedback(
                1, len(document), len(document) + 1, enabled, writes, admitted, records, update,
                model_hash, _digest(proposed, work), memory_hash, _digest(vars(result), work), work)
            self.controller.validate_stage(proposal, work)
            self.controller.commit_validated(proposal, work)
            self.last_controller_resolution = update
            return result, feedback
        except BaseException:
            self.controller.discard_stage(stage)
            raise

    def ingest_with_receipt(self, documents, state=None, *, learning_enabled=True,
                            writes_enabled=True, work=None):
        if (not isinstance(documents, Sequence) or isinstance(documents, (bytes, str))
                or len(documents) != 1 or type(documents[0]) is not bytes):
            raise ValueError("recurrent fact admission requires exactly one raw document")
        return self._source(documents[0], state, admitted=True, learning_enabled=learning_enabled,
                            writes_enabled=writes_enabled, work=work)

    def observe_source(self, document, state, *, learning_enabled=True, work=None):
        """One transient causal source; return the identical, unmodified tape object."""
        return self._source(document, state, admitted=False, learning_enabled=learning_enabled,
                            writes_enabled=False, work=work)

    def accounting(self):
        result = super().accounting()
        result.update({
            "schema": "recurrent-content-accounting-v2",
            "controller_tensor_bytes": self.controller.tensor_bytes,
            "controller_weight_bytes": 24623616,
            "controller_eligibility_bytes": 3 * 24623616,
            "resolution_fork_eligibility_bytes": 3 * 24623616,
            "controller_causal_prefix_bytes": 2 * 2802448 * 8,
            "controller_literal_direct_bytes": 2 * 275504 * 8,
            "controller_pending_literal_max_bytes": 16 * 257 + 16 * 1072,
            "resolution_fork_additional_credit_bytes": 2 * 24623616,
            "weights_root_fork_credit_proposal_bytes": 12 * 24623616,
            "checkpoint_scope": "idle only; no unresolved eligibility/credit/frame/private RNG saved",
            "binding_stage_tensor_bytes": 0,
            "binding_correction_prior": "held complete V7 decision once per causal teacher prefix",
            "binding_source_prior": "no source or input-scan prior calls",
            "scan_offer_scope": (
                "present field bytes/EOS plus13 flag/status symbols; absent untrained"),
            "transient_source_scope": "new controller experience; zero tape allocations/appends",
        })
        return result


class RecurrentContentSpliceLearner(DendriticSequenceSpliceLearner):
    """Existing staged motor heads and the same recurrent reading/payload actor."""

    def __init__(self, *, mode="learned", seed=733, reader=None):
        if mode not in self.MODES:
            raise ValueError("invalid recurrent action mode")
        if reader is None:
            reader = RecurrentContentMemoryLearner(mode=self.MODES[mode], seed=seed)
        if not isinstance(reader, RecurrentContentMemoryLearner):
            raise ValueError("recurrent actions require the same recurrent reader")
        super().__init__(mode=mode, seed=seed, reader=reader)
        self._resolution_gate = None

    def _require_idle(self):
        super()._require_idle()
        self.reader._require_idle()

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "recurrent-content-splice-learner-v2",
            "reader": "recurrent-content-memory-learner-v2",
            "publication": "all motor proposals then controller/RNG then motor copies",
            "source": "one admitted source or explicit transient observe_source; no load replay",
            "public_reward": "validated actual execution failure=-1; otherwise0; STOP0",
        })
        return metadata

    def load_v7_state_dict(self, state, *, work=None):
        """Preserve the complete old combined actor, extending only declared metadata."""
        self._require_idle()
        if work is None:
            work = {}
        expected = {name for name in self.state_dict()
                    if not name.startswith("reader.controller.")}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("V7 combined adaptation requires every retained state key exactly")
        mapped = self.reader._v7_mapping(state, "reader.", {
            "_extra_state": DendriticSequenceSpliceLearner.get_extra_state(self),
            "reader._extra_state": DendriticSequenceMemoryLearner.get_extra_state(self.reader),
        }, work)
        mapped["_extra_state"] = self.get_extra_state()
        return self.load_state_dict(mapped, strict=True, assign=False, work=work)

    def _validate_execution(self, prediction, observation, state, receipt, work):
        super()._validate_execution(prediction, observation, state, receipt, work)
        if self._resolution_gate is not None:
            execution = receipt["execution"]
            failed = (not execution["accepted"] or execution["returncode"] != 0
                      or execution["timeout"] or not execution["capture_complete"]
                      or not execution["files_complete"] or execution["output_limit_exceeded"])
            reward = -int(failed and prediction.proposal.operation != "STOP")
            self.reader._resolution_context = (prediction.payload, _digest(receipt, work),
                                                reward, self._resolution_gate)

    def teach(self, prediction, observation, state, teacher, execution_receipt,
              learning_rate=0.01, *, learning_enabled=True):
        if type(learning_enabled) is not bool or self._resolution_gate is not None:
            raise ValueError("action resolution needs one explicit Boolean learning gate")
        self._resolution_gate = learning_enabled
        try:
            return super().teach(prediction, observation, state, teacher, execution_receipt,
                                 learning_rate, learning_enabled=learning_enabled)
        finally:
            self._resolution_gate = None
            self.reader._resolution_context = None

    def finish_without_teaching(self, prediction, observation, state, execution_receipt,
                                *, learning_enabled=True):
        if type(learning_enabled) is not bool or self._resolution_gate is not None:
            raise ValueError("action acknowledgement needs one explicit Boolean learning gate")
        self._resolution_gate = learning_enabled
        try:
            return super().finish_without_teaching(
                prediction, observation, state, execution_receipt)
        finally:
            self._resolution_gate = None
            self.reader._resolution_context = None

    def observe_source(self, document, state, *, learning_enabled=True):
        self._require_idle()
        work = {}
        self._check(work)
        return self.reader.observe_source(
            document, state, learning_enabled=learning_enabled, work=work)
