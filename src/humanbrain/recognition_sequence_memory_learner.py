"""Prepared-owner recognition and local output teaching in one persistent reader.

The inherited shared router remains a numerical prior. Recognition gains are
private to input-defined event owners; current target bytes train outputs only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS, ByteGeneration
from humanbrain.factorized_sequence_features import (
    build_candidate_view,
    build_context,
)
from humanbrain.factorized_sequence_memory_learner import (
    FactorizedSequenceMemoryLearner,
    FactorizedSequenceStep,
    _view_record,
)
from humanbrain.input_defined_recognition import (
    InputDefinedRecognition,
    _self_from_validated_view,
    check_self_activity,
    validate_self_activity,
)
from humanbrain.local_memory_learner import _finite
from humanbrain.neuronal_splice_learner import _digest
from humanbrain.persistent_coincidence_memory import PersistentCoincidenceMemory2048
from humanbrain.persistent_coincidence_reader import compete2048
from humanbrain.sequence_assembly_cells import SequenceAssemblyCells1024
from humanbrain.sequence_assembly_memory_learner import (
    INPUT_WIDTH,
    PREFIX_SLOTS,
    SequenceAssemblyFeedback,
    SequenceAssemblyPrediction,
    _add,
    _trace,
)
from humanbrain.sequence_assembly_observation import MAX_REPLACEMENT_BYTES
from humanbrain.sequence_assembly_prefix import RollingPrefixCoincidence2048


@dataclass(frozen=True)
class RecognitionSequenceFeedback(SequenceAssemblyFeedback):
    recognition_updated: bool = False
    routing_updated: bool = False


def _recognition_record(decision):
    return {
        "schema": "input-defined-recognition-decision-v1",
        "selected_index": decision.selected_index,
        "selected_score": decision.selected_score,
        "selected_baseline": decision.selected_baseline,
        "selected_kappa": decision.selected_kappa,
        "energy": decision.energy,
        "context_sha256": decision.context_sha256,
        "prior_scores": decision.prior_scores,
        "kappas": decision.kappas,
        "competition_work": (
            {} if decision.competition is None else dict(decision.competition.work)),
    }


class RecognitionSequenceMemoryLearner(FactorizedSequenceMemoryLearner):
    """All 2048 fact addresses and all 1024 event owners remain eligible."""

    def __init__(self, *, mode="learned", seed=0):
        super().__init__(mode=mode, seed=seed)
        self.memory = PersistentCoincidenceMemory2048()
        self.assemblies = SequenceAssemblyCells1024(INPUT_WIDTH)
        self.recognition = InputDefinedRecognition()
        self._view_source_sha256 = self._prototype_hash(
            self.assemblies.prototypes, self.assemblies.used)
        self._self_wirings = ()
        self._self_wiring_sha256 = _digest(_view_record(self._self_wirings))

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "recognition-sequence-memory-learner-v1",
            "capacity": 2048,
            "competition_capacity": 2048,
            "memory": "append-only bool[2048,32,18]; all occupied facts eligible",
            "routing": "all owners; frozen prior plus private rank-one recognition current",
            "routing_credit": "none; inherited shared theta is held",
            "recognition_credit": "input-prepared owner activation one before current byte",
            "recognition_gains": "float64[1024]; zero initial gains, no old-lesson replay",
            "self_wiring": "canonical self activity from authoritative receptive pattern",
            "publication": "joint validated recognition/assembly stages and derived wiring",
            "initialization": "retain all prior tensors/events; zero new recognition gains",
            "teacher_phase": "input preparation then recognition association then current byte",
        })
        return metadata

    def _check_conditioned_parameters(self, work=None):
        super()._check_conditioned_parameters(work)
        if hasattr(self, "recognition"):
            self.recognition.check_parameters(work)
            used = int(self.assemblies.used.item())
            if bool(self.recognition.gains[used:].ne(0).any()):
                raise ValueError("unoccupied recognition gains must be zero")
            if len(self._self_wirings) != used:
                raise ValueError("self wiring must cover every occupied event")
            for wiring, view in zip(self._self_wirings, self._candidate_views, strict=True):
                check_self_activity(wiring, view, work)
            if _digest(_view_record(self._self_wirings), work) != self._self_wiring_sha256:
                raise ValueError("derived self wiring changed")
            if work is not None:
                _add(work, "recognition_unused_gain_values", 1024 - used)
                _add(work, "recognition_wiring_custody_checks", 1)

    def prepare_state_views(self, state, prefix="", work=None):
        base = super().prepare_state_views(state, prefix, work)
        self.recognition.validate_state(state, prefix + "recognition.", work)
        used = int(state[prefix + "assemblies.used"].item())
        if bool(state[prefix + "recognition.gains"][used:].ne(0).any()):
            raise ValueError("checkpoint has gains outside occupied event owners")
        # The base preparation just constructed each canonical view from its
        # authoritative pattern. Derive self wiring without rebuilding the view.
        wirings = tuple(_self_from_validated_view(view, work) for view in base[0])
        return base, wirings, _digest(_view_record(wirings), work)

    def install_validated_views(self, prepared):
        base, self._self_wirings, self._self_wiring_sha256 = prepared
        super().install_validated_views(base)

    def _context_forward(self, prepared, prefix, cache, work):
        """Same held address/readout arithmetic with explicit2048 competition."""
        q, xq, contexts, legacy, sensor = prepared
        xp = cache.features()
        scores = (xq * self.address_weight).sum((1, 2)) + (xp * self.prefix_weight).sum((1, 2))
        _finite(scores, "recognition memory address scores")
        competition = compete2048(scores)
        for key, value in competition.work.items():
            _add(work, key, value)
        selected = competition.winner_index
        selected_context = contexts[selected:selected + 1]
        position = len(prefix)
        p = self._rails([BYTE_BOS, *prefix], PREFIX_SLOTS, work)
        if q is not None and position <= 16:
            _, logits, _ = self._conditional(
                q, p[:17], selected_context, legacy[selected:selected + 1],
                position, self.output_weight, work)
            fallback = logits[0]
            _add(work, "sequence_conditioned_fallbacks", 1)
        else:
            fallback = legacy[selected].clone()
            _add(work, "sequence_codec_fallbacks", 1)
        position_cells = torch.zeros(PREFIX_SLOTS, dtype=torch.bool)
        position_cells[position] = True
        inputs = torch.cat((sensor, p.flatten(), selected_context.flatten(), position_cells))
        _add(work, "sequence_input_values", inputs.numel())
        _add(work, "output_decisions_contexts", 1)
        _add(work, "output_address_steps", 1)
        _add(work, "output_address_synapse_products", xq.shape[0] * 2048)
        return inputs, fallback, selected, scores

    def _recognize(self, inputs, fallback, views, wirings, outgoing, work, *,
                   stage=None, owned_free=None):
        context = build_context(inputs, work=work)
        if owned_free is None:
            decision = self.recognition.evaluate(
                context, views, wirings, self.router.theta, stage=stage, work=work)
        else:
            if stage is not None:
                raise ValueError("owned free recognition cannot use a teacher stage")
            decision = self.recognition._evaluate_owned_free(
                context, views, wirings, owned_free, work=work)
        logits = (fallback.clone() if decision.selected_index is None
                  else outgoing[decision.selected_index].clone())
        _finite(logits, "recognition selected outgoing currents")
        _add(work, "recognition_selected_output_reads",
             257 * int(decision.selected_index is not None))
        _add(work, "recognition_fallback_output_reads",
             257 * int(decision.selected_index is None))
        return context, decision, logits

    @torch.no_grad()
    def predict_with_receipt(self, observation, state, max_bytes):
        self._require_idle()
        if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_REPLACEMENT_BYTES:
            raise ValueError("recognition response max_bytes must be0..128")
        work = {}
        prepared = self._prepare_observation(observation, state, work)
        model_hash = _digest(self.state_dict(), work)
        memory_hash = _digest(vars(state), work)
        cache = RollingPrefixCoincidence2048(state.keys[:state.used], work=work)
        assemblies, router, recognition = self.assemblies, self.router, self.recognition
        views, wirings = self._candidate_views, self._self_wirings
        prototypes, outgoing, used = assemblies.prototypes, assemblies.outgoing, assemblies.used
        theta, gains = router.theta, recognition.gains
        owned = recognition._begin_validated_free_response(theta, work)
        held_tensors = tuple(tensor for view in views
                             for tensor in (view.indices, view.values, view.flags))
        held_wirings = tuple(tensor for wiring in wirings
                             for tensor in (wiring.indices, wiring.values))
        versions = tuple(tensor._version for tensor in (*held_tensors, *held_wirings))
        output, steps = bytearray(), []
        for position in range(max_bytes + 1):
            prefix = bytes(output)
            inputs, fallback, address, scores = self._context_forward(prepared, prefix, cache, work)
            _, decision, logits = self._recognize(
                inputs, fallback, views, wirings, outgoing, work, owned_free=owned)
            token = int(logits.argmax().item())
            nll = float(-F.log_softmax(logits, dim=0)[token].item())
            _add(work, "recognition_output_argmaxes", 1)
            _add(work, "recognition_output_log_softmaxes", 1)
            steps.append(FactorizedSequenceStep(
                position, prefix, token, address, scores, decision.selected_index,
                decision.scores, None, logits, nll, _recognition_record(decision)))
            _add(work, "output_decisions", 1)
            if token == BYTE_EOS or position == max_bytes:
                break
            output.append(token)
            cache.advance(token)
        current_tensors = tuple(tensor for view in views
                                for tensor in (view.indices, view.values, view.flags))
        current_wirings = tuple(tensor for wiring in wirings
                                for tensor in (wiring.indices, wiring.values))
        if (self.assemblies is not assemblies or self.router is not router
                or self.recognition is not recognition or self._candidate_views is not views
                or self._self_wirings is not wirings or assemblies.prototypes is not prototypes
                or assemblies.outgoing is not outgoing or assemblies.used is not used
                or router.theta is not theta or recognition.gains is not gains
                or any(a is not b for a, b in zip(
                    (*held_tensors, *held_wirings), (*current_tensors, *current_wirings),
                    strict=True))
                or versions != tuple(tensor._version for tensor in
                                     (*current_tensors, *current_wirings))):
            raise RuntimeError("free recognition response rebound or mutated owned storage")
        recognition._check_owned_free_response(owned)
        self._check_conditioned_parameters(work)
        if (_digest(self.state_dict(), work) != model_hash
                or _digest(vars(state), work) != memory_hash):
            raise RuntimeError("free recognition response mutated model or memory")
        prediction = SequenceAssemblyPrediction(
            ByteGeneration(bytes(output), "eos" if token == BYTE_EOS else "limit"),
            tuple(steps), work, observation, max_bytes,
            f"{id(self)}:{self._active_experience + 1}", model_hash, memory_hash, "")
        trace_hash = _digest(_trace(prediction), work)
        object.__setattr__(prediction, "trace_sha256", trace_hash)
        self._sequence_pending = (prediction, observation.sha256, model_hash, memory_hash,
                                  trace_hash, _digest(work))
        self._active_experience += 1
        return prediction

    @torch.no_grad()
    def teach(self, observation, teacher, state, learning_rate=0.01, *,
              prediction, learning_enabled=True):
        if (not isinstance(teacher, bytes) or len(teacher) > MAX_REPLACEMENT_BYTES
                or type(learning_rate) not in (float, int) or learning_rate != 0.01
                or type(learning_enabled) is not bool):
            raise ValueError("recognition teacher needs bounded bytes, eta.01 and Boolean gate")
        work = {}
        self.validate_pending(prediction, state, observation=observation, work=work)
        if len(teacher) > prediction.max_bytes:
            raise ValueError("teacher exceeds the actual response capacity")
        prepared = self._prepare_observation(observation, state, work)
        cache = RollingPrefixCoincidence2048(state.keys[:state.used], work=work)
        assembly_stage = self.assemblies.begin_stage(work)
        recognition_stage = self.recognition.begin_stage(work)
        views, wirings = list(self._candidate_views), list(self._self_wirings)
        enabled = learning_enabled and self.mode != "slow_frozen"
        records, loss = [], 0.0
        for position in range(len(teacher) + 1):
            prefix = teacher[:position]
            inputs, fallback, address, scores = self._context_forward(prepared, prefix, cache, work)
            context, decision, logits = self._recognize(
                inputs, fallback, views, wirings, assembly_stage.outgoing, work,
                stage=recognition_stage)
            emitted = int(logits.argmax().item())
            _add(work, "recognition_output_argmaxes", 1)
            exact = self.assemblies.prepare_context(
                inputs, fallback, stage=assembly_stage, work=work)
            proposed_view = (build_candidate_view(exact.inputs, work=work)
                             if exact.recruited else None)
            proposed_wiring = (_self_from_validated_view(proposed_view, work)
                               if exact.recruited else None)
            recognition_update = None
            if exact.selected_index is None:
                _add(work, "recognition_exhausted_contexts", 1)
            else:
                view = proposed_view if exact.recruited else views[exact.selected_index]
                wiring = proposed_wiring if exact.recruited else wirings[exact.selected_index]
                recognition_context = self.recognition.prepare_context(
                    context, view, wiring, exact.selected_index, self.router.theta,
                    stage=recognition_stage, work=work)
                recognition_update = self.recognition.prepare_recognition(
                    recognition_context, stage=recognition_stage,
                    learning_enabled=enabled, work=work)
            # No current target has entered either owner preparation or recognition.
            target = teacher[position] if position < len(teacher) else BYTE_EOS
            update = self.assemblies.prepare_teach(
                exact, target, stage=assembly_stage, learning_enabled=enabled, work=work)
            if update.staged and update.recruited:
                views.append(proposed_view)
                wirings.append(proposed_wiring)
            nll = float(-F.log_softmax(logits, dim=0)[target].item())
            _add(work, "recognition_output_log_softmaxes", 1)
            loss += nll
            records.append({
                "position": position, "prefix_hex": prefix.hex(), "target": target,
                "memory_index": address, "address_scores": scores,
                "input_sha256": _digest({"input": inputs}, work),
                "free_assembly_index": decision.selected_index,
                "assembly_scores": decision.scores, "free_currents": logits,
                "emitted_byte": emitted, "routing": _recognition_record(decision),
                "recognition_update": None if recognition_update is None else vars(
                    recognition_update).copy(),
                "recruitment_scores": exact.evaluation.agreement_counts,
                "recruitment_winner": exact.evaluation.winner_index,
                "selected_index": update.selected_index, "exact": exact.evaluation.exact,
                "recruited": update.recruited, "exhausted": update.exhausted,
                "staged": update.staged, "currents_before": update.currents_before,
                "currents_after": update.currents_after, "nll": nll,
            })
            if target != BYTE_EOS:
                cache.advance(target)
        self.assemblies.validate_stage(assembly_stage, work)
        self.recognition.validate_stage(recognition_stage, work)
        if len(views) != len(wirings) or len(views) != int(assembly_stage.used.item()):
            raise ValueError("staged recognition wiring differs from staged owners")
        if bool(recognition_stage.gains[len(views):].ne(0).any()):
            raise ValueError("staged gains escaped occupied event owners")
        for index, (view, wiring) in enumerate(zip(views, wirings, strict=True)):
            validate_self_activity(wiring, view, assembly_stage.prototypes[index], work)
        base_views = (tuple(views),
                      self._prototype_hash(assembly_stage.prototypes, assembly_stage.used, work),
                      _digest(_view_record(views), work))
        derived = base_views, tuple(wirings), _digest(_view_record(wirings), work)
        self.validate_pending(prediction, state, observation=observation, work=work)
        changed = enabled and assembly_stage.revision > 0
        recognition_changed = enabled and recognition_stage.revision > 0
        feedback = RecognitionSequenceFeedback(
            state, prediction.attempt_id, len(teacher) + 1, loss, loss / (len(teacher) + 1),
            changed, tuple(records), work, recognition_updated=recognition_changed)
        # Both stage proposals and all derived wiring are validated before publication.
        if changed:
            self.assemblies.commit_validated(assembly_stage)
        if recognition_changed:
            self.recognition.commit_validated(recognition_stage)
        if changed:
            self.install_validated_views(derived)
        self._sequence_pending = None
        if changed:
            _add(work, "assembly_publications", 1)
            _add(work, "assembly_publication_copy_bytes", self.assemblies.tensor_bytes)
        if recognition_changed:
            _add(work, "recognition_publications", 1)
            _add(work, "recognition_publication_copy_bytes", self.recognition.tensor_bytes)
        return feedback

    def accounting(self):
        result = super().accounting()
        result.update({
            "schema": "recognition-sequence-accounting-v1",
            "factual_memory_capacity": 2048,
            "factual_tape_bytes": 3_301_384,
            "maximum_address_comparator_neurons": 2048 * 2047 // 2,
            "maximum_address_winner_neurons": 2048,
            "maximum_address_feature_bank_float32_bytes": 2048 * 1024 * 4,
            "maximum_raw_context_bytes": 2048 * 64 * 18,
            "maximum_legacy_current_bytes": 2048 * 257 * 8,
            "assembly_capacity": 1024,
            "router_stage_bytes": 0,
            "recognition_gain_bytes": self.recognition.tensor_bytes,
            "recognition_stage_bytes": self.recognition.tensor_bytes,
            "self_wiring_tensor_bytes": sum(
                wiring.indices.numel() * 4 + wiring.values.numel() * 8
                for wiring in self._self_wirings),
            "recognition_wiring_storage": (
                "derived canonical runtime tuple; no checkpoint duplicate"),
            "inherited_synaptic_learning": "none; shared theta, B/C/U and codec held",
        })
        return result
