"""Persistent all-owner routing with direct local outcome and byte teaching.

The factual tape and inherited numerical priors remain in the same reader.
Sparse owner views are derived only from actual receptive patterns. Free routing
and teacher-only exact-context recruitment have deliberately separate circuits.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS, ByteGeneration
from humanbrain.factorized_sequence_features import (
    FEATURE_WIDTH,
    MAX_SPARSE_ENTRIES,
    build_candidate_view,
    build_context,
    check_candidate_view,
    validate_candidate_view,
)
from humanbrain.factorized_sequence_router import FactorizedSequenceRouter
from humanbrain.local_memory_learner import _finite
from humanbrain.neuronal_splice_learner import _digest
from humanbrain.persistent_coincidence_memory import PersistentCoincidenceMemory1024
from humanbrain.persistent_coincidence_reader import compete1024
from humanbrain.sequence_assembly_memory_learner import (
    PREFIX_SLOTS,
    SequenceAssemblyFeedback,
    SequenceAssemblyMemoryLearner,
    SequenceAssemblyPrediction,
    SequenceAssemblyStep,
    _add,
    _trace,
)
from humanbrain.sequence_assembly_observation import MAX_REPLACEMENT_BYTES
from humanbrain.sequence_assembly_prefix import RollingPrefixCoincidence1024


@dataclass(frozen=True)
class FactorizedSequenceStep(SequenceAssemblyStep):
    # No exact-Hamming circuit runs in free generation. None means unevaluated.
    exact_context: bool | None
    routing: dict


@dataclass(frozen=True)
class FactorizedSequenceFeedback(SequenceAssemblyFeedback):
    routing_updated: bool = False


def _routing_record(decision):
    return {
        "schema": "factorized-routing-decision-v1",
        "selected_index": decision.selected_index,
        "selected_score": decision.selected_score,
        "selected_baseline": decision.selected_baseline,
        "energy": decision.energy,
        "context_sha256": decision.context_sha256,
        "competition_work": (
            {} if decision.competition is None else dict(decision.competition.work)),
    }


def _view_record(views):
    return {str(index): vars(view) for index, view in enumerate(views)}


class FactorizedSequenceMemoryLearner(SequenceAssemblyMemoryLearner):
    """The same raw-observation actor, with 1024 facts and all event owners."""

    def __init__(self, *, mode="learned", seed=0):
        super().__init__(mode=mode, seed=seed)
        self.memory = PersistentCoincidenceMemory1024()
        self.router = FactorizedSequenceRouter()
        self._candidate_views = ()
        self._view_source_sha256 = self._prototype_hash(
            self.assemblies.prototypes, self.assemblies.used)
        self._view_sha256 = _digest(_view_record(self._candidate_views))

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "factorized-sequence-memory-learner-v1",
            "capacity": 1024,
            "competition_capacity": 1024,
            "memory": "append-only bool[1024,32,18]; all occupied facts eligible",
            "query_limit_scope": "complete raw request<=256 via fixed31-byte chunks",
            "routing": "every occupied event owner; float64 factorized potential competition",
            "routing_feature_width": FEATURE_WIDTH,
            "routing_credit": "selected pre-clamp emitted byte outcome; own presynaptic activity",
            "recruitment": "separate teacher-only full exact Hamming preparation before target",
            "publication": "joint validated router/assembly stages and derived owner views",
            "candidate_views": "canonical sparse derived cache; receptive patterns authoritative",
            "initialization": "retain prior tensors and assemblies; new zero router; derive views",
            "loss": "routed pre-clamp currents NLL at causal teacher prefix; no posterior credit",
            "codec_credit": "assembly output clamp only; inherited U and codec held",
        })
        return metadata

    @staticmethod
    def _prototype_hash(prototypes, used, work=None):
        return _digest({"prototypes": prototypes[:int(used.item())], "used": used}, work)

    def _check_conditioned_parameters(self, work=None):
        super()._check_conditioned_parameters(work)
        if hasattr(self, "router"):
            self.router.check_parameters(work)
        if hasattr(self, "_candidate_views"):
            for view in self._candidate_views:
                check_candidate_view(view, work)
            if (len(self._candidate_views) != int(self.assemblies.used.item())
                    or self._prototype_hash(self.assemblies.prototypes, self.assemblies.used,
                                            work) != self._view_source_sha256
                    or _digest(_view_record(self._candidate_views), work) != self._view_sha256):
                raise ValueError("derived candidate views or their receptive source changed")
            if work is not None:
                _add(work, "factorized_view_custody_checks", 1)

    def prepare_state_views(self, state, prefix="", work=None):
        """Validate new child state and derive every view before any owner copy.

        The combined actor calls this before recursive loading too: PyTorch does
        not invoke a child's custom load_state_dict during a parent load.
        """
        self.router.validate_state(state, prefix + "router.", work)
        self.assemblies.validate_state(state, prefix + "assemblies.", work)
        prototypes, used = (state[prefix + "assemblies.prototypes"],
                            state[prefix + "assemblies.used"])
        # The builder validates canonical raw decoding and its resulting sparse
        # columns/rates. There is no supplied view to compare, so rebuilding the
        # same canonical result a second time would add no load validation.
        views = tuple(build_candidate_view(prototypes[index], work=work)
                      for index in range(int(used.item())))
        return (views, self._prototype_hash(prototypes, used, work),
                _digest(_view_record(views), work))

    def install_validated_views(self, prepared):
        """Pointer-only cache installation after the owner's prevalidation."""
        self._candidate_views, self._view_source_sha256, self._view_sha256 = prepared

    def load_state_dict(self, state_dict, strict=True, assign=False, *, work=None):
        self._require_idle()
        self._check_conditioned_parameters(work)
        if not strict or assign or not isinstance(state_dict, Mapping):
            raise ValueError("factorized loading requires strict in-place state")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("factorized checkpoint fields mismatch")
        for name, prototype in expected.items():
            value = state_dict[name]
            if isinstance(prototype, Tensor):
                if (not isinstance(value, Tensor) or value.shape != prototype.shape
                        or value.dtype != prototype.dtype or value.device != prototype.device
                        or value.requires_grad or value.grad_fn is not None
                        or value.grad is not None):
                    raise ValueError("factorized checkpoint tensor contract mismatch")
                _finite(value, "factorized checkpoint tensor")
            elif value != prototype:
                raise ValueError("factorized checkpoint metadata mismatch")
        for name in self.FIXED_STATE_NAMES:
            if not torch.equal(state_dict[name], expected[name]):
                raise ValueError("checkpoint changes fixed reader wiring")
        self.context.validate_fixed_state(state_dict, "context.")
        prepared = self.prepare_state_views(state_dict, work=work)
        # All tensor, metadata and derived-view checks precede recursive copies.
        result = nn.Module.load_state_dict(self, state_dict, strict=True, assign=False)
        self.install_validated_views(prepared)
        return result

    def _context_forward(self, prepared, prefix, cache, work):
        # Same held-address/raw-context construction as the preceding reader;
        # only the explicit competition capacity changes from512 to1024.
        q, xq, contexts, legacy, sensor = prepared
        xp = cache.features()
        scores = (xq * self.address_weight).sum((1, 2)) + (xp * self.prefix_weight).sum((1, 2))
        _finite(scores, "factorized memory address scores")
        competition = compete1024(scores)
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

    def _route(self, inputs, fallback, views, outgoing, work, *, stage=None):
        context = build_context(inputs, work=work)
        decision = self.router.evaluate(context, views, stage=stage, work=work)
        logits = (fallback.clone() if decision.selected_index is None
                  else outgoing[decision.selected_index].clone())
        _finite(logits, "factorized selected outgoing currents")
        _add(work, "factorized_selected_output_reads",
             int(decision.selected_index is not None) * 257)
        _add(work, "factorized_fallback_output_reads", int(decision.selected_index is None) * 257)
        return decision, logits

    @torch.no_grad()
    def predict_with_receipt(self, observation, state, max_bytes):
        self._require_idle()
        if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_REPLACEMENT_BYTES:
            raise ValueError("factorized response max_bytes must be0..128")
        work = {}
        prepared = self._prepare_observation(observation, state, work)
        model_hash = _digest(self.state_dict(), work)
        memory_hash = _digest(vars(state), work)
        cache = RollingPrefixCoincidence1024(state.keys[:state.used], work=work)
        assemblies, router = self.assemblies, self.router
        views, outgoing = self._candidate_views, assemblies.outgoing
        prototypes, used, theta = assemblies.prototypes, assemblies.used, router.theta
        view_tensors = tuple((view.indices, view.values, view.flags) for view in views)
        output, steps = bytearray(), []
        for position in range(max_bytes + 1):
            prefix = bytes(output)
            inputs, fallback, address, scores = self._context_forward(prepared, prefix, cache, work)
            decision, logits = self._route(inputs, fallback, views, outgoing, work)
            token = int(logits.argmax().item())
            nll = float(-F.log_softmax(logits, dim=0)[token].item())
            _add(work, "factorized_output_argmaxes", 1)
            _add(work, "factorized_output_log_softmaxes", 1)
            steps.append(FactorizedSequenceStep(
                position, prefix, token, address, scores, decision.selected_index,
                decision.scores, None, logits, nll, _routing_record(decision)))
            _add(work, "output_decisions", 1)
            if token == BYTE_EOS or position == max_bytes:
                break
            output.append(token)
            cache.advance(token)
        if (self.assemblies is not assemblies or self.router is not router
                or self._candidate_views is not views or assemblies.outgoing is not outgoing
                or assemblies.prototypes is not prototypes or assemblies.used is not used
                or router.theta is not theta
                or any(view.indices is not held[0] or view.values is not held[1]
                       or view.flags is not held[2]
                       for view, held in zip(views, view_tensors, strict=True))):
            raise RuntimeError("free factorized response rebound owned routing storage")
        self._check_conditioned_parameters(work)
        if (_digest(self.state_dict(), work) != model_hash
                or _digest(vars(state), work) != memory_hash):
            raise RuntimeError("free factorized response mutated model or memory")
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
            raise ValueError("factorized teacher needs bounded bytes, eta.01 and Boolean gate")
        work = {}
        self.validate_pending(prediction, state, observation=observation, work=work)
        if len(teacher) > prediction.max_bytes:
            raise ValueError("teacher exceeds the actual response capacity")
        prepared = self._prepare_observation(observation, state, work)
        cache = RollingPrefixCoincidence1024(state.keys[:state.used], work=work)
        assembly_stage = self.assemblies.begin_stage(work)
        router_stage = self.router.begin_stage(work)
        views = list(self._candidate_views)
        enabled = learning_enabled and self.mode != "slow_frozen"
        records, loss = [], 0.0
        for position, target in enumerate((*teacher, BYTE_EOS)):
            prefix = teacher[:position]
            inputs, fallback, address, scores = self._context_forward(prepared, prefix, cache, work)
            decision, logits = self._route(
                inputs, fallback, views, assembly_stage.outgoing, work, stage=router_stage)
            emitted = int(logits.argmax().item())
            _add(work, "factorized_output_argmaxes", 1)
            # Separate old exact-existence/recruitment circuit, before target.
            exact = self.assemblies.prepare_context(
                inputs, fallback, stage=assembly_stage, work=work)
            proposed_view = (build_candidate_view(exact.inputs, work=work)
                             if exact.recruited else None)
            routing_update = self.router.prepare_outcome(
                decision, emitted, target, stage=router_stage, learning_enabled=enabled, work=work)
            update = self.assemblies.prepare_teach(
                exact, target, stage=assembly_stage, learning_enabled=enabled, work=work)
            if update.staged and update.recruited:
                views.append(proposed_view)
            nll = float(-F.log_softmax(logits, dim=0)[target].item())
            _add(work, "factorized_output_log_softmaxes", 1)
            loss += nll
            records.append({
                "position": position, "prefix_hex": prefix.hex(), "target": target,
                "memory_index": address, "address_scores": scores,
                "input_sha256": _digest({"input": inputs}, work),
                "free_assembly_index": decision.selected_index,
                "assembly_scores": decision.scores, "free_currents": logits,
                "emitted_byte": emitted, "routing": _routing_record(decision),
                "routing_outcome": {
                    "desired": routing_update.desired, "error": routing_update.error,
                    "score_before": routing_update.score_before,
                    "score_after": routing_update.score_after, "energy": routing_update.energy,
                    "staged": routing_update.staged, "skipped": routing_update.skipped,
                    "local_values": routing_update.indices.numel(),
                },
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
        self.router.validate_stage(router_stage, work)
        if len(views) != int(assembly_stage.used.item()):
            raise ValueError("staged view count differs from staged event owners")
        for index, view in enumerate(views):
            validate_candidate_view(view, assembly_stage.prototypes[index], work=work)
        prepared_views = (
            tuple(views),
            self._prototype_hash(assembly_stage.prototypes, assembly_stage.used, work),
            _digest(_view_record(views), work))
        self.validate_pending(prediction, state, observation=observation, work=work)
        changed = enabled and assembly_stage.revision > 0
        routing_changed = enabled and router_stage.revision > 0
        feedback = FactorizedSequenceFeedback(
            state, prediction.attempt_id, len(teacher) + 1, loss, loss / (len(teacher) + 1),
            changed, tuple(records), work, routing_updated=routing_changed)
        # Every allocation, finite check, seal and candidate reconstruction is
        # complete. Both neuronal stages and the preallocated view tuple publish
        # as one synchronous copy/pointer-only section.
        if changed:
            self.assemblies.commit_validated(assembly_stage)
        if routing_changed:
            self.router.commit_validated(router_stage)
        if changed:
            self.install_validated_views(prepared_views)
        self._sequence_pending = None
        if changed:
            _add(work, "assembly_publications", 1)
            _add(work, "assembly_publication_copy_bytes", self.assemblies.tensor_bytes)
        if routing_changed:
            _add(work, "router_publications", 1)
            _add(work, "router_publication_copy_bytes", self.router.tensor_bytes)
        return feedback

    def accounting(self):
        self._check_conditioned_parameters()
        return {
            "schema": "factorized-sequence-accounting-v1",
            "model_tensor_bytes": sum(
                value.numel() * value.element_size() for value in self.state_dict().values()
                if isinstance(value, Tensor)),
            "factual_memory_capacity": 1024,
            "factual_tape_bytes": 1_650_696,
            "maximum_address_comparator_neurons": 1024 * 1023 // 2,
            "maximum_address_winner_neurons": 1024,
            "maximum_address_feature_bank_float32_bytes": 1024 * 1024 * 4,
            "maximum_raw_context_bytes": 1024 * 64 * 18,
            "maximum_legacy_current_bytes": 1024 * 257 * 8,
            "router_feature_width": FEATURE_WIDTH,
            "router_tensor_bytes": self.router.tensor_bytes,
            "router_stage_bytes": self.router.tensor_bytes,
            "assembly_capacity": 512,
            "assembly_occupied": int(self.assemblies.used.item()),
            "assembly_input_width": self.assemblies.input_width,
            "assembly_bank_bytes": self.assemblies.tensor_bytes,
            "assembly_stage_bytes": self.assemblies.tensor_bytes,
            "candidate_view_tensor_bytes": sum(
                view.indices.numel() * 4 + view.values.numel() * 4 + view.flags.numel()
                for view in self._candidate_views),
            "maximum_candidate_sparse_entries_per_owner": MAX_SPARSE_ENTRIES,
            "routing_owners": len(self._candidate_views),
            "maximum_response_raw_bytes": MAX_REPLACEMENT_BYTES,
            "maximum_query_chunks": 9,
            "derived_view_storage": "runtime-only canonical sparse tuples; not checkpoint tensors",
            "inherited_synaptic_learning": "none; B/C/U and codec held",
        }
