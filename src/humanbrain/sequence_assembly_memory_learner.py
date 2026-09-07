"""Persistent raw-observation reader with input-recruited local sequence cells.

The inherited numerical reader remains in this model. Its address and output
synapses are held; only the assembly bank receives new sequence teaching.
Actual responses and causal teacher-prefix forwards have separate receipts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from humanbrain.active_path_memory_learner import ActivePathMemoryLearner
from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS, ByteGeneration
from humanbrain.local_memory_learner import _finite
from humanbrain.neuronal_byte_coincidence import encode_symbol_rails
from humanbrain.neuronal_splice_learner import _digest
from humanbrain.persistent_coincidence_reader import compete512
from humanbrain.sequence_assembly_cells import SequenceAssemblyCells
from humanbrain.sequence_assembly_observation import (
    MAX_REPLACEMENT_BYTES,
    SENSOR_FLAG_NAMES,
    SENSOR_SYMBOLS,
    RawObservation,
)
from humanbrain.sequence_assembly_prefix import RollingPrefixCoincidence

PREFIX_SLOTS = MAX_REPLACEMENT_BYTES + 1
INPUT_WIDTH = SENSOR_SYMBOLS * 18 + len(SENSOR_FLAG_NAMES) + PREFIX_SLOTS * 18 + 64 * 18 + PREFIX_SLOTS


def _add(work, key, value):
    work[key] = work.get(key, 0) + value


@dataclass(frozen=True)
class SequenceAssemblyStep:
    output_step: int
    prefix: bytes
    observed_token: int
    selected_index: int
    address_scores: Tensor
    assembly_index: int | None
    assembly_scores: Tensor
    exact_context: bool
    logits: Tensor
    nll: float


@dataclass(frozen=True)
class SequenceAssemblyPrediction:
    generation: ByteGeneration
    steps: tuple[SequenceAssemblyStep, ...]
    work: dict
    observation: RawObservation
    max_bytes: int
    attempt_id: str
    model_sha256: str
    memory_sha256: str
    trace_sha256: str


@dataclass(frozen=True)
class SequenceAssemblyFeedback:
    state: object
    attempt_id: str
    teacher_count: int
    total_loss: float
    mean_loss: float
    output_updated: bool
    teacher_steps: tuple[dict, ...]
    work: dict
    address_updated: bool = False
    legacy_codec_updated: bool = False


def _trace(prediction):
    return {
        "observation": prediction.observation.to_record(),
        "max_bytes": prediction.max_bytes,
        "attempt_id": prediction.attempt_id,
        "model_sha256": prediction.model_sha256,
        "memory_sha256": prediction.memory_sha256,
        "generation": {"data": prediction.generation.data.hex(),
                       "stop_reason": prediction.generation.stop_reason},
        "steps": {str(i): {**vars(step), "prefix": step.prefix.hex()}
                  for i, step in enumerate(prediction.steps)},
    }


class SequenceAssemblyMemoryLearner(ActivePathMemoryLearner):
    """One pending response, one persistent memory, and one local assembly bank."""

    def __init__(self, *, mode="learned", seed=0):
        super().__init__(mode=mode, seed=seed)
        self.assemblies = SequenceAssemblyCells(INPUT_WIDTH)
        self._sequence_pending = None

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "sequence-assembly-memory-learner-v1",
            "query": "whole request<=256; fixed31-byte chunks each followed by EOS",
            "address": "mean separately normalized chunk X; held B/C; rolling32 prefix X",
            "address_credit": "none; inherited B/C frozen",
            "address_synapse_credit": "none; inherited B/C frozen",
            "free_belief": "selected memory and assembly; scores retained, no posterior recurrence",
            "loss": "selected assembly free-current NLL before current target; no mixture or post-update probe",
            "codec": "inherited numeric codec/U retained as fallback; new sparse assembly outgoing bank",
            "codec_credit": "inherited U and codec frozen; only assembly local outgoing errors",
            "codec_updated_semantics": "legacy codec and U unchanged; output_updated means assembly publication",
            "output_feature_geometry_scope": "inherited292/energy32/context256 fields describe U fallback only",
            "teacher_phase": "one target-free context preparation per offered prefix then target",
            "publication": "one private assembly stage; disabled never advances",
            "fallback": "old conditional U only request<=31 and position<=16; otherwise old codec",
            "assembly_inputs": "whole framed observation, BOS/full prefix, selected64-row context, position",
            "assembly_input_width": INPUT_WIDTH,
            "teacher_capacity": "raw target<=pending max_bytes<=128 plus EOS",
            "initialization": "retain all prior numeric values; new empty assembly bank",
        })
        return metadata

    def _check_conditioned_parameters(self, work=None):
        super()._check_conditioned_parameters()
        # Base constructors validate before this subclass creates its new bank.
        if hasattr(self, "assemblies"):
            self.assemblies.check_parameters(work)

    def _require_idle(self):
        if getattr(self, "_sequence_pending", None) is not None:
            raise ValueError("resolve the pending sequence-assembly response first")
        super()._require_idle()

    @property
    def pending_prediction(self):
        return None if self._sequence_pending is None else self._sequence_pending[0]

    def is_pending(self, prediction):
        return self._sequence_pending is not None and self._sequence_pending[0] is prediction

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._require_idle()
        self.assemblies.validate_state(state_dict, "assemblies.")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _prepare_observation(self, observation, state, work):
        if not isinstance(observation, RawObservation):
            raise TypeError("the reader requires a complete RawObservation")
        self._check_conditioned_parameters(work)
        self._check_state(state)
        # Mechanical fixed chunks cover every byte. Each chunk has the same
        # inherited lag geometry; their normalized activities are averaged in
        # source order. No chunk selection depends on a target or memory score.
        request = observation.request
        chunks = [request[i:i + 31] for i in range(0, len(request), 31)] or [b""]
        xq = None
        old_q = None
        for chunk in chunks:
            q = self._rails([*chunk, BYTE_EOS], 32, work)
            lags = torch.zeros_like(q)
            lags[:len(chunk) + 1].copy_(q[:len(chunk) + 1].flip(0))
            features = self._feature_bank(lags, state, work)
            xq = features if xq is None else xq + features
            if len(chunks) == 1:
                old_q = q
        if len(chunks) > 1:
            xq = xq / len(chunks)
        _add(work, "sequence_query_chunks", len(chunks))
        _add(work, "sequence_query_mean_additions", (len(chunks) - 1) * xq.numel())
        _add(work, "sequence_query_mean_divisions", xq.numel() if len(chunks) > 1 else 0)
        _add(work, "query_raw_bytes", len(request))
        _add(work, "query_eos_symbols", len(chunks))
        sensory = state.sensory[:state.used]
        active = sensory.sum(1)
        tokens = sensory.argmax(1)
        rails = encode_symbol_rails(tokens, work=work)
        rails &= active.to(torch.bool)[:, None]
        context = torch.empty(state.used, 64, 18, dtype=torch.bool)
        context[:, :32].copy_(state.keys[:state.used])
        cursors = torch.arange(state.used, dtype=torch.int64)
        for position in range(32):
            context[:, 32 + position].copy_(rails.index_select(0, cursors))
            if position < 31:
                cursors = self.memory._advance_checked(state, cursors)
                _add(work, "output_context_cursor_advances", state.used)
        legacy = self.codec.weight.T.index_select(0, tokens) * active[:, None] + self.codec.bias
        _finite(legacy, "sequence legacy currents")
        frame = observation.sensor_frame()
        symbols = torch.tensor(frame.symbols, dtype=torch.int64)
        present = symbols >= 0
        sensor = encode_symbol_rails(symbols.clamp_min(0), work=work)
        sensor &= present[:, None]
        sensor = torch.cat((sensor.flatten(), torch.tensor(frame.flags, dtype=torch.bool)))
        _add(work, "sequence_sensor_symbols", len(frame.symbols))
        _add(work, "sequence_sensor_flags", len(frame.flags))
        _add(work, "output_context_bank_bytes", context.numel())
        _add(work, "output_legacy_current_potentials", legacy.numel())
        return old_q, xq, context, legacy, sensor

    def _context_forward(self, prepared, prefix, cache, work):
        q, xq, contexts, legacy, sensor = prepared
        xp = cache.features()
        scores = (xq * self.address_weight).sum((1, 2)) + (xp * self.prefix_weight).sum((1, 2))
        _finite(scores, "sequence memory address scores")
        competition = compete512(scores)
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
        _add(work, "output_address_steps", 1)
        _add(work, "output_address_synapse_products", xq.shape[0] * 2048)
        return inputs, fallback, selected, scores

    @torch.no_grad()
    def predict_with_receipt(self, observation, state, max_bytes):
        self._require_idle()
        if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_REPLACEMENT_BYTES:
            raise ValueError("sequence response max_bytes must be0..128")
        work = {}
        prepared = self._prepare_observation(observation, state, work)
        model_hash = _digest(self.state_dict(), work)
        memory_hash = _digest(vars(state), work)
        cache = RollingPrefixCoincidence(state.keys[:state.used], work=work)
        output, steps = bytearray(), []
        for position in range(max_bytes + 1):
            prefix = bytes(output)
            inputs, fallback, address, scores = self._context_forward(prepared, prefix, cache, work)
            evaluation = self.assemblies.evaluate(inputs, fallback, work=work)
            logits = evaluation.currents
            token = int(logits.argmax().item())
            nll = float(-F.log_softmax(logits, dim=0)[token].item())
            steps.append(SequenceAssemblyStep(
                position, prefix, token, address, scores, evaluation.winner_index,
                evaluation.agreement_counts, evaluation.exact, logits, nll))
            _add(work, "output_decisions", 1)
            if token == BYTE_EOS or position == max_bytes:
                break
            output.append(token)
            cache.advance(token)
        if (_digest(self.state_dict(), work) != model_hash
                or _digest(vars(state), work) != memory_hash):
            raise RuntimeError("free sequence response mutated model or memory")
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

    def validate_pending(self, prediction, state, *, observation=None, work=None):
        if work is None:
            work = {}
        self._check_conditioned_parameters(work)
        self._check_state(state)
        if not isinstance(prediction, SequenceAssemblyPrediction) or not self.is_pending(prediction):
            raise ValueError("response belongs to another learner or is already resolved")
        _, obs_hash, model_hash, memory_hash, trace_hash, work_hash = self._sequence_pending
        if (prediction.observation.sha256 != obs_hash
                or observation is not None and observation.sha256 != obs_hash):
            raise ValueError("response observation changed")
        if (prediction.model_sha256 != model_hash or prediction.memory_sha256 != memory_hash
                or _digest(self.state_dict(), work) != model_hash
                or _digest(vars(state), work) != memory_hash):
            raise ValueError("pending sequence model or memory changed")
        if (prediction.trace_sha256 != trace_hash or _digest(_trace(prediction), work) != trace_hash
                or _digest(prediction.work) != work_hash):
            raise ValueError("pending sequence response or work changed")
        _add(work, "sequence_trace_validations", 1)

    def acknowledge(self, prediction, state):
        work = {}
        self.validate_pending(prediction, state, work=work)
        self._sequence_pending = None
        _add(work, "sequence_acknowledgements", 1)
        return work

    @torch.no_grad()
    def teach(self, observation, teacher, state, learning_rate=0.01, *,
              prediction, learning_enabled=True):
        if (not isinstance(teacher, bytes) or len(teacher) > MAX_REPLACEMENT_BYTES
                or type(learning_rate) not in (float, int) or learning_rate != 0.01
                or type(learning_enabled) is not bool):
            raise ValueError("sequence teacher requires bounded raw bytes, eta.01 and Boolean gate")
        work = {}
        self.validate_pending(prediction, state, observation=observation, work=work)
        if len(teacher) > prediction.max_bytes:
            raise ValueError("teacher exceeds the actual response capacity")
        prepared = self._prepare_observation(observation, state, work)
        cache = RollingPrefixCoincidence(state.keys[:state.used], work=work)
        stage = self.assemblies.begin_stage(work)
        enabled = learning_enabled and self.mode != "slow_frozen"
        records, loss = [], 0.0
        for position, target in enumerate((*teacher, BYTE_EOS)):
            prefix = teacher[:position]
            inputs, fallback, address, scores = self._context_forward(prepared, prefix, cache, work)
            context = self.assemblies.prepare_context(inputs, fallback, stage=stage, work=work)
            # Current target is first passed after the complete input-only
            # preparation. This is a teacher-prefix state, not the actual trace.
            update = self.assemblies.prepare_teach(
                context, target, stage=stage, learning_enabled=enabled, work=work)
            nll = float(-F.log_softmax(context.evaluation.currents, dim=0)[target].item())
            loss += nll
            records.append({
                "position": position, "prefix_hex": prefix.hex(), "target": target,
                "memory_index": address, "address_scores": scores,
                "input_sha256": _digest({"input": inputs}, work),
                "free_assembly_index": context.evaluation.winner_index,
                "assembly_scores": context.evaluation.agreement_counts,
                "free_currents": context.evaluation.currents,
                "selected_index": update.selected_index, "exact": context.evaluation.exact,
                "recruited": update.recruited, "exhausted": update.exhausted,
                "staged": update.staged, "currents_before": update.currents_before,
                "currents_after": update.currents_after, "nll": nll,
            })
            if target != BYTE_EOS:
                cache.advance(target)
        self.assemblies.validate_stage(stage, work)
        self.validate_pending(prediction, state, observation=observation, work=work)
        changed = enabled and stage.revision > 0
        feedback = SequenceAssemblyFeedback(state, prediction.attempt_id, len(teacher) + 1,
                                            loss, loss / (len(teacher) + 1), changed,
                                            tuple(records), work)
        if changed:
            self.assemblies.publish(stage, work)
        self._sequence_pending = None
        return feedback

    def accounting(self):
        accounting = super().accounting()
        accounting.update({
            "assembly_capacity": 512,
            "assembly_occupied": int(self.assemblies.used.item()),
            "assembly_input_width": INPUT_WIDTH,
            "assembly_receptive_bytes": self.assemblies.prototypes.numel(),
            "assembly_outgoing_bytes": self.assemblies.outgoing.numel() * 8,
            "assembly_bank_bytes": self.assemblies.tensor_bytes,
            "assembly_stage_bytes": self.assemblies.tensor_bytes,
            "assembly_active_outgoing_values_per_selected_context": 257,
            "assembly_max_agreement_cells_per_context": 512 * INPUT_WIDTH,
            "assembly_max_agreement_thresholds_per_context": 3 * 512 * INPUT_WIDTH,
            "maximum_response_raw_bytes": MAX_REPLACEMENT_BYTES,
            "maximum_query_chunks": 9,
            "inherited_synaptic_learning": "none; B/C/U and codec held",
            "assembly_claim": "allocated/occupied local episodic cells; no generalization claim",
        })
        return accounting
