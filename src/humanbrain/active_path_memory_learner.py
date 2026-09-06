"""One actual-path output correction with separately staged local address teaching.

The retained free response supplies the selected cells and their own activity.
No target chooses an address, and no error traverses the fixed feature circuit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS, ByteGeneration
from humanbrain.incremental_prefix_coincidence import WORK_KEYS as PREFIX_WORK_KEYS
from humanbrain.incremental_prefix_coincidence import IncrementalPrefixCoincidence
from humanbrain.local_memory_learner import _finite
from humanbrain.neuronal_splice_learner import _digest
from humanbrain.persistent_coincidence_reader import compete512
from humanbrain.request_conditioned_memory_learner import (
    OUTPUT_WIDTH,
    TARGET_CURRENT,
    RequestConditionedFeedback,
    RequestConditionedMemoryLearner,
    RequestConditionedPrediction,
    RequestConditionedStep,
    _add,
)

ACTIVE_WORK_KEYS = (
    "active_custody_tensor_bytes_hashed",
    "active_retained_feature_values",
    "active_retained_feature_bytes",
    "active_trace_validations",
    "active_response_acknowledgements",
    "active_output_event_comparisons",
    "active_output_clamp_offers",
    "active_output_error_values",
    "active_output_energy_products",
    "active_output_energy_reductions",
    "active_output_local_products",
    "active_output_local_divisions",
    "active_output_parameter_additions",
)


@dataclass(frozen=True)
class ActivePathPrediction(RequestConditionedPrediction):
    selected_features: tuple[Tensor, ...]
    request: bytes
    max_bytes: int
    attempt_id: str
    model_sha256: str
    memory_sha256: str
    trace_sha256: str


@dataclass(frozen=True)
class ActivePathFeedback(RequestConditionedFeedback):
    correction_position: int | None = None
    correction_target: int | None = None
    active_output_offers: int = 0


def _hash(value, work):
    counted = {}
    result = _digest(value, counted)
    _add(work, "active_custody_tensor_bytes_hashed", counted.get("custody_tensor_bytes_hashed", 0))
    return result


def _trace_record(prediction):
    # A mapping (rather than a list of tensors) uses the established digest's
    # exact dtype/shape/byte framing. The work receipt is bound separately.
    return {
        "request_hex": prediction.request.hex(),
        "max_bytes": prediction.max_bytes,
        "attempt_id": prediction.attempt_id,
        "model_sha256": prediction.model_sha256,
        "memory_sha256": prediction.memory_sha256,
        "generation": {
            "data_hex": prediction.generation.data.hex(),
            "stop_reason": prediction.generation.stop_reason,
        },
        "steps": {
            str(i): {
                "output_step": step.output_step,
                "observed_token": step.observed_token,
                "nll": step.nll,
                "probabilities": step.probabilities,
                "prior_log_belief": step.prior_log_belief,
                "posterior_log_belief": step.posterior_log_belief,
                "selected_index": step.selected_index,
                "competition": vars(step.competition),
                "logits": step.logits,
                "prefix_hex": step.prefix.hex(),
            }
            for i, step in enumerate(prediction.steps)
        },
        "features": {str(i): value for i, value in enumerate(prediction.selected_features)},
    }


class ActivePathMemoryLearner(RequestConditionedMemoryLearner):
    """Continue existing weights while binding output teaching to actual activity."""

    WORK_KEYS = tuple(dict.fromkeys((
        *RequestConditionedMemoryLearner.WORK_KEYS, *PREFIX_WORK_KEYS, *ACTIVE_WORK_KEYS,
    )))

    def __init__(self, *, mode="learned", seed=0):
        super().__init__(mode=mode, seed=seed)
        self._active_pending = None
        self._active_experience = 0

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "active-path-memory-learner-v1",
            "address_activity": (
                "hold Xrequest; exact incremental Boolean prefix gates then fresh X"),
            "codec_credit": "zero/one first actual mismatch; own selected x/current unitrate clamp",
            "teacher_phase": "all targets teach B/C from current p-b with U held; no all-address U",
            "publication": (
                "private B/C stages plus one actual-path U proposal; atomic final copies"),
            "loss": (
                "held-U teacher mixture NLL; same-feature active clamp has no sequence guarantee"),
            "response_ownership": (
                "one identity-bound pending actual trace; explicit teach or acknowledgement"),
            "teacher_capacity": "raw target length must fit the pending response max_bytes<=16",
            "initialization": (
                "same inherited numerical tensors; active-credit and exact-cache policy change"),
        })
        return metadata

    def _require_idle(self):
        if self._active_pending is not None:
            raise ValueError("resolve the actual pending reader response first")

    def is_pending(self, prediction):
        return self._active_pending is not None and self._active_pending[0] is prediction

    @property
    def pending_prediction(self):
        return None if self._active_pending is None else self._active_pending[0]

    def ingest(self, documents, state=None):
        self._require_idle()
        return super().ingest(documents, state)

    def snapshot(self, state):
        self._require_idle()
        return super().snapshot(state)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._require_idle()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def restore_snapshot(self, snapshot):
        self._require_idle()
        return super().restore_snapshot(snapshot)

    def validate_pending(self, prediction, state, *, query=None, work=None):
        """Check both private ownership and mutable trace bytes before publication."""
        if work is None:
            work = {}
        self._check_conditioned_parameters()
        self._check_state(state)
        if not isinstance(prediction, ActivePathPrediction) or not self.is_pending(prediction):
            raise ValueError("teacher must bind this instance's actual pending response")
        _, request, model_hash, memory_hash, trace_hash, work_hash = self._active_pending
        if (query is not None and query != request) or prediction.request != request:
            raise ValueError("pending response request changed")
        if (prediction.model_sha256 != model_hash or prediction.memory_sha256 != memory_hash
                or _hash(self.state_dict(), work) != model_hash
                or _hash(vars(state), work) != memory_hash):
            raise ValueError("pending reader model or memory changed")
        try:
            if (not isinstance(prediction.steps, tuple)
                    or not isinstance(prediction.selected_features, tuple)
                    or len(prediction.steps) != len(prediction.selected_features)
                    or not 1 <= len(prediction.steps) <= 17):
                raise ValueError("actual output trace geometry changed")
            for step, feature in zip(prediction.steps, prediction.selected_features, strict=True):
                for value, shape in ((feature, (OUTPUT_WIDTH,)), (step.logits, (1, 257))):
                    if (not isinstance(value, Tensor) or tuple(value.shape) != shape
                            or value.dtype != torch.float64 or value.device.type != "cpu"
                            or value.requires_grad or value.grad_fn is not None
                            or value.grad is not None):
                        raise ValueError("actual selected activity must stay detached CPU float64")
                    _finite(value, "actual selected output activity")
            actual_trace = _hash(_trace_record(prediction), work)
            actual_work = _digest(prediction.work)
        except (AttributeError, TypeError, RuntimeError, ValueError, IndexError) as error:
            raise ValueError("pending actual trace is malformed") from error
        if (prediction.trace_sha256 != trace_hash or actual_trace != trace_hash
                or actual_work != work_hash):
            raise ValueError("pending actual trace, features or work changed")
        _add(work, "active_trace_validations", 1)

    def acknowledge(self, prediction, state):
        """Resolve reading, STOP or evaluation with zero forward/teacher calls."""
        work = dict.fromkeys(self.WORK_KEYS, 0)
        self.validate_pending(prediction, state, work=work)
        _add(work, "active_response_acknowledgements", 1)
        self._active_pending = None
        return work

    def _incremental_address(self, prefix, cache, state, xq, b_weight, c_weight, work):
        p = self._rails([BYTE_BOS, *prefix], 17, work)
        xp = cache.features()
        scores = (xq * b_weight).sum((1, 2)) + (xp * c_weight).sum((1, 2))
        _finite(scores, "active-path address scores")
        log_prior = F.log_softmax(scores.to(torch.float64), dim=0)
        _finite(log_prior, "active-path address prior")
        _add(work, "output_prefix_encodings", 1)
        _add(work, "output_prefix_symbols", len(prefix) + 1)
        _add(work, "output_prefix_feature_slots", state.used)
        _add(work, "output_address_steps", 1)
        _add(work, "output_address_synapse_products", state.used * 2048)
        return p, xp, scores, log_prior

    @torch.no_grad()
    def predict_with_receipt(self, query, state, max_bytes):
        self._require_idle()
        if type(max_bytes) is not int or not 0 <= max_bytes <= 16:
            raise ValueError("active output max_bytes must be0..16")
        work = dict.fromkeys(self.WORK_KEYS, 0)
        q, xq, context, legacy = self._prepare(query, state, work)
        model_hash = _hash(self.state_dict(), work)
        memory_hash = _hash(vars(state), work)
        cache = IncrementalPrefixCoincidence(state.keys[:state.used], work=work)
        output, steps, selected_features = bytearray(), [], []
        for position in range(max_bytes + 1):
            prefix = bytes(output)
            p, _, scores, log_prior = self._incremental_address(
                prefix, cache, state, xq, self.address_weight, self.prefix_weight, work)
            competition = compete512(scores)
            for name, value in competition.work.items():
                _add(work, name, value)
            selected = competition.winner_index
            index = torch.tensor([selected], dtype=torch.int64)
            x, logits, log_probabilities = self._conditional(
                q, p, context.index_select(0, index), legacy.index_select(0, index),
                position, self.output_weight, work)
            _add(work, "output_held_context_gathers", 64 * 18)
            _add(work, "output_held_current_gathers", 257)
            selected_features.append(x[0].detach().clone())
            _add(work, "active_retained_feature_values", OUTPUT_WIDTH)
            _add(work, "active_retained_feature_bytes", OUTPUT_WIDTH * 8)
            token = int(logits[0].argmax().item())
            steps.append(RequestConditionedStep(
                position, token, float(-log_probabilities[0, token].item()),
                log_probabilities[0].exp(), log_prior, None, selected,
                competition, logits, prefix))
            _add(work, "output_decisions", 1)
            if token == BYTE_EOS or position == max_bytes:
                break
            output.append(token)
            cache.advance(token)
        del cache
        if (_hash(self.state_dict(), work) != model_hash
                or _hash(vars(state), work) != memory_hash):
            raise RuntimeError("free response changed neuronal model or memory")
        attempt_id = f"{id(self)}:{self._active_experience + 1}"
        prediction = ActivePathPrediction(
            ByteGeneration(bytes(output), "eos" if token == BYTE_EOS else "limit"),
            tuple(steps), work, tuple(selected_features), query, max_bytes, attempt_id,
            model_hash, memory_hash, "")
        trace_hash = _hash(_trace_record(prediction), work)
        object.__setattr__(prediction, "trace_sha256", trace_hash)
        self._active_pending = (
            prediction, query, model_hash, memory_hash, trace_hash, _digest(work))
        self._active_experience += 1
        return prediction

    @torch.no_grad()
    def teach(self, query, teacher, state, learning_rate, *, prediction, learning_enabled=True):
        if (not isinstance(teacher, bytes) or len(teacher) > 16
                or type(learning_rate) not in (float, int) or learning_rate != 0.01
                or type(learning_enabled) is not bool):
            raise ValueError("active teacher requires raw<=16 bytes, eta.01 and boolean gate")
        work = dict.fromkeys(self.WORK_KEYS, 0)
        self.validate_pending(prediction, state, query=query, work=work)
        if len(teacher) > prediction.max_bytes:
            raise ValueError("teacher target exceeds the actual response capacity")
        targets = (*teacher, BYTE_EOS)
        correction_position = None
        for position, step in enumerate(prediction.steps):
            _add(work, "active_output_event_comparisons", 1)
            if step.observed_token != targets[position]:
                correction_position = position
                break
            if step.observed_token == BYTE_EOS:
                break
        # A bounded teacher EOS always creates an actual mismatch unless the
        # complete response already agrees. No new output event is generated.
        correction_target = None
        u_next = self.output_weight.clone()
        if correction_position is not None:
            correction_target = targets[correction_position]
            x_actual = prediction.selected_features[correction_position]
            z_actual = prediction.steps[correction_position].logits[0]
            desired = torch.zeros(257, dtype=torch.float64)
            desired[correction_target] = TARGET_CURRENT
            energy = x_actual.square().sum()
            if not bool(energy > 0):
                raise ValueError("actual output presynaptic energy must be positive")
            change = torch.outer(desired - z_actual, x_actual) / energy
            _finite(change, "active output local contribution")
            u_next.add_(change)
            _finite(u_next, "active output parameter proposal")
            _add(work, "active_output_clamp_offers", 1)
            _add(work, "active_output_error_values", 257)
            _add(work, "active_output_energy_products", OUTPUT_WIDTH)
            _add(work, "active_output_energy_reductions", OUTPUT_WIDTH - 1)
            _add(work, "active_output_local_products", 257 * OUTPUT_WIDTH)
            _add(work, "active_output_local_divisions", 257 * OUTPUT_WIDTH)
            _add(work, "active_output_parameter_additions", 257 * OUTPUT_WIDTH)
        q, xq, context, legacy = self._prepare(query, state, work)
        cache = IncrementalPrefixCoincidence(state.keys[:state.used], work=work)
        b_stage, c_stage = self.address_weight.clone(), self.prefix_weight.clone()
        work["output_staging_tensor_bytes"] = 2 * 4096 + 257 * OUTPUT_WIDTH * 8
        output_enabled = learning_enabled and self.mode != "slow_frozen"
        address_enabled = learning_enabled and self.mode not in ("slow_frozen", "address_frozen")
        steps, total_loss = [], 0.0
        for position, target in enumerate(targets):
            prefix = teacher[:position]
            p, xp, _, log_prior = self._incremental_address(
                prefix, cache, state, xq, b_stage, c_stage, work)
            _, logits, log_conditional = self._conditional(
                q, p, context, legacy, position, self.output_weight, work)
            mixture_log = torch.logsumexp(log_prior[:, None] + log_conditional, dim=0)
            _finite(mixture_log, "active teacher mixture")
            log_posterior = F.log_softmax(log_prior + log_conditional[:, target], dim=0)
            _finite(log_posterior, "active current teacher responsibility")
            address_error = log_prior.exp() - log_posterior.exp()
            b_change = (xq.to(torch.float64) * address_error[:, None, None]).sum(0)
            c_change = (xp.to(torch.float64) * address_error[:, None, None]).sum(0)
            _finite(b_change, "active local query address contribution")
            _finite(c_change, "active local prefix address contribution")
            b_next = b_stage - learning_rate * b_change.to(torch.float32)
            c_next = c_stage - learning_rate * c_change.to(torch.float32)
            _finite(b_next, "active query address parameter proposal")
            _finite(c_next, "active prefix address parameter proposal")
            loss = float(-mixture_log[target].item())
            total_loss += loss
            if not math.isfinite(total_loss):
                raise ValueError("active teacher loss must be finite")
            steps.append(RequestConditionedStep(
                position, target, loss, mixture_log.exp(), log_prior, log_posterior,
                None, None, logits, prefix))
            if address_enabled:
                b_stage, c_stage = b_next, c_next
            _add(work, "output_target_offers", 1)
            _add(work, "teacher_offers", 1)
            _add(work, "output_address_local_products", state.used * 2048)
            _add(work, "output_staged_update_steps", int(address_enabled))
            if target != BYTE_EOS:
                cache.advance(target)
        del cache
        # Revalidate original ownership and every final proposal before the
        # first live copy. A failure retains the pending actual response.
        self.validate_pending(prediction, state, query=query, work=work)
        _finite(b_stage, "final active query address proposal")
        _finite(c_stage, "final active prefix address proposal")
        _finite(u_next, "final active output parameter proposal")
        b_changed = address_enabled and not torch.equal(self.address_weight, b_stage)
        c_changed = address_enabled and not torch.equal(self.prefix_weight, c_stage)
        u_changed = output_enabled and not torch.equal(self.output_weight, u_next)
        work["output_query_address_enabled_groups"] = int(address_enabled)
        work["output_prefix_address_enabled_groups"] = int(address_enabled)
        work["output_enabled_groups"] = int(output_enabled and correction_position is not None)
        feedback = ActivePathFeedback(
            state, len(targets), total_loss, total_loss / len(targets),
            u_changed, b_changed or c_changed, tuple(steps), work,
            b_changed, c_changed, u_changed, False,
            correction_position, correction_target, int(correction_position is not None))
        if address_enabled:
            self.address_weight.copy_(b_stage)
            self.prefix_weight.copy_(c_stage)
        if output_enabled and correction_position is not None:
            self.output_weight.copy_(u_next)
        self._active_pending = None
        return feedback

    def accounting(self):
        account = super().accounting()
        account.update({
            "schema": "active-path-memory-accounting-v1",
            "maximum_active_output_clamps": 1,
            "maximum_active_output_products": 257 * OUTPUT_WIDTH,
            "maximum_all_address_output_update_products": 0,
            "maximum_retained_selected_feature_bytes": 17 * OUTPUT_WIDTH * 8,
            "maximum_prefix_boolean_cache_bytes": 512 * 32 * 32,
            "pending_state": "one actual response; excluded from checkpoint tensors",
            "publication": "B/C privately staged with U held; zero/one actual-path U; atomic final",
        })
        return account
