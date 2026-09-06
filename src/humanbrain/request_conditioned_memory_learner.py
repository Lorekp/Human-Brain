"""Causal request/prefix-conditioned output with direct local plasticity.

Every output position can select any occupied memory address. Teacher targets
enter only after forward currents, as a postsynaptic clamp and a declared latent
address responsibility. No error propagates through fixed input/threshold cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS, ByteGeneration
from humanbrain.local_memory_learner import _finite
from humanbrain.neuronal_byte_coincidence import coincidence_features, encode_symbol_rails
from humanbrain.neuronal_sequence_memory import TapeState
from humanbrain.persistent_coincidence_reader import PersistentCoincidenceReader512, compete512
from humanbrain.request_conditioned_output_cells import WORK_KEYS as CONTEXT_WORK_KEYS
from humanbrain.request_conditioned_output_cells import RequestConditionedOutputCells

OUTPUT_WIDTH = 292
TARGET_CURRENT = math.log(256)
EXTRA_WORK_KEYS = (
    "output_query_encodings",
    "output_prefix_encodings",
    "output_prefix_symbols",
    "output_context_bank_values",
    "output_context_bank_bytes",
    "output_context_cursor_advances",
    "output_legacy_current_slots",
    "output_legacy_current_potentials",
    "output_address_steps",
    "output_prefix_feature_slots",
    "output_query_feature_slots",
    "output_address_synapse_products",
    "output_conditional_populations",
    "output_synapse_products",
    "output_softmaxes",
    "output_decisions",
    "output_target_offers",
    "output_local_synapse_products",
    "output_local_slot_reductions",
    "output_local_denominator_values",
    "output_address_local_products",
    "output_staging_tensor_bytes",
    "output_staged_update_steps",
    "output_query_address_enabled_groups",
    "output_prefix_address_enabled_groups",
    "output_enabled_groups",
    "output_legacy_codec_enabled_groups",
    "output_held_context_gathers",
    "output_held_current_gathers",
)


def _add(work, key, count):
    work[key] = work.get(key, 0) + count


@dataclass(frozen=True)
class RequestConditionedStep:
    output_step: int
    observed_token: int
    nll: float
    probabilities: Tensor
    prior_log_belief: Tensor
    posterior_log_belief: Tensor | None
    selected_index: int | None
    competition: object | None
    logits: Tensor
    prefix: bytes


@dataclass(frozen=True)
class RequestConditionedPrediction:
    generation: ByteGeneration
    steps: tuple[RequestConditionedStep, ...]
    work: dict[str, int]


@dataclass(frozen=True)
class RequestConditionedFeedback:
    state: TapeState
    teacher_count: int
    total_loss: float
    mean_loss: float
    codec_updated: bool  # new-schema compatibility alias for output_updated, NOT legacy codec
    address_updated: bool
    steps: tuple[RequestConditionedStep, ...]
    work: dict[str, int]
    query_address_updated: bool
    prefix_address_updated: bool
    output_updated: bool
    legacy_codec_updated: bool = False


class RequestConditionedMemoryLearner(PersistentCoincidenceReader512):
    """One persistent prior plus new prefix/address and conditional output cells."""

    FIXED_STATE_NAMES = ("input_codes", "context.indices", "context.signs")
    WORK_KEYS = (*PersistentCoincidenceReader512.WORK_KEYS, *CONTEXT_WORK_KEYS, *EXTRA_WORK_KEYS)

    def __init__(self, *, mode="learned", seed=0):
        # Do not override the base constructor's dynamic parameter check: the
        # new groups do not exist until after that original construction.
        super().__init__(mode=mode, seed=seed)
        self.prefix_weight = nn.Parameter(
            torch.zeros(32, 32, dtype=torch.float32), requires_grad=False
        )
        self.output_weight = nn.Parameter(
            torch.zeros(257, OUTPUT_WIDTH, dtype=torch.float64), requires_grad=False
        )
        self.context = RequestConditionedOutputCells(seed=seed)
        self._check_conditioned_parameters()

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update(
            {
                "schema": "request-conditioned-memory-learner-v1",
                "recall": (
                    "all occupied addresses reselected at every actual output prefix; no advance"
                ),
                "query": "whole raw request<=31 plus EOS; prefix BOS plus earlier raw outputs<=16",
                "address": (
                    "separate float32 sum(B*Xrequest)+sum(C*Xprefix); all addresses each step"
                ),
                "address_credit": "each teacher position current p-b; local own X; staged eta.01",
                "address_synapse_credit": (
                    "separate own-Xrequest and own-Xprefix products with current p-b; "
                    "float64 reductions cast tofloat32 at each private teacher step"
                ),
                "address_activity": (
                    "hold full Xrequest once; rebuild full Xprefix at every output position"
                ),
                "free_belief": (
                    "full diagnostic prior at each position; selected conditional output only; "
                    "no recurrent free posterior"
                ),
                "teacher_phase": (
                    "all candidates reopened independently per earlier teacher prefix; "
                    "current forward likelihood responsibility, no final-posterior history"
                ),
                "codec": (
                    "legacy numeric codec held fixed; new257x292 float64 residual output currents"
                ),
                "codec_activity": (
                    "hold legacy currents only; recompute conditional logits and softmax"
                ),
                "codec_credit": (
                    "U unitrate normalized output potential clamp; legacy codec unchanged"
                ),
                "codec_updated_semantics": "alias for U output_updated; legacy_codec_updated=False",
                "output_target_current": TARGET_CURRENT,
                "output_feature_energy": 32,
                "output_feature_width": OUTPUT_WIDTH,
                "output_context_cells": 256,
                "output_context_fan_in": 32,
                "output_context_seed_offset": 104729,
                "output_context": (
                    "32 preceding raw rows plus current and31 EOS-absorbing successors"
                ),
                "publication": (
                    "sequential private B/C/U stages; all finite before one final publication"
                ),
                "disabled_staging": "compute candidate checks but no advance of disabled groups",
                "loss": (
                    "reported pre-clamp mixture NLL; local clamp descent is fixed-b quadratic only"
                ),
                "initialization": (
                    "C/Uzero; old numerical tensors retained; no free trajectory parity"
                ),
            }
        )
        return metadata

    def _check_conditioned_parameters(self):
        super()._check_parameters()
        self.codec.check_parameters()
        for value, shape, dtype in (
            (self.prefix_weight, (32, 32), torch.float32),
            (self.output_weight, (257, OUTPUT_WIDTH), torch.float64),
        ):
            if (
                value.shape != shape
                or value.dtype != dtype
                or value.device.type != "cpu"
                or value.requires_grad
                or value.grad_fn is not None
                or value.grad is not None
            ):
                raise ValueError("conditioned output parameters must retain detached CPU geometry")
            _finite(value, "conditioned output parameters")
        self.context.check_parameters()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._check_conditioned_parameters()
        self.context.validate_fixed_state(state_dict, "context.")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._check_conditioned_parameters()
        return result

    def ingest(self, documents, state=None):
        self._check_conditioned_parameters()
        return super().ingest(documents, state)

    def snapshot(self, state):
        self._check_conditioned_parameters()
        return super().snapshot(state)

    @staticmethod
    def _rails(tokens, width, work):
        if len(tokens) > width:
            raise ValueError("raw conditioned population exceeds capacity")
        encoded = encode_symbol_rails(torch.tensor(tokens, dtype=torch.int64), work=work)
        rows = torch.zeros(width, 18, dtype=torch.bool)
        rows[: len(tokens)].copy_(encoded)
        return rows

    def _feature_bank(self, lags, state, work):
        features = torch.empty(state.used, 32, 32, dtype=torch.float32)
        for first in range(0, state.used, 128):
            last = min(first + 128, state.used)
            block = coincidence_features(lags, state.keys[first:last], work=work)
            features[first:last].copy_(block)
            _add(work, "persistent_feature_block_count", 1)
        _add(work, "persistent_feature_assembly_bytes", features.numel() * 4)
        _add(work, "persistent_feature_copy_values", features.numel())
        return features

    def _prepare(self, query, state, work):
        self._check_conditioned_parameters()
        self._check_state(state)
        if not isinstance(query, bytes) or len(query) > 31:
            raise ValueError("conditioned output request must be raw bytes of length<=31")
        q = self._rails([*query, BYTE_EOS], 32, work)
        lags = torch.zeros_like(q)
        lags[: len(query) + 1].copy_(q[: len(query) + 1].flip(0))
        xq = self._feature_bank(lags, state, work)
        work["query_raw_bytes"], work["query_eos_symbols"] = len(query), 1
        work["output_query_encodings"] = 1
        work["output_query_feature_slots"] = state.used
        sensory = state.sensory[: state.used]
        active = sensory.sum(1)
        tokens = sensory.argmax(1)
        rails = encode_symbol_rails(tokens, work=work)
        rails &= active.to(torch.bool)[:, None]
        context = torch.empty(state.used, 64, 18, dtype=torch.bool)
        context[:, :32].copy_(state.keys[: state.used])
        cursors = torch.arange(state.used, dtype=torch.int64)
        for position in range(32):
            context[:, 32 + position].copy_(rails.index_select(0, cursors))
            if position < 31:
                cursors = self.memory._advance_checked(state, cursors)
                _add(work, "output_context_cursor_advances", state.used)
        # Only immutable legacy currents are cached. Prefix-conditioned output
        # probabilities do not exist until the current prefix has been supplied.
        legacy = self.codec.weight.T.index_select(0, tokens) * active[:, None] + self.codec.bias
        _finite(legacy, "held legacy output currents")
        work["output_context_bank_values"] = context.numel()
        work["output_context_bank_bytes"] = context.numel()
        work["output_legacy_current_slots"] = state.used
        work["output_legacy_current_potentials"] = legacy.numel()
        return q, xq, context, legacy

    def _address(self, prefix, state, xq, b_weight, c_weight, work):
        p = self._rails([BYTE_BOS, *prefix], 17, work)
        lags = torch.zeros(32, 18, dtype=torch.bool)
        lags[: len(prefix) + 1].copy_(p[: len(prefix) + 1].flip(0))
        xp = self._feature_bank(lags, state, work)
        scores = (xq * b_weight).sum((1, 2)) + (xp * c_weight).sum((1, 2))
        _finite(scores, "conditioned address scores")
        log_prior = F.log_softmax(scores.to(torch.float64), dim=0)
        _finite(log_prior, "conditioned address prior")
        _add(work, "output_prefix_encodings", 1)
        _add(work, "output_prefix_symbols", len(prefix) + 1)
        _add(work, "output_prefix_feature_slots", state.used)
        _add(work, "output_address_steps", 1)
        _add(work, "output_address_synapse_products", state.used * 2048)
        return p, xp, scores, log_prior

    def _conditional(self, q, p, context, legacy, position, weight, work):
        x = self.context(q, p, context, context[:, 32], position, work=work)
        logits = legacy + x @ weight.T
        _finite(logits, "conditioned output currents")
        log_probabilities = F.log_softmax(logits, dim=1)
        _finite(log_probabilities, "conditioned output log probabilities")
        _add(work, "output_conditional_populations", x.shape[0])
        _add(work, "output_synapse_products", x.shape[0] * 257 * OUTPUT_WIDTH)
        _add(work, "output_softmaxes", x.shape[0])
        return x, logits, log_probabilities

    @torch.no_grad()
    def predict_with_receipt(self, query: bytes, state: TapeState, max_bytes: int):
        if type(max_bytes) is not int or not 0 <= max_bytes <= 16:
            raise ValueError("conditioned output max_bytes must be0..16")
        work = dict.fromkeys(self.WORK_KEYS, 0)
        q, xq, context, legacy = self._prepare(query, state, work)
        output, steps = bytearray(), []
        for position in range(max_bytes + 1):
            prefix = bytes(output)
            p, _, scores, log_prior = self._address(
                prefix, state, xq, self.address_weight, self.prefix_weight, work
            )
            competition = compete512(scores)
            for name, value in competition.work.items():
                _add(work, name, value)
            selected = competition.winner_index
            index = torch.tensor([selected], dtype=torch.int64)
            _, logits, log_probabilities = self._conditional(
                q,
                p,
                context.index_select(0, index),
                legacy.index_select(0, index),
                position,
                self.output_weight,
                work,
            )
            _add(work, "output_held_context_gathers", 64 * 18)
            _add(work, "output_held_current_gathers", 257)
            token = int(logits[0].argmax().item())
            steps.append(
                RequestConditionedStep(
                    position,
                    token,
                    float(-log_probabilities[0, token].item()),
                    log_probabilities[0].exp(),
                    log_prior,
                    None,
                    selected,
                    competition,
                    logits,
                    prefix,
                )
            )
            _add(work, "output_decisions", 1)
            if token == BYTE_EOS or position == max_bytes:
                return RequestConditionedPrediction(
                    ByteGeneration(bytes(output), "eos" if token == BYTE_EOS else "limit"),
                    tuple(steps),
                    work,
                )
            output.append(token)
        raise AssertionError("bounded conditioned generation did not terminate")

    @torch.no_grad()
    def teach(self, query, teacher, state, learning_rate, *, learning_enabled=True):
        if (
            not isinstance(teacher, bytes)
            or len(teacher) > 16
            or type(learning_rate) not in (float, int)
            or learning_rate != 0.01
            or type(learning_enabled) is not bool
        ):
            raise ValueError("conditioned teacher requires raw<=16 bytes, eta.01 and boolean gate")
        work = dict.fromkeys(self.WORK_KEYS, 0)
        q, xq, context, legacy = self._prepare(query, state, work)
        b_stage, c_stage, u_stage = (
            self.address_weight.clone(),
            self.prefix_weight.clone(),
            self.output_weight.clone(),
        )
        work["output_staging_tensor_bytes"] = 2 * 4096 + 257 * OUTPUT_WIDTH * 8
        output_enabled = learning_enabled and self.mode != "slow_frozen"
        address_enabled = learning_enabled and self.mode not in ("slow_frozen", "address_frozen")
        steps, total_loss = [], 0.0
        for position, target in enumerate((*teacher, BYTE_EOS)):
            prefix = teacher[:position]
            p, xp, _, log_prior = self._address(prefix, state, xq, b_stage, c_stage, work)
            x, logits, log_conditional = self._conditional(
                q, p, context, legacy, position, u_stage, work
            )
            mixture_log = torch.logsumexp(log_prior[:, None] + log_conditional, dim=0)
            _finite(mixture_log, "conditioned teacher mixture")
            log_posterior = F.log_softmax(log_prior + log_conditional[:, target], dim=0)
            _finite(log_posterior, "conditioned current teacher responsibility")
            responsibility = log_posterior.exp()
            desired = torch.zeros(257, dtype=torch.float64)
            desired[target] = TARGET_CURRENT
            energy = x.square().sum(1)
            if not bool((energy > 0).all()):
                raise ValueError("conditioned presynaptic energy must be positive")
            potential_error = (desired - logits) * (responsibility / energy)[:, None]
            u_change = potential_error.T @ x
            address_error = log_prior.exp() - responsibility
            b_change = (xq.to(torch.float64) * address_error[:, None, None]).sum(0)
            c_change = (xp.to(torch.float64) * address_error[:, None, None]).sum(0)
            _finite(u_change, "conditioned local output contribution")
            _finite(b_change, "conditioned local query address contribution")
            _finite(c_change, "conditioned local prefix address contribution")
            b_next = b_stage - learning_rate * b_change.to(torch.float32)
            c_next = c_stage - learning_rate * c_change.to(torch.float32)
            u_next = u_stage + u_change
            # These are private proposals; no group publishes on an earlier
            # teacher byte. A fault on the final offer preserves every live group.
            _finite(b_next, "conditioned query address parameter proposal")
            _finite(c_next, "conditioned prefix address parameter proposal")
            _finite(u_next, "conditioned output parameter proposal")
            loss = float(-mixture_log[target].item())
            total_loss += loss
            if not math.isfinite(total_loss):
                raise ValueError("conditioned teacher loss must be finite")
            steps.append(
                RequestConditionedStep(
                    position,
                    target,
                    loss,
                    mixture_log.exp(),
                    log_prior,
                    log_posterior,
                    None,
                    None,
                    logits,
                    prefix,
                )
            )
            if output_enabled:
                u_stage = u_next
            if address_enabled:
                b_stage, c_stage = b_next, c_next
            _add(work, "output_target_offers", 1)
            _add(work, "teacher_offers", 1)
            _add(work, "output_local_synapse_products", state.used * 257 * OUTPUT_WIDTH)
            _add(work, "output_local_slot_reductions", (state.used - 1) * 257 * OUTPUT_WIDTH)
            _add(work, "output_local_denominator_values", state.used * OUTPUT_WIDTH)
            _add(work, "output_address_local_products", state.used * 2048)
            _add(work, "output_staged_update_steps", int(output_enabled or address_enabled))
        b_changed = address_enabled and not torch.equal(self.address_weight, b_stage)
        c_changed = address_enabled and not torch.equal(self.prefix_weight, c_stage)
        u_changed = output_enabled and not torch.equal(self.output_weight, u_stage)
        work["output_query_address_enabled_groups"] = int(address_enabled)
        work["output_prefix_address_enabled_groups"] = int(address_enabled)
        work["output_enabled_groups"] = int(output_enabled)
        work["output_legacy_codec_enabled_groups"] = 0
        feedback = RequestConditionedFeedback(
            state,
            len(steps),
            total_loss,
            total_loss / len(steps),
            u_changed,
            b_changed or c_changed,
            tuple(steps),
            work,
            b_changed,
            c_changed,
            u_changed,
        )
        if address_enabled:
            self.address_weight.copy_(b_stage)
            self.prefix_weight.copy_(c_stage)
        if output_enabled:
            self.output_weight.copy_(u_stage)
        return feedback

    def accounting(self):
        self._check_conditioned_parameters()
        return {
            "schema": "request-conditioned-memory-accounting-v1",
            "model_tensor_bytes": sum(
                v.numel() * v.element_size()
                for v in self.state_dict().values()
                if isinstance(v, Tensor)
            ),
            "tape": self.memory.accounting(self.memory.initial_state()),
            "new_slow_tensor_bytes": 604448,
            "fixed_context_wiring_bytes": 73728,
            "output_neurons": 257,
            "context_threshold_neurons": 256,
            "conditional_fan_in": OUTPUT_WIDTH,
            "maximum_output_decisions": 17,
            "maximum_teacher_conditional_populations": 512 * 17,
            "maximum_teacher_output_products": 512 * 17 * 257 * OUTPUT_WIDTH,
            "maximum_free_output_products": 17 * 257 * OUTPUT_WIDTH,
            "maximum_raw_context_bytes": 512 * 64 * 18,
            "maximum_legacy_current_bytes": 512 * 257 * 8,
            "maximum_teacher_logit_receipt_bytes": 17 * 512 * 257 * 8,
            "publication": "sequential private stages; one atomic B/C/U publication",
            "legacy_codec": "fixed numerical prior; no learned publication",
        }
