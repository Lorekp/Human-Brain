"""Local address learning over fixed query lags and a literal neuronal sequence.

Answer-free sensory associations store every document byte. One shared query
projection addresses preceding-context keys. Forward cursors and causal prefix
beliefs recall ordered content through a shared conditional byte codec. Codec
filtering credit remains explicitly approximate; address credit is exact
conditional on the fixed normalized query features and held current codec.
"""

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS, ByteGeneration
from humanbrain.neuronal_sequence_memory import NeuronalSequenceMemory, TapeState
from humanbrain.plastic_memory import _all_finite


def _finite(value: Tensor, name: str) -> None:
    if not _all_finite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ConditionalByteStep:
    """One shared-circuit pass, saved until its current local errors are formed."""

    sensory: Tensor
    logits: Tensor
    log_probabilities: Tensor
    probabilities: Tensor


class ConditionalByteCodec(nn.Module):
    """Shared 257-way codec, time-multiplexed over branch sensory activities.

    One-hot sensory input activates one weight column; zero input uses bias.
    Sparse column currents never inspect the teacher. The smooth identity prior
    gives probability 1/2 to the sensory identity and 1/512 to every alternative.
    There is no punctuation-to-EOS exception or separately tied slot codec.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            math.log(256) * torch.eye(257, dtype=torch.float64, device="cpu")
        )
        self.bias = nn.Parameter(torch.zeros(257, dtype=torch.float64, device="cpu"))

    def check_parameters(self) -> None:
        for name, shape in (("weight", (257, 257)), ("bias", (257,))):
            value = getattr(self, name)
            if value.shape != shape or value.dtype != torch.float64 or value.device.type != "cpu":
                raise ValueError("conditional codec must retain CPU float64 geometry")
            if value.grad is not None or value.grad_fn is not None:
                raise ValueError("codec must not retain autograd gradients/history")
            _finite(value, "conditional codec parameters")

    @torch.no_grad()
    def evaluate(self, sensory: Tensor) -> ConditionalByteStep:
        self.check_parameters()
        if (
            not isinstance(sensory, Tensor) or sensory.ndim != 2 or sensory.shape[1] != 257
            or sensory.shape[0] < 1 or sensory.dtype != torch.float32
            or sensory.device.type != "cpu" or sensory.requires_grad or sensory.grad_fn is not None
        ):
            raise ValueError("sensory input must be detached CPU float32 [branches,257]")
        _finite(sensory, "sensory activity")
        if not bool(((sensory == 0) | (sensory == 1)).all()) or bool((sensory.sum(1) > 1).any()):
            raise ValueError("sensory activity must be one-hot or all zero")
        active = sensory.sum(1).to(torch.float64)
        columns = sensory.argmax(1)
        logits = self.weight.T.index_select(0, columns) * active[:, None] + self.bias
        _finite(logits, "conditional logits")
        log_probabilities = F.log_softmax(logits, dim=1)
        _finite(log_probabilities, "conditional log probabilities")
        probabilities = log_probabilities.exp()
        _finite(probabilities, "conditional probabilities")
        return ConditionalByteStep(sensory, logits, log_probabilities, probabilities)


@dataclass(frozen=True)
class SequenceQuery:
    activity: Tensor
    projected_query: Tensor
    log_prior: Tensor
    work: dict[str, int]


@dataclass(frozen=True)
class SequenceStep:
    output_step: int
    observed_token: int
    nll: float
    mixture_probabilities: Tensor
    prior_log_belief: Tensor
    posterior_log_belief: Tensor
    branch_cursors: Tensor


@dataclass(frozen=True)
class SequencePrediction:
    generation: ByteGeneration
    steps: tuple[SequenceStep, ...]
    work: dict[str, int]
    initial_log_belief: Tensor
    final_log_belief: Tensor


@dataclass(frozen=True)
class SequenceLessonFeedback:
    state: TapeState
    teacher_count: int
    total_loss: float
    mean_loss: float
    codec_updated: bool
    address_updated: bool
    steps: tuple[SequenceStep, ...]
    work: dict[str, int]
    initial_log_belief: Tensor
    final_log_belief: Tensor
    address_error: Tensor
    loss_kind: str = "sequence_mixture_negative_log_likelihood"


class LocalMemoryLearner(nn.Module):
    """Four matched controls over one literal neuronal episode.

    learned: address/codec learn. address_frozen: only codec learns.
    slow_frozen: all slow parameters fixed, sensory storage still active.
    writes_disabled: allocation/boundary clocks advance with zero keys/content.
    No writer tags, event encoder, static phase decoder or compatibility mode.
    Geometry is fixed; generation and teacher lengths are generic bounded bytes.
    """

    MODES = frozenset({"learned", "address_frozen", "slow_frozen", "writes_disabled"})
    WORK_KEYS = (
        "query_raw_bytes", "query_eos_symbols", "query_lag_shift_steps",
        "query_population_normalizations", "address_projection_neurons",
        "address_projection_synapse_evaluations", "address_score_evaluations",
        "address_local_synapse_contributions",
        "address_normalizations", "codec_slot_evaluations", "codec_output_potentials",
        "codec_normalizations", "mixture_output_reductions", "prefix_belief_updates",
        "branch_cursor_initializations", "branch_cursor_advances", "teacher_offers",
        "codec_update_group_offers",
        "address_update_group_offers",
        "enabled_codec_update_groups", "enabled_address_update_groups",
    )

    def __init__(self, *, mode: str = "learned", seed: int = 0) -> None:
        super().__init__()
        if mode not in self.MODES or type(seed) is not int:
            raise ValueError("invalid sequence learner mode or seed")
        self.mode, self.seed = mode, seed
        self.memory = NeuronalSequenceMemory()
        generator = torch.Generator(device="cpu").manual_seed(seed + 2)
        self.register_buffer(
            "input_codes",
            torch.empty(BYTE_BOS + 1, 32, dtype=torch.float32, device="cpu").normal_(
                generator=generator
            ),
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + 4)
        self.address_weight = nn.Parameter(
            torch.empty(1024, 1024, dtype=torch.float32, device="cpu")
        )
        nn.init.xavier_uniform_(self.address_weight, generator=generator)
        self.codec = ConditionalByteCodec()
        if mode in {"address_frozen", "slow_frozen"}:
            self.address_weight.requires_grad_(False)
        if mode == "slow_frozen":
            self.codec.requires_grad_(False)
        self._check_parameters()

    def get_extra_state(self) -> dict:
        return {
            "schema": "local-neuronal-lag-query-learner-v1", "mode": self.mode, "seed": self.seed,
            "capacity": 128, "lags": 32, "input_code_size": 32, "query_size": 1024,
            "input_code": "fixed normal258x32; private seed+2; raw bytes plus EOS256",
            "address": "one shared1024x1024 Xavier projection; private seed+4; no slot parameters",
            "query": "fixed32x32 newest-first raw-code lags including EOS; reset each query",
            "query_normalization": (
                "float32 lag/norm.clamp_min(1e-8)*sqrt(32); "
                "squared energy at most32 in real arithmetic"
            ),
            "codec": "shared257x257; log(256)*identity and zero bias; float64",
            "prior": "same-identity probability1/2; every other byte/EOS1/512",
            "recall": "all starts; forward EOS-absorbing adjacency; causal prefix belief",
            "loss": "sequence_mixture_negative_log_likelihood",
            "address_credit": "initial probability minus final prefix posterior",
            "codec_credit": "current prefix responsibility; filtering approximation; local sum",
            "publication": "all weights held through teacher sequence; eta0.01 summed once",
            "upstream_dtype": "torch.float32", "codec_signal_dtype": "torch.float64",
            "retention": "literal sensory/key synapses and layout; no semantic compression claim",
        }

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("sequence learner checkpoint configuration mismatch")

    def _check_parameters(self) -> None:
        if self.mode not in self.MODES:
            raise ValueError("invalid sequence learner mode")
        tensors = [(self.input_codes, (258, 32)), (self.address_weight, (1024, 1024))]
        for value, shape in tensors:
            if value.shape != shape or value.dtype != torch.float32 or value.device.type != "cpu":
                raise ValueError("query/address/codes must retain CPU float32 geometry")
            if value.grad_fn is not None or value.grad is not None:
                raise ValueError("local parameters must not retain autograd history or gradients")
            _finite(value, "query/address/codes")
        if self.input_codes.requires_grad:
            raise ValueError("input codes must stay fixed")
        self.codec.check_parameters()
        if self.mode in {"address_frozen", "slow_frozen"} and self.address_weight.requires_grad:
            raise ValueError("address parameters must remain frozen in this control")
        if self.mode == "slow_frozen" and any(p.requires_grad for p in self.codec.parameters()):
            raise ValueError("slow-frozen codec parameters must remain frozen")

    def _check_state(self, state: TapeState, *, populated: bool = True) -> None:
        self.memory.check_state(state)
        if populated and state.used == 0:
            raise ValueError("recall requires at least one allocated sensory slot")
        if self.mode == "writes_disabled" and (
            bool(state.keys.count_nonzero()) or bool(state.sensory.count_nonzero())
        ):
            raise ValueError("writes-disabled state must contain zero sensory/key activity")

    @torch.no_grad()
    def ingest(self, documents: Sequence[bytes], state: TapeState | None = None) -> TapeState:
        """Teacher-free append; temporary lag state resets at each document EOS."""
        self._check_parameters()
        if state is not None:
            self._check_state(state, populated=False)
        result = self.memory.ingest(
            documents, self.input_codes, state, writes_enabled=self.mode != "writes_disabled"
        )
        self._check_state(result, populated=False)
        return result

    @torch.no_grad()
    def _query(self, query: bytes, state: TapeState) -> SequenceQuery:
        self._check_parameters()
        self._check_state(state)
        if not isinstance(query, bytes):
            raise ValueError("query must be raw bytes")
        tokens = torch.tensor([*query, BYTE_EOS], dtype=torch.long, device="cpu")
        codes = F.embedding(tokens, self.input_codes)
        lags = torch.zeros(32, 32, dtype=torch.float32, device="cpu")
        for inputs in codes.unbind(0):
            # Fixed forward adjacency; every byte and EOS take the same route.
            lags[1:].copy_(lags[:-1].clone())
            lags[0].copy_(inputs)
        norm = torch.linalg.vector_norm(lags)
        _finite(norm, "query lag norm")
        activity = (lags.flatten() / norm.clamp_min(1e-8) * math.sqrt(32))[None]
        _finite(activity, "normalized query lag activity")
        projected = F.linear(activity, self.address_weight.T)[0]
        _finite(projected, "projected query")
        scores = F.linear(projected, state.keys[:state.used])
        _finite(scores, "address scores")
        log_prior = F.log_softmax(scores.to(torch.float64), dim=0)
        _finite(log_prior, "initial log belief")
        work = dict.fromkeys(self.WORK_KEYS, 0)
        work.update({
            "query_raw_bytes": len(query), "query_eos_symbols": 1,
            "query_lag_shift_steps": len(query) + 1,
            "query_population_normalizations": 1,
            "address_projection_neurons": 1024, "address_score_evaluations": state.used,
            "address_projection_synapse_evaluations": 1024 * 1024,
            "address_normalizations": 1,
        })
        return SequenceQuery(activity, projected, log_prior, work)

    @torch.no_grad()
    def query_receipt(self, query: bytes, state: TapeState) -> SequenceQuery:
        """One read-only query/address traversal for declared delay distractors."""
        return self._query(query, state)

    @staticmethod
    def _mixture(step: ConditionalByteStep, log_belief: Tensor) -> Tensor:
        result = torch.logsumexp(log_belief[:, None] + step.log_probabilities, dim=0)
        _finite(result, "mixture log probabilities")
        return result

    @staticmethod
    def _observe(
        position: int, token: int, cursors: Tensor, log_belief: Tensor,
        codec_step: ConditionalByteStep, log_mixture: Tensor,
    ) -> SequenceStep:
        posterior = log_belief + codec_step.log_probabilities[:, token] - log_mixture[token]
        _finite(posterior, "prefix log belief")
        loss = -float(log_mixture[token])
        if not math.isfinite(loss):
            raise ValueError("sequence loss must be finite")
        return SequenceStep(
            position, token, loss, log_mixture.exp(), log_belief.clone(), posterior, cursors.clone()
        )

    @staticmethod
    def _count_step(work: dict[str, int], branches: int) -> None:
        work["codec_slot_evaluations"] += branches
        work["codec_output_potentials"] += branches * 257
        work["codec_normalizations"] += branches
        work["mixture_output_reductions"] += 257
        work["prefix_belief_updates"] += branches

    @torch.no_grad()
    def predict_with_receipt(
        self, query: bytes, state: TapeState, max_bytes: int
    ) -> SequencePrediction:
        """Single actual free traversal, filtering only on its own output choices."""
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        event = self._query(query, state)
        cursors = self.memory.initial_recall(state)
        log_belief = event.log_prior
        output, steps, work = bytearray(), [], dict(event.work)
        work["branch_cursor_initializations"] = state.used
        for position in range(max_bytes + 1):
            conditional = self.codec.evaluate(self.memory._content_checked(state, cursors))
            mixture = self._mixture(conditional, log_belief)
            token = int(mixture.argmax())
            receipt = self._observe(position, token, cursors, log_belief, conditional, mixture)
            steps.append(receipt)
            log_belief = receipt.posterior_log_belief
            self._count_step(work, state.used)
            if token == BYTE_EOS or position == max_bytes:
                return SequencePrediction(
                    ByteGeneration(bytes(output), "eos" if token == BYTE_EOS else "limit"),
                    tuple(steps), work, event.log_prior.clone(), log_belief.clone(),
                )
            output.append(token)
            cursors = self.memory._advance_checked(state, cursors)
            work["branch_cursor_advances"] += state.used
        raise AssertionError("bounded generation must have a stopping decision")

    @torch.no_grad()
    def predict(self, query: bytes, state: TapeState, max_bytes: int) -> ByteGeneration:
        return self.predict_with_receipt(query, state, max_bytes).generation

    @torch.no_grad()
    def teach(
        self, query: bytes, teacher: bytes, state: TapeState, learning_rate: float,
        *, learning_enabled: bool = True,
    ) -> SequenceLessonFeedback:
        """Held-sequence local teaching; no backward history or payload correction.

        Codec errors use each current prefix posterior when the teacher arrives,
        not the final posterior on past errors. All teacher bytes/EOS are offered.
        Tape is caller-owned and unchanged; fixed query lags carry no eligibility.
        """
        if not isinstance(teacher, bytes):
            raise ValueError("teacher must be raw bytes")
        if type(learning_rate) not in {float, int} or learning_rate != 0.01:
            raise ValueError("sequence learning_rate must be 0.01")
        if type(learning_enabled) is not bool:
            raise ValueError("learning_enabled must be boolean")
        event = self._query(query, state)
        cursors = self.memory.initial_recall(state)
        log_belief = event.log_prior
        weight_error = torch.zeros_like(self.codec.weight)
        bias_error = torch.zeros_like(self.codec.bias)
        steps, work, total_loss = [], dict(event.work), 0.0
        work["branch_cursor_initializations"] = state.used
        targets = (*teacher, BYTE_EOS)
        for position, token in enumerate(targets):
            conditional = self.codec.evaluate(self.memory._content_checked(state, cursors))
            mixture = self._mixture(conditional, log_belief)
            receipt = self._observe(position, token, cursors, log_belief, conditional, mixture)
            steps.append(receipt)
            log_belief = receipt.posterior_log_belief
            responsibility = log_belief.exp()
            error = conditional.probabilities.clone()
            error[:, token] -= 1
            error *= responsibility[:, None]
            _finite(error, "current prefix codec error")
            # Sparse shared-synapse update. Zero sensory activity has no column
            # contribution, but reaches the same bias neurons.
            active = conditional.sensory.sum(1).to(torch.float64)
            columns = conditional.sensory.argmax(1)
            weight_error.index_add_(1, columns, (error * active[:, None]).T)
            bias_error.add_(error.sum(0))
            _finite(weight_error, "accumulated codec weight error")
            _finite(bias_error, "accumulated codec bias error")
            total_loss += receipt.nll
            if not math.isfinite(total_loss):
                raise ValueError("accumulated sequence loss must be finite")
            self._count_step(work, state.used)
            work["teacher_offers"] += 1
            if position + 1 < len(targets):
                cursors = self.memory._advance_checked(state, cursors)
                work["branch_cursor_advances"] += state.used
        address_error = event.log_prior.exp() - log_belief.exp()
        _finite(address_error, "final address error")
        projection_signal = F.linear(address_error, state.keys[:state.used].T.to(torch.float64))
        _finite(projection_signal, "address projection neuron error")
        address_change64 = event.activity[0].to(torch.float64)[:, None] * projection_signal[None]
        _finite(address_change64, "address synapse local contribution")
        work["address_local_synapse_contributions"] = self.address_weight.numel()
        address_change = address_change64.to(torch.float32)
        _finite(address_change, "cast address synapse contribution")

        codec_enabled = learning_enabled and self.mode != "slow_frozen"
        address_enabled = learning_enabled and self.mode not in {"address_frozen", "slow_frozen"}
        codec_proposals = []
        if codec_enabled:
            pairs = ((self.codec.weight, weight_error), (self.codec.bias, bias_error))
            for parameter, error in pairs:
                if parameter.requires_grad:
                    candidate = parameter - learning_rate * error
                    _finite(candidate, "codec parameter proposal")
                    codec_proposals.append((parameter, candidate))
        address_proposal = None
        if address_enabled and self.address_weight.requires_grad:
            address_proposal = self.address_weight - learning_rate * address_change
            _finite(address_proposal, "address parameter proposal")
        # Every candidate is finite before either group publishes. All signals
        # and contributions use held old values, with no query-history update.
        codec_updated = any(not torch.equal(old, new) for old, new in codec_proposals)
        address_updated = (
            address_proposal is not None and not torch.equal(self.address_weight, address_proposal)
        )
        for parameter, candidate in codec_proposals:
            parameter.copy_(candidate)
        if address_proposal is not None:
            self.address_weight.copy_(address_proposal)
        for group in ("codec", "address"):
            work[f"{group}_update_group_offers"] = 1
        work["enabled_codec_update_groups"] = int(bool(codec_proposals))
        work["enabled_address_update_groups"] = int(address_proposal is not None)
        return SequenceLessonFeedback(
            self.memory.restore_state(state.snapshot()), len(targets), total_loss,
            total_loss / len(targets), codec_updated, address_updated,
            tuple(steps), work, event.log_prior.clone(), log_belief.clone(), address_error,
        )

    def snapshot(self, state: TapeState) -> dict:
        self._check_parameters()
        self._check_state(state, populated=False)
        return {"model": copy.deepcopy(self.state_dict()), "memory": state.snapshot()}

    def load_state_dict(self, state_dict: Mapping, strict: bool = True, assign: bool = False):
        if not strict or assign or not isinstance(state_dict, Mapping):
            raise ValueError("sequence checkpoints require strict in-place loading")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("sequence checkpoint fields mismatch")
        for name, prototype in expected.items():
            value = state_dict[name]
            if isinstance(prototype, Tensor):
                if (
                    not isinstance(value, Tensor) or value.shape != prototype.shape
                    or value.dtype != prototype.dtype or value.device != prototype.device
                    or value.requires_grad or value.grad_fn is not None
                ):
                    raise ValueError("sequence checkpoint tensor contract mismatch")
                _finite(value, "checkpoint tensor")
                if name == "input_codes" and not torch.equal(value, prototype):
                    raise ValueError("checkpoint changes fixed input codes")
            elif value != prototype:
                raise ValueError("sequence checkpoint configuration mismatch")
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self._check_parameters()
        return result

    def restore_snapshot(self, snapshot: dict) -> TapeState:
        if not isinstance(snapshot, dict) or set(snapshot) != {"model", "memory"}:
            raise ValueError("snapshot must contain model and memory")
        state = self.memory.restore_state(snapshot["memory"])
        self._check_state(state, populated=False)
        self.load_state_dict(snapshot["model"])
        return state

    def accounting(self) -> dict:
        self._check_parameters()
        return {
            "tape": self.memory.accounting(self.memory.initial_state()),
            "query_lag_neurons": 1024, "normalized_query_neurons": 1024,
            "query_lag_forward_connections": 31 * 32,
            "query_population_squared_energy": 32,
            "address_projection_neurons": 1024,
            "allocated_address_competition_neurons": 128,
            "shared_codec_input_neurons": 257, "shared_codec_output_neurons": 257,
            "address_slow_synapses": 1024 * 1024,
            "codec_slow_synapses_including_bias": 66306, "fixed_input_code_values": 8256,
            "model_tensor_bytes": sum(
                v.numel() * v.element_size() for v in self.state_dict().values()
                if isinstance(v, Tensor)
            ),
            "maximum_saved_codec_probabilities_bytes": 128 * 257 * 8,
            "maximum_conditional_step_cache_bytes": {
                "sensory_float32": 128 * 257 * 4,
                "logits_float64": 128 * 257 * 8,
                "log_probabilities_float64": 128 * 257 * 8,
                "probabilities_float64": 128 * 257 * 8,
                "total": 128 * 257 * (4 + 3 * 8),
            },
            "maximum_log_belief_bytes": 128 * 8,
            "codec_weight_error_bytes": 257 * 257 * 8, "codec_bias_error_bytes": 257 * 8,
            "maximum_current_codec_error_bytes": 128 * 257 * 8,
            "maximum_scaled_codec_column_error_bytes": 128 * 257 * 8,
            "maximum_responsibility_bytes": 128 * 8,
            "query_lag_bytes": 1024 * 4, "normalized_query_activity_bytes": 1024 * 4,
            "query_lag_shift_copy_bytes": 31 * 32 * 4,
            "query_activity_cast_bytes": 1024 * 8,
            "maximum_address_key_cast_bytes": 128 * 1024 * 8,
            "projection_signal_bytes": 1024 * 8,
            "address_local_contribution_float64_bytes": 1024 * 1024 * 8,
            "address_local_contribution_float32_bytes": 1024 * 1024 * 4,
            "address_parameter_proposal_bytes": 1024 * 1024 * 4,
            "transient_scope": (
                "named tensor payloads, not allocator/peak bounds; old/new conditional cache "
                "may overlap across iterations; parameter proposals, returned decision receipts, "
                "encoded query, validation temporaries and snapshots also contribute; "
                "the process-tree watchdog measures actual peak"
            ),
            "codec_clock": "one shared circuit time-multiplexed over occupied branches",
            "retained_query_or_teacher_history": False,
        }
