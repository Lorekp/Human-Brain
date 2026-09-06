"""Fast synapses and local rate-cell learning; no tokenizer or input cache.

A learned controller writes M += beta * (v - M k) k^T with unit-L2 keys,
bounded values and beta in (0, 1). The differentiable reference can train its
controller/maps with offline gradients; a write needs no future query or answer.
The local path uses cell self-path eligibility and query-addressed writer/map
tags, with explicitly supplied neuron-level feedback. It omits indirect paths
and is not the full memory derivative. This is a small delta-rule mechanism,
not established semantic compression or lossless storage.
Interference remains possible between nonorthogonal keys.
Unit-L2 normalization is our adaptation, not the component-sum normalization
used by the cited fast-weight programmer recipe.

Live differentiable states retain a temporary autograd graph during training.
Use the matching snapshot/restore methods before discarding source inputs.
Purge writer activity for source-absent reads. Local snapshots also retain
source-derived eligibility, which is used only for authorized feedback and must
be counted. This module cannot emit bytes or authenticate source provenance.

Reference state contains V*K + H scalars (16,640 bytes at default float32
dimensions). Tagged local state adds O(P*K) credit for P writer/map parameters
and the live cell traces: 3,777,280 bytes per stream at D32/H64/K64/V64, excluding
slow weights, read events and scratch copies. Aggregate credit costs O(P).
Offline reference autograd storage grows with sequence length.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _all_finite(value: Tensor) -> bool:
    """Exact predicate through a temporary CPU view; never mutate/retain it.

    NumPy avoids Torch's finite-mask overhead on the supported dense CPU types.
    Detached conversion shares storage, including supported noncontiguous views.
    Other devices/layouts/dtypes and unsupported view bits retain the Torch path.
    """
    if (
        value.device.type == "cpu"
        and value.layout == torch.strided
        and value.dtype in {torch.float32, torch.float64}
    ):
        try:
            array = value.detach().numpy()
        except (TypeError, RuntimeError):
            pass
        else:
            array.setflags(write=False)
            return bool(np.isfinite(array).all())
    return bool(torch.isfinite(value).all())


def _size(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("dimensions must be positive integers")
    return value


@dataclass(frozen=True)
class RateCellLearningState:
    """Caller-owned activity and local eligibility; no input history or graph.

    Per stream this costs H + H*D + H*H + H scalars for a cell; bypass omits
    the recurrent array. These traces are retained model state if feedback is
    delayed beyond source removal, and must be counted separately from memory.
    """

    activity: Tensor
    input_eligibility: Tensor
    recurrent_eligibility: Tensor | None
    bias_eligibility: Tensor

    def snapshot(self) -> dict[str, Tensor | None]:
        return {
            name: None if value is None else value.detach().clone()
            for name, value in vars(self).items()
        }


class StreamingRateCell(nn.Module):
    """One leaky tanh step, or its instantaneous producer bypass.

    Cell: h' = h + .5 * (tanh(W*x + U*h + b) - h), U initialized to .9 I.
    Bypass: h' = tanh(W*x + b); it has no recurrent parameter. Input weights
    use Xavier uniform, biases zero. A private CPU generator preserves the
    caller's RNG; construction is on CPU, then normal nn.Module.to() applies.
    No states are retained in the module. There is no implicit normalization.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, kind: str = "cell", seed: int = 0):
        super().__init__()
        self.input_size, self.hidden_size = _size(input_size), _size(hidden_size)
        if kind not in {"cell", "bypass"}:
            raise ValueError("kind must be cell or bypass")
        self.kind, self.seed = kind, seed
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size, device="cpu"))
        self.bias = nn.Parameter(torch.zeros(hidden_size, device="cpu"))
        nn.init.xavier_uniform_(self.weight, generator=generator)
        if kind == "cell":
            self.recurrent = nn.Parameter(0.9 * torch.eye(hidden_size, device="cpu"))

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> Tensor:
        return torch.zeros(
            _size(batch_size),
            self.hidden_size,
            device=self.weight.device if device is None else device,
            dtype=self.weight.dtype if dtype is None else dtype,
        )

    def forward(self, inputs: Tensor, state: Tensor | None = None) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError("inputs must have shape [batch, input_size]")
        if state is None:
            state = self.initial_state(inputs.shape[0])
        if state.shape != (inputs.shape[0], self.hidden_size):
            raise ValueError("state must have shape [batch, hidden_size]")
        return self._step(inputs, state)[0]

    def _step(self, inputs: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """Shared forward arithmetic for differentiable and local-only paths."""
        drive = F.linear(inputs, self.weight, self.bias)
        if self.kind == "bypass":
            activation = torch.tanh(drive)
            return activation, activation
        activation = torch.tanh(drive + F.linear(state, self.recurrent))
        return state + 0.5 * (activation - state), activation

    def initial_learning_state(self, batch_size: int) -> RateCellLearningState:
        activity = self.initial_state(batch_size)
        return RateCellLearningState(
            activity,
            activity.new_zeros(batch_size, self.hidden_size, self.input_size),
            activity.new_zeros(batch_size, self.hidden_size, self.hidden_size)
            if self.kind == "cell"
            else None,
            torch.zeros_like(activity),
        )

    def _check_learning_state(self, state: RateCellLearningState) -> int:
        if not isinstance(state, RateCellLearningState) or state.activity.ndim != 2:
            raise ValueError("state must be a RateCellLearningState")
        batch = _size(state.activity.shape[0])
        shapes = {
            "activity": (batch, self.hidden_size),
            "input_eligibility": (batch, self.hidden_size, self.input_size),
            "bias_eligibility": (batch, self.hidden_size),
        }
        if self.kind == "cell":
            shapes["recurrent_eligibility"] = (batch, self.hidden_size, self.hidden_size)
        elif state.recurrent_eligibility is not None:
            raise ValueError("bypass must not contain recurrent eligibility")
        for name, shape in shapes.items():
            value = getattr(state, name)
            if not isinstance(value, Tensor) or value.shape != shape:
                raise ValueError(f"invalid {name} shape")
            if value.dtype != self.weight.dtype or value.device != self.weight.device:
                raise ValueError("learning state dtype/device must match parameters")
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError("local learning state must not retain autograd history")
            if not _all_finite(value):
                raise ValueError("learning state must be finite")
        return batch

    def _active_mask(self, mask: Tensor | None, batch: int) -> Tensor:
        if mask is None:
            return torch.ones(batch, dtype=torch.bool, device=self.weight.device)
        if mask.shape != (batch,) or mask.dtype != torch.bool or mask.device != self.weight.device:
            raise ValueError("mask must be boolean [batch] on the parameter device")
        return mask

    def restore_learning_state(self, snapshot: dict[str, Tensor | None]) -> RateCellLearningState:
        if set(snapshot) != {
            "activity",
            "input_eligibility",
            "recurrent_eligibility",
            "bias_eligibility",
        }:
            raise ValueError("invalid local learning snapshot fields")
        state = RateCellLearningState(**snapshot)
        self._check_learning_state(state)
        return RateCellLearningState(**state.snapshot())

    def reset_eligibility(
        self, state: RateCellLearningState, *, preserve_activity: bool = True
    ) -> RateCellLearningState:
        """Explicitly end credit retention; optionally reset activation as well."""
        batch = self._check_learning_state(state)
        empty = self.initial_learning_state(batch)
        return RateCellLearningState(
            state.activity.detach().clone() if preserve_activity else empty.activity,
            empty.input_eligibility,
            empty.recurrent_eligibility,
            empty.bias_eligibility,
        )

    @torch.no_grad()
    def local_step(
        self,
        inputs: Tensor,
        state: RateCellLearningState | None = None,
        *,
        mask: Tensor | None = None,
    ) -> RateCellLearningState:
        """Advance forward activity and approximate, synapse-local self-path traces.

        For cell j, d=.5*(1-tanh(s_j)^2), carry=.5+d*U_jj, and
        e_ji'=carry*e_ji+d*pre_i (input, previous activity, or one).
        Off-diagonal recurrent paths are deliberately omitted: these are NOT
        full recurrent gradients. Bypass uses instantaneous (1-tanh(s)^2)*pre
        with no trace carry. Parameter changes do not retroactively recompute
        old traces. Padding preserves every state tensor and ignores input NaNs.
        """
        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError("inputs must have shape [batch, input_size]")
        if inputs.dtype != self.weight.dtype or inputs.device != self.weight.device:
            raise ValueError("inputs dtype/device must match parameters")
        if state is None:
            state = self.initial_learning_state(inputs.shape[0])
        batch = self._check_learning_state(state)
        if inputs.shape[0] != batch:
            raise ValueError("inputs and learning state batch must match")
        active = self._active_mask(mask, batch)
        safe = torch.where(active[:, None], inputs, 0)
        if not _all_finite(safe):
            raise ValueError("active inputs must be finite")
        return self._local_step_checked_inputs(safe, state, active)

    def _local_step_checked_inputs(
        self, safe: Tensor, state: RateCellLearningState, active: Tensor
    ) -> RateCellLearningState:
        """No-grad caller validates inputs/state/mask; every output stays checked.

        A memory write checks its entire input sequence and initial state once.
        Each returned cell state is checked here before becoming the next input,
        so the loop need not repeat the public entry checks for every byte.
        """
        activity, activation = self._step(safe, state.activity)
        derivative = 1 - activation.square()
        if self.kind == "cell":
            derivative = 0.5 * derivative
            carry = 0.5 + derivative * self.recurrent.diagonal()
            recurrent = (
                carry[:, :, None] * state.recurrent_eligibility
                + derivative[:, :, None] * state.activity[:, None, :]
            )
            recurrent = torch.where(active[:, None, None], recurrent, state.recurrent_eligibility)
        else:
            carry = torch.zeros_like(derivative)
            recurrent = None
        result = RateCellLearningState(
            torch.where(active[:, None], activity, state.activity),
            torch.where(
                active[:, None, None],
                carry[:, :, None] * state.input_eligibility
                + derivative[:, :, None] * safe[:, None, :],
                state.input_eligibility,
            ),
            recurrent,
            torch.where(
                active[:, None], carry * state.bias_eligibility + derivative, state.bias_eligibility
            ),
        )
        self._check_learning_state(result)
        return result

    @torch.no_grad()
    def apply_learning_signal(
        self,
        state: RateCellLearningState,
        signal: Tensor,
        learning_rate: float,
        *,
        mask: Tensor | None = None,
        learning_enabled: bool = True,
    ) -> bool:
        """Mutate owned trainable parameters by -eta*mean_active(L_j*e_ji).

        L is one externally declared scalar per neuron, NOT a per-synapse
        gradient. This method does not compute it. Frozen requires_grad=False
        parameters, disabled learning, and an empty active batch never change.
        State/traces are unchanged; caller explicitly resets or continues them.
        Return whether any parameter actually changed; no optimizer or .grad use.
        """
        batch = self._check_learning_state(state)
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if not isinstance(learning_enabled, bool):
            raise ValueError("learning_enabled must be boolean")
        if signal.shape != (batch, self.hidden_size):
            raise ValueError("signal must have shape [batch, hidden_size]")
        if signal.device != self.weight.device or signal.dtype != self.weight.dtype:
            raise ValueError("signal dtype/device must match parameters")
        active = self._active_mask(mask, batch)
        safe = torch.where(active[:, None], signal, 0)
        if not _all_finite(safe):
            raise ValueError("active signals must be finite")
        count = int(active.sum())
        if not learning_enabled or learning_rate == 0 or count == 0:
            return False
        pairs = [(self.weight, state.input_eligibility), (self.bias, state.bias_eligibility)]
        if self.kind == "cell":
            pairs.append((self.recurrent, state.recurrent_eligibility))
        # Check all proposals before mutation, so overflow cannot partially update.
        proposals = []
        for parameter, trace in pairs:
            if parameter.requires_grad:
                factor = safe if trace.ndim == 2 else safe[:, :, None]
                change = (factor * trace).sum(dim=0) * (learning_rate / count)
                candidate = parameter - change
                if not _all_finite(candidate):
                    raise ValueError("local update produced nonfinite parameters")
                proposals.append((parameter, candidate))
        changed = any(not torch.equal(parameter, candidate) for parameter, candidate in proposals)
        for parameter, candidate in proposals:
            parameter.copy_(candidate)
        return changed


@dataclass(frozen=True)
class PlasticMemoryState:
    """Explicit tensors only; the owning caller controls their lifetime."""

    synapses: Tensor
    writer: Tensor

    def snapshot(self) -> dict[str, Tensor]:
        """Independent persistent tensors, with no retained training/input graph."""
        return {
            "synapses": self.synapses.detach().clone(),
            "writer": self.writer.detach().clone(),
        }


@dataclass(frozen=True)
class LocalMemoryState:
    """Fast synapses plus fixed-size source-derived local credit, never inputs.

    Tagged tensors have [batch, *parameter_shape, key_size]; aggregate tensors
    omit the final axis. They are teaching state, never inputs to read/decoding.
    """

    synapses: Tensor
    cell: RateCellLearningState
    credit: dict[str, Tensor]
    credit_mode: str

    def snapshot(self) -> dict:
        return {
            "synapses": self.synapses.detach().clone(),
            "cell": self.cell.snapshot(),
            "credit": {name: value.detach().clone() for name, value in self.credit.items()},
            "credit_mode": self.credit_mode,
        }


@dataclass(frozen=True)
class LocalMemoryRead:
    """One read with raw query-map presynaptic traces and unclamped drive norm.

    Normalization belongs in the neuron learning signal, once. Writer-map credit
    retains its separate diagonal approximation. Caller owns read retirement.
    """

    value: Tensor
    key: Tensor
    query_norm: Tensor
    query_weight_eligibility: Tensor
    query_bias_eligibility: Tensor
    mask: Tensor

    def snapshot(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in vars(self).items()}


class PlasticMemory(nn.Module):
    """Learned streaming writer and query map around explicit fast synapses.

    Keys/queries are linear maps normalized by max(L2 norm, 1e-8); a zero map
    produces a zero key/read. Values are tanh(linear(writer)); write strength is
    sigmoid(linear(writer)). Projections have Xavier-uniform weights and zero
    biases, using a separate seed+1 stream in key/value/strength/query order.
    The controller uses seed unchanged. Cell/bypass shared initial arrays match.

    read() uses only synapses, never the final writer activation. Disabling writes
    still advances the controller, but changes no synapses. There is no automatic
    per-document reset or memory decay. Continuing writes uses the supplied state;
    purge_writer() explicitly separates document-local carry from memory.

    Restore learned parameters with ordinary state_dict/load_state_dict using
    the same constructor configuration. Restore per-stream state separately.
    Optimizer state and parameter freezing remain caller-owned policy decisions.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        key_size: int = 64,
        value_size: int = 64,
        kind: str = "cell",
        seed: int = 0,
    ):
        super().__init__()
        self.input_size, self.hidden_size = _size(input_size), _size(hidden_size)
        self.key_size, self.value_size = _size(key_size), _size(value_size)
        self.kind, self.seed = kind, seed
        self.controller = StreamingRateCell(input_size, hidden_size, kind, seed)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        # Direct tensors avoid implicit random draws by nn.Linear constructors.
        for name, output_size, inputs in (
            ("key", key_size, hidden_size),
            ("value", value_size, hidden_size),
            ("strength", 1, hidden_size),
            ("query", key_size, input_size),
        ):
            weight = nn.Parameter(torch.empty(output_size, inputs, device="cpu"))
            nn.init.xavier_uniform_(weight, generator=generator)
            self.register_parameter(name + "_weight", weight)
            self.register_parameter(
                name + "_bias", nn.Parameter(torch.zeros(output_size, device="cpu"))
            )

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> PlasticMemoryState:
        writer = self.controller.initial_state(batch_size, device=device, dtype=dtype)
        return PlasticMemoryState(
            writer.new_zeros(batch_size, self.value_size, self.key_size), writer
        )

    def _check_state(self, state: PlasticMemoryState, batch_size: int) -> None:
        if not isinstance(state, PlasticMemoryState):
            raise ValueError("state must be PlasticMemoryState")
        if state.synapses.shape != (batch_size, self.value_size, self.key_size):
            raise ValueError("synapse shape does not match memory configuration")
        if state.writer.shape != (batch_size, self.hidden_size):
            raise ValueError("writer shape does not match memory configuration")
        for tensor in (state.synapses, state.writer):
            if tensor.dtype != self.key_weight.dtype or tensor.device != self.key_weight.device:
                raise ValueError("state dtype/device must match memory parameters")
            if not _all_finite(tensor):
                raise ValueError("state must be finite")

    def restore_state(self, snapshot: dict[str, Tensor]) -> PlasticMemoryState:
        if set(snapshot) != {"synapses", "writer"}:
            raise ValueError("snapshot must contain only synapses and writer")
        if not all(isinstance(tensor, Tensor) for tensor in snapshot.values()):
            raise ValueError("snapshot entries must be tensors")
        state = PlasticMemoryState(snapshot["synapses"], snapshot["writer"])
        if state.writer.ndim != 2:
            raise ValueError("snapshot writer must have batch/hidden dimensions")
        self._check_state(state, state.writer.shape[0])
        return PlasticMemoryState(**state.snapshot())

    def purge_writer(self, state: PlasticMemoryState) -> PlasticMemoryState:
        self._check_state(state, state.writer.shape[0])
        return PlasticMemoryState(state.synapses.clone(), torch.zeros_like(state.writer))

    def write(
        self,
        values: Tensor,
        mask: Tensor,
        state: PlasticMemoryState | None = None,
        *,
        writes_enabled: bool = True,
    ) -> PlasticMemoryState:
        if values.ndim != 3 or values.shape[2] != self.input_size:
            raise ValueError("values must have shape [batch, tokens, input_size]")
        if mask.dtype != torch.bool or mask.shape != values.shape[:2]:
            raise ValueError("mask must be boolean with matching batch/token dimensions")
        if not isinstance(writes_enabled, bool):
            raise ValueError("writes_enabled must be boolean")
        if values.dtype != self.key_weight.dtype or values.device != self.key_weight.device:
            raise ValueError("input dtype/device must match memory parameters")
        if mask.device != values.device:
            raise ValueError("mask must share the input device")
        clean = values.masked_fill(~mask[..., None], 0)
        if not _all_finite(clean):
            raise ValueError("valid inputs must be finite")
        if state is None:
            state = self.initial_state(values.shape[0])
        self._check_state(state, values.shape[0])
        synapses, writer = state.synapses, state.writer
        for position in range(values.shape[1]):
            active = mask[:, position, None]
            candidate = self.controller(clean[:, position], writer)
            writer = torch.where(active, candidate, writer)
            if writes_enabled:
                key = F.normalize(
                    F.linear(writer, self.key_weight, self.key_bias), dim=-1, eps=1e-8
                )
                value = torch.tanh(F.linear(writer, self.value_weight, self.value_bias))
                strength = torch.sigmoid(F.linear(writer, self.strength_weight, self.strength_bias))
                predicted = torch.bmm(synapses, key.unsqueeze(-1)).squeeze(-1)
                correction = (strength * (value - predicted)).unsqueeze(-1) * key.unsqueeze(1)
                synapses = torch.where(active.unsqueeze(-1), synapses + correction, synapses)
        return PlasticMemoryState(synapses, writer)

    def read(self, query: Tensor, state: PlasticMemoryState) -> Tensor:
        if query.ndim != 2 or query.shape[1] != self.input_size:
            raise ValueError("query must have shape [batch, input_size]")
        if query.dtype != self.query_weight.dtype or query.device != self.query_weight.device:
            raise ValueError("query dtype/device must match memory parameters")
        if not _all_finite(query):
            raise ValueError("query must be finite")
        self._check_state(state, query.shape[0])
        key = F.normalize(F.linear(query, self.query_weight, self.query_bias), dim=-1, eps=1e-8)
        return torch.bmm(state.synapses, key.unsqueeze(-1)).squeeze(-1)

    def _writer_parameters(self) -> dict[str, Tensor]:
        return {name: p for name, p in self.named_parameters() if not name.startswith("query_")}

    def initial_local_state(
        self, batch_size: int, *, credit_mode: str = "tagged"
    ) -> LocalMemoryState:
        if credit_mode not in {"tagged", "aggregate"}:
            raise ValueError("credit_mode must be tagged or aggregate")
        cell = self.controller.initial_learning_state(batch_size)
        extra = (self.key_size,) if credit_mode == "tagged" else ()
        return LocalMemoryState(
            cell.activity.new_zeros(batch_size, self.value_size, self.key_size),
            cell,
            {
                name: p.new_zeros((batch_size, *p.shape, *extra))
                for name, p in self._writer_parameters().items()
            },
            credit_mode,
        )

    def _check_local_state(self, state: LocalMemoryState) -> int:
        if not isinstance(state, LocalMemoryState) or state.credit_mode not in {
            "tagged",
            "aggregate",
        }:
            raise ValueError("invalid local memory state")
        batch = self.controller._check_learning_state(state.cell)
        self._check_state(PlasticMemoryState(state.synapses, state.cell.activity), batch)
        parameters = self._writer_parameters()
        if set(state.credit) != set(parameters):
            raise ValueError("credit must match writer/map parameter names")
        extra = (self.key_size,) if state.credit_mode == "tagged" else ()
        for name, p in parameters.items():
            tensor = state.credit[name]
            if tensor.shape != (batch, *p.shape, *extra):
                raise ValueError("credit shape mismatch")
            self._check_local_tensor(tensor)
        self._check_local_tensor(state.synapses)
        return batch

    def _check_local_tensor(self, value: Tensor) -> None:
        if (
            value.dtype != self.key_weight.dtype
            or value.device != self.key_weight.device
            or value.requires_grad
            or value.grad_fn is not None
            or not _all_finite(value)
        ):
            raise ValueError("local tensors must be finite, detached and match parameters")

    def restore_local_state(self, snapshot: dict) -> LocalMemoryState:
        if set(snapshot) != {"synapses", "cell", "credit", "credit_mode"}:
            raise ValueError("invalid local memory snapshot fields")
        state = LocalMemoryState(
            snapshot["synapses"],
            self.controller.restore_learning_state(snapshot["cell"]),
            snapshot["credit"],
            snapshot["credit_mode"],
        )
        self._check_local_state(state)
        return LocalMemoryState(
            state.synapses.detach().clone(),
            state.cell,
            {name: value.detach().clone() for name, value in state.credit.items()},
            state.credit_mode,
        )

    def purge_local_writer(self, state: LocalMemoryState) -> LocalMemoryState:
        """Remove activity and untagged cell carry; retain M and delayed credit."""
        self._check_local_state(state)
        return LocalMemoryState(
            state.synapses.detach().clone(),
            self.controller.reset_eligibility(state.cell, preserve_activity=False),
            {name: value.detach().clone() for name, value in state.credit.items()},
            state.credit_mode,
        )

    def retire_local_traces(self, state: LocalMemoryState) -> LocalMemoryState:
        """End lesson credit without rebuilding M after slow-parameter updates."""
        self._check_local_state(state)
        return LocalMemoryState(
            state.synapses.detach().clone(),
            self.controller.reset_eligibility(state.cell),
            {name: torch.zeros_like(value) for name, value in state.credit.items()},
            state.credit_mode,
        )

    @staticmethod
    def _normalized_local_map(drive: Tensor) -> tuple[Tensor, Tensor]:
        norm = drive.norm(dim=-1, keepdim=True)
        key = drive / norm.clamp_min(1e-8)
        # Only the normalization Jacobian's diagonal is retained; no cross-unit credit.
        gain = torch.where(
            norm > 1e-8, (1 - key.square()) / norm.clamp_min(1e-8), torch.full_like(key, 1e8)
        )
        return key, gain

    @torch.no_grad()
    def write_local(
        self,
        values: Tensor,
        mask: Tensor,
        state: LocalMemoryState | None = None,
        *,
        writes_enabled: bool = True,
    ) -> LocalMemoryState:
        """Answer-free byte steps, fast writes and local credit; no slow updates.

        Tagged: T' = T + beta*(e - T dot k)*k, all RHS from old T.
        Aggregate control: E' = (1-beta)*E + beta*e, without key/query addressing.
        Map e uses local diagonal derivatives; controller e uses self-path traces.
        Keys/strength derivatives of memory overwrite and cross-neuron paths are
        omitted. This is a declared heuristic, not the full memory derivative.
        Caller holds slow parameters fixed through the lesson. Disabled writes
        advance cell state but neither synapses nor credit. Padding changes none.
        """
        if values.ndim != 3 or values.shape[-1] != self.input_size:
            raise ValueError("values must be [batch, tokens, input_size]")
        if (
            mask.dtype != torch.bool
            or mask.shape != values.shape[:2]
            or mask.device != values.device
        ):
            raise ValueError("mask must be boolean with matching batch/token dimensions/device")
        if type(writes_enabled) is not bool:
            raise ValueError("writes_enabled must be boolean")
        clean = torch.where(mask[:, :, None], values, 0)
        if (
            clean.dtype != self.key_weight.dtype
            or clean.device != self.key_weight.device
            or not _all_finite(clean)
        ):
            raise ValueError("active inputs must be finite and match parameters")
        if state is None:
            state = self.initial_local_state(values.shape[0])
        if self._check_local_state(state) != values.shape[0]:
            raise ValueError("state batch mismatch")
        synapses = state.synapses.clone()
        credit = {name: value.clone() for name, value in state.credit.items()}
        cell = RateCellLearningState(**state.cell.snapshot())
        batch = values.shape[0]
        # Reuse method-private storage; caller-owned credit remains untouched.
        # Keep the original elementwise/reduction order rather than changing to
        # a matrix-vector kernel with a different floating-point reduction.
        tag_scratch = {}
        if writes_enabled and state.credit_mode == "tagged" and values.shape[1]:
            for name, old in credit.items():
                flat = old.reshape(batch, -1, self.key_size)
                # reshape can copy a noncontiguous restored tensor. Publish the
                # owned flat storage, not an earlier clone that it no longer views.
                credit[name] = flat.reshape_as(old)
                tag_scratch[name] = (
                    flat,
                    torch.empty_like(flat),
                    flat.new_empty(batch, flat.shape[1]),
                )
        for position in range(values.shape[1]):
            active = mask[:, position]
            cell = self.controller._local_step_checked_inputs(clean[:, position], cell, active)
            if not writes_enabled:
                continue
            h = cell.activity
            key, key_gain = self._normalized_local_map(F.linear(h, self.key_weight, self.key_bias))
            value = torch.tanh(F.linear(h, self.value_weight, self.value_bias))
            strength = torch.sigmoid(F.linear(h, self.strength_weight, self.strength_bias))
            beta = strength * active[:, None]
            predicted = torch.bmm(synapses, key[:, :, None]).squeeze(-1)
            synapses = synapses + (beta * (value - predicted))[:, :, None] * key[:, None, :]
            traces = {
                "controller.weight": cell.input_eligibility,
                "controller.bias": cell.bias_eligibility,
            }
            if self.kind == "cell":
                traces["controller.recurrent"] = cell.recurrent_eligibility
            for name, gain in (
                ("key", key_gain),
                ("value", 1 - value.square()),
                ("strength", strength * (1 - strength)),
            ):
                traces[name + "_weight"] = gain[:, :, None] * h[:, None, :]
                traces[name + "_bias"] = gain
            for name, trace in traces.items():
                e = trace.reshape(batch, -1)
                if state.credit_mode == "tagged":
                    flat, product, correction = tag_scratch[name]
                    torch.mul(flat, key[:, None, :], out=product)
                    torch.sum(product, dim=-1, out=correction)
                    torch.sub(e, correction, out=correction)
                    torch.mul(beta, correction, out=correction)
                    torch.mul(correction[:, :, None], key[:, None, :], out=product)
                    flat.add_(product)
                else:
                    old = credit[name]
                    flat = old.reshape(batch, -1)
                    updated = flat + beta * (e - flat)
                    credit[name] = updated.reshape_as(old)
        result = LocalMemoryState(synapses, cell, credit, state.credit_mode)
        self._check_local_state(result)
        return result

    @torch.no_grad()
    def read_local(
        self, query: Tensor, state: LocalMemoryState, *, mask: Tensor | None = None
    ) -> LocalMemoryRead:
        batch = self._check_local_state(state)
        if query.shape != (batch, self.input_size):
            raise ValueError("query must be [batch, input_size]")
        active = self.controller._active_mask(mask, batch)
        safe = torch.where(active[:, None], query, 0)
        if (
            safe.dtype != self.query_weight.dtype
            or safe.device != self.query_weight.device
            or not _all_finite(safe)
        ):
            raise ValueError("active query must be finite and match parameters")
        drive = F.linear(safe, self.query_weight, self.query_bias)
        norm = drive.norm(dim=-1, keepdim=True)
        key = drive / norm.clamp_min(1e-8)
        key = torch.where(active[:, None], key, 0)
        event = LocalMemoryRead(
            torch.bmm(state.synapses, key[:, :, None]).squeeze(-1),
            key,
            norm,
            safe[:, None, :].expand(batch, self.key_size, self.input_size).clone(),
            active[:, None].expand(batch, self.key_size).to(dtype=safe.dtype),
            active.detach().clone(),
        )
        self._check_local_read(event, batch)
        return event

    def _check_local_read(self, event: LocalMemoryRead, batch: int) -> None:
        shapes = {
            "value": (batch, self.value_size),
            "key": (batch, self.key_size),
            "query_norm": (batch, 1),
            "query_weight_eligibility": (batch, self.key_size, self.input_size),
            "query_bias_eligibility": (batch, self.key_size),
        }
        for name, shape in shapes.items():
            value = getattr(event, name)
            if value.shape != shape:
                raise ValueError("local read shape mismatch")
            self._check_local_tensor(value)
        self.controller._active_mask(event.mask, batch)
        if bool((event.query_norm < 0).any()):
            raise ValueError("query norm must be nonnegative")

    def restore_local_read(self, snapshot: dict) -> LocalMemoryRead:
        if set(snapshot) != {
            "value",
            "key",
            "query_norm",
            "query_weight_eligibility",
            "query_bias_eligibility",
            "mask",
        }:
            raise ValueError("invalid local read snapshot fields")
        event = LocalMemoryRead(**snapshot)
        self._check_local_read(event, event.value.shape[0])
        return LocalMemoryRead(**event.snapshot())

    @torch.no_grad()
    def read_learning_signals(
        self, state: LocalMemoryState, event: LocalMemoryRead, read_signal: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Current read-path neuron errors, before any query/map mutation.

        With g at read neurons, e_q=M^T g. The normalization error pool supplies
        e_a=(e_q-q*(q dot e_q))/norm above epsilon and e_q/epsilon otherwise.
        At equality the clamped branch is the declared subgradient. Return e_a
        and Q_pre^T e_a; query-map synapses use raw preactivity, not another gain.
        This instantaneous weight transport does not differentiate cell history.
        """
        batch = self._check_local_state(state)
        self._check_local_read(event, batch)
        if read_signal.shape != (batch, self.value_size):
            raise ValueError("read signal must be per-neuron [batch, value_size]")
        safe = torch.where(event.mask[:, None], read_signal, 0)
        self._check_local_tensor(safe)
        key_signal = torch.bmm(state.synapses.transpose(1, 2), safe[:, :, None]).squeeze(-1)
        self._check_local_tensor(key_signal)
        pooled = (event.key * key_signal).sum(-1, keepdim=True)
        query_signal = torch.where(
            event.query_norm > 1e-8, key_signal - event.key * pooled, key_signal
        ) / event.query_norm.clamp_min(1e-8)
        self._check_local_tensor(query_signal)
        indirect_signal = F.linear(query_signal, self.query_weight.T)
        self._check_local_tensor(indirect_signal)
        return query_signal, indirect_signal

    @torch.no_grad()
    def apply_local_feedback(
        self,
        state: LocalMemoryState,
        event: LocalMemoryRead,
        signals: dict[str, Tensor],
        learning_rate: float,
        *,
        learning_enabled: bool = True,
    ) -> bool:
        """Apply -eta*mean_active(L_j*retrieved_e_ji) once; never rebuild memory.

        Caller supplies five neuron-level broadcasts, not per-synapse updates.
        Query-map credit is current local e, writer/map credit is T dot query key
        (or aggregate E). Frozen parameters and disabled learning remain unchanged.
        Events are not consumed automatically; caller owns lesson authority and
        retires traces after its one upstream update. No graph or optimizer is used.
        """
        batch = self._check_local_state(state)
        self._check_local_read(event, batch)
        if (
            not math.isfinite(learning_rate)
            or learning_rate < 0
            or type(learning_enabled) is not bool
        ):
            raise ValueError("invalid learning rate/enabled flag")
        sizes = {
            "controller": self.hidden_size,
            "key": self.key_size,
            "value": self.value_size,
            "strength": 1,
            "query": self.key_size,
        }
        if set(signals) != set(sizes):
            raise ValueError("provide exactly the five neuron learning signals")
        clean = {}
        for name, size in sizes.items():
            signal = signals[name]
            if signal.shape != (batch, size):
                raise ValueError("signal must be per-neuron [batch, group_size]")
            clean[name] = torch.where(event.mask[:, None], signal, 0)
            self._check_local_tensor(clean[name])
        count = int(event.mask.sum())
        if not learning_enabled or learning_rate == 0 or count == 0:
            return False
        traces = {}
        for name, credit in state.credit.items():
            if state.credit_mode == "tagged":
                traces[name] = (
                    (credit.reshape(batch, -1, self.key_size) * event.key[:, None, :])
                    .sum(-1)
                    .reshape(credit.shape[:-1])
                )
            else:
                traces[name] = credit
        traces.update(
            query_weight=event.query_weight_eligibility, query_bias=event.query_bias_eligibility
        )
        proposals = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            group = "controller" if name.startswith("controller.") else name.split("_")[0]
            trace = traces[name]
            signal = clean[group] if parameter.ndim == 1 else clean[group][:, :, None]
            candidate = parameter - (learning_rate / count) * (signal * trace).sum(0)
            if not _all_finite(candidate):
                raise ValueError("nonfinite proposed local memory update")
            proposals.append((parameter, candidate))
        changed = any(not torch.equal(p, candidate) for p, candidate in proposals)
        for parameter, candidate in proposals:
            parameter.copy_(candidate)
        return changed
