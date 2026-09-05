"""Fast synapses and local rate-cell learning; no tokenizer or input cache.

A learned controller writes M += beta * (v - M k) k^T with unit-L2 keys,
bounded values and beta in (0, 1). The differentiable reference can train its
controller/maps with offline gradients; a write needs no future query or answer.
StreamingRateCell also provides graph-free local eligibility/signal updates.
Local credit through the complete memory writer/maps remains a separate task.
This is a small
delta-rule mechanism, not established semantic compression or lossless storage.
Interference remains possible between nonorthogonal keys.
Unit-L2 normalization is our adaptation, not the component-sum normalization
used by the cited fast-weight programmer recipe.

Live differentiable states retain a temporary autograd graph during training.
Use state.snapshot() and restore_state() before discarding source inputs: only
detached synapses and writer activations cross that boundary. Clear the latter
with purge_writer() for source-absent reads. This module cannot emit bytes or
authenticate provenance; those require separately accounted interfaces.

Per stream, persistent tensors contain V*K + H scalars (16,640 bytes with the
default float32 dimensions). Writes cost O(H*D + H*H + H*(K+V) + V*K) for cells;
reads cost O(K*D + V*K). Offline autograd storage grows with sequence length.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
            if not bool(torch.isfinite(value).all()):
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
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("active inputs must be finite")
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
        if not bool(torch.isfinite(safe).all()):
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
                if not bool(torch.isfinite(candidate).all()):
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
            if not bool(torch.isfinite(tensor).all()):
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
        if not bool(torch.isfinite(clean).all()):
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
        if not bool(torch.isfinite(query).all()):
            raise ValueError("query must be finite")
        self._check_state(state, query.shape[0])
        key = F.normalize(F.linear(query, self.query_weight, self.query_bias), dim=-1, eps=1e-8)
        return torch.bmm(state.synapses, key.unsqueeze(-1)).squeeze(-1)
