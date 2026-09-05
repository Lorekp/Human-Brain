"""Explicit, differentiable fast synapses; no tokenizer, answer labels or input cache.

A learned controller writes M += beta * (v - M k) k^T with unit-L2 keys,
bounded values and beta in (0, 1). Offline gradients train the controller and
maps; a document write itself needs no future query or answer. This is a small
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

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _size(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("dimensions must be positive integers")
    return value


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
        drive = F.linear(inputs, self.weight, self.bias)
        if self.kind == "bypass":
            return torch.tanh(drive)
        activation = torch.tanh(drive + F.linear(state, self.recurrent))
        return state + 0.5 * (activation - state)


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
