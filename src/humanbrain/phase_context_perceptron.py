"""Local integer-synapse reference with a supplied neuronal phase representation.

The fixed clock and generic conjunctions are gifted priors, not learned reading
or a variable-length policy. Only byte/EOS associations learn from teacher labels.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS, ByteGeneration


@dataclass(frozen=True)
class PhaseContextState:
    """Caller-owned next clock activity; no module activity or token history."""

    phase: Tensor

    @property
    def tensor_bytes(self) -> int:
        return self.phase.numel() * self.phase.element_size()

    def snapshot(self) -> dict:
        return {"phase": self.phase.detach().clone()}


@dataclass(frozen=True)
class PhaseContextStep:
    """Current presynaptic activity/scores and independently owned next phase."""

    state: PhaseContextState
    features: Tensor
    scores: Tensor

    @property
    def tensor_bytes(self) -> int:
        return self.state.tensor_bytes + sum(
            value.numel() * value.element_size() for value in (self.features, self.scores)
        )

    def snapshot(self) -> dict:
        return {
            "state": self.state.snapshot(),
            "features": self.features.detach().clone(),
            "scores": self.scores.detach().clone(),
        }


@dataclass(frozen=True)
class PhaseContextFeedback:
    """Mistake loss is 0/1, not the byte decoder's cross-entropy."""

    loss: float
    teacher_count: int
    updated: bool
    winner: int
    target: int
    phase: int


class PhaseContextPerceptron(nn.Module):
    """Fixed 96-context/three-phase/257-output CPU neuronal reference.

    Every phase connects to every context coordinate through a two-input
    threshold conjunction neuron. Output neurons receive phase and conjunction
    activity; all output synapses start at zero. Each int32 tick denotes 0.01.
    Winners use exact int64 scores and lowest-index ties. No witness separator,
    character map, answer-slot selection, bias update or input history exists.
    """

    context_size = 96
    phase_size = 3
    feature_size = 291
    tick_size = 0.01

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("output_ticks", torch.zeros(257, 291, dtype=torch.int32, device="cpu"))
        self.register_buffer(
            "clock_transition",
            torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.int32, device="cpu"),
        )
        self.register_buffer(
            "clock_start", torch.tensor([1, 0, 0], dtype=torch.int32, device="cpu")
        )
        self.register_buffer(
            "conjunction_synapses", torch.ones(3, 96, 2, dtype=torch.int32, device="cpu")
        )
        self.register_buffer(
            "conjunction_thresholds", torch.full((3, 96), 2, dtype=torch.int32, device="cpu")
        )

    def get_extra_state(self) -> dict:
        return {
            "schema": "phase-context-perceptron-v1", "context_size": 96,
            "phase_size": 3, "feature_size": 291, "outputs": 257,
            "tick_size": 0.01, "features": "phase-then-all-phase-context-conjunctions",
            "learning": "immediate-integer-teacher-minus-winner",
            "clock": "fixed-three-neuron-chain", "max_bytes": 2,
        }

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("reference checkpoint configuration mismatch")

    @staticmethod
    def _tensor(value: Tensor, shape: tuple, dtype: torch.dtype) -> None:
        if (
            not isinstance(value, Tensor) or value.shape != shape or value.dtype != dtype
            or value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None
        ):
            raise ValueError("reference tensor shape/dtype/device/history mismatch")

    def _check_fixed(self) -> None:
        for value, shape in (
            (self.output_ticks, (257, 291)), (self.clock_transition, (3, 3)),
            (self.clock_start, (3,)), (self.conjunction_synapses, (3, 96, 2)),
            (self.conjunction_thresholds, (3, 96)),
        ):
            self._tensor(value, shape, torch.int32)
        if (
            self.clock_transition.tolist() != [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
            or self.clock_start.tolist() != [1, 0, 0]
            or not bool((self.conjunction_synapses == 1).all())
            or not bool((self.conjunction_thresholds == 2).all())
        ):
            raise ValueError("reference clock and generic conjunction dynamics must remain fixed")

    def _context(self, context: Tensor) -> Tensor:
        if (
            not isinstance(context, Tensor) or context.shape != (1, 96)
            or context.device.type != "cpu" or context.dtype not in (torch.float32, torch.float64)
            or not bool(((context == 0) | (context == 1)).all())
        ):
            raise ValueError("context must be binary CPU float32/64 with shape [1, 96]")
        return context.detach().to(torch.int32)

    def _check_state(self, state: PhaseContextState) -> None:
        if not isinstance(state, PhaseContextState):
            raise ValueError("state must be a PhaseContextState")
        self._tensor(state.phase, (1, 3), torch.int32)
        if not bool(((state.phase == 0) | (state.phase == 1)).all()) or int(state.phase.sum()) > 1:
            raise ValueError("clock activity must be one-hot or exhausted")

    def initial_state(self) -> PhaseContextState:
        self._check_fixed()
        return PhaseContextState(self.clock_start.unsqueeze(0).clone())

    def _scores(self, features: Tensor) -> Tensor:
        # Binary factors cannot overflow int32 products. Sum in int64 so even
        # arbitrary legal int32 synapses have exact scores before overflow checks.
        return (self.output_ticks * features).sum(-1, dtype=torch.int64).unsqueeze(0)

    @torch.no_grad()
    def local_step(
        self, context: Tensor, input_tokens: Tensor, state: PhaseContextState | None = None
    ) -> PhaseContextStep:
        """Advance fixed neuronal dynamics; previous-byte identity supplies no feature."""
        self._check_fixed()
        binary = self._context(context)
        self._tensor(input_tokens, (1,), torch.int64)
        if int(input_tokens[0]) < 0 or int(input_tokens[0]) > BYTE_BOS:
            raise ValueError("input must be byte, EOS or BOS")
        if state is None:
            if int(input_tokens[0]) != BYTE_BOS:
                raise ValueError("a fresh clock requires BOS")
            state = self.initial_state()
        self._check_state(state)
        if int(state.phase.sum()) != 1:
            raise ValueError("three-phase clock exhausted")
        drive = (
            state.phase[:, :, None] * self.conjunction_synapses[:, :, 0]
            + binary[:, None, :] * self.conjunction_synapses[:, :, 1]
        )
        conjunction = (drive >= self.conjunction_thresholds).to(torch.int32)
        features = torch.cat((state.phase, conjunction.reshape(1, 288)), dim=-1)
        next_state = PhaseContextState(state.phase @ self.clock_transition)
        return PhaseContextStep(next_state, features, self._scores(features))

    def _check_step(self, step: PhaseContextStep) -> None:
        self._check_fixed()
        if not isinstance(step, PhaseContextStep):
            raise ValueError("step must be a PhaseContextStep")
        self._check_state(step.state)
        self._tensor(step.features, (1, 291), torch.int32)
        self._tensor(step.scores, (1, 257), torch.int64)
        phase = step.features[:, :3]
        if not bool(((step.features == 0) | (step.features == 1)).all()) or int(phase.sum()) != 1:
            raise ValueError("step features must be binary with one active phase")
        conjunction = step.features[:, 3:].reshape(1, 3, 96)
        if (
            bool((conjunction * (1 - phase[:, :, None])).any())
            or not torch.equal(step.state.phase, phase @ self.clock_transition)
        ):
            raise ValueError("step features and neuronal clock are inconsistent")
        if not torch.equal(step.scores, self._scores(step.features)):
            raise ValueError("teaching requires current integer scores; stale step rejected")

    @torch.no_grad()
    def teach_local(
        self, step: PhaseContextStep, targets: Tensor, learning_rate: float,
        *, learning_enabled: bool = True,
    ) -> PhaseContextFeedback:
        """Apply one local mistake update; labels never select or change features."""
        self._check_step(step)
        self._tensor(targets, (1,), torch.int64)
        target = int(targets[0])
        if target < 0 or target > BYTE_EOS:
            raise ValueError("target must be a byte or EOS")
        if learning_rate != self.tick_size:
            raise ValueError("reference learning_rate must equal one 0.01 tick")
        if type(learning_enabled) is not bool:
            raise ValueError("learning_enabled must be boolean")
        winner = int(step.scores.argmax(-1)[0])
        mistake = target != winner
        updated = mistake and learning_enabled
        if updated:
            pre = step.features[0].to(torch.int64)
            positive = self.output_ticks[target].to(torch.int64) + pre
            negative = self.output_ticks[winner].to(torch.int64) - pre
            bounds = torch.iinfo(torch.int32)
            if bool((positive > bounds.max).any()) or bool((negative < bounds.min).any()):
                raise ValueError("local integer synapse update would overflow int32")
            self.output_ticks[target].copy_(positive.to(torch.int32))
            self.output_ticks[winner].copy_(negative.to(torch.int32))
        return PhaseContextFeedback(
            float(mistake), 1, updated, winner, target, int(step.features[0, :3].argmax()),
        )

    @torch.no_grad()
    def generate(self, context: Tensor, max_bytes: int) -> tuple[ByteGeneration, ...]:
        """Read-only competition; only a learned EOS winner or byte cap stops output."""
        self._check_fixed()
        self._context(context)
        if type(max_bytes) is not int or not 0 <= max_bytes <= 2:
            raise ValueError("three-phase reference max_bytes must be between zero and two")
        output, state = bytearray(), None
        token = BYTE_BOS
        for position in range(max_bytes + 1):
            step = self.local_step(context, torch.tensor([token], device="cpu"), state)
            state = step.state
            token = int(step.scores.argmax(-1)[0])
            if token == BYTE_EOS:
                return (ByteGeneration(bytes(output), "eos"),)
            if position == max_bytes:
                return (ByteGeneration(bytes(output), "limit"),)
            output.append(token)
        raise AssertionError("bounded generation must have a stopping decision")

    def accounting(self) -> dict:
        """Allocated tensor payload; process memory also includes Python and scratch."""
        self._check_fixed()
        return {
            "neurons": {"supplied_context": 96, "clock": 3, "conjunction": 288, "output": 257},
            "total_neurons": 644, "learned_output_synapses": 74787,
            "fixed_allocated_synapses": 585, "fixed_nonzero_synapses": 578,
            "fixed_thresholds": 288, "clock_initial_values": 3,
            "serialized_tensor_bytes": sum(
                value.numel() * value.element_size() for value in self.buffers()
            ),
            "state_tensor_bytes": 12, "step_tensor_bytes": 3232,
            "supplied_float32_context_bytes": 384,
        }
