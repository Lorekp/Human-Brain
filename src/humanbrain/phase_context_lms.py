"""Local potential-error learning with an explicitly supplied neuronal clock.

Generic continuous phase/context gating is a gifted representation, not learned
reading. The binary development proof does not certify arbitrary memory inputs.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS, ByteGeneration
from humanbrain.plastic_memory import _all_finite


@dataclass(frozen=True)
class PhaseLMSState:
    """Independently owned next clock activity, without input history."""

    phase: Tensor

    @property
    def tensor_bytes(self) -> int:
        return self.phase.numel() * self.phase.element_size()

    def snapshot(self) -> dict:
        return {"phase": self.phase.detach().clone()}


@dataclass(frozen=True)
class PhaseLMSStep:
    """Current local presynaptic rates/potentials and the next neuronal phase."""

    state: PhaseLMSState
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
class PhaseLMSFeedback:
    """Half-sum squared potential loss; residual is teacher minus saved potential."""

    loss: float
    teacher_count: int
    updated: bool
    winner: int
    target: int
    phase: int
    residual: Tensor


class PhaseContextLMS(nn.Module):
    """Fixed CPU float64, 96-context/three-phase/257-output local learner.

    Each multiplication neuron receives one phase and one context coordinate,
    through fixed unit synapses. Output neurons receive all phase and gated
    context rates. Every output potential learns from its own residual, including
    after a correct winner. Generation has no teacher, forced EOS or byte repair.
    """

    context_size = 96
    phase_size = 3
    feature_size = 291

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "output_weights", torch.zeros(257, 291, dtype=torch.float64, device="cpu")
        )
        self.register_buffer(
            "clock_transition",
            torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.int32, device="cpu"),
        )
        self.register_buffer(
            "clock_start", torch.tensor([1, 0, 0], dtype=torch.int32, device="cpu")
        )
        self.register_buffer(
            "gating_synapses", torch.ones(3, 96, 2, dtype=torch.float64, device="cpu")
        )

    def get_extra_state(self) -> dict:
        return {
            "schema": "phase-context-lms-v1", "context_size": 96,
            "phase_size": 3, "feature_size": 291, "outputs": 257,
            "dtype": "float64", "device": "cpu", "learning_rate": 0.01,
            "features": "phase-then-all-continuous-phase-context-products",
            "learning": "immediate-all-output-potential-residual",
            "loss": "half_sum_squared_potential_error",
            "clock": "fixed-three-neuron-chain", "max_bytes": 2,
        }

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("LMS checkpoint configuration mismatch")

    @staticmethod
    def _tensor(value: Tensor, shape: tuple, dtype: torch.dtype) -> None:
        if (
            not isinstance(value, Tensor) or value.shape != shape or value.dtype != dtype
            or value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None
        ):
            raise ValueError("LMS tensor shape/dtype/device/history mismatch")
        if value.is_floating_point() and not _all_finite(value):
            raise ValueError("LMS tensors must be finite")

    def _check_fixed(self) -> None:
        self._tensor(self.output_weights, (257, 291), torch.float64)
        self._tensor(self.clock_transition, (3, 3), torch.int32)
        self._tensor(self.clock_start, (3,), torch.int32)
        self._tensor(self.gating_synapses, (3, 96, 2), torch.float64)
        if (
            self.clock_transition.tolist() != [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
            or self.clock_start.tolist() != [1, 0, 0]
            or not bool((self.gating_synapses == 1).all())
        ):
            raise ValueError("LMS clock and generic gating dynamics must remain fixed")

    def _context(self, context: Tensor) -> Tensor:
        if (
            not isinstance(context, Tensor) or context.shape != (1, 96)
            or context.device.type != "cpu" or context.dtype not in (torch.float32, torch.float64)
            or not _all_finite(context)
        ):
            raise ValueError("context must be finite CPU float32/64 with shape [1, 96]")
        return context.detach().to(torch.float64)

    def _check_state(self, state: PhaseLMSState) -> None:
        if not isinstance(state, PhaseLMSState):
            raise ValueError("state must be a PhaseLMSState")
        self._tensor(state.phase, (1, 3), torch.int32)
        if not bool(((state.phase == 0) | (state.phase == 1)).all()) or int(state.phase.sum()) > 1:
            raise ValueError("clock activity must be one-hot or exhausted")

    def initial_state(self) -> PhaseLMSState:
        self._check_fixed()
        return PhaseLMSState(self.clock_start.unsqueeze(0).clone())

    def _scores(self, features: Tensor) -> Tensor:
        scores = F.linear(features, self.output_weights)
        if not _all_finite(scores):
            raise ValueError("LMS output potentials must be finite")
        return scores

    @torch.no_grad()
    def local_step(
        self, context: Tensor, input_tokens: Tensor, state: PhaseLMSState | None = None
    ) -> PhaseLMSStep:
        """Advance generic neuronal gating; previous-byte identity supplies no feature."""
        self._check_fixed()
        rates = self._context(context)
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
        phase_rates = state.phase.to(torch.float64)
        gated = (
            phase_rates[:, :, None] * self.gating_synapses[:, :, 0]
        ) * (rates[:, None, :] * self.gating_synapses[:, :, 1])
        features = torch.cat((phase_rates, gated.reshape(1, 288)), dim=-1)
        if not _all_finite(features):
            raise ValueError("LMS presynaptic features must be finite")
        next_state = PhaseLMSState(state.phase @ self.clock_transition)
        return PhaseLMSStep(next_state, features, self._scores(features))

    def _check_step(self, step: PhaseLMSStep) -> None:
        self._check_fixed()
        if not isinstance(step, PhaseLMSStep):
            raise ValueError("step must be a PhaseLMSStep")
        self._check_state(step.state)
        self._tensor(step.features, (1, 291), torch.float64)
        self._tensor(step.scores, (1, 257), torch.float64)
        phase = step.features[:, :3]
        if not bool(((phase == 0) | (phase == 1)).all()) or float(phase.sum()) != 1:
            raise ValueError("step must contain one active phase")
        gated = step.features[:, 3:].reshape(1, 3, 96)
        if (
            bool((gated * (1 - phase[:, :, None])).any())
            or not torch.equal(step.state.phase, phase.to(torch.int32) @ self.clock_transition)
        ):
            raise ValueError("step features and neuronal clock are inconsistent")
        if not torch.equal(step.scores, self._scores(step.features)):
            raise ValueError("teaching requires current potentials; stale step rejected")

    @torch.no_grad()
    def teach_local(
        self, step: PhaseLMSStep, targets: Tensor, learning_rate: float,
        *, learning_enabled: bool = True,
    ) -> PhaseLMSFeedback:
        """Apply each neuron's residual times local presynaptic activity, at rate 0.01."""
        self._check_step(step)
        self._tensor(targets, (1,), torch.int64)
        target = int(targets[0])
        if target < 0 or target > BYTE_EOS:
            raise ValueError("target must be a byte or EOS")
        if learning_rate != 0.01:
            raise ValueError("LMS learning_rate must be 0.01")
        if type(learning_enabled) is not bool:
            raise ValueError("learning_enabled must be boolean")
        winner = int(step.scores.argmax(-1)[0])
        residual = -step.scores
        residual[0, target] += 1
        if not _all_finite(residual):
            raise ValueError("LMS teaching residual must be finite")
        loss = 0.5 * float(residual.square().sum())
        if not math.isfinite(loss):
            raise ValueError("LMS squared potential loss must be finite")
        updated = False
        if learning_enabled:
            candidate = self.output_weights + (learning_rate * residual.T) * step.features
            if not _all_finite(candidate):
                raise ValueError("LMS proposed synapses must be finite")
            updated = not torch.equal(candidate, self.output_weights)
            self.output_weights.copy_(candidate)
        return PhaseLMSFeedback(
            loss, 1, updated, winner, target, int(step.features[0, :3].argmax()), residual,
        )

    @torch.no_grad()
    def generate(self, context: Tensor, max_bytes: int) -> tuple[ByteGeneration, ...]:
        """Read-only competition; stop only for a learned EOS winner or byte cap."""
        self._check_fixed()
        self._context(context)
        if type(max_bytes) is not int or not 0 <= max_bytes <= 2:
            raise ValueError("three-phase LMS max_bytes must be between zero and two")
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
        """Tensor payload only; Python, copies and proposal scratch also consume memory."""
        self._check_fixed()
        return {
            "neurons": {"supplied_context": 96, "clock": 3, "gating": 288, "output": 257},
            "total_neurons": 644, "learned_output_synapses": 74787,
            "fixed_allocated_synapses": 585, "fixed_nonzero_synapses": 578,
            "clock_initial_values": 3,
            "serialized_tensor_bytes": sum(
                value.numel() * value.element_size() for value in self.buffers()
            ),
            "state_tensor_bytes": 12, "step_tensor_bytes": 4396,
            "feedback_residual_tensor_bytes": 2056, "supplied_float64_context_bytes": 768,
        }
