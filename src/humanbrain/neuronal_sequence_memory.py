"""Literal neuronal sequence associations and sparse forward-only cursor activity.

This is finite episodic storage, not semantic compression or learned admission.
Sparse indices implement one-hot cursor populations and their declared adjacency;
no question, answer or selected span enters association writes or cursor advances.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from humanbrain.plastic_memory import _all_finite

CAPACITY = 128
LAGS = 32
CODE_SIZE = 32
SENSORY_SIZE = 257
KEY_SIZE = LAGS * CODE_SIZE
EOS = 256


@dataclass(frozen=True)
class TapeState:
    """Caller-owned associations and sparse allocation/adjacency; no raw cache."""

    keys: Tensor
    sensory: Tensor
    successor: Tensor
    write_index: Tensor

    @property
    def used(self) -> int:
        return int(self.write_index.item())

    def snapshot(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in vars(self).items()}


class NeuronalSequenceMemory:
    """128 slots; each key contains 32 preceding codes, newest lag first.

    The write cursor has 128 logical neurons, represented by one int64 index;
    index 128 means exhausted/no active write neuron. Each of 128 possible start
    branches has a 128-neuron logical recall cursor, stored as one active index
    only when occupied. Shared adjacency has forward links and EOS-gated self
    links. EOS is absorbing for recall but allocation always advances.

    Associations at a fresh active slot implement the local unit-rate rule
    w += cursor*(sensory-w), including normalized lag-key sensory activity.
    Source strings and the temporary lag population do not survive ingestion.
    """

    def initial_state(self) -> TapeState:
        return TapeState(
            torch.zeros(CAPACITY, KEY_SIZE, dtype=torch.float32, device="cpu"),
            torch.zeros(CAPACITY, SENSORY_SIZE, dtype=torch.float32, device="cpu"),
            torch.full((CAPACITY,), -1, dtype=torch.int64, device="cpu"),
            torch.zeros((), dtype=torch.int64, device="cpu"),
        )

    def check_state(self, state: TapeState) -> None:
        if not isinstance(state, TapeState):
            raise ValueError("state must be a TapeState")
        for name, shape, dtype in (
            ("keys", (CAPACITY, KEY_SIZE), torch.float32),
            ("sensory", (CAPACITY, SENSORY_SIZE), torch.float32),
            ("successor", (CAPACITY,), torch.int64),
            ("write_index", (), torch.int64),
        ):
            value = getattr(state, name)
            if (
                not isinstance(value, Tensor) or value.shape != shape or value.dtype != dtype
                or value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None
                or not _all_finite(value)
            ):
                raise ValueError(
                    "tape tensors must be finite detached CPU tensors of declared shape"
                )
        used = state.used
        if not 0 <= used <= CAPACITY:
            raise ValueError("write cursor must be within tape capacity")
        if (
            bool(state.keys[used:].count_nonzero())
            or bool(state.sensory[used:].count_nonzero())
            or bool((state.successor[used:] != -1).any())
        ):
            raise ValueError("unused tape slots must remain empty")
        sensory = state.sensory[:used]
        sums = sensory.sum(-1)
        if bool(((sensory != 0) & (sensory != 1)).any()) or bool(((sums != 0) & (sums != 1)).any()):
            raise ValueError("sensory associations must be one-hot or disabled zeros")
        if used:
            indices = torch.arange(used, dtype=torch.int64, device="cpu")
            successor = state.successor[:used]
            if bool(((successor != indices) & (successor != indices + 1)).any()):
                raise ValueError("recall adjacency must be forward or absorbing")
            if int(successor[-1]) != used - 1:
                raise ValueError("last occupied slot must be an absorbing document end")
            active = sums == 1
            if bool((active & ((sensory[:, EOS] == 1) != (successor == indices))).any()):
                raise ValueError("stored EOS and absorbing adjacency must agree")

    def restore_state(self, payload: dict[str, Tensor]) -> TapeState:
        if not isinstance(payload, dict) or set(payload) != {
            "keys", "sensory", "successor", "write_index"
        }:
            raise ValueError("invalid tape snapshot fields")
        state = TapeState(**payload)
        self.check_state(state)
        return TapeState(**state.snapshot())

    @torch.no_grad()
    def ingest(
        self, documents: Sequence[bytes], input_codes: Tensor, state: TapeState | None = None,
        *, writes_enabled: bool = True,
    ) -> TapeState:
        """Append complete raw documents/EOS, rejecting overflow before any write.

        Disabled sensory writes retain identical allocation and EOS boundaries,
        but write no sensory or key activity, including at EOS. Existing slots
        always survive; all mutations are to an independent snapshot.
        """
        if (
            isinstance(documents, (bytes, str)) or not isinstance(documents, Sequence)
            or any(not isinstance(document, bytes) for document in documents)
            or type(writes_enabled) is not bool
        ):
            raise ValueError("documents must be a sequence of raw bytes and enabled a boolean")
        if (
            not isinstance(input_codes, Tensor) or input_codes.shape != (258, CODE_SIZE)
            or input_codes.dtype != torch.float32 or input_codes.device.type != "cpu"
            or input_codes.requires_grad or input_codes.grad_fn is not None
            or not _all_finite(input_codes)
        ):
            raise ValueError("input codes must be fixed finite CPU float32 [258,32]")
        if state is None:
            state = self.initial_state()
        self.check_state(state)
        required = sum(len(document) + 1 for document in documents)
        if state.used + required > CAPACITY:
            raise ValueError("neuronal sequence memory capacity exceeded")
        result = TapeState(**state.snapshot())
        slot = state.used
        for document in documents:
            lags = torch.zeros(LAGS, CODE_SIZE, dtype=torch.float32, device="cpu")
            for token in (*document, EOS):
                if writes_enabled:
                    prefix = lags.reshape(KEY_SIZE)
                    norm = prefix.norm()
                    if not _all_finite(norm):
                        raise ValueError("preceding-key normalization must remain finite")
                    key = prefix / norm.clamp_min(1e-8)
                    sensory = torch.zeros(SENSORY_SIZE, dtype=torch.float32, device="cpu")
                    sensory[token] = 1
                    # The current cursor is one, every other cursor is zero.
                    result.keys[slot].add_(key - result.keys[slot])
                    result.sensory[slot].add_(sensory - result.sensory[slot])
                result.successor[slot] = slot if token == EOS else slot + 1
                slot += 1
                # EOS resets the lag population at the next document boundary.
                if token != EOS:
                    lags[1:] = lags[:-1].clone()
                    lags[0] = input_codes[token]
            result.write_index.fill_(slot)
        self.check_state(result)
        return result

    def initial_recall(self, state: TapeState) -> Tensor:
        """One branch at every occupied start, not a selected winning address."""
        self.check_state(state)
        return torch.arange(state.used, dtype=torch.int64, device="cpu")

    def _check_cursors(self, state: TapeState, cursors: Tensor) -> None:
        if (
            not isinstance(cursors, Tensor) or cursors.ndim != 1
            or cursors.numel() > CAPACITY or cursors.dtype != torch.int64
            or cursors.device.type != "cpu" or cursors.requires_grad
            or bool(((cursors < 0) | (cursors >= state.used)).any())
        ):
            raise ValueError("recall cursors must be occupied CPU int64 indices")

    def content(self, state: TapeState, cursors: Tensor) -> Tensor:
        self.check_state(state)
        return self._content_checked(state, cursors)

    def _content_checked(self, state: TapeState, cursors: Tensor) -> Tensor:
        """Caller validated immutable tape; still validate current sparse activity."""
        self._check_cursors(state, cursors)
        return state.sensory.index_select(0, cursors)

    def advance(self, state: TapeState, cursors: Tensor) -> Tensor:
        """Exact sparse evaluation of forward/EOS-absorbing neuronal adjacency."""
        self.check_state(state)
        return self._advance_checked(state, cursors)

    def _advance_checked(self, state: TapeState, cursors: Tensor) -> Tensor:
        """Caller validated immutable tape; only forward sparse activity changes."""
        self._check_cursors(state, cursors)
        return state.successor.index_select(0, cursors)

    def accounting(self, state: TapeState) -> dict[str, int]:
        self.check_state(state)
        return {
            "capacity": CAPACITY, "occupied_slots": state.used,
            "key_features_per_slot": KEY_SIZE, "sensory_neurons": SENSORY_SIZE,
            "key_sensory_association_synapses": CAPACITY * (KEY_SIZE + SENSORY_SIZE),
            "association_tensor_bytes": (KEY_SIZE + SENSORY_SIZE) * CAPACITY * 4,
            "logical_write_cursor_neurons": CAPACITY,
            "logical_recall_branches": CAPACITY,
            "logical_recall_cursor_neurons": CAPACITY * CAPACITY,
            "logical_write_forward_connections": CAPACITY - 1,
            "logical_shared_recall_forward_connections": CAPACITY - 1,
            "logical_shared_recall_self_connections": CAPACITY,
            "logical_recall_connections_all_branches": CAPACITY * (2 * CAPACITY - 1),
            "enabled_shared_recall_connections": state.used,
            "stored_successor_bytes": CAPACITY * 8,
            "stored_write_cursor_bytes": 8,
            "stored_initial_recall_cursor_bytes": state.used * 8,
            "temporary_lag_neurons": KEY_SIZE,
            "temporary_lag_bytes": KEY_SIZE * 4,
            "logical_lag_shift_connections": (LAGS - 1) * CODE_SIZE,
            "state_tensor_bytes": sum(v.numel() * v.element_size() for v in vars(state).values()),
        }
