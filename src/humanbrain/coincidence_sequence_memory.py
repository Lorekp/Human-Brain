"""Binary literal associations for a generic neuronal coincidence frontend.

Keys retain preceding raw-symbol rails, never normalized random codes. Sensory
content and forward/EOS cursor wiring keep the existing sequence-memory meaning.
"""

from collections.abc import Sequence

import torch

from humanbrain.neuronal_byte_coincidence import encode_symbol_rails, validate_symbol_rails
from humanbrain.neuronal_sequence_memory import (
    CAPACITY,
    EOS,
    LAGS,
    SENSORY_SIZE,
    NeuronalSequenceMemory,
    TapeState,
)
from humanbrain.plastic_memory import _all_finite

RAILS = 18
TAPE_BYTES = CAPACITY * (LAGS * RAILS + SENSORY_SIZE * 4 + 8) + 8


def _add(work: dict | None, name: str, value: int) -> None:
    if work is not None:
        work[name] = work.get(name, 0) + value


class CoincidenceSequenceMemory(NeuronalSequenceMemory):
    """128 literal slots with binary [32,18] preceding-symbol associations.

    A fresh selected slot publishes the exact local unit-rate endpoints
    ``w += cursor * (rail - w)``. Binary copy implements that update without
    pretending Boolean subtraction or normalized random-key writes are valid.
    Old tape snapshots are incompatible and are rejected by the new contract.
    """

    def initial_state(self) -> TapeState:
        return TapeState(
            torch.zeros(CAPACITY, LAGS, RAILS, dtype=torch.bool, device="cpu"),
            torch.zeros(CAPACITY, SENSORY_SIZE, dtype=torch.float32, device="cpu"),
            torch.full((CAPACITY,), -1, dtype=torch.int64, device="cpu"),
            torch.zeros((), dtype=torch.int64, device="cpu"),
        )

    def check_state(self, state: TapeState) -> None:
        if not isinstance(state, TapeState):
            raise ValueError("coincidence state must be a TapeState")
        for name, shape, dtype in (
            ("keys", (CAPACITY, LAGS, RAILS), torch.bool),
            ("sensory", (CAPACITY, SENSORY_SIZE), torch.float32),
            ("successor", (CAPACITY,), torch.int64),
            ("write_index", (), torch.int64),
        ):
            value = getattr(state, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or value.dtype != dtype
                or value.device.type != "cpu"
                or value.requires_grad
                or value.grad_fn is not None
                or value.grad is not None
                or not _all_finite(value)
            ):
                raise ValueError("coincidence tape must retain its finite detached CPU geometry")
        validate_symbol_rails(state.keys, allow_special=False)
        used = state.used
        if not 0 <= used <= CAPACITY:
            raise ValueError("write cursor must be within tape capacity")
        if (
            bool(state.keys[used:].count_nonzero())
            or bool(state.sensory[used:].count_nonzero())
            or bool((state.successor[used:] != -1).any())
        ):
            raise ValueError("unused coincidence slots must remain empty")
        sensory = state.sensory[:used]
        sums = sensory.sum(-1)
        if bool(((sensory != 0) & (sensory != 1)).any()) or bool(
            ((sums != 0) & (sums != 1)).any()
        ):
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

    @torch.no_grad()
    def ingest(
        self,
        documents: Sequence[bytes],
        state: TapeState | None = None,
        *,
        writes_enabled: bool = True,
        work: dict | None = None,
    ) -> TapeState:
        """Append once, retaining raw rails and preserving every existing slot.

        Encoding is input fan-out. It sees no query or teacher. EOS controls
        sensory publication and adjacency but never enters a preceding key.
        Disabled writes still consume the same documents and allocation clocks.
        """
        if (
            isinstance(documents, (bytes, str))
            or not isinstance(documents, Sequence)
            or any(not isinstance(document, bytes) for document in documents)
            or type(writes_enabled) is not bool
        ):
            raise ValueError("documents must be a sequence of raw bytes and enabled a boolean")
        if state is not None:
            self.check_state(state)
        used = 0 if state is None else state.used
        required = sum(len(document) + 1 for document in documents)
        if used + required > CAPACITY:
            raise ValueError("neuronal coincidence memory capacity exceeded")
        result = self.initial_state() if state is None else TapeState(**state.snapshot())
        slot = used
        for document in documents:
            rails = encode_symbol_rails(
                torch.tensor(list(document), dtype=torch.int64, device="cpu"), work=work
            )
            lags = torch.zeros(LAGS, RAILS, dtype=torch.bool, device="cpu")
            for position, token in enumerate((*document, EOS)):
                if writes_enabled:
                    result.keys[slot].copy_(lags)
                    # Allocation selected an unused slot, whose other outputs are zero.
                    result.sensory[slot, token] = 1
                    _add(work, "coincidence_ingest_key_publications", LAGS * RAILS)
                    _add(work, "coincidence_ingest_sensory_eligible_synapses", SENSORY_SIZE)
                    _add(work, "coincidence_ingest_sensory_publications", 1)
                result.successor[slot] = slot if token == EOS else slot + 1
                slot += 1
                _add(work, "coincidence_ingest_allocation_steps", 1)
                if token != EOS:
                    lags[1:].copy_(lags[:-1].clone())
                    lags[0].copy_(rails[position])
                    _add(work, "coincidence_ingest_lag_shift_values", (LAGS - 1) * RAILS)
                    _add(work, "coincidence_ingest_lag_shift_copy_values", (LAGS - 1) * RAILS)
                    _add(work, "coincidence_ingest_input_rail_values", RAILS)
            result.write_index.fill_(slot)
            _add(work, "coincidence_ingest_document_resets", 1)
        self.check_state(result)
        return result

    def accounting(self, state: TapeState) -> dict[str, int]:
        self.check_state(state)
        return {
            "capacity": CAPACITY,
            "occupied_slots": state.used,
            "key_features_per_slot": LAGS * RAILS,
            "sensory_neurons": SENSORY_SIZE,
            "key_sensory_association_synapses": CAPACITY * (LAGS * RAILS + SENSORY_SIZE),
            "association_tensor_bytes": CAPACITY * (LAGS * RAILS + SENSORY_SIZE * 4),
            "binary_key_tensor_bytes": CAPACITY * LAGS * RAILS,
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
            "temporary_lag_neurons": LAGS * RAILS,
            "temporary_lag_bytes": LAGS * RAILS,
            "temporary_lag_shift_copy_bytes": (LAGS - 1) * RAILS,
            "logical_lag_shift_connections": (LAGS - 1) * RAILS,
            "state_tensor_bytes": sum(v.numel() * v.element_size() for v in vars(state).values()),
        }
