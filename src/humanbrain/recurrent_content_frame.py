"""Causal public framing and fixed sparse one-hot content relays.

The frame has no learned weights, targets, semantic selectors or mutable tensor
backing. Public rows become visible only after reception. Factual rows are a
copy of the already admitted tape, never a privately staged source suffix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256

import torch

from humanbrain.neuronal_sequence_memory import TapeState
from humanbrain.recurrent_content_controller import SparseActivity
from humanbrain.sequence_assembly_observation import RawObservation

EOS = 256
BOS = 257
ABSENT = 258
PUBLIC_CAPACITY = 3485
FACT_CAPACITY = 2048
ADDRESS_CAPACITY = PUBLIC_CAPACITY + FACT_CAPACITY
REQUEST_WIDTH = 68868
CONTEXT_WIDTH = 69954
EMISSION_WIDTH = 1072
FIELD_COUNT = 14
GATES = (
    "HOLD", "BIND1_CURRENT", "BIND2_CURRENT", "BIND1_RETRIEVED",
    "BIND2_RETRIEVED", "ADVANCE1", "ADVANCE2", "SWAP",
)
WORK_KEYS = tuple("content_frame_" + name for name in (
    "constructions", "public_slots", "fact_slots", "fact_validation_values",
    "fact_identity_products", "fact_identity_additions", "fact_population_additions",
    "copied_scalar_values", "slot_initializations",
    "custody_bytes_hashed", "field_entries", "received_symbols", "raw_offers",
    "eos_offers", "status_offers", "absent_markers", "query_incidence_writes",
    "query_unique_entries", "register_reads", "visibility_checks",
    "read_active_synapses", "read_products", "read_identity_products",
    "read_reduction_additions", "invalid_reads", "gate_choices",
    "bind_relays", "successor_relays", "swap_relays", "hold_relays",
    "activity_builds", "activity_nonzeros", "activity_square_products",
    "activity_sum_additions", "activity_square_roots", "activity_divisions",
    "activity_tensor_bytes", "output_advances",
))


def _add(work, name, value=1):
    if work is not None:
        key = "content_frame_" + name
        work[key] = work.get(key, 0) + int(value)


def _seal(value, work=None):
    raw = repr(value).encode("utf-8")
    _add(work, "custody_bytes_hashed", len(raw))
    return sha256(b"humanbrain.recurrent-content-frame.v1\0" + raw).hexdigest()


def _integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value < upper:
        raise ValueError(f"{label} outside its fixed population")


@dataclass(frozen=True)
class InputField:
    field: int
    tokens: tuple[int, ...]
    addresses: tuple[int, ...]
    supervised: bool
    status: bool = False


@dataclass(frozen=True)
class ContentFrame:
    fields: tuple[InputField, ...]
    symbols: tuple[int, ...]
    successors: tuple[int, ...]
    reveal: tuple[int, ...]
    fact_used: int
    source: bool
    seal: str
    _owner: object = field(repr=False, compare=False)

    @property
    def offer_count(self):
        return sum(len(item.tokens) for item in self.fields if item.supervised)

    @property
    def token_count(self):
        return sum(len(item.tokens) for item in self.fields)


@dataclass(frozen=True)
class FramePosition:
    """Immutable causal cursor; field offsets count received symbols, not draws."""

    frame_seal: str
    field_index: int
    offset: int
    arrived: int
    last: int
    current: int | None
    phase: int  # source=0, public input=1, output=2
    request: bytes
    request_eos: bool
    query_indices: tuple[int, ...]
    _owner: object = field(repr=False, compare=False)

    @property
    def seal(self):
        return _seal((self.frame_seal, self.field_index, self.offset, self.arrived,
                      self.last, self.current, self.phase, self.request,
                      self.request_eos, self.query_indices))


def _facts(state, work):
    if state is None:
        return (), ()
    if not isinstance(state, TapeState):
        raise ValueError("frame facts require a TapeState or None")
    for value, shape, dtype in (
        (state.sensory, (FACT_CAPACITY, 257), torch.float32),
        (state.successor, (FACT_CAPACITY,), torch.int64),
        (state.write_index, (), torch.int64),
    ):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or value.dtype != dtype or value.device.type != "cpu"
                or value.requires_grad or value.grad_fn is not None or value.grad is not None):
            raise ValueError("frame facts require detached CPU2048 tape populations")
    used = int(state.write_index)
    _integer(used, 0, FACT_CAPACITY + 1, "fact occupancy")
    # Exact sparse conversion of one-hot sensory wiring, including disabled zeros.
    # Every occupied column is inspected; no candidate or target-based filter.
    symbols = []
    rows = state.sensory[:used].tolist()
    successors = state.successor[:used].tolist()
    _add(work, "fact_validation_values", used * 258)
    for index, (row, successor) in enumerate(zip(rows, successors, strict=True)):
        if any(value not in (0.0, 1.0) for value in row):
            raise ValueError("factual sensory rows must be binary one-hot or empty")
        active = sum(row)
        _add(work, "fact_population_additions", 256)
        if active not in (0, 1):
            raise ValueError("factual sensory row has multiple active symbols")
        token = sum(column * value for column, value in enumerate(row))
        _add(work, "fact_identity_products", 257)
        _add(work, "fact_identity_additions", 256)
        if successor not in (index, index + 1) or successor >= used:
            raise ValueError("factual successor is not an admitted forward/absorbing link")
        if active and ((token == EOS) != (successor == index)):
            raise ValueError("factual EOS does not agree with its absorbing relay")
        symbols.append(int(token) if active else -1)
    _add(work, "fact_slots", used)
    _add(work, "copied_scalar_values", used * 258)
    return tuple(symbols), tuple(successors)


def _frame(fields, state, source, work):
    facts, links = _facts(state, work)
    symbols = [-1] * ADDRESS_CAPACITY
    successors = [-1] * ADDRESS_CAPACITY
    reveal = [-1] * ADDRESS_CAPACITY
    _add(work, "slot_initializations", ADDRESS_CAPACITY * 3)
    rank = 0
    for item in fields:
        for index, (token, address) in enumerate(zip(item.tokens, item.addresses, strict=True)):
            symbols[address] = token
            reveal[address] = rank
            rank += 1
            if index + 1 < len(item.addresses):
                successors[address] = item.addresses[index + 1]
    for index, (token, successor) in enumerate(zip(facts, links, strict=True)):
        address = PUBLIC_CAPACITY + index
        symbols[address] = token
        successors[address] = PUBLIC_CAPACITY + successor
        reveal[address] = -2  # All and only admitted facts are visible from entry.
    payload = (tuple(fields), tuple(symbols), tuple(successors), tuple(reveal), len(facts), source)
    _add(work, "constructions")
    _add(work, "public_slots", PUBLIC_CAPACITY)
    _add(work, "copied_scalar_values", ADDRESS_CAPACITY * 3)
    return ContentFrame(*payload, _seal(payload, work), object())


def public_frame(observation: RawObservation, state: TapeState | None, work=None) -> ContentFrame:
    if not isinstance(observation, RawObservation):
        raise ValueError("frame requires a complete bounded RawObservation")
    sensor = observation.sensor_frame()
    # Preserve all3471 old sensor coordinates. The extra returncode EOS follows
    # that population; flags occupy3472..3484. Scan order remains structural.
    spans = [(0, 257), (257, 513), (770, 513), (1283, 8)]
    spans.extend((1291 + file * 545 + shift, length)
                 for file in range(4) for shift, length in ((0, 32), (32, 513)))
    fields = []
    for field_id, (start, length) in enumerate(spans):
        raw = sensor.symbols[start:start + length]
        present = raw[0] != -1
        if not present:
            fields.append(InputField(field_id, (ABSENT,), (start,), False))
            continue
        tokens = tuple(token for token in raw if token != -1)
        addresses = tuple(range(start, start + len(tokens)))
        if field_id == 3:
            tokens += (EOS,)
            addresses += (3471,)
        fields.append(InputField(field_id, tokens, addresses, True))
    fields.append(InputField(12, (*map(int, sensor.flags), EOS), tuple(range(3472, 3485)), True, True))
    return _frame(fields, state, False, work)


def source_frame(raw: bytes, state: TapeState | None = None, work=None) -> ContentFrame:
    if type(raw) is not bytes or len(raw) >= PUBLIC_CAPACITY:
        raise ValueError("source must fit3484 raw bytes plus EOS in its expiring frame")
    item = InputField(13, (*raw, EOS), tuple(range(len(raw) + 1)), True)
    return _frame((item,), state, True, work)


def initial_position(frame: ContentFrame) -> FramePosition:
    return FramePosition(frame.seal, -1, 0, 0, BOS, None, int(not frame.source),
                         b"", False, (), frame._owner)


def check_position(frame, position):
    if (not isinstance(frame, ContentFrame) or not isinstance(position, FramePosition)
            or position._owner is not frame._owner or position.frame_seal != frame.seal):
        raise ValueError("position belongs to another immutable content frame")


def enter_field(frame, position, field_index, work=None):
    check_position(frame, position)
    _integer(field_index, 0, len(frame.fields), "field index")
    if (position.phase == 2 or field_index != position.field_index + 1
            or (position.field_index >= 0
                and position.offset != len(frame.fields[position.field_index].tokens))):
        raise ValueError("fields must be consumed once in their fixed public order")
    _add(work, "field_entries")
    return FramePosition(frame.seal, field_index, 0, position.arrived, BOS, None,
                         position.phase, position.request, position.request_eos,
                         position.query_indices, frame._owner)


def receive(frame, position, symbol, work=None):
    check_position(frame, position)
    _integer(symbol, 0, ABSENT + 1, "received symbol")
    if position.phase == 2 or position.field_index < 0:
        raise ValueError("receive requires an open source/public field")
    item = frame.fields[position.field_index]
    if position.offset >= len(item.tokens) or item.tokens[position.offset] != symbol:
        raise ValueError("received symbol must be the next actual public/source symbol")
    request, request_eos = position.request, position.request_eos
    query = position.query_indices
    if not frame.source and item.field == 0:
        indices = set(query)
        previous = request[-1] if request else BOS
        indices.add(256 + previous * 258 + symbol)
        writes = 1
        if symbol != EOS:
            indices.add(symbol)
            indices.add(256 + 258 * 258 + (len(request) // 32) * 256 + symbol)
            request += bytes((symbol,))
            writes += 2
        else:
            request_eos = True
        query = tuple(sorted(indices))
        _add(work, "query_incidence_writes", writes)
        _add(work, "query_unique_entries", len(query))
    _add(work, "received_symbols")
    if not item.supervised:
        _add(work, "absent_markers")
    elif item.status:
        _add(work, "status_offers")
    elif symbol == EOS:
        _add(work, "eos_offers")
    else:
        _add(work, "raw_offers")
    return FramePosition(frame.seal, position.field_index, position.offset + 1,
                         position.arrived + 1, symbol, item.addresses[position.offset],
                         position.phase, request, request_eos, query, frame._owner)


def output_position(frame, position):
    check_position(frame, position)
    if (position.phase == 2 or position.field_index != len(frame.fields) - 1
            or position.offset != len(frame.fields[-1].tokens)):
        raise ValueError("output must follow the complete causal input scan")
    return FramePosition(frame.seal, position.field_index, position.offset,
                         position.arrived, BOS, None, 2, position.request,
                         position.request_eos, position.query_indices, frame._owner)


def advance_output(position, token, work=None):
    if not isinstance(position, FramePosition) or position.phase != 2:
        raise ValueError("output advance requires an output position")
    _integer(token, 0, EOS + 1, "emitted symbol")
    if position.last == EOS:
        raise ValueError("cannot advance beyond emitted EOS")
    _add(work, "output_advances")
    return FramePosition(position.frame_seal, position.field_index, position.offset,
                         position.arrived, token, None, 2, position.request,
                         position.request_eos, position.query_indices, position._owner)


def _cursor(value):
    if value is not None:
        _integer(value, 0, ADDRESS_CAPACITY, "one-hot address")


def _visible(frame, position, address, work):
    _add(work, "visibility_checks")
    rank = frame.reveal[address]
    return rank == -2 or 0 <= rank < position.arrived


def read_registers(frame, position, cursors, work=None):
    check_position(frame, position)
    if type(cursors) is not tuple or len(cursors) != 2:
        raise ValueError("two sparse one-hot registers required")
    result = []
    for address in cursors:
        _cursor(address)
        _add(work, "register_reads")
        if address is None:
            result.append(None)
            _add(work, "invalid_reads")
            continue
        visible = int(_visible(frame, position, address, work))
        token = frame.symbols[address]
        # Sparse evaluation visits the only active address, then the only stored
        # sensory afferent. All other one-hot products are exactly structural zero.
        # The framing-only ABSENT marker cannot become a literal byte emission.
        # Its read is represented by the absent symbol cell and validity zero.
        activity = 1 * visible * int(0 <= token <= EOS)
        _add(work, "read_products", 2)
        _add(work, "read_active_synapses", int(token >= 0))
        identity = activity * max(token, 0)
        _add(work, "read_identity_products")
        result.append(identity if activity else None)
        _add(work, "invalid_reads", not activity)
    return tuple(result)


def apply_gate(frame, position, cursors, gate, *, retrieved_index=None, work=None):
    check_position(frame, position)
    _integer(gate, 0, len(GATES), "gate choice")
    if type(cursors) is not tuple or len(cursors) != 2:
        raise ValueError("two sparse one-hot registers required")
    for cursor in cursors:
        _cursor(cursor)
    retrieved = None
    if retrieved_index is not None:
        _integer(retrieved_index, 0, frame.fact_used, "actual factual winner")
        if position.phase != 2:
            raise ValueError("source/public scanning has no retrieved factual winner")
        retrieved = PUBLIC_CAPACITY + retrieved_index
    _add(work, "gate_choices")
    values = list(cursors)
    if gate == 0:
        _add(work, "hold_relays", 2)
    elif gate in (1, 2, 3, 4):
        target = (gate - 1) % 2
        values[target] = position.current if gate < 3 else retrieved
        _add(work, "bind_relays")
    elif gate in (5, 6):
        target = gate - 5
        address = values[target]
        if address is not None:
            successor = frame.successors[address]
            values[target] = successor if successor >= 0 else None
            _add(work, "successor_relays")
    else:
        values = [values[1], values[0]]
        _add(work, "swap_relays", 2)
    return tuple(values)


def _activity(frame, position, h, cursors, prior_token, emission, work):
    check_position(frame, position)
    _integer(h, 0, 32, "recurrent context")
    if position.field_index < 0:
        raise ValueError("activity requires a declared input field")
    if prior_token is not None:
        _integer(prior_token, 0, 257, "actual prior byte/EOS")
        if position.phase != 2:
            raise ValueError("scanning cannot use an output prior")
    registers = read_registers(frame, position, cursors, work)
    blocks = []
    if emission:
        blocks.append((32, (h,)))
    else:
        blocks.append((REQUEST_WIDTH, position.query_indices))
    blocks.append((259, (position.last,)))
    if not emission:
        blocks.extend(((14, (frame.fields[position.field_index].field,)),
                       (3, (position.phase,)), (32, (h,))))
    for token in registers:
        blocks.append((259, (ABSENT if token is None else token,)))
    blocks.append((2, tuple(i for i, token in enumerate(registers) if token is not None)))
    if emission:
        blocks.append((3, (position.phase,)))
    blocks.extend(((257, () if prior_token is None else (prior_token,)), (1, (0,))))
    entries, offset = [], 0
    for width, active in blocks:
        entries.extend(offset + index for index in active)
        offset += width
    expected = EMISSION_WIDTH if emission else CONTEXT_WIDTH
    if offset != expected:
        raise RuntimeError("fixed frame population widths disagree")
    indices = torch.tensor(entries, dtype=torch.int64)
    raw = torch.ones(len(entries), dtype=torch.float64)
    squares = raw.square()
    energy = squares.sum()
    norm = energy.sqrt()
    values = raw / norm
    _add(work, "activity_builds")
    _add(work, "activity_nonzeros", len(entries))
    _add(work, "activity_square_products", len(entries))
    _add(work, "activity_sum_additions", len(entries) - 1)
    _add(work, "activity_square_roots")
    _add(work, "activity_divisions", len(entries))
    _add(work, "activity_tensor_bytes", 32 * len(entries) + 16)
    return SparseActivity(indices=indices, values=values, width=expected)


def context_activity(frame, position, h, cursors, *, prior_token=None, work=None):
    """Used for context and again for gate currents with the newly sampled h."""
    return _activity(frame, position, h, cursors, prior_token, False, work)


def emission_activity(frame, position, h, cursors, *, prior_token=None, work=None):
    return _activity(frame, position, h, cursors, prior_token, True, work)
