"""Compositional fitted-cell prompt-factor bridge, self-posed slice.

This module is deliberately isolated from the canonical TRAIN capsule and its
annotation manifest.  It implements the eventual learner seam on a tiny
self-posed curriculum: a private three-bank projector, independent one-vs-rest
factor heads, one local error rule, and a frozen ``infer`` boundary.

Only canonical tokens enter the plastic path.  Whole-span and whole-prompt
digests authenticate curriculum membership; neither digest is projected into a
cell, a current, or a local update.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.resting_glif_race import CompiledRestingGlifRace


BRIDGE_SCOPE = "PUBLIC_TRAIN_PROMPT_FACTOR_COMPOSITIONAL_BRIDGE_V2"

FACTOR_REGISTRY = (
    "add",
    "subtract",
    "minimum",
    "maximum",
    "absolute",
    "less-than",
    "map",
    "sum",
    "count",
    "contains",
    "filter",
    "reverse",
    "class:ascii-alphanumeric",
    "class:ascii-decimal-digits",
    "class:ascii-hex-digits",
    "class:ascii-letters",
    "class:ascii-lowercase",
    "class:ascii-octal-digits",
    "class:ascii-printable",
    "class:ascii-punctuation",
    "class:ascii-uppercase",
    "class:ascii-whitespace",
    "class:english-consonants",
    "class:english-vowels",
)
_FACTOR_INDEX = MappingProxyType(
    {factor: index for index, factor in enumerate(FACTOR_REGISTRY)}
)
_SLICE_FACTORS = ("add", "subtract")

HEAD_COUNT = len(FACTOR_REGISTRY)
BANK_CELL_COUNT = 512
BANK_COUNT = 3
INPUT_CELL_COUNT = BANK_COUNT * BANK_CELL_COUNT
ASSEMBLY_ACTIVE_CELL_COUNT = 12
MAX_RELATIVE_SLOTS = 4

TICK_DTYPE = np.dtype("<u2")
TICK_CAP = 50
TICK_CURRENT_PA = 32.0
PRESENT_BASE_CURRENT_PA = 600.0
STOP_CURRENT_PA = 650.0
PRESENT_MAX_CURRENT_PA = 2_200.0
RACE_STEPS = 600
TRAINING_EPOCHS = 4

STOP_CELL = 0
PRESENT_CELL = 1
STOP = "STOP"
PRESENT = "PRESENT"

PHYSICAL_DRIVE_CURRENT_PA = 5_000.0
PHYSICAL_DRIVE_STEPS = 30

LEARNED_STATE_BYTES = (
    HEAD_COUNT * INPUT_CELL_COUNT * TICK_DTYPE.itemsize * 2
)
if LEARNED_STATE_BYTES != 147_456:
    raise RuntimeError("compositional bridge learned-state anatomy changed")

_TOKEN_BANK = 0
_SLOT_BANK = 1
_TRANSITION_BANK = 2
_BANK_NAMES = ("canonical-token", "relative-slot-token", "ordered-transition")
_BANK_NAMESPACES = (
    b"prompt-factor-compositional-v2-token-assembly\0",
    b"prompt-factor-compositional-v2-slot-token-assembly\0",
    b"prompt-factor-compositional-v2-transition-assembly\0",
)
_SPAN_NAMESPACE = b"prompt-factor-compositional-v2-whole-span\0"
_AUDIT_NAMESPACE = b"prompt-factor-compositional-v2-whole-prompt-audit\0"
_CHECKPOINT_NAMESPACE = b"prompt-factor-compositional-bridge-v2-checkpoint\0"

_TOKEN_PATTERN = re.compile(
    r"<[A-Za-z][A-Za-z0-9_]*>"
    r"|[A-Za-z]+(?:-[A-Za-z]+)*"
    r"|[+-]?\d+"
    r"|[^\s]"
)
_ROLE_TOKEN = re.compile(r"<[A-Za-z][A-Za-z0-9_]*>\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


FACTOR_REGISTRY_SHA256 = _namespaced_sha256(
    b"prompt-factor-compositional-v2-registry\0",
    FACTOR_REGISTRY,
)


def _canonical_token(token: str) -> str:
    if _ROLE_TOKEN.fullmatch(token):
        return token.upper()
    if re.fullmatch(r"[+-]?\d+", token):
        return "<INT>"
    return token.casefold()


def _tokenize(surface: str) -> tuple[str, ...]:
    if type(surface) is not str or not surface.strip():
        raise ValueError("compositional input requires nonempty text")
    tokens: list[str] = []
    cursor = 0
    for match in _TOKEN_PATTERN.finditer(surface):
        if surface[cursor : match.start()].strip():
            raise ValueError("text lies outside the compositional tokenizer")
        tokens.append(_canonical_token(match.group(0)))
        cursor = match.end()
    if surface[cursor:].strip():
        raise ValueError("text lies outside the compositional tokenizer")
    if not 1 <= len(tokens) <= MAX_RELATIVE_SLOTS:
        raise ValueError("a compositional sequence needs one to four tokens")
    return tuple(tokens)


@dataclass(frozen=True, slots=True)
class CompositionalPromptInput:
    """One authenticated ordered prompt sequence supplied to frozen inference."""

    start_token: int
    stop_token: int
    canonical_tokens: tuple[str, ...]
    whole_span_sha256: str
    audit_prompt_sha256: str

    def __post_init__(self) -> None:
        if type(self.start_token) is not int or self.start_token < 0:
            raise ValueError("start_token must be a nonnegative integer")
        if (
            type(self.stop_token) is not int
            or self.stop_token - self.start_token != len(self.canonical_tokens)
        ):
            raise ValueError("stop_token must delimit the canonical sequence")
        if (
            type(self.canonical_tokens) is not tuple
            or not 1 <= len(self.canonical_tokens) <= MAX_RELATIVE_SLOTS
            or any(type(token) is not str or not token for token in self.canonical_tokens)
            or tuple(map(_canonical_token, self.canonical_tokens))
            != self.canonical_tokens
        ):
            raise ValueError("canonical_tokens must contain one to four canonical tokens")
        for field_name, value in (
            ("whole_span_sha256", self.whole_span_sha256),
            ("audit_prompt_sha256", self.audit_prompt_sha256),
        ):
            if type(value) is not str or _HEX_SHA256.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256")


def _make_prompt_input(
    surface: str,
    *,
    start_token: int = 0,
    audit_prompt: str | None = None,
) -> CompositionalPromptInput:
    """Private host tokenizer/hash boundary used by self-posed fixtures."""

    tokens = _tokenize(surface)
    if type(start_token) is not int or start_token < 0:
        raise ValueError("start_token must be a nonnegative integer")
    prompt = surface if audit_prompt is None else audit_prompt
    if type(prompt) is not str or not prompt:
        raise ValueError("audit_prompt must be nonempty text")
    return CompositionalPromptInput(
        start_token=start_token,
        stop_token=start_token + len(tokens),
        canonical_tokens=tokens,
        whole_span_sha256=_namespaced_sha256(_SPAN_NAMESPACE, tokens),
        audit_prompt_sha256=_namespaced_sha256(_AUDIT_NAMESPACE, prompt),
    )


@dataclass(frozen=True, slots=True)
class CompositionalLesson:
    """Atomic factor teacher bits for one authenticated prompt sequence."""

    prompt_input: CompositionalPromptInput
    present_factors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_input, CompositionalPromptInput):
            raise TypeError("prompt_input must be CompositionalPromptInput")
        if (
            type(self.present_factors) is not tuple
            or not self.present_factors
            or len(set(self.present_factors)) != len(self.present_factors)
            or any(factor not in _FACTOR_INDEX for factor in self.present_factors)
        ):
            raise ValueError("present_factors must be unique registry factors")
        ordered = tuple(
            factor for factor in FACTOR_REGISTRY if factor in self.present_factors
        )
        if self.present_factors != ordered:
            raise ValueError("present_factors must use registry order")


@dataclass(frozen=True, slots=True)
class CompositionalSplit:
    """Authentication-only membership for the self-posed capacity split."""

    fit_span_sha256s: tuple[str, ...]
    generalization_span_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, values in (
            ("fit_span_sha256s", self.fit_span_sha256s),
            ("generalization_span_sha256s", self.generalization_span_sha256s),
        ):
            if (
                type(values) is not tuple
                or any(type(value) is not str or _HEX_SHA256.fullmatch(value) is None for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"{field_name} must contain unique SHA-256 values")
        if set(self.fit_span_sha256s) & set(self.generalization_span_sha256s):
            raise ValueError("split memberships must be disjoint")

    @property
    def sha256(self) -> str:
        return _namespaced_sha256(
            b"prompt-factor-compositional-v2-split\0",
            {
                "fit": self.fit_span_sha256s,
                "generalization": self.generalization_span_sha256s,
            },
        )


@dataclass(frozen=True, slots=True)
class _BankAnatomy:
    bank_index: int
    name: str
    global_offset: int
    cell_count: int
    namespace: bytes


@dataclass(frozen=True, slots=True)
class _Anatomy:
    banks: tuple[_BankAnatomy, ...]
    input_cell_count: int
    head_count: int
    stop_cell: int
    present_cell: int


_ANATOMY = _Anatomy(
    banks=tuple(
        _BankAnatomy(
            bank_index=index,
            name=name,
            global_offset=index * BANK_CELL_COUNT,
            cell_count=BANK_CELL_COUNT,
            namespace=_BANK_NAMESPACES[index],
        )
        for index, name in enumerate(_BANK_NAMES)
    ),
    input_cell_count=INPUT_CELL_COUNT,
    head_count=HEAD_COUNT,
    stop_cell=STOP_CELL,
    present_cell=PRESENT_CELL,
)


@dataclass(frozen=True, slots=True)
class _FeatureAssembly:
    bank_index: int
    key: tuple[object, ...]
    local_cells: tuple[int, ...]
    global_cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _BankDrive:
    bank_index: int
    active_local_cells: tuple[int, ...]
    first_spike_step: int
    spike_local_cells: tuple[int, ...]
    trace_sha256: str


@dataclass(frozen=True, slots=True)
class _ProjectedSequence:
    canonical_tokens: tuple[str, ...]
    token_assemblies: tuple[_FeatureAssembly, ...]
    slot_assemblies: tuple[_FeatureAssembly, ...]
    transition_assemblies: tuple[_FeatureAssembly, ...]
    active_global_cells: tuple[int, ...]
    bank_drives: tuple[_BankDrive, ...]
    physical_drive_sha256: str

    @property
    def active_cell_count(self) -> int:
        return len(self.active_global_cells)


@lru_cache(maxsize=16_384)
def _assembly_cells(
    bank_index: int,
    key: tuple[object, ...],
) -> tuple[int, ...]:
    anatomy = _ANATOMY.banks[bank_index]
    key_bytes = _canonical_json(key)
    selected: set[int] = set()
    counter = 0
    while len(selected) < ASSEMBLY_ACTIVE_CELL_COUNT:
        digest = sha256()
        digest.update(anatomy.namespace)
        digest.update(key_bytes)
        digest.update(counter.to_bytes(8, "little"))
        block = digest.digest()
        for offset in range(0, len(block), 4):
            selected.add(
                int.from_bytes(block[offset : offset + 4], "little")
                % anatomy.cell_count
            )
            if len(selected) == ASSEMBLY_ACTIVE_CELL_COUNT:
                break
        counter += 1
    return tuple(sorted(selected))


def _feature_assembly(
    bank_index: int,
    key: tuple[object, ...],
) -> _FeatureAssembly:
    anatomy = _ANATOMY.banks[bank_index]
    local_cells = _assembly_cells(bank_index, key)
    return _FeatureAssembly(
        bank_index=bank_index,
        key=key,
        local_cells=local_cells,
        global_cells=tuple(anatomy.global_offset + cell for cell in local_cells),
    )


@lru_cache(maxsize=16_384)
def _drive_bank(
    bank_index: int,
    active_local_cells: tuple[int, ...],
) -> _BankDrive:
    anatomy = _ANATOMY.banks[bank_index]
    if (
        tuple(sorted(set(active_local_cells))) != active_local_cells
        or any(not 0 <= cell < anatomy.cell_count for cell in active_local_cells)
    ):
        raise ValueError("bank drive requires sorted unique local cells")
    currents = np.zeros(anatomy.cell_count, dtype=np.float64)
    if active_local_cells:
        currents[np.asarray(active_local_cells, dtype=np.intp)] = (
            PHYSICAL_DRIVE_CURRENT_PA
        )
    population = AdaptiveGlifPopulation(anatomy.cell_count)
    seen = np.zeros(anatomy.cell_count, dtype=np.bool_)
    first_spike_step = -1
    trace = sha256()
    trace.update(b"prompt-factor-compositional-v2-bank-drive\0")
    trace.update(anatomy.name.encode("ascii"))
    trace.update(np.asarray(active_local_cells, dtype="<i8").tobytes())
    for step in range(PHYSICAL_DRIVE_STEPS):
        fired = np.asarray(population.step(currents), dtype=np.bool_)
        trace.update(np.packbits(fired, bitorder="little").tobytes())
        if np.any(fired):
            if first_spike_step < 0:
                first_spike_step = step
            seen |= fired
    spike_cells = tuple(int(cell) for cell in np.flatnonzero(seen))
    if spike_cells != active_local_cells or (
        active_local_cells and first_spike_step < 0
    ):
        raise RuntimeError("compositional assembly failed fitted-cell drive")
    return _BankDrive(
        bank_index=bank_index,
        active_local_cells=active_local_cells,
        first_spike_step=first_spike_step,
        spike_local_cells=spike_cells,
        trace_sha256=trace.hexdigest(),
    )


@lru_cache(maxsize=8_192)
def _project_tokens(tokens: tuple[str, ...]) -> _ProjectedSequence:
    if (
        not 1 <= len(tokens) <= MAX_RELATIVE_SLOTS
        or tuple(map(_canonical_token, tokens)) != tokens
    ):
        raise ValueError("projector requires one to four canonical tokens")
    token_assemblies = tuple(
        _feature_assembly(_TOKEN_BANK, (token,)) for token in tokens
    )
    slot_assemblies = tuple(
        _feature_assembly(_SLOT_BANK, (slot, token))
        for slot, token in enumerate(tokens)
    )
    transition_assemblies = tuple(
        _feature_assembly(_TRANSITION_BANK, (left, right))
        for left, right in zip(tokens, tokens[1:])
    )
    groups = (token_assemblies, slot_assemblies, transition_assemblies)
    active_by_bank = tuple(
        tuple(
            sorted(
                {
                    cell
                    for assembly in assemblies
                    for cell in assembly.local_cells
                }
            )
        )
        for assemblies in groups
    )
    bank_drives = tuple(
        _drive_bank(bank_index, active)
        for bank_index, active in enumerate(active_by_bank)
    )
    active_global = tuple(
        sorted(
            {
                cell
                for assemblies in groups
                for assembly in assemblies
                for cell in assembly.global_cells
            }
        )
    )
    drive_sha = _namespaced_sha256(
        b"prompt-factor-compositional-v2-projected-drive\0",
        {
            "tokens": tokens,
            "active_global_cells": active_global,
            "bank_traces": tuple(drive.trace_sha256 for drive in bank_drives),
        },
    )
    return _ProjectedSequence(
        canonical_tokens=tokens,
        token_assemblies=token_assemblies,
        slot_assemblies=slot_assemblies,
        transition_assemblies=transition_assemblies,
        active_global_cells=active_global,
        bank_drives=bank_drives,
        physical_drive_sha256=drive_sha,
    )


def _project_prompt(prompt_input: CompositionalPromptInput) -> _ProjectedSequence:
    if not isinstance(prompt_input, CompositionalPromptInput):
        raise TypeError("infer requires CompositionalPromptInput")
    # Authentication hashes and absolute prompt position are intentionally not
    # arguments to the projector cache or any assembly identity.
    return _project_tokens(prompt_input.canonical_tokens)


_COMPILED_RACE = CompiledRestingGlifRace.compile(step_count=RACE_STEPS)


@dataclass(frozen=True, slots=True)
class FactorHeadRace:
    factor: str
    decision: str
    difference_ticks: int
    present_current_pa: float
    stop_current_pa: float
    winner_cell: int
    spike_step: int
    simultaneous_spike_cells: int


def _validate_tick_arrays(
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (HEAD_COUNT, INPUT_CELL_COUNT)
    for name, values in (
        ("excitatory_ticks", excitatory_ticks),
        ("inhibitory_ticks", inhibitory_ticks),
    ):
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if values.shape != expected_shape or values.dtype != TICK_DTYPE:
            raise ValueError(f"{name} must be uint16[24,1536]")
        if np.any(values > TICK_CAP):
            raise ValueError(f"{name} exceeds the tick cap")
    return excitatory_ticks, inhibitory_ticks


def _race_heads(
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    projected: _ProjectedSequence,
) -> tuple[FactorHeadRace, ...]:
    excitatory, inhibitory = _validate_tick_arrays(
        excitatory_ticks, inhibitory_ticks
    )
    active = np.asarray(projected.active_global_cells, dtype=np.intp)
    active_count = int(active.size)
    if active_count < 1:
        raise RuntimeError("compositional projection produced no active cells")
    differences = np.sum(
        excitatory[:, active].astype(np.int32)
        - inhibitory[:, active].astype(np.int32),
        axis=1,
        dtype=np.int64,
    )
    present_currents = np.clip(
        PRESENT_BASE_CURRENT_PA
        + (TICK_CURRENT_PA / active_count) * differences.astype(np.float64),
        PRESENT_BASE_CURRENT_PA,
        PRESENT_MAX_CURRENT_PA,
    )
    currents = np.empty((HEAD_COUNT, 2), dtype=np.float64)
    currents[:, STOP_CELL] = STOP_CURRENT_PA
    currents[:, PRESENT_CELL] = present_currents
    physical = _COMPILED_RACE.race_many(currents, require_unique=False)
    rows: list[FactorHeadRace] = []
    for index, race in enumerate(physical):
        # A tier containing STOP is absent.  PRESENT is legal only as the sole
        # first spiker, so no semantic decision depends on a cell-index tie.
        decision = (
            PRESENT
            if race.winner_cell == PRESENT_CELL
            and race.simultaneous_spike_candidates == 1
            else STOP
        )
        rows.append(
            FactorHeadRace(
                factor=FACTOR_REGISTRY[index],
                decision=decision,
                difference_ticks=int(differences[index]),
                present_current_pa=float(present_currents[index]),
                stop_current_pa=STOP_CURRENT_PA,
                winner_cell=race.winner_cell,
                spike_step=race.spike_step,
                simultaneous_spike_cells=race.simultaneous_spike_candidates,
            )
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _SequenceLearningResult:
    races: tuple[FactorHeadRace, ...]
    excitatory_head_updates: tuple[int, ...]
    inhibitory_head_updates: tuple[int, ...]
    committed_cell_ticks: int


def _increment_tick_row(
    ticks: np.ndarray,
    head: int,
    active: np.ndarray,
) -> int:
    before = ticks[head, active].astype(np.uint32)
    after = np.minimum(before + 1, TICK_CAP).astype(TICK_DTYPE)
    ticks[head, active] = after
    return int(np.sum(after.astype(np.int32) - before.astype(np.int32)))


def _learn_sequence(
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
    prompt_input: CompositionalPromptInput,
    present_factors: tuple[str, ...],
) -> _SequenceLearningResult:
    """Race a sequence snapshot, accumulate local errors, then commit once."""

    excitatory, inhibitory = _validate_tick_arrays(
        excitatory_ticks, inhibitory_ticks
    )
    if not excitatory.flags.writeable or not inhibitory.flags.writeable:
        raise ValueError("learning requires writable tick arrays")
    if (
        type(present_factors) is not tuple
        or len(set(present_factors)) != len(present_factors)
        or any(factor not in _FACTOR_INDEX for factor in present_factors)
    ):
        raise ValueError("present_factors must be unique registry factors")
    projected = _project_prompt(prompt_input)
    # Copying both rows is intentional: every factor race for this sequence
    # observes exactly the state present at sequence entry.
    snapshot_e = excitatory.copy(order="C")
    snapshot_h = inhibitory.copy(order="C")
    races = _race_heads(snapshot_e, snapshot_h, projected)
    teacher = {_FACTOR_INDEX[factor] for factor in present_factors}
    excitatory_heads = tuple(
        index
        for index, race in enumerate(races)
        if index in teacher and race.decision == STOP
    )
    inhibitory_heads = tuple(
        index
        for index, race in enumerate(races)
        if index not in teacher and race.decision == PRESENT
    )
    active = np.asarray(projected.active_global_cells, dtype=np.intp)
    committed = 0
    # No row is touched until every decision above has been materialized.
    for head in excitatory_heads:
        committed += _increment_tick_row(excitatory, head, active)
    for head in inhibitory_heads:
        committed += _increment_tick_row(inhibitory, head, active)
    return _SequenceLearningResult(
        races=races,
        excitatory_head_updates=excitatory_heads,
        inhibitory_head_updates=inhibitory_heads,
        committed_cell_ticks=committed,
    )


@dataclass(frozen=True, slots=True)
class FactorEvent:
    factor: str
    start_token: int
    stop_token: int
    spike_step: int


@dataclass(frozen=True, slots=True)
class CompositionalInference:
    events: tuple[FactorEvent, ...]
    head_races: tuple[FactorHeadRace, ...]
    active_cell_count: int
    bank_active_cell_counts: tuple[int, int, int]
    bank_first_spike_steps: tuple[int, int, int]
    physical_drive_sha256: str
    semantic_sha256: str

    @property
    def present_factors(self) -> tuple[str, ...]:
        return tuple(event.factor for event in self.events)


@dataclass(frozen=True, slots=True)
class BridgeBuildSummary:
    epochs: int
    fit_sequence_count: int
    generalization_sequence_count: int
    matched_contrast_group_count: int
    excitatory_head_update_events: int
    inhibitory_head_update_events: int
    committed_cell_ticks: int


def _array_frame(name: str, values: np.ndarray) -> bytes:
    header = _canonical_json(
        {"name": name, "dtype": values.dtype.str, "shape": values.shape}
    )
    raw = np.ascontiguousarray(values, dtype=TICK_DTYPE).tobytes(order="C")
    return (
        len(header).to_bytes(8, "little")
        + header
        + len(raw).to_bytes(8, "little")
        + raw
    )


def _checkpoint_payload(
    metadata: dict[str, object],
    excitatory_ticks: np.ndarray,
    inhibitory_ticks: np.ndarray,
) -> bytes:
    metadata_bytes = _canonical_json(metadata)
    return b"".join(
        (
            _CHECKPOINT_NAMESPACE,
            len(metadata_bytes).to_bytes(8, "little"),
            metadata_bytes,
            _array_frame("excitatory_ticks", excitatory_ticks),
            _array_frame("inhibitory_ticks", inhibitory_ticks),
        )
    )


class FrozenCompositionalBridge:
    """Read-only 24-head bridge whose semantic boundary is fitted-cell races."""

    __slots__ = (
        "_excitatory_ticks",
        "_inhibitory_ticks",
        "_metadata",
        "_checkpoint_sha256",
        "build_summary",
    )

    def __init__(
        self,
        *,
        excitatory_ticks: np.ndarray,
        inhibitory_ticks: np.ndarray,
        metadata: dict[str, object],
        build_summary: BridgeBuildSummary,
    ) -> None:
        excitatory, inhibitory = _validate_tick_arrays(
            excitatory_ticks, inhibitory_ticks
        )
        self._excitatory_ticks = np.array(
            excitatory, dtype=TICK_DTYPE, order="C", copy=True
        )
        self._inhibitory_ticks = np.array(
            inhibitory, dtype=TICK_DTYPE, order="C", copy=True
        )
        self._excitatory_ticks.setflags(write=False)
        self._inhibitory_ticks.setflags(write=False)
        self._metadata = dict(metadata)
        self.build_summary = build_summary
        self._checkpoint_sha256 = sha256(self.checkpoint_bytes()).hexdigest()

    @property
    def excitatory_ticks(self) -> np.ndarray:
        snapshot = self._excitatory_ticks.copy(order="C")
        snapshot.setflags(write=False)
        return snapshot

    @property
    def inhibitory_ticks(self) -> np.ndarray:
        snapshot = self._inhibitory_ticks.copy(order="C")
        snapshot.setflags(write=False)
        return snapshot

    @property
    def learned_state_bytes(self) -> int:
        return self._excitatory_ticks.nbytes + self._inhibitory_ticks.nbytes

    @property
    def checkpoint_sha256(self) -> str:
        self._authenticate_checkpoint()
        return self._checkpoint_sha256

    def checkpoint_bytes(self) -> bytes:
        return _checkpoint_payload(
            self._metadata,
            self._excitatory_ticks,
            self._inhibitory_ticks,
        )

    def _authenticate_checkpoint(self) -> None:
        if sha256(self.checkpoint_bytes()).hexdigest() != self._checkpoint_sha256:
            raise RuntimeError("frozen compositional checkpoint changed")

    def infer(
        self,
        prompt_input: CompositionalPromptInput,
    ) -> CompositionalInference:
        """Infer ordered factor events without mutating the frozen checkpoint."""

        self._authenticate_checkpoint()
        projected = _project_prompt(prompt_input)
        races = _race_heads(
            self._excitatory_ticks,
            self._inhibitory_ticks,
            projected,
        )
        events = tuple(
            FactorEvent(
                factor=race.factor,
                start_token=prompt_input.start_token,
                stop_token=prompt_input.stop_token,
                spike_step=race.spike_step,
            )
            for race in races
            if race.decision == PRESENT
        )
        semantic_record = {
            "canonical_tokens": prompt_input.canonical_tokens,
            "start_token": prompt_input.start_token,
            "stop_token": prompt_input.stop_token,
            "events": tuple(
                (event.factor, event.start_token, event.stop_token, event.spike_step)
                for event in events
            ),
            "head_currents": tuple(
                (race.factor, race.difference_ticks, race.present_current_pa)
                for race in races
            ),
            "physical_drive_sha256": projected.physical_drive_sha256,
        }
        result = CompositionalInference(
            events=events,
            head_races=races,
            active_cell_count=projected.active_cell_count,
            bank_active_cell_counts=tuple(
                len(drive.active_local_cells) for drive in projected.bank_drives
            ),
            bank_first_spike_steps=tuple(
                drive.first_spike_step for drive in projected.bank_drives
            ),
            physical_drive_sha256=projected.physical_drive_sha256,
            semantic_sha256=_namespaced_sha256(
                b"prompt-factor-compositional-v2-inference\0",
                semantic_record,
            ),
        )
        self._authenticate_checkpoint()
        return result


def _feature_keys(
    tokens: tuple[str, ...],
) -> tuple[set[str], set[tuple[int, str]], set[tuple[str, str]]]:
    return (
        set(tokens),
        set(enumerate(tokens)),
        set(zip(tokens, tokens[1:])),
    )


def _curriculum_sha256(lessons: tuple[CompositionalLesson, ...]) -> str:
    return _namespaced_sha256(
        b"prompt-factor-compositional-v2-curriculum\0",
        tuple(
            {
                "start_token": lesson.prompt_input.start_token,
                "stop_token": lesson.prompt_input.stop_token,
                "canonical_tokens": lesson.prompt_input.canonical_tokens,
                "whole_span_sha256": lesson.prompt_input.whole_span_sha256,
                "audit_prompt_sha256": lesson.prompt_input.audit_prompt_sha256,
                "present_factors": lesson.present_factors,
            }
            for lesson in lessons
        ),
    )


def _validate_curriculum(
    curriculum: Sequence[CompositionalLesson],
    split: CompositionalSplit,
) -> tuple[
    tuple[CompositionalLesson, ...],
    tuple[CompositionalLesson, ...],
    tuple[CompositionalLesson, ...],
]:
    if isinstance(curriculum, (str, bytes)):
        raise TypeError("curriculum must be a sequence of lessons")
    lessons = tuple(curriculum)
    if not lessons or not all(
        isinstance(lesson, CompositionalLesson) for lesson in lessons
    ):
        raise TypeError("curriculum must contain CompositionalLesson values")
    if not isinstance(split, CompositionalSplit):
        raise TypeError("split must be CompositionalSplit")
    identities = tuple(
        lesson.prompt_input.whole_span_sha256 for lesson in lessons
    )
    if len(identities) != len(set(identities)):
        raise ValueError("curriculum whole-span identities must be unique")
    expected = set(identities)
    fit_ids = set(split.fit_span_sha256s)
    generalization_ids = set(split.generalization_span_sha256s)
    if expected != fit_ids | generalization_ids:
        raise ValueError("split must partition the complete curriculum")
    fit = tuple(
        lesson
        for lesson in lessons
        if lesson.prompt_input.whole_span_sha256 in fit_ids
    )
    generalization = tuple(
        lesson
        for lesson in lessons
        if lesson.prompt_input.whole_span_sha256 in generalization_ids
    )
    if any(
        lesson.present_factors not in (("add",), ("subtract",))
        for lesson in lessons
    ):
        raise ValueError("the first compositional slice trains add/subtract only")

    groups: dict[tuple[str, ...], list[CompositionalLesson]] = defaultdict(list)
    for lesson in lessons:
        groups[tuple(sorted(lesson.prompt_input.canonical_tokens))].append(lesson)
    if len(groups) != 6:
        raise ValueError("the slice requires six matched contrast groups")
    fit_groups = 0
    generalization_groups = 0
    for grouped in groups.values():
        if len(grouped) != 2 or {
            lesson.present_factors for lesson in grouped
        } != {("add",), ("subtract",)}:
            raise ValueError("each contrast group needs one add and one subtract sequence")
        memberships = {
            "fit"
            if lesson.prompt_input.whole_span_sha256 in fit_ids
            else "generalization"
            for lesson in grouped
        }
        if len(memberships) != 1:
            raise ValueError("a matched contrast group cannot cross the split")
        fit_groups += int("fit" in memberships)
        generalization_groups += int("generalization" in memberships)
    if fit_groups != 4 or generalization_groups != 2:
        raise ValueError("the slice requires exactly four FIT and two generalization groups")
    for factor in _SLICE_FACTORS:
        if sum(lesson.present_factors == (factor,) for lesson in fit) != 4:
            raise ValueError(f"FIT requires four {factor} sequences")
        if sum(
            lesson.present_factors == (factor,) for lesson in generalization
        ) != 2:
            raise ValueError(f"generalization requires two {factor} sequences")

    fit_sequences = {lesson.prompt_input.canonical_tokens for lesson in fit}
    if any(
        lesson.prompt_input.canonical_tokens in fit_sequences
        for lesson in generalization
    ):
        raise ValueError("generalization exact spans must be absent from FIT")
    fit_features = tuple(set() for _ in range(3))
    for lesson in fit:
        for destination, source in zip(
            fit_features,
            _feature_keys(lesson.prompt_input.canonical_tokens),
            strict=True,
        ):
            destination.update(source)
    for lesson in generalization:
        token_keys, slot_keys, transition_keys = _feature_keys(
            lesson.prompt_input.canonical_tokens
        )
        if not token_keys <= fit_features[0] or not slot_keys <= fit_features[1]:
            raise ValueError("generalization token/slot features must occur in FIT")
        if len(transition_keys & fit_features[2]) < min(2, len(transition_keys)):
            raise ValueError("generalization needs overlapping ordered transitions")
    return lessons, fit, generalization


def build_frozen_bridge(
    curriculum: Sequence[CompositionalLesson],
    split: CompositionalSplit,
) -> FrozenCompositionalBridge:
    """Fit exactly four epochs and return an immutable compositional bridge."""

    lessons, fit, generalization = _validate_curriculum(curriculum, split)
    excitatory = np.zeros((HEAD_COUNT, INPUT_CELL_COUNT), dtype=TICK_DTYPE)
    inhibitory = np.zeros((HEAD_COUNT, INPUT_CELL_COUNT), dtype=TICK_DTYPE)
    ordered_fit = tuple(
        sorted(
            fit,
            key=lambda lesson: (
                lesson.prompt_input.canonical_tokens,
                lesson.present_factors,
            ),
        )
    )
    excitatory_updates = 0
    inhibitory_updates = 0
    committed_ticks = 0
    for _epoch in range(TRAINING_EPOCHS):
        for lesson in ordered_fit:
            learned = _learn_sequence(
                excitatory,
                inhibitory,
                lesson.prompt_input,
                lesson.present_factors,
            )
            excitatory_updates += len(learned.excitatory_head_updates)
            inhibitory_updates += len(learned.inhibitory_head_updates)
            committed_ticks += learned.committed_cell_ticks

    summary = BridgeBuildSummary(
        epochs=TRAINING_EPOCHS,
        fit_sequence_count=len(fit),
        generalization_sequence_count=len(generalization),
        matched_contrast_group_count=6,
        excitatory_head_update_events=excitatory_updates,
        inhibitory_head_update_events=inhibitory_updates,
        committed_cell_ticks=committed_ticks,
    )
    metadata = {
        "schema": "prompt-factor-compositional-bridge-v2-checkpoint",
        "scope": BRIDGE_SCOPE,
        "factor_registry_sha256": FACTOR_REGISTRY_SHA256,
        "head_count": HEAD_COUNT,
        "bank_cell_count": BANK_CELL_COUNT,
        "input_cell_count": INPUT_CELL_COUNT,
        "assembly_active_cell_count": ASSEMBLY_ACTIVE_CELL_COUNT,
        "max_relative_slots": MAX_RELATIVE_SLOTS,
        "tick_dtype": TICK_DTYPE.str,
        "tick_cap": TICK_CAP,
        "tick_current_pa": TICK_CURRENT_PA,
        "present_base_current_pa": PRESENT_BASE_CURRENT_PA,
        "stop_current_pa": STOP_CURRENT_PA,
        "present_max_current_pa": PRESENT_MAX_CURRENT_PA,
        "race_steps": RACE_STEPS,
        "stop_precedes_complete_tie": True,
        "training_epochs": TRAINING_EPOCHS,
        "curriculum_sha256": _curriculum_sha256(lessons),
        "split_sha256": split.sha256,
        "fit_sequence_count": len(fit),
        "generalization_sequence_count": len(generalization),
        "excitatory_head_update_events": excitatory_updates,
        "inhibitory_head_update_events": inhibitory_updates,
        "committed_cell_ticks": committed_ticks,
        "array_names": ("excitatory_ticks", "inhibitory_ticks"),
        "array_shape": (HEAD_COUNT, INPUT_CELL_COUNT),
        "learned_state_bytes": LEARNED_STATE_BYTES,
    }
    return FrozenCompositionalBridge(
        excitatory_ticks=excitatory,
        inhibitory_ticks=inhibitory,
        metadata=metadata,
        build_summary=summary,
    )


__all__ = [
    "ASSEMBLY_ACTIVE_CELL_COUNT",
    "BANK_CELL_COUNT",
    "BRIDGE_SCOPE",
    "FACTOR_REGISTRY",
    "FACTOR_REGISTRY_SHA256",
    "HEAD_COUNT",
    "INPUT_CELL_COUNT",
    "LEARNED_STATE_BYTES",
    "MAX_RELATIVE_SLOTS",
    "PRESENT",
    "PRESENT_BASE_CURRENT_PA",
    "PRESENT_CELL",
    "PRESENT_MAX_CURRENT_PA",
    "RACE_STEPS",
    "STOP",
    "STOP_CELL",
    "STOP_CURRENT_PA",
    "TICK_CAP",
    "TICK_CURRENT_PA",
    "TICK_DTYPE",
    "TRAINING_EPOCHS",
    "BridgeBuildSummary",
    "CompositionalInference",
    "CompositionalLesson",
    "CompositionalPromptInput",
    "CompositionalSplit",
    "FactorEvent",
    "FactorHeadRace",
    "FrozenCompositionalBridge",
    "build_frozen_bridge",
]
