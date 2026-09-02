"""Task-neutral variable-length fragment memory for assembly projection.

The memory grows only from complete successful pairs of opaque afferent and
output token sequences.  Ordered tokenizer n-grams recruit afferent cells;
maximal common output context separates independently selectable optional
chunks.  The resulting projection exposes every learned owner and signed drive
to the recurrent stack.  It never selects a winner itself.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256

_SCHEMA = "humanbrain-factorized-fragment-memory-v2"
_FactorKey = tuple[str, tuple[int, ...], tuple[tuple[int, ...], ...]]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(namespace: bytes, value: object) -> str:
    return sha256(
        namespace + b"\0" + _canonical_json(value).encode("ascii")
    ).hexdigest()


def _opaque_tokens(value: Sequence[int], *, name: str) -> tuple[int, ...]:
    tokens = tuple(value)
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError(f"{name} must contain nonnegative integer tokens")
    return tokens


def _lcs(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    """Return one deterministic maximal ordered common token context."""

    lengths = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for left_index, left_value in enumerate(left, start=1):
        for right_index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                lengths[left_index][right_index] = (
                    lengths[left_index - 1][right_index - 1] + 1
                )
            else:
                lengths[left_index][right_index] = max(
                    lengths[left_index - 1][right_index],
                    lengths[left_index][right_index - 1],
                )
    result: list[int] = []
    left_index = len(left)
    right_index = len(right)
    while left_index and right_index:
        if left[left_index - 1] == right[right_index - 1]:
            result.append(left[left_index - 1])
            left_index -= 1
            right_index -= 1
        elif lengths[left_index - 1][right_index] >= lengths[left_index][
            right_index - 1
        ]:
            left_index -= 1
        else:
            right_index -= 1
    result.reverse()
    return tuple(result)


def _optional_chunks(
    sequence: Sequence[int],
    backbone: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    chunks: list[tuple[int, ...]] = []
    cursor = 0
    for token in backbone:
        scan = cursor
        while scan < len(sequence) and sequence[scan] != token:
            scan += 1
        if scan >= len(sequence):
            raise RuntimeError("fragment backbone is absent from a learned sequence")
        chunks.append(tuple(sequence[cursor:scan]))
        cursor = scan + 1
    chunks.append(tuple(sequence[cursor:]))
    return tuple(chunks)


def _ordered_ngrams(
    tokens: Sequence[int],
    max_order: int,
    boundary_tokens: frozenset[int] = frozenset(),
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        ngram
        for ngram, _, _ in _ordered_ngram_occurrences(
            tokens,
            max_order,
            boundary_tokens,
        )
    )


def _ordered_ngram_occurrences(
    tokens: Sequence[int],
    max_order: int,
    boundary_tokens: frozenset[int] = frozenset(),
) -> tuple[tuple[tuple[int, ...], int, int], ...]:
    result: list[tuple[tuple[int, ...], int, int]] = []
    for start in range(len(tokens)):
        if tokens[start] in boundary_tokens:
            continue
        for order in range(1, min(max_order, len(tokens) - start) + 1):
            if tokens[start + order - 1] in boundary_tokens:
                break
            result.append(
                (tuple(tokens[start : start + order]), start, start + order)
            )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FragmentMemoryConfig:
    """Explicit structural capacities, independent of raw byte vocabulary size."""

    max_examples: int = 512
    max_ngram_order: int = 3
    max_afferent_cells: int = 65_536
    max_fragment_cells: int = 8_192
    max_candidate_owners: int = 4_096
    max_anchor_tokens: int = 256
    drive_scale: float = 5.0

    def __post_init__(self) -> None:
        bounds = (
            self.max_examples,
            self.max_ngram_order,
            self.max_afferent_cells,
            self.max_fragment_cells,
            self.max_candidate_owners,
            self.max_anchor_tokens,
        )
        if any(type(value) is not int or value < 1 for value in bounds):
            raise ValueError("fragment memory capacities must be positive integers")
        if not isinstance(self.drive_scale, float | int) or self.drive_scale <= 1.0:
            raise ValueError("drive_scale must exceed the physical STOP baseline")


@dataclass(frozen=True, slots=True)
class _FragmentSuccess:
    observation_tokens: tuple[int, ...]
    output_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        _opaque_tokens(self.observation_tokens, name="observation_tokens")
        _opaque_tokens(self.output_tokens, name="output_tokens")


@dataclass(frozen=True, slots=True)
class _RoleLoopSpec:
    """One learned inventory-to-end run encoded by an internal marker token."""

    marker_token: int
    start_ordinal: int
    separator_tokens: tuple[int, ...]
    max_roles: int = 256

    def __post_init__(self) -> None:
        if type(self.marker_token) is not int or self.marker_token < 0:
            raise ValueError("role-loop marker token must be nonnegative")
        if type(self.start_ordinal) is not int or self.start_ordinal < 0:
            raise ValueError("role-loop start ordinal must be nonnegative")
        if any(type(token) is not int or token < 0 for token in self.separator_tokens):
            raise ValueError("role-loop separator tokens must be nonnegative")
        if type(self.max_roles) is not int or self.max_roles <= self.start_ordinal:
            raise ValueError("role-loop role capacity must exceed its start ordinal")


@dataclass(frozen=True, slots=True)
class _ProjectedRoleLoop:
    marker_cell: int
    start_ordinal: int
    separator_cells: tuple[int, ...]
    role_cells: tuple[int, ...]
    prefix_cells: tuple[int, ...]
    digit_cells: tuple[int, ...]
    suffix_cell: int


@dataclass(frozen=True, slots=True)
class _ProjectedSlotCandidate:
    owner_cell: int
    cohort_cell: int
    signed_drive: float
    fragment_cell_path: tuple[int, ...]
    slots: tuple[_ProjectedSlot | None, ...]
    canonical_continuation: int
    span_edge: str


@dataclass(frozen=True, slots=True)
class _ProjectedSlot:
    slot_cell: int
    factor_cell: int
    candidates: tuple[_ProjectedSlotCandidate, ...]


@dataclass(frozen=True, slots=True)
class _ProjectedFactorRefinement:
    relay_cell: int
    source_factor_cell: int
    source_cohort_cell: int
    source_support_count: int
    target_factor_cell: int
    target_cohort_cell: int


@dataclass(frozen=True, slots=True)
class _ProjectedProgram:
    owner_cell: int
    root_edge: str
    signed_drive: float
    fragment_cell_path: tuple[int, ...]
    slots: tuple[_ProjectedSlot | None, ...]
    direct: bool


@dataclass(frozen=True, slots=True)
class _FragmentProjection:
    owner_digest: str
    memory_digest: str
    query_sha256: str
    ordered_afferent_cells: tuple[int, ...]
    programs: tuple[_ProjectedProgram, ...]
    refinements: tuple[_ProjectedFactorRefinement, ...]
    role_loops: tuple[_ProjectedRoleLoop, ...]
    stop_owner_cell: int
    stop_signed_drive: float
    sha256: str

    @classmethod
    def seal(
        cls,
        *,
        owner_digest: str,
        memory_digest: str,
        query_sha256: str,
        ordered_afferent_cells: tuple[int, ...],
        programs: tuple[_ProjectedProgram, ...],
        refinements: tuple[_ProjectedFactorRefinement, ...],
        role_loops: tuple[_ProjectedRoleLoop, ...],
        stop_owner_cell: int,
        stop_signed_drive: float,
    ) -> _FragmentProjection:
        payload = {
            "memory_digest": memory_digest,
            "ordered_afferent_cells": ordered_afferent_cells,
            "owner_digest": owner_digest,
            "programs": _program_payload(programs),
            "query_sha256": query_sha256,
            "refinements": [asdict(item) for item in refinements],
            "role_loops": [asdict(item) for item in role_loops],
            "stop_owner_cell": stop_owner_cell,
            "stop_signed_drive": stop_signed_drive,
        }
        return cls(
            owner_digest=owner_digest,
            memory_digest=memory_digest,
            query_sha256=query_sha256,
            ordered_afferent_cells=ordered_afferent_cells,
            programs=programs,
            refinements=refinements,
            role_loops=role_loops,
            stop_owner_cell=stop_owner_cell,
            stop_signed_drive=stop_signed_drive,
            sha256=_digest(b"humanbrain-fragment-projection-v1", payload),
        )

    def verify(self) -> None:
        rebuilt = self.seal(
            owner_digest=self.owner_digest,
            memory_digest=self.memory_digest,
            query_sha256=self.query_sha256,
            ordered_afferent_cells=self.ordered_afferent_cells,
            programs=self.programs,
            refinements=self.refinements,
            role_loops=self.role_loops,
            stop_owner_cell=self.stop_owner_cell,
            stop_signed_drive=self.stop_signed_drive,
        )
        if rebuilt.sha256 != self.sha256:
            raise ValueError("fragment projection seal is not authentic")


def _program_payload(programs: Sequence[_ProjectedProgram]) -> list[object]:
    return [
        {
            "direct": program.direct,
            "fragment_cell_path": program.fragment_cell_path,
            "owner_cell": program.owner_cell,
            "root_edge": program.root_edge,
            "signed_drive": program.signed_drive,
            "slots": [
                None
                if slot is None
                else {
                    "candidates": [asdict(candidate) for candidate in slot.candidates],
                    "factor_cell": slot.factor_cell,
                    "slot_cell": slot.slot_cell,
                }
                for slot in program.slots
            ],
        }
        for program in programs
    ]


@dataclass(frozen=True, slots=True)
class _LearnedCandidate:
    owner_cell: int
    fragment_cell_path: tuple[int, ...]
    slots: tuple[_LearnedSlot | None, ...]
    canonical_continuation: int
    span_edge: str
    support_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _LearnedSlot:
    slot_cell: int
    candidates: tuple[_LearnedCandidate, ...]


@dataclass(frozen=True, slots=True)
class _LearnedProgram:
    owner_cell: int
    root_edge: str
    fragment_cell_path: tuple[int, ...]
    slots: tuple[_LearnedSlot | None, ...]
    direct: bool


@dataclass(frozen=True, slots=True)
class _SpanRow:
    sequence: tuple[int, ...]
    observation_tokens: tuple[int, ...]
    row_id: int


class FactorizedFragmentMemory:
    """Learn ordered afferents and expose unfiltered fragment projections."""

    def __init__(
        self,
        config: FragmentMemoryConfig | None = None,
        *,
        transport_token_ids: Iterable[int] = (),
        boundary_token_ids: Iterable[int] = (),
    ) -> None:
        self.config = config or FragmentMemoryConfig()
        self._transport_token_ids = frozenset(transport_token_ids)
        self._boundary_token_ids = frozenset(boundary_token_ids)
        if any(
            type(token) is not int or token < 0
            for token in self._transport_token_ids
        ):
            raise ValueError("transport token IDs must be nonnegative integers")
        if any(
            type(token) is not int or token < 0
            for token in self._boundary_token_ids
        ):
            raise ValueError("boundary token IDs must be nonnegative integers")
        self._cell_by_key: dict[tuple[object, ...], int] = {}
        self._key_by_cell: dict[int, tuple[object, ...]] = {}
        self._atom_by_cell: dict[int, tuple[str, int]] = {}
        self._afferent_by_ngram: dict[tuple[int, ...], int] = {}
        self._feature_counts: dict[str, Counter[int]] = defaultdict(Counter)
        self._edge_observations: Counter[str] = Counter()
        self._programs: tuple[_LearnedProgram, ...] = ()
        self._factor_root_key = ""
        self._factor_by_slot: dict[int, _FactorKey] = {}
        self._factor_cell_by_slot: dict[int, int] = {}
        self._cohort_cell_by_candidate: dict[tuple[int, int], int] = {}
        self._feature_factor_weight: dict[tuple[int, _FactorKey], Fraction] = {}
        self._factor_refinements: tuple[_ProjectedFactorRefinement, ...] = ()
        self._role_loops: tuple[_ProjectedRoleLoop, ...] = ()
        self._stop_owner_cell = self._cell_for(("owner", "stop"))
        self._prior_contribution_scan_count = 0
        self._last_coactivity_write_count = 0
        self._owner_digest = ""

    @classmethod
    def grow(
        cls,
        successes: Iterable[_FragmentSuccess],
        config: FragmentMemoryConfig | None = None,
        *,
        transport_token_ids: Iterable[int] = (),
        boundary_token_ids: Iterable[int] = (),
        role_loops: Iterable[_RoleLoopSpec] = (),
    ) -> FactorizedFragmentMemory:
        rows = tuple(
            sorted(
                set(successes),
                key=lambda row: (row.output_tokens, row.observation_tokens),
            )
        )
        if not rows:
            raise ValueError("fragment memory requires at least one public success")
        loop_specs = tuple(
            sorted(
                set(role_loops),
                key=lambda item: (
                    item.marker_token,
                    item.start_ordinal,
                    item.separator_tokens,
                    item.max_roles,
                ),
            )
        )
        if any(not isinstance(item, _RoleLoopSpec) for item in loop_specs):
            raise TypeError("role_loops must contain _RoleLoopSpec rows")
        if len({item.marker_token for item in loop_specs}) != len(loop_specs):
            raise ValueError("role-loop marker tokens must be unique")
        memory = cls(
            config,
            transport_token_ids=transport_token_ids,
            boundary_token_ids=boundary_token_ids,
        )
        if len(rows) > memory.config.max_examples:
            raise RuntimeError("fragment example capacity is exhausted")

        ngrams = {
            ngram
            for row in rows
            for ngram in _ordered_ngrams(
                row.observation_tokens,
                memory.config.max_ngram_order,
                memory._boundary_token_ids,
            )
        }
        if len(ngrams) > memory.config.max_afferent_cells:
            raise RuntimeError("fragment afferent-cell capacity is exhausted")
        for ngram in sorted(ngrams):
            memory._afferent_by_ngram[ngram] = memory._cell_for(
                ("afferent", *ngram)
            )

        programs: list[_LearnedProgram] = []
        multi_rows = tuple(row for row in rows if len(row.output_tokens) > 1)
        variable = (
            len(multi_rows) >= 3
            and (
                len({len(row.observation_tokens) for row in multi_rows}) > 1
                or len({len(row.output_tokens) for row in multi_rows}) > 1
            )
        )
        if variable:
            template = memory._grow_template(multi_rows)
            if template is not None:
                programs.append(template)
        programs.extend(memory._grow_direct(rows))
        owner_count = sum(
            1 for key in memory._cell_by_key if key and key[0] == "owner"
        )
        if owner_count > memory.config.max_candidate_owners:
            raise RuntimeError("fragment candidate-owner capacity is exhausted")
        memory._programs = tuple(sorted(programs, key=lambda program: program.owner_cell))
        memory._role_loops = tuple(
            memory._recruit_role_loop(spec) for spec in loop_specs
        )
        memory._refresh_feature_ownership()
        memory._owner_digest = memory._compute_owner_digest()
        return memory

    @property
    def has_variable_template(self) -> bool:
        return any(not program.direct for program in self._programs)

    @property
    def owner_digest(self) -> str:
        return self._owner_digest

    @property
    def model_digest(self) -> str:
        return _digest(
            b"humanbrain-factorized-fragment-state-v1",
            {
                "edge_observations": tuple(sorted(self._edge_observations.items())),
                "feature_counts": tuple(
                    (edge, tuple(sorted(counts.items())))
                    for edge, counts in sorted(self._feature_counts.items())
                ),
                "factor_by_slot": tuple(sorted(self._factor_by_slot.items())),
                "feature_factor_weight": tuple(
                    (
                        cell,
                        factor,
                        weight.numerator,
                        weight.denominator,
                    )
                    for (cell, factor), weight in sorted(
                        self._feature_factor_weight.items()
                    )
                ),
                "factor_refinements": tuple(
                    asdict(item) for item in self._factor_refinements
                ),
                "role_loops": tuple(asdict(item) for item in self._role_loops),
                "owner_digest": self._owner_digest,
            },
        )

    @property
    def projected_cell_ids(self) -> tuple[int, ...]:
        cells = {self._stop_owner_cell}
        for program in self._programs:
            cells.add(program.owner_cell)
            cells.update(program.fragment_cell_path)
            for slot in program.slots:
                if slot is None:
                    continue
                self._collect_slot_cells(slot, cells)
        for refinement in self._factor_refinements:
            cells.update(
                (
                    refinement.relay_cell,
                    refinement.source_factor_cell,
                    refinement.source_cohort_cell,
                    refinement.target_factor_cell,
                    refinement.target_cohort_cell,
                )
            )
        for loop in self._role_loops:
            cells.add(loop.marker_cell)
            cells.update(loop.separator_cells)
            cells.update(loop.role_cells)
            cells.update(loop.prefix_cells)
            cells.update(loop.digit_cells)
            cells.add(loop.suffix_cell)
        return tuple(sorted(cells))

    def _collect_slot_cells(self, slot: _LearnedSlot, cells: set[int]) -> None:
        cells.add(slot.slot_cell)
        cells.add(self._factor_cell_by_slot[slot.slot_cell])
        for candidate in slot.candidates:
            cells.add(candidate.owner_cell)
            cells.add(
                self._cohort_cell_by_candidate[
                    (slot.slot_cell, candidate.owner_cell)
                ]
            )
            cells.add(candidate.canonical_continuation)
            cells.update(candidate.fragment_cell_path)
            for child in candidate.slots:
                if child is not None:
                    self._collect_slot_cells(child, cells)

    @property
    def prior_contribution_scan_count(self) -> int:
        return self._prior_contribution_scan_count

    @property
    def last_coactivity_write_count(self) -> int:
        return self._last_coactivity_write_count

    def project(
        self,
        observation_tokens: Sequence[int],
        *,
        query_sha256: str,
    ) -> _FragmentProjection:
        tokens = _opaque_tokens(observation_tokens, name="observation_tokens")
        if not isinstance(query_sha256, str) or not query_sha256:
            raise ValueError("query_sha256 must be non-empty")
        occurrences = tuple(
            (cell, start, end)
            for ngram, start, end in _ordered_ngram_occurrences(
                tokens,
                self.config.max_ngram_order,
                self._boundary_token_ids,
            )
            if (cell := self._afferent_by_ngram.get(ngram)) is not None
        )
        ordered_afferents = tuple(cell for cell, _, _ in occurrences)
        intervals_by_cell: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for cell, start, end in occurrences:
            intervals_by_cell[cell].append((start, end))
        root_edges = tuple(program.root_edge for program in self._programs)
        root_feature_weights = self._positional_feature_weights(
            root_edges,
            intervals_by_cell,
        )
        root_drives = self._bounded_competition_drives(
            tuple(
                self._signed_drive(
                    program.root_edge,
                    root_edges,
                    root_feature_weights,
                )
                for program in self._programs
            )
        )
        projected_programs: list[_ProjectedProgram] = []
        for program, root_drive in zip(
            self._programs,
            root_drives,
            strict=True,
        ):
            projected_programs.append(
                _ProjectedProgram(
                    owner_cell=program.owner_cell,
                    root_edge=program.root_edge,
                    signed_drive=root_drive,
                    fragment_cell_path=program.fragment_cell_path,
                    slots=tuple(
                        None
                        if slot is None
                        else self._project_slot(slot, intervals_by_cell)
                        for slot in program.slots
                    ),
                    direct=program.direct,
                )
            )
        return _FragmentProjection.seal(
            owner_digest=self._owner_digest,
            memory_digest=self.model_digest,
            query_sha256=query_sha256,
            ordered_afferent_cells=ordered_afferents,
            programs=tuple(projected_programs),
            refinements=self._factor_refinements,
            role_loops=self._role_loops,
            stop_owner_cell=self._stop_owner_cell,
            stop_signed_drive=1.0,
        )

    def _project_slot(
        self,
        slot: _LearnedSlot,
        intervals_by_cell: Mapping[int, Sequence[tuple[int, int]]],
    ) -> _ProjectedSlot:
        competitor_edges = tuple(candidate.span_edge for candidate in slot.candidates)
        singleton_drive = (
            float(self.config.drive_scale)
            if len(slot.candidates) == 1
            else None
        )
        factor = self._factor_by_slot[slot.slot_cell]
        feature_weights = self._positional_feature_weights(
            competitor_edges,
            intervals_by_cell,
            factor=factor,
        )
        candidate_drives = (
            (singleton_drive,)
            if singleton_drive is not None
            else self._bounded_competition_drives(
                tuple(
                    self._signed_drive(
                        candidate.span_edge,
                        competitor_edges,
                        feature_weights,
                    )
                    for candidate in slot.candidates
                )
            )
        )
        return _ProjectedSlot(
            slot_cell=slot.slot_cell,
            factor_cell=self._factor_cell_by_slot[slot.slot_cell],
            candidates=tuple(
                _ProjectedSlotCandidate(
                    owner_cell=candidate.owner_cell,
                    cohort_cell=self._cohort_cell_by_candidate[
                        (slot.slot_cell, candidate.owner_cell)
                    ],
                    signed_drive=candidate_drive,
                    fragment_cell_path=candidate.fragment_cell_path,
                    slots=tuple(
                        None
                        if child is None
                        else self._project_slot(child, intervals_by_cell)
                        for child in candidate.slots
                    ),
                    canonical_continuation=candidate.canonical_continuation,
                    span_edge=candidate.span_edge,
                )
                for candidate, candidate_drive in zip(
                    slot.candidates,
                    candidate_drives,
                    strict=True,
                )
            ),
        )

    def materialize(
        self,
        fragment_cells: Sequence[int],
        observation_tokens: Sequence[int],
    ) -> tuple[int, ...]:
        observation = _opaque_tokens(observation_tokens, name="observation_tokens")
        result: list[int] = []
        for cell in fragment_cells:
            atom = self._atom_by_cell.get(cell)
            if atom is None:
                raise ValueError("projection emitted a non-fragment owner cell")
            kind, value = atom
            if kind == "literal":
                result.append(value)
            elif 0 <= value < len(observation):
                result.append(observation[value])
            else:
                raise ValueError("projection copy role is outside the observation")
        return tuple(result)

    def observe_emitted_success(
        self,
        observation_tokens: Sequence[int],
        selected_span_edges: Sequence[str],
    ) -> None:
        """Increment only selected span paths; never replay historical rows."""

        tokens = _opaque_tokens(observation_tokens, name="observation_tokens")
        selected = tuple(dict.fromkeys(selected_span_edges))
        known_edges: set[str] = set()
        for program in self._programs:
            for slot in program.slots:
                if slot is not None:
                    self._collect_span_edges(slot, known_edges)
        if any(edge not in known_edges for edge in selected):
            raise ValueError("selected span edge is not owned by fragment memory")
        query_cells = {
            cell
            for ngram in _ordered_ngrams(
                tokens,
                self.config.max_ngram_order,
                self._boundary_token_ids,
            )
            if (cell := self._afferent_by_ngram.get(ngram)) is not None
        }
        self._last_coactivity_write_count = 0
        for edge in selected:
            self._edge_observations[edge] += 1
            self._last_coactivity_write_count += 1
            counts = self._feature_counts[edge]
            for cell in query_cells:
                counts[cell] += 1
                self._last_coactivity_write_count += 1
        self._refresh_feature_ownership()

    def _collect_span_edges(self, slot: _LearnedSlot, edges: set[str]) -> None:
        for candidate in slot.candidates:
            edges.add(candidate.span_edge)
            for child in candidate.slots:
                if child is not None:
                    self._collect_span_edges(child, edges)

    def _grow_template(
        self,
        rows: Sequence[_FragmentSuccess],
    ) -> _LearnedProgram | None:
        backbone = rows[0].output_tokens
        for row in rows[1:]:
            backbone = _lcs(backbone, row.output_tokens)
        if not backbone:
            return None
        chunks_by_row = tuple(
            _optional_chunks(row.output_tokens, backbone) for row in rows
        )
        backbone_cells = tuple(self._literal_cell(token) for token in backbone)
        root_key = _digest(b"fragment-template-root-v1", backbone)[:24]
        self._factor_root_key = root_key
        root_edge = f"projected-root:{root_key}"
        owner_cell = self._cell_for(("owner", "template", root_key))
        slots: list[_LearnedSlot | None] = []
        for slot_index in range(len(backbone) + 1):
            chunks = tuple(row_chunks[slot_index] for row_chunks in chunks_by_row)
            if len(set(chunks)) == 1 and not chunks[0]:
                slots.append(None)
                continue
            continuation = (
                backbone_cells[slot_index]
                if slot_index < len(backbone_cells)
                else owner_cell
            )
            slots.append(
                self._build_slot(
                    tuple(
                        _SpanRow(chunk, row.observation_tokens, row_id)
                        for row_id, (row, chunk) in enumerate(
                            zip(rows, chunks, strict=True)
                        )
                    ),
                    continuation=continuation,
                    path_key=(root_key, slot_index),
                )
            )
        self._observe_program_root(root_edge, rows)
        return _LearnedProgram(
            owner_cell=owner_cell,
            root_edge=root_edge,
            fragment_cell_path=backbone_cells,
            slots=tuple(slots),
            direct=False,
        )

    def _build_slot(
        self,
        rows: Sequence[_SpanRow],
        *,
        continuation: int,
        path_key: tuple[object, ...],
    ) -> _LearnedSlot | None:
        if not rows or all(not row.sequence for row in rows):
            return None
        slot_key = _digest(b"fragment-recursive-slot-v1", path_key)[:24]
        candidates = self._build_candidates(
            rows,
            continuation=continuation,
            path_key=(*path_key, slot_key),
        )
        return _LearnedSlot(
            slot_cell=self._cell_for(("slot", *path_key, slot_key)),
            candidates=tuple(sorted(candidates, key=lambda item: item.owner_cell)),
        )

    def _build_candidates(
        self,
        rows: Sequence[_SpanRow],
        *,
        continuation: int,
        path_key: tuple[object, ...],
    ) -> tuple[_LearnedCandidate, ...]:
        anchor = self._best_anchor(rows)
        if anchor is None:
            grouped: dict[tuple[int, ...], list[_SpanRow]] = defaultdict(list)
            for row in rows:
                grouped[row.sequence].append(row)
            return tuple(
                self._literal_candidate(
                    sequence,
                    sequence_rows,
                    continuation=continuation,
                    path_key=(*path_key, "literal", sequence),
                )
                for sequence, sequence_rows in sorted(grouped.items())
            )

        containing: list[tuple[_SpanRow, int]] = []
        remainder: list[_SpanRow] = []
        for row in rows:
            offset = self._first_occurrence(row.sequence, anchor)
            if offset is None:
                remainder.append(row)
            else:
                containing.append((row, offset))
        if len(containing) < 2:
            raise RuntimeError("recursive fragment anchor lost its learned support")

        anchor_key = _digest(
            b"fragment-recursive-anchor-v1",
            (*path_key, anchor),
        )[:24]
        owner_cell = self._cell_for(("owner", "anchor", *path_key, anchor_key))
        span_edge = f"projected-span:{anchor_key}"
        anchor_cells = tuple(self._literal_cell(token) for token in anchor)
        prefix_rows = tuple(
            _SpanRow(row.sequence[:offset], row.observation_tokens, row.row_id)
            for row, offset in containing
        )
        suffix_rows = tuple(
            _SpanRow(
                row.sequence[offset + len(anchor) :],
                row.observation_tokens,
                row.row_id,
            )
            for row, offset in containing
        )
        prefix_slot = self._build_slot(
            prefix_rows,
            continuation=anchor_cells[0],
            path_key=(*path_key, anchor_key, "prefix"),
        )
        suffix_slot = self._build_slot(
            suffix_rows,
            continuation=owner_cell,
            path_key=(*path_key, anchor_key, "suffix"),
        )
        slots = (
            prefix_slot,
            *(None for _ in range(len(anchor_cells) - 1)),
            suffix_slot,
        )
        for row, _ in containing:
            self._observe_edge(span_edge, row.observation_tokens)
        result = [
            _LearnedCandidate(
                owner_cell=owner_cell,
                fragment_cell_path=anchor_cells,
                slots=slots,
                canonical_continuation=continuation,
                span_edge=span_edge,
                support_rows=tuple(sorted(row.row_id for row, _ in containing)),
            )
        ]
        if remainder:
            result.extend(
                self._build_candidates(
                    remainder,
                    continuation=continuation,
                    path_key=(*path_key, anchor_key, "remainder"),
                )
            )
        return tuple(result)

    def _literal_candidate(
        self,
        sequence: tuple[int, ...],
        rows: Sequence[_SpanRow],
        *,
        continuation: int,
        path_key: tuple[object, ...],
    ) -> _LearnedCandidate:
        candidate_key = _digest(b"fragment-literal-candidate-v1", path_key)[:24]
        span_edge = f"projected-span:{candidate_key}"
        for row in rows:
            self._observe_edge(span_edge, row.observation_tokens)
        return _LearnedCandidate(
            owner_cell=self._cell_for(("owner", "literal", *path_key, candidate_key)),
            fragment_cell_path=tuple(self._literal_cell(token) for token in sequence),
            slots=(),
            canonical_continuation=continuation,
            span_edge=span_edge,
            support_rows=tuple(sorted(row.row_id for row in rows)),
        )

    def _best_anchor(self, rows: Sequence[_SpanRow]) -> tuple[int, ...] | None:
        support: Counter[tuple[int, ...]] = Counter()
        for row in rows:
            seen: set[tuple[int, ...]] = set()
            limit = min(len(row.sequence), self.config.max_anchor_tokens)
            for start in range(len(row.sequence)):
                for length in range(1, min(limit, len(row.sequence) - start) + 1):
                    seen.add(row.sequence[start : start + length])
            support.update(seen)
        candidates = tuple(
            span
            for span, count in support.items()
            if count >= 2
            and span
            and not all(token in self._transport_token_ids for token in span)
        )
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda span: (
                -((support[span] - 1) * len(span)),
                -len(span),
                -support[span],
                span,
            ),
        )

    @staticmethod
    def _first_occurrence(
        sequence: Sequence[int],
        anchor: Sequence[int],
    ) -> int | None:
        width = len(anchor)
        for start in range(len(sequence) - width + 1):
            if tuple(sequence[start : start + width]) == tuple(anchor):
                return start
        return None

    def _grow_direct(
        self,
        rows: Sequence[_FragmentSuccess],
    ) -> tuple[_LearnedProgram, ...]:
        grouped: dict[tuple[object, ...], list[_FragmentSuccess]] = defaultdict(list)
        for row in rows:
            if len(row.output_tokens) != 1:
                continue
            output = row.output_tokens[0]
            matches = tuple(
                index
                for index, token in enumerate(row.observation_tokens)
                if token == output
            )
            atom = ("copy", matches[0]) if len(matches) == 1 else ("literal", output)
            grouped[(len(row.observation_tokens), atom)].append(row)
        programs: list[_LearnedProgram] = []
        for key, direct_rows in sorted(grouped.items(), key=str):
            direct_key = _digest(b"fragment-direct-v1", key)[:24]
            root_edge = f"projected-root:{direct_key}"
            _, atom = key
            assert isinstance(atom, tuple)
            fragment_cell = self._atom_cell(atom[0], atom[1])
            self._observe_program_root(root_edge, direct_rows)
            programs.append(
                _LearnedProgram(
                    owner_cell=self._cell_for(("owner", "direct", direct_key)),
                    root_edge=root_edge,
                    fragment_cell_path=(fragment_cell,),
                    slots=(),
                    direct=True,
                )
            )
        return tuple(programs)

    def _observe_program_root(
        self,
        edge: str,
        rows: Sequence[_FragmentSuccess],
    ) -> None:
        for row in rows:
            self._observe_edge(edge, row.observation_tokens)

    def _observe_edge(self, edge: str, observation_tokens: Sequence[int]) -> None:
        cells = {
            self._afferent_by_ngram[ngram]
            for ngram in _ordered_ngrams(
                observation_tokens,
                self.config.max_ngram_order,
                self._boundary_token_ids,
            )
        }
        self._edge_observations[edge] += 1
        for cell in cells:
            self._feature_counts[edge][cell] += 1

    def _signed_drive(
        self,
        edge: str,
        competitors: Sequence[str],
        feature_weights: Mapping[int, Fraction],
    ) -> float:
        own_total = self._edge_observations.get(edge, 0)
        if own_total < 1 or not feature_weights:
            return 0.0

        def affinity(candidate: str) -> float:
            observations = self._edge_observations.get(candidate, 0)
            if observations < 1:
                return 0.0
            return sum(
                float(feature_weight)
                * self._feature_counts[candidate].get(cell, 0)
                / observations
                for cell, feature_weight in feature_weights.items()
            )

        own_affinity = affinity(edge)
        competitor_affinity = max(
            (
                affinity(competitor)
                for competitor in competitors
                if competitor != edge
            ),
            default=0.0,
        )
        return float(
            self.config.drive_scale * (own_affinity - competitor_affinity)
        )

    def _bounded_competition_drives(
        self,
        drives: Sequence[float],
    ) -> tuple[float, ...]:
        """Give one uniquely supported race a fixed positive synaptic budget."""

        raw = tuple(float(drive) for drive in drives)
        maximum = max(raw, default=0.0)
        if maximum <= 0.0 or sum(drive == maximum for drive in raw) != 1:
            return tuple(0.0 for _ in raw)
        positive_total = sum(max(0.0, drive) for drive in raw)
        if positive_total <= 0.0:
            return tuple(0.0 for _ in raw)
        return tuple(
            self.config.drive_scale * max(0.0, drive) / positive_total
            for drive in raw
        )

    def _positional_feature_weights(
        self,
        competitors: Sequence[str],
        intervals_by_cell: Mapping[int, Sequence[tuple[int, int]]],
        *,
        factor: _FactorKey | None = None,
    ) -> dict[int, Fraction]:
        raw_weights: dict[int, Fraction] = {}
        for cell in intervals_by_cell:
            factor_weight = (
                Fraction(1, 1)
                if factor is None
                else self._feature_factor_weight.get(
                    (cell, factor),
                    Fraction(0, 1),
                )
            )
            if factor_weight <= 0:
                continue
            rates = tuple(
                Fraction(
                    self._feature_counts[edge].get(cell, 0),
                    self._edge_observations[edge],
                )
                for edge in competitors
                if self._edge_observations.get(edge, 0) > 0
            )
            if not rates or (len(rates) > 1 and max(rates) == min(rates)):
                continue
            raw_weights[cell] = factor_weight
        if not raw_weights:
            return {}

        parent = {cell: cell for cell in raw_weights}

        def find(cell: int) -> int:
            while parent[cell] != cell:
                parent[cell] = parent[parent[cell]]
                cell = parent[cell]
            return cell

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        occurrences = sorted(
            (start, end, cell)
            for cell in raw_weights
            for start, end in intervals_by_cell[cell]
        )
        active: list[tuple[int, int]] = []
        for start, end, cell in occurrences:
            active = [
                (active_end, active_cell)
                for active_end, active_cell in active
                if active_end > start
            ]
            for _, active_cell in active:
                union(cell, active_cell)
            active.append((end, cell))

        groups: dict[int, list[int]] = defaultdict(list)
        for cell in raw_weights:
            groups[find(cell)].append(cell)
        result: dict[int, Fraction] = {}
        for cells in groups.values():
            total = sum(
                (raw_weights[cell] for cell in cells),
                start=Fraction(0, 1),
            )
            for cell in cells:
                result[cell] = raw_weights[cell] / total
        return result

    def _refresh_feature_ownership(self) -> None:
        """Bind discriminative afferents to one learned support-row factor."""

        slots: list[_LearnedSlot] = []
        for program in self._programs:
            for slot in program.slots:
                if slot is not None:
                    self._collect_slots(slot, slots)
        self._factor_by_slot = {
            slot.slot_cell: self._factor_key(slot) for slot in slots
        }
        self._factor_cell_by_slot = {
            slot.slot_cell: self._cell_for(
                ("factor", *self._factor_by_slot[slot.slot_cell])
            )
            for slot in slots
        }
        self._cohort_cell_by_candidate = {
            (slot.slot_cell, candidate.owner_cell): self._cell_for(
                (
                    "factor-cohort",
                    *self._factor_by_slot[slot.slot_cell],
                    candidate.support_rows,
                )
            )
            for slot in slots
            for candidate in slot.candidates
        }
        self._factor_refinements = self._build_factor_refinements(slots)
        gain_by_feature: dict[int, dict[_FactorKey, Fraction]] = defaultdict(dict)
        for slot in slots:
            edges = tuple(candidate.span_edge for candidate in slot.candidates)
            if len(edges) < 2:
                continue
            factor = self._factor_by_slot[slot.slot_cell]
            features = {
                cell for edge in edges for cell in self._feature_counts[edge]
            }
            for cell in features:
                rates = sorted(
                    (
                        Fraction(
                            self._feature_counts[edge].get(cell, 0),
                            self._edge_observations[edge],
                        )
                        for edge in edges
                        if self._edge_observations.get(edge, 0) > 0
                    ),
                    reverse=True,
                )
                if len(rates) < 2:
                    continue
                gain = rates[0] - rates[1]
                if gain > 0:
                    prior = gain_by_feature[cell].get(factor, Fraction(0, 1))
                    gain_by_feature[cell][factor] = max(prior, gain)
        weights: dict[tuple[int, _FactorKey], Fraction] = {}
        for cell, factor_gains in gain_by_feature.items():
            total_gain = sum(factor_gains.values(), start=Fraction(0, 1))
            for factor, gain in factor_gains.items():
                weights[(cell, factor)] = gain / total_gain
        self._feature_factor_weight = weights

    def _build_factor_refinements(
        self,
        slots: Sequence[_LearnedSlot],
    ) -> tuple[_ProjectedFactorRefinement, ...]:
        refinements: dict[
            tuple[int, int, int, int, int],
            _ProjectedFactorRefinement,
        ] = {}
        for source_slot in slots:
            source_factor = self._factor_by_slot[source_slot.slot_cell]
            source_root, source_support, source_blocks = source_factor
            source_factor_cell = self._factor_cell_by_slot[source_slot.slot_cell]
            for target_slot in slots:
                target_factor = self._factor_by_slot[target_slot.slot_cell]
                target_root, target_support, _ = target_factor
                if source_root != target_root or not set(source_support).issubset(
                    target_support
                ):
                    continue
                target_candidates = tuple(
                    (candidate, frozenset(candidate.support_rows))
                    for candidate in target_slot.candidates
                )
                if any(
                    sum(
                        frozenset(block).issubset(target_rows)
                        for _, target_rows in target_candidates
                    )
                    != 1
                    for block in source_blocks
                ):
                    continue
                for source_candidate in source_slot.candidates:
                    source_rows = frozenset(source_candidate.support_rows)
                    targets = tuple(
                        target_candidate
                        for target_candidate, target_rows in target_candidates
                        if source_rows.issubset(target_rows)
                    )
                    if len(targets) != 1:
                        continue
                    target_candidate = targets[0]
                    source_cohort_cell = self._cohort_cell_by_candidate[
                        (source_slot.slot_cell, source_candidate.owner_cell)
                    ]
                    target_factor_cell = self._factor_cell_by_slot[
                        target_slot.slot_cell
                    ]
                    target_cohort_cell = self._cohort_cell_by_candidate[
                        (target_slot.slot_cell, target_candidate.owner_cell)
                    ]
                    relation_key = (
                        source_factor_cell,
                        source_cohort_cell,
                        len(source_support),
                        target_factor_cell,
                        target_cohort_cell,
                    )
                    refinements[relation_key] = _ProjectedFactorRefinement(
                        relay_cell=self._cell_for(
                            ("factor-refinement", *relation_key)
                        ),
                        source_factor_cell=source_factor_cell,
                        source_cohort_cell=source_cohort_cell,
                        source_support_count=len(source_support),
                        target_factor_cell=target_factor_cell,
                        target_cohort_cell=target_cohort_cell,
                    )
        return tuple(refinements[key] for key in sorted(refinements))

    def _factor_key(self, slot: _LearnedSlot) -> _FactorKey:
        blocks = tuple(
            sorted(candidate.support_rows for candidate in slot.candidates)
        )
        if any(not block for block in blocks):
            raise RuntimeError("factor support blocks must be non-empty")
        cohort = tuple(sorted(row for block in blocks for row in block))
        if len(cohort) != len(set(cohort)):
            raise RuntimeError("factor support blocks must be disjoint")
        return self._factor_root_key, cohort, blocks

    def _collect_slots(
        self,
        slot: _LearnedSlot,
        result: list[_LearnedSlot],
    ) -> None:
        result.append(slot)
        for candidate in slot.candidates:
            for child in candidate.slots:
                if child is not None:
                    self._collect_slots(child, result)

    def _recruit_role_loop(self, spec: _RoleLoopSpec) -> _ProjectedRoleLoop:
        """Recruit immutable role, syntax, and separator cells before inference."""

        return _ProjectedRoleLoop(
            marker_cell=self._literal_cell(spec.marker_token),
            start_ordinal=spec.start_ordinal,
            separator_cells=tuple(
                self._literal_cell(token) for token in spec.separator_tokens
            ),
            role_cells=tuple(
                self._cell_for(("role-loop", "role", ordinal))
                for ordinal in range(spec.max_roles)
            ),
            prefix_cells=(self._literal_cell(ord("<")), self._literal_cell(ord("@"))),
            digit_cells=tuple(
                self._literal_cell(ord("0") + digit) for digit in range(10)
            ),
            suffix_cell=self._literal_cell(ord(">")),
        )

    def _literal_cell(self, token: int) -> int:
        return self._atom_cell("literal", token)

    def _atom_cell(self, kind: str, value: int) -> int:
        cell = self._cell_for(("fragment", kind, value))
        self._atom_by_cell[cell] = (kind, value)
        if len(self._atom_by_cell) > self.config.max_fragment_cells:
            raise RuntimeError("fragment-cell capacity is exhausted")
        return cell

    def _cell_for(self, key: tuple[object, ...]) -> int:
        existing = self._cell_by_key.get(key)
        if existing is not None:
            return existing
        cell = len(self._cell_by_key)
        self._cell_by_key[key] = cell
        self._key_by_cell[cell] = key
        return cell

    def _compute_owner_digest(self) -> str:
        programs = []
        for program in self._programs:
            programs.append(
                {
                    "direct": program.direct,
                    "fragment_cell_path": program.fragment_cell_path,
                    "owner_cell": program.owner_cell,
                    "root_edge": program.root_edge,
                    "slots": [
                        None
                        if slot is None
                        else {
                            "candidates": [asdict(candidate) for candidate in slot.candidates],
                            "slot_cell": slot.slot_cell,
                        }
                        for slot in program.slots
                    ],
                }
            )
        return _digest(
            b"humanbrain-factorized-fragment-owner-v1",
            {
                "atoms": tuple(sorted(self._atom_by_cell.items())),
                "boundary_token_ids": tuple(sorted(self._boundary_token_ids)),
                "cells": tuple(
                    (repr(key), cell)
                    for key, cell in sorted(self._cell_by_key.items(), key=str)
                ),
                "config": asdict(self.config),
                "cohort_cell_by_candidate": tuple(
                    sorted(self._cohort_cell_by_candidate.items())
                ),
                "factor_by_slot": tuple(sorted(self._factor_by_slot.items())),
                "factor_cell_by_slot": tuple(
                    sorted(self._factor_cell_by_slot.items())
                ),
                "factor_refinements": tuple(
                    asdict(item) for item in self._factor_refinements
                ),
                "role_loops": tuple(asdict(item) for item in self._role_loops),
                "factor_root_key": self._factor_root_key,
                "programs": programs,
                "schema": _SCHEMA,
                "stop_owner_cell": self._stop_owner_cell,
                "transport_token_ids": tuple(sorted(self._transport_token_ids)),
            },
        )
