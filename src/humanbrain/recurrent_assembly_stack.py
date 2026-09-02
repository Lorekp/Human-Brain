"""Bounded recurrent assembly programs learned from opaque successful sequences.

The module deliberately knows nothing about Python, code, task types, or expected
fresh answers.  It discovers cue-conditioned positions and equality-derived copy
roles from complete successful sequences of opaque afferent GIDs.  Properly nested
position intervals become reusable assembly programs.  Every inference decision is the winner of a
fitted-cell race; the small amount of host state only carries those winners
through a bounded neuronal frame array.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256

from humanbrain.cortex import DendriticCortex
from humanbrain.factorized_fragment_memory import (
    _FragmentProjection,
    _ProjectedRoleLoop,
    _ProjectedSlot,
    _ProjectedSlotCandidate,
)

__all__ = [
    "AssemblyAblation",
    "AssemblyConsequenceUpdate",
    "AssemblyRaceStep",
    "AssemblyStackConfig",
    "AssemblyTrace",
    "PendingAssemblyDecision",
    "RecurrentAssemblyStack",
    "Valence",
]


class AssemblyAblation(Enum):
    NONE = "none"
    STACK_OFF = "stack_off"


class Valence(Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


@dataclass(frozen=True, slots=True)
class AssemblyStackConfig:
    max_cells: int = 2_048
    max_examples: int = 64
    max_programs: int = 256
    max_cue_tokens: int = 32
    max_sequence_tokens: int = 128
    max_depth: int = 4
    max_steps: int = 256
    max_output_tokens: int = 128
    random_seed: int = 7

    def __post_init__(self) -> None:
        positive = (
            self.max_cells,
            self.max_examples,
            self.max_programs,
            self.max_cue_tokens,
            self.max_sequence_tokens,
            self.max_depth,
            self.max_steps,
            self.max_output_tokens,
        )
        if any(type(value) is not int or value < 1 for value in positive):
            raise ValueError("assembly bounds must be positive integers")
        if self.max_cells <= self.max_depth + 8:
            raise ValueError("max_cells is too small for controls and frame cells")
        if type(self.random_seed) is not int:
            raise ValueError("random_seed must be an integer")


@dataclass(frozen=True, slots=True)
class AssemblyRaceStep:
    phase: str
    operation: str
    depth: int
    candidate_cells: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    winner_cell: int
    spike_time_ms: float
    simulated_steps: int


@dataclass(frozen=True, slots=True)
class AssemblyTrace:
    steps: tuple[AssemblyRaceStep, ...]
    winner_cells: tuple[int, ...]
    operations: tuple[str, ...]
    simulated_steps: int
    max_depth: int
    selected_structural_edges: tuple[str, ...]
    selected_choice_edges: tuple[str, ...]
    selected_stop: bool


@dataclass(frozen=True, slots=True)
class AssemblyConsequenceUpdate:
    valence: Valence
    changed_edges: tuple[str, ...]
    prior_digest: str
    model_digest: str


@dataclass(frozen=True, slots=True)
class _Literal:
    token: int


@dataclass(frozen=True, slots=True)
class _CopyRole:
    cue_index: int


_Atom = _Literal | _CopyRole


@dataclass(frozen=True, slots=True)
class _Example:
    cue: tuple[int, ...]
    sequence: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Action:
    kind: str
    state_cell: int
    atom: _Atom | None = None
    atom_cell: int | None = None
    emission_slot: str | None = None
    child_group: str | None = None


@dataclass(frozen=True, slots=True)
class _Program:
    key: str
    group: str
    root_cell: int
    cue_arity: int
    controller_index: int | None
    cue_value: int | None
    exact_cue: tuple[int, ...] | None
    actions: tuple[_Action, ...]
    is_root: bool


@dataclass(frozen=True, slots=True)
class _Interval:
    controller: int
    start: int
    end: int
    parent: int | None
    children: tuple[int, ...]
    owned_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Contribution:
    atoms: tuple[_Atom, ...]


@dataclass(slots=True)
class _ShapeCoactivity:
    cue_arity: int
    output_length: int
    contributions: dict[tuple[int, ...], _Contribution] = field(default_factory=dict)
    direct_programs: dict[tuple[int, ...], _Program] = field(default_factory=dict)
    cue_value_support: Counter[tuple[int, int]] = field(default_factory=Counter)
    position_atom_support: dict[int, Counter[_Atom]] = field(
        default_factory=dict
    )
    conditional_support: dict[tuple[int, int, int], Counter[_Atom]] = field(
        default_factory=dict
    )
    active_programs: dict[str, _Program] = field(default_factory=dict)
    published_keys: set[str] = field(default_factory=set)
    compositional: bool = False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _opaque_tuple(value: Sequence[int], *, name: str) -> tuple[int, ...]:
    result = tuple(value)
    if not result or any(type(token) is not int or token < 0 for token in result):
        raise ValueError(f"{name} must contain nonnegative integer GIDs")
    return result


def _atom_payload(atom: _Atom) -> tuple[str, int]:
    if isinstance(atom, _CopyRole):
        return ("copy", atom.cue_index)
    return ("literal", atom.token)


class PendingAssemblyDecision:
    """One unresolved neuronal decision and its consequence eligibility."""

    __slots__ = ("_nonce", "_resolved", "_stack", "_tokens", "_trace")

    def __init__(
        self,
        stack: RecurrentAssemblyStack,
        nonce: int,
        tokens: tuple[int, ...],
        trace: AssemblyTrace,
    ) -> None:
        self._stack = stack
        self._nonce = nonce
        self._tokens = tokens
        self._trace = trace
        self._resolved = False

    @property
    def tokens(self) -> tuple[int, ...]:
        return self._tokens

    @property
    def trace(self) -> AssemblyTrace:
        return self._trace

    def resolve(self, valence: Valence) -> AssemblyConsequenceUpdate:
        if self._resolved:
            raise RuntimeError("assembly decision consequence was already resolved")
        if not isinstance(valence, Valence):
            raise TypeError("valence must be a Valence")
        update = self._stack._resolve(self, self._nonce, valence)
        self._resolved = True
        return update


class RecurrentAssemblyStack:
    """Learn and run bounded opaque sequence assemblies through cell races."""

    _PUSH = "push"
    _EMIT = "emit"
    _POP = "pop"
    _STOP = "stop"
    _ROLE_CONTINUE = "role-continue"
    _ROLE_EXIT = "role-exit"
    _CONTROL_ORDER = (_PUSH, _EMIT, _POP, _STOP)
    _FACTOR_RECURRENT_DRIVE = 5.0
    _NOGO_TICKS = 5
    _MODEL_SCHEMA = "humanbrain-recurrent-assembly-stack-v3"

    def __init__(self, config: AssemblyStackConfig | None = None) -> None:
        self.config = config or AssemblyStackConfig()
        self._cortex = DendriticCortex(
            cell_count=self.config.max_cells,
            max_order=1,
            seed=self.config.random_seed,
        )
        self._basket_cell = self.config.max_cells - 1
        self._next_cell = 0
        self._cell_by_key: dict[tuple[object, ...], int] = {}
        self._key_by_cell: dict[int, tuple[object, ...]] = {}
        self._control_cells = {
            operation: self._cell_for(("control", operation))
            for operation in self._CONTROL_ORDER
        }
        self._frame_cells = tuple(
            self._cell_for(("frame", depth))
            for depth in range(self.config.max_depth)
        )
        self._role_control_cells = {
            self._ROLE_CONTINUE: self._cell_for(("role-loop-control", self._ROLE_CONTINUE)),
            self._ROLE_EXIT: self._cell_for(("role-loop-control", self._ROLE_EXIT)),
        }
        self._shape_states: dict[tuple[int, int], _ShapeCoactivity] = {}
        self._shape_by_cue: dict[tuple[int, ...], tuple[int, int]] = {}
        self._contribution_count = 0
        self._incremental_update_count = 0
        self._last_coactivity_write_count = 0
        self._prior_contribution_scan_count = 0
        self._programs: dict[str, _Program] = {}
        self._variants_by_group: dict[str, tuple[str, ...]] = {}
        self._emission_cells_by_slot: dict[str, tuple[int, ...]] = {}
        self._atom_by_cell: dict[int, _Atom] = {}
        self._root_postings: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._exact_root_postings: dict[tuple[int, ...], set[str]] = defaultdict(set)
        self._plasticity_ticks: dict[str, int] = {}
        self._projection_owner_cells: dict[str, frozenset[int]] = {}
        self._pending: PendingAssemblyDecision | None = None
        self._nonce = 0

    @property
    def model_digest(self) -> str:
        if self._pending is not None:
            raise RuntimeError("model digest is unavailable while a decision is unresolved")
        return self._compute_digest()

    @property
    def incremental_update_count(self) -> int:
        """Number of distinct additions or exact-cue revisions applied."""

        return self._incremental_update_count

    @property
    def last_coactivity_write_count(self) -> int:
        """Counter writes made by the most recent successful observation."""

        return self._last_coactivity_write_count

    @property
    def prior_contribution_scan_count(self) -> int:
        """Historical contributions replayed into learning counters (always zero)."""

        return self._prior_contribution_scan_count

    def observe_success(
        self,
        cue: Sequence[int],
        sequence: Sequence[int],
    ) -> None:
        """Observe one complete successful opaque cue/sequence pair."""

        self._require_resolved()
        cue_tokens = _opaque_tuple(cue, name="cue")
        output_tokens = _opaque_tuple(sequence, name="sequence")
        if len(cue_tokens) > self.config.max_cue_tokens:
            raise ValueError("cue exceeds max_cue_tokens")
        if len(output_tokens) > self.config.max_sequence_tokens:
            raise ValueError("sequence exceeds max_sequence_tokens")
        old_shape_key = self._shape_by_cue.get(cue_tokens)
        if (
            old_shape_key is None
            and self._contribution_count >= self.config.max_examples
        ):
            raise RuntimeError("assembly example capacity is exhausted")

        atoms = self._generalize(_Example(cue_tokens, output_tokens))
        new_shape_key = (len(cue_tokens), len(output_tokens))
        if old_shape_key is not None:
            old_state = self._shape_states[old_shape_key]
            old_contribution = old_state.contributions[cue_tokens]
            if old_shape_key == new_shape_key and old_contribution.atoms == atoms:
                self._last_coactivity_write_count = 0
                return
            self._last_coactivity_write_count = 0
            self._remove_contribution(old_shape_key, old_state, cue_tokens)
            if old_shape_key != new_shape_key:
                self._refresh_shape(old_shape_key, old_state)
        else:
            self._last_coactivity_write_count = 0
            self._contribution_count += 1

        state = self._shape_states.get(new_shape_key)
        if state is None:
            state = _ShapeCoactivity(*new_shape_key)
            self._shape_states[new_shape_key] = state
        contribution = _Contribution(atoms=atoms)
        state.contributions[cue_tokens] = contribution
        if len(output_tokens) == 1:
            direct_program = self._compile_direct(
                _Example(cue_tokens, output_tokens)
            )
            state.direct_programs[cue_tokens] = direct_program
            state.active_programs[direct_program.key] = direct_program
        self._apply_contribution(state, cue_tokens, contribution, delta=1)
        self._shape_by_cue[cue_tokens] = new_shape_key
        self._incremental_update_count += 1
        self._refresh_shape(new_shape_key, state)

    def respond(
        self,
        cue: Sequence[int],
        ablation: AssemblyAblation = AssemblyAblation.NONE,
    ) -> PendingAssemblyDecision:
        """Return one consequence-bound decision made only by physical races."""

        self._require_resolved()
        if not isinstance(ablation, AssemblyAblation):
            raise TypeError("ablation must be an AssemblyAblation")
        cue_tokens = _opaque_tuple(cue, name="cue")
        if len(cue_tokens) > self.config.max_cue_tokens:
            raise ValueError("cue exceeds max_cue_tokens")

        steps: list[AssemblyRaceStep] = []
        selected_edges: list[str] = []
        emitted: list[int] = []
        selected_stop = False
        max_depth = 0

        root_keys: set[str] = set(self._exact_root_postings.get(cue_tokens, ()))
        for cue_index, cue_token in enumerate(cue_tokens):
            root_keys.update(self._root_postings.get((cue_index, cue_token), ()))
        root_programs = tuple(
            self._programs[key]
            for key in sorted(root_keys)
            if self._programs[key].is_root
        )
        root_cells = [program.root_cell for program in root_programs]
        root_scores = [self._root_score(program, cue_tokens) for program in root_programs]
        root_cells.append(self._control_cells[self._STOP])
        root_scores.append(1.0)
        root_winner = self._race(
            phase="root",
            operation_by_cell={
                **{program.root_cell: "root" for program in root_programs},
                self._control_cells[self._STOP]: self._STOP,
            },
            depth=0,
            cells=root_cells,
            scores=root_scores,
            steps=steps,
        )
        program_by_root = {program.root_cell: program for program in root_programs}
        current_program = program_by_root.get(root_winner)
        if current_program is None:
            selected_stop = True
        else:
            selected_edges.append(self._root_edge(current_program))

        frame_programs: list[_Program | None] = [None] * self.config.max_depth
        frame_indices = [0] * self.config.max_depth
        frame_return_cells = [-1] * self.config.max_depth
        depth = 0
        action_index = 0
        race_budget_exhausted = False

        while current_program is not None and not selected_stop:
            if len(steps) >= self.config.max_steps:
                selected_stop = True
                race_budget_exhausted = True
                break
            action = current_program.actions[action_index]
            control_scores = [
                5.0 if operation == action.kind else 0.0
                for operation in self._CONTROL_ORDER
            ]
            control_winner = self._race(
                phase="control",
                operation_by_cell={
                    self._control_cells[operation]: operation
                    for operation in self._CONTROL_ORDER
                },
                depth=depth,
                cells=[self._control_cells[operation] for operation in self._CONTROL_ORDER],
                scores=control_scores,
                steps=steps,
            )
            operation = self._operation_for_control(control_winner)
            if operation == self._STOP:
                selected_stop = action.kind != self._STOP
                break
            if operation != action.kind:
                selected_stop = True
                break

            if operation == self._EMIT:
                assert action.atom is not None
                assert action.atom_cell is not None
                assert action.emission_slot is not None
                population = self._emission_cells_by_slot[action.emission_slot]
                emission_cells = [*population, self._control_cells[self._STOP]]
                emission_scores = [
                    5.0 if cell == action.atom_cell else 0.0 for cell in population
                ]
                emission_scores.append(1.0)
                emission_winner = self._race(
                    phase="emission",
                    operation_by_cell={
                        **{cell: self._EMIT for cell in population},
                        self._control_cells[self._STOP]: self._STOP,
                    },
                    depth=depth,
                    cells=emission_cells,
                    scores=emission_scores,
                    steps=steps,
                )
                if emission_winner == self._control_cells[self._STOP]:
                    selected_stop = True
                    break
                emitted.append(self._materialize(self._atom_by_cell[emission_winner], cue_tokens))
                if len(emitted) > self.config.max_output_tokens:
                    selected_stop = True
                    break
                action_index += 1
                continue

            if operation == self._PUSH:
                if depth >= self.config.max_depth:
                    selected_stop = True
                    break
                frame_cells = [*self._frame_cells, self._control_cells[self._STOP]]
                frame_scores = [
                    (
                        5.0
                        if frame_index == depth
                        and ablation is AssemblyAblation.NONE
                        else 0.0
                    )
                    for frame_index in range(self.config.max_depth)
                ]
                frame_scores.append(1.0)
                frame_winner = self._race(
                    phase="frame",
                    operation_by_cell={
                        **{cell: self._PUSH for cell in self._frame_cells},
                        self._control_cells[self._STOP]: self._STOP,
                    },
                    depth=depth,
                    cells=frame_cells,
                    scores=frame_scores,
                    steps=steps,
                )
                if frame_winner == self._control_cells[self._STOP]:
                    selected_stop = True
                    break
                assert action.child_group is not None
                child_programs = tuple(
                    self._programs[key]
                    for key in self._variants_by_group[action.child_group]
                )
                child_cells = [program.root_cell for program in child_programs]
                child_scores = [
                    self._child_score(current_program, action_index, child, cue_tokens)
                    for child in child_programs
                ]
                child_cells.append(self._control_cells[self._STOP])
                child_scores.append(1.0)
                child_winner = self._race(
                    phase="child",
                    operation_by_cell={
                        **{program.root_cell: self._PUSH for program in child_programs},
                        self._control_cells[self._STOP]: self._STOP,
                    },
                    depth=depth,
                    cells=child_cells,
                    scores=child_scores,
                    steps=steps,
                )
                child_by_cell = {program.root_cell: program for program in child_programs}
                child = child_by_cell.get(child_winner)
                if child is None:
                    selected_stop = True
                    break
                edge = self._child_edge(current_program, action_index, child)
                selected_edges.append(edge)
                continuation_index = action_index + 1
                frame_programs[depth] = current_program
                frame_indices[depth] = continuation_index
                frame_return_cells[depth] = current_program.actions[
                    continuation_index
                ].state_cell
                depth += 1
                max_depth = max(max_depth, depth)
                current_program = child
                action_index = 0
                continue

            if operation == self._POP:
                if depth == 0:
                    selected_stop = True
                    break
                return_cell = frame_return_cells[depth - 1]
                continuation_winner = self._race(
                    phase="continuation",
                    operation_by_cell={
                        return_cell: self._POP,
                        self._control_cells[self._STOP]: self._STOP,
                    },
                    depth=depth,
                    cells=[return_cell, self._control_cells[self._STOP]],
                    scores=[
                        5.0 if ablation is AssemblyAblation.NONE else 0.0,
                        1.0,
                    ],
                    steps=steps,
                )
                if continuation_winner == self._control_cells[self._STOP]:
                    selected_stop = True
                    break
                depth -= 1
                parent = frame_programs[depth]
                if parent is None:
                    selected_stop = True
                    break
                current_program = parent
                action_index = frame_indices[depth]
                frame_programs[depth] = None
                frame_indices[depth] = 0
                frame_return_cells[depth] = -1

        if race_budget_exhausted:
            # The bounded stop is still represented by a physical STOP winner.
            self._race(
                phase="budget",
                operation_by_cell={self._control_cells[self._STOP]: self._STOP},
                depth=depth,
                cells=[self._control_cells[self._STOP]],
                scores=[1.0],
                steps=steps,
            )
        tokens = () if selected_stop else tuple(emitted)
        trace = AssemblyTrace(
            steps=tuple(steps),
            winner_cells=tuple(step.winner_cell for step in steps),
            operations=tuple(step.operation for step in steps),
            simulated_steps=sum(step.simulated_steps for step in steps),
            max_depth=max_depth,
            selected_structural_edges=tuple(selected_edges),
            selected_choice_edges=(),
            selected_stop=selected_stop,
        )
        self._nonce += 1
        pending = PendingAssemblyDecision(self, self._nonce, tokens, trace)
        self._pending = pending
        return pending

    def _register_projection_owner(
        self,
        owner_digest: str,
        owner_cells: Sequence[int],
    ) -> None:
        """Preallocate one private projected-cell sheet before inference."""

        self._require_resolved()
        if not isinstance(owner_digest, str) or not owner_digest:
            raise ValueError("projection owner digest must be non-empty")
        cells = frozenset(owner_cells)
        if not cells or any(type(cell) is not int or cell < 0 for cell in cells):
            raise ValueError("projection owner cells must be nonnegative integers")
        existing = self._projection_owner_cells.get(owner_digest)
        if existing is not None and existing != cells:
            raise ValueError("projection owner digest was rebound to different cells")
        for owner_cell in sorted(cells):
            self._cell_for(("projection", owner_digest, owner_cell))
        self._projection_owner_cells[owner_digest] = cells

    def _respond_projection(
        self,
        projection: _FragmentProjection,
        *,
        role_count: int,
        ablation: AssemblyAblation = AssemblyAblation.NONE,
    ) -> PendingAssemblyDecision:
        """Run one sealed fragment projection entirely through physical races."""

        self._require_resolved()
        if not isinstance(projection, _FragmentProjection):
            raise TypeError("projection must be a _FragmentProjection")
        if not isinstance(ablation, AssemblyAblation):
            raise TypeError("ablation must be an AssemblyAblation")
        if type(role_count) is not int or role_count < 1:
            raise ValueError("role_count must be a positive integer")
        if any(role_count > len(loop.role_cells) for loop in projection.role_loops):
            raise ValueError("role inventory exceeds learned loop capacity")
        projection.verify()
        registered = self._projection_owner_cells.get(projection.owner_digest)
        referenced = self._projection_referenced_cells(projection)
        if registered is None or not referenced.issubset(registered):
            raise ValueError("fragment projection is not bound to its owner sheet")

        steps: list[AssemblyRaceStep] = []
        selected_edges: list[str] = []
        selected_choice_edges: list[str] = []
        emitted: list[int] = []
        selected_stop = False
        max_depth = 0
        stop_cell = self._projection_cell(
            projection.owner_digest,
            projection.stop_owner_cell,
        )

        program_cells = [
            self._projection_cell(projection.owner_digest, program.owner_cell)
            for program in projection.programs
        ]
        root_cells = [*program_cells, stop_cell]
        root_choice_edges = [
            self._projected_choice_edge((), program.root_edge)
            for program in projection.programs
        ]
        root_scores = [
            max(
                0.0,
                program.signed_drive + self._plasticity_ticks.get(choice_edge, 0),
            )
            if program.signed_drive > 0.0
            else 0.0
            for program, choice_edge in zip(
                projection.programs,
                root_choice_edges,
                strict=True,
            )
        ]
        root_scores.append(max(0.0, projection.stop_signed_drive))
        root_winner = self._race(
            phase="projected-root",
            operation_by_cell={
                **{cell: "root" for cell in program_cells},
                stop_cell: self._STOP,
            },
            depth=0,
            cells=root_cells,
            scores=root_scores,
            steps=steps,
        )
        program_by_cell = dict(zip(program_cells, projection.programs, strict=True))
        program = program_by_cell.get(root_winner)
        if program is None:
            selected_stop = True
        else:
            root_index = projection.programs.index(program)
            if len(projection.programs) > 1:
                selected_choice_edges.append(root_choice_edges[root_index])
            deepest = [0]
            factor_state: dict[int, int] = {}
            planned = self._projected_emit_pattern(
                projection,
                program.fragment_cell_path,
                program.slots,
                owner_end=program.owner_cell,
                ablation=ablation,
                role_count=role_count,
                depth=0,
                steps=steps,
                emitted=[],
                selected_edges=[],
                selected_choice_edges=[],
                choice_prefix=[program.root_edge],
                deepest=deepest,
                factor_state=factor_state,
            )
            selected_stop = not planned or not self._projected_emit_pattern(
                projection,
                program.fragment_cell_path,
                program.slots,
                owner_end=program.owner_cell,
                ablation=ablation,
                role_count=role_count,
                depth=0,
                steps=steps,
                emitted=emitted,
                selected_edges=selected_edges,
                selected_choice_edges=selected_choice_edges,
                choice_prefix=[program.root_edge],
                deepest=deepest,
                factor_state=factor_state,
            )
            max_depth = deepest[0]

        if not selected_stop and not self._projected_control(
            self._STOP,
            depth=0,
            steps=steps,
        ):
            selected_stop = True
        if len(steps) > self.config.max_steps or len(emitted) > self.config.max_output_tokens:
            selected_stop = True
            self._race(
                phase="budget",
                operation_by_cell={stop_cell: self._STOP},
                depth=0,
                cells=[stop_cell],
                scores=[1.0],
                steps=steps,
            )
        tokens = () if selected_stop else tuple(emitted)
        trace = AssemblyTrace(
            steps=tuple(steps),
            winner_cells=tuple(step.winner_cell for step in steps),
            operations=tuple(step.operation for step in steps),
            simulated_steps=sum(step.simulated_steps for step in steps),
            max_depth=max_depth,
            selected_structural_edges=tuple(selected_edges),
            selected_choice_edges=tuple(selected_choice_edges),
            selected_stop=selected_stop,
        )
        self._nonce += 1
        pending = PendingAssemblyDecision(self, self._nonce, tokens, trace)
        self._pending = pending
        return pending

    def _projected_emit_pattern(
        self,
        projection: _FragmentProjection,
        fragment_path: Sequence[int],
        slots: Sequence[_ProjectedSlot | None],
        *,
        owner_end: int,
        ablation: AssemblyAblation,
        role_count: int,
        depth: int,
        steps: list[AssemblyRaceStep],
        emitted: list[int],
        selected_edges: list[str],
        selected_choice_edges: list[str],
        choice_prefix: list[str],
        deepest: list[int],
        factor_state: dict[int, int],
    ) -> bool:
        normalized_slots = (
            tuple(slots)
            if slots
            else tuple(None for _ in range(len(fragment_path) + 1))
        )
        if len(normalized_slots) != len(fragment_path) + 1:
            raise ValueError("projected slots do not match their fragment path")
        stop_cell = self._projection_cell(
            projection.owner_digest,
            projection.stop_owner_cell,
        )
        for slot_index, slot in enumerate(normalized_slots):
            if slot is not None:
                chosen = self._projected_slot_race(
                    projection,
                    slot,
                    stop_cell=stop_cell,
                    ablation=ablation,
                    depth=depth,
                    steps=steps,
                    factor_state=factor_state,
                    choice_prefix=tuple(choice_prefix),
                )
                if chosen is None:
                    return False
                factor_state[slot.factor_cell] = chosen.cohort_cell
                selected_edges.append(chosen.span_edge)
                if len(slot.candidates) > 1:
                    selected_choice_edges.append(
                        self._projected_choice_edge(
                            tuple(choice_prefix),
                            chosen.span_edge,
                        )
                    )
                choice_prefix.append(chosen.span_edge)
                if not self._projected_control(
                    self._PUSH,
                    depth=depth,
                    steps=steps,
                ):
                    return False
                frame_cells = [*self._frame_cells, stop_cell]
                frame_scores = [
                    (
                        5.0
                        if frame_index == depth
                        and ablation is AssemblyAblation.NONE
                        else 0.0
                    )
                    for frame_index in range(self.config.max_depth)
                ]
                frame_scores.append(1.0)
                frame_winner = self._race(
                    phase="frame",
                    operation_by_cell={
                        **{cell: self._PUSH for cell in self._frame_cells},
                        stop_cell: self._STOP,
                    },
                    depth=depth,
                    cells=frame_cells,
                    scores=frame_scores,
                    steps=steps,
                )
                if frame_winner == stop_cell:
                    return False
                child_depth = depth + 1
                deepest[0] = max(deepest[0], child_depth)
                if not self._projected_emit_pattern(
                    projection,
                    chosen.fragment_cell_path,
                    chosen.slots,
                    owner_end=chosen.owner_cell,
                    ablation=ablation,
                    role_count=role_count,
                    depth=child_depth,
                    steps=steps,
                    emitted=emitted,
                    selected_edges=selected_edges,
                    selected_choice_edges=selected_choice_edges,
                    choice_prefix=choice_prefix,
                    deepest=deepest,
                    factor_state=factor_state,
                ):
                    return False
                if not self._projected_control(
                    self._POP,
                    depth=child_depth,
                    steps=steps,
                ):
                    return False
                continuation_cell = self._projection_cell(
                    projection.owner_digest,
                    chosen.canonical_continuation,
                )
                continuation_winner = self._race(
                    phase="continuation",
                    operation_by_cell={
                        continuation_cell: self._POP,
                        stop_cell: self._STOP,
                    },
                    depth=child_depth,
                    cells=[continuation_cell, stop_cell],
                    scores=[
                        5.0 if ablation is AssemblyAblation.NONE else 0.0,
                        1.0,
                    ],
                    steps=steps,
                )
                if continuation_winner == stop_cell:
                    return False
                expected = (
                    fragment_path[slot_index]
                    if slot_index < len(fragment_path)
                    else owner_end
                )
                if chosen.canonical_continuation != expected:
                    raise ValueError("projected candidate continuation is not canonical")
            if slot_index < len(fragment_path):
                canonical_cell = fragment_path[slot_index]
                if not self._projected_emit_path(
                    projection,
                    (canonical_cell,),
                    ((canonical_cell,),),
                    role_count=role_count,
                    ablation=ablation,
                    depth=depth,
                    steps=steps,
                    emitted=emitted,
                ):
                    return False
        return True

    def _projected_slot_race(
        self,
        projection: _FragmentProjection,
        slot: _ProjectedSlot,
        *,
        stop_cell: int,
        ablation: AssemblyAblation,
        depth: int,
        steps: list[AssemblyRaceStep],
        factor_state: dict[int, int],
        choice_prefix: tuple[str, ...],
    ) -> _ProjectedSlotCandidate | None:
        candidate_cells = [
            self._projection_cell(projection.owner_digest, candidate.owner_cell)
            for candidate in slot.candidates
        ]
        choice_edges = [
            self._projected_choice_edge(choice_prefix, candidate.span_edge)
            for candidate in slot.candidates
        ]
        target_refinements = tuple(
            refinement
            for refinement in projection.refinements
            if refinement.target_factor_cell == slot.factor_cell
        )
        active_refinements = tuple(
            refinement
            for refinement in target_refinements
            if factor_state.get(refinement.source_factor_cell)
            == refinement.source_cohort_cell
        )
        if not active_refinements:
            scores = [
                (
                    min(
                        self._FACTOR_RECURRENT_DRIVE,
                        max(
                            0.0,
                            candidate.signed_drive
                            + self._plasticity_ticks.get(choice_edge, 0),
                        ),
                    )
                    if candidate.signed_drive > 0.0
                    else 0.0
                )
                for candidate, choice_edge in zip(
                    slot.candidates,
                    choice_edges,
                    strict=True,
                )
            ]
        else:
            most_specific = min(
                refinement.source_support_count
                for refinement in active_refinements
            )
            leading_targets = {
                refinement.target_cohort_cell
                for refinement in active_refinements
                if refinement.source_support_count == most_specific
            }
            unambiguous_target = (
                next(iter(leading_targets))
                if len(leading_targets) == 1
                else None
            )
            relay_cells = [
                self._projection_cell(
                    projection.owner_digest,
                    refinement.relay_cell,
                )
                for refinement in target_refinements
            ]
            relay_scores = [
                (
                    self._FACTOR_RECURRENT_DRIVE
                    * most_specific
                    / refinement.source_support_count
                    if refinement in active_refinements
                    and refinement.target_cohort_cell == unambiguous_target
                    and ablation is AssemblyAblation.NONE
                    else 0.0
                )
                for refinement in target_refinements
            ]
            relay_winner = self._race(
                phase="factor-relay",
                operation_by_cell={
                    **{cell: self._PUSH for cell in relay_cells},
                    stop_cell: self._STOP,
                },
                depth=depth,
                cells=[*relay_cells, stop_cell],
                scores=[*relay_scores, 1.0],
                steps=steps,
            )
            if relay_winner == stop_cell:
                return None
            refinement_by_cell = dict(
                zip(relay_cells, target_refinements, strict=True)
            )
            recalled_cohort = refinement_by_cell[relay_winner].target_cohort_cell
            cohort_owner_cells = tuple(
                dict.fromkeys(candidate.cohort_cell for candidate in slot.candidates)
            )
            cohort_cells = [
                self._projection_cell(projection.owner_digest, cohort_cell)
                for cohort_cell in cohort_owner_cells
            ]
            recalled_cell = self._race(
                phase="factor-recall",
                operation_by_cell={
                    **{cell: self._PUSH for cell in cohort_cells},
                    stop_cell: self._STOP,
                },
                depth=depth,
                cells=[*cohort_cells, stop_cell],
                scores=[
                    *(
                        self._FACTOR_RECURRENT_DRIVE
                        if cohort_cell == recalled_cohort
                        else 0.0
                        for cohort_cell in cohort_owner_cells
                    ),
                    1.0,
                ],
                steps=steps,
            )
            if recalled_cell == stop_cell:
                return None
            recalled_cohort = cohort_owner_cells[cohort_cells.index(recalled_cell)]
            revising = any(
                self._plasticity_ticks.get(choice_edge, 0) < 0
                for choice_edge in choice_edges
            )
            scores = [
                max(
                    0.0,
                    (
                        self._FACTOR_RECURRENT_DRIVE
                        if candidate.cohort_cell == recalled_cohort
                        else candidate.signed_drive if revising else 0.0
                    )
                    + self._plasticity_ticks.get(choice_edge, 0),
                )
                for candidate, choice_edge in zip(
                    slot.candidates,
                    choice_edges,
                    strict=True,
                )
            ]
        winner = self._race(
            phase="projected-slot",
            operation_by_cell={
                **{cell: self._PUSH for cell in candidate_cells},
                stop_cell: self._STOP,
            },
            depth=depth,
            cells=[*candidate_cells, stop_cell],
            scores=[*scores, 1.0],
            steps=steps,
        )
        by_cell = dict(zip(candidate_cells, slot.candidates, strict=True))
        return by_cell.get(winner)

    def _projected_control(
        self,
        expected: str,
        *,
        depth: int,
        steps: list[AssemblyRaceStep],
    ) -> bool:
        winner = self._race(
            phase="control",
            operation_by_cell={
                self._control_cells[operation]: operation
                for operation in self._CONTROL_ORDER
            },
            depth=depth,
            cells=[self._control_cells[operation] for operation in self._CONTROL_ORDER],
            scores=[
                5.0 if operation == expected else 0.0
                for operation in self._CONTROL_ORDER
            ],
            steps=steps,
        )
        return self._operation_for_control(winner) == expected

    def _projected_emit_path(
        self,
        projection: _FragmentProjection,
        selected_path: Sequence[int],
        peer_paths: Sequence[Sequence[int]],
        *,
        role_count: int,
        ablation: AssemblyAblation,
        depth: int,
        steps: list[AssemblyRaceStep],
        emitted: list[int],
    ) -> bool:
        stop_cell = self._projection_cell(
            projection.owner_digest,
            projection.stop_owner_cell,
        )
        role_loop_by_cell = {
            loop.marker_cell: loop for loop in projection.role_loops
        }
        for path_index, fragment_cell in enumerate(selected_path):
            role_loop = role_loop_by_cell.get(fragment_cell)
            if role_loop is not None:
                if not self._projected_emit_role_loop(
                    projection,
                    role_loop,
                    role_count=role_count,
                    ablation=ablation,
                    depth=depth,
                    steps=steps,
                    emitted=emitted,
                ):
                    return False
                continue
            if not self._projected_control(
                self._EMIT,
                depth=depth,
                steps=steps,
            ):
                return False
            population = tuple(
                sorted(
                    {
                        path[path_index]
                        for path in peer_paths
                        if path_index < len(path)
                    }
                )
            )
            cells = [
                self._projection_cell(projection.owner_digest, cell)
                for cell in population
            ]
            selected = self._projection_cell(
                projection.owner_digest,
                fragment_cell,
            )
            winner = self._race(
                phase="emission",
                operation_by_cell={
                    **{cell: self._EMIT for cell in cells},
                    stop_cell: self._STOP,
                },
                depth=depth,
                cells=[*cells, stop_cell],
                scores=[*(5.0 if cell == selected else 0.0 for cell in cells), 1.0],
                steps=steps,
            )
            if winner == stop_cell:
                return False
            emitted.append(population[cells.index(winner)])
        return True

    def _projected_emit_role_loop(
        self,
        projection: _FragmentProjection,
        loop: _ProjectedRoleLoop,
        *,
        role_count: int,
        ablation: AssemblyAblation,
        depth: int,
        steps: list[AssemblyRaceStep],
        emitted: list[int],
    ) -> bool:
        """Bind successive live role cells until physical exhaustion wins."""

        stop_cell = self._projection_cell(
            projection.owner_digest,
            projection.stop_owner_cell,
        )
        ordinal = loop.start_ordinal
        while ordinal < role_count:
            role_cells = [
                self._projection_cell(projection.owner_digest, cell)
                for cell in loop.role_cells[:role_count]
            ]
            selected_role = self._projection_cell(
                projection.owner_digest,
                loop.role_cells[ordinal],
            )
            role_winner = self._race(
                phase="role-binding",
                operation_by_cell={
                    **{cell: "role-bind" for cell in role_cells},
                    stop_cell: self._STOP,
                },
                depth=depth,
                cells=[*role_cells, stop_cell],
                scores=[
                    *(5.0 if cell == selected_role else 0.0 for cell in role_cells),
                    1.0,
                ],
                steps=steps,
            )
            if role_winner != selected_role:
                return False
            pointer_cells = (
                *loop.prefix_cells,
                *(
                    loop.digit_cells[int(character)]
                    for character in str(ordinal)
                ),
                loop.suffix_cell,
            )
            if not self._projected_emit_literal_cells(
                projection,
                pointer_cells,
                depth=depth,
                steps=steps,
                emitted=emitted,
            ):
                return False

            next_ordinal = ordinal + 1
            has_next = next_ordinal < role_count
            continue_cell = self._role_control_cells[self._ROLE_CONTINUE]
            exit_cell = self._role_control_cells[self._ROLE_EXIT]
            control_winner = self._race(
                phase="role-boundary",
                operation_by_cell={
                    continue_cell: self._ROLE_CONTINUE,
                    exit_cell: self._ROLE_EXIT,
                    stop_cell: self._STOP,
                },
                depth=depth,
                cells=[continue_cell, exit_cell, stop_cell],
                scores=[
                    5.0
                    if has_next and ablation is AssemblyAblation.NONE
                    else 0.0,
                    5.0 if not has_next else 0.0,
                    1.0,
                ],
                steps=steps,
            )
            if has_next:
                if control_winner != continue_cell:
                    return False
                if not self._projected_emit_literal_cells(
                    projection,
                    loop.separator_cells,
                    depth=depth,
                    steps=steps,
                    emitted=emitted,
                ):
                    return False
                ordinal = next_ordinal
                continue
            return control_winner == exit_cell
        self._race(
            phase="role-binding",
            operation_by_cell={stop_cell: self._STOP},
            depth=depth,
            cells=[stop_cell],
            scores=[1.0],
            steps=steps,
        )
        return False

    def _projected_emit_literal_cells(
        self,
        projection: _FragmentProjection,
        fragment_cells: Sequence[int],
        *,
        depth: int,
        steps: list[AssemblyRaceStep],
        emitted: list[int],
    ) -> bool:
        stop_cell = self._projection_cell(
            projection.owner_digest,
            projection.stop_owner_cell,
        )
        for fragment_cell in fragment_cells:
            if not self._projected_control(self._EMIT, depth=depth, steps=steps):
                return False
            selected = self._projection_cell(
                projection.owner_digest,
                fragment_cell,
            )
            winner = self._race(
                phase="role-emission",
                operation_by_cell={selected: self._EMIT, stop_cell: self._STOP},
                depth=depth,
                cells=[selected, stop_cell],
                scores=[5.0, 1.0],
                steps=steps,
            )
            if winner == stop_cell:
                return False
            emitted.append(fragment_cell)
        return True

    @staticmethod
    def _projection_referenced_cells(
        projection: _FragmentProjection,
    ) -> frozenset[int]:
        cells = {projection.stop_owner_cell}
        for program in projection.programs:
            cells.add(program.owner_cell)
            cells.update(program.fragment_cell_path)
            for slot in program.slots:
                if slot is None:
                    continue
                RecurrentAssemblyStack._collect_projected_slot_cells(slot, cells)
        for refinement in projection.refinements:
            cells.update(
                (
                    refinement.relay_cell,
                    refinement.source_factor_cell,
                    refinement.source_cohort_cell,
                    refinement.target_factor_cell,
                    refinement.target_cohort_cell,
                )
            )
        for loop in projection.role_loops:
            cells.add(loop.marker_cell)
            cells.update(loop.separator_cells)
            cells.update(loop.role_cells)
            cells.update(loop.prefix_cells)
            cells.update(loop.digit_cells)
            cells.add(loop.suffix_cell)
        return frozenset(cells)

    @staticmethod
    def _collect_projected_slot_cells(
        slot: _ProjectedSlot,
        cells: set[int],
    ) -> None:
        cells.add(slot.slot_cell)
        cells.add(slot.factor_cell)
        for candidate in slot.candidates:
            cells.add(candidate.owner_cell)
            cells.add(candidate.cohort_cell)
            cells.add(candidate.canonical_continuation)
            cells.update(candidate.fragment_cell_path)
            for child in candidate.slots:
                if child is not None:
                    RecurrentAssemblyStack._collect_projected_slot_cells(
                        child,
                        cells,
                    )

    def _projection_cell(self, owner_digest: str, owner_cell: int) -> int:
        cell = self._cell_by_key.get(("projection", owner_digest, owner_cell))
        if cell is None:
            raise ValueError("projected cell was not registered before inference")
        return cell

    @staticmethod
    def _projected_choice_edge(prefix: Sequence[str], edge: str) -> str:
        payload = _canonical_json((tuple(prefix), edge))
        return f"projected-choice:{sha256(payload.encode('ascii')).hexdigest()[:24]}"

    def _resolve(
        self,
        pending: PendingAssemblyDecision,
        nonce: int,
        valence: Valence,
    ) -> AssemblyConsequenceUpdate:
        if self._pending is not pending or nonce != self._nonce:
            raise RuntimeError("assembly decision is not the active pending decision")
        prior = self._compute_digest()
        structural = pending.trace.selected_structural_edges
        choices = pending.trace.selected_choice_edges
        selected = (*structural, *choices)
        if valence is Valence.POSITIVE:
            changed = tuple(dict.fromkeys(structural))
            for edge in changed:
                self._plasticity_ticks[edge] = self._plasticity_ticks.get(edge, 0) + 1
        elif valence is Valence.NEGATIVE and selected:
            # The latest real choice is contextualized by its selected prefix.
            projected_choice = choices[-1] if choices else None
            edge = projected_choice or selected[-1]
            changed = (edge,)
            decrement = self._NOGO_TICKS if projected_choice is not None else 1
            self._plasticity_ticks[edge] = (
                self._plasticity_ticks.get(edge, 0) - decrement
            )
        else:
            changed = ()
        self._pending = None
        return AssemblyConsequenceUpdate(
            valence=valence,
            changed_edges=changed,
            prior_digest=prior,
            model_digest=self._compute_digest(),
        )

    def _require_resolved(self) -> None:
        if self._pending is not None:
            raise RuntimeError("resolve the pending assembly decision first")

    def _cell_for(self, key: tuple[object, ...]) -> int:
        existing = self._cell_by_key.get(key)
        if existing is not None:
            return existing
        if self._next_cell >= self._basket_cell:
            raise RuntimeError("assembly cell capacity is exhausted")
        cell = self._next_cell
        self._next_cell += 1
        self._cell_by_key[key] = cell
        self._key_by_cell[cell] = key
        return cell

    def _remove_contribution(
        self,
        shape_key: tuple[int, int],
        state: _ShapeCoactivity,
        cue: tuple[int, ...],
    ) -> None:
        contribution = state.contributions.pop(cue)
        direct_program = state.direct_programs.pop(cue, None)
        if direct_program is not None:
            state.active_programs.pop(direct_program.key, None)
        self._apply_contribution(state, cue, contribution, delta=-1)
        self._shape_by_cue.pop(cue, None)
        if self._shape_states.get(shape_key) is not state:
            raise RuntimeError("shape contribution ownership is inconsistent")

    def _apply_contribution(
        self,
        state: _ShapeCoactivity,
        cue: tuple[int, ...],
        contribution: _Contribution,
        *,
        delta: int,
    ) -> None:
        for cue_index, cue_value in enumerate(cue):
            self._counter_delta(
                state.cue_value_support,
                (cue_index, cue_value),
                delta,
            )
        for position, atom in enumerate(contribution.atoms):
            position_counter = state.position_atom_support.setdefault(
                position,
                Counter(),
            )
            self._counter_delta(position_counter, atom, delta)
            if not position_counter:
                del state.position_atom_support[position]
            for cue_index, cue_value in enumerate(cue):
                conditional_key = (cue_index, cue_value, position)
                conditional_counter = state.conditional_support.setdefault(
                    conditional_key,
                    Counter(),
                )
                self._counter_delta(conditional_counter, atom, delta)
                if not conditional_counter:
                    del state.conditional_support[conditional_key]

    def _counter_delta(
        self,
        counter: Counter[object],
        key: object,
        delta: int,
    ) -> None:
        updated = counter.get(key, 0) + delta
        if updated < 0:
            raise RuntimeError("coactivity support cannot become negative")
        if updated:
            counter[key] = updated
        else:
            counter.pop(key, None)
        self._last_coactivity_write_count += 1

    def _refresh_shape(
        self,
        shape_key: tuple[int, int],
        state: _ShapeCoactivity,
    ) -> None:
        compiled = self._compile_compositional(state)
        if compiled is not None:
            state.active_programs = {program.key: program for program in compiled}
            state.compositional = True
        else:
            # Multi-token evidence remains STOP until it supports one coherent
            # nested structure.  Only atomic copy/literal routes are immediate.
            state.active_programs = {
                program.key: program for program in state.direct_programs.values()
            }
            state.compositional = False
        self._publish_shape(state)
        if not state.contributions:
            self._shape_states.pop(shape_key, None)

    def _publish_shape(self, state: _ShapeCoactivity) -> None:
        next_count = (
            len(self._programs)
            - len(state.published_keys)
            + len(state.active_programs)
        )
        if next_count > self.config.max_programs:
            raise RuntimeError("assembly program capacity is exhausted")
        for key in state.published_keys:
            self._programs.pop(key, None)
        self._programs.update(state.active_programs)
        state.published_keys = set(state.active_programs)
        self._index_programs()

    def _compile_compositional(
        self,
        state: _ShapeCoactivity,
    ) -> tuple[_Program, ...] | None:
        cue_arity = state.cue_arity
        output_length = state.output_length
        if len(state.contributions) < 3 or output_length < 2:
            return None

        mature_values: dict[int, tuple[int, ...]] = {}
        for cue_index in range(cue_arity):
            values = tuple(
                sorted(
                    cue_value
                    for (axis, cue_value), support in state.cue_value_support.items()
                    if axis == cue_index and support >= 2
                )
            )
            if len(values) >= 2:
                mature_values[cue_index] = values

        dependencies: list[int | None] = []
        for position in range(output_length):
            atoms = tuple(state.position_atom_support.get(position, ()))
            if len(atoms) == 1:
                dependencies.append(None)
                continue
            candidates: list[int] = []
            for cue_index, cue_values in mature_values.items():
                mapping: list[_Atom] = []
                for cue_value in cue_values:
                    supported_atoms = tuple(
                        state.conditional_support.get(
                            (cue_index, cue_value, position),
                            (),
                        )
                    )
                    if len(supported_atoms) != 1:
                        valid = False
                        break
                    mapping.append(supported_atoms[0])
                else:
                    valid = True
                if valid and len(set(mapping)) > 1:
                    candidates.append(cue_index)
            if len(candidates) != 1:
                return None
            dependencies.append(candidates[0])

        positions_by_controller: dict[int, list[int]] = defaultdict(list)
        for position, controller in enumerate(dependencies):
            if controller is not None:
                positions_by_controller[controller].append(position)
        if not positions_by_controller:
            return None
        spans = {
            controller: (min(positions), max(positions))
            for controller, positions in positions_by_controller.items()
        }
        controllers = sorted(spans)
        for left_index, left in enumerate(controllers):
            left_start, left_end = spans[left]
            for right in controllers[left_index + 1 :]:
                right_start, right_end = spans[right]
                crossing = (
                    left_start < right_start <= left_end < right_end
                    or right_start < left_start <= right_end < left_end
                )
                if crossing or (left_start, left_end) == (right_start, right_end):
                    return None

        parents: dict[int, int | None] = {}
        for controller in controllers:
            start, end = spans[controller]
            containers = [
                other
                for other in controllers
                if other != controller
                and spans[other][0] <= start
                and end <= spans[other][1]
            ]
            parents[controller] = (
                min(containers, key=lambda item: spans[item][1] - spans[item][0])
                if containers
                else None
            )
        roots = [controller for controller in controllers if parents[controller] is None]
        if len(roots) != 1 or spans[roots[0]] != (0, output_length - 1):
            return None

        assigned_invariants: dict[int, list[int]] = defaultdict(list)
        for position, dependency in enumerate(dependencies):
            if dependency is not None:
                continue
            containers = [
                controller
                for controller in controllers
                if spans[controller][0] <= position <= spans[controller][1]
            ]
            if not containers:
                return None
            owner = min(containers, key=lambda item: spans[item][1] - spans[item][0])
            assigned_invariants[owner].append(position)

        intervals: dict[int, _Interval] = {}
        for controller in controllers:
            children = tuple(
                sorted(
                    (child for child in controllers if parents[child] == controller),
                    key=lambda child: spans[child][0],
                )
            )
            owned = tuple(
                sorted(
                    [*positions_by_controller[controller], *assigned_invariants[controller]]
                )
            )
            intervals[controller] = _Interval(
                controller=controller,
                start=spans[controller][0],
                end=spans[controller][1],
                parent=parents[controller],
                children=children,
                owned_positions=owned,
            )

        def interval_depth(controller: int) -> int:
            depth = 0
            parent = intervals[controller].parent
            while parent is not None:
                depth += 1
                parent = intervals[parent].parent
            return depth

        if max(interval_depth(controller) for controller in controllers) > self.config.max_depth:
            return None

        cohort_name = f"shape:{cue_arity}:{output_length}"
        result: list[_Program] = []
        for controller in controllers:
            interval = intervals[controller]
            group = f"{cohort_name}:interval:{interval.start}:{interval.end}:{controller}"
            values = mature_values[controller]
            child_by_start = {
                intervals[child].start: child for child in interval.children
            }
            owned = set(interval.owned_positions)
            for cue_value in values:
                key = self._stable_key("program", group, cue_value)
                root_cell = self._cell_for(("root", key))
                actions: list[_Action] = []
                cursor = interval.start
                while cursor <= interval.end:
                    child_controller = child_by_start.get(cursor)
                    if child_controller is not None:
                        child = intervals[child_controller]
                        child_group = (
                            f"{cohort_name}:interval:{child.start}:{child.end}:"
                            f"{child.controller}"
                        )
                        actions.append(
                            _Action(
                                kind=self._PUSH,
                                state_cell=self._cell_for(
                                    ("state", key, len(actions))
                                ),
                                child_group=child_group,
                            )
                        )
                        cursor = child.end + 1
                        continue
                    if cursor not in owned:
                        return None
                    supported_atoms = tuple(
                        state.conditional_support.get(
                            (controller, cue_value, cursor),
                            (),
                        )
                    )
                    if len(supported_atoms) != 1:
                        return None
                    atom = supported_atoms[0]
                    slot = f"{group}:position:{cursor}"
                    atom_cell = self._atom_cell(atom)
                    actions.append(
                        _Action(
                            kind=self._EMIT,
                            state_cell=self._cell_for(("state", key, len(actions))),
                            atom=atom,
                            atom_cell=atom_cell,
                            emission_slot=slot,
                        )
                    )
                    cursor += 1
                actions.append(
                    _Action(
                        kind=self._STOP if interval.parent is None else self._POP,
                        state_cell=self._cell_for(("state", key, len(actions))),
                    )
                )
                result.append(
                    _Program(
                        key=key,
                        group=group,
                        root_cell=root_cell,
                        cue_arity=cue_arity,
                        controller_index=controller,
                        cue_value=cue_value,
                        exact_cue=None,
                        actions=tuple(actions),
                        is_root=interval.parent is None,
                    )
                )
        return tuple(result)

    def _compile_direct(self, example: _Example) -> _Program:
        key = self._stable_key("direct", example.cue, example.sequence)
        group = f"direct:{key}"
        actions: list[_Action] = []
        atoms = self._generalize(example)
        for position, atom in enumerate(atoms):
            slot = f"{group}:position:{position}"
            actions.append(
                _Action(
                    kind=self._EMIT,
                    state_cell=self._cell_for(("state", key, len(actions))),
                    atom=atom,
                    atom_cell=self._atom_cell(atom),
                    emission_slot=slot,
                )
            )
        actions.append(
            _Action(
                kind=self._STOP,
                state_cell=self._cell_for(("state", key, len(actions))),
            )
        )
        copied_indices = {
            atom.cue_index for atom in atoms if isinstance(atom, _CopyRole)
        }
        selector_indices = [
            index for index in range(len(example.cue)) if index not in copied_indices
        ]
        controller_index = selector_indices[0] if len(selector_indices) == 1 else None
        return _Program(
            key=key,
            group=group,
            root_cell=self._cell_for(("root", key)),
            cue_arity=len(example.cue),
            controller_index=controller_index,
            cue_value=(
                example.cue[controller_index] if controller_index is not None else None
            ),
            exact_cue=example.cue if controller_index is None else None,
            actions=tuple(actions),
            is_root=True,
        )

    def _index_programs(self) -> None:
        variants: dict[str, list[str]] = defaultdict(list)
        emissions: dict[str, set[int]] = defaultdict(set)
        roots: dict[tuple[int, int], set[str]] = defaultdict(set)
        exact_roots: dict[tuple[int, ...], set[str]] = defaultdict(set)
        for program in self._programs.values():
            variants[program.group].append(program.key)
            for action in program.actions:
                if action.emission_slot is not None and action.atom_cell is not None:
                    emissions[action.emission_slot].add(action.atom_cell)
            if not program.is_root:
                continue
            if program.exact_cue is not None:
                exact_roots[program.exact_cue].add(program.key)
            elif program.controller_index is not None and program.cue_value is not None:
                roots[(program.controller_index, program.cue_value)].add(program.key)
        self._variants_by_group = {
            group: tuple(sorted(keys)) for group, keys in variants.items()
        }
        self._emission_cells_by_slot = {
            slot: tuple(sorted(cells)) for slot, cells in emissions.items()
        }
        self._root_postings = roots
        self._exact_root_postings = exact_roots

    @staticmethod
    def _generalize(example: _Example) -> tuple[_Atom, ...]:
        atoms: list[_Atom] = []
        for token in example.sequence:
            matches = [
                index for index, cue_token in enumerate(example.cue) if cue_token == token
            ]
            atoms.append(_CopyRole(matches[0]) if len(matches) == 1 else _Literal(token))
        return tuple(atoms)

    def _atom_cell(self, atom: _Atom) -> int:
        payload = _atom_payload(atom)
        cell = self._cell_for(("atom", *payload))
        self._atom_by_cell[cell] = atom
        return cell

    @staticmethod
    def _stable_key(kind: str, *parts: object) -> str:
        payload = _canonical_json((kind, *parts))
        return sha256(payload.encode("ascii")).hexdigest()[:24]

    def _root_score(self, program: _Program, cue: tuple[int, ...]) -> float:
        edge = self._root_edge(program)
        plasticity = self._plasticity_ticks.get(edge, 0)
        if program.exact_cue is not None:
            supported = cue == program.exact_cue
        else:
            index = program.controller_index
            supported = (
                index is not None
                and len(cue) == program.cue_arity
                and cue[index] == program.cue_value
            )
        return max(0.0, (5.0 if supported else 0.0) + plasticity)

    def _child_score(
        self,
        parent: _Program,
        action_index: int,
        child: _Program,
        cue: tuple[int, ...],
    ) -> float:
        index = child.controller_index
        supported = (
            index is not None
            and len(cue) == child.cue_arity
            and cue[index] == child.cue_value
        )
        edge = self._child_edge(parent, action_index, child)
        return max(
            0.0,
            (5.0 if supported else 0.0) + self._plasticity_ticks.get(edge, 0),
        )

    @staticmethod
    def _root_edge(program: _Program) -> str:
        return f"root:{program.key}"

    @staticmethod
    def _child_edge(parent: _Program, action_index: int, child: _Program) -> str:
        return f"child:{parent.key}:{action_index}:{child.key}"

    @staticmethod
    def _materialize(atom: _Atom, cue: tuple[int, ...]) -> int:
        if isinstance(atom, _CopyRole):
            return cue[atom.cue_index]
        return atom.token

    def _operation_for_control(self, cell: int) -> str:
        for operation, control_cell in self._control_cells.items():
            if cell == control_cell:
                return operation
        raise RuntimeError("non-control cell won a control race")

    def _race(
        self,
        *,
        phase: str,
        operation_by_cell: dict[int, str],
        depth: int,
        cells: Sequence[int],
        scores: Sequence[float],
        steps: list[AssemblyRaceStep],
    ) -> int:
        if len(cells) != len(scores):
            raise RuntimeError("race cells and synaptic scores disagree")
        merged: dict[int, float] = {}
        for cell, score in zip(cells, scores, strict=True):
            merged[cell] = max(merged.get(cell, 0.0), max(0.0, float(score)))
        candidate_cells = tuple(merged)
        candidate_scores = tuple(merged[cell] for cell in candidate_cells)
        maximum = max(candidate_scores, default=0.0)
        currents = tuple(
            1_800.0 if maximum > 0.0 and score == maximum else 500.0
            for score in candidate_scores
        )
        race = self._cortex.race_currents_first_spike(
            candidate_cells,
            currents,
        )
        winner = race.winner
        steps.append(
            AssemblyRaceStep(
                phase=phase,
                operation=operation_by_cell[winner],
                depth=depth,
                candidate_cells=candidate_cells,
                candidate_scores=candidate_scores,
                winner_cell=winner,
                spike_time_ms=race.spike_time_ms,
                simulated_steps=race.simulated_steps,
            )
        )
        return winner

    def _compute_digest(self) -> str:
        programs = []
        for program in sorted(self._programs.values(), key=lambda value: value.key):
            programs.append(
                {
                    "actions": [
                        {
                            "atom": None if action.atom is None else _atom_payload(action.atom),
                            "atom_cell": action.atom_cell,
                            "child_group": action.child_group,
                            "emission_slot": action.emission_slot,
                            "kind": action.kind,
                            "state_cell": action.state_cell,
                        }
                        for action in program.actions
                    ],
                    "controller_index": program.controller_index,
                    "cue_arity": program.cue_arity,
                    "cue_value": program.cue_value,
                    "exact_cue": program.exact_cue,
                    "group": program.group,
                    "is_root": program.is_root,
                    "key": program.key,
                    "root_cell": program.root_cell,
                }
            )
        shapes = []
        for shape_key, state in sorted(self._shape_states.items()):
            position_support = []
            for position, counter in sorted(state.position_atom_support.items()):
                position_support.append(
                    (
                        position,
                        tuple(
                            sorted(
                                (
                                    (_atom_payload(atom), support)
                                    for atom, support in counter.items()
                                ),
                                key=str,
                            )
                        ),
                    )
                )
            conditional_support = []
            for condition, counter in sorted(state.conditional_support.items()):
                conditional_support.append(
                    (
                        condition,
                        tuple(
                            sorted(
                                (
                                    (_atom_payload(atom), support)
                                    for atom, support in counter.items()
                                ),
                                key=str,
                            )
                        ),
                    )
                )
            shapes.append(
                {
                    "compositional": state.compositional,
                    "conditional_support": conditional_support,
                    "contributions": tuple(
                        (
                            cue,
                            tuple(_atom_payload(atom) for atom in contribution.atoms),
                        )
                        for cue, contribution in sorted(state.contributions.items())
                    ),
                    "cue_value_support": tuple(
                        (*key, support)
                        for key, support in sorted(state.cue_value_support.items())
                    ),
                    "position_atom_support": position_support,
                    "shape": shape_key,
                }
            )
        payload = {
            "cells": tuple(
                (repr(key), cell) for key, cell in sorted(self._cell_by_key.items(), key=str)
            ),
            "config": asdict(self.config),
            "plasticity": tuple(sorted(self._plasticity_ticks.items())),
            "programs": programs,
            "schema": self._MODEL_SCHEMA,
            "shapes": shapes,
        }
        return sha256(_canonical_json(payload).encode("ascii")).hexdigest()
