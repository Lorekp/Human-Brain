"""Public scalar typed-program editing through cell-raced proposals.

This is an experiment-only architecture slice.  The host applies exactly one
cell-elected, type-valid edit, executes explicit public cases, and returns raw
outcomes plus the signed pass-count delta.  The public oracle accepts only a
strict improvement and supplies that delta as a local policy modulation; this
is a cell-raced proposal/hill-climb plumbing probe, not cell-only feedback
learning.  It never enumerates sibling programs, retains a best candidate, or
derives an edit from expected outputs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.cells import AdaptiveGlifPopulation  # noqa: E402
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

_SEED = 17
_BASE_CURRENT_PA = 600.0
_NOVELTY_CURRENT_PA = 48.0
_FEEDBACK_STEP_PA = 8.0
_MAX_POLICY_PA = 32.0
_DECISION_CURRENT_PA = 64.0
_RACE_STEPS = 600
_OPERATIONS = ("absolute", "add", "subtract")
_LITERALS = tuple(range(-9, 10))
_ORIENTATIONS = ("existing-left", "existing-right")
_DECISIONS = ("KEEP", "UNDO")


@dataclass(frozen=True, slots=True)
class ScalarCase:
    arguments: tuple[int, ...]
    expected: int


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    arguments: tuple[int, ...]
    expected: int
    actual: int | None
    error: str | None
    passed: bool


@dataclass(frozen=True, slots=True)
class NodeSite:
    path: tuple[int, ...]
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class CellRace:
    bank: str
    candidates: tuple[str, ...]
    currents_pa: tuple[float, ...]
    winner_cell: int
    winner: str
    spike_step: int
    simultaneous_spike_candidates: int


@dataclass(frozen=True, slots=True)
class LocalWeightUpdate:
    bank: str
    candidate: str
    before_pa: float
    after_pa: float


@dataclass(frozen=True, slots=True)
class EditStep:
    execution: int
    before_tokens: tuple[str, ...]
    proposal_tokens: tuple[str, ...]
    after_tokens: tuple[str, ...]
    selected_path: tuple[int, ...]
    operation: str
    literal: int | None
    orientation: str | None
    before_score: int
    proposal_score: int
    signed_delta: int
    decision: str
    outcomes: tuple[CaseOutcome, ...]
    races: tuple[CellRace, ...]
    local_updates: tuple[LocalWeightUpdate, ...]


@dataclass(frozen=True, slots=True)
class EditLoopResult:
    seed: int
    budget: int
    initial_program: tp.Program
    initial_race: CellRace
    final_program: tp.Program
    initial_score: int
    final_score: int
    initial_outcomes: tuple[CaseOutcome, ...]
    steps: tuple[EditStep, ...]
    controller_before_sha256: str
    controller_after_sha256: str

    @property
    def strict_improvement(self) -> bool:
        return self.final_score > self.initial_score


class _CellRacedProposalController:
    """Host-stored novelty/policy currents consumed by fitted-cell races."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._policy_pa: dict[tuple[str, str], float] = {}
        self._uses: dict[tuple[str, str], int] = {}

    def checkpoint_sha256(self) -> str:
        record = {
            "seed": self.seed,
            "policy_pa": sorted(
                (bank, candidate, value)
                for (bank, candidate), value in self._policy_pa.items()
            ),
            "uses": sorted(
                (bank, candidate, value)
                for (bank, candidate), value in self._uses.items()
            ),
        }
        return sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()

    def choose(
        self,
        bank: str,
        candidates: Sequence[Hashable],
    ) -> tuple[Hashable, CellRace]:
        materialized = tuple(candidates)
        if not materialized:
            raise ValueError(f"{bank} race needs at least one candidate")
        labels = tuple(_candidate_label(candidate) for candidate in materialized)
        use_counts = tuple(self._uses.get((bank, label), 0) for label in labels)
        minimum_use = min(use_counts)
        currents = np.asarray(
            [
                _BASE_CURRENT_PA
                + _fixed_heterogeneity_pa(self.seed, bank, label)
                + self._policy_pa.get((bank, label), 0.0)
                + (_NOVELTY_CURRENT_PA if uses == minimum_use else 0.0)
                for label, uses in zip(labels, use_counts)
            ],
            dtype=np.float64,
        )
        winner_cell, step, simultaneous = _physical_race(currents)
        winner = materialized[winner_cell]
        winner_label = labels[winner_cell]
        self._uses[(bank, winner_label)] = use_counts[winner_cell] + 1
        return winner, CellRace(
            bank=bank,
            candidates=labels,
            currents_pa=tuple(float(value) for value in currents),
            winner_cell=winner_cell,
            winner=winner_label,
            spike_step=step,
            simultaneous_spike_candidates=simultaneous,
        )

    def decide(self, delta: int) -> tuple[str, CellRace]:
        currents = np.asarray(
            [
                _BASE_CURRENT_PA
                + (float(delta > 0) * _DECISION_CURRENT_PA),
                _BASE_CURRENT_PA
                + (float(delta <= 0) * _DECISION_CURRENT_PA),
            ],
            dtype=np.float64,
        )
        winner_cell, step, simultaneous = _physical_race(currents)
        winner = _DECISIONS[winner_cell]
        return winner, CellRace(
            bank="decision",
            candidates=_DECISIONS,
            currents_pa=tuple(float(value) for value in currents),
            winner_cell=winner_cell,
            winner=winner,
            spike_step=step,
            simultaneous_spike_candidates=simultaneous,
        )

    def learn(
        self,
        selected: Sequence[tuple[str, Hashable]],
        delta: int,
    ) -> tuple[LocalWeightUpdate, ...]:
        modulation = 1 if delta > 0 else -1 if delta < 0 else 0
        updates: list[LocalWeightUpdate] = []
        for bank, candidate in selected:
            label = _candidate_label(candidate)
            key = (bank, label)
            before = self._policy_pa.get(key, 0.0)
            after = float(
                np.clip(
                    before + modulation * _FEEDBACK_STEP_PA,
                    -_MAX_POLICY_PA,
                    _MAX_POLICY_PA,
                )
            )
            if after != before:
                self._policy_pa[key] = after
                updates.append(LocalWeightUpdate(bank, label, before, after))
        return tuple(updates)


def _candidate_label(candidate: Hashable) -> str:
    if isinstance(candidate, tuple):
        return "root" if not candidate else ".".join(str(part) for part in candidate)
    return str(candidate)


def _fixed_heterogeneity_pa(seed: int, bank: str, candidate: str) -> float:
    digest = sha256(f"{seed}:{bank}:{candidate}".encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") / 4096.0


def _physical_race(currents_pa: np.ndarray) -> tuple[int, int, int]:
    population = AdaptiveGlifPopulation(int(currents_pa.size))
    for step in range(_RACE_STEPS):
        fired = np.flatnonzero(population.step(currents_pa))
        if fired.size:
            return int(fired[0]), step, int(fired.size)
    raise RuntimeError("cell edit race did not spike within its bound")


def _literal_tokens(value: int) -> tuple[str, ...]:
    sign = "-" if value < 0 else "+"
    return ("integer", sign, *tuple(str(abs(value))), "end")


def _node_sites(tokens: tuple[str, ...]) -> tuple[NodeSite, ...]:
    sites: list[NodeSite] = []

    def walk(index: int, path: tuple[int, ...]) -> int:
        if index >= len(tokens):
            raise tp.ProgramFormatError("edit-site traversal reached an incomplete program")
        try:
            spec = tp._SPEC_BY_TOKEN[tokens[index]]
        except KeyError as error:
            raise tp.ProgramFormatError("edit-site traversal found an unknown token") from error
        start = index
        if spec.role == "variable":
            stop = index + 1
        elif spec.role == "literal":
            stop = index + 1
            while stop < len(tokens) and tokens[stop] != "end":
                stop += 1
            if stop >= len(tokens):
                raise tp.ProgramFormatError("edit-site traversal found an open literal")
            stop += 1
        elif spec.role in {"operation", "binder"}:
            stop = index + 1
            child_count = len(spec.arguments) if spec.role == "operation" else 1
            for child in range(child_count):
                stop = walk(stop, (*path, child))
        else:
            raise tp.ProgramFormatError("structural token cannot start an edit site")
        sites.append(NodeSite(path, start, stop))
        return stop

    stop = walk(0, ())
    if stop != len(tokens):
        raise tp.ProgramFormatError("edit-site traversal left trailing tokens")
    return tuple(sorted(sites, key=lambda site: (len(site.path), site.path)))


def _proposal(
    program: tp.Program,
    site: NodeSite,
    operation: str,
    literal: int | None,
    orientation: str | None,
) -> tp.Program:
    existing = program.tokens[site.start : site.stop]
    if operation == "absolute":
        replacement = (operation, *existing)
    else:
        if literal is None or orientation not in _ORIENTATIONS:
            raise ValueError("binary edit needs a cell-selected literal and orientation")
        encoded_literal = _literal_tokens(literal)
        if orientation == "existing-left":
            replacement = (operation, *existing, *encoded_literal)
        else:
            replacement = (operation, *encoded_literal, *existing)
    tokens = (*program.tokens[: site.start], *replacement, *program.tokens[site.stop :])
    proposal = tp.Program(tuple(tokens), program.signature)
    tp.compile(proposal)
    return proposal


def _execute(
    program: tp.Program,
    cases: Sequence[ScalarCase],
) -> tuple[int, tuple[CaseOutcome, ...]]:
    try:
        compiled = tp.compile(program)
        namespace: dict[str, object] = {}
        exec(
            compiled.source,
            {"__builtins__": {"abs": abs, "max": max, "min": min, "sum": sum}},
            namespace,
        )
        function = namespace[program.signature.entry_point]
    except Exception as error:
        outcomes = tuple(
            CaseOutcome(
                case.arguments,
                case.expected,
                None,
                f"{type(error).__name__}: {error}",
                False,
            )
            for case in cases
        )
        return 0, outcomes

    outcomes_list: list[CaseOutcome] = []
    for case in cases:
        try:
            actual = function(*case.arguments)  # type: ignore[operator]
            passed = type(actual) is int and actual == case.expected
            outcomes_list.append(
                CaseOutcome(case.arguments, case.expected, actual, None, passed)
            )
        except Exception as error:
            outcomes_list.append(
                CaseOutcome(
                    case.arguments,
                    case.expected,
                    None,
                    f"{type(error).__name__}: {error}",
                    False,
                )
            )
    outcomes = tuple(outcomes_list)
    return sum(outcome.passed for outcome in outcomes), outcomes


def _initial_program(
    signature: TypedSignature,
    controller: _CellRacedProposalController,
) -> tuple[tp.Program, CellRace]:
    state = tp._TypedHoleState(signature)
    variables = tuple(
        spec.token for spec in state.allowed() if spec.role == "variable"
    )
    selected, race = controller.choose("initial-leaf", variables)
    program = tp.Program((str(selected),), signature)
    tp.compile(program)
    return program, race


def run_edit_loop(
    signature: TypedSignature,
    cases: Sequence[ScalarCase],
    *,
    budget: int = 8,
    seed: int = _SEED,
) -> EditLoopResult:
    """Run one fixed-seed, last-incumbent-only public feedback trajectory."""
    if budget < 1:
        raise ValueError("edit budget must be positive")
    materialized_cases = tuple(cases)
    if not materialized_cases:
        raise ValueError("edit loop needs explicit public cases")
    controller = _CellRacedProposalController(seed)
    controller_before = controller.checkpoint_sha256()
    incumbent, initial_race = _initial_program(signature, controller)
    initial_program = incumbent
    initial_score, initial_outcomes = _execute(incumbent, materialized_cases)
    score = initial_score
    steps: list[EditStep] = []

    for execution in range(1, budget + 1):
        before = incumbent
        score_before = score
        sites = _node_sites(before.tokens)
        paths = tuple(site.path for site in sites)
        selected_path, path_race = controller.choose("path", paths)
        site = next(site for site in sites if site.path == selected_path)
        operation, operation_race = controller.choose("operation", _OPERATIONS)
        selected: list[tuple[str, Hashable]] = [
            ("path", selected_path),
            ("operation", operation),
        ]
        races: list[CellRace] = [path_race, operation_race]
        literal: int | None = None
        orientation: str | None = None
        if operation != "absolute":
            literal_value, literal_race = controller.choose("literal", _LITERALS)
            orientation_value, orientation_race = controller.choose(
                "orientation", _ORIENTATIONS
            )
            literal = int(literal_value)
            orientation = str(orientation_value)
            selected.extend(
                (("literal", literal), ("orientation", orientation))
            )
            races.extend((literal_race, orientation_race))

        proposal = _proposal(before, site, str(operation), literal, orientation)
        proposal_score, outcomes = _execute(proposal, materialized_cases)
        delta = proposal_score - score_before
        decision, decision_race = controller.decide(delta)
        races.append(decision_race)
        local_updates = controller.learn(selected, delta)
        if decision == "KEEP":
            incumbent = proposal
            score = proposal_score
        elif decision != "UNDO":
            raise AssertionError("decision race returned an unknown action")
        steps.append(
            EditStep(
                execution=execution,
                before_tokens=before.tokens,
                proposal_tokens=proposal.tokens,
                after_tokens=incumbent.tokens,
                selected_path=tuple(selected_path),
                operation=str(operation),
                literal=literal,
                orientation=orientation,
                before_score=score_before,
                proposal_score=proposal_score,
                signed_delta=delta,
                decision=decision,
                outcomes=outcomes,
                races=tuple(races),
                local_updates=local_updates,
            )
        )

    return EditLoopResult(
        seed=seed,
        budget=budget,
        initial_program=initial_program,
        initial_race=initial_race,
        final_program=incumbent,
        initial_score=initial_score,
        final_score=score,
        initial_outcomes=initial_outcomes,
        steps=tuple(steps),
        controller_before_sha256=controller_before,
        controller_after_sha256=controller.checkpoint_sha256(),
    )


def public_canary() -> tuple[TypedSignature, tuple[ScalarCase, ...]]:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("value", integer),),
        result=integer,
    )
    cases = (
        ScalarCase((-5,), 8),
        ScalarCase((-2,), 5),
        ScalarCase((0,), 3),
        ScalarCase((4,), 7),
    )
    return signature, cases


def run_public_canary() -> EditLoopResult:
    signature, cases = public_canary()
    return run_edit_loop(signature, cases, budget=8, seed=_SEED)


def main() -> int:
    print(json.dumps(asdict(run_public_canary()), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
