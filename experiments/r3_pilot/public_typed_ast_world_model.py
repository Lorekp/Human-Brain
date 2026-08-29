"""Cell-raced world model for one typed wrap edit inside an incumbent AST.

The observation seam may inspect only the incumbent program and its subtree
outputs.  Prediction never constructs or executes an edited program: it sends
the selected subtree values through the target-free total successor, then
replays each supported scalar ancestor as another physical successor race.
Only :func:`apply_selected_wrap` owns the typed replacement actuator.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import public_typed_total_successor as total  # noqa: E402
from humanbrain import typed_program as tp  # noqa: E402

UNKNOWN = total.UNKNOWN
SuccessorState = total.SuccessorState

_ACTION_BY_LABEL = {
    action.label: action for action in total.banked.banked.ACTIONS
}
WRAP_ACTION_LABELS = tuple(
    action.label for action in total.banked.banked.ACTIONS if action.kind != "done"
)
_UNKNOWN_PROPAGATION_ACTION = "DONE"


@dataclass(frozen=True, slots=True)
class SubtreeObservation:
    path: tuple[int, ...]
    tokens: tuple[str, ...]
    outputs: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class CurrentProgramObservation:
    program_sha256: str
    arguments: tuple[int, ...]
    sites: tuple[SubtreeObservation, ...]
    sha256: str

    def site(self, path: tuple[int, ...]) -> SubtreeObservation:
        for site in self.sites:
            if site.path == path:
                return site
        raise ValueError(f"incumbent program has no edit site at path {path!r}")


class _UnsupportedIncumbent(RuntimeError):
    pass


def _literal(tokens: tuple[str, ...], index: int) -> tuple[int, int]:
    if index + 3 >= len(tokens) or tokens[index] != "integer":
        raise _UnsupportedIncumbent("not a canonical integer literal")
    sign = tokens[index + 1]
    if sign not in {"+", "-"}:
        raise _UnsupportedIncumbent("literal sign is unsupported")
    cursor = index + 2
    digits: list[str] = []
    while cursor < len(tokens) and tokens[cursor] != "end":
        if len(tokens[cursor]) != 1 or not tokens[cursor].isdigit():
            raise _UnsupportedIncumbent("literal digit is unsupported")
        digits.append(tokens[cursor])
        cursor += 1
    if cursor >= len(tokens) or not digits:
        raise _UnsupportedIncumbent("literal is incomplete")
    magnitude = int("".join(digits))
    return (-magnitude if sign == "-" else magnitude), cursor + 1


def _observe_tokens(
    tokens: tuple[str, ...],
    signature: tp.TypedSignature,
    arguments: tuple[int, ...],
) -> tuple[int, ...] | None:
    try:
        incumbent_subtree = tp.Program(tokens, signature)
        return total.banked.banked._execute(incumbent_subtree, arguments)
    except (tp.ProgramFormatError, TypeError, ValueError):
        return None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def observe_current_program(
    program: tp.Program,
    arguments: Sequence[int],
) -> CurrentProgramObservation:
    """Materialize path-stable outputs for the incumbent program only."""
    values = tuple(arguments)
    if not values or not all(type(value) is int for value in values):
        raise ValueError("current-program observation needs scalar integer arguments")
    sites = tuple(
        SubtreeObservation(
            path=site.path,
            tokens=site.tokens,
            outputs=_observe_tokens(site.tokens, program.signature, values),
        )
        for site in tp.program_edit_sites(program)
    )
    record = {
        "program_sha256": program.sha256,
        "arguments": values,
        "sites": tuple(asdict(site) for site in sites),
    }
    return CurrentProgramObservation(
        program_sha256=program.sha256,
        arguments=values,
        sites=sites,
        sha256=sha256(_canonical_json(record)).hexdigest(),
    )


def _positive_literal_amount(tokens: tuple[str, ...]) -> int | None:
    try:
        value, stop = _literal(tokens, 0)
    except _UnsupportedIncumbent:
        return None
    if stop != len(tokens) or not 1 <= value <= 4:
        return None
    return value


def _ancestor_action(
    observation: CurrentProgramObservation,
    parent_path: tuple[int, ...],
    child_index: int,
) -> tuple[str | None, str | None]:
    parent = observation.site(parent_path)
    head = parent.tokens[0]
    if head == "absolute" and child_index == 0:
        return "WRAP_ABS", None
    if head not in {"add", "subtract"} or child_index not in {0, 1}:
        return None, f"ancestor {parent_path!r} is outside the scalar wrap grammar"
    if head == "subtract" and child_index == 1:
        return None, f"right-child subtraction at {parent_path!r} is not unary-recomposable"
    sibling_path = (*parent_path, 1 - child_index)
    amount = _positive_literal_amount(observation.site(sibling_path).tokens)
    if amount is None:
        return None, f"ancestor {parent_path!r} lacks a supported positive literal"
    if head == "add":
        return f"WRAP_ADD[{amount}]", None
    return f"WRAP_SUB[{amount}]", None


@dataclass(frozen=True, slots=True)
class AncestorPrediction:
    path: tuple[int, ...]
    action: str | None
    predicted_outputs: tuple[SuccessorState, ...]
    races: tuple[total.banked.CellRace, ...]
    supported: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class AstEditPrediction:
    path: tuple[int, ...]
    action: str
    observation_sha256: str
    current_subtree_outputs: tuple[int, ...] | None
    predicted_local_outputs: tuple[SuccessorState, ...]
    ancestor_actions: tuple[str | None, ...]
    ancestors: tuple[AncestorPrediction, ...]
    predicted_root_outputs: tuple[SuccessorState, ...]
    races: tuple[total.banked.CellRace, ...]
    supported: bool
    unsupported_reason: str | None
    successor_checkpoint_before_sha256: str
    successor_checkpoint_after_sha256: str


def _total_state(value: int) -> SuccessorState:
    if total.banked.VALUE_MIN <= value <= total.banked.VALUE_MAX:
        return value
    return UNKNOWN


def _world_checkpoint_sha256(successor_sha256: str) -> str:
    digest = sha256()
    digest.update(successor_sha256.encode("ascii"))
    digest.update(_canonical_json(WRAP_ACTION_LABELS))
    digest.update(b"ast-world-model-v1")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenAstWorldModel:
    successor: total.FrozenTotalSuccessor
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        return _world_checkpoint_sha256(self.successor.live_checkpoint_sha256)

    def predict(
        self,
        observation: CurrentProgramObservation,
        path: tuple[int, ...],
        action_label: str,
    ) -> AstEditPrediction:
        """Predict one wrap and its incumbent ancestor chain in physical cells."""
        if action_label not in WRAP_ACTION_LABELS:
            raise ValueError("AST world model needs one fixed generic wrap action")
        before_sha = self.live_checkpoint_sha256
        if before_sha != self.checkpoint_sha256:
            raise RuntimeError("AST world-model checkpoint changed before prediction")
        selected = observation.site(path)
        unsupported_reason: str | None = None
        if selected.outputs is None:
            states: tuple[SuccessorState, ...] = tuple(
                UNKNOWN for _ in observation.arguments
            )
            unsupported_reason = "selected incumbent subtree has no scalar observation"
        else:
            states = tuple(_total_state(value) for value in selected.outputs)
            if any(state is UNKNOWN for state in states):
                unsupported_reason = "selected subtree left the total raw-value range"

        local_outputs, local_races = self.successor.predict_many(states, action_label)
        if any(state is UNKNOWN for state in local_outputs) and unsupported_reason is None:
            unsupported_reason = "selected pre/action transition is outside learned support"
        current = local_outputs
        all_races: list[total.banked.CellRace] = list(local_races)
        ancestors: list[AncestorPrediction] = []
        for depth in range(len(path) - 1, -1, -1):
            parent_path = path[:depth]
            action, reason = _ancestor_action(
                observation,
                parent_path,
                path[depth],
            )
            physical_action = action or _UNKNOWN_PROPAGATION_ACTION
            physical_inputs = (
                current
                if action is not None
                else tuple(UNKNOWN for _ in observation.arguments)
            )
            current, races = self.successor.predict_many(
                physical_inputs,
                physical_action,
            )
            all_races.extend(races)
            ancestors.append(
                AncestorPrediction(
                    path=parent_path,
                    action=action,
                    predicted_outputs=current,
                    races=races,
                    supported=action is not None,
                    reason=reason,
                )
            )
            if reason is not None and unsupported_reason is None:
                unsupported_reason = reason

        after_sha = self.live_checkpoint_sha256
        if after_sha != before_sha:
            raise RuntimeError("frozen AST world model changed during prediction")
        supported = unsupported_reason is None and all(
            state is not UNKNOWN for state in current
        )
        return AstEditPrediction(
            path=path,
            action=action_label,
            observation_sha256=observation.sha256,
            current_subtree_outputs=selected.outputs,
            predicted_local_outputs=local_outputs,
            ancestor_actions=tuple(ancestor.action for ancestor in ancestors),
            ancestors=tuple(ancestors),
            predicted_root_outputs=current,
            races=tuple(all_races),
            supported=supported,
            unsupported_reason=unsupported_reason,
            successor_checkpoint_before_sha256=before_sha,
            successor_checkpoint_after_sha256=after_sha,
        )


@dataclass(frozen=True, slots=True)
class AstWorldTrainingSummary:
    total_successor_checkpoint_sha256: str
    checkpoint_sha256: str
    target_free_observations: int
    wrap_actions: tuple[str, ...]
    claim_boundary: str


def train_public_ast_world_model(
) -> tuple[FrozenAstWorldModel, AstWorldTrainingSummary]:
    """Train only the target-free total successor and freeze AST recomposition."""
    successor, summary = total.train_public_total_successor()
    checkpoint_sha = _world_checkpoint_sha256(successor.checkpoint_sha256)
    model = FrozenAstWorldModel(successor, checkpoint_sha)
    return model, AstWorldTrainingSummary(
        total_successor_checkpoint_sha256=successor.checkpoint_sha256,
        checkpoint_sha256=checkpoint_sha,
        target_free_observations=summary.observations,
        wrap_actions=WRAP_ACTION_LABELS,
        claim_boundary=(
            "Incumbent-AST observation and target-free one-edit prediction only; "
            "no candidate execution, composite target, planner result, or gate evidence."
        ),
    )


def _literal_tokens(value: int) -> tuple[str, ...]:
    return ("integer", "+", *tuple(str(value)), "end")


def apply_selected_wrap(
    program: tp.Program,
    path: tuple[int, ...],
    action_label: str,
) -> tp.Program:
    """Actuate one already-selected edit; this is the sole mutation seam."""
    if action_label not in WRAP_ACTION_LABELS:
        raise ValueError("selected AST action is not a generic wrap")
    selected = next(
        (site for site in tp.program_edit_sites(program) if site.path == path),
        None,
    )
    if selected is None:
        raise ValueError(f"program has no edit site at path {path!r}")
    action = _ACTION_BY_LABEL[action_label]
    if action.kind == "absolute":
        replacement = ("absolute", *selected.tokens)
    elif action.kind in {"add", "subtract"}:
        replacement = (
            action.kind,
            *selected.tokens,
            *_literal_tokens(action.amount),
        )
    else:
        raise AssertionError("wrap action bank contains an unknown operation")
    return tp.replace_subtree(program, path, replacement)
