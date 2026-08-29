"""Reusable factorized H=2 planner for public typed Int development.

The incumbent alone is compiled and observed.  Hypothetical edits are carried
by a target-free typed operator successor plus the learned structural successor;
no candidate program is constructed, compiled, executed, or host-ranked.  The
compiler owns action legality through ``FactorActionSite`` descriptors.  A
replaceable behavior adapter keeps the Int value-reassembly boundary separate
from enumeration and planning so List behavior can be added independently.

This module is a development mechanism, not a result runner or evidence claim.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Protocol

import numpy as np
from experiments.r3_pilot import public_typed_factor_h1_root as base_h1
from experiments.r3_pilot import public_typed_factor_motor as factor_motor
from experiments.r3_pilot import public_typed_factor_value_race as value_race
from experiments.r3_pilot import public_typed_operator_successor as operator
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_program_observation as root_observation
from experiments.r3_pilot import public_typed_prompt_factors as prompt_factors
from experiments.r3_pilot import public_typed_structure_successor as structure
from humanbrain import typed_program as tp
from humanbrain.resting_glif_race import (
    CompiledRestingGlifRace,
    RestingGlifRaceFailure,
    RestingGlifRaceResult,
)
from humanbrain.resting_glif_stop_bounded_tiers import (
    CompiledStopBoundedRestingGlifTiers,
    StopBoundedTierResult,
)
from humanbrain.resting_glif_tier_beam import (
    RETAINED,
    STOP_LEADING_TIER,
    CompiledRestingGlifTierBeam,
    RestingGlifTierBeamResult,
)
from humanbrain.typed_program_types import TypedSignature

BEAM_WIDTH = 4
STATUS_SELECTED = "SELECTED"
STATUS_DONE = "DONE"
STATUS_STOP_FIRST_FRONTIER = "STOP_FIRST_FRONTIER"
STATUS_STOP_TERMINAL_VALUE = "STOP_TERMINAL_VALUE"
STATUS_STOP_TERMINAL_STRUCTURE = "STOP_TERMINAL_STRUCTURE"
STATUS_STOP_PROMPT = "STOP_PROMPT"

_VALUE_MIN_CURRENT_PA = -100_000.0
_VALUE_MAX_CURRENT_PA = 10_000.0
_VALUE_INHIBITION_PA = 120_000.0
_STRUCTURE_SUPPORTED_CURRENT_PA = 1_200.0
_STRUCTURE_STOP_CURRENT_PA = 650.0
_PROMPT_STOP_CURRENT_PA = 650.0
_EDIT_COST_CURRENT_PA = 40.0
_UNKNOWN_VALUE = object()
_DONE = tp.FactorAction((), "DONE", "NONE", "NONE", "NONE", None)
_PROMPT_RACE = CompiledRestingGlifRace.compile(step_count=value_race._RACE_STEPS)
_TRUSTED_BUILTINS = {"abs": abs, "max": max, "min": min, "sum": sum}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _action_record(action: tp.FactorAction) -> dict[str, object]:
    return asdict(action)


def _output_type(type_label: str) -> output_cells.OutputType | None:
    return {
        "Int": output_cells.OutputType.INT,
        "Bool": output_cells.OutputType.BOOL,
        "List[Int]": output_cells.OutputType.LIST_INT,
    }.get(type_label)


def _unknown(type_label: str) -> output_cells.TypedOutputEncoding:
    declared = _output_type(type_label)
    if declared is None:
        return output_cells.TypedOutputCellBank.encode(_UNKNOWN_VALUE)
    return output_cells.TypedOutputCellBank.encode(
        _UNKNOWN_VALUE,
        declared_type=declared,
    )


def _encode(type_label: str, value: object) -> output_cells.TypedOutputEncoding:
    declared = _output_type(type_label)
    if declared is None:
        return _unknown(type_label)
    materialized = list(value) if declared is output_cells.OutputType.LIST_INT else value
    return output_cells.TypedOutputCellBank.encode(
        materialized,
        declared_type=declared,
    )


def _literal_value(tokens: tuple[str, ...]) -> int | None:
    if len(tokens) < 4 or tokens[0] != "integer" or tokens[1] not in {"+", "-"}:
        return None
    if tokens[-1] != "end" or not all(token.isdigit() for token in tokens[2:-1]):
        return None
    magnitude = int("".join(tokens[2:-1]))
    return -magnitude if tokens[1] == "-" else magnitude


def _site_callable(compiled: tp.CompiledEditSite):
    namespace: dict[str, object] = {"__builtins__": dict(_TRUSTED_BUILTINS)}
    exec(compiled.source, namespace)
    result = namespace.get("edit_site")
    if not callable(result):
        raise RuntimeError("trusted edit-site compiler emitted no callable")
    return result


@dataclass(frozen=True, slots=True)
class FactorSiteValue:
    site: tp.FactorActionSite
    outputs: tuple[output_cells.TypedOutputEncoding, ...]
    literal_value: int | None


@dataclass(frozen=True, slots=True)
class FactorStateObservation:
    """Physical value/structure state for a real or hypothetical incumbent."""

    program_sha256: str | None
    signature: TypedSignature
    cases: tuple[tuple[int, ...], ...]
    structure: structure.StructureObservation
    sites: tuple[FactorSiteValue, ...]
    root_scope_rows: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...]
    sha256: str

    def site(self, path: tuple[int, ...]) -> FactorSiteValue:
        for item in self.sites:
            if item.site.path == path:
                return item
        raise ValueError(f"factor state has no site at path {path!r}")

    @property
    def action_sites(self) -> tuple[tp.FactorActionSite, ...]:
        return tuple(item.site for item in self.sites)

    @property
    def root_outputs(self) -> tuple[output_cells.TypedOutputEncoding, ...]:
        return self.site(()).outputs


def _state_observation(
    *,
    program_sha256: str | None,
    signature: TypedSignature,
    cases: tuple[tuple[int, ...], ...],
    observed_structure: structure.StructureObservation,
    sites: tuple[FactorSiteValue, ...],
    root_scope_rows: tuple[tuple[output_cells.TypedOutputEncoding, ...], ...],
) -> FactorStateObservation:
    record = {
        "program_sha256": program_sha256,
        "signature": signature.to_record(),
        "cases": cases,
        "structure_sha256": observed_structure.sha256,
        "sites": tuple(
            {
                "site": asdict(item.site),
                "outputs": tuple(output.sha256 for output in item.outputs),
                "literal_value": item.literal_value,
            }
            for item in sites
        ),
        "scopes": tuple(
            tuple(value.sha256 for value in row) for row in root_scope_rows
        ),
    }
    return FactorStateObservation(
        program_sha256=program_sha256,
        signature=signature,
        cases=cases,
        structure=observed_structure,
        sites=sites,
        root_scope_rows=root_scope_rows,
        sha256=sha256(_canonical_json(record)).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class FactorBehaviorPrediction:
    action: tp.FactorAction
    local_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    ancestor_outputs: tuple[
        tuple[tuple[int, ...], tuple[output_cells.TypedOutputEncoding, ...]], ...
    ]
    root_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    races: tuple[operator.OperatorRace, ...]
    supported: bool
    unsupported_reason: str | None
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


class FactorBehaviorAdapter(Protocol):
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str: ...

    def observe(
        self,
        program: tp.Program,
        cases: Sequence[tuple[object, ...]],
    ) -> FactorStateObservation: ...

    def predict_root_many(
        self,
        state: FactorStateObservation,
        actions: Sequence[tp.FactorAction],
    ) -> tuple[FactorBehaviorPrediction, ...]: ...

    def predict_state_values(
        self,
        state: FactorStateObservation,
        action: tp.FactorAction,
        predicted_structure: structure.StructureObservation,
        behavior: FactorBehaviorPrediction,
    ) -> tuple[tuple[FactorSiteValue, ...], tuple[operator.OperatorRace, ...]]: ...


def _behavior_checkpoint_sha256(operator_sha256: str) -> str:
    digest = sha256()
    digest.update(b"public-typed-factor-h2-int-behavior-v1\0")
    digest.update(operator_sha256.encode("ascii"))
    return digest.hexdigest()


def _inside_binder(state: FactorStateObservation, path: tuple[int, ...]) -> bool:
    return any(
        item.site.is_binder
        and len(path) > len(item.site.path)
        and path[: len(item.site.path)] == item.site.path
        for item in state.sites
    )


@dataclass(frozen=True, slots=True)
class FrozenIntFactorBehaviorAdapter:
    """Target-free learned operator effects plus bounded Int ancestor replay."""

    operator_successor: operator.FrozenTypedOperatorSuccessor
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        return _behavior_checkpoint_sha256(
            self.operator_successor.live_checkpoint_sha256
        )

    def observe(
        self,
        program: tp.Program,
        cases: Sequence[tuple[object, ...]],
    ) -> FactorStateObservation:
        if not isinstance(program, tp.Program):
            raise TypeError("factor behavior observation requires a Program")
        if (
            base_h1._declared_output_type(program.signature.result)
            is not output_cells.OutputType.INT
        ):
            raise ValueError("this behavior adapter supports Int-result programs only")
        current = root_observation.observe_current_program_root(program, cases)
        if not current.all_supported:
            raise ValueError("incumbent root observation must be physically supported")
        normalized_cases = tuple(
            tuple(int(value) for value in row.arguments) for row in current.cases
        )
        if not all(
            type(value) is int
            for row in current.cases
            for value in row.arguments
        ):
            raise ValueError("Int H=2 observation requires only scalar Int cases")
        root_scopes = base_h1._scope_rows(program, current)
        edit_sites = {site.path: site for site in tp.program_edit_sites(program)}
        action_sites = tp.factor_action_sites(program)
        values: list[FactorSiteValue] = []
        for action_site in action_sites:
            edit_site = edit_sites[action_site.path]
            if action_site.path == ():
                outputs = tuple(case.output_encoding for case in current.cases)
            else:
                declared = _output_type(action_site.result_type_label)
                try:
                    compiled = tp.compile_edit_site(program, action_site.path)
                    callable_site = _site_callable(compiled)
                    raw_scopes = tuple(
                        tuple(reversed(row.arguments)) for row in current.cases
                    )
                    if declared is None or any(
                        len(row) != len(compiled.scope_type_labels)
                        for row in raw_scopes
                    ):
                        raise ValueError("site scope is outside the Int adapter")
                    outputs = tuple(
                        _encode(action_site.result_type_label, callable_site(*row))
                        for row in raw_scopes
                    )
                except (RuntimeError, TypeError, ValueError, tp.ProgramFormatError):
                    outputs = tuple(
                        _unknown(action_site.result_type_label)
                        for _ in current.cases
                    )
            values.append(
                FactorSiteValue(
                    site=action_site,
                    outputs=outputs,
                    literal_value=_literal_value(edit_site.tokens),
                )
            )
        return _state_observation(
            program_sha256=program.sha256,
            signature=program.signature,
            cases=normalized_cases,
            observed_structure=structure.observe_program_structure(program),
            sites=tuple(values),
            root_scope_rows=root_scopes,
        )

    def _copy_rows(
        self,
        rows: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[
        tuple[output_cells.TypedOutputEncoding, ...],
        tuple[operator.OperatorRace, ...],
    ]:
        effects = self.operator_successor.predict_actions_many(rows, (_DONE,))[0]
        return (
            tuple(effect.output for effect in effects),
            tuple(race for effect in effects for race in effect.races),
        )

    @staticmethod
    def _source_hint(site: FactorSiteValue) -> tuple[str, int | None] | None:
        if site.site.selected_scope_index in {0, 1}:
            return f"SCOPE{site.site.selected_scope_index}", None
        if site.literal_value is not None:
            return "LITERAL", site.literal_value
        return None

    def _unknown_rows(
        self,
        type_label: str,
        case_count: int,
    ) -> tuple[
        tuple[output_cells.TypedOutputEncoding, ...],
        tuple[operator.OperatorRace, ...],
    ]:
        return self._copy_rows(tuple(_unknown(type_label) for _ in range(case_count)))

    def _recompose(
        self,
        state: FactorStateObservation,
        action: tp.FactorAction,
        local: tuple[output_cells.TypedOutputEncoding, ...],
    ) -> tuple[
        tuple[output_cells.TypedOutputEncoding, ...],
        tuple[
            tuple[tuple[int, ...], tuple[output_cells.TypedOutputEncoding, ...]],
            ...,
        ],
        tuple[operator.OperatorRace, ...],
        str | None,
    ]:
        current = local
        ancestors: list[
            tuple[tuple[int, ...], tuple[output_cells.TypedOutputEncoding, ...]]
        ] = []
        races: list[operator.OperatorRace] = []
        reason: str | None = None
        for depth in range(len(action.path) - 1, -1, -1):
            parent_path = action.path[:depth]
            child = action.path[depth]
            parent = state.site(parent_path)
            head = parent.site.head_production
            ancestor_action: tp.FactorAction | None = None
            if head == "absolute" and child == 0:
                ancestor_action = tp.FactorAction(
                    parent_path,
                    "APPLY1",
                    "absolute",
                    "SELECTED",
                    "NONE",
                    None,
                )
            elif head in {"add", "subtract"} and child in {0, 1}:
                sibling_path = (*parent_path, 1 - child)
                try:
                    hint = self._source_hint(state.site(sibling_path))
                except ValueError:
                    hint = None
                if hint is not None:
                    source, literal = hint
                    left = "SELECTED" if child == 0 else source
                    right = source if child == 0 else "SELECTED"
                    ancestor_action = tp.FactorAction(
                        parent_path,
                        "APPLY2",
                        head,
                        left,
                        right,
                        literal,
                    )
            if ancestor_action is None:
                current, unknown_races = self._unknown_rows(
                    parent.site.result_type_label,
                    len(state.cases),
                )
                races.extend(unknown_races)
                if reason is None:
                    reason = (
                        f"ancestor {parent_path!r} is outside compiler-representable "
                        "Int reassembly"
                    )
            else:
                effects = self.operator_successor.predict_actions_many(
                    current,
                    (ancestor_action,),
                    scope_rows=state.root_scope_rows,
                )[0]
                current = tuple(effect.output for effect in effects)
                races.extend(race for effect in effects for race in effect.races)
            ancestors.append((parent_path, current))
        return current, tuple(ancestors), tuple(races), reason

    def predict_root_many(
        self,
        state: FactorStateObservation,
        actions: Sequence[tp.FactorAction],
    ) -> tuple[FactorBehaviorPrediction, ...]:
        supplied = tuple(actions)
        if not supplied or not all(type(action) is tp.FactorAction for action in supplied):
            raise TypeError("behavior prediction requires FactorAction values")
        before = self.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("Int behavior checkpoint changed before prediction")

        predictions: dict[tp.FactorAction, FactorBehaviorPrediction] = {}
        paths: list[tuple[int, ...]] = []
        for action in supplied:
            if action.path not in paths:
                paths.append(action.path)
        for path in paths:
            path_actions = tuple(action for action in supplied if action.path == path)
            try:
                selected = state.site(path)
            except ValueError:
                for action in path_actions:
                    unknown_rows, unknown_races = self._unknown_rows(
                        "Int", len(state.cases)
                    )
                    predictions[action] = FactorBehaviorPrediction(
                        action,
                        unknown_rows,
                        (),
                        unknown_rows,
                        unknown_races,
                        False,
                        "action path is absent from the predicted state",
                        before,
                        before,
                    )
                continue
            effects_by_action = self.operator_successor.predict_actions_many(
                selected.outputs,
                path_actions,
                scope_rows=state.root_scope_rows,
            )
            for action, effects in zip(path_actions, effects_by_action, strict=True):
                local = tuple(effect.output for effect in effects)
                local_races = tuple(race for effect in effects for race in effect.races)
                invalid_bound = (
                    _inside_binder(state, action.path)
                    and "BOUND0" in {action.left_source, action.right_source}
                )
                if invalid_bound:
                    local, local_races = self._unknown_rows(
                        selected.site.result_type_label,
                        len(state.cases),
                    )
                root, ancestors, ancestor_races, reason = self._recompose(
                    state,
                    action,
                    local,
                )
                if invalid_bound and reason is None:
                    reason = "BOUND0 values are outside the Int behavior adapter"
                supported = reason is None and all(item.supported for item in root)
                predictions[action] = FactorBehaviorPrediction(
                    action=action,
                    local_outputs=local,
                    ancestor_outputs=ancestors,
                    root_outputs=root,
                    races=(*local_races, *ancestor_races),
                    supported=supported,
                    unsupported_reason=reason,
                    checkpoint_before_sha256=before,
                    checkpoint_after_sha256=before,
                )
        after = self.live_checkpoint_sha256
        if after != before:
            raise RuntimeError("frozen Int behavior model changed during prediction")
        return tuple(predictions[action] for action in supplied)

    def predict_state_values(
        self,
        state: FactorStateObservation,
        action: tp.FactorAction,
        predicted_structure: structure.StructureObservation,
        behavior: FactorBehaviorPrediction,
    ) -> tuple[tuple[FactorSiteValue, ...], tuple[operator.OperatorRace, ...]]:
        """Project one predicted structure into physical Int site values."""

        if behavior.action != action:
            raise ValueError("behavior prediction and transition action differ")
        action_sites = _factor_sites_from_structure(predicted_structure)
        ancestor_rows = dict(behavior.ancestor_outputs)
        values: list[FactorSiteValue] = []
        races: list[operator.OperatorRace] = []
        for site in action_sites:
            route = _value_route(action, site.path)
            literal_value: int | None = None
            if site.path in ancestor_rows:
                rows = ancestor_rows[site.path]
            elif route.kind == "LOCAL":
                rows = behavior.local_outputs
            elif route.kind in {"SAME", "SELECTED"}:
                assert route.source_path is not None
                try:
                    source = state.site(route.source_path)
                except ValueError:
                    rows, copy_races = self._unknown_rows(
                        site.result_type_label,
                        len(state.cases),
                    )
                else:
                    rows, copy_races = self._copy_rows(source.outputs)
                    literal_value = source.literal_value
                races.extend(copy_races)
            elif route.kind in {"SCOPE0", "SCOPE1"}:
                scope_index = int(route.kind[-1])
                source_rows = tuple(
                    row[scope_index]
                    if scope_index < len(row)
                    else _unknown(site.result_type_label)
                    for row in state.root_scope_rows
                )
                rows, copy_races = self._copy_rows(source_rows)
                races.extend(copy_races)
            elif route.kind == "LITERAL" and action.literal is not None:
                source_rows = tuple(
                    _encode("Int", action.literal) for _ in state.cases
                )
                rows, copy_races = self._copy_rows(source_rows)
                races.extend(copy_races)
                literal_value = action.literal
            else:
                rows, copy_races = self._unknown_rows(
                    site.result_type_label,
                    len(state.cases),
                )
                races.extend(copy_races)
            values.append(FactorSiteValue(site, rows, literal_value))
        return tuple(values), tuple(races)


@dataclass(frozen=True, slots=True)
class _ValueRoute:
    kind: str
    source_path: tuple[int, ...] | None = None


def _starts_with(path: tuple[int, ...], prefix: tuple[int, ...]) -> bool:
    return path[: len(prefix)] == prefix


def _operand_value_route(
    action: tp.FactorAction,
    source: str,
    suffix: tuple[int, ...],
) -> _ValueRoute:
    if source == "SELECTED":
        return _ValueRoute("SELECTED", (*action.path, *suffix))
    if suffix:
        return _ValueRoute("UNKNOWN")
    if source in {"SCOPE0", "SCOPE1", "LITERAL"}:
        return _ValueRoute(source)
    return _ValueRoute("UNKNOWN")


def _value_route(
    action: tp.FactorAction,
    output_path: tuple[int, ...],
) -> _ValueRoute:
    if action.schema == "DONE":
        return _ValueRoute("SAME", output_path)
    if not _starts_with(output_path, action.path):
        return _ValueRoute("SAME", output_path)
    relative = output_path[len(action.path) :]
    if not relative:
        return _ValueRoute("LOCAL")
    child, suffix = relative[0], relative[1:]
    if action.schema == "APPLY1" and child == 0:
        return _operand_value_route(action, action.left_source, suffix)
    if action.schema == "APPLY2" and child == 0:
        return _operand_value_route(action, action.left_source, suffix)
    if action.schema == "APPLY2" and child == 1:
        return _operand_value_route(action, action.right_source, suffix)
    # List-body value propagation belongs to a separate behavior adapter.
    return _ValueRoute("UNKNOWN")


def _factor_sites_from_structure(
    observation: structure.StructureObservation,
) -> tuple[tp.FactorActionSite, ...]:
    sites: list[tp.FactorActionSite] = []
    for item in observation.paths:
        semantic = (
            item.production,
            item.result_type,
            item.scope0_type,
            item.scope1_type,
            item.binder_marker,
        )
        if item.presence != structure.PRESENT or any(
            value in {None, structure.UNKNOWN} for value in semantic
        ):
            continue
        assert item.production is not None
        assert item.result_type is not None
        assert item.scope0_type is not None
        assert item.scope1_type is not None
        selected_scope_index = (
            int(item.production[-1])
            if item.production in {"variable-0", "variable-1"}
            else None
        )
        sites.append(
            tp.FactorActionSite(
                path=item.path,
                result_type_label=item.result_type,
                head_production=item.production,
                scope_binding_type_labels=(item.scope0_type, item.scope1_type),
                is_binder=item.binder_marker == "binder",
                selected_scope_index=selected_scope_index,
            )
        )
    return tuple(sites)


@dataclass(frozen=True, slots=True)
class FactorStateTransition:
    action: tp.FactorAction
    before_sha256: str
    behavior: FactorBehaviorPrediction
    structure_prediction: structure.StructurePrediction
    predicted: FactorStateObservation
    value_races: tuple[operator.OperatorRace, ...]
    supported: bool
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


def _world_checkpoint_sha256(
    behavior_sha256: str,
    structure_sha256: str,
    literals: tuple[int, ...],
) -> str:
    digest = sha256()
    digest.update(b"public-typed-factor-h2-world-v1\0")
    digest.update(behavior_sha256.encode("ascii"))
    digest.update(structure_sha256.encode("ascii"))
    digest.update(np.asarray(literals, dtype=np.int64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenFactorH2WorldModel:
    behavior: FactorBehaviorAdapter
    structure_successor: structure.FrozenStructureSuccessor
    literals: tuple[int, ...]
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        return _world_checkpoint_sha256(
            self.behavior.live_checkpoint_sha256,
            self.structure_successor.live_checkpoint_sha256,
            self.literals,
        )

    def candidate_actions(
        self,
        state: FactorStateObservation,
    ) -> tuple[tp.FactorAction, ...]:
        return tp.factor_actions_for_sites(
            state.action_sites,
            literals=self.literals,
        )

    def predict_transition(
        self,
        state: FactorStateObservation,
        action: tp.FactorAction,
        *,
        behavior: FactorBehaviorPrediction | None = None,
    ) -> FactorStateTransition:
        before = self.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("factor H=2 world model changed before prediction")
        predicted_behavior = (
            self.behavior.predict_root_many(state, (action,))[0]
            if behavior is None
            else behavior
        )
        if predicted_behavior.action != action:
            raise ValueError("cached behavior belongs to another action")
        structural = self.structure_successor.predict(state.structure, action)
        site_values, value_races = self.behavior.predict_state_values(
            state,
            action,
            structural.predicted,
            predicted_behavior,
        )
        predicted = _state_observation(
            program_sha256=None,
            signature=state.signature,
            cases=state.cases,
            observed_structure=structural.predicted,
            sites=site_values,
            root_scope_rows=state.root_scope_rows,
        )
        supported = structural.supported and all(
            output.supported for output in predicted.root_outputs
        )
        after = self.live_checkpoint_sha256
        if after != before:
            raise RuntimeError("frozen factor H=2 world model changed in prediction")
        return FactorStateTransition(
            action=action,
            before_sha256=state.sha256,
            behavior=predicted_behavior,
            structure_prediction=structural,
            predicted=predicted,
            value_races=value_races,
            supported=supported,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
        )


@dataclass(frozen=True, slots=True)
class FactorH2Sequence:
    first: tp.FactorAction
    second: tp.FactorAction

    @property
    def edit_count(self) -> int:
        return int(self.first.schema != "DONE") + int(self.second.schema != "DONE")

    @property
    def record(self) -> dict[str, object]:
        return {
            "first": _action_record(self.first),
            "second": _action_record(self.second),
        }


@dataclass(frozen=True, slots=True)
class FactorH2Prediction:
    first_candidate_cell: int
    sequence: FactorH2Sequence
    first_transition: FactorStateTransition
    second_behavior: FactorBehaviorPrediction

    @property
    def predicted_final_outputs(
        self,
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        return self.second_behavior.root_outputs


@dataclass(frozen=True, slots=True)
class FactorSequenceValueFeatures:
    cells: np.ndarray
    exact_case_counts: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class FixedFactorSequenceValueProjection:
    """Fixed typed-output synapses over unique two-ply sequence cells."""

    sequences: tuple[FactorH2Sequence, ...]
    desired: tuple[output_cells.TypedOutputEncoding, ...]
    weights_pa: np.ndarray
    bias_pa: np.ndarray
    stop_current_pa: float
    sha256: str

    @classmethod
    def build(
        cls,
        sequences: Sequence[FactorH2Sequence],
        desired: Sequence[output_cells.TypedOutputEncoding],
    ) -> FixedFactorSequenceValueProjection:
        bank = tuple(sequences)
        targets = tuple(desired)
        records = tuple(_canonical_json(item.record) for item in bank)
        if not bank or len(set(records)) != len(records):
            raise ValueError("terminal sequence bank must be nonempty and unique")
        if not targets or not all(target.supported for target in targets):
            raise ValueError("terminal sequence projection needs supported targets")
        feature_count = output_cells.TOTAL_CELL_COUNT + 1
        coincidence_feature = output_cells.TOTAL_CELL_COUNT
        weights = np.zeros(
            (len(bank), len(targets), feature_count),
            dtype=np.float64,
        )
        weights[:, :, output_cells.TYPE_SLICE] = value_race._SUPPORTED_CASE_WEIGHT_PA
        weights[:, :, coincidence_feature] = value_race._EXACT_MATCH_WEIGHT_PA
        weights[:, :, value_race._unknown_cell_indices()] = (
            -value_race._UNKNOWN_INHIBITION_PA
        )
        bias = np.asarray(
            tuple(
                value_race._BASE_CURRENT_PA
                - _EDIT_COST_CURRENT_PA * sequence.edit_count
                for sequence in bank
            ),
            dtype=np.float64,
        )
        frozen_weights = np.ascontiguousarray(weights)
        frozen_bias = np.ascontiguousarray(bias)
        frozen_weights.setflags(write=False)
        frozen_bias.setflags(write=False)
        digest = sha256()
        digest.update(b"public-typed-factor-h2-sequence-value-v1\0")
        for record in records:
            digest.update(record)
        digest.update(_canonical_json(tuple(target.sha256 for target in targets)))
        digest.update(frozen_weights.tobytes())
        digest.update(frozen_bias.tobytes())
        digest.update(np.asarray((value_race._STOP_CURRENT_PA,)).tobytes())
        return cls(
            sequences=bank,
            desired=targets,
            weights_pa=frozen_weights,
            bias_pa=frozen_bias,
            stop_current_pa=value_race._STOP_CURRENT_PA,
            sha256=digest.hexdigest(),
        )

    def features(
        self,
        predictions: Sequence[Sequence[output_cells.TypedOutputEncoding]],
    ) -> FactorSequenceValueFeatures:
        rows = tuple(tuple(row) for row in predictions)
        if len(rows) != len(self.sequences) or any(
            len(row) != len(self.desired) for row in rows
        ):
            raise ValueError("terminal outputs must align with sequences and cases")
        cells = np.zeros(
            (
                len(self.sequences),
                len(self.desired),
                output_cells.TOTAL_CELL_COUNT + 1,
            ),
            dtype=np.float64,
        )
        detectors = tuple(
            output_cells.FixedTypedCoincidenceDetector.for_target(target)
            for target in self.desired
        )
        exact_counts: list[int] = []
        for sequence_cell, row in enumerate(rows):
            exact = 0
            for case, (predicted, detector) in enumerate(
                zip(row, detectors, strict=True)
            ):
                cells[
                    sequence_cell,
                    case,
                    : output_cells.TOTAL_CELL_COUNT,
                ] = predicted.cells
                if detector.detect(predicted).spiked:
                    cells[sequence_cell, case, output_cells.TOTAL_CELL_COUNT] = 1.0
                    exact += 1
            exact_counts.append(exact)
        frozen = np.ascontiguousarray(cells)
        frozen.setflags(write=False)
        return FactorSequenceValueFeatures(
            cells=frozen,
            exact_case_counts=tuple(exact_counts),
            sha256=sha256(frozen.tobytes()).hexdigest(),
        )

    def currents(self, features: FactorSequenceValueFeatures) -> np.ndarray:
        if features.cells.shape != self.weights_pa.shape:
            raise ValueError("terminal features do not match the fixed projection")
        candidates = self.bias_pa + np.einsum(
            "skf,skf->s",
            self.weights_pa,
            features.cells,
        )
        return np.ascontiguousarray(
            np.concatenate((candidates, np.asarray((self.stop_current_pa,))))
        )


def _leading_tier(currents: np.ndarray) -> RestingGlifTierBeamResult:
    selector = CompiledRestingGlifTierBeam.compile(
        candidate_count=len(currents),
        step_count=value_race._RACE_STEPS,
        min_current_pa=_VALUE_MIN_CURRENT_PA,
        max_current_pa=_VALUE_MAX_CURRENT_PA,
        inhibition_current_pa=_VALUE_INHIBITION_PA,
        beam_width=1,
    )
    return selector.select(currents, stop_cell=len(currents) - 1)


@dataclass(frozen=True, slots=True)
class FactorH2TerminalAttempt:
    expanded_first_cells: tuple[int, ...]
    sequence_count: int
    value_projection_sha256: str
    value_feature_sha256: str
    value_currents_pa: tuple[float, ...]
    value_tier: RestingGlifTierBeamResult | None
    exact_leading_sequence_cells: tuple[int, ...]
    final_structure_supported: tuple[bool, ...]
    structure_tier: RestingGlifTierBeamResult | None
    structure_leading_sequence_cells: tuple[int, ...]
    prompt_decision: prompt_factors.PromptBindingProjectionDecision | None
    prompt_sequence_currents_pa: tuple[float, ...]
    prompt_race: RestingGlifRaceResult | None
    selected_sequence_cell: int | None
    retryable: bool
    stop_reason: str | None


@dataclass(frozen=True, slots=True)
class _TerminalResolution:
    attempt: FactorH2TerminalAttempt
    selected_cell: int | None
    final_transitions: tuple[tuple[int, FactorStateTransition], ...]


def _prompt_sequence_race(
    predictions: tuple[FactorH2Prediction, ...],
    leading_cells: tuple[int, ...],
    prompt: str,
    signature: TypedSignature,
) -> tuple[
    prompt_factors.PromptBindingProjectionDecision,
    tuple[float, ...],
    RestingGlifRaceResult,
    int | None,
]:
    actions: list[tp.FactorAction] = []
    for cell in leading_cells:
        sequence = predictions[cell].sequence
        for action in (sequence.first, sequence.second):
            if action not in actions:
                actions.append(action)
    projection = prompt_factors.FixedPromptBindingProjection.build(tuple(actions))
    decision = projection.project(prompt, signature)
    action_index = {action: index for index, action in enumerate(actions)}
    currents: list[float] = []
    for cell in leading_cells:
        sequence = predictions[cell].sequence
        sequence_actions = (
            (sequence.first,)
            if sequence.first == sequence.second
            else (sequence.first, sequence.second)
        )
        currents.append(
            sum(
                decision.action_currents_pa[action_index[action]]
                for action in sequence_actions
            )
        )
    physical_currents = (*currents, _PROMPT_STOP_CURRENT_PA)
    race = _PROMPT_RACE.race_many(
        np.asarray((physical_currents,), dtype=np.float64)
    )[0]
    selected = (
        leading_cells[race.winner_cell]
        if race.simultaneous_spike_candidates == 1
        and race.winner_cell < len(leading_cells)
        else None
    )
    return decision, tuple(float(value) for value in physical_currents), race, selected


@dataclass(frozen=True, slots=True)
class FactorizedH2Decision:
    status: str
    stop_reason: str | None
    observation: FactorStateObservation
    desired: tuple[output_cells.TypedOutputEncoding, ...]
    candidate_actions: tuple[tp.FactorAction, ...]
    first_predictions: tuple[FactorBehaviorPrediction, ...]
    first_projection_sha256: str | None
    first_feature_sha256: str | None
    first_currents_pa: tuple[float, ...]
    first_tiers: StopBoundedTierResult | None
    expanded_first_cells: tuple[int, ...]
    predictions: tuple[FactorH2Prediction, ...]
    terminal_attempts: tuple[FactorH2TerminalAttempt, ...]
    selected_sequence_cell: int | None
    selected_sequence: FactorH2Sequence | None
    selected_first_transition: FactorStateTransition | None
    motor_decision: factor_motor.FactorMotorDecision | None
    unique_first_expansions: int
    unique_sequence_predictions: int
    cumulative_sequence_evaluations: int
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str


@dataclass(frozen=True, slots=True)
class FactorH2Actuation:
    program: tp.Program
    observation: FactorStateObservation
    actuator_calls: int
    structure_parity: bool
    action_site_parity: bool
    root_output_parity: bool


def _planner_checkpoint_sha256(world_sha256: str, beam_width: int) -> str:
    digest = sha256()
    digest.update(b"public-typed-factor-horizon2-int-v1\0")
    digest.update(world_sha256.encode("ascii"))
    digest.update(np.asarray((beam_width,), dtype=np.int64).tobytes())
    digest.update(
        np.asarray(
            (
                _VALUE_MIN_CURRENT_PA,
                _VALUE_MAX_CURRENT_PA,
                _VALUE_INHIBITION_PA,
                _STRUCTURE_SUPPORTED_CURRENT_PA,
                _STRUCTURE_STOP_CURRENT_PA,
                _PROMPT_STOP_CURRENT_PA,
                _EDIT_COST_CURRENT_PA,
            ),
            dtype=np.float64,
        ).tobytes()
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenFactorizedHorizon2Int:
    world: FrozenFactorH2WorldModel
    beam_width: int
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        return _planner_checkpoint_sha256(
            self.world.live_checkpoint_sha256,
            self.beam_width,
        )

    def observe(
        self,
        program: tp.Program,
        cases: Sequence[tuple[object, ...]],
    ) -> FactorStateObservation:
        return self.world.behavior.observe(program, cases)

    def _terminal_resolution(
        self,
        predictions: tuple[FactorH2Prediction, ...],
        expanded_first_cells: tuple[int, ...],
        desired: tuple[output_cells.TypedOutputEncoding, ...],
        prompt: str,
        final_cache: dict[FactorH2Sequence, FactorStateTransition],
    ) -> _TerminalResolution:
        sequences = tuple(item.sequence for item in predictions)
        projection = FixedFactorSequenceValueProjection.build(sequences, desired)
        features = projection.features(
            tuple(item.predicted_final_outputs for item in predictions)
        )
        currents = projection.currents(features)
        value_tier: RestingGlifTierBeamResult | None = None
        exact_cells: tuple[int, ...] = ()
        structure_flags: tuple[bool, ...] = ()
        structure_tier: RestingGlifTierBeamResult | None = None
        structure_leading: tuple[int, ...] = ()
        prompt_decision: prompt_factors.PromptBindingProjectionDecision | None = None
        prompt_currents: tuple[float, ...] = ()
        prompt_race: RestingGlifRaceResult | None = None
        final_transitions: list[tuple[int, FactorStateTransition]] = []

        def finish(
            *,
            selected: int | None,
            retryable: bool,
            reason: str | None,
        ) -> _TerminalResolution:
            return _TerminalResolution(
                attempt=FactorH2TerminalAttempt(
                    expanded_first_cells=expanded_first_cells,
                    sequence_count=len(predictions),
                    value_projection_sha256=projection.sha256,
                    value_feature_sha256=features.sha256,
                    value_currents_pa=tuple(float(value) for value in currents),
                    value_tier=value_tier,
                    exact_leading_sequence_cells=exact_cells,
                    final_structure_supported=structure_flags,
                    structure_tier=structure_tier,
                    structure_leading_sequence_cells=structure_leading,
                    prompt_decision=prompt_decision,
                    prompt_sequence_currents_pa=prompt_currents,
                    prompt_race=prompt_race,
                    selected_sequence_cell=selected,
                    retryable=retryable,
                    stop_reason=reason,
                ),
                selected_cell=selected,
                final_transitions=tuple(final_transitions),
            )

        try:
            value_tier = _leading_tier(currents)
        except RestingGlifRaceFailure as error:
            return finish(selected=None, retryable=True, reason=str(error))
        if value_tier.status == STOP_LEADING_TIER:
            return finish(
                selected=None,
                retryable=True,
                reason="physical terminal value STOP occupies the leading tier",
            )
        if value_tier.status != RETAINED or len(value_tier.rounds) != 1:
            raise RuntimeError("terminal value race did not produce one complete tier")
        leading = value_tier.retained_candidate_cells
        exact_cells = tuple(
            cell
            for cell in leading
            if features.exact_case_counts[cell] == len(desired)
        )
        if not exact_cells:
            return finish(
                selected=None,
                retryable=True,
                reason="leading terminal value tier lacks all-case coincidence",
            )

        for cell in exact_cells:
            item = predictions[cell]
            transition = final_cache.get(item.sequence)
            if transition is None:
                transition = self.world.predict_transition(
                    item.first_transition.predicted,
                    item.sequence.second,
                    behavior=item.second_behavior,
                )
                final_cache[item.sequence] = transition
            final_transitions.append((cell, transition))
        structure_flags = tuple(
            transition.supported for _, transition in final_transitions
        )
        structure_currents = np.asarray(
            (
                *tuple(
                    _STRUCTURE_SUPPORTED_CURRENT_PA if supported else 0.0
                    for supported in structure_flags
                ),
                _STRUCTURE_STOP_CURRENT_PA,
            ),
            dtype=np.float64,
        )
        try:
            structure_tier = _leading_tier(structure_currents)
        except RestingGlifRaceFailure as error:
            return finish(selected=None, retryable=True, reason=str(error))
        if structure_tier.status == STOP_LEADING_TIER:
            return finish(
                selected=None,
                retryable=True,
                reason="all exact terminal sequences produced structural UNKNOWN",
            )
        if structure_tier.status != RETAINED or len(structure_tier.rounds) != 1:
            raise RuntimeError("terminal structure race lost its complete tier")
        structure_leading = tuple(
            exact_cells[local_cell]
            for local_cell in structure_tier.retained_candidate_cells
        )
        if len(structure_leading) == 1:
            return finish(selected=structure_leading[0], retryable=False, reason=None)

        try:
            (
                prompt_decision,
                prompt_currents,
                prompt_race,
                selected,
            ) = _prompt_sequence_race(
                predictions,
                structure_leading,
                prompt,
                predictions[0].first_transition.predicted.signature,
            )
        except (RestingGlifRaceFailure, RuntimeError) as error:
            return finish(selected=None, retryable=False, reason=str(error))
        if selected is None:
            return finish(
                selected=None,
                retryable=False,
                reason="prompt evidence did not uniquely resolve the leading structure tier",
            )
        return finish(selected=selected, retryable=False, reason=None)

    def plan(
        self,
        observation: FactorStateObservation,
        prompt: str,
        desired_outputs: Sequence[object],
    ) -> FactorizedH2Decision:
        """Choose one first edit from genuine learned-cell second-ply outcomes."""

        if not isinstance(observation, FactorStateObservation):
            raise TypeError("factorized H=2 planning requires a FactorStateObservation")
        if type(prompt) is not str or not prompt:
            raise ValueError("factorized H=2 planning requires a nonempty prompt")
        before = self.live_checkpoint_sha256
        if before != self.checkpoint_sha256:
            raise RuntimeError("factorized H=2 checkpoint changed before planning")
        supplied_desired = tuple(desired_outputs)
        if len(supplied_desired) != len(observation.cases):
            raise ValueError("desired outputs must align with the observed cases")
        desired = tuple(_encode("Int", value) for value in supplied_desired)
        if not all(item.supported for item in desired):
            raise ValueError("desired Int outputs must inhabit the physical bank")
        candidate_actions = self.world.candidate_actions(observation)

        def done_decision() -> FactorizedH2Decision:
            motor_projection = factor_motor.FixedFactorMotorProjection.build(
                candidate_actions
            )
            spikes = np.zeros(len(candidate_actions), dtype=np.float64)
            spikes[candidate_actions.index(_DONE)] = 1.0
            motor = motor_projection.resolve(spikes)
            after = self.live_checkpoint_sha256
            return FactorizedH2Decision(
                status=STATUS_DONE,
                stop_reason=None,
                observation=observation,
                desired=desired,
                candidate_actions=candidate_actions,
                first_predictions=(),
                first_projection_sha256=None,
                first_feature_sha256=None,
                first_currents_pa=(),
                first_tiers=None,
                expanded_first_cells=(),
                predictions=(),
                terminal_attempts=(),
                selected_sequence_cell=None,
                selected_sequence=FactorH2Sequence(_DONE, _DONE),
                selected_first_transition=None,
                motor_decision=motor,
                unique_first_expansions=0,
                unique_sequence_predictions=0,
                cumulative_sequence_evaluations=0,
                checkpoint_before_sha256=before,
                checkpoint_after_sha256=after,
            )

        detectors = tuple(
            output_cells.FixedTypedCoincidenceDetector.for_target(target)
            for target in desired
        )
        if all(
            detector.detect(current).spiked
            for detector, current in zip(
                detectors,
                observation.root_outputs,
                strict=True,
            )
        ):
            return done_decision()

        first_predictions = self.world.behavior.predict_root_many(
            observation,
            candidate_actions,
        )
        first_projection = value_race.FixedTypedFactorValueProjection.build(
            candidate_actions,
            desired,
        )
        first_features = first_projection.features(
            tuple(item.root_outputs for item in first_predictions)
        )
        first_currents = first_projection.tier_currents(first_features)
        first_selector = CompiledStopBoundedRestingGlifTiers.compile(
            candidate_count=len(first_currents),
            step_count=value_race._RACE_STEPS,
            min_current_pa=_VALUE_MIN_CURRENT_PA,
            max_current_pa=_VALUE_MAX_CURRENT_PA,
            inhibition_current_pa=_VALUE_INHIBITION_PA,
        )
        first_tiers = first_selector.enumerate(
            first_currents,
            stop_cell=len(candidate_actions),
        )
        tier_cells = tuple(
            tuple(cell for cell in round_.tier_cells if cell != len(candidate_actions))
            for round_ in first_tiers.rounds
        )

        expanded_cells: list[int] = []
        expansion_cache: dict[int, tuple[FactorH2Prediction, ...]] = {}
        final_cache: dict[FactorH2Sequence, FactorStateTransition] = {}
        attempts: list[FactorH2TerminalAttempt] = []
        predictions: tuple[FactorH2Prediction, ...] = ()
        selected_cell: int | None = None
        stop_reason: str | None = None
        cumulative_evaluations = 0
        tier_cursor = 0

        while tier_cursor < len(tier_cells):
            while tier_cursor < len(tier_cells) and (
                len(expanded_cells) < self.beam_width or not expanded_cells
            ):
                expanded_cells.extend(tier_cells[tier_cursor])
                tier_cursor += 1
            if not expanded_cells:
                break
            for first_cell in expanded_cells:
                if first_cell in expansion_cache:
                    continue
                first_action = candidate_actions[first_cell]
                first_transition = self.world.predict_transition(
                    observation,
                    first_action,
                    behavior=first_predictions[first_cell],
                )
                second_actions = (
                    (_DONE,)
                    if first_action.schema == "DONE"
                    else self.world.candidate_actions(first_transition.predicted)
                )
                second_predictions = self.world.behavior.predict_root_many(
                    first_transition.predicted,
                    second_actions,
                )
                expansion_cache[first_cell] = tuple(
                    FactorH2Prediction(
                        first_candidate_cell=first_cell,
                        sequence=FactorH2Sequence(first_action, second_action),
                        first_transition=first_transition,
                        second_behavior=second_prediction,
                    )
                    for second_action, second_prediction in zip(
                        second_actions,
                        second_predictions,
                        strict=True,
                    )
                )
            predictions = tuple(
                item
                for first_cell in expanded_cells
                for item in expansion_cache[first_cell]
            )
            cumulative_evaluations += len(predictions)
            resolution = self._terminal_resolution(
                predictions,
                tuple(expanded_cells),
                desired,
                prompt,
                final_cache,
            )
            attempts.append(resolution.attempt)
            if resolution.selected_cell is not None:
                selected_cell = resolution.selected_cell
                break
            stop_reason = resolution.attempt.stop_reason
            if not resolution.attempt.retryable or tier_cursor >= len(tier_cells):
                break
            expanded_cells.extend(tier_cells[tier_cursor])
            tier_cursor += 1

        selected_sequence = (
            None if selected_cell is None else predictions[selected_cell].sequence
        )
        motor: factor_motor.FactorMotorDecision | None = None
        selected_first_transition: FactorStateTransition | None = None
        status = STATUS_STOP_TERMINAL_VALUE
        if selected_cell is not None and selected_sequence is not None:
            first_cell = predictions[selected_cell].first_candidate_cell
            selected_first_transition = predictions[selected_cell].first_transition
            motor_projection = factor_motor.FixedFactorMotorProjection.build(
                candidate_actions
            )
            spikes = np.zeros(len(candidate_actions), dtype=np.float64)
            spikes[first_cell] = 1.0
            motor = motor_projection.resolve(spikes)
            if motor.source_action != selected_sequence.first:
                raise RuntimeError("terminal H=2 winner and factor motor lost parity")
            status = (
                STATUS_DONE
                if selected_sequence.first.schema == "DONE"
                else STATUS_SELECTED
            )
            stop_reason = None
        elif attempts and attempts[-1].structure_tier is not None:
            status = STATUS_STOP_PROMPT
            if attempts[-1].retryable:
                status = STATUS_STOP_TERMINAL_STRUCTURE
        elif not expanded_cells:
            status = STATUS_STOP_FIRST_FRONTIER
            stop_reason = "physical first-ply STOP admitted no candidates"

        after = self.live_checkpoint_sha256
        if after != before or after != self.checkpoint_sha256:
            raise RuntimeError("frozen factorized H=2 planner changed in inference")
        return FactorizedH2Decision(
            status=status,
            stop_reason=stop_reason,
            observation=observation,
            desired=desired,
            candidate_actions=candidate_actions,
            first_predictions=first_predictions,
            first_projection_sha256=first_projection.sha256,
            first_feature_sha256=first_features.sha256,
            first_currents_pa=tuple(float(value) for value in first_currents),
            first_tiers=first_tiers,
            expanded_first_cells=tuple(expanded_cells),
            predictions=predictions,
            terminal_attempts=tuple(attempts),
            selected_sequence_cell=selected_cell,
            selected_sequence=selected_sequence,
            selected_first_transition=selected_first_transition,
            motor_decision=motor,
            unique_first_expansions=len(expansion_cache),
            unique_sequence_predictions=len(predictions),
            cumulative_sequence_evaluations=cumulative_evaluations,
            checkpoint_before_sha256=before,
            checkpoint_after_sha256=after,
        )

    def actuate_and_reobserve(
        self,
        incumbent: tp.Program,
        decision: FactorizedH2Decision,
    ) -> FactorH2Actuation:
        """Execute only the selected first edit, then reobserve for replanning."""

        if incumbent.sha256 != decision.observation.program_sha256:
            raise ValueError("actuator incumbent differs from the planned observation")
        if decision.status == STATUS_DONE:
            return FactorH2Actuation(
                program=incumbent,
                observation=decision.observation,
                actuator_calls=0,
                structure_parity=True,
                action_site_parity=True,
                root_output_parity=True,
            )
        if (
            decision.status != STATUS_SELECTED
            or decision.motor_decision is None
            or decision.selected_first_transition is None
        ):
            raise ValueError("only a selected H=2 decision may reach the actuator")
        edited = factor_motor.actuate_selected_factor_action(
            incumbent,
            decision.motor_decision,
        )
        observed = self.observe(edited, decision.observation.cases)
        predicted = decision.selected_first_transition.predicted
        structure_parity = observed.structure.paths == predicted.structure.paths
        action_site_parity = observed.action_sites == predicted.action_sites
        root_output_parity = tuple(
            item.sha256 for item in observed.root_outputs
        ) == tuple(item.sha256 for item in predicted.root_outputs)
        if not (structure_parity and action_site_parity and root_output_parity):
            raise RuntimeError("first learned transition differs from real reobservation")
        return FactorH2Actuation(
            program=edited,
            observation=observed,
            actuator_calls=1,
            structure_parity=structure_parity,
            action_site_parity=action_site_parity,
            root_output_parity=root_output_parity,
        )


def build_public_factorized_horizon2_int(
    *,
    literals: Sequence[int] = tuple(range(-4, 5)),
) -> FrozenFactorizedHorizon2Int:
    """Build the target-free reusable mechanism without running a task cohort."""

    supplied = tuple(literals)
    tp.factor_actions_for_sites((), literals=supplied)
    literal_bank = tuple(sorted(supplied))
    operator_model, _ = operator.train_public_typed_operator_successor()
    structure_model, _ = structure.train_public_structure_successor()
    behavior_checkpoint = _behavior_checkpoint_sha256(
        operator_model.checkpoint_sha256
    )
    behavior = FrozenIntFactorBehaviorAdapter(
        operator_successor=operator_model,
        checkpoint_sha256=behavior_checkpoint,
    )
    world_checkpoint = _world_checkpoint_sha256(
        behavior.checkpoint_sha256,
        structure_model.checkpoint_sha256,
        literal_bank,
    )
    world = FrozenFactorH2WorldModel(
        behavior=behavior,
        structure_successor=structure_model,
        literals=literal_bank,
        checkpoint_sha256=world_checkpoint,
    )
    return FrozenFactorizedHorizon2Int(
        world=world,
        beam_width=BEAM_WIDTH,
        checkpoint_sha256=_planner_checkpoint_sha256(world_checkpoint, BEAM_WIDTH),
    )


__all__ = [
    "BEAM_WIDTH",
    "STATUS_DONE",
    "STATUS_SELECTED",
    "FactorBehaviorAdapter",
    "FactorBehaviorPrediction",
    "FactorH2Actuation",
    "FactorH2Prediction",
    "FactorH2Sequence",
    "FactorSiteValue",
    "FactorStateObservation",
    "FactorStateTransition",
    "FactorizedH2Decision",
    "FixedFactorSequenceValueProjection",
    "FrozenFactorH2WorldModel",
    "FrozenFactorizedHorizon2Int",
    "FrozenIntFactorBehaviorAdapter",
    "build_public_factorized_horizon2_int",
]
