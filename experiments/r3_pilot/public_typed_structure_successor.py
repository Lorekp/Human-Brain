"""Local physical successor for the public typed-program structure.

Target-free compiler observations teach two frozen layers: admissible generic
factor patterns and local rules for nodes created by those factors.  Existing
and SELECTED subtrees travel through fixed same-label relay synapses, so an
edit learned on one incumbent composes with unseen surrounding structure.
Prediction races only structural cells; it never constructs, edits, or
compiles a candidate program.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
for search_path in (REPO_ROOT, SRC):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from experiments.r3_pilot import (  # noqa: E402
    public_typed_horizon2_successor_planner as physical,
)
from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.resting_glif_race import (  # noqa: E402
    CompiledRestingGlifRace,
    RestingGlifRaceFailure,
)
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

CellRace = physical.CellRace

ABSENT = "ABSENT"
PRESENT = "PRESENT"
UNKNOWN = "UNKNOWN"

MAX_PATH_DEPTH = 3
MAX_CHILDREN = 3


def _path_universe() -> tuple[tuple[int, ...], ...]:
    paths: list[tuple[int, ...]] = []

    def add(path: tuple[int, ...]) -> None:
        paths.append(path)
        if len(path) < MAX_PATH_DEPTH:
            for child in range(MAX_CHILDREN):
                add((*path, child))

    add(())
    return tuple(paths)


PATH_UNIVERSE = _path_universe()
PRESENCE_LABELS = (ABSENT, PRESENT, UNKNOWN)
PRODUCTION_LABELS = (
    "add",
    "subtract",
    "absolute",
    "less-than",
    "if",
    "map",
    "sum",
    "bind",
    "variable-0",
    "variable-1",
    "integer",
    UNKNOWN,
)
RESULT_TYPE_LABELS = ("Int", "Bool", "List[Int]", "internal-function", UNKNOWN)
SCOPE_TYPE_LABELS = (
    "unbound",
    "Int",
    "Bool",
    "List[Int]",
    "internal-function",
    UNKNOWN,
)
BINDER_MARKER_LABELS = ("not-binder", "binder", UNKNOWN)
PATH_ROLE_LABELS = ("root", "child-0", "child-1", "child-2", UNKNOWN)

_FIELD_SPECS = (
    ("presence", PRESENCE_LABELS),
    ("production", PRODUCTION_LABELS),
    ("result_type", RESULT_TYPE_LABELS),
    ("scope0_type", SCOPE_TYPE_LABELS),
    ("scope1_type", SCOPE_TYPE_LABELS),
    ("binder_marker", BINDER_MARKER_LABELS),
    ("path_role", PATH_ROLE_LABELS),
)
_SEMANTIC_FIELD_SPECS = _FIELD_SPECS[1:]

_SCHEMA_LABELS = ("DONE", "APPLY1", "APPLY2", "MAP_BIND")
_OPERATOR_LABELS = ("NONE", "absolute", "add", "subtract", "less-than", "map")
_SOURCE_LABELS = ("NONE", "SELECTED", "SCOPE0", "SCOPE1", "BOUND0", "LITERAL")
_LITERAL_LABELS = ("NONE", "PRESENT")
_DONE_PATH_LABELS = ("ROOT", "NONROOT", "EDIT")
_READINESS_LABELS = ("READY", UNKNOWN)

_COPY_SAME = "COPY_SAME"
_COPY_SELECTED = "COPY_SELECTED"
_CREATE_OPERATOR = "CREATE_OPERATOR"
_CREATE_BINDER = "CREATE_BINDER"
_CREATE_MAP_BOUND = "CREATE_MAP_BOUND"
_CREATE_BOUND = "CREATE_BOUND"
_CREATE_SCOPE0 = "CREATE_SCOPE0"
_CREATE_SCOPE1 = "CREATE_SCOPE1"
_CREATE_LITERAL = "CREATE_LITERAL"
_ROUTE_ABSENT = "ROUTE_ABSENT"
_ROUTE_UNKNOWN = "ROUTE_UNKNOWN"
_ROUTE_LABELS = (
    _COPY_SAME,
    _COPY_SELECTED,
    _CREATE_OPERATOR,
    _CREATE_BINDER,
    _CREATE_MAP_BOUND,
    _CREATE_BOUND,
    _CREATE_SCOPE0,
    _CREATE_SCOPE1,
    _CREATE_LITERAL,
    _ROUTE_ABSENT,
    _ROUTE_UNKNOWN,
)

_RELAY_SOURCE = "RELAY_SOURCE"
_RELAY_SCOPE0 = "RELAY_SCOPE0"
_RELAY_SCOPE1 = "RELAY_SCOPE1"
_RELAY_TARGET_ROLE = "RELAY_TARGET_ROLE"
_LEARNED_RULE = "LEARNED_RULE"

_FULL_MATCH_CURRENT_PA = 5_000.0
_CONTACT_GAP_PA = 5_000.0
_UNKNOWN_CURRENT_PA = 1_200.0
_OUTPUT_CURRENT_PA = 5_000.0
_RACE = CompiledRestingGlifRace.compile(step_count=physical._RACE_STEPS)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _path_role(path: tuple[int, ...]) -> str:
    return "root" if not path else f"child-{path[-1]}"


@dataclass(frozen=True, slots=True)
class PathStructure:
    path: tuple[int, ...]
    presence: str
    production: str | None
    result_type: str | None
    scope0_type: str | None
    scope1_type: str | None
    binder_marker: str | None
    path_role: str | None


@dataclass(frozen=True, slots=True)
class StructureObservation:
    program_sha256: str | None
    paths: tuple[PathStructure, ...]
    sha256: str

    def site(self, path: tuple[int, ...]) -> PathStructure:
        try:
            return self.paths[PATH_UNIVERSE.index(path)]
        except ValueError as error:
            raise ValueError(f"path {path!r} is outside the frozen universe") from error


@dataclass(frozen=True, slots=True)
class FieldRace:
    path: tuple[int, ...]
    field: str
    race: CellRace


@dataclass(frozen=True, slots=True)
class StructurePrediction:
    predicted: StructureObservation
    supported: bool
    transition_race: CellRace
    rule_races: tuple[FieldRace, ...]
    field_races: tuple[FieldRace, ...]
    feature_sha256: str
    checkpoint_before_sha256: str
    checkpoint_after_sha256: str

    @property
    def races(self) -> tuple[CellRace, ...]:
        return (
            self.transition_race,
            *(item.race for item in self.rule_races),
            *(item.race for item in self.field_races),
        )


@dataclass(frozen=True, slots=True)
class StructureTrainingSummary:
    observations: int
    transition_rows: int
    local_rule_rows: int
    path_cells: int
    action_feature_cells: int
    local_feature_cells: int
    recognition_synapses: int
    projection_synapses: int
    training_digest: str
    checkpoint_sha256: str
    host_execution_scope: str


@dataclass(frozen=True, slots=True)
class _Route:
    kind: str
    source_path: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class _ObservedTransition:
    before: StructureObservation
    action: tp.FactorAction
    after: StructureObservation


@dataclass(frozen=True, slots=True)
class FrozenFieldRuleBank:
    field: str
    labels: tuple[str, ...]
    contacts: np.ndarray
    contact_counts: np.ndarray
    projection: np.ndarray


def _observation(
    program_sha256: str | None,
    paths: Sequence[PathStructure],
) -> StructureObservation:
    path_tuple = tuple(paths)
    record = {
        "program_sha256": program_sha256,
        "paths": tuple(asdict(item) for item in path_tuple),
    }
    return StructureObservation(
        program_sha256=program_sha256,
        paths=path_tuple,
        sha256=sha256(_canonical_json(record)).hexdigest(),
    )


def unknown_structure_observation(
    program_sha256: str | None = None,
) -> StructureObservation:
    return _observation(
        program_sha256,
        tuple(
            PathStructure(
                path,
                UNKNOWN,
                UNKNOWN,
                UNKNOWN,
                UNKNOWN,
                UNKNOWN,
                UNKNOWN,
                UNKNOWN,
            )
            for path in PATH_UNIVERSE
        ),
    )


def _known_or_unknown(value: str, labels: tuple[str, ...]) -> str:
    return value if value in labels[:-1] else UNKNOWN


def observe_program_structure(program: tp.Program) -> StructureObservation:
    """Observe only an incumbent through public path/type metadata."""

    if not isinstance(program, tp.Program):
        raise TypeError("structure observation requires a typed Program")
    sites = tp.program_edit_sites(program)
    if any(site.path not in PATH_UNIVERSE for site in sites):
        return unknown_structure_observation(program.sha256)
    by_path = {site.path: site for site in sites}
    records: list[PathStructure] = []
    for path in PATH_UNIVERSE:
        site = by_path.get(path)
        if site is None:
            records.append(
                PathStructure(path, ABSENT, None, None, None, None, None, None)
            )
            continue
        records.append(
            PathStructure(
                path=path,
                presence=PRESENT,
                production=_known_or_unknown(site.head_production, PRODUCTION_LABELS),
                result_type=_known_or_unknown(
                    site.result_type_label,
                    RESULT_TYPE_LABELS,
                ),
                scope0_type=_known_or_unknown(
                    site.scope_binding_type_labels[0],
                    SCOPE_TYPE_LABELS,
                ),
                scope1_type=_known_or_unknown(
                    site.scope_binding_type_labels[1],
                    SCOPE_TYPE_LABELS,
                ),
                binder_marker="binder" if site.is_binder else "not-binder",
                path_role=_path_role(path),
            )
        )
    return _observation(program.sha256, records)


def _validate_observation(observation: StructureObservation) -> None:
    if not isinstance(observation, StructureObservation):
        raise TypeError("successor input must be a StructureObservation")
    if tuple(item.path for item in observation.paths) != PATH_UNIVERSE:
        raise ValueError("structure observation does not cover the frozen path universe")
    for item in observation.paths:
        if item.presence not in PRESENCE_LABELS:
            raise ValueError("structure observation has an unknown presence label")
        values = tuple(getattr(item, field) for field, _ in _SEMANTIC_FIELD_SPECS)
        if item.presence == ABSENT:
            if any(value is not None for value in values):
                raise ValueError("absent paths cannot carry semantic fields")
            continue
        for (field, labels), value in zip(
            _SEMANTIC_FIELD_SPECS,
            values,
            strict=True,
        ):
            if value not in labels:
                raise ValueError(f"structure observation has an invalid {field} label")
        if item.presence == UNKNOWN and any(value != UNKNOWN for value in values):
            raise ValueError("physical UNKNOWN paths must propagate through every field")
    if observation.sha256 != _observation(
        observation.program_sha256,
        observation.paths,
    ).sha256:
        raise ValueError("structure observation digest does not match its fields")


def _starts_with(path: tuple[int, ...], prefix: tuple[int, ...]) -> bool:
    return path[: len(prefix)] == prefix


def _inside_binder(
    observation: StructureObservation,
    path: tuple[int, ...],
) -> bool:
    return any(
        item.presence == PRESENT
        and item.binder_marker == "binder"
        and len(path) > len(item.path)
        and _starts_with(path, item.path)
        for item in observation.paths
    )


def _selected_would_overflow(
    observation: StructureObservation,
    action_path: tuple[int, ...],
) -> bool:
    return any(
        item.presence == PRESENT
        and _starts_with(item.path, action_path)
        and len(item.path) + 1 > MAX_PATH_DEPTH
        for item in observation.paths
    )


def _site_ready(
    observation: StructureObservation,
    action: tp.FactorAction,
) -> bool:
    if action.path not in PATH_UNIVERSE:
        return False
    site = observation.site(action.path)
    semantic = tuple(getattr(site, field) for field, _ in _SEMANTIC_FIELD_SPECS)
    if site.presence != PRESENT or any(value == UNKNOWN for value in semantic):
        return False
    if action.schema == "DONE":
        return action.path == ()
    required_depth = 2 if action.schema == "MAP_BIND" else 1
    if len(action.path) + required_depth > MAX_PATH_DEPTH:
        return False
    sources = (action.left_source, action.right_source)
    for source in sources:
        if source == "SCOPE0" and site.scope0_type in {"unbound", UNKNOWN}:
            return False
        if source == "SCOPE1" and site.scope1_type in {"unbound", UNKNOWN}:
            return False
        if (
            source == "BOUND0"
            and action.schema != "MAP_BIND"
            and not _inside_binder(observation, action.path)
        ):
            return False
    if "SELECTED" in sources and _selected_would_overflow(
        observation,
        action.path,
    ):
        return False
    return True


def _literal_label(literal: int | None) -> str:
    return "NONE" if literal is None else "PRESENT"


def _done_path_label(action: tp.FactorAction) -> str:
    if action.schema != "DONE":
        return "EDIT"
    return "ROOT" if action.path == () else "NONROOT"


def _action_feature_labels() -> tuple[str, ...]:
    labels: list[str] = []
    domains = (
        ("done-path", _DONE_PATH_LABELS),
        ("schema", _SCHEMA_LABELS),
        ("operator", _OPERATOR_LABELS),
        ("left", _SOURCE_LABELS),
        ("right", _SOURCE_LABELS),
        ("literal", _LITERAL_LABELS),
        ("readiness", _READINESS_LABELS),
    )
    for factor, values in domains:
        labels.extend(f"action:{factor}:{value}" for value in values)
    return tuple(labels)


_ACTION_FEATURE_LABELS = _action_feature_labels()
_ACTION_FEATURE_INDEX = {
    label: index for index, label in enumerate(_ACTION_FEATURE_LABELS)
}


def _encode_action_features(
    action: tp.FactorAction,
    *,
    ready: bool,
) -> np.ndarray:
    if type(action) is not tp.FactorAction:
        raise TypeError("successor action must be a FactorAction")
    features = np.zeros(len(_ACTION_FEATURE_LABELS), dtype=np.float64)
    active = (
        f"action:done-path:{_done_path_label(action)}",
        f"action:schema:{action.schema}",
        f"action:operator:{action.operator}",
        f"action:left:{action.left_source}",
        f"action:right:{action.right_source}",
        f"action:literal:{_literal_label(action.literal)}",
        f"action:readiness:{'READY' if ready else UNKNOWN}",
    )
    for label in active:
        features[_ACTION_FEATURE_INDEX[label]] = 1.0
    return np.ascontiguousarray(features)


def _local_feature_count() -> int:
    return (
        len(_ACTION_FEATURE_LABELS)
        + len(_ROUTE_LABELS)
        + len(RESULT_TYPE_LABELS)
    )


def _one_hot(value: str, labels: tuple[str, ...]) -> np.ndarray:
    cells = np.zeros(len(labels), dtype=np.float64)
    cells[labels.index(value if value in labels else UNKNOWN)] = 1.0
    return cells


def _encode_local_features(
    action: tp.FactorAction,
    route: _Route,
    site: PathStructure,
) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate(
            (
                _encode_action_features(action, ready=True),
                _one_hot(route.kind, _ROUTE_LABELS),
                _one_hot(str(site.result_type), RESULT_TYPE_LABELS),
            )
        )
    )


def _operand_route(
    action: tp.FactorAction,
    source: str,
    suffix: tuple[int, ...],
) -> _Route:
    if source == "SELECTED":
        return _Route(_COPY_SELECTED, (*action.path, *suffix))
    if suffix:
        return _Route(_ROUTE_ABSENT)
    return _Route(
        {
            "BOUND0": _CREATE_BOUND,
            "SCOPE0": _CREATE_SCOPE0,
            "SCOPE1": _CREATE_SCOPE1,
            "LITERAL": _CREATE_LITERAL,
        }.get(source, _ROUTE_UNKNOWN)
    )


def _route_for(
    action: tp.FactorAction,
    output_path: tuple[int, ...],
) -> _Route:
    if action.schema == "DONE":
        return _Route(_COPY_SAME, output_path)
    if not _starts_with(output_path, action.path):
        return _Route(_COPY_SAME, output_path)
    relative = output_path[len(action.path) :]
    if not relative:
        return _Route(_CREATE_OPERATOR)
    child, suffix = relative[0], relative[1:]
    if action.schema == "APPLY1":
        if child != 0:
            return _Route(_ROUTE_ABSENT)
        return _operand_route(action, action.left_source, suffix)
    if action.schema == "APPLY2":
        if child == 0:
            return _operand_route(action, action.left_source, suffix)
        if child == 1:
            return _operand_route(action, action.right_source, suffix)
        return _Route(_ROUTE_ABSENT)
    if action.schema == "MAP_BIND":
        if child == 0:
            if not suffix:
                return _Route(_CREATE_BINDER)
            if suffix == (0,):
                return _Route(_CREATE_MAP_BOUND)
            return _Route(_ROUTE_ABSENT)
        if child == 1:
            return _operand_route(action, action.right_source, suffix)
        return _Route(_ROUTE_ABSENT)
    return _Route(_ROUTE_UNKNOWN)


def _field_mode(route: _Route, field: str) -> str:
    if field == "presence":
        if route.kind in {_COPY_SAME, _COPY_SELECTED}:
            return _RELAY_SOURCE
        return _LEARNED_RULE
    if field == "path_role":
        return _RELAY_TARGET_ROLE
    if route.kind in {_COPY_SAME, _COPY_SELECTED}:
        return _RELAY_SOURCE
    if field == "scope0_type":
        return _LEARNED_RULE if route.kind == _CREATE_MAP_BOUND else _RELAY_SCOPE0
    if field == "scope1_type":
        return _RELAY_SCOPE0 if route.kind == _CREATE_MAP_BOUND else _RELAY_SCOPE1
    if field == "result_type":
        if route.kind in {_CREATE_BOUND, _CREATE_SCOPE0}:
            return _RELAY_SCOPE0
        if route.kind == _CREATE_SCOPE1:
            return _RELAY_SCOPE1
    return _LEARNED_RULE


def _relay_value(
    observation: StructureObservation,
    site: PathStructure,
    output_path: tuple[int, ...],
    route: _Route,
    field: str,
    mode: str,
) -> str | None:
    if mode == _RELAY_SOURCE:
        if route.source_path not in PATH_UNIVERSE:
            return UNKNOWN
        return getattr(observation.site(route.source_path), field)
    if mode == _RELAY_SCOPE0:
        return site.scope0_type
    if mode == _RELAY_SCOPE1:
        return site.scope1_type
    if mode == _RELAY_TARGET_ROLE:
        return _path_role(output_path)
    raise AssertionError("learned local rule cannot use a relay value")


def _done() -> tp.FactorAction:
    return tp.FactorAction((), "DONE", "NONE", "NONE", "NONE", None)


def _public_synthetic_incumbents() -> tuple[tuple[tp.Program, tp.FactorAction], ...]:
    int_signature = TypedSignature(
        "public_int_structure",
        (ProgramParameter("value", ProgramType.Int()),),
        ProgramType.Int(),
    )
    bool_signature = TypedSignature(
        "public_bool_structure",
        (ProgramParameter("value", ProgramType.Int()),),
        ProgramType.Bool(),
    )
    list_int = ProgramType.List(ProgramType.Int())
    list_signature = TypedSignature(
        "public_list_structure",
        (ProgramParameter("items", list_int),),
        list_int,
    )
    two_int_signature = TypedSignature(
        "public_two_int_structure",
        (
            ProgramParameter("left", ProgramType.Int()),
            ProgramParameter("right", ProgramType.Int()),
        ),
        ProgramType.Int(),
    )
    int_identity = tp.Program(("variable-0",), int_signature)
    two_int_identity = tp.Program(("variable-0",), two_int_signature)
    bool_less_than_zero = tp.Program(
        ("less-than", "variable-0", "integer", "+", "0", "end"),
        bool_signature,
    )
    list_identity = tp.Program(("variable-0",), list_signature)
    map_identity_action = tp.FactorAction(
        (), "MAP_BIND", "map", "BOUND0", "SELECTED", None
    )
    map_identity = tp.apply_factor_action(list_identity, map_identity_action)
    return (
        (int_identity, _done()),
        (
            int_identity,
            tp.FactorAction((), "APPLY1", "absolute", "SELECTED", "NONE", None),
        ),
        (
            int_identity,
            tp.FactorAction((), "APPLY2", "add", "SELECTED", "LITERAL", 2),
        ),
        (
            int_identity,
            tp.FactorAction(
                (), "APPLY2", "subtract", "SELECTED", "LITERAL", 2
            ),
        ),
        (bool_less_than_zero, _done()),
        (
            bool_less_than_zero,
            tp.FactorAction((), "APPLY2", "less-than", "SCOPE0", "LITERAL", 2),
        ),
        (
            bool_less_than_zero,
            tp.FactorAction((), "APPLY2", "less-than", "LITERAL", "SCOPE0", 2),
        ),
        (
            two_int_identity,
            tp.FactorAction((), "APPLY2", "subtract", "SCOPE1", "SCOPE0", None),
        ),
        (
            two_int_identity,
            tp.FactorAction((), "APPLY2", "subtract", "SCOPE0", "SCOPE1", None),
        ),
        (list_identity, _done()),
        (list_identity, map_identity_action),
        (map_identity, _done()),
        (
            map_identity,
            tp.FactorAction(
                (0, 0), "APPLY1", "absolute", "BOUND0", "NONE", None
            ),
        ),
        (
            map_identity,
            tp.FactorAction(
                (0, 0), "APPLY2", "add", "BOUND0", "LITERAL", 2
            ),
        ),
    )


def _collect_transitions() -> tuple[_ObservedTransition, ...]:
    observations: list[_ObservedTransition] = []
    for incumbent, action in _public_synthetic_incumbents():
        before = observe_program_structure(incumbent)
        after = observe_program_structure(tp.apply_factor_action(incumbent, action))
        if not _site_ready(before, action):
            raise RuntimeError("public synthetic transition exceeds local support")
        observations.append(_ObservedTransition(before, action, after))
    return tuple(observations)


def _freeze_array(value: np.ndarray, dtype: object) -> np.ndarray:
    frozen = np.ascontiguousarray(value, dtype=dtype)
    frozen.setflags(write=False)
    return frozen


def _race_currents(
    contacts: np.ndarray,
    contact_counts: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    known = np.ascontiguousarray(
        _FULL_MATCH_CURRENT_PA
        + _CONTACT_GAP_PA * (contacts @ features - contact_counts)
    )
    return np.concatenate(
        (known, np.array([_UNKNOWN_CURRENT_PA], dtype=np.float64))
    )


def _checkpoint_sha256(
    contacts: np.ndarray,
    contact_counts: np.ndarray,
    output_weights: tuple[np.ndarray, ...],
    rule_banks: tuple[FrozenFieldRuleBank, ...],
    transition_labels: tuple[str, ...],
    training_digest: str,
) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(contacts, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(contact_counts, dtype=np.int64).tobytes())
    for weights in output_weights:
        digest.update(np.ascontiguousarray(weights, dtype=np.float64).tobytes())
    for bank in rule_banks:
        digest.update(bank.field.encode("ascii"))
        digest.update(bank.contacts.tobytes())
        digest.update(bank.contact_counts.tobytes())
        digest.update(bank.projection.tobytes())
    digest.update(_canonical_json(PATH_UNIVERSE))
    digest.update(_canonical_json(_FIELD_SPECS))
    digest.update(_canonical_json(_ACTION_FEATURE_LABELS))
    digest.update(_canonical_json(_ROUTE_LABELS))
    digest.update(_canonical_json(transition_labels))
    digest.update(training_digest.encode("ascii"))
    return digest.hexdigest()


def _physical_race(
    currents_pa: np.ndarray,
    labels: Sequence[str],
    *,
    bank: str,
) -> CellRace:
    """Run the established resting fitted-cell race without rebuilding cells."""

    currents = np.ascontiguousarray(currents_pa, dtype=np.float64)
    candidates = tuple(labels)
    if currents.shape != (len(candidates),) or not candidates:
        raise ValueError("physical race needs one current per candidate")
    try:
        result = _RACE.race_many(currents[None, :], require_unique=False)[0]
    except RestingGlifRaceFailure as error:
        raise physical.CellRaceFailure(
            f"{bank} produced no spike within its fixed bound"
        ) from error
    if result.simultaneous_spike_candidates != 1:
        raise physical.CellRaceFailure(
            f"{bank} produced {result.simultaneous_spike_candidates} "
            "simultaneous first spikes"
        )
    winner = result.winner_cell
    return CellRace(
        bank=bank,
        candidates=candidates,
        currents_pa=tuple(float(value) for value in currents),
        winner_cell=winner,
        winner=candidates[winner],
        spike_step=result.spike_step,
        simultaneous_spike_candidates=1,
    )


def _relay_race(
    value: str | None,
    labels: tuple[str, ...],
    weights: np.ndarray,
    *,
    bank: str,
) -> CellRace:
    source = value if value in labels else UNKNOWN
    spikes = _one_hot(str(source), labels)
    currents = _OUTPUT_CURRENT_PA * (weights @ spikes)
    currents[labels.index(UNKNOWN)] += _UNKNOWN_CURRENT_PA
    return _physical_race(currents, labels, bank=bank)


def _rule_races(
    rule_bank: FrozenFieldRuleBank,
    features: np.ndarray,
    *,
    bank: str,
) -> tuple[CellRace, CellRace]:
    rule_labels = tuple(
        f"{rule_bank.field}-rule-{index}"
        for index in range(rule_bank.contacts.shape[0])
    ) + (UNKNOWN,)
    rule_race = _physical_race(
        _race_currents(rule_bank.contacts, rule_bank.contact_counts, features),
        rule_labels,
        bank=f"{bank}:local-rule",
    )
    spikes = np.zeros(len(rule_labels), dtype=np.float64)
    spikes[rule_race.winner_cell] = 1.0
    currents = _OUTPUT_CURRENT_PA * (rule_bank.projection @ spikes)
    currents[rule_bank.labels.index(UNKNOWN)] += _UNKNOWN_CURRENT_PA
    field_race = _physical_race(currents, rule_bank.labels, bank=bank)
    return rule_race, field_race


@dataclass(frozen=True, slots=True)
class FrozenStructureSuccessor:
    """Readonly factor recognition, local-rule, and identity-relay synapses."""

    contacts: np.ndarray
    contact_counts: np.ndarray
    output_weights: tuple[np.ndarray, ...]
    rule_banks: tuple[FrozenFieldRuleBank, ...]
    transition_labels: tuple[str, ...]
    training_digest: str
    checkpoint_sha256: str

    @property
    def live_checkpoint_sha256(self) -> str:
        return _checkpoint_sha256(
            self.contacts,
            self.contact_counts,
            self.output_weights,
            self.rule_banks,
            self.transition_labels,
            self.training_digest,
        )

    def predict(
        self,
        observation: StructureObservation,
        action: tp.FactorAction,
    ) -> StructurePrediction:
        """Route and race one local edit without materializing a program."""

        before_sha = self.live_checkpoint_sha256
        if before_sha != self.checkpoint_sha256:
            raise RuntimeError("structure-successor checkpoint changed before prediction")
        _validate_observation(observation)
        ready = _site_ready(observation, action)
        action_features = _encode_action_features(action, ready=ready)
        transition_race = _physical_race(
            _race_currents(self.contacts, self.contact_counts, action_features),
            self.transition_labels,
            bank="typed-structure-factor",
        )
        factor_supported = transition_race.winner != UNKNOWN
        site = observation.site(action.path) if ready else None
        rule_races: list[FieldRace] = []
        field_races: list[FieldRace] = []
        predicted_paths: list[PathStructure] = []

        for path in PATH_UNIVERSE:
            route = _route_for(action, path) if factor_supported else _Route(_ROUTE_UNKNOWN)
            local_features = (
                _encode_local_features(action, route, site)
                if factor_supported and site is not None
                else None
            )

            def predict_field(field_index: int, field: str) -> str:
                labels = _FIELD_SPECS[field_index][1]
                bank_name = f"typed-structure:{PATH_UNIVERSE.index(path)}:{field}"
                if route.kind == _ROUTE_UNKNOWN or site is None:
                    race = _relay_race(
                        UNKNOWN,
                        labels,
                        self.output_weights[field_index],
                        bank=bank_name,
                    )
                else:
                    mode = _field_mode(route, field)
                    if mode == _LEARNED_RULE:
                        assert local_features is not None
                        rule_race, race = _rule_races(
                            self.rule_banks[field_index],
                            local_features,
                            bank=bank_name,
                        )
                        rule_races.append(FieldRace(path, field, rule_race))
                    else:
                        value = _relay_value(
                            observation,
                            site,
                            path,
                            route,
                            field,
                            mode,
                        )
                        race = _relay_race(
                            value,
                            labels,
                            self.output_weights[field_index],
                            bank=bank_name,
                        )
                field_races.append(FieldRace(path, field, race))
                return race.winner

            presence = predict_field(0, "presence")
            values: dict[str, str | None] = {
                field: None for field, _ in _SEMANTIC_FIELD_SPECS
            }
            if presence == UNKNOWN:
                for field_index, (field, _) in enumerate(
                    _SEMANTIC_FIELD_SPECS,
                    start=1,
                ):
                    race = _relay_race(
                        UNKNOWN,
                        _FIELD_SPECS[field_index][1],
                        self.output_weights[field_index],
                        bank=(
                            f"typed-structure:{PATH_UNIVERSE.index(path)}:{field}"
                        ),
                    )
                    field_races.append(FieldRace(path, field, race))
                    values[field] = race.winner
            elif presence == PRESENT:
                for field_index, (field, _) in enumerate(
                    _SEMANTIC_FIELD_SPECS,
                    start=1,
                ):
                    values[field] = predict_field(field_index, field)
            predicted_paths.append(
                PathStructure(
                    path=path,
                    presence=presence,
                    production=values["production"],
                    result_type=values["result_type"],
                    scope0_type=values["scope0_type"],
                    scope1_type=values["scope1_type"],
                    binder_marker=values["binder_marker"],
                    path_role=values["path_role"],
                )
            )

        predicted = _observation(None, predicted_paths)
        has_unknown = any(
            item.presence == UNKNOWN
            or any(
                getattr(item, field) == UNKNOWN
                for field, _ in _SEMANTIC_FIELD_SPECS
            )
            for item in predicted.paths
        )
        after_sha = self.live_checkpoint_sha256
        if before_sha != after_sha or after_sha != self.checkpoint_sha256:
            raise RuntimeError("frozen structure successor changed during prediction")
        return StructurePrediction(
            predicted=predicted,
            supported=factor_supported and not has_unknown,
            transition_race=transition_race,
            rule_races=tuple(rule_races),
            field_races=tuple(field_races),
            feature_sha256=sha256(action_features.tobytes()).hexdigest(),
            checkpoint_before_sha256=before_sha,
            checkpoint_after_sha256=after_sha,
        )


def _freeze_rule_bank(
    field: str,
    labels: tuple[str, ...],
    rows: list[np.ndarray],
    outputs: list[str],
) -> FrozenFieldRuleBank:
    if rows:
        contacts = np.stack(rows)
    else:
        contacts = np.zeros((0, _local_feature_count()), dtype=np.float64)
    counts = np.count_nonzero(contacts, axis=1).astype(np.int64)
    projection = np.zeros((len(labels), len(rows) + 1), dtype=np.float64)
    for row, output in enumerate(outputs):
        projection[labels.index(output), row] = 1.0
    projection[labels.index(UNKNOWN), -1] = 1.0
    return FrozenFieldRuleBank(
        field=field,
        labels=labels,
        contacts=_freeze_array(contacts, np.float64),
        contact_counts=_freeze_array(counts, np.int64),
        projection=_freeze_array(projection, np.float64),
    )


def train_public_structure_successor(
) -> tuple[FrozenStructureSuccessor, StructureTrainingSummary]:
    """Observe generic edits and freeze factor-local structural synapses."""

    observations = _collect_transitions()
    factor_rows: list[np.ndarray] = []
    factor_keys: set[bytes] = set()
    for item in observations:
        features = _encode_action_features(item.action, ready=True)
        if features.tobytes() not in factor_keys:
            factor_keys.add(features.tobytes())
            factor_rows.append(features)

    rule_rows: dict[str, list[np.ndarray]] = {
        field: [] for field, _ in _FIELD_SPECS
    }
    rule_outputs: dict[str, list[str]] = {
        field: [] for field, _ in _FIELD_SPECS
    }
    rule_indices: dict[str, dict[bytes, int]] = {
        field: {} for field, _ in _FIELD_SPECS
    }
    for item in observations:
        site = item.before.site(item.action.path)
        for path in PATH_UNIVERSE:
            route = _route_for(item.action, path)
            if route.kind == _ROUTE_UNKNOWN:
                raise RuntimeError("training action lacks a local structural route")
            post = item.after.site(path)
            for field, _ in _FIELD_SPECS:
                output = getattr(post, field)
                if output is None:
                    continue
                mode = _field_mode(route, field)
                if mode != _LEARNED_RULE:
                    relayed = _relay_value(
                        item.before,
                        site,
                        path,
                        route,
                        field,
                        mode,
                    )
                    if relayed != output:
                        raise RuntimeError(
                            f"{field} does not obey its generic local relay"
                        )
                    continue
                features = _encode_local_features(item.action, route, site)
                key = features.tobytes()
                previous = rule_indices[field].get(key)
                if previous is not None:
                    if rule_outputs[field][previous] != output:
                        raise RuntimeError("one local coincidence observed conflicting posts")
                    continue
                rule_indices[field][key] = len(rule_rows[field])
                rule_rows[field].append(features)
                rule_outputs[field].append(output)

    contacts = _freeze_array(np.stack(factor_rows), np.float64)
    contact_counts = _freeze_array(
        np.count_nonzero(contacts, axis=1).astype(np.int64),
        np.int64,
    )
    transition_labels = tuple(
        f"factor-{index}" for index in range(len(factor_rows))
    ) + (UNKNOWN,)
    output_weights = tuple(
        _freeze_array(np.eye(len(labels), dtype=np.float64), np.float64)
        for _, labels in _FIELD_SPECS
    )
    rule_banks = tuple(
        _freeze_rule_bank(field, labels, rule_rows[field], rule_outputs[field])
        for field, labels in _FIELD_SPECS
    )
    training_digest = sha256(
        _canonical_json(
            tuple(
                {
                    "before": item.before.sha256,
                    "action": asdict(item.action),
                    "after": item.after.sha256,
                }
                for item in observations
            )
        )
    ).hexdigest()
    checkpoint_sha = _checkpoint_sha256(
        contacts,
        contact_counts,
        output_weights,
        rule_banks,
        transition_labels,
        training_digest,
    )
    model = FrozenStructureSuccessor(
        contacts=contacts,
        contact_counts=contact_counts,
        output_weights=output_weights,
        rule_banks=rule_banks,
        transition_labels=transition_labels,
        training_digest=training_digest,
        checkpoint_sha256=checkpoint_sha,
    )
    local_rule_rows = sum(bank.contacts.shape[0] for bank in rule_banks)
    summary = StructureTrainingSummary(
        observations=len(observations),
        transition_rows=len(factor_rows),
        local_rule_rows=local_rule_rows,
        path_cells=len(PATH_UNIVERSE),
        action_feature_cells=len(_ACTION_FEATURE_LABELS),
        local_feature_cells=_local_feature_count(),
        recognition_synapses=(
            int(np.count_nonzero(contacts))
            + sum(int(np.count_nonzero(bank.contacts)) for bank in rule_banks)
        ),
        projection_synapses=(
            sum(int(np.count_nonzero(weights)) for weights in output_weights)
            + sum(int(np.count_nonzero(bank.projection)) for bank in rule_banks)
        ),
        training_digest=training_digest,
        checkpoint_sha256=checkpoint_sha,
        host_execution_scope="generic synthetic incumbent edits only",
    )
    return model, summary
