"""Canonical public-TRAIN engineering panel for the packed generic breadth.

The six rows are the ordinal-zero public TRAIN tasks from the already frozen
six-family surface.  Their task identities, typed signatures, and four visible
regression rows are authenticated as one roster before the first packed build.
Property rows, prompts, expected programs, DEV/eval material, and protected
artifacts are neither opened nor passed to this panel.

Task and family labels are audit metadata only.  The engine receives exactly
``build(signature, cases, literals, max_cost)``; literals and cost are selected
only from the typed signature.  Desired outputs enter only after a READY build
has demonstrated zero desired reads and zero proof materialization.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.r3_pilot import (
    typed_semantic_packed_generic_train_breadth_v2 as breadth,
)
from humanbrain.simple_swe.protocol import load_protocol
from humanbrain.simple_swe.tasks import GeneratedTask, generate_visible_partition
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "packed-generic-typed-canonical-train-freeze-v1"
PASS_VERDICT = "PASS_PACKED_GENERIC_TYPED_CANONICAL_TRAIN_FREEZE_V1"
STOP_VERDICT = "STOP_PACKED_GENERIC_TYPED_CANONICAL_TRAIN_FREEZE_V1"
STOP_UNSUPPORTED_SIGNATURE = (
    "STOP_PACKED_GENERIC_TYPED_CANONICAL_TRAIN_FREEZE_V1_UNSUPPORTED_SIGNATURE"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
EXPECTED_PROTOCOL_SHA256 = "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
BREADTH_SOURCE_PATH = (
    REPO_ROOT / "experiments/r3_pilot/typed_semantic_packed_generic_train_breadth_v2.py"
)
BREADTH_TEST_PATH = REPO_ROOT / "tests/test_typed_semantic_packed_generic_train_breadth_v2.py"
EXPECTED_BREADTH_SOURCE_SHA256 = "4c6035992a773a18c848f57c362e0dea24f28b7e4c0dcb244553bb1ca98141de"
EXPECTED_BREADTH_TEST_SHA256 = "02d54c1b1b2ba0871ea8c751d06a8b14298a506c5e3e39ea1b933db072473f9c"

PUBLIC_FAMILIES = (
    "t0_scalar_clamp",
    "t0_scalar_nonnegative",
    "t1_string_filter_class",
    "t2_clamp_then_sum",
    "t2_map_then_count",
    "t2_filter_then_reverse",
)
PUBLIC_FAMILY_TIERS = (0, 0, 1, 2, 2, 2)
VISIBLE_REGRESSION_CASES = 4
PUBLIC_LITERAL_BANK = tuple(range(-9, 10))

EXPECTED_TASK_IDS = tuple(
    f"simple-swe-v1/train/tier-{tier}/{family}/000"
    for family, tier in zip(PUBLIC_FAMILIES, PUBLIC_FAMILY_TIERS, strict=True)
)
EXPECTED_VISIBLE_PROBLEM_SHA256S = (
    "a56f697d445a63e972f19f7110b4d1f4b0851b5b5e5024d54ef7ee81981fe7a3",
    "0832d0691fdbff1e84c0caec71f21986c48bb68f9c889f016dcbac66def2c762",
    "43e21d341536a6c4d441532c400944498dfc676c8fca6b2ffd75dde2a40cc25a",
    "bd996fbb2efa2e11f3b6eed433ed52b274140fe36fc56f685b952caae59bd01f",
    "ef6e953102768a8ed6dc0f6003b2094300f2884222a1b0c77092e428bf7dd1e6",
    "ec44534db6ef0ba095c5b6a97aca89dea14f99748f3645b8caf4e571aa2ac0c9",
)
EXPECTED_ROSTER_SHA256 = "632601526021eb047056a83087ddc3690e22716cb864a137d7ba2446ad26c328"

_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)
_STRING = ProgramType.String()
_EXPECTED_SIGNATURES = (
    ("n", _INT, _INT),
    ("n", _INT, _INT),
    ("text", _STRING, _STRING),
    ("values", _LIST_INT, _INT),
    ("values", _LIST_INT, _INT),
    ("text", _STRING, _STRING),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    digest = sha256()
    digest.update(b"packed-generic-typed-canonical-train-freeze-v1\0")
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _snapshot(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(_snapshot(item) for item in value)
    return value


class CanonicalTrainRow(NamedTuple):
    ordinal: int
    family: str
    task_id: str
    signature: TypedSignature
    cases: tuple[tuple[object, ...], ...]
    desired: tuple[object, ...]
    visible_problem_sha256: str
    sha256: str


def _row_payload(row: CanonicalTrainRow) -> dict[str, object]:
    return {
        "ordinal": row.ordinal,
        "family": row.family,
        "task_id": row.task_id,
        "signature": row.signature.to_record(),
        "cases": row.cases,
        "desired": row.desired,
        "visible_problem_sha256": row.visible_problem_sha256,
    }


def _visible_problem(
    task: GeneratedTask,
) -> tuple[TypedSignature, tuple[tuple[object, ...], ...], tuple[object, ...]]:
    regressions = task.private.regression_cases
    if len(regressions) != VISIBLE_REGRESSION_CASES:
        raise RuntimeError("canonical visible regression count changed")
    cases: list[tuple[object, ...]] = []
    desired: list[object] = []
    for case in regressions:
        row = _snapshot(case.args)
        if not isinstance(row, tuple):
            raise RuntimeError("canonical visible arguments are not a tuple")
        cases.append(row)
        desired.append(_snapshot(case.expected))
    return task.public.signature, tuple(cases), tuple(desired)


def _visible_problem_sha256(
    signature: TypedSignature,
    cases: tuple[tuple[object, ...], ...],
    desired: tuple[object, ...],
) -> str:
    return sha256(
        _canonical_json(
            {
                "signature": signature.to_record(),
                "arguments": cases,
                "desired": desired,
            }
        )
    ).hexdigest()


def _roster_sha256(rows: Sequence[CanonicalTrainRow]) -> str:
    return sha256(
        _canonical_json(tuple((row.task_id, row.visible_problem_sha256) for row in rows))
    ).hexdigest()


def _require_breadth_freeze() -> tuple[str, str]:
    source_sha256 = sha256(BREADTH_SOURCE_PATH.read_bytes()).hexdigest()
    test_sha256 = sha256(BREADTH_TEST_PATH.read_bytes()).hexdigest()
    if (
        source_sha256 != EXPECTED_BREADTH_SOURCE_SHA256
        or test_sha256 != EXPECTED_BREADTH_TEST_SHA256
    ):
        raise RuntimeError("packed generic breadth source/test freeze changed")
    if (
        breadth.MAX_RECIPE_RELATION_WORK != 30_000
        or breadth.MAX_PHYSICAL_PREDICTIONS != 30_000
        or breadth.MAX_PHYSICAL_RACES != 80_000
    ):
        raise RuntimeError("packed generic breadth work caps changed")
    return source_sha256, test_sha256


def load_canonical_train_rows() -> tuple[CanonicalTrainRow, ...]:
    """Authenticate all six visible rows before any packed outcome exists."""

    if sha256(PROTOCOL_PATH.read_bytes()).hexdigest() != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("canonical Simple-SWE protocol bytes changed")
    protocol = load_protocol(PROTOCOL_PATH)
    if protocol.path != PROTOCOL_PATH.resolve():
        raise RuntimeError("canonical TRAIN protocol path changed")
    generated = generate_visible_partition(protocol, "train")
    index = {
        (task.public.family, task.public.ordinal): task
        for task in generated
        if task.public.family in PUBLIC_FAMILIES and task.public.ordinal == 0
    }
    if set(index) != {(family, 0) for family in PUBLIC_FAMILIES}:
        raise RuntimeError("canonical six-family ordinal-zero roster changed")

    rows: list[CanonicalTrainRow] = []
    for ordinal, (family, task_id, expected_problem, signature_shape) in enumerate(
        zip(
            PUBLIC_FAMILIES,
            EXPECTED_TASK_IDS,
            EXPECTED_VISIBLE_PROBLEM_SHA256S,
            _EXPECTED_SIGNATURES,
            strict=True,
        )
    ):
        task = index[(family, 0)]
        if (
            task.public.partition != "train"
            or task.public.family != family
            or task.public.ordinal != 0
            or task.public.task_id != task_id
        ):
            raise RuntimeError("canonical public TRAIN row identity changed")
        signature, cases, desired = _visible_problem(task)
        parameter_name, parameter_type, result_type = signature_shape
        if (
            len(signature.parameters) != 1
            or signature.parameters[0].name != parameter_name
            or signature.parameters[0].type != parameter_type
            or signature.result != result_type
        ):
            raise RuntimeError("canonical public TRAIN signature changed")
        problem_sha256 = _visible_problem_sha256(signature, cases, desired)
        if problem_sha256 != expected_problem:
            raise RuntimeError("canonical visible TRAIN problem changed")
        partial = CanonicalTrainRow(
            ordinal=ordinal,
            family=family,
            task_id=task_id,
            signature=signature,
            cases=cases,
            desired=desired,
            visible_problem_sha256=problem_sha256,
            sha256="",
        )
        rows.append(partial._replace(sha256=_digest("canonical-row", _row_payload(partial))))
    frozen = tuple(rows)
    if _roster_sha256(frozen) != EXPECTED_ROSTER_SHA256:
        raise RuntimeError("canonical visible TRAIN roster root changed")
    return frozen


def _engine_value(value: object, value_type: ProgramType) -> object:
    if value_type == _LIST_INT:
        if not isinstance(value, tuple):
            raise RuntimeError("canonical List[Int] row lost its immutable snapshot")
        return list(value)
    return value


def _engine_cases(row: CanonicalTrainRow) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(
            _engine_value(value, parameter.type)
            for parameter, value in zip(row.signature.parameters, case, strict=True)
        )
        for case in row.cases
    )


def _engine_desired(row: CanonicalTrainRow) -> tuple[object, ...]:
    return tuple(_engine_value(value, row.signature.result) for value in row.desired)


def _signature_request(signature: TypedSignature) -> tuple[tuple[int, ...], int]:
    """Choose the fixed generic request without task or family metadata."""

    if (
        len(signature.parameters) == 1
        and signature.parameters[0].type == _INT
        and signature.result == _INT
    ):
        return PUBLIC_LITERAL_BANK, 5
    if (
        len(signature.parameters) == 1
        and signature.parameters[0].type == _STRING
        and signature.result == _STRING
    ):
        return (), 7
    return PUBLIC_LITERAL_BANK, 9


class CanonicalTrainRowResult(NamedTuple):
    row_sha256: str
    task_id: str
    family: str
    status: str
    error: str | None
    frontier_sha256: str | None
    root_eclass_ids: tuple[str, ...]
    build_work_sha256: str | None
    build_relation_units: int
    build_predictions: int
    build_cache_hits: int
    build_races: int
    build_proofs: int
    selection_sha256: str | None
    selected_eclass_id: str | None
    selection_relation_units: int
    selection_races: int
    selection_proofs: int
    selected_source: str | None
    sha256: str


def _result_payload(result: CanonicalTrainRowResult) -> dict[str, object]:
    return {
        "row_sha256": result.row_sha256,
        "task_id": result.task_id,
        "family": result.family,
        "status": result.status,
        "error": result.error,
        "frontier_sha256": result.frontier_sha256,
        "root_eclass_ids": result.root_eclass_ids,
        "build_work_sha256": result.build_work_sha256,
        "build_relation_units": result.build_relation_units,
        "build_predictions": result.build_predictions,
        "build_cache_hits": result.build_cache_hits,
        "build_races": result.build_races,
        "build_proofs": result.build_proofs,
        "selection_sha256": result.selection_sha256,
        "selected_eclass_id": result.selected_eclass_id,
        "selection_relation_units": result.selection_relation_units,
        "selection_races": result.selection_races,
        "selection_proofs": result.selection_proofs,
        "selected_source": result.selected_source,
    }


def _unsupported_result(row: CanonicalTrainRow, error: ValueError) -> CanonicalTrainRowResult:
    partial = CanonicalTrainRowResult(
        row_sha256=row.sha256,
        task_id=row.task_id,
        family=row.family,
        status=STOP_UNSUPPORTED_SIGNATURE,
        error=str(error),
        frontier_sha256=None,
        root_eclass_ids=(),
        build_work_sha256=None,
        build_relation_units=0,
        build_predictions=0,
        build_cache_hits=0,
        build_races=0,
        build_proofs=0,
        selection_sha256=None,
        selected_eclass_id=None,
        selection_relation_units=0,
        selection_races=0,
        selection_proofs=0,
        selected_source=None,
        sha256="",
    )
    return partial._replace(sha256=_digest("canonical-row-result", _result_payload(partial)))


def _run_row(row: CanonicalTrainRow) -> CanonicalTrainRowResult:
    if row.sha256 != _digest("canonical-row", _row_payload(row)):
        raise RuntimeError("canonical row changed after roster freeze")
    literals, max_cost = _signature_request(row.signature)
    try:
        frontier = breadth.PackedGenericTypedTrainBreadthV2.build(
            row.signature,
            _engine_cases(row),
            literals,
            max_cost,
        )
    except ValueError as error:
        if "packed generic breadth requires" not in str(error):
            raise
        return _unsupported_result(row, error)

    build_work = frontier.work
    build_events = dict(build_work.relation.event_counts)
    if (
        build_work.concrete_proof_witnesses != 0
        or build_work.materialized_witnesses != 0
        or build_events.get("frontier:desired-value-read", 0) != 0
        or build_events.get("operation:materialize-selected-lazy-proof-plan", 0) != 0
        or build_events.get("operation:workset-materialize-selected-proof", 0) != 0
    ):
        raise RuntimeError("canonical packed build observed desired/proof work")

    selection = None
    status = frontier.status
    if frontier.status == breadth.FRONTIER_READY:
        selection = frontier.select(_engine_desired(row))
        status = selection.status
    selected_source = None
    if selection is not None and len(selection.witnesses) == 1:
        selected_source = selection.witnesses[0].source
    selection_work = build_work if selection is None else selection.work
    partial = CanonicalTrainRowResult(
        row_sha256=row.sha256,
        task_id=row.task_id,
        family=row.family,
        status=status,
        error=None,
        frontier_sha256=frontier.sha256,
        root_eclass_ids=frontier.root_eclass_ids,
        build_work_sha256=build_work.sha256,
        build_relation_units=build_work.relation.completed_units,
        build_predictions=build_work.physical_predictions,
        build_cache_hits=build_work.prediction_cache_hits,
        build_races=build_work.physical_races,
        build_proofs=build_work.concrete_proof_witnesses,
        selection_sha256=None if selection is None else selection.sha256,
        selected_eclass_id=(None if selection is None else selection.selected_eclass_id),
        selection_relation_units=selection_work.relation.completed_units,
        selection_races=selection_work.physical_races,
        selection_proofs=(0 if selection is None else selection.exact_proof_count),
        selected_source=selected_source,
        sha256="",
    )
    return partial._replace(sha256=_digest("canonical-row-result", _result_payload(partial)))


class CanonicalTrainPanel(NamedTuple):
    schema: str
    verdict: str
    roster_sha256: str
    breadth_source_sha256: str
    breadth_test_sha256: str
    results: tuple[CanonicalTrainRowResult, ...]
    selected_rows: int
    stopped_rows: int
    sha256: str


def _panel_payload(panel: CanonicalTrainPanel) -> dict[str, object]:
    return {
        "schema": panel.schema,
        "verdict": panel.verdict,
        "roster_sha256": panel.roster_sha256,
        "breadth_source_sha256": panel.breadth_source_sha256,
        "breadth_test_sha256": panel.breadth_test_sha256,
        "results": tuple(result.sha256 for result in panel.results),
        "selected_rows": panel.selected_rows,
        "stopped_rows": panel.stopped_rows,
    }


def run_canonical_train_freeze() -> CanonicalTrainPanel:
    """Run all six immutable TRAIN rows once without retuning or substitution."""

    source_sha256, test_sha256 = _require_breadth_freeze()
    rows = load_canonical_train_rows()
    roster_sha256 = _roster_sha256(rows)
    if roster_sha256 != EXPECTED_ROSTER_SHA256:
        raise RuntimeError("canonical roster changed before packed execution")
    results = tuple(_run_row(row) for row in rows)
    selected = sum(result.status == breadth.SELECTED for result in results)
    stopped = len(results) - selected
    verdict = PASS_VERDICT if stopped == 0 else STOP_VERDICT
    partial = CanonicalTrainPanel(
        schema=SCHEMA,
        verdict=verdict,
        roster_sha256=roster_sha256,
        breadth_source_sha256=source_sha256,
        breadth_test_sha256=test_sha256,
        results=results,
        selected_rows=selected,
        stopped_rows=stopped,
        sha256="",
    )
    return partial._replace(sha256=_digest("canonical-panel", _panel_payload(partial)))


__all__ = [
    "EXPECTED_BREADTH_SOURCE_SHA256",
    "EXPECTED_BREADTH_TEST_SHA256",
    "EXPECTED_ROSTER_SHA256",
    "EXPECTED_TASK_IDS",
    "EXPECTED_VISIBLE_PROBLEM_SHA256S",
    "PASS_VERDICT",
    "PUBLIC_FAMILIES",
    "SCHEMA",
    "STOP_UNSUPPORTED_SIGNATURE",
    "STOP_VERDICT",
    "CanonicalTrainPanel",
    "CanonicalTrainRow",
    "CanonicalTrainRowResult",
    "load_canonical_train_rows",
    "run_canonical_train_freeze",
]
