"""Fresh fixed-roster TRAIN panel over generic packed breadth V4.

The banked V1 panel is used only as the authenticated authority for the exact
six visible public-TRAIN rows and their signature-derived literal/cost policy.
Its packed outcome is not imported.  Breadth V4 is the sole construction and
selection path.

All six row identities are authenticated before the first build.  Construction
receives only ``signature``, visible argument rows, caller-supplied literals,
and maximum cost.  Desired values are supplied only to ``select`` after the
retained frontier reports zero desired reads and zero proof publication.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.r3_pilot import (
    packed_generic_typed_canonical_train_freeze_v1 as roster_v1,
)
from experiments.r3_pilot import (
    typed_semantic_packed_generic_train_breadth_v2 as breadth_v2,
)
from experiments.r3_pilot import (
    typed_semantic_packed_generic_train_breadth_v3 as breadth_v3,
)
from experiments.r3_pilot import (
    typed_semantic_packed_generic_train_breadth_v4 as breadth_v4,
)
from humanbrain.typed_program_types import ProgramType, TypedSignature

SCHEMA = "packed-generic-typed-canonical-train-freeze-v3"
PASS_VERDICT = "PASS_PACKED_GENERIC_TYPED_CANONICAL_TRAIN_FREEZE_V3"
STOP_VERDICT = "STOP_PACKED_GENERIC_TYPED_CANONICAL_TRAIN_FREEZE_V3"
STOP_UNSUPPORTED_SIGNATURE = (
    "STOP_PACKED_GENERIC_TYPED_CANONICAL_TRAIN_FREEZE_V3_UNSUPPORTED_SIGNATURE"
)

REPO_ROOT = Path(__file__).resolve().parents[2]

ROSTER_V1_BANKED_COMMIT = "0eeae2d4daf1b02b537bc62dbe3dcb3398489c60"
BREADTH_V2_BANKED_COMMIT = "9e889ed5fa3c21ce57de3b6b83b3ea663f50c582"
BREADTH_V3_BANKED_COMMIT = "6ad0f51e19828da8ed76d73566bd914580be1d0b"
BREADTH_V4_BANKED_COMMIT = "490fc3580f2a1d9b1cc70cad5a3233e4c9863f8f"

ROSTER_V1_SOURCE_PATH = (
    REPO_ROOT / "experiments/r3_pilot/packed_generic_typed_canonical_train_freeze_v1.py"
)
ROSTER_V1_TEST_PATH = REPO_ROOT / "tests/test_packed_generic_typed_canonical_train_freeze_v1.py"
BREADTH_V2_SOURCE_PATH = (
    REPO_ROOT / "experiments/r3_pilot/typed_semantic_packed_generic_train_breadth_v2.py"
)
BREADTH_V2_TEST_PATH = REPO_ROOT / "tests/test_typed_semantic_packed_generic_train_breadth_v2.py"
BREADTH_V3_SOURCE_PATH = (
    REPO_ROOT / "experiments/r3_pilot/typed_semantic_packed_generic_train_breadth_v3.py"
)
BREADTH_V3_TEST_PATH = REPO_ROOT / "tests/test_typed_semantic_packed_generic_train_breadth_v3.py"
BREADTH_V4_SOURCE_PATH = (
    REPO_ROOT / "experiments/r3_pilot/typed_semantic_packed_generic_train_breadth_v4.py"
)
BREADTH_V4_TEST_PATH = REPO_ROOT / "tests/test_typed_semantic_packed_generic_train_breadth_v4.py"

EXPECTED_ROSTER_V1_SOURCE_SHA256 = (
    "7e78e8208c82516d9ca58261ffa223737836387404bd0c9ae7fcf34fb1f6fd0f"
)
EXPECTED_ROSTER_V1_TEST_SHA256 = "0c7091ddb5310b3dcc1cd52946d117f3bf12ca9cda968c66ea098469d97406aa"
EXPECTED_BREADTH_V2_SOURCE_SHA256 = (
    "4c6035992a773a18c848f57c362e0dea24f28b7e4c0dcb244553bb1ca98141de"
)
EXPECTED_BREADTH_V2_TEST_SHA256 = "02d54c1b1b2ba0871ea8c751d06a8b14298a506c5e3e39ea1b933db072473f9c"
EXPECTED_BREADTH_V3_SOURCE_SHA256 = (
    "e8a4c3f846afc96e321a484b0b2c230bf5123851f390c2615bc9775655960ba6"
)
EXPECTED_BREADTH_V3_TEST_SHA256 = "f8d11131b9e0849e765d6e5fbb46491c98896197a7154bc8fb97d139e8318bf4"
EXPECTED_BREADTH_V4_SOURCE_SHA256 = (
    "160f89c08b24eb3f1bca13d447f57bd6a06be60d2293fb1abd6674fecd004280"
)
EXPECTED_BREADTH_V4_TEST_SHA256 = "3ded3acf066be81ee6d7482183a191aca37f936e467ccd214d642a79437bdb1e"

EXPECTED_ROSTER_SHA256 = "632601526021eb047056a83087ddc3690e22716cb864a137d7ba2446ad26c328"
EXPECTED_ROW_SHA256S = (
    "4763d89f0c132006d3670ad562898d73a01b0ca8ad6b3b2fc4710fb433965b45",
    "5ad54b9cecb2a6f8e7d577147425491de31647eefab9d7e2e100b3cb03358275",
    "ec1fbb2086db0ef2a084c3cd6be973decf3e2d5dae2752341bc7e079cea55349",
    "ac8655e3f6090a721b9a67517d8358aefdf14859230be656395a99818cd7016b",
    "36d303800a53368998082269ba73641d51479929ba7ad538713f493e7207cb42",
    "f26980e8257fc7dc8424aa6be995747921c08475e9ea120ec8bdc8e1d1eabd4b",
)

WORK_CAPS = (30_000, 30_000, 80_000)
EXPECTED_DEPENDENCY_ROOT = "daf708124d098985f75580548ef15305c96c186071ce7e637612408481295a72"
EXPECTED_SOURCE_ROOT = "41e30bb34e42f4655d4f6909e9fb8dab7d1a156f8a26b36e9eaf4e1126e9f80a"
EXPECTED_ROW_RESULT_SHA256S = (
    "8b3991de691d2b01190a96897bcd7ce86896bd243e0488fe68564e45eea12869",
    "bb7bde8cd80d697e2eebf6e63dd1d1eb888f1ecd6647b8dc027792254a4c5241",
    "f8af9f88a48b74d7b3dde9c00b2a176d100266f5cfba4bfc5d7790a85dd5b94a",
    "85d2bc277ed6f01b4c649ab3d262daea2361f5f0010465d9598618e4c0e6347f",
    "51e94565b592378e0aaf225d079a51d584fcc31d9b1465a61ed3e82bfa7c4393",
    "d2e42b26314c142893982c345a0d2b321b08c8577e53e97e7ef8c87375ca399a",
)
EXPECTED_PANEL_SHA256 = "aaf3a03f6cb0bcd2142c266fb55ec5de0912c1e476a6c279cabfd33f9bd8436b"

_NAMESPACE = b"packed-generic-typed-canonical-train-freeze-v3\0"
_INT = ProgramType.Int()
_LIST_INT = ProgramType.List(_INT)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    digest = sha256()
    digest.update(_NAMESPACE)
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class DependencyReceipt(NamedTuple):
    roster_v1_source_sha256: str
    roster_v1_test_sha256: str
    breadth_v2_source_sha256: str
    breadth_v2_test_sha256: str
    breadth_v3_source_sha256: str
    breadth_v3_test_sha256: str
    breadth_v4_source_sha256: str
    breadth_v4_test_sha256: str
    protocol_sha256: str
    roster_sha256: str
    row_sha256s: tuple[str, ...]
    work_caps: tuple[int, int, int]
    sha256: str


def _dependency_payload(receipt: DependencyReceipt) -> dict[str, object]:
    return {
        "roster_v1_banked_commit": ROSTER_V1_BANKED_COMMIT,
        "breadth_v2_banked_commit": BREADTH_V2_BANKED_COMMIT,
        "breadth_v3_banked_commit": BREADTH_V3_BANKED_COMMIT,
        "breadth_v4_banked_commit": BREADTH_V4_BANKED_COMMIT,
        "roster_v1_source_sha256": receipt.roster_v1_source_sha256,
        "roster_v1_test_sha256": receipt.roster_v1_test_sha256,
        "breadth_v2_source_sha256": receipt.breadth_v2_source_sha256,
        "breadth_v2_test_sha256": receipt.breadth_v2_test_sha256,
        "breadth_v3_source_sha256": receipt.breadth_v3_source_sha256,
        "breadth_v3_test_sha256": receipt.breadth_v3_test_sha256,
        "breadth_v4_source_sha256": receipt.breadth_v4_source_sha256,
        "breadth_v4_test_sha256": receipt.breadth_v4_test_sha256,
        "protocol_sha256": receipt.protocol_sha256,
        "roster_sha256": receipt.roster_sha256,
        "row_sha256s": receipt.row_sha256s,
        "work_caps": receipt.work_caps,
    }


def _source_root(receipt: DependencyReceipt) -> str:
    request_policy = (
        (tuple(range(-9, 10)), 5),
        (tuple(range(-9, 10)), 5),
        ((), 7),
        (tuple(range(-9, 10)), 9),
        (tuple(range(-9, 10)), 9),
        ((), 7),
    )
    return _digest(
        "panel-source",
        {
            "schema": SCHEMA,
            "dependency_root": receipt.sha256,
            "row_sha256s": receipt.row_sha256s,
            "request_policy": request_policy,
            "engine_seam": "build(signature,cases,literals,max_cost)->select(desired)",
            "work_caps": WORK_CAPS,
        },
    )


def _authenticate_dependencies() -> tuple[
    DependencyReceipt,
    tuple[roster_v1.CanonicalTrainRow, ...],
    str,
]:
    """Authenticate source bytes and the complete roster before any build."""

    if Path(roster_v1.__file__).resolve() != ROSTER_V1_SOURCE_PATH.resolve():
        raise RuntimeError("canonical V1 roster authority path changed")
    if Path(breadth_v4.__file__).resolve() != BREADTH_V4_SOURCE_PATH.resolve():
        raise RuntimeError("generic breadth V4 authority path changed")

    actual_hashes = (
        _file_sha256(ROSTER_V1_SOURCE_PATH),
        _file_sha256(ROSTER_V1_TEST_PATH),
        _file_sha256(BREADTH_V2_SOURCE_PATH),
        _file_sha256(BREADTH_V2_TEST_PATH),
        _file_sha256(BREADTH_V3_SOURCE_PATH),
        _file_sha256(BREADTH_V3_TEST_PATH),
        _file_sha256(BREADTH_V4_SOURCE_PATH),
        _file_sha256(BREADTH_V4_TEST_PATH),
    )
    expected_hashes = (
        EXPECTED_ROSTER_V1_SOURCE_SHA256,
        EXPECTED_ROSTER_V1_TEST_SHA256,
        EXPECTED_BREADTH_V2_SOURCE_SHA256,
        EXPECTED_BREADTH_V2_TEST_SHA256,
        EXPECTED_BREADTH_V3_SOURCE_SHA256,
        EXPECTED_BREADTH_V3_TEST_SHA256,
        EXPECTED_BREADTH_V4_SOURCE_SHA256,
        EXPECTED_BREADTH_V4_TEST_SHA256,
    )
    if actual_hashes != expected_hashes:
        raise RuntimeError("canonical panel dependency source/test freeze changed")
    if (
        roster_v1.EXPECTED_ROSTER_SHA256 != EXPECTED_ROSTER_SHA256
        or roster_v1.EXPECTED_BREADTH_SOURCE_SHA256 != EXPECTED_BREADTH_V2_SOURCE_SHA256
        or roster_v1.EXPECTED_BREADTH_TEST_SHA256 != EXPECTED_BREADTH_V2_TEST_SHA256
    ):
        raise RuntimeError("canonical V1 roster provenance constants changed")

    caps = (
        breadth_v4.MAX_RECIPE_RELATION_WORK,
        breadth_v4.MAX_PHYSICAL_PREDICTIONS,
        breadth_v4.MAX_PHYSICAL_RACES,
    )
    if caps != WORK_CAPS or caps != (
        breadth_v3.MAX_RECIPE_RELATION_WORK,
        breadth_v3.MAX_PHYSICAL_PREDICTIONS,
        breadth_v3.MAX_PHYSICAL_RACES,
    ):
        raise RuntimeError("generic breadth V4/V3 work caps changed")
    if caps != (
        breadth_v2.MAX_RECIPE_RELATION_WORK,
        breadth_v2.MAX_PHYSICAL_PREDICTIONS,
        breadth_v2.MAX_PHYSICAL_RACES,
    ):
        raise RuntimeError("generic breadth V2 work caps changed")

    rows = roster_v1.load_canonical_train_rows()
    row_sha256s = tuple(row.sha256 for row in rows)
    if row_sha256s != EXPECTED_ROW_SHA256S:
        raise RuntimeError("canonical six-row identities changed")
    if roster_v1._roster_sha256(rows) != EXPECTED_ROSTER_SHA256:
        raise RuntimeError("canonical six-row roster root changed")
    request_policy = tuple(roster_v1._signature_request(row.signature) for row in rows)
    expected_policy = (
        (tuple(range(-9, 10)), 5),
        (tuple(range(-9, 10)), 5),
        ((), 7),
        (tuple(range(-9, 10)), 9),
        (tuple(range(-9, 10)), 9),
        ((), 7),
    )
    if request_policy != expected_policy:
        raise RuntimeError("canonical signature-derived request policy changed")

    partial = DependencyReceipt(
        roster_v1_source_sha256=actual_hashes[0],
        roster_v1_test_sha256=actual_hashes[1],
        breadth_v2_source_sha256=actual_hashes[2],
        breadth_v2_test_sha256=actual_hashes[3],
        breadth_v3_source_sha256=actual_hashes[4],
        breadth_v3_test_sha256=actual_hashes[5],
        breadth_v4_source_sha256=actual_hashes[6],
        breadth_v4_test_sha256=actual_hashes[7],
        protocol_sha256=_file_sha256(roster_v1.PROTOCOL_PATH),
        roster_sha256=EXPECTED_ROSTER_SHA256,
        row_sha256s=row_sha256s,
        work_caps=caps,
        sha256="",
    )
    receipt = partial._replace(sha256=_digest("dependency-receipt", _dependency_payload(partial)))
    source_root = _source_root(receipt)
    if EXPECTED_DEPENDENCY_ROOT and receipt.sha256 != EXPECTED_DEPENDENCY_ROOT:
        raise RuntimeError("canonical dependency root changed")
    if EXPECTED_SOURCE_ROOT and source_root != EXPECTED_SOURCE_ROOT:
        raise RuntimeError("canonical panel source root changed")
    return receipt, rows, source_root


def _is_cross_type(signature: TypedSignature) -> bool:
    return (
        signature.result == _INT
        and len(signature.parameters) == 1
        and signature.parameters[0].type == _LIST_INT
    )


def _factorization_events(
    events: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, count)
        for name, count in events
        if name.startswith("workset:selected-factor-")
        or name.startswith("workset:selected-relation-receipt-")
    )


def _root_sha256(root_eclass_ids: tuple[str, ...]) -> str:
    return sha256(_canonical_json(root_eclass_ids)).hexdigest()


class CanonicalTrainRowResult(NamedTuple):
    row_sha256: str
    task_id: str
    family: str
    cross_type_route: bool
    status: str
    error: str | None
    frontier_sha256: str | None
    root_eclass_ids: tuple[str, ...]
    root_eclass_sha256: str | None
    build_work_sha256: str | None
    build_relation_units: int
    build_predictions: int
    build_cache_hits: int
    build_races: int
    build_proofs: int
    build_factorization_events: tuple[tuple[str, int], ...]
    selection_sha256: str | None
    selected_eclass_id: str | None
    selected_proof_forest_sha256: str | None
    race_sha256: str | None
    checkpoint_before_sha256: str | None
    checkpoint_after_sha256: str | None
    selection_relation_units: int
    selection_races: int
    selection_proofs: int
    selection_factorization_events: tuple[tuple[str, int], ...]
    selected_source_sha256: str | None
    sha256: str


def _result_payload(result: CanonicalTrainRowResult) -> dict[str, object]:
    return {field: getattr(result, field) for field in result._fields if field != "sha256"}


def _unsupported_result(
    row: roster_v1.CanonicalTrainRow,
    error: ValueError,
) -> CanonicalTrainRowResult:
    partial = CanonicalTrainRowResult(
        row_sha256=row.sha256,
        task_id=row.task_id,
        family=row.family,
        cross_type_route=_is_cross_type(row.signature),
        status=STOP_UNSUPPORTED_SIGNATURE,
        error=str(error),
        frontier_sha256=None,
        root_eclass_ids=(),
        root_eclass_sha256=None,
        build_work_sha256=None,
        build_relation_units=0,
        build_predictions=0,
        build_cache_hits=0,
        build_races=0,
        build_proofs=0,
        build_factorization_events=(),
        selection_sha256=None,
        selected_eclass_id=None,
        selected_proof_forest_sha256=None,
        race_sha256=None,
        checkpoint_before_sha256=None,
        checkpoint_after_sha256=None,
        selection_relation_units=0,
        selection_races=0,
        selection_proofs=0,
        selection_factorization_events=(),
        selected_source_sha256=None,
        sha256="",
    )
    return partial._replace(sha256=_digest("row-result", _result_payload(partial)))


def _run_row(row: roster_v1.CanonicalTrainRow) -> CanonicalTrainRowResult:
    literals, max_cost = roster_v1._signature_request(row.signature)
    try:
        frontier = breadth_v4.PackedGenericTypedTrainBreadthV4.build(
            row.signature,
            roster_v1._engine_cases(row),
            literals,
            max_cost,
        )
    except ValueError as error:
        if "packed generic breadth" not in str(error):
            raise
        return _unsupported_result(row, error)

    build_work = frontier.work
    build_events = tuple(build_work.relation.event_counts)
    build_factorization = _factorization_events(build_events)
    if (
        build_work.concrete_proof_witnesses != 0
        or build_work.materialized_witnesses != 0
        or dict(build_events).get("frontier:desired-value-read", 0) != 0
        or dict(build_events).get("frontier:materialize-witness", 0) != 0
        or dict(build_events).get(
            "operation:materialize-selected-lazy-proof-plan",
            0,
        )
        != 0
        or dict(build_events).get(
            "operation:workset-materialize-selected-proof",
            0,
        )
        != 0
        or build_factorization
    ):
        raise RuntimeError("canonical V3 build observed desired/proof/factorization work")

    selection = None
    status = frontier.status
    if frontier.status == breadth_v4.FRONTIER_READY:
        selection = frontier.select(roster_v1._engine_desired(row))
        status = selection.status
    selection_work = build_work if selection is None else selection.work
    selection_events = tuple(selection_work.relation.event_counts)
    selection_factorization = _factorization_events(selection_events)
    if not _is_cross_type(row.signature) and selection_factorization:
        raise RuntimeError("V4 factorization entered a frozen scalar/List/String route")
    if selection is not None and (
        selection.checkpoint_before_sha256 != frontier.state.operator_checkpoint_sha256
        or selection.checkpoint_after_sha256 != frontier.state.operator_checkpoint_sha256
    ):
        raise RuntimeError("typed operator checkpoint changed during canonical selection")

    selected_source_sha256 = None
    if selection is not None and len(selection.witnesses) == 1:
        selected_source_sha256 = sha256(selection.witnesses[0].source.encode("utf-8")).hexdigest()
    roots = frontier.root_eclass_ids
    partial = CanonicalTrainRowResult(
        row_sha256=row.sha256,
        task_id=row.task_id,
        family=row.family,
        cross_type_route=_is_cross_type(row.signature),
        status=status,
        error=None,
        frontier_sha256=frontier.sha256,
        root_eclass_ids=roots,
        root_eclass_sha256=_root_sha256(roots),
        build_work_sha256=build_work.sha256,
        build_relation_units=build_work.relation.completed_units,
        build_predictions=build_work.physical_predictions,
        build_cache_hits=build_work.prediction_cache_hits,
        build_races=build_work.physical_races,
        build_proofs=build_work.concrete_proof_witnesses,
        build_factorization_events=build_factorization,
        selection_sha256=None if selection is None else selection.sha256,
        selected_eclass_id=None if selection is None else selection.selected_eclass_id,
        selected_proof_forest_sha256=(
            None if selection is None else selection.selected_proof_forest_sha256
        ),
        race_sha256=(
            None if selection is None or selection.race is None else selection.race.sha256
        ),
        checkpoint_before_sha256=(
            None if selection is None else selection.checkpoint_before_sha256
        ),
        checkpoint_after_sha256=(None if selection is None else selection.checkpoint_after_sha256),
        selection_relation_units=selection_work.relation.completed_units,
        selection_races=selection_work.physical_races,
        selection_proofs=0 if selection is None else selection.exact_proof_count,
        selection_factorization_events=selection_factorization,
        selected_source_sha256=selected_source_sha256,
        sha256="",
    )
    return partial._replace(sha256=_digest("row-result", _result_payload(partial)))


class FrozenRowOutcome(NamedTuple):
    row_sha256: str
    status: str
    frontier_sha256: str | None
    root_count: int
    root_eclass_sha256: str | None
    build_work_sha256: str | None
    build_relation_units: int
    build_predictions: int
    build_cache_hits: int
    build_races: int
    build_proofs: int
    selection_sha256: str | None
    selected_eclass_id: str | None
    selected_proof_forest_sha256: str | None
    race_sha256: str | None
    checkpoint_sha256: str | None
    selection_relation_units: int
    selection_races: int
    selection_proofs: int
    selection_factorization_events: tuple[tuple[str, int], ...]
    result_sha256: str


FROZEN_ROW_OUTCOMES = (
    FrozenRowOutcome(
        row_sha256=EXPECTED_ROW_SHA256S[0],
        status=breadth_v4.SELECTED,
        frontier_sha256="76692dcc4e3c04af79e2f0b9cc36b40033c0c474854e136fada2472d43e20c71",
        root_count=60,
        root_eclass_sha256="5d4e79af63e04798eba84b7ec016746a9ff29aa8bccacb2218939e61fe29f5ad",
        build_work_sha256="362a49d206a54df41b800c914a16e91faa144ba01e30823552c210deccf2da11",
        build_relation_units=19_428,
        build_predictions=278,
        build_cache_hits=266,
        build_races=954,
        build_proofs=0,
        selection_sha256="4213277d69d81b789c814435919270fd7cef466189753e3bd882fa0dd5f1bc6b",
        selected_eclass_id="de5e1879b0c7238e62fee89fa124255ecd9d6b33129fa42aeace601eb111ff3d",
        selected_proof_forest_sha256="2779f86243e26fde7d5ff416197cb3239c80b5ad0b02f8bf8f7b3fbaa4b8be56",
        race_sha256="f3d0f10b45929414eb434ff8da7e42049f738f8933a1c7aa5c6aed281626501d",
        checkpoint_sha256="92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e",
        selection_relation_units=19_453,
        selection_races=955,
        selection_proofs=1,
        selection_factorization_events=(),
        result_sha256=EXPECTED_ROW_RESULT_SHA256S[0],
    ),
    FrozenRowOutcome(
        row_sha256=EXPECTED_ROW_SHA256S[1],
        status=breadth_v4.STOP_AMBIGUOUS,
        frontier_sha256="b9f123b7d379cbd8d74a04292e8c20c324894487014756cba8e45074be440c8e",
        root_count=211,
        root_eclass_sha256="f5389bdbd9084869efb6645d8500b4bacf2c8f7236b50a1f66d3592ae31b8c4b",
        build_work_sha256="7e076a1dd67180947b5f517d6ae21c540f49c1010f73393da7300ac642b7c505",
        build_relation_units=28_973,
        build_predictions=379,
        build_cache_hits=537,
        build_races=1_356,
        build_proofs=0,
        selection_sha256="3287e7fd27b8107536ca02c4dacf8747170fca5deda2fa90d76f4ad3af3bd05e",
        selected_eclass_id="bd46b7e77942994d26e73315964f3f6d880b1af1d78460418cc7417a27299355",
        selected_proof_forest_sha256="e027b90a98b63bc987d76b4bd88c5f6db20240f8dc11b39bb247a331424f4d97",
        race_sha256="34ef9e4bbf5450cdac1a038a41015b24655d6bc196095ae4c871556f7d4da572",
        checkpoint_sha256="92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e",
        selection_relation_units=29_009,
        selection_races=1_357,
        selection_proofs=2,
        selection_factorization_events=(),
        result_sha256=EXPECTED_ROW_RESULT_SHA256S[1],
    ),
    FrozenRowOutcome(
        row_sha256=EXPECTED_ROW_SHA256S[2],
        status=breadth_v4.SELECTED,
        frontier_sha256="e54d250840039be07a29c45d2cc5710b4a316c66f19addb35bede6e24cb75612",
        root_count=7,
        root_eclass_sha256="0a427db1b1f1599532f7e299f4f99179c100dcacac34eddcace3e522abee867c",
        build_work_sha256="91668cee4cb882ec7ca9c25db5f9c48c63d60a20a7b98466bd0efeba7b9abfb5",
        build_relation_units=919,
        build_predictions=108,
        build_cache_hits=34,
        build_races=241,
        build_proofs=0,
        selection_sha256="c406404e5cf0d87072a1289f4c7160b92698b5b3dceb89f0888256e05671efb7",
        selected_eclass_id="b341ecb3e7f2def76feada51b2b05da9809d8513eb323389366ad8620f9f40cf",
        selected_proof_forest_sha256="a641d58f7ba8238030beaf8ac50e184986ddae404a512d989a53e111f8477534",
        race_sha256="6624810658e88ea29614fb913129f836683fffdfd2366e8f363061c13e02d8ef",
        checkpoint_sha256="92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e",
        selection_relation_units=1_236,
        selection_races=242,
        selection_proofs=1,
        selection_factorization_events=(),
        result_sha256=EXPECTED_ROW_RESULT_SHA256S[2],
    ),
    FrozenRowOutcome(
        row_sha256=EXPECTED_ROW_SHA256S[3],
        status=breadth_v4.STOP_AMBIGUOUS,
        frontier_sha256="e9b9243e9f6a5e4b4b1529f64490ca2222d33fc1a4beab5f142a0076436817a4",
        root_count=122,
        root_eclass_sha256="1f5f592f4aa5c3b7dd406f1aaf80d956d58f31fd9e43d78902f0cf4a5d92df44",
        build_work_sha256="7a754b531d9f36aacc9b780c933efd3c99168787fc5d4459088fecd901c970e0",
        build_relation_units=25_762,
        build_predictions=1_743,
        build_cache_hits=6_303,
        build_races=4_571,
        build_proofs=0,
        selection_sha256="1b37ab90b45a2405987c37547d2787641ecceb7e7ef9d3233fb7dc1d3700a975",
        selected_eclass_id="fd04cf98729705ae34e679af78edcbe4cb00937d00994656423db2895e4f8aad",
        selected_proof_forest_sha256="0ab38bea56604f76daa62f122cf93604c47754a5a0c206f0f860665647543242",
        race_sha256="ddffb3cc669d7ea236b77a42612962f77e7a2eb3b4d70e8adb5afd4fb911849d",
        checkpoint_sha256="92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e",
        selection_relation_units=28_529,
        selection_races=4_572,
        selection_proofs=10,
        selection_factorization_events=(
            ("workset:selected-factor-branch", 51),
            ("workset:selected-factor-child-lookup", 53),
            ("workset:selected-factor-position", 44),
            ("workset:selected-factor-shape-lookup", 33),
            ("workset:selected-relation-receipt-authenticate", 122),
            ("workset:selected-relation-receipt-issue", 81),
            ("workset:selected-relation-receipt-registry-open", 1),
        ),
        result_sha256=EXPECTED_ROW_RESULT_SHA256S[3],
    ),
    FrozenRowOutcome(
        row_sha256=EXPECTED_ROW_SHA256S[4],
        status=breadth_v4.STOP_WORK_BOUND,
        frontier_sha256="3c44d3a625ce26a7522d79326f879791073663b55a5c1a4d32f5282afe1b7654",
        root_count=171,
        root_eclass_sha256="4931d857a3e76b740215937321ac9f9e152fd02663bb490694b32f115a80c972",
        build_work_sha256="5eeeaf69e9389a13ad7683d395ad55b6f08e3d171f734f05a97607f806c59316",
        build_relation_units=27_477,
        build_predictions=1_874,
        build_cache_hits=5_548,
        build_races=5_023,
        build_proofs=0,
        selection_sha256="c338129de0cf0650bc148c661ff1db8187a3b3a4336c92640aacb85c410d0ded",
        selected_eclass_id=None,
        selected_proof_forest_sha256=None,
        race_sha256=None,
        checkpoint_sha256="92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e",
        selection_relation_units=30_000,
        selection_races=5_024,
        selection_proofs=0,
        selection_factorization_events=(
            ("workset:selected-factor-branch", 47),
            ("workset:selected-factor-child-lookup", 49),
            ("workset:selected-factor-position", 38),
            ("workset:selected-factor-shape-lookup", 27),
            ("workset:selected-relation-receipt-authenticate", 113),
            ("workset:selected-relation-receipt-issue", 88),
            ("workset:selected-relation-receipt-registry-open", 1),
        ),
        result_sha256=EXPECTED_ROW_RESULT_SHA256S[4],
    ),
    FrozenRowOutcome(
        row_sha256=EXPECTED_ROW_SHA256S[5],
        status=breadth_v4.SELECTED,
        frontier_sha256="18f274cb77307b01a392edb7c6e68e06c68f1f301f46a79806aaaaddaa28f09b",
        root_count=7,
        root_eclass_sha256="0a427db1b1f1599532f7e299f4f99179c100dcacac34eddcace3e522abee867c",
        build_work_sha256="91668cee4cb882ec7ca9c25db5f9c48c63d60a20a7b98466bd0efeba7b9abfb5",
        build_relation_units=919,
        build_predictions=108,
        build_cache_hits=34,
        build_races=241,
        build_proofs=0,
        selection_sha256="33d50629c37f3286b7d10eb924ca2f5c6e00b8083c62db1de5f9a5fd7a3938a9",
        selected_eclass_id="5631777e0da50fcb3cfcf56ead7e451e47e2cd3374d498515ca1c27593590530",
        selected_proof_forest_sha256="74d571c3b28217f1a6bb3cb228f46688293d149807539545bda4c72c572bc2c1",
        race_sha256="476e1f93e598e463f8e8c8ff8803ebd327e116a7ce3172f6072ef1ad27440c90",
        checkpoint_sha256="92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e",
        selection_relation_units=1_283,
        selection_races=242,
        selection_proofs=1,
        selection_factorization_events=(),
        result_sha256=EXPECTED_ROW_RESULT_SHA256S[5],
    ),
)


class CanonicalTrainPanel(NamedTuple):
    schema: str
    verdict: str
    dependency_root: str
    source_root: str
    roster_sha256: str
    results: tuple[CanonicalTrainRowResult, ...]
    selected_rows: int
    stopped_rows: int
    sha256: str


def _panel_payload(panel: CanonicalTrainPanel) -> dict[str, object]:
    return {
        "schema": panel.schema,
        "verdict": panel.verdict,
        "dependency_root": panel.dependency_root,
        "source_root": panel.source_root,
        "roster_sha256": panel.roster_sha256,
        "results": tuple(result.sha256 for result in panel.results),
        "selected_rows": panel.selected_rows,
        "stopped_rows": panel.stopped_rows,
    }


def run_canonical_train_freeze() -> CanonicalTrainPanel:
    """Spend the immutable roster without substitution, retuning, or tie-breaks."""

    dependency, rows, source_root = _authenticate_dependencies()
    results = tuple(_run_row(row) for row in rows)
    selected = sum(result.status == breadth_v4.SELECTED for result in results)
    stopped = len(results) - selected
    partial = CanonicalTrainPanel(
        schema=SCHEMA,
        verdict=PASS_VERDICT if stopped == 0 else STOP_VERDICT,
        dependency_root=dependency.sha256,
        source_root=source_root,
        roster_sha256=dependency.roster_sha256,
        results=results,
        selected_rows=selected,
        stopped_rows=stopped,
        sha256="",
    )
    panel = partial._replace(sha256=_digest("canonical-panel", _panel_payload(partial)))
    result_sha256s = tuple(result.sha256 for result in panel.results)
    if EXPECTED_ROW_RESULT_SHA256S and result_sha256s != EXPECTED_ROW_RESULT_SHA256S:
        raise RuntimeError("frozen canonical V3 row outcomes changed")
    if EXPECTED_PANEL_SHA256 and panel.sha256 != EXPECTED_PANEL_SHA256:
        raise RuntimeError("frozen canonical V3 panel outcome changed")
    return panel


__all__ = [
    "BREADTH_V4_BANKED_COMMIT",
    "EXPECTED_DEPENDENCY_ROOT",
    "EXPECTED_PANEL_SHA256",
    "EXPECTED_ROSTER_SHA256",
    "EXPECTED_ROW_RESULT_SHA256S",
    "EXPECTED_ROW_SHA256S",
    "EXPECTED_SOURCE_ROOT",
    "FROZEN_ROW_OUTCOMES",
    "PASS_VERDICT",
    "SCHEMA",
    "STOP_UNSUPPORTED_SIGNATURE",
    "STOP_VERDICT",
    "CanonicalTrainPanel",
    "CanonicalTrainRowResult",
    "DependencyReceipt",
    "FrozenRowOutcome",
    "run_canonical_train_freeze",
]
