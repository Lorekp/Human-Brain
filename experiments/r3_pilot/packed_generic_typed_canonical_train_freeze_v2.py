"""Fixed-roster canonical TRAIN spend through packed generic breadth V3.

This successor authenticates the banked V1 panel source, its focused test, the
frozen V2 breadth bytes, and the separately banked V3 cross-type mechanism
before opening the already-frozen six-row visible TRAIN roster.  V1 remains the
sole row provider; its exact row identities and roster root are checked before
the first V3 build.

Commit identifiers, file hashes, task labels, family labels, and row hashes are
host-side audit metadata only.  The engine receives exactly the typed
signature, V1-visible argument cases, signature-derived literal/cost request,
and—after a desired-free READY build—the V1-visible desired vector.  No prompt,
property row, DEV/eval/protected input, expected program, or host tie-break is
available to the packed selection.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.r3_pilot import packed_generic_typed_canonical_train_freeze_v1 as v1
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v2 as breadth_v2
from experiments.r3_pilot import typed_semantic_packed_generic_train_breadth_v3 as breadth_v3

SCHEMA = "packed-generic-typed-canonical-train-freeze-v2"
PASS_VERDICT = "PASS_PACKED_GENERIC_TYPED_CANONICAL_TRAIN_FREEZE_V2"
STOP_VERDICT = "STOP_PACKED_GENERIC_TYPED_CANONICAL_TRAIN_FREEZE_V2"

V1_COMMIT = "0eeae2d"
V1_COMMIT_FULL = "0eeae2d4daf1b02b537bc62dbe3dcb3398489c60"
V3_COMMIT = "6ad0f51"
V3_COMMIT_FULL = "6ad0f51e19828da8ed76d73566bd914580be1d0b"

EXPECTED_V1_PANEL_SHA256 = "2e5f4ff70dac02c095ced103af03b432cc7d57b6f9a710a18b64d1cda3c6ffaa"
EXPECTED_ROSTER_SHA256 = "632601526021eb047056a83087ddc3690e22716cb864a137d7ba2446ad26c328"
EXPECTED_V1_SOURCE_SHA256 = "7e78e8208c82516d9ca58261ffa223737836387404bd0c9ae7fcf34fb1f6fd0f"
EXPECTED_V1_TEST_SHA256 = "0c7091ddb5310b3dcc1cd52946d117f3bf12ca9cda968c66ea098469d97406aa"
EXPECTED_V2_BREADTH_SOURCE_SHA256 = (
    "4c6035992a773a18c848f57c362e0dea24f28b7e4c0dcb244553bb1ca98141de"
)
EXPECTED_V2_BREADTH_TEST_SHA256 = "02d54c1b1b2ba0871ea8c751d06a8b14298a506c5e3e39ea1b933db072473f9c"
EXPECTED_V3_BREADTH_SOURCE_SHA256 = (
    "e8a4c3f846afc96e321a484b0b2c230bf5123851f390c2615bc9775655960ba6"
)
EXPECTED_V3_BREADTH_TEST_SHA256 = "f8d11131b9e0849e765d6e5fbb46491c98896197a7154bc8fb97d139e8318bf4"
EXPECTED_V1_ROW_SHA256S = (
    "4763d89f0c132006d3670ad562898d73a01b0ca8ad6b3b2fc4710fb433965b45",
    "5ad54b9cecb2a6f8e7d577147425491de31647eefab9d7e2e100b3cb03358275",
    "ec1fbb2086db0ef2a084c3cd6be973decf3e2d5dae2752341bc7e079cea55349",
    "ac8655e3f6090a721b9a67517d8358aefdf14859230be656395a99818cd7016b",
    "36d303800a53368998082269ba73641d51479929ba7ad538713f493e7207cb42",
    "f26980e8257fc7dc8424aa6be995747921c08475e9ea120ec8bdc8e1d1eabd4b",
)
EXPECTED_DEPENDENCY_SHA256 = "407f7928b18f46a94333030679c026110f7c5e0bd6404df0e4e728ac271bf988"
EXPECTED_PANEL_SHA256 = "25144b5d34f21f2be2dcbb826847e3491b83863a31f0ea5837332c33d111c1dc"
EXPECTED_RESULT_SHA256S = (
    "b62483402128311c466c442a507ff5983e0f105aeec5be95af3215bf1a1b1bbd",
    "132002386585d1754adfcc366143fc46dd0d0edf9d8ad673f47b1897c7dbdfc5",
    "7ba21721e7ed41e5500ea04421306137d7683a44367374eaed442aa35e158203",
    "66e21027c93de6f8d8e59eb3d8a29442c89aa4827bcf961eebe5ac908b115b94",
    "af66606d972d9845ba2073c719815aef3ea2ac056d6c52b1ebba2cca12087c92",
    "2fc19658d453261a5346a7ec4ca809dc1838695262b6b7a0d102355d5860222f",
)
EXPECTED_STATUSES = (
    breadth_v3.SELECTED,
    breadth_v3.STOP_AMBIGUOUS,
    breadth_v3.SELECTED,
    breadth_v3.STOP_AMBIGUOUS,
    breadth_v3.STOP_WORK_BOUND,
    breadth_v3.SELECTED,
)
EXPECTED_FRONTIER_SHA256S = (
    "76692dcc4e3c04af79e2f0b9cc36b40033c0c474854e136fada2472d43e20c71",
    "b9f123b7d379cbd8d74a04292e8c20c324894487014756cba8e45074be440c8e",
    "e54d250840039be07a29c45d2cc5710b4a316c66f19addb35bede6e24cb75612",
    "e9b9243e9f6a5e4b4b1529f64490ca2222d33fc1a4beab5f142a0076436817a4",
    "3c44d3a625ce26a7522d79326f879791073663b55a5c1a4d32f5282afe1b7654",
    "18f274cb77307b01a392edb7c6e68e06c68f1f301f46a79806aaaaddaa28f09b",
)
EXPECTED_ROOT_COUNTS = (60, 211, 7, 122, 171, 7)
EXPECTED_BUILD_WORK_SHA256S = (
    "362a49d206a54df41b800c914a16e91faa144ba01e30823552c210deccf2da11",
    "7e076a1dd67180947b5f517d6ae21c540f49c1010f73393da7300ac642b7c505",
    "91668cee4cb882ec7ca9c25db5f9c48c63d60a20a7b98466bd0efeba7b9abfb5",
    "7a754b531d9f36aacc9b780c933efd3c99168787fc5d4459088fecd901c970e0",
    "5eeeaf69e9389a13ad7683d395ad55b6f08e3d171f734f05a97607f806c59316",
    "91668cee4cb882ec7ca9c25db5f9c48c63d60a20a7b98466bd0efeba7b9abfb5",
)
EXPECTED_BUILD_RELATION_UNITS = (19_428, 28_973, 919, 25_762, 27_477, 919)
EXPECTED_BUILD_PREDICTIONS = (278, 379, 108, 1_743, 1_874, 108)
EXPECTED_BUILD_CACHE_HITS = (266, 537, 34, 6_303, 5_548, 34)
EXPECTED_BUILD_RACES = (954, 1_356, 241, 4_571, 5_023, 241)
EXPECTED_SELECTION_SHA256S = (
    "4213277d69d81b789c814435919270fd7cef466189753e3bd882fa0dd5f1bc6b",
    "3287e7fd27b8107536ca02c4dacf8747170fca5deda2fa90d76f4ad3af3bd05e",
    "c406404e5cf0d87072a1289f4c7160b92698b5b3dceb89f0888256e05671efb7",
    "07fa0acf1c1862031b9fbd977bcc40ec4544bcf93fe8a1b8f4d402c24497e110",
    "594663ef7c0284684ea79e8ccb1eef4777ab563e21039ba47039bc311680ec5e",
    "33d50629c37f3286b7d10eb924ca2f5c6e00b8083c62db1de5f9a5fd7a3938a9",
)
EXPECTED_SELECTION_RELATION_UNITS = (19_453, 29_009, 1_236, 29_758, 30_000, 1_283)
EXPECTED_SELECTION_RACES = (955, 1_357, 242, 4_572, 5_024, 242)
EXPECTED_SELECTION_PROOFS = (1, 2, 1, 10, 0, 1)
EXPECTED_SELECTED_SOURCES = (
    "def solve_86f0ac8ef86cc2ae(n):\n    return min(max(n, -2), 2)\n",
    None,
    (
        "def solve_e9123d138c4c0a05(text):\n"
        "    return ''.join((char_0 for char_0 in text if char_0 in "
        "'abcdefghijklmnopqrstuvwxyz'))\n"
    ),
    None,
    None,
    (
        "def solve_02dc6afe28e67f48(text):\n"
        "    return ''.join((char_0 for char_0 in text if char_0 in "
        "'abcdefghijklmnopqrstuvwxyz'))[::-1]\n"
    ),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_SOURCE_PATH = Path(v1.__file__).resolve()
V1_TEST_PATH = REPO_ROOT / "tests/test_packed_generic_typed_canonical_train_freeze_v1.py"
V2_BREADTH_SOURCE_PATH = Path(breadth_v2.__file__).resolve()
V2_BREADTH_TEST_PATH = REPO_ROOT / "tests/test_typed_semantic_packed_generic_train_breadth_v2.py"
V3_BREADTH_SOURCE_PATH = Path(breadth_v3.__file__).resolve()
V3_BREADTH_TEST_PATH = REPO_ROOT / "tests/test_typed_semantic_packed_generic_train_breadth_v3.py"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    digest = sha256()
    digest.update(b"packed-generic-typed-canonical-train-freeze-v2\0")
    digest.update(label.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(value))
    return digest.hexdigest()


class FrozenDependencies(NamedTuple):
    v1_commit: str
    v1_commit_full: str
    v1_panel_sha256: str
    v1_source_sha256: str
    v1_test_sha256: str
    v2_breadth_source_sha256: str
    v2_breadth_test_sha256: str
    v3_commit: str
    v3_commit_full: str
    v3_breadth_source_sha256: str
    v3_breadth_test_sha256: str
    roster_sha256: str
    work_caps: tuple[int, int, int]
    sha256: str


def _dependency_payload(value: FrozenDependencies) -> dict[str, object]:
    return {
        "v1_commit": value.v1_commit,
        "v1_commit_full": value.v1_commit_full,
        "v1_panel_sha256": value.v1_panel_sha256,
        "v1_source_sha256": value.v1_source_sha256,
        "v1_test_sha256": value.v1_test_sha256,
        "v2_breadth_source_sha256": value.v2_breadth_source_sha256,
        "v2_breadth_test_sha256": value.v2_breadth_test_sha256,
        "v3_commit": value.v3_commit,
        "v3_commit_full": value.v3_commit_full,
        "v3_breadth_source_sha256": value.v3_breadth_source_sha256,
        "v3_breadth_test_sha256": value.v3_breadth_test_sha256,
        "roster_sha256": value.roster_sha256,
        "work_caps": value.work_caps,
    }


def _authenticate_dependencies() -> FrozenDependencies:
    hashes = (
        sha256(V1_SOURCE_PATH.read_bytes()).hexdigest(),
        sha256(V1_TEST_PATH.read_bytes()).hexdigest(),
        sha256(V2_BREADTH_SOURCE_PATH.read_bytes()).hexdigest(),
        sha256(V2_BREADTH_TEST_PATH.read_bytes()).hexdigest(),
        sha256(V3_BREADTH_SOURCE_PATH.read_bytes()).hexdigest(),
        sha256(V3_BREADTH_TEST_PATH.read_bytes()).hexdigest(),
    )
    expected = (
        EXPECTED_V1_SOURCE_SHA256,
        EXPECTED_V1_TEST_SHA256,
        EXPECTED_V2_BREADTH_SOURCE_SHA256,
        EXPECTED_V2_BREADTH_TEST_SHA256,
        EXPECTED_V3_BREADTH_SOURCE_SHA256,
        EXPECTED_V3_BREADTH_TEST_SHA256,
    )
    if hashes != expected:
        raise RuntimeError("canonical V2 dependency bytes changed")
    if (
        not V1_COMMIT_FULL.startswith(V1_COMMIT)
        or not V3_COMMIT_FULL.startswith(V3_COMMIT)
        or v1.EXPECTED_ROSTER_SHA256 != EXPECTED_ROSTER_SHA256
        or v1.EXPECTED_BREADTH_SOURCE_SHA256 != EXPECTED_V2_BREADTH_SOURCE_SHA256
        or v1.EXPECTED_BREADTH_TEST_SHA256 != EXPECTED_V2_BREADTH_TEST_SHA256
    ):
        raise RuntimeError("canonical V2 banked dependency identity changed")
    work_caps = (
        breadth_v3.MAX_RECIPE_RELATION_WORK,
        breadth_v3.MAX_PHYSICAL_PREDICTIONS,
        breadth_v3.MAX_PHYSICAL_RACES,
    )
    if work_caps != (30_000, 30_000, 80_000) or work_caps != (
        breadth_v2.MAX_RECIPE_RELATION_WORK,
        breadth_v2.MAX_PHYSICAL_PREDICTIONS,
        breadth_v2.MAX_PHYSICAL_RACES,
    ):
        raise RuntimeError("canonical V2 packed work caps changed")
    partial = FrozenDependencies(
        v1_commit=V1_COMMIT,
        v1_commit_full=V1_COMMIT_FULL,
        v1_panel_sha256=EXPECTED_V1_PANEL_SHA256,
        v1_source_sha256=hashes[0],
        v1_test_sha256=hashes[1],
        v2_breadth_source_sha256=hashes[2],
        v2_breadth_test_sha256=hashes[3],
        v3_commit=V3_COMMIT,
        v3_commit_full=V3_COMMIT_FULL,
        v3_breadth_source_sha256=hashes[4],
        v3_breadth_test_sha256=hashes[5],
        roster_sha256=v1.EXPECTED_ROSTER_SHA256,
        work_caps=work_caps,
        sha256="",
    )
    frozen = partial._replace(sha256=_digest("frozen-dependencies", _dependency_payload(partial)))
    if frozen.sha256 != EXPECTED_DEPENDENCY_SHA256:
        raise RuntimeError("canonical V2 dependency root changed")
    return frozen


def _frozen_rows() -> tuple[v1.CanonicalTrainRow, ...]:
    rows = v1.load_canonical_train_rows()
    if (
        tuple(row.sha256 for row in rows) != EXPECTED_V1_ROW_SHA256S
        or tuple(row.task_id for row in rows) != v1.EXPECTED_TASK_IDS
        or tuple(row.visible_problem_sha256 for row in rows) != v1.EXPECTED_VISIBLE_PROBLEM_SHA256S
        or v1._roster_sha256(rows) != v1.EXPECTED_ROSTER_SHA256
    ):
        raise RuntimeError("canonical V1 row identities changed")
    return rows


def load_frozen_train_rows() -> tuple[v1.CanonicalTrainRow, ...]:
    """Authenticate dependencies and all six V1 rows without engine outcomes."""

    _authenticate_dependencies()
    return _frozen_rows()


class CanonicalTrainRowResult(NamedTuple):
    row_sha256: str
    task_id: str
    family: str
    status: str
    frontier_sha256: str
    root_eclass_ids: tuple[str, ...]
    build_work_sha256: str
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


def _run_row(row: v1.CanonicalTrainRow) -> CanonicalTrainRowResult:
    if row.sha256 not in EXPECTED_V1_ROW_SHA256S:
        raise RuntimeError("canonical V1 row changed after roster freeze")
    literals, max_cost = v1._signature_request(row.signature)
    frontier = breadth_v3.PackedGenericTypedTrainBreadthV3.build(
        row.signature,
        v1._engine_cases(row),
        literals,
        max_cost,
    )
    build_work = frontier.work
    build_events = dict(build_work.relation.event_counts)
    if (
        build_work.concrete_proof_witnesses != 0
        or build_work.materialized_witnesses != 0
        or build_events.get("frontier:desired-value-read", 0) != 0
        or build_events.get(
            "operation:workset-materialize-selected-lazy-proof-plan",
            0,
        )
        != 0
        or build_events.get("operation:workset-materialize-selected-proof", 0) != 0
        or build_events.get("frontier:materialize-witness", 0) != 0
    ):
        raise RuntimeError("canonical packed V3 build observed desired/proof work")

    selection = None
    status = frontier.status
    if frontier.status == breadth_v3.FRONTIER_READY:
        selection = frontier.select(v1._engine_desired(row))
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
        selection_proofs=0 if selection is None else selection.exact_proof_count,
        selected_source=selected_source,
        sha256="",
    )
    return partial._replace(sha256=_digest("canonical-row-result", _result_payload(partial)))


class CanonicalTrainPanel(NamedTuple):
    schema: str
    verdict: str
    dependency_sha256: str
    roster_sha256: str
    row_sha256s: tuple[str, ...]
    results: tuple[CanonicalTrainRowResult, ...]
    selected_rows: int
    ambiguous_rows: int
    stopped_rows: int
    sha256: str


def _panel_payload(panel: CanonicalTrainPanel) -> dict[str, object]:
    return {
        "schema": panel.schema,
        "verdict": panel.verdict,
        "dependency_sha256": panel.dependency_sha256,
        "roster_sha256": panel.roster_sha256,
        "row_sha256s": panel.row_sha256s,
        "results": tuple(result.sha256 for result in panel.results),
        "selected_rows": panel.selected_rows,
        "ambiguous_rows": panel.ambiguous_rows,
        "stopped_rows": panel.stopped_rows,
    }


def _assert_spent_panel(panel: CanonicalTrainPanel) -> None:
    results = panel.results
    if (
        panel.verdict != STOP_VERDICT
        or panel.dependency_sha256 != EXPECTED_DEPENDENCY_SHA256
        or panel.roster_sha256 != EXPECTED_ROSTER_SHA256
        or panel.row_sha256s != EXPECTED_V1_ROW_SHA256S
        or panel.selected_rows != 3
        or panel.ambiguous_rows != 2
        or panel.stopped_rows != 3
        or panel.sha256 != EXPECTED_PANEL_SHA256
        or tuple(result.sha256 for result in results) != EXPECTED_RESULT_SHA256S
        or tuple(result.status for result in results) != EXPECTED_STATUSES
        or tuple(result.frontier_sha256 for result in results) != EXPECTED_FRONTIER_SHA256S
        or tuple(len(result.root_eclass_ids) for result in results) != EXPECTED_ROOT_COUNTS
        or tuple(result.build_work_sha256 for result in results) != EXPECTED_BUILD_WORK_SHA256S
        or tuple(result.build_relation_units for result in results) != EXPECTED_BUILD_RELATION_UNITS
        or tuple(result.build_predictions for result in results) != EXPECTED_BUILD_PREDICTIONS
        or tuple(result.build_cache_hits for result in results) != EXPECTED_BUILD_CACHE_HITS
        or tuple(result.build_races for result in results) != EXPECTED_BUILD_RACES
        or any(result.build_proofs != 0 for result in results)
        or tuple(result.selection_sha256 for result in results) != EXPECTED_SELECTION_SHA256S
        or tuple(result.selection_relation_units for result in results)
        != EXPECTED_SELECTION_RELATION_UNITS
        or tuple(result.selection_races for result in results) != EXPECTED_SELECTION_RACES
        or tuple(result.selection_proofs for result in results) != EXPECTED_SELECTION_PROOFS
        or tuple(result.selected_source for result in results) != EXPECTED_SELECTED_SOURCES
    ):
        raise RuntimeError("spent canonical V2 panel outcome changed")


def run_canonical_train_freeze() -> CanonicalTrainPanel:
    """Spend the one fixed six-row V2 panel without substitution or retuning."""

    dependencies = _authenticate_dependencies()
    rows = _frozen_rows()
    results = tuple(_run_row(row) for row in rows)
    selected = sum(result.status == breadth_v3.SELECTED for result in results)
    ambiguous = sum(result.status == breadth_v3.STOP_AMBIGUOUS for result in results)
    stopped = len(results) - selected
    verdict = PASS_VERDICT if selected == len(results) else STOP_VERDICT
    partial = CanonicalTrainPanel(
        schema=SCHEMA,
        verdict=verdict,
        dependency_sha256=dependencies.sha256,
        roster_sha256=v1._roster_sha256(rows),
        row_sha256s=tuple(row.sha256 for row in rows),
        results=results,
        selected_rows=selected,
        ambiguous_rows=ambiguous,
        stopped_rows=stopped,
        sha256="",
    )
    panel = partial._replace(sha256=_digest("canonical-panel", _panel_payload(partial)))
    _assert_spent_panel(panel)
    return panel


__all__ = [
    "EXPECTED_PANEL_SHA256",
    "EXPECTED_RESULT_SHA256S",
    "EXPECTED_STATUSES",
    "EXPECTED_V1_ROW_SHA256S",
    "PASS_VERDICT",
    "SCHEMA",
    "STOP_VERDICT",
    "V1_COMMIT",
    "V1_COMMIT_FULL",
    "V3_COMMIT",
    "V3_COMMIT_FULL",
    "CanonicalTrainPanel",
    "CanonicalTrainRowResult",
    "FrozenDependencies",
    "load_frozen_train_rows",
    "run_canonical_train_freeze",
]
