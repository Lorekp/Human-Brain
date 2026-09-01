from __future__ import annotations

import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import FrozenInstanceError, asdict, fields, replace
from hashlib import sha256
from inspect import signature
from threading import Barrier

import pytest
from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from experiments.r3_pilot import common_rust_action_intervention_v1 as interventions
from humanbrain import rust_selected_action_executor as rust_executor

DOUBLE_SOURCE = """\
fn double_value(value: i64) -> i64 {
    value * 2
}

pub fn solve(values: &[i64]) -> Vec<i64> {
    values.iter().map(|value| double_value(*value)).collect()
}
"""

NONNEGATIVE_SOURCE = """\
pub fn solve(values: &[i64]) -> Vec<i64> {
    values.iter().copied().filter(|value| *value >= 0).collect()
}
"""


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _candidate(
    candidate_id: str,
    symbol: str,
) -> knowledge_action.KnowledgeActionCandidateProjectionV1:
    return knowledge_action.project_knowledge_action_candidate_v1(
        candidate_id,
        f"def solve(values):\n    return tools.{symbol}(values)\n",
    )


def _catalogue():
    double = _candidate("slot_q7", "kavun")
    nonnegative = _candidate("slot_z4", "pelto")
    sources = (
        (double, DOUBLE_SOURCE),
        (nonnegative, NONNEGATIVE_SOURCE),
    )
    return double, nonnegative, sources


def test_prepare_binds_one_opaque_train_catalogue_without_task_or_outcome_state() -> None:
    double, nonnegative, sources = _catalogue()

    prepared = interventions.prepare_rust_action_intervention_v1(tuple(reversed(sources)))
    equivalent = interventions.prepare_rust_action_intervention_v1(sources)

    assert type(prepared) is interventions.PreparedRustActionInterventionV1
    assert type(prepared.catalogue) is interventions.RustActionInterventionCatalogueV1
    assert type(prepared.authority) is interventions.RustActionInterventionAuthorityV1
    assert prepared.catalogue.split == prepared.authority.split == interventions.SPLIT
    assert prepared.catalogue.action_count == 2
    assert prepared.candidate_sha256s == tuple(sorted((double.sha256, nonnegative.sha256)))
    assert prepared.catalogue_sha256 == equivalent.catalogue_sha256
    assert prepared.catalogue == equivalent.catalogue
    assert prepared.authority_sha256 != equivalent.authority_sha256
    assert prepared.authority.catalogue_sha256 == prepared.catalogue_sha256
    assert prepared.catalogue.rust_source_sha256s == tuple(
        sha256(source.encode("ascii")).hexdigest()
        for _candidate_projection, source in sorted(
            sources,
            key=lambda pair: pair[0].sha256,
        )
    )
    assert prepared.work_counters == interventions.RustActionInterventionWorkCountersV1(
        issued_selection_count=0,
        spent_selection_count=0,
        source_release_count=0,
        execution_attempt_count=0,
        successful_execution_count=0,
    )
    assert DOUBLE_SOURCE not in repr(prepared.catalogue)
    assert NONNEGATIVE_SOURCE not in repr(prepared.authority)
    public_field_names = {
        field.name
        for record_type in (
            interventions.RustActionInterventionCatalogueV1,
            interventions.RustActionInterventionAuthorityV1,
            interventions.RustActionInterventionSelectionV1,
        )
        for field in fields(record_type)
    }
    assert not {"task", "task_text", "expected_outputs", "memory", "memory_state"}.intersection(
        public_field_names
    )
    assert tuple(signature(interventions.prepare_rust_action_intervention_v1).parameters) == (
        "candidate_sources",
    )
    assert tuple(signature(prepared.issue).parameters) == (
        "candidate_sha256",
        "intervention_sha256",
    )
    assert tuple(signature(prepared.execute_selected).parameters) == (
        "selection",
        "batch",
    )

    with pytest.raises(FrozenInstanceError):
        prepared.catalogue.action_count = 3  # type: ignore[misc]


def test_prepare_rejects_inexact_unbounded_duplicate_and_non_ascii_catalogues() -> None:
    double, nonnegative, sources = _catalogue()
    third = _candidate("slot_m2", "zimra")
    fourth = _candidate("slot_r5", "noxel")
    fifth = _candidate("slot_x8", "tavik")
    generated = lambda value: (  # noqa: E731
        f"pub fn solve(values: &[i64]) -> Vec<i64> {{ "
        f"values.iter().map(|item| *item + {value}).collect() }}\n"
    )

    with pytest.raises(TypeError, match="exact tuple"):
        interventions.prepare_rust_action_intervention_v1(list(sources))  # type: ignore[arg-type]
    with pytest.raises(interventions.RustActionInterventionError, match="2 to 4"):
        interventions.prepare_rust_action_intervention_v1((sources[0],))
    with pytest.raises(interventions.RustActionInterventionError, match="2 to 4"):
        interventions.prepare_rust_action_intervention_v1(
            (
                *sources,
                (third, generated(3)),
                (fourth, generated(4)),
                (fifth, generated(5)),
            )
        )
    with pytest.raises(interventions.RustActionInterventionError, match="duplicated"):
        interventions.prepare_rust_action_intervention_v1(
            ((double, DOUBLE_SOURCE), (double, NONNEGATIVE_SOURCE))
        )
    with pytest.raises(interventions.RustActionInterventionError, match="source is duplicated"):
        interventions.prepare_rust_action_intervention_v1(
            ((double, DOUBLE_SOURCE), (nonnegative, DOUBLE_SOURCE))
        )
    with pytest.raises(interventions.RustActionInterventionError, match="ASCII"):
        interventions.prepare_rust_action_intervention_v1(
            ((double, DOUBLE_SOURCE), (nonnegative, f"{NONNEGATIVE_SOURCE}// ©"))
        )
    with pytest.raises(knowledge_action.KnowledgeActionMemoryError):
        interventions.prepare_rust_action_intervention_v1(
            (
                (replace(double, sha256="0" * 64), DOUBLE_SOURCE),
                (nonnegative, NONNEGATIVE_SOURCE),
            )
        )


def test_issue_is_unique_immutable_and_rejects_forged_or_foreign_selection_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    double, nonnegative, sources = _catalogue()
    prepared = interventions.prepare_rust_action_intervention_v1(sources)
    selection = prepared.issue(double.sha256, _root("intervention:one"))

    selection.authenticate_structure()
    assert selection.split == interventions.SPLIT
    assert selection.authority_sha256 == prepared.authority_sha256
    assert selection.catalogue_sha256 == prepared.catalogue_sha256
    assert selection.candidate_sha256 == double.sha256
    assert prepared.work_counters.issued_selection_count == 1
    with pytest.raises(FrozenInstanceError):
        selection.candidate_sha256 = nonnegative.sha256  # type: ignore[misc]
    with pytest.raises(interventions.RustActionInterventionError, match="already issued"):
        prepared.issue(nonnegative.sha256, selection.intervention_sha256)
    with pytest.raises(interventions.RustActionInterventionError, match="not in"):
        prepared.issue(_root("foreign-candidate"), _root("intervention:foreign-candidate"))

    partial = replace(
        selection,
        candidate_sha256=nonnegative.sha256,
        sha256="",
    )
    payload = asdict(partial)
    payload.pop("sha256")
    forged = replace(
        partial,
        sha256=interventions._digest(interventions._SELECTION_NAMESPACE, payload),
    )
    forged.authenticate_structure()
    bad_root = replace(selection, sha256="0" * 64)
    other = interventions.prepare_rust_action_intervention_v1(sources)
    foreign = other.issue(double.sha256, _root("intervention:other-authority"))

    prepare_calls = 0

    def forbidden_prepare(_authority_sha256, _batch):
        nonlocal prepare_calls
        prepare_calls += 1
        raise AssertionError("invalid selections must fail before execution preparation")

    monkeypatch.setattr(interventions, "_prepare_execution", forbidden_prepare)
    with pytest.raises(interventions.RustActionInterventionError, match="authentication failed"):
        prepared.execute_selected(bad_root, ((1,),))
    with pytest.raises(interventions.RustActionInterventionError, match="not issued"):
        prepared.execute_selected(forged, ((1,),))
    with pytest.raises(interventions.RustActionInterventionError, match="different"):
        prepared.execute_selected(foreign, ((1,),))

    assert prepare_calls == 0
    assert prepared.work_counters == interventions.RustActionInterventionWorkCountersV1(
        issued_selection_count=1,
        spent_selection_count=0,
        source_release_count=0,
        execution_attempt_count=0,
        successful_execution_count=0,
    )


def test_selection_is_spent_and_release_counted_before_source_factory_cannot_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    double, _nonnegative, sources = _catalogue()
    prepared = interventions.prepare_rust_action_intervention_v1(sources)
    selection = prepared.issue(double.sha256, _root("intervention:seal-failure"))
    source_factory_calls = 0

    monkeypatch.setattr(interventions, "_prepare_execution", lambda _root, _batch: object())

    def unavailable_source_factory(**_fields):
        nonlocal source_factory_calls
        source_factory_calls += 1
        assert prepared.work_counters.spent_selection_count == 1
        assert prepared.work_counters.source_release_count == 1
        assert prepared.work_counters.execution_attempt_count == 0
        raise RuntimeError("synthetic source factory failure")

    monkeypatch.setattr(
        rust_executor,
        "make_rust_selected_source_v2",
        unavailable_source_factory,
    )
    with pytest.raises(RuntimeError, match="synthetic source factory failure"):
        prepared.execute_selected(selection, ((1,),))

    assert source_factory_calls == 1
    assert prepared.work_counters == interventions.RustActionInterventionWorkCountersV1(
        issued_selection_count=1,
        spent_selection_count=1,
        source_release_count=1,
        execution_attempt_count=0,
        successful_execution_count=0,
    )

    def forbidden_prepare(_authority_sha256, _batch):
        raise AssertionError("a spent selection must fail before execution preparation")

    monkeypatch.setattr(interventions, "_prepare_execution", forbidden_prepare)
    with pytest.raises(interventions.RustActionInterventionError, match="already consumed"):
        prepared.execute_selected(selection, ((1,),))
    assert source_factory_calls == 1


def test_concurrent_selection_loser_is_rejected_before_single_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    double, _nonnegative, sources = _catalogue()
    prepared = interventions.prepare_rust_action_intervention_v1(sources)
    selection = prepared.issue(double.sha256, _root("intervention:concurrent"))
    callers_start = Barrier(3)
    preparation_started = Barrier(2)
    release_preparation = Barrier(2)
    prepare_calls = 0

    def blocked_prepare(_authority_sha256, _batch):
        nonlocal prepare_calls
        prepare_calls += 1
        preparation_started.wait(timeout=5)
        release_preparation.wait(timeout=5)
        raise RuntimeError("synthetic preparation failure")

    def forbidden_source_factory(**_fields):
        raise AssertionError("failed preparation must not release source")

    def invoke():
        callers_start.wait(timeout=5)
        try:
            prepared.execute_selected(selection, ((1,),))
        except BaseException as exc:
            return exc
        raise AssertionError("synthetic preparation cannot succeed")

    monkeypatch.setattr(interventions, "_prepare_execution", blocked_prepare)
    monkeypatch.setattr(
        rust_executor,
        "make_rust_selected_source_v2",
        forbidden_source_factory,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(pool.submit(invoke) for _index in range(2))
        callers_start.wait(timeout=5)
        preparation_started.wait(timeout=5)
        completed, pending = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
        assert len(completed) == 1
        assert len(pending) == 1
        loser = next(iter(completed)).result()
        assert type(loser) is interventions.RustActionInterventionError
        assert "already consumed" in str(loser)
        release_preparation.wait(timeout=5)
        outcomes = tuple(future.result(timeout=5) for future in futures)

    assert prepare_calls == 1
    assert sum(type(outcome) is RuntimeError for outcome in outcomes) == 1
    assert (
        sum(type(outcome) is interventions.RustActionInterventionError for outcome in outcomes) == 1
    )
    assert prepared.work_counters == interventions.RustActionInterventionWorkCountersV1(
        issued_selection_count=1,
        spent_selection_count=1,
        source_release_count=0,
        execution_attempt_count=0,
        successful_execution_count=0,
    )


@pytest.mark.skipif(
    os.name != "nt" or shutil.which("rustc") is None,
    reason="the real V2 execution requires Windows and rustc",
)
def test_execute_selected_compiles_and_runs_once_with_exact_v2_receipts() -> None:
    double, _nonnegative, sources = _catalogue()
    prepared = interventions.prepare_rust_action_intervention_v1(sources)
    selection = prepared.issue(double.sha256, _root("intervention:real-execution"))
    batch = ((), (1, -2, 3), (9,))

    result = prepared.execute_selected(selection, batch)

    assert type(result) is rust_executor.RustSelectedActionExecutionV2
    result.authenticate_structure()
    assert result.outputs == ((), (2, -4, 6), (18,))
    assert result.selected_source.authority_sha256 == prepared.authority_sha256
    assert result.selected_source.selection_sha256 == selection.sha256
    assert result.selected_source.action_handle == double.candidate_id
    assert result.selected_source.source_sha256 == sha256(DOUBLE_SOURCE.encode("ascii")).hexdigest()
    assert result.selected_source.source == DOUBLE_SOURCE
    assert type(result.preparation) is rust_executor.RustSelectedSourcePreparationV2
    assert type(result.dependency_receipt) is rust_executor.RustSelectedSourceDependencyReceiptV2
    assert type(result.compile_receipt) is rust_executor.RustSelectedSourceCompileReceiptV2
    assert type(result.run_receipt) is rust_executor.RustSelectedSourceRunReceiptV2
    assert result.compile_receipt.compile_invocations == 1
    assert result.run_receipt.run_invocations == 1
    assert prepared.work_counters == interventions.RustActionInterventionWorkCountersV1(
        issued_selection_count=1,
        spent_selection_count=1,
        source_release_count=1,
        execution_attempt_count=1,
        successful_execution_count=1,
    )

    with pytest.raises(interventions.RustActionInterventionError, match="already consumed"):
        prepared.execute_selected(selection, batch)
    assert prepared.work_counters.successful_execution_count == 1
