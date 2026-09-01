from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
from inspect import signature
from pathlib import Path
from threading import Barrier

import pytest
from experiments.r3_pilot import common_knowledge_action_memory_v1 as action_memory
from humanbrain import rust_selected_action_executor as executor
from humanbrain.consolidated_knowledge_cell_projection import (
    project_consolidated_knowledge_to_cells_v1,
)
from humanbrain.knowledge_consolidation import consolidate_knowledge_corpus_v1
from humanbrain.knowledge_corpus import (
    BenchmarkDenyManifest,
    KnowledgeCorpusCaps,
    KnowledgeCorpusManifest,
    KnowledgeFileSpec,
    KnowledgeSourceSpec,
    build_local_knowledge_corpus,
)
from humanbrain.rust_action_registry import (
    PreparedRustActionRegistryV1,
    RustActionRegistryError,
    prepare_rust_action_registry_v1,
)

pytestmark = pytest.mark.skipif(
    os.name != "nt" or shutil.which("rustc") is None or shutil.which("rustfmt") is None,
    reason="the v1 executor requires Windows, rustc, and rustfmt",
)

_BATCH = ((), (1, -2, 3), (9,))
_EMPTY_SHA256 = sha256(b"").hexdigest()

_ALTERNATE_SOURCE = """\
fn alternate_transform(value: i64) -> i64 {
    value + 7
}

pub fn solve(values: &[i64]) -> Vec<i64> {
    values.iter().map(|value| alternate_transform(*value)).collect()
}
"""

_SUCCESS_SOURCE = """\
fn primary_transform(value: i64) -> i64 {
    value * 2
}

pub fn solve(values: &[i64]) -> Vec<i64> {
    values.iter().map(|value| primary_transform(*value)).collect()
}
"""

_COMPILE_FAILURE_SOURCE = """\
fn primary_transform(values: &[i64]) -> Vec<i64> {
    missing_transform(values)
}

pub fn solve(values: &[i64]) -> Vec<i64> {
    primary_transform(values)
}
"""

_TIMEOUT_SOURCE = """\
fn primary_transform(values: &[i64]) -> Vec<i64> {
    std::thread::sleep(std::time::Duration::from_secs(10));
    values.to_vec()
}

pub fn solve(values: &[i64]) -> Vec<i64> {
    primary_transform(values)
}
"""

_AGGREGATE_OUTPUT_SOURCE = """\
use std::io::{self, Write as _};

fn primary_transform(values: &[i64]) -> Vec<i64> {
    let output = vec![b'x'; 40 * 1024];
    io::stdout().write_all(&output).expect("stdout write failed");
    io::stderr().write_all(&output).expect("stderr write failed");
    values.to_vec()
}

pub fn solve(values: &[i64]) -> Vec<i64> {
    primary_transform(values)
}
"""

_PROTOCOL_FAILURE_SOURCE = """\
fn primary_transform(values: &[i64]) -> Vec<i64> {
    println!("synthetic protocol noise");
    values.to_vec()
}

pub fn solve(values: &[i64]) -> Vec<i64> {
    primary_transform(values)
}
"""


def _build_registry_and_recall(
    root: Path,
    selected_source: str,
    *,
    cell_gate: bool = True,
) -> tuple[PreparedRustActionRegistryV1, action_memory.KnowledgeActionRecallV1]:
    source_root = root / "corpus"
    source_root.mkdir(parents=True)
    primary = b"primary transform input first\n\nprimary transform input second\n"
    alternate = b"alternate transform input first\n\nalternate transform input second\n"
    (source_root / "primary.txt").write_bytes(primary)
    (source_root / "alternate.txt").write_bytes(alternate)
    source_id = "synthetic-rust-executor"
    manifest = KnowledgeCorpusManifest(
        sources=(
            KnowledgeSourceSpec(
                source_id=source_id,
                canonical_url="https://docs.example.org/synthetic-rust-executor/",
                upstream_revision="executor-fixture-v1",
                license_spdx="MIT",
                license_evidence_url="https://docs.example.org/license",
                attribution="Synthetic Rust executor fixture",
                provenance_confidence=1.0,
                files=(
                    KnowledgeFileSpec(
                        path="primary.txt",
                        expected_raw_sha256=sha256(primary).hexdigest(),
                        split="acquire",
                        content_type="plain-text",
                        chapter="primary-transform",
                        symbol="synthetic.primary",
                    ),
                    KnowledgeFileSpec(
                        path="alternate.txt",
                        expected_raw_sha256=sha256(alternate).hexdigest(),
                        split="acquire",
                        content_type="plain-text",
                        chapter="alternate-transform",
                        symbol="synthetic.alternate",
                    ),
                ),
            ),
        )
    )
    corpus = build_local_knowledge_corpus(
        manifest,
        {source_id: source_root},
        BenchmarkDenyManifest(),
        KnowledgeCorpusCaps(
            max_bytes_per_source=16_384,
            max_total_bytes=16_384,
            max_tokenish_per_source=2_048,
            max_total_tokenish=2_048,
        ),
    )
    consolidation = consolidate_knowledge_corpus_v1(corpus, split="acquire")
    projection = project_consolidated_knowledge_to_cells_v1(
        consolidation,
        corpus=corpus,
    )
    registry = prepare_rust_action_registry_v1(
        corpus,
        consolidation,
        sources=(selected_source, _ALTERNATE_SOURCE),
    )
    memory = registry.fit_memory(projection)
    recall = registry.recall(
        memory,
        action_memory.project_knowledge_action_task_v1("primary transform"),
        cell_gate=cell_gate,
    )
    expected_status = action_memory.RECALLED if cell_gate else action_memory.STOP_CELL_OFF
    assert recall.status == expected_status
    return registry, recall


@pytest.fixture(scope="module")
def successful_execution(
    tmp_path_factory: pytest.TempPathFactory,
):
    registry, recall = _build_registry_and_recall(
        tmp_path_factory.mktemp("rust-selected-success"),
        _SUCCESS_SOURCE,
    )
    prepared = executor.prepare_rust_selected_action_executor_v1(registry, _BATCH)
    result = prepared.execute_selected_once(recall)
    return result, registry


def _resign(value: object, namespace: bytes):
    payload = asdict(value)  # type: ignore[arg-type]
    payload.pop("sha256")
    return replace(  # type: ignore[call-overload]
        value,
        sha256=executor._digest(namespace, payload),
    )


def _replace_and_resign_execution(result, **changes):  # type: ignore[no-untyped-def]
    changed = replace(result, **changes)
    payload = {
        "materialization_sha256": changed.materialization.sha256,
        "rustc_identity_sha256": changed.rustc_identity.sha256,
        "preparation_sha256": changed.preparation.sha256,
        "dependency_receipt_sha256": changed.dependency_receipt.sha256,
        "compile_receipt_sha256": changed.compile_receipt.sha256,
        "run_receipt_sha256": changed.run_receipt.sha256,
        "outputs": changed.outputs,
    }
    return replace(
        changed,
        sha256=executor._digest(executor._EXECUTION_NAMESPACE, payload),
    )


def test_registry_recall_compiles_only_selected_source_and_runs_frozen_batch(
    successful_execution,
) -> None:
    result, registry = successful_execution

    result.authenticate_structure()
    assert result.outputs == ((), (2, -4, 6), (18,))
    assert result.materialization.source == _SUCCESS_SOURCE
    assert result.preparation.expected_registry_sha256 == registry.receipt.sha256
    assert result.preparation.rustc_identity_sha256 == result.rustc_identity.sha256
    assert result.preparation.case_count == len(_BATCH)
    assert result.run_receipt.case_count == len(_BATCH) == len(result.outputs)
    assert result.compile_receipt.edition == "2021"
    assert result.compile_receipt.compile_invocations == 1
    assert result.run_receipt.run_invocations == 1
    assert 0 < result.compile_receipt.binary_size_bytes <= executor.MAX_BINARY_BYTES
    assert result.dependency_receipt.dependency_paths == (
        executor.HARNESS_FILENAME,
        executor.SELECTED_FILENAME,
    )
    assert result.dependency_receipt.dependency_content_sha256s == (
        (executor.HARNESS_FILENAME, executor.HARNESS_SHA256),
        (executor.SELECTED_FILENAME, result.materialization.source_sha256),
    )
    assert result.compile_receipt.stdout_sha256 == _EMPTY_SHA256
    assert result.compile_receipt.stderr_sha256 == _EMPTY_SHA256
    assert result.run_receipt.stderr_sha256 == _EMPTY_SHA256
    assert (
        result.run_receipt.stdout_sha256
        == sha256(executor._encode_batch(result.outputs)).hexdigest()
    )


def test_v2_registry_authority_runs_selected_source_with_v2_receipts(
    tmp_path: Path,
) -> None:
    registry, recall = _build_registry_and_recall(tmp_path / "v2-success", _SUCCESS_SOURCE)
    authority = executor.adapt_rust_action_registry_authority_v2(registry)
    prepared = executor.prepare_rust_selected_action_executor_v2(authority, _BATCH)

    result = prepared.execute_selected_once(recall)

    result.authenticate_structure()
    assert isinstance(result, executor.RustSelectedActionExecutionV2)
    assert result.outputs == ((), (2, -4, 6), (18,))
    assert result.selected_source.source == _SUCCESS_SOURCE
    assert (
        result.selected_source.source_sha256 == sha256(_SUCCESS_SOURCE.encode("ascii")).hexdigest()
    )
    assert result.selected_source.selection_sha256 == recall.sha256
    assert result.selected_source.action_handle == recall.selected_candidate_id
    assert (
        result.selected_source.authority_sha256
        == authority.authority_sha256
        == result.preparation.expected_authority_sha256
    )
    assert result.preparation.schema == executor.V2_SCHEMA
    assert result.dependency_receipt.schema == executor.V2_SCHEMA
    assert result.compile_receipt.schema == executor.V2_SCHEMA
    assert result.run_receipt.schema == executor.V2_SCHEMA
    assert type(result.preparation) is executor.RustSelectedSourcePreparationV2
    assert type(result.dependency_receipt) is executor.RustSelectedSourceDependencyReceiptV2
    assert type(result.compile_receipt) is executor.RustSelectedSourceCompileReceiptV2
    assert type(result.run_receipt) is executor.RustSelectedSourceRunReceiptV2
    assert result.compile_receipt.compile_invocations == 1
    assert result.run_receipt.run_invocations == 1


def test_v2_selected_source_accepts_full_candidate_identifier_contract() -> None:
    root = sha256(b"v2-handle-contract").hexdigest()
    selected = executor.make_rust_selected_source_v2(
        authority_sha256=root,
        selection_sha256=root,
        release_sha256=root,
        action_handle="Map-Action",
        source=_SUCCESS_SOURCE,
    )

    selected.authenticate_structure()
    assert selected.action_handle == "Map-Action"


def test_v2_equivalent_registries_have_distinct_authority_custody(
    tmp_path: Path,
) -> None:
    first_registry, first_recall = _build_registry_and_recall(
        tmp_path / "v2-instance-first",
        _SUCCESS_SOURCE,
    )
    second_registry, second_recall = _build_registry_and_recall(
        tmp_path / "v2-instance-second",
        _SUCCESS_SOURCE,
    )
    first = executor.adapt_rust_action_registry_authority_v2(first_registry)
    second = executor.adapt_rust_action_registry_authority_v2(second_registry)

    first_source = first.execute_selected(first_recall, lambda value: value)
    second_source = second.execute_selected(second_recall, lambda value: value)

    assert first_registry.receipt == second_registry.receipt
    assert first_recall == second_recall
    assert first.receipt.registry_receipt_sha256 == first_registry.receipt.sha256
    assert second.receipt.registry_receipt_sha256 == second_registry.receipt.sha256
    assert first.authority_sha256 != second.authority_sha256
    assert first_source.authority_sha256 == first.authority_sha256
    assert second_source.authority_sha256 == second.authority_sha256
    assert first_source.release_sha256 == second_source.release_sha256
    assert first_source.sha256 != second_source.sha256


def test_v2_registry_adapter_rejects_spliced_grant_translation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, recall = _build_registry_and_recall(
        tmp_path / "v2-spliced-grant",
        _SUCCESS_SOURCE,
    )
    authority = executor.adapt_rust_action_registry_authority_v2(registry)
    seal = executor.make_rust_selected_source_v2
    callback_count = 0

    def splice(**fields):
        return seal(
            authority_sha256=fields["authority_sha256"],
            selection_sha256=fields["selection_sha256"],
            release_sha256="f" * 64,
            action_handle="Spliced-Handle",
            source=fields["source"],
        )

    def callback(_selected):
        nonlocal callback_count
        callback_count += 1

    monkeypatch.setattr(executor, "make_rust_selected_source_v2", splice)
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="registry selected-source authority changed",
    ):
        authority.execute_selected(recall, callback)

    assert callback_count == 0


@pytest.mark.parametrize("replacement", ("invalid", "inexact"))
def test_v2_registry_adapter_authenticates_exact_grant_before_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    registry, recall = _build_registry_and_recall(
        tmp_path / f"v2-{replacement}-grant",
        _SUCCESS_SOURCE,
    )
    authority = executor.adapt_rust_action_registry_authority_v2(registry)
    seal = executor.make_rust_selected_source_v2
    callback_count = 0

    def substitute(**fields):
        grant = seal(**fields)
        if replacement == "invalid":
            return replace(grant, sha256="0" * 64)
        return object()

    def callback(_selected):
        nonlocal callback_count
        callback_count += 1

    monkeypatch.setattr(executor, "make_rust_selected_source_v2", substitute)
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match=r"authentication failed|inexact grant",
    ):
        authority.execute_selected(recall, callback)

    assert callback_count == 0


def test_v2_registry_authority_releases_once_across_two_executors(
    tmp_path: Path,
) -> None:
    registry, recall = _build_registry_and_recall(tmp_path / "v2-shared", _SUCCESS_SOURCE)
    authority = executor.adapt_rust_action_registry_authority_v2(registry)
    prepared = (
        executor.prepare_rust_selected_action_executor_v2(authority, _BATCH),
        executor.prepare_rust_selected_action_executor_v2(authority, _BATCH),
    )
    start = Barrier(2)

    def invoke(index: int):
        start.wait()
        try:
            return prepared[index].execute_selected_once(recall)
        except BaseException as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(invoke, range(2)))

    results = [
        value for value in outcomes if isinstance(value, executor.RustSelectedActionExecutionV2)
    ]
    errors = [value for value in outcomes if isinstance(value, BaseException)]
    assert len(results) == 1
    assert results[0].selected_source.selection_sha256 == recall.sha256
    assert results[0].compile_receipt.compile_invocations == 1
    assert results[0].run_receipt.run_invocations == 1
    assert len(errors) == 1
    assert "release failed" in str(errors[0])


def test_v2_compile_failure_spends_executor_and_shared_registry_authority(
    tmp_path: Path,
) -> None:
    registry, recall = _build_registry_and_recall(
        tmp_path / "v2-compile-failure",
        _COMPILE_FAILURE_SOURCE,
    )
    authority = executor.adapt_rust_action_registry_authority_v2(registry)
    failed = executor.prepare_rust_selected_action_executor_v2(authority, _BATCH)
    retry = executor.prepare_rust_selected_action_executor_v2(authority, _BATCH)

    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="failed compilation",
    ):
        failed.execute_selected_once(recall)
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="prepared Rust executor was consumed",
    ):
        failed.execute_selected_once(recall)
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="registry selected-source release failed",
    ):
        retry.execute_selected_once(recall)
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="prepared Rust executor was consumed",
    ):
        retry.execute_selected_once(recall)


def test_v2_malformed_and_stop_selections_fail_before_compilation(
    tmp_path: Path,
) -> None:
    malformed_registry, _recall = _build_registry_and_recall(
        tmp_path / "v2-malformed",
        _COMPILE_FAILURE_SOURCE,
    )
    malformed_authority = executor.adapt_rust_action_registry_authority_v2(malformed_registry)
    malformed = executor.prepare_rust_selected_action_executor_v2(
        malformed_authority,
        _BATCH,
    )

    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="selection root is invalid",
    ) as malformed_error:
        malformed.execute_selected_once(object())
    assert "failed compilation" not in str(malformed_error.value)
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="prepared Rust executor was consumed",
    ):
        malformed.execute_selected_once(object())

    stop_registry, stop_recall = _build_registry_and_recall(
        tmp_path / "v2-stop",
        _COMPILE_FAILURE_SOURCE,
        cell_gate=False,
    )
    stop_authority = executor.adapt_rust_action_registry_authority_v2(stop_registry)
    stopped = executor.prepare_rust_selected_action_executor_v2(stop_authority, _BATCH)

    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="registry selected-source release failed",
    ) as stop_error:
        stopped.execute_selected_once(stop_recall)
    assert "failed compilation" not in str(stop_error.value)
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="prepared Rust executor was consumed",
    ):
        stopped.execute_selected_once(stop_recall)


def test_public_authority_seam_is_factory_owned_and_consumes_before_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    successful_execution,
) -> None:
    result, completed_registry = successful_execution
    registry, recall = _build_registry_and_recall(tmp_path / "authority", _SUCCESS_SOURCE)
    prepared = executor.prepare_rust_selected_action_executor_v1(registry, _BATCH)

    assert tuple(signature(executor.prepare_rust_selected_action_executor_v1).parameters) == (
        "registry",
        "batch",
    )
    assert tuple(signature(prepared.execute_selected_once).parameters) == ("recall",)
    assert not hasattr(prepared, "execute_once")
    with pytest.raises((TypeError, RustActionRegistryError)):
        prepared.execute_selected_once(object())  # type: ignore[arg-type]
    with pytest.raises(executor.RustSelectedActionExecutionError, match="consumed"):
        prepared.execute_selected_once(recall)
    with pytest.raises(TypeError, match="factory-owned"):
        executor.PreparedRustSelectedActionExecutorV1(
            rustc=result.rustc_identity,
            preparation=result.preparation,
            registry=completed_registry,
            cases=_BATCH,
            stdin=executor._encode_batch(_BATCH),
            construction_token=object(),
        )

    def unexpected_discovery():
        raise AssertionError("tool discovery must follow authority and batch validation")

    monkeypatch.setattr(executor, "_discover_rustc", unexpected_discovery)
    with pytest.raises(TypeError, match="exact prepared"):
        executor.prepare_rust_selected_action_executor_v1(object(), _BATCH)  # type: ignore[arg-type]
    with pytest.raises(executor.RustSelectedActionExecutionError, match="batch"):
        executor.prepare_rust_selected_action_executor_v1(registry, ())


def test_concurrent_public_calls_compile_and_run_at_most_once(tmp_path: Path) -> None:
    registry, recall = _build_registry_and_recall(tmp_path / "concurrent", _SUCCESS_SOURCE)
    prepared = executor.prepare_rust_selected_action_executor_v1(registry, _BATCH)

    def invoke():
        try:
            return prepared.execute_selected_once(recall)
        except BaseException as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(lambda _index: invoke(), range(2)))

    results = [
        value for value in outcomes if isinstance(value, executor.RustSelectedActionExecutionV1)
    ]
    errors = [value for value in outcomes if isinstance(value, BaseException)]
    assert len(results) == 1
    assert results[0].compile_receipt.compile_invocations == 1
    assert results[0].run_receipt.run_invocations == 1
    assert len(errors) == 1
    assert "consumed" in str(errors[0])


def test_semantic_compile_failure_remains_consumed(tmp_path: Path) -> None:
    registry, recall = _build_registry_and_recall(
        tmp_path / "compile-failure",
        _COMPILE_FAILURE_SOURCE,
    )
    prepared = executor.prepare_rust_selected_action_executor_v1(registry, _BATCH)

    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="failed compilation",
    ):
        prepared.execute_selected_once(recall)
    with pytest.raises(executor.RustSelectedActionExecutionError, match="consumed"):
        prepared.execute_selected_once(recall)


def test_dep_info_rejects_an_extra_selected_source_dependency(tmp_path: Path) -> None:
    helper = (tmp_path / "synthetic_helper.rs").resolve()
    helper.write_text(
        "pub fn double(value: i64) -> i64 { value * 2 }\n",
        encoding="ascii",
    )
    selected_source = f"""\
#[path = r"{helper.as_posix()}"]
mod synthetic_helper;

fn primary_transform(value: i64) -> i64 {{
    synthetic_helper::double(value)
}}

pub fn solve(values: &[i64]) -> Vec<i64> {{
    values.iter().map(|value| primary_transform(*value)).collect()
}}
"""
    registry, recall = _build_registry_and_recall(
        tmp_path / "dependency",
        selected_source,
    )
    prepared = executor.prepare_rust_selected_action_executor_v1(registry, _BATCH)

    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match=r"dep-info|dependency",
    ):
        prepared.execute_selected_once(recall)


@pytest.mark.parametrize(
    ("selected_source", "message"),
    (
        (_TIMEOUT_SOURCE, "timed out"),
        (_AGGREGATE_OUTPUT_SOURCE, "output limit exceeded"),
        (_PROTOCOL_FAILURE_SOURCE, "protocol mismatch"),
    ),
)
def test_runtime_failures_are_bounded_and_fail_closed(
    tmp_path: Path,
    selected_source: str,
    message: str,
) -> None:
    registry, recall = _build_registry_and_recall(
        tmp_path / sha256(selected_source.encode("ascii")).hexdigest()[:12],
        selected_source,
    )
    prepared = executor.prepare_rust_selected_action_executor_v1(registry, _BATCH)

    with pytest.raises(executor.RustSelectedActionExecutionError, match=message):
        prepared.execute_selected_once(recall)


def test_receipts_cross_authenticate_identity_cases_and_canonical_output(
    successful_execution,
) -> None:
    result, _registry = successful_execution

    changed_stdout = _resign(
        replace(result.run_receipt, stdout_sha256="f" * 64),
        executor._RUN_NAMESPACE,
    )
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="execution receipt",
    ):
        _replace_and_resign_execution(
            result,
            run_receipt=changed_stdout,
        ).authenticate_structure()

    changed_cases = _resign(
        replace(result.run_receipt, case_count=result.run_receipt.case_count - 1),
        executor._RUN_NAMESPACE,
    )
    with pytest.raises(executor.RustSelectedActionExecutionError, match="case count"):
        _replace_and_resign_execution(
            result,
            run_receipt=changed_cases,
        ).authenticate_structure()

    changed_preparation = _resign(
        replace(result.preparation, rustc_identity_sha256="0" * 64),
        executor._PREPARATION_NAMESPACE,
    )
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="execution receipt",
    ):
        _replace_and_resign_execution(
            result,
            preparation=changed_preparation,
        ).authenticate_structure()

    changed_compile = _resign(
        replace(result.compile_receipt, stdout_sha256="f" * 64),
        executor._COMPILE_NAMESPACE,
    )
    with pytest.raises(
        executor.RustSelectedActionExecutionError,
        match="compile receipt",
    ):
        changed_compile.authenticate_structure()


def test_environment_manifests_and_claim_stay_within_trusted_host_boundary(
    successful_execution,
) -> None:
    result, _registry = successful_execution
    compile_environment = dict(result.preparation.compile_environment)
    run_environment = dict(result.preparation.run_environment)

    assert compile_environment["LIB"] == os.pathsep.join(result.rustc_identity.link_search_paths)
    assert not {"PATH", "PATHEXT", "INCLUDE", "LIB", "LIBPATH"}.intersection(run_environment)
    assert result.rustc_identity.target_libdir_file_count > 0
    assert result.rustc_identity.target_libdir_total_bytes > 0
    assert all(value > 0 for value in result.rustc_identity.link_search_file_counts)
    assert all(value > 0 for value in result.rustc_identity.link_search_total_bytes)
    assert "not hostile-code" in result.preparation.claim_boundary
    assert "filesystem or network isolation" in result.preparation.claim_boundary


def test_bounded_artifact_helpers_reject_oversize_and_manifest_file_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = tmp_path / "oversized.bin"
    oversized.write_bytes(b"12345")
    with pytest.raises(executor.RustSelectedActionExecutionError, match="bounded"):
        executor._read_bounded_regular_file(
            oversized,
            label="synthetic artifact",
            maximum_bytes=4,
        )

    manifest = tmp_path / "manifest"
    manifest.mkdir()
    (manifest / "one.lib").write_bytes(b"one")
    (manifest / "two.lib").write_bytes(b"two")
    monkeypatch.setattr(executor, "MAX_TOOLCHAIN_FILES", 1)
    with pytest.raises(executor.RustSelectedActionExecutionError, match="file-count"):
        executor._directory_manifest(manifest)
