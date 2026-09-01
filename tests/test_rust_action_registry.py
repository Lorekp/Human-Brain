from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from inspect import Parameter, signature
from pathlib import Path
from threading import Event, Thread

import pytest
from experiments.r3_pilot import common_knowledge_action_memory_v1 as action_memory
from humanbrain import rust_action_registry as rust_registry
from humanbrain.consolidated_knowledge_cell_projection import (
    project_consolidated_knowledge_to_cells_v1,
)
from humanbrain.knowledge_consolidation import consolidate_knowledge_corpus_v1
from humanbrain.knowledge_corpus import (
    BenchmarkDenyManifest,
    KnowledgeCorpus,
    KnowledgeCorpusCaps,
    KnowledgeCorpusManifest,
    KnowledgeFileSpec,
    KnowledgeSourceSpec,
    build_local_knowledge_corpus,
)
from humanbrain.rust_action_registry import (
    BoundRustActionMemoryV1,
    RustActionMaterializationV1,
    RustActionRegistryError,
    prepare_rust_action_registry_v1,
)
from tools import build_rust_book_knowledge_corpus as rust_book

SEND_SOURCE = """\
#!/usr/bin/env mutex_lock
#![allow(dead_code)]
pub fn send_channel<'a>(worker: &'a str) -> &'a str {
    opaque_literals!(123mutex, "lock"mutex, b'p'poison);
    let cooked_annotation = "mutex lock poison";
    let raw_annotation = r###"mutex lock poison"###;
    let byte_annotation = b"mutex lock";
    let raw_byte_annotation = br##"mutex lock"##;
    let c_annotation = c"mutex lock";
    let raw_c_annotation = cr#"mutex lock"#;
    let byte_character = b'm';
    let character = 'm';
    let numeric_annotation = 123usize;
    /* mutex outer /* lock nested */ poison */
    let _ = (
        cooked_annotation,
        raw_annotation,
        byte_annotation,
        raw_byte_annotation,
        c_annotation,
        raw_c_annotation,
        byte_character,
        character,
        numeric_annotation,
    );
    worker
}
"""

LOCK_SOURCE = """\
#![allow(dead_code)]
pub fn lock_mutex<'a>(worker: &'a str) -> &'a str {
    let cooked_annotation = "send channel ownership";
    let raw_annotation = r##"send channel ownership"##;
    // send channel ownership
    /** send channel ownership */
    let _ = (cooked_annotation, raw_annotation);
    worker
}
"""


def _build_corpus(
    root: Path,
    *,
    chapters: tuple[str, str] = ("message-passing", "shared-state"),
    symbols: tuple[str, str] = ("audit.message", "audit.shared"),
    source_id: str = "synthetic-rust-book",
    raw_texts: tuple[bytes, bytes] = (
        b"send channel owned message worker first\n\nsend channel owned message worker second\n",
        b"lock mutex shared state worker first\n\nlock mutex shared state worker second\n",
    ),
) -> KnowledgeCorpus:
    root.mkdir(parents=True, exist_ok=True)
    values = (
        (
            "sending.txt",
            chapters[0],
            symbols[0],
            raw_texts[0],
        ),
        (
            "locking.txt",
            chapters[1],
            symbols[1],
            raw_texts[1],
        ),
    )
    files: list[KnowledgeFileSpec] = []
    for path, chapter, symbol, raw in values:
        (root / path).write_bytes(raw)
        files.append(
            KnowledgeFileSpec(
                path=path,
                expected_raw_sha256=sha256(raw).hexdigest(),
                split="acquire",
                content_type="plain-text",
                chapter=chapter,
                symbol=symbol,
            )
        )
    manifest = KnowledgeCorpusManifest(
        sources=(
            KnowledgeSourceSpec(
                source_id=source_id,
                canonical_url=f"https://docs.example.org/{source_id}/",
                upstream_revision="registry-fixture-v1",
                license_spdx="MIT",
                license_evidence_url="https://docs.example.org/license",
                attribution="Synthetic Rust registry fixture",
                provenance_confidence=1.0,
                files=tuple(files),
            ),
        )
    )
    return build_local_knowledge_corpus(
        manifest,
        {source_id: root},
        BenchmarkDenyManifest(),
        KnowledgeCorpusCaps(
            max_bytes_per_source=16_384,
            max_total_bytes=16_384,
            max_tokenish_per_source=2_048,
            max_total_tokenish=2_048,
        ),
    )


def _prepare(root: Path):
    corpus = _build_corpus(root)
    consolidation = consolidate_knowledge_corpus_v1(corpus, split="acquire")
    registry = prepare_rust_action_registry_v1(
        corpus,
        consolidation,
        sources=(SEND_SOURCE, LOCK_SOURCE),
    )
    return corpus, consolidation, registry


def _fit(root: Path):
    corpus, consolidation, registry = _prepare(root)
    projection = project_consolidated_knowledge_to_cells_v1(
        consolidation,
        corpus=corpus,
    )
    return corpus, consolidation, projection, registry, registry.fit_memory(projection)


def test_registry_builder_is_task_free_and_exactly_two_source_only() -> None:
    parameters = signature(prepare_rust_action_registry_v1).parameters

    assert tuple(parameters) == ("corpus", "consolidation", "sources")
    assert parameters["sources"].kind is Parameter.KEYWORD_ONLY
    assert not {
        "task",
        "chapter",
        "path",
        "label",
        "candidate",
        "oracle",
        "expected",
        "execute",
    }.intersection(parameters)


def test_exact_overlap_masks_noncode_and_roots_concrete_rustfmt(
    tmp_path: Path,
) -> None:
    _corpus, _consolidation, registry = _prepare(tmp_path / "normal")
    view = registry.selector_view
    receipt = registry.receipt

    symbols_by_factor: dict[str, set[str]] = {}
    for binding in view.factor_bindings:
        symbols_by_factor.setdefault(binding.factor, set()).add(binding.opaque_symbol)
    send_symbol = symbols_by_factor["tok:send"]
    lock_symbol = symbols_by_factor["tok:lock"]

    assert send_symbol == symbols_by_factor["tok:channel"]
    assert lock_symbol == symbols_by_factor["tok:mutex"]
    assert send_symbol.isdisjoint(lock_symbol)
    assert len(view.candidate_projections) == 2
    assert all(candidate.candidate_id.startswith("ra_") for candidate in view.candidate_projections)
    assert all(
        candidate.symbol == f"rust_action.{candidate.candidate_id}"
        for candidate in view.candidate_projections
    )
    assert Path(receipt.rustfmt_path).is_absolute()
    assert Path(receipt.rustfmt_path).is_file()
    assert receipt.rustfmt_version
    assert len(receipt.rustfmt_executable_sha256) == 64
    assert len(receipt.rustfmt_identity_sha256) == 64
    view.authenticate()
    receipt.authenticate_structure()


def test_source_order_and_audit_label_values_cannot_route_actions(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path / "first")
    consolidation = consolidate_knowledge_corpus_v1(corpus, split="acquire")
    forward = prepare_rust_action_registry_v1(
        corpus,
        consolidation,
        sources=(SEND_SOURCE, LOCK_SOURCE),
    )
    reverse = prepare_rust_action_registry_v1(
        corpus,
        consolidation,
        sources=(LOCK_SOURCE, SEND_SOURCE),
    )
    renamed_corpus = _build_corpus(
        tmp_path / "renamed",
        chapters=("opaque-audit-b", "opaque-audit-a"),
        symbols=("audit.changed_b", "audit.changed_a"),
    )
    renamed_consolidation = consolidate_knowledge_corpus_v1(
        renamed_corpus,
        split="acquire",
    )
    renamed = prepare_rust_action_registry_v1(
        renamed_corpus,
        renamed_consolidation,
        sources=(SEND_SOURCE, LOCK_SOURCE),
    )

    assert forward.selector_view == reverse.selector_view == renamed.selector_view
    assert forward.receipt == reverse.receipt
    assert forward.receipt != renamed.receipt


def test_cross_corpus_and_duplicate_structural_sources_fail_closed(
    tmp_path: Path,
) -> None:
    first = _build_corpus(tmp_path / "first")
    first_consolidation = consolidate_knowledge_corpus_v1(first, split="acquire")
    other = _build_corpus(
        tmp_path / "other",
        source_id="other-rust-book",
    )
    with pytest.raises(ValueError):
        prepare_rust_action_registry_v1(
            other,
            first_consolidation,
            sources=(SEND_SOURCE, LOCK_SOURCE),
        )

    comment_only_variant = SEND_SOURCE.replace(
        "mutex outer",
        "changed audit prose",
    ).replace("mutex lock poison", "different hidden payload")
    with pytest.raises(RustActionRegistryError, match="distinct exact and structural"):
        prepare_rust_action_registry_v1(
            first,
            first_consolidation,
            sources=(SEND_SOURCE, comment_only_variant),
        )


def test_bound_memory_recalls_and_materializes_only_the_selected_source(
    tmp_path: Path,
) -> None:
    _corpus, _consolidation, projection, registry, bound = _fit(tmp_path / "bound")
    task = action_memory.project_knowledge_action_task_v1("send channel owned message")

    assert type(bound) is BoundRustActionMemoryV1
    assert bound.receipt.factor_count == projection.distinct_factor_count
    assert bound.receipt.association_count == len(registry.selector_view.factor_bindings)
    assert bound._association_projection.cognition_sha256 == (bound.receipt.binder_cognition_sha256)
    assert set(bound._association_projection.authorization_roots) == {
        bound.receipt.binder_authorization_sha256,
        registry.receipt.sha256,
        registry.selector_view.sha256,
        *registry.receipt.authorization_sha256s,
    }
    assert {
        projection.provenance_sha256,
        projection.sha256,
        registry.receipt.sha256,
        registry.selector_view.sha256,
        bound.receipt.binder_cognition_sha256,
        bound.receipt.binder_authorization_sha256,
        bound.receipt.association_projection_sha256,
    }.issubset(bound.provenance_roots)
    recall = registry.recall(bound, task)
    assert recall.status == action_memory.RECALLED

    materializations: list[RustActionMaterializationV1] = []
    result = registry.execute_selected(recall, materializations.append)

    assert result is None
    assert len(materializations) == 1
    assert materializations[0].source == SEND_SOURCE
    materializations[0].authenticate_structure()
    assert not hasattr(materializations[0], "authenticate")
    with pytest.raises(RustActionRegistryError, match="cannot materialize"):
        registry.materialize_selected(recall)


def test_recall_accepts_only_builder_bound_memory_and_no_task_time_candidates(
    tmp_path: Path,
) -> None:
    _corpus, _consolidation, _projection, registry, bound = _fit(tmp_path / "api")
    task = action_memory.project_knowledge_action_task_v1("send channel")

    fit_parameters = signature(registry.fit_memory).parameters
    recall_parameters = signature(registry.recall).parameters
    assert tuple(fit_parameters) == ("knowledge", "plasticity_gate")
    assert tuple(recall_parameters) == ("memory", "task", "cell_gate")
    assert fit_parameters["plasticity_gate"].kind is Parameter.KEYWORD_ONLY
    assert recall_parameters["cell_gate"].kind is Parameter.KEYWORD_ONLY
    assert "candidates" not in recall_parameters

    with pytest.raises(TypeError, match="exact bound"):
        registry.recall(bound._memory, task)
    with pytest.raises(TypeError, match="builder-owned"):
        BoundRustActionMemoryV1(
            knowledge_projection=bound._knowledge_projection,
            association_projection=bound._association_projection,
            memory=bound._memory,
            receipt=bound.receipt,
            construction_token=object(),
        )


def test_shared_factor_support_stops_equal_physical_race(tmp_path: Path) -> None:
    _corpus, _consolidation, _projection, registry, bound = _fit(tmp_path / "tie")

    recall = registry.recall(
        bound,
        action_memory.project_knowledge_action_task_v1("worker"),
    )

    assert recall.status == action_memory.STOP_PHYSICAL_RACE
    assert tuple(score for _symbol, score in recall.candidate_scores) == (1, 1)
    with pytest.raises(RustActionRegistryError, match="not authenticated"):
        registry.materialize_selected(recall)


def test_cross_projection_registry_and_candidate_roots_fail_closed(
    tmp_path: Path,
) -> None:
    corpus, consolidation, projection, registry, bound = _fit(tmp_path / "original")
    foreign_corpus = _build_corpus(
        tmp_path / "foreign-corpus",
        chapters=("renamed-message", "renamed-shared"),
        source_id="foreign-rust-book",
    )
    foreign_consolidation = consolidate_knowledge_corpus_v1(
        foreign_corpus,
        split="acquire",
    )
    foreign_projection = project_consolidated_knowledge_to_cells_v1(
        foreign_consolidation,
        corpus=foreign_corpus,
    )
    with pytest.raises(RustActionRegistryError, match="not bound"):
        registry.fit_memory(foreign_projection)

    foreign_provenance_registry = prepare_rust_action_registry_v1(
        foreign_corpus,
        foreign_consolidation,
        sources=(SEND_SOURCE, LOCK_SOURCE),
    )
    foreign_provenance_bound = foreign_provenance_registry.fit_memory(foreign_projection)
    task = action_memory.project_knowledge_action_task_v1("send channel")
    assert foreign_provenance_registry.selector_view == registry.selector_view
    with pytest.raises(RustActionRegistryError, match="not from this registry corpus"):
        registry.recall(foreign_provenance_bound, task)

    foreign_candidate_registry = prepare_rust_action_registry_v1(
        corpus,
        consolidation,
        sources=(SEND_SOURCE.replace("send_channel", "send_channel_fast"), LOCK_SOURCE),
    )
    foreign_candidate_bound = foreign_candidate_registry.fit_memory(projection)
    with pytest.raises(RustActionRegistryError, match="association projection changed"):
        registry.recall(foreign_candidate_bound, task)

    tampered = replace(
        bound.receipt,
        candidate_set_sha256="0" * 64,
    )
    object.__setattr__(bound, "_receipt", tampered)
    with pytest.raises(RustActionRegistryError, match="receipt"):
        registry.recall(bound, task)


def test_materialization_is_atomically_consumed_before_slow_callback(
    tmp_path: Path,
) -> None:
    _corpus, _consolidation, _projection, registry, bound = _fit(tmp_path / "concurrent")
    recall = registry.recall(
        bound,
        action_memory.project_knowledge_action_task_v1("send channel"),
    )
    entered = Event()
    release = Event()
    seen: list[RustActionMaterializationV1] = []
    results: list[str] = []
    errors: list[BaseException] = []

    def slow_callback(materialization: RustActionMaterializationV1) -> str:
        seen.append(materialization)
        entered.set()
        if not release.wait(10):
            raise AssertionError("callback release timed out")
        return materialization.source_sha256

    def run() -> None:
        try:
            results.append(registry.execute_selected(recall, slow_callback))
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=run, daemon=True)
    thread.start()
    assert entered.wait(10)
    with pytest.raises(RustActionRegistryError, match="cannot materialize"):
        registry.execute_selected(recall, slow_callback)
    release.set()
    thread.join(10)

    assert not thread.is_alive()
    assert errors == []
    assert len(seen) == 1
    assert results == [seen[0].source_sha256]


def test_callback_and_selected_projection_failures_remain_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _corpus, _consolidation, _projection, callback_registry, callback_bound = _fit(
        tmp_path / "callback"
    )
    callback_recall = callback_registry.recall(
        callback_bound,
        action_memory.project_knowledge_action_task_v1("send channel"),
    )

    def fail_callback(_materialization: RustActionMaterializationV1) -> None:
        raise RuntimeError("host callback failed")

    with pytest.raises(RuntimeError, match="host callback failed"):
        callback_registry.execute_selected(callback_recall, fail_callback)
    with pytest.raises(RustActionRegistryError, match="cannot materialize"):
        callback_registry.materialize_selected(callback_recall)

    _corpus, _consolidation, _projection, projection_registry, projection_bound = _fit(
        tmp_path / "projection"
    )
    projection_recall = projection_registry.recall(
        projection_bound,
        action_memory.project_knowledge_action_task_v1("send channel"),
    )
    original_project_source = rust_registry._project_source
    calls = 0

    def fail_selected_projection(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RustActionRegistryError("selected projection failed")
        return original_project_source(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(rust_registry, "_project_source", fail_selected_projection)
        with pytest.raises(RustActionRegistryError, match="selected projection failed"):
            projection_registry.materialize_selected(projection_recall)
    with pytest.raises(RustActionRegistryError, match="cannot materialize"):
        projection_registry.materialize_selected(projection_recall)


def test_non_ascii_or_invalid_rust_source_fails_before_authorization(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path / "corpus")
    consolidation = consolidate_knowledge_corpus_v1(corpus, split="acquire")

    with pytest.raises(RustActionRegistryError, match="ASCII"):
        prepare_rust_action_registry_v1(
            corpus,
            consolidation,
            sources=(SEND_SOURCE + "// café\n", LOCK_SOURCE),
        )
    with pytest.raises(RustActionRegistryError, match="projection failed"):
        prepare_rust_action_registry_v1(
            corpus,
            consolidation,
            sources=("fn broken( {", LOCK_SOURCE),
        )
    with pytest.raises(RustActionRegistryError, match="skip directives"):
        prepare_rust_action_registry_v1(
            corpus,
            consolidation,
            sources=(
                "#[rustfmt::skip]\nfn send_channel() {}\n",
                LOCK_SOURCE,
            ),
        )


@pytest.mark.parametrize(
    "attribute",
    (
        "#! [rustfmt::skip]",
        "#!/*x*/[rustfmt::skip]",
        "#! /*x*/ [rustfmt::skip]",
    ),
)
def test_ambiguous_byte_zero_inner_attributes_fail_closed(
    tmp_path: Path,
    attribute: str,
) -> None:
    corpus = _build_corpus(tmp_path / "corpus")
    consolidation = consolidate_knowledge_corpus_v1(corpus, split="acquire")

    with pytest.raises(RustActionRegistryError, match="ambiguous Rust shebang"):
        prepare_rust_action_registry_v1(
            corpus,
            consolidation,
            sources=(
                f"{attribute}\npub fn send_channel() {{}}\n",
                LOCK_SOURCE,
            ),
        )


def test_registry_receipt_rejects_malformed_authorization_roots(tmp_path: Path) -> None:
    _corpus, _consolidation, registry = _prepare(tmp_path / "receipt")
    tampered = replace(
        registry.receipt,
        authorization_sha256s=("not-a-sha256",),
        sha256="",
    )
    payload = {key: value for key, value in asdict(tampered).items() if key != "sha256"}
    object.__setattr__(
        tampered,
        "sha256",
        rust_registry._digest(rust_registry._RECEIPT_NAMESPACE, payload),
    )

    with pytest.raises(RustActionRegistryError, match="receipt authentication"):
        tampered.authenticate_structure()


def test_boolean_literals_are_masked_but_raw_and_numbered_identifiers_remain(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(
        tmp_path / "literal-corpus",
        raw_texts=(
            b"true sha256sum value first\n\ntrue sha256sum value second\n",
            b"false mutex state first\n\nfalse mutex state second\n",
        ),
    )
    consolidation = consolidate_knowledge_corpus_v1(corpus, split="acquire")

    with pytest.raises(RustActionRegistryError, match="exact authenticated factor overlap"):
        prepare_rust_action_registry_v1(
            corpus,
            consolidation,
            sources=(
                "pub fn alpha() { let flag = true; }\n",
                "pub fn beta() { let flag = false; }\n",
            ),
        )

    registry = prepare_rust_action_registry_v1(
        corpus,
        consolidation,
        sources=(
            "pub fn r#true() { sha256sum(); }\n",
            "pub fn mutex() {}\n",
        ),
    )
    factors = {binding.factor for binding in registry.selector_view.factor_bindings}
    assert {"tok:true", "tok:sha256sum", "tok:mutex"} <= factors


def test_numeric_payloads_cannot_authorize_but_field_method_and_range_identifiers_can(
    tmp_path: Path,
) -> None:
    numeric_corpus = _build_corpus(
        tmp_path / "numeric",
        raw_texts=(
            b"e2 exponent first\n\ne2 exponent second\n",
            b"mutex shared first\n\nmutex shared second\n",
        ),
    )
    numeric_consolidation = consolidate_knowledge_corpus_v1(
        numeric_corpus,
        split="acquire",
    )
    with pytest.raises(RustActionRegistryError, match="exact authenticated factor overlap"):
        prepare_rust_action_registry_v1(
            numeric_corpus,
            numeric_consolidation,
            sources=(
                "pub fn alpha() { let _ = (1e2, 1.0E+2_f64, 2.); }\n",
                "pub fn mutex() {}\n",
            ),
        )

    method_corpus = _build_corpus(
        tmp_path / "method",
        raw_texts=(
            b"e2 foo method first\n\ne2 foo method second\n",
            b"mutex shared first\n\nmutex shared second\n",
        ),
    )
    method_consolidation = consolidate_knowledge_corpus_v1(
        method_corpus,
        split="acquire",
    )
    registry = prepare_rust_action_registry_v1(
        method_corpus,
        method_consolidation,
        sources=(
            "pub fn alpha() { let _ = 1.e2; 1.foo(); for _ in 1..2 { foo(); } }\n",
            "pub fn mutex() {}\n",
        ),
    )
    factors = {binding.factor for binding in registry.selector_view.factor_bindings}
    assert {"tok:e2", "tok:foo"} <= factors


def test_lifetime_identifier_bodies_survive_boolean_literal_masking() -> None:
    masked = rust_registry._mask_rust_noncode_v1("opaque!('true, 'r#true, true);")

    assert masked.count("true") == 2
    assert "r#true" not in masked


def test_real_ten_card_projection_fits_within_frozen_memory_caps() -> None:
    archive = rust_book.DEFAULT_CACHE_DIR / rust_book.ARCHIVE_PATH
    if not archive.is_file():
        pytest.skip("ignored authenticated Rust Book cache is unavailable")

    corpus, _receipt = rust_book.build_from_cache()
    consolidation = consolidate_knowledge_corpus_v1(corpus, split="acquire")
    projection = project_consolidated_knowledge_to_cells_v1(
        consolidation,
        corpus=corpus,
    )
    registry = prepare_rust_action_registry_v1(
        corpus,
        consolidation,
        sources=(SEND_SOURCE, LOCK_SOURCE),
    )
    bound = registry.fit_memory(projection)
    summary = bound.training_summary

    assert projection.concept_count == 10
    assert projection.factor_association_count == 1_280
    assert projection.distinct_factor_count == bound.receipt.factor_count == 791
    assert summary.checkpoint_array_bytes < 64 * 1024
    assert summary.training_cell_steps <= 1_800_000
    assert {
        projection.sha256,
        projection.provenance_sha256,
        registry.receipt.sha256,
        registry.selector_view.sha256,
        registry.selector_view.candidate_set_sha256,
        *registry.receipt.authorization_sha256s,
        bound.receipt.binder_cognition_sha256,
        bound.receipt.binder_authorization_sha256,
        bound.receipt.association_projection_sha256,
    }.issubset(bound.provenance_roots)
