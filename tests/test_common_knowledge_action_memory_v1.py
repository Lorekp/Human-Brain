from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from experiments.r3_pilot import common_knowledge_action_memory_v1 as memory_v1
from humanbrain.knowledge_cell_projection import (
    KnowledgeCellProjectionError,
    project_knowledge_corpus_to_cells_v1,
)
from humanbrain.knowledge_corpus import (
    BenchmarkDenyManifest,
    KnowledgeCorpusCaps,
    KnowledgeCorpusManifest,
    KnowledgeFileSpec,
    KnowledgeSourceSpec,
    build_local_knowledge_corpus,
)


LSTRIP_DESCRIPTION = (
    "Return a copy after removing leading characters. The chars argument is a "
    "set of individual characters, not one exact sequence."
)
REMOVEPREFIX_DESCRIPTION = (
    "When the string starts with one complete leading sequence, remove that "
    "sequence. Otherwise preserve the original string unchanged."
)
TASK_TEXT = (
    "Remove one complete leading sequence only when it matches; otherwise "
    "preserve the original string unchanged."
)
MESSAGE_PASSING = "rust.concurrent.message_passing"
SHARED_STATE = "rust.concurrent.shared_state"
MESSAGE_TEXT = (
    "Transmit owned values between isolated workers through a channel while "
    "preserving independent state."
)
SHARED_TEXT = (
    "Coordinate worker access to shared synchronized state through mutex locking."
)
CONCURRENCY_TASK = (
    "Transmit owned values between isolated workers through a channel while "
    "keeping their state independent."
)


def _projection(
    root: Path,
    *,
    first_description: str = LSTRIP_DESCRIPTION,
    second_description: str = REMOVEPREFIX_DESCRIPTION,
    split: str = "acquire",
    single_group: bool = False,
):
    root.mkdir(parents=True, exist_ok=True)
    raw = (
        "str.lstrip(chars=None, /)\n\n"
        f"{first_description}\n\n"
        "str.removeprefix(prefix, /)\n\n"
        f"{second_description}\n"
    ).encode("utf-8")
    source_path = root / "stdtypes.txt"
    source_path.write_bytes(raw)
    manifest = KnowledgeCorpusManifest(
        sources=(
            KnowledgeSourceSpec(
                source_id="synthetic-python-docs",
                canonical_url="https://docs.example.org/stdtypes",
                upstream_revision="unit-fixture-v1",
                license_spdx="MIT",
                license_evidence_url="https://docs.example.org/license",
                attribution="Synthetic contrastive API documentation fixture",
                provenance_confidence=1.0,
                files=(
                    KnowledgeFileSpec(
                        path="stdtypes.txt",
                        expected_raw_sha256=sha256(raw).hexdigest(),
                        split=split,
                        content_type="plain-text",
                        chapter="Text sequence methods",
                        symbol=None,
                    ),
                ),
            ),
        )
    )
    corpus = build_local_knowledge_corpus(
        manifest,
        {"synthetic-python-docs": source_path},
        BenchmarkDenyManifest(),
        KnowledgeCorpusCaps(
            max_bytes_per_source=4096,
            max_total_bytes=4096,
            max_tokenish_per_source=512,
            max_total_tokenish=512,
        ),
    )
    assert len(corpus.chunks) == 4
    assert all(chunk.symbol is None for chunk in corpus.chunks)
    groups = (
        ((tuple(chunk.chunk_id for chunk in corpus.chunks)),)
        if single_group
        else (
            (corpus.chunks[0].chunk_id, corpus.chunks[1].chunk_id),
            (corpus.chunks[2].chunk_id, corpus.chunks[3].chunk_id),
        )
    )
    return project_knowledge_corpus_to_cells_v1(
        corpus,
        admitted_chunk_groups=groups,
        split=split,
    )


def _candidates() -> tuple[memory_v1.KnowledgePatchCandidateV1, ...]:
    return (
        memory_v1.KnowledgePatchCandidateV1.from_source(
            "strip_character_set",
            "def drop_marker(value, marker):\n"
            "    return str.lstrip(value, marker)\n",
        ),
        memory_v1.KnowledgePatchCandidateV1.from_source(
            "remove_whole_marker",
            "def drop_marker(value, marker):\n"
            "    return str.removeprefix(value, marker)\n",
        ),
    )


def _declared_projection(
    root: Path,
    labels: tuple[str | None, str | None],
):
    root.mkdir(parents=True, exist_ok=True)
    files = (
        ("message.txt", MESSAGE_TEXT, labels[0]),
        ("shared.txt", SHARED_TEXT, labels[1]),
    )
    for path, text, _symbol in files:
        (root / path).write_text(text, encoding="utf-8", newline="\n")
    manifest = KnowledgeCorpusManifest(
        sources=(
            KnowledgeSourceSpec(
                source_id="synthetic-rust-concepts",
                canonical_url="https://docs.example.org/rust-concepts",
                upstream_revision="unit-fixture-v1",
                license_spdx="MIT",
                license_evidence_url="https://docs.example.org/license",
                attribution="Synthetic neutral concurrency concept fixture",
                provenance_confidence=1.0,
                files=tuple(
                    KnowledgeFileSpec(
                        path=path,
                        expected_raw_sha256=sha256(text.encode("utf-8")).hexdigest(),
                        split="acquire",
                        content_type="plain-text",
                        chapter="Concurrency concepts",
                        symbol=symbol,
                    )
                    for path, text, symbol in files
                ),
            ),
        )
    )
    corpus = build_local_knowledge_corpus(
        manifest,
        {"synthetic-rust-concepts": root},
        BenchmarkDenyManifest(),
        KnowledgeCorpusCaps(
            max_bytes_per_source=4096,
            max_total_bytes=4096,
            max_tokenish_per_source=512,
            max_total_tokenish=512,
        ),
    )
    projection = project_knowledge_corpus_to_cells_v1(
        corpus,
        admitted_chunk_groups=tuple((chunk.chunk_id,) for chunk in corpus.chunks),
        split="acquire",
    )
    assert all(not chunk.document_symbols for chunk in projection.chunks)
    return projection


def _neutral_candidates() -> tuple[memory_v1.KnowledgePatchCandidateV1, ...]:
    return (
        memory_v1.KnowledgePatchCandidateV1.from_source(
            "message_passing",
            "def choose():\n    return rust.concurrent.message_passing()\n",
        ),
        memory_v1.KnowledgePatchCandidateV1.from_source(
            "shared_state",
            "def choose():\n    return rust.concurrent.shared_state()\n",
        ),
    )


def test_raw_document_cells_select_and_execute_one_exact_candidate(
    tmp_path: Path,
) -> None:
    projection = _projection(tmp_path / "knowledge-on")
    memory = memory_v1.fit_common_knowledge_action_memory_v1(projection)
    candidates = _candidates()
    environment = memory_v1.KnowledgePatchEnvironmentV1(memory, candidates)

    result = memory_v1.run_knowledge_action_pilot_v1(
        memory,
        task_text=TASK_TEXT,
        environment=environment,
        function_name="drop_marker",
        cases=(("xyxpayload", "xy"), ("payload", "xy")),
    )

    assert result.recall.status == memory_v1.RECALLED
    assert result.recall.selected_symbol == "str.removeprefix"
    assert result.recall.selected_candidate_id == "remove_whole_marker"
    assert result.outputs == ("xpayload", "payload")
    assert result.materialized_candidate_ids == ("remove_whole_marker",)
    assert result.execution_count == 1
    assert result.recall.physical_output_record["decision"] == "TOKEN"
    assert (
        memory.symbol_vocabulary[result.recall.physical_output_record["token"]]
        == result.recall.selected_symbol
    )
    assert memory.training_summary.admitted_chunk_count == 4
    assert (
        memory.training_summary.config_sha256
        == "26f7386982c2e74daa4b581473c0b24ad8d4f7c84000a187e81eba5b40a53430"
    )
    assert (
        memory.checkpoint_sha256
        == "9aa03c4461cd115625ed4aa622dcd59d1bc0a5ec89b2837066a9a7ac07882ac1"
    )
    assert memory.training_summary.factor_hash_collision_count == 0
    assert result.recall.query_bucket_collision_count == 0
    assert memory.training_summary.checkpoint_array_bytes <= 64 * 1024
    assert memory.training_summary.training_cell_steps <= 1_000_000
    assert result.recall.work.total_cell_steps <= 200_000
    assert all("text" not in name for name in memory.__slots__)
    assert memory_v1.RAW_TEXT_RETENTION_COUNT == 0
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="one materialization"):
        environment.execute_selected(
            result.recall,
            function_name="drop_marker",
            cases=(("again", "a"),),
        )


def test_required_ablation_shuffle_conflict_splice_and_order_controls(
    tmp_path: Path,
) -> None:
    projection = _projection(tmp_path / "normal")
    memory = memory_v1.fit_common_knowledge_action_memory_v1(projection)
    no_plasticity = memory_v1.fit_common_knowledge_action_memory_v1(
        projection, plasticity_gate=False
    )
    task = memory_v1.project_knowledge_action_task_v1(TASK_TEXT)
    candidates = tuple(value.projection for value in _candidates())

    gate_off = memory.recall(task, candidates, cell_gate=False)
    absent = no_plasticity.recall(task, candidates)
    forward = memory.recall(task, candidates)
    reverse = memory.recall(task, tuple(reversed(candidates)))

    assert gate_off.status == memory_v1.STOP_CELL_OFF
    assert gate_off.selected_candidate_id is None
    assert absent.status != memory_v1.RECALLED
    assert absent.selected_candidate_id is None
    assert no_plasticity.learned_state_shapes == memory.learned_state_shapes
    assert forward.record() == reverse.record()
    assert forward.query_bucket_collision_count == 0

    shuffled_projection = _projection(
        tmp_path / "shuffled",
        first_description=REMOVEPREFIX_DESCRIPTION,
        second_description=LSTRIP_DESCRIPTION,
    )
    shuffled = memory_v1.fit_common_knowledge_action_memory_v1(
        shuffled_projection
    ).recall(task, candidates)
    assert shuffled.selected_symbol != "str.removeprefix"

    conflict_projection = _projection(
        tmp_path / "conflict",
        first_description=REMOVEPREFIX_DESCRIPTION,
        second_description=REMOVEPREFIX_DESCRIPTION,
    )
    conflict = memory_v1.fit_common_knowledge_action_memory_v1(
        conflict_projection
    ).recall(task, candidates)
    assert conflict.status == memory_v1.STOP_PHYSICAL_RACE
    assert conflict.selected_candidate_id is None
    assert (
        conflict.checkpoint_sha256
        == "3c0aaf1097f47bc7de91a2bdd526d0a33bb49d97a6c4910ea87544bbc7e9f808"
    )
    assert conflict.candidate_scores == (
        ("str.lstrip", 12),
        ("str.removeprefix", 12),
    )
    assert conflict.physical_output_record["decision"] == "STOP"
    assert conflict.physical_output_record["stop_reason"] == "COLLISION"

    spliced = replace(
        projection,
        producer_receipt=shuffled_projection.producer_receipt,
    )
    with pytest.raises(KnowledgeCellProjectionError):
        memory_v1.fit_common_knowledge_action_memory_v1(spliced)

    multi_signature = memory_v1.fit_common_knowledge_action_memory_v1(
        _projection(tmp_path / "multi-signature", single_group=True)
    ).recall(task, candidates)
    assert multi_signature.status == memory_v1.STOP_PHYSICAL_RACE

    non_acquire = _projection(tmp_path / "non-acquire", split="not-acquire")
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="only.*acquire"):
        memory_v1.fit_common_knowledge_action_memory_v1(non_acquire)

    tampered = replace(forward, selected_candidate_id="strip_character_set")
    tamper_environment = memory_v1.KnowledgePatchEnvironmentV1(
        memory, _candidates()
    )
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="recall root"):
        tamper_environment.materialize_selected(tampered)

    altered_candidates = (
        _candidates()[0],
        memory_v1.KnowledgePatchCandidateV1.from_source(
            "remove_whole_marker",
            "def drop_marker(value, marker):\n"
            "    unchanged = value\n"
            "    return str.removeprefix(unchanged, marker)\n",
        ),
    )
    altered_recall = memory.recall(
        task, tuple(value.projection for value in altered_candidates)
    )
    source_splice_environment = memory_v1.KnowledgePatchEnvironmentV1(
        memory, _candidates()
    )
    with pytest.raises(
        memory_v1.KnowledgeActionMemoryError, match="candidate set"
    ):
        source_splice_environment.materialize_selected(altered_recall)


def test_candidate_contract_rejects_hidden_second_call() -> None:
    with pytest.raises(
        memory_v1.KnowledgeActionMemoryError,
        match="exactly one qualified AST call",
    ):
        memory_v1.project_knowledge_action_candidate_v1(
            "hidden_second_call",
            "def drop_marker(value, marker):\n"
            "    eval('1 + 1')\n"
            "    return str.removeprefix(value, marker)\n",
        )


def test_declared_symbols_learn_neutral_lanes_and_permute_causally(
    tmp_path: Path,
) -> None:
    candidates = _neutral_candidates()
    candidate_projections = tuple(value.projection for value in candidates)
    task = memory_v1.project_knowledge_action_task_v1(CONCURRENCY_TASK)

    projection = _declared_projection(
        tmp_path / "declared",
        (MESSAGE_PASSING, SHARED_STATE),
    )
    memory = memory_v1.fit_common_knowledge_action_memory_v1(projection)
    recalled = memory.recall(task, candidate_projections)

    assert memory.symbol_vocabulary == (MESSAGE_PASSING, SHARED_STATE)
    assert recalled.status == memory_v1.RECALLED
    assert recalled.selected_symbol == MESSAGE_PASSING
    assert recalled.selected_candidate_id == "message_passing"

    permuted_projection = _declared_projection(
        tmp_path / "permuted",
        (SHARED_STATE, MESSAGE_PASSING),
    )
    permuted_memory = memory_v1.fit_common_knowledge_action_memory_v1(
        permuted_projection
    )
    permuted = permuted_memory.recall(task, candidate_projections)

    assert permuted.status == memory_v1.RECALLED
    assert permuted.selected_symbol == SHARED_STATE
    assert permuted.selected_candidate_id == "shared_state"
    assert permuted_memory.checkpoint_sha256 != memory.checkpoint_sha256


def test_declared_symbol_ablation_and_splices_fail_closed(tmp_path: Path) -> None:
    without_labels = _declared_projection(tmp_path / "without-labels", (None, None))
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="symbol vocabulary"):
        memory_v1.fit_common_knowledge_action_memory_v1(without_labels)

    projection = _declared_projection(
        tmp_path / "projection",
        (MESSAGE_PASSING, SHARED_STATE),
    )
    permuted_projection = _declared_projection(
        tmp_path / "projection-splice",
        (SHARED_STATE, MESSAGE_PASSING),
    )
    with pytest.raises(KnowledgeCellProjectionError):
        memory_v1.fit_common_knowledge_action_memory_v1(
            replace(
                projection,
                producer_receipt=permuted_projection.producer_receipt,
            )
        )

    message, shared = _neutral_candidates()
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="patch candidate"):
        replace(message, source=shared.source).authenticate()
    with pytest.raises(memory_v1.KnowledgeActionMemoryError, match="candidate projection"):
        replace(message.projection, symbol=shared.projection.symbol).authenticate()
