from __future__ import annotations

import inspect
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import common_continual_token_edit_schema_v1 as continual
from experiments.r3_pilot import (
    continual_rust_token_edit_schema_transfer_cohort_v1 as cohort,
)

EXPECTED_SPEC_SHA256 = "70bfdbc00bafb4e6a2d4d569dd74583d4dfafc104254ef2fb068f70fea09eafe"
EXPECTED_EPISODE_ROOTS = (
    (
        "7f3db0db3743232bd5d8ef3053ac3f568d8c748f1ca375074a1a34808d735882",
        "e96009a74af6a75903360f3de3b9535a48a9e1f00fcda143ebec81cf7e867264",
        "b194e68189d5d0a77e3860b7fbdfdba3b41da6dcbf292134fd622003e6c838a7",
        "c763dfbc9d9585384de52c0845aece419b0e21e3c795a103c4feb47aa4ff2a8e",
        "f80160a1694d6582821d7b81e0eb2f98693f0f1617b69272842496e26ea4425b",
        "8822e0d7267937db140d9316825e7035044ad0b923c29f809f5fce322f487ddb",
        "d335b523ec67621ba6925c48c84f1d4cce2f38f00956099297db443d1fe80f50",
        "d00939b3ee635b6011a6e3296550dfef6fa1fa1d3211f42514822300bb37332c",
    ),
    (
        "a87b5833d976865ee59e741daf824c7bc3f163570e87ea27117896de20c6126d",
        "6a00e08d35b246c35a967e5f8f2a5e1c4f9749915209f1b42e048c1aa27f9aa7",
        "50d5eaa57f084ec54507210834f49e1b154265cf908670b67d9b4af258f0b6f2",
        "2a5e271db94afad038f4cced88470a1f144487ad22a33078ac1c376338894b99",
        "aba480f6943f55066001b118d170d51c6be17b810eb735d890ce44310a8aa054",
        "4778fc07bb1f0d6cf2ed2130e70eedb9dad740118165edff9aa8986efc4a44b3",
        "9156378419e5a875a0aa88632bbada13c3261e07ca979ce1ea9afbf650025975",
        "a35874ffedbaabac33977d554916c049a823434b35d57e5ba82be0770ab088a7",
    ),
)
EXPECTED_TRANSFER_ROOTS = (
    "40a1ad002922c1c8a08fc1a860a40b606943952b54778916c3c2c8fe75778751",
    "b3a97174aee37cf68c86a29c099db321446142414102372d83462415f0cf059a",
    "76383769b37cbad7dd70972bc8fedcc4e906d11b9d982cbccbb49b467c279887",
    "31079c9d393b1b76a1e371ea57cf515f27be8603fd3c02f44a1ed34fd9ed88b2",
    "7642127956daa6b9d8728e74f144e762fe8ba761a1fe5a88809046cff95bede2",
    "8ddd9235627ed61cf2950bad13117e880472d6807941bdd3b916a6999fc73fa8",
    "66703237df4e8f68e7f9e678239970c93e218e7267bae1aeb4b8833e7a475a93",
)
EXPECTED_C1_EXCLUSIONS = (
    "126b0ce796c248396e311e31a332e28dcdfbb328e0b994d64b39a1f6e3e0877b",
    "16504bb28d57532c6621aa0a36c6db23204d54fe7ea0c56132efa8e38b234704",
    "8b225ebb888c282c0e6293215cdaecb36427efcbe3ca7317bbf322e5e1888c49",
    "a5979ba178a792517f4e675fba66fe1a61f96c1a383e8453f32c5320f22fd04a",
    "a8102ce87ef9940510e2921ce401d3a66aa1b6ac08a6865fc84b733bc322cd71",
    "c5bc2f2af0a579325f857006e222f4f59f4f44bd58eceaa0dd8e82baba16f9c5",
)


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _transfer_event(
    *,
    ordinal: int,
    phase: str,
    previous_event_sha256: str,
) -> cohort.C2ExecutionEventV1:
    record: dict[str, object] = {
        "ordinal": ordinal,
        "domain": cohort.TRANSFER,
        "phase": phase,
        "previous_event_sha256": previous_event_sha256,
        "candidate_sha256": _root(f"event:{ordinal}:candidate"),
        "authority_sha256": _root(f"event:{ordinal}:authority"),
        "catalogue_sha256": _root(f"event:{ordinal}:catalogue"),
        "intervention_sha256": _root(f"event:{ordinal}:intervention"),
        "selection_sha256": _root(f"event:{ordinal}:selection"),
        "selected_source_grant_sha256": _root(f"event:{ordinal}:grant"),
        "release_sha256": _root(f"event:{ordinal}:release"),
        "source_frame_sha256": _root(f"event:{ordinal}:source-frame"),
        "source_sha256": _root(f"event:{ordinal}:source"),
        "batch_sha256": _root(f"event:{ordinal}:batch"),
        "stdin_sha256": _root(f"event:{ordinal}:stdin"),
        "execution_sha256": _root(f"event:{ordinal}:execution"),
        "observed_output_sha256": _root(f"event:{ordinal}:output"),
    }
    event = cohort.C2ExecutionEventV1(
        **record,
        sha256=cohort._digest(cohort._EVENT_NAMESPACE, record),
    )
    event.authenticate()
    return event


def _seal_lexical_episode(
    material: cohort.C2EpisodeMaterialV1,
    *,
    failed_source: str,
    successful_source: str,
) -> continual.FailedSuccessfulTokenEditEpisodeV1:
    failed_frame = continual.project_token_source_v1(failed_source)
    successful_frame = continual.project_token_source_v1(successful_source)
    batch_sha256 = _root(f"c2-static:{material.label}:batch")
    failed_consequence = continual.make_token_edit_consequence_v1(
        source_sha256=failed_frame.source_sha256,
        batch_sha256=batch_sha256,
        test_manifest_sha256=material.test_manifest_sha256,
        execution_sha256=_root(f"c2-static:{material.label}:failed"),
        observed_outputs=material.batch,
        expected_outputs=material.expected_outputs,
    )
    successful_consequence = continual.make_token_edit_consequence_v1(
        source_sha256=successful_frame.source_sha256,
        batch_sha256=batch_sha256,
        test_manifest_sha256=material.test_manifest_sha256,
        execution_sha256=_root(f"c2-static:{material.label}:successful"),
        observed_outputs=material.expected_outputs,
        expected_outputs=material.expected_outputs,
    )
    return continual.seal_failed_successful_token_edit_episode_v1(
        repository_sha256=material.repository_sha256,
        task_sha256=material.task_sha256,
        task_instance_sha256=material.task_instance_sha256,
        task_source_sha256=material.task_source_sha256,
        test_source_sha256=material.test_manifest_sha256,
        failed_source=failed_frame,
        failed_consequence=failed_consequence,
        successful_source=successful_frame,
        successful_consequence=successful_consequence,
    )


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(keys for item in value.values() if (keys := _nested_keys(item))))
    if isinstance(value, (list, tuple)):
        return set().union(*(_nested_keys(item) for item in value))
    return set()


def test_frozen_public_spec_pins_fresh_material_without_answer_bytes() -> None:
    spec = cohort.DEFAULT_PUBLIC_SPEC
    spec.authenticate()

    assert spec.sha256 == EXPECTED_SPEC_SHA256
    assert spec.frozen_common_file_sha256 == cohort.FROZEN_COMMON_FILE_SHA256
    assert spec.frozen_learner_config_sha256 == continual.CONFIG_SHA256
    assert spec.frozen_mechanism_commit == "f2de425d5e06104538a0260da30ff0b287aa5f55"
    assert (spec.frozen_schema_index, spec.frozen_schema_sha256) == (
        3,
        "7cf44511bced68635e3c77a199542061b404b5861fe6a51c7c6ba88b01c2b014",
    )

    actual_episode_roots = tuple(
        (
            material.sha256,
            material.repository_sha256,
            material.task_sha256,
            material.task_instance_sha256,
            material.task_source_sha256,
            material.test_manifest_sha256,
            material.failed_source_sha256,
            material.successful_source_sha256,
        )
        for material in spec.episodes
    )
    transfer = spec.transfer
    actual_transfer_roots = (
        transfer.sha256,
        transfer.repository_sha256,
        transfer.task_sha256,
        transfer.task_instance_sha256,
        transfer.task_source_sha256,
        transfer.test_manifest_sha256,
        transfer.input_source_sha256,
    )
    assert actual_episode_roots == EXPECTED_EPISODE_ROOTS
    assert actual_transfer_roots == EXPECTED_TRANSFER_ROOTS
    assert spec.excluded_c1_source_sha256s == EXPECTED_C1_EXCLUSIONS
    assert not set(spec.training_source_sha256s).intersection(EXPECTED_C1_EXCLUSIONS)
    assert transfer.input_source_sha256 not in EXPECTED_C1_EXCLUSIONS

    record = spec.record()
    keys = _nested_keys(record)
    assert "expected_edited_source" not in keys
    assert "expected_edited_source_sha256" not in keys
    assert record["expected_edited_source_committed"] is False
    assert record["expected_edited_source_sha256_committed"] is False
    assert record["transfer_oracle_outputs_committed"] is False
    assert record["transfer_oracle_output_sha256_committed"] is False
    assert "expected_outputs" not in record["transfer"]


def test_fresh_episodes_replicate_singleton_schema_and_bind_transfer_query() -> None:
    spec = cohort.DEFAULT_PUBLIC_SPEC
    episode_a = _seal_lexical_episode(
        spec.episodes[0],
        failed_source=cohort.A_FAILED_SOURCE,
        successful_source=cohort.A_SUCCESSFUL_SOURCE,
    )
    episode_b = _seal_lexical_episode(
        spec.episodes[1],
        failed_source=cohort.B_FAILED_SOURCE,
        successful_source=cohort.B_SUCCESSFUL_SOURCE,
    )
    query = continual.project_token_source_v1(cohort.TRANSFER_SOURCE)

    empty = continual.start_continual_token_edit_schema_v1()
    one_source = continual.advance_continual_token_edit_schema_v1(empty, episode_a)
    assert one_source.provenance[0].matched_schema_indices == (3,)
    assert one_source.support_counts[3] == 1
    one_source_proposal = one_source.propose(query)
    assert one_source_proposal.status == continual.STOP_INSUFFICIENT_INDEPENDENT_SOURCES
    assert one_source_proposal.edited_source is None

    ready = continual.advance_continual_token_edit_schema_v1(one_source, episode_b)
    assert ready.provenance[1].matched_schema_indices == (3,)
    assert ready.support_counts[3] == 2
    assert ready.training_summary.promoted_schema_indices == (3,)

    cell_off = ready.propose(query, cell_gate=False)
    schema_off = ready.propose(query, schema_gate=False)
    assert (cell_off.status, cell_off.edited_source) == (continual.STOP_CELL_OFF, None)
    assert (schema_off.status, schema_off.edited_source) == (
        continual.STOP_SCHEMA_OFF,
        None,
    )

    proposal = ready.propose(query)
    assert proposal.status == continual.READY
    assert (proposal.selected_schema_index, proposal.selected_schema_sha256) == (
        3,
        cohort.FROZEN_SCHEMA_SHA256,
    )
    assert (
        proposal.anchor_token_index,
        proposal.target_token_index,
        proposal.donor_token_index,
    ) == (75, 76, 78)
    assert query.tokens[75:79] == (">=", "signal", "&&", "threshold")
    assert proposal.edited_source is not None
    edited = continual.project_token_source_v1(proposal.edited_source)
    changed_indices = tuple(
        index
        for index, (before, after) in enumerate(zip(query.tokens, edited.tokens, strict=True))
        if before != after
    )
    assert changed_indices == (proposal.target_token_index,)
    assert edited.tokens[proposal.target_token_index] == query.tokens[proposal.donor_token_index]
    assert proposal.input_source_sha256 == spec.transfer.input_source_sha256
    proposal.authenticate()
    ready.authenticate(expected_checkpoint_sha256=ready.checkpoint_sha256)


def test_transfer_oracle_refuses_early_then_opens_once_after_exact_event_chain() -> None:
    retained_events: list[cohort.C2ExecutionEventV1] = []
    proposal_event_sha256 = _root("event-chain:proposal")
    open_once = cohort._make_guarded_transfer_oracle_v1(
        proposal_event_sha256,
        lambda: tuple(retained_events),
    )

    with pytest.raises(
        cohort.ContinualRustTokenEditSchemaTransferCohortError,
        match="requires two retained execution events",
    ):
        open_once()

    baseline_event = _transfer_event(
        ordinal=6,
        phase=cohort.BASELINE,
        previous_event_sha256=_root("event-chain:detached-proposal"),
    )
    retained_events.append(baseline_event)
    with pytest.raises(
        cohort.ContinualRustTokenEditSchemaTransferCohortError,
        match="requires two retained execution events",
    ):
        open_once()

    retained_events.append(
        _transfer_event(
            ordinal=7,
            phase=cohort.PROPOSAL,
            previous_event_sha256=baseline_event.sha256,
        )
    )

    with pytest.raises(
        cohort.ContinualRustTokenEditSchemaTransferCohortError,
        match="event chain changed",
    ):
        open_once()

    baseline_event = _transfer_event(
        ordinal=6,
        phase=cohort.BASELINE,
        previous_event_sha256=proposal_event_sha256,
    )
    retained_events[:] = [
        baseline_event,
        _transfer_event(
            ordinal=7,
            phase=cohort.PROPOSAL,
            previous_event_sha256=baseline_event.sha256,
        ),
    ]
    assert open_once() == ((0, 1, 1, 0), (1, 0, 1))
    with pytest.raises(
        cohort.ContinualRustTokenEditSchemaTransferCohortError,
        match="already opened",
    ):
        open_once()


def test_authority_snapshot_rejects_rehashed_hidden_attempts_and_binds_predecessor() -> None:
    selection_roots = (_root("snapshot:selection:0"), _root("snapshot:selection:1"))
    execution_roots = (_root("snapshot:execution:0"), _root("snapshot:execution:1"))
    event_roots = (_root("snapshot:event:0"), _root("snapshot:event:1"))
    record: dict[str, object] = {
        "domain": cohort.TRANSFER,
        "previous_event_sha256": event_roots[-1],
        "authority_sha256": _root("snapshot:authority"),
        "catalogue_sha256": _root("snapshot:catalogue"),
        "issued_selection_sha256s": selection_roots,
        "spent_selection_sha256s": selection_roots,
        "successful_execution_sha256s": execution_roots,
        "execution_event_sha256s": event_roots,
        "issued_selection_count": 2,
        "spent_selection_count": 2,
        "source_release_count": 2,
        "execution_attempt_count": 2,
        "successful_execution_count": 2,
    }
    snapshot = cohort.C2AuthorityWorkSnapshotV1(
        **record,
        sha256=cohort._digest(cohort._AUTHORITY_SNAPSHOT_NAMESPACE, record),
    )
    snapshot.authenticate()

    rehashed_hidden_attempt = replace(snapshot, execution_attempt_count=3, sha256="")
    rehashed_hidden_attempt = replace(
        rehashed_hidden_attempt,
        sha256=cohort._digest(
            cohort._AUTHORITY_SNAPSHOT_NAMESPACE,
            rehashed_hidden_attempt.record(include_sha256=False),
        ),
    )
    with pytest.raises(
        cohort.ContinualRustTokenEditSchemaTransferCohortError,
        match="authority work snapshot authentication failed",
    ):
        rehashed_hidden_attempt.authenticate()

    rehashed_detached = replace(
        snapshot,
        previous_event_sha256=_root("snapshot:detached"),
        sha256="",
    )
    rehashed_detached = replace(
        rehashed_detached,
        sha256=cohort._digest(
            cohort._AUTHORITY_SNAPSHOT_NAMESPACE,
            rehashed_detached.record(include_sha256=False),
        ),
    )
    assert rehashed_detached.sha256 != snapshot.sha256
    with pytest.raises(
        cohort.ContinualRustTokenEditSchemaTransferCohortError,
        match="authority work snapshot authentication failed",
    ):
        rehashed_detached.authenticate()
    assert (
        "expected_result_sha256"
        in inspect.signature(cohort.C2TokenEditSchemaTransferResultV1.authenticate).parameters
    )
    assert "Git custody" in inspect.getdoc(cohort.C2TokenEditSchemaTransferResultV1.authenticate)


def test_authority_snapshot_link_rejects_fully_rehashed_extra_spent_selection() -> None:
    authority = SimpleNamespace(sha256=_root("aggregate:authority"))
    catalogue = SimpleNamespace(sha256=_root("aggregate:catalogue"))
    selections = tuple(
        SimpleNamespace(sha256=_root(f"aggregate:selection:{index}")) for index in range(2)
    )
    executions = tuple(
        SimpleNamespace(sha256=_root(f"aggregate:execution:{index}")) for index in range(2)
    )
    events = tuple(SimpleNamespace(sha256=_root(f"aggregate:event:{index}")) for index in range(2))
    record: dict[str, object] = {
        "domain": cohort.TRANSFER,
        "previous_event_sha256": events[-1].sha256,
        "authority_sha256": authority.sha256,
        "catalogue_sha256": catalogue.sha256,
        "issued_selection_sha256s": tuple(item.sha256 for item in selections),
        "spent_selection_sha256s": tuple(item.sha256 for item in selections),
        "successful_execution_sha256s": tuple(item.sha256 for item in executions),
        "execution_event_sha256s": tuple(item.sha256 for item in events),
        "issued_selection_count": 2,
        "spent_selection_count": 2,
        "source_release_count": 2,
        "execution_attempt_count": 2,
        "successful_execution_count": 2,
    }
    snapshot = cohort.C2AuthorityWorkSnapshotV1(
        **record,
        sha256=cohort._digest(cohort._AUTHORITY_SNAPSHOT_NAMESPACE, record),
    )
    cohort._authenticate_authority_snapshot_link_v1(
        snapshot=snapshot,
        domain=cohort.TRANSFER,
        previous_event_sha256=events[-1].sha256,
        authority=authority,
        catalogue=catalogue,
        selections=selections,
        executions=executions,
        events=events,
    )

    extra_selection_sha256 = _root("aggregate:selection:extra")
    rehashed_record = snapshot.record(include_sha256=False)
    rehashed_record.update(
        issued_selection_sha256s=(*snapshot.issued_selection_sha256s, extra_selection_sha256),
        spent_selection_sha256s=(*snapshot.spent_selection_sha256s, extra_selection_sha256),
        issued_selection_count=3,
        spent_selection_count=3,
    )
    rehashed_extra_selection = cohort.C2AuthorityWorkSnapshotV1(
        **rehashed_record,
        sha256=cohort._digest(cohort._AUTHORITY_SNAPSHOT_NAMESPACE, rehashed_record),
    )
    rehashed_extra_selection.authenticate()
    with pytest.raises(
        cohort.ContinualRustTokenEditSchemaTransferCohortError,
        match="snapshot differs from the retained causal work",
    ):
        cohort._authenticate_authority_snapshot_link_v1(
            snapshot=rehashed_extra_selection,
            domain=cohort.TRANSFER,
            previous_event_sha256=events[-1].sha256,
            authority=authority,
            catalogue=catalogue,
            selections=selections,
            executions=executions,
            events=events,
        )


def test_persisted_result_document_rederives_compact_root_and_transcript_links() -> None:
    compact_body: dict[str, object] = {
        "schema": cohort.SCHEMA,
        "claim_boundary": cohort.CLAIM_BOUNDARY,
        "status": cohort.STATUS_PASS,
        "spec_sha256": EXPECTED_SPEC_SHA256,
    }
    result_sha256 = cohort._digest(cohort._RESULT_NAMESPACE, compact_body)
    compact_record = {**compact_body, "sha256": result_sha256}
    transcript = {"status": cohort.STATUS_PASS, "sha256": result_sha256}
    body: dict[str, object] = {
        "schema": cohort.RESULT_DOCUMENT_SCHEMA,
        "terminal_status": cohort.STATUS_PASS,
        "result_sha256": result_sha256,
        "authenticated_result_record": compact_record,
        "full_authenticated_transcript": transcript,
    }
    document = {
        **body,
        "document_sha256": cohort._digest(cohort._DOCUMENT_NAMESPACE, body),
    }
    calls: list[str] = []
    cohort._verify_document_bytes(
        document=document,
        payload=cohort._canonical_json(document) + b"\n",
        authenticate_terminal=lambda: calls.append("authenticated"),
        expected_document_sha256=document["document_sha256"],
        expected_result_sha256=result_sha256,
    )
    assert calls == ["authenticated"]

    changed_compact = {**compact_record, "claim_boundary": "rehashed alternate"}
    changed_body = {**body, "authenticated_result_record": changed_compact}
    changed_document = {
        **changed_body,
        "document_sha256": cohort._digest(cohort._DOCUMENT_NAMESPACE, changed_body),
    }
    with pytest.raises(
        cohort.ContinualRustTokenEditSchemaTransferCohortError,
        match="persisted result record or transcript linkage changed",
    ):
        cohort._verify_document_bytes(
            document=changed_document,
            payload=cohort._canonical_json(changed_document) + b"\n",
            authenticate_terminal=lambda: None,
            expected_document_sha256=changed_document["document_sha256"],
            expected_result_sha256=result_sha256,
        )

    nonmapping_body = {**body, "full_authenticated_transcript": [transcript]}
    nonmapping_document = {
        **nonmapping_body,
        "document_sha256": cohort._digest(cohort._DOCUMENT_NAMESPACE, nonmapping_body),
    }
    with pytest.raises(
        cohort.ContinualRustTokenEditSchemaTransferCohortError,
        match="result document linkage changed",
    ):
        cohort._verify_document_bytes(
            document=nonmapping_document,
            payload=cohort._canonical_json(nonmapping_document) + b"\n",
            authenticate_terminal=lambda: None,
            expected_document_sha256=nonmapping_document["document_sha256"],
            expected_result_sha256=result_sha256,
        )


def test_public_surface_exposes_only_the_exact_default_path_writer(tmp_path: Path) -> None:
    assert not hasattr(cohort, "run_cohort_v1")
    writer_signature = inspect.signature(cohort.run_and_write_cohort_v1)
    assert tuple(writer_signature.parameters) == (
        "spec",
        "expected_result_sha256",
        "expected_document_sha256",
    )
    assert (
        writer_signature.parameters["expected_result_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        writer_signature.parameters["expected_document_sha256"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert "Git custody" in inspect.getdoc(cohort.run_and_write_cohort_v1)

    nondefault_path = tmp_path / "nondefault.result.json"
    with pytest.raises(TypeError, match="one exact transfer specification"):
        cohort.run_and_write_cohort_v1(nondefault_path)
    assert not nondefault_path.exists()

    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_evidence_directory = outside_directory / cohort.DEFAULT_RESULT_PATH.name
    with pytest.raises(ValueError, match="exact anchored evidence leaf"):
        cohort._publish_terminal_document(
            destination=outside_evidence_directory,
            expected_destination=cohort.DEFAULT_RESULT_PATH,
            document={},
            authenticate_terminal=lambda: None,
        )
    assert not outside_evidence_directory.exists()


def test_run_and_write_persists_sibling_reservation_and_pre_rust_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "c2-static-abort.result.json"
    reservation_path = Path(f"{output_path}.reservation.json")
    calls: list[str] = []

    def fail_before_rust(
        spec: cohort.C2TokenEditSchemaTransferSpecV1,
        *,
        ledger: cohort._RunLedgerV1,
    ) -> None:
        calls.append(spec.sha256)
        ledger.mark("SYNTHETIC_PRE_RUST_ABORT", transfer_events=())
        raise RuntimeError("synthetic pre-Rust protocol abort")

    monkeypatch.setattr(cohort, "_run_cohort_v1", fail_before_rust)

    with pytest.raises(RuntimeError, match="synthetic pre-Rust protocol abort"):
        cohort._run_and_write_at_paths_for_test_v1(
            destination=output_path,
            reservation_path=reservation_path,
        )

    assert reservation_path.parent == output_path.parent
    reservation_record = json.loads(reservation_path.read_text(encoding="utf-8"))
    reservation = cohort.C2RunReservationV1(**reservation_record)
    reservation.authenticate(expected_destination=output_path)
    assert reservation.destination == str(output_path)
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["schema"] == cohort.RESULT_DOCUMENT_SCHEMA
    assert document["terminal_status"] == cohort.STATUS_EXECUTION_ABORT
    abort = cohort.C2ExecutionAbortV1(**document["authenticated_abort"])
    abort.authenticate()
    assert abort.failure_kind == "PROTOCOL_ABORT"
    assert abort.partial_transcript["stage"] == "SYNTHETIC_PRE_RUST_ABORT"
    assert calls == [EXPECTED_SPEC_SHA256]

    with pytest.raises(FileExistsError):
        cohort._run_and_write_at_paths_for_test_v1(
            destination=output_path,
            reservation_path=reservation_path,
        )
    assert calls == [EXPECTED_SPEC_SHA256]


def test_pre_consume_failure_persists_issued_never_spent_abort_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "c2-pre-consume-abort.result.json"
    reservation_path = Path(f"{output_path}.reservation.json")

    def fail_before_consumption(
        prepared: cohort.intervention.PreparedRustActionInterventionV1,
        selection: cohort.intervention.RustActionInterventionSelectionV1,
    ) -> None:
        selection_sha256 = selection.sha256
        assert selection_sha256 in prepared._issued_selections
        assert selection_sha256 not in prepared._spent_selection_sha256s
        raise RuntimeError("synthetic failure before selection consumption")

    def forbid_executor(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Rust executor boundary must remain unreachable")

    monkeypatch.setattr(
        cohort.intervention.PreparedRustActionInterventionV1,
        "_consume_issued_selection",
        fail_before_consumption,
    )
    monkeypatch.setattr(cohort.intervention, "_execute_prepared", forbid_executor)

    with pytest.raises(RuntimeError, match="before selection consumption"):
        cohort._run_and_write_at_paths_for_test_v1(
            destination=output_path,
            reservation_path=reservation_path,
        )

    document_text = output_path.read_text(encoding="utf-8")
    document = json.loads(document_text)
    abort = cohort.C2ExecutionAbortV1(**document["authenticated_abort"])
    abort.authenticate()
    assert abort.partial_transcript["stage"] == "TRAINING_SELECTION_1_ISSUED"
    assert "SPENT" not in abort.partial_transcript["stage"]
    assert "SELECTION_SPENT" not in document_text
