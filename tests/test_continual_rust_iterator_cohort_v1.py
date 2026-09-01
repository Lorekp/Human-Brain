from __future__ import annotations

import inspect
import json
from dataclasses import asdict, replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import continual_rust_iterator_cohort_v1 as cohort


def _root(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _synthetic_ready() -> cohort.ReadyUnspentV1:
    payload = {
        "schema": cohort.READY_SCHEMA,
        "status": "READY_UNSPENT",
        "claim_boundary": cohort.CLAIM_BOUNDARY,
        "implementation_sha256": _root("synthetic-implementation"),
        "implementation_file_sha256s": tuple(
            (path, _root(f"synthetic-implementation-file-{index}"))
            for index, path in enumerate(cohort.IMPLEMENTATION_RELATIVE_PATHS)
        ),
        "batch_sha256": cohort.EXPECTED_BATCH_SHA256,
        "oracle_filename": cohort.ORACLE_PATH.name,
        "oracle_sha256": cohort.EXPECTED_ORACLE_SHA256,
        "rust_book_revision": cohort.RUST_BOOK_REVISION,
        "rust_book_iterator_url": cohort.RUST_BOOK_ITERATOR_URL,
        "rust_book_iterator_source_sha256": cohort.RUST_BOOK_ITERATOR_SOURCE_SHA256,
        "rust_book_file_inventory_sha256": cohort.RUST_BOOK_FILE_INVENTORY_SHA256,
        "lesson_sha256s": tuple(_root(f"synthetic-lesson-{index}") for index in range(4)),
        "checkpoint_sha256s": tuple(_root(f"synthetic-checkpoint-{index}") for index in range(4)),
        "final_checkpoint_array_sha256s": tuple(
            _root(f"synthetic-array-{index}") for index in range(3)
        ),
        "candidate_sha256s": tuple(_root(f"synthetic-candidate-{index}") for index in range(2)),
        "candidate_projection_source_sha256s": tuple(
            _root(f"synthetic-projection-source-{index}") for index in range(2)
        ),
        "promoted_candidate_sha256s": tuple(
            _root(f"synthetic-candidate-{index}") for index in range(2)
        ),
        "provenance_sha256s": tuple(_root(f"synthetic-provenance-{index}") for index in range(4)),
        "vault_receipt_sha256": _root("synthetic-vault-receipt"),
        "vault_binding_sha256s": tuple(
            _root(f"synthetic-vault-binding-{index}") for index in range(2)
        ),
        "actual_rust_source_sha256s": tuple(
            _root(f"synthetic-rust-source-{index}") for index in range(2)
        ),
        "rustc_identity_sha256": _root("synthetic-rustc-identity"),
        "v2_preparation_template_sha256s": tuple(
            _root(f"synthetic-preparation-template-{index}") for index in range(2)
        ),
        "v2_batch_sha256s": tuple(_root(f"synthetic-v2-batch-{index}") for index in range(2)),
        "v2_stdin_sha256s": tuple(_root(f"synthetic-v2-stdin-{index}") for index in range(2)),
        "work_counters": cohort.READY_COUNTERS,
    }
    return cohort.ReadyUnspentV1(
        **payload,
        sha256=cohort._digest(cohort._READY_NAMESPACE, payload),
    )


def _synthetic_protocol_ready(prepared) -> cohort.ReadyUnspentV1:
    payload = _synthetic_ready().record(include_sha256=False)
    payload.update(
        {
            "lesson_sha256s": tuple(lesson.sha256 for lesson in prepared.lessons),
            "checkpoint_sha256s": prepared.checkpoint_sha256s,
            "final_checkpoint_array_sha256s": prepared.memory.checkpoint_array_sha256s,
            "candidate_sha256s": tuple(
                candidate.sha256 for candidate in prepared.memory.candidate_projections
            ),
            "candidate_projection_source_sha256s": tuple(
                candidate.source_sha256 for candidate in prepared.memory.candidate_projections
            ),
            "promoted_candidate_sha256s": prepared.memory.promoted_candidate_sha256s,
            "provenance_sha256s": tuple(row.sha256 for row in prepared.memory.provenance),
            "vault_receipt_sha256": prepared.vault.receipt.sha256,
            "vault_binding_sha256s": prepared.vault.receipt.binding_sha256s,
            "actual_rust_source_sha256s": prepared.vault.receipt.rust_source_sha256s,
        }
    )
    return cohort.ReadyUnspentV1(
        **payload,
        sha256=cohort._digest(cohort._READY_NAMESPACE, payload),
    )


def test_synthetic_ready_authenticates_zero_evaluation_work_and_detects_tamper() -> None:
    ready = _synthetic_ready()
    ready.authenticate_structure()
    counters = dict(ready.work_counters)
    assert counters["synthetic_training_lesson_count"] == 4
    assert counters["training_compile_invocation_count"] == 0
    assert counters["training_run_invocation_count"] == 0
    assert counters["evaluation_vault_recall_count"] == 0
    assert counters["oracle_read_count"] == 0

    with pytest.raises(cohort.ContinualRustIteratorCohortError):
        replace(ready, oracle_sha256=_root("changed-oracle")).authenticate_structure()


def test_synthetic_ready_round_trips_through_published_json_shape() -> None:
    ready = _synthetic_ready()
    serialized = json.loads(json.dumps(ready.record()))

    assert cohort._ready_from_record(serialized) == ready


def test_oracle_gate_rejects_early_access_before_touching_a_file(tmp_path) -> None:
    missing = tmp_path / "synthetic-oracle.json"
    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="two exact authenticated V2 execution trees",
    ):
        cohort._authenticate_oracle_gate(
            (object(), object()),
            (object(), object()),
            ready=_synthetic_ready(),
            authority_sha256=_root("synthetic-authority"),
        )
    assert not missing.exists()


def test_oracle_parser_accepts_synthetic_bytes_without_file_access() -> None:
    payload = {
        "schema": cohort.ORACLE_SCHEMA,
        "rows": [
            {"name": "X_prime", "expected_outputs": [[101], [-202, 303]]},
            {"name": "Y_prime", "expected_outputs": [[], [404]]},
        ],
    }
    raw = json.dumps(payload, indent=2).encode("utf-8") + b"\n"

    rows = cohort._parse_oracle_bytes(
        raw,
        expected_sha256=sha256(raw).hexdigest(),
        expected_names=("X_prime", "Y_prime"),
    )

    assert rows == (
        ("X_prime", ((101,), (-202, 303))),
        ("Y_prime", ((), (404,))),
    )


def test_exclusive_spend_writer_never_overwrites_existing_bytes(tmp_path) -> None:
    path = tmp_path / "synthetic-spent.json"
    first = {"status": "SPENT", "sha256": _root("first")}
    file_sha256 = cohort._write_exclusive_json(path, first)
    original = path.read_bytes()
    assert file_sha256 == sha256(original).hexdigest()

    with pytest.raises(cohort.ContinualRustIteratorCohortError, match="already exists"):
        cohort._write_exclusive_json(
            path,
            {"status": "RETRY", "sha256": _root("second")},
        )

    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob("*.pending"))


def test_exclusive_writer_cleans_pending_file_when_publication_fails(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "synthetic-result.json"

    def fail_publication(_pending, _path) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(cohort, "_publish_pending_exclusive", fail_publication)
    with pytest.raises(OSError, match="synthetic publication failure"):
        cohort._write_exclusive_json(path, {"status": "STOP"})

    assert not path.exists()
    assert not tuple(tmp_path.glob("*.pending"))


def test_reserved_terminal_path_is_held_and_filled_exactly_once(tmp_path) -> None:
    path = tmp_path / "synthetic-result.json"
    terminal = {"status": "STOP", "sha256": _root("synthetic-terminal")}
    expected = cohort._canonical_json(terminal) + b"\n"
    reservation = cohort._reserve_result_path(path)
    try:
        if reservation.windows:
            with pytest.raises(OSError):
                path.read_bytes()
            with pytest.raises(OSError):
                path.write_bytes(b"{}\n")
        else:
            assert path.read_bytes() == b""
        file_sha256 = cohort._write_reserved_result_json(reservation, terminal)
        if reservation.windows:
            with pytest.raises(OSError):
                path.read_bytes()
        else:
            assert path.read_bytes() == expected
        assert file_sha256 == sha256(expected).hexdigest()
        with pytest.raises(
            cohort.ContinualRustIteratorCohortError,
            match="already consumed",
        ):
            cohort._write_reserved_result_json(reservation, terminal)
    finally:
        cohort._release_result_reservation(reservation)
    assert path.read_bytes() == expected

    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="already exists",
    ):
        cohort._reserve_result_path(path)


def test_empty_terminal_reservation_is_discarded_before_any_spend(tmp_path) -> None:
    path = tmp_path / "synthetic-unspent-result.json"
    spend_path = tmp_path / "synthetic-unspent-spend.json"
    reservation = cohort._reserve_result_path(path)

    cohort._discard_empty_result_reservation(reservation, spend_path=spend_path)

    assert reservation.released
    assert not path.exists()
    retry = cohort._reserve_result_path(path)
    cohort._discard_empty_result_reservation(retry, spend_path=spend_path)


def test_terminal_reservation_is_retained_when_a_spend_path_exists(tmp_path) -> None:
    path = tmp_path / "synthetic-ambiguous-result.json"
    spend_path = tmp_path / "synthetic-ambiguous-spend.json"
    reservation = cohort._reserve_result_path(path)
    spend_path.write_bytes(b"{}\n")
    try:
        with pytest.raises(
            cohort.ContinualRustIteratorCohortError,
            match="cannot be discarded",
        ):
            cohort._discard_empty_result_reservation(
                reservation,
                spend_path=spend_path,
            )
    finally:
        cohort._release_result_reservation(reservation)
    assert path.read_bytes() == b""


def test_current_spend_authentication_rejects_replacement(tmp_path) -> None:
    path = tmp_path / "synthetic-spent.json"
    ready = _synthetic_ready()
    spend = cohort._make_spend_record("a" * 40, ready)
    file_sha256 = cohort._write_exclusive_json(path, spend)
    cohort._authenticate_spend_current(
        path,
        expected=spend,
        expected_file_sha256=file_sha256,
    )

    path.write_bytes(b"{}\n")
    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="current spend marker authentication failed",
    ):
        cohort._authenticate_spend_current(
            path,
            expected=spend,
            expected_file_sha256=file_sha256,
        )


def test_spend_publication_state_distinguishes_absent_exact_and_ambiguous(tmp_path) -> None:
    path = tmp_path / "synthetic-spent.json"
    ready = _synthetic_ready()
    spend = cohort._make_spend_record("f" * 40, ready)
    expected_file_sha256 = sha256(cohort._canonical_json(spend) + b"\n").hexdigest()

    assert (
        cohort._spend_publication_state(
            path,
            expected=spend,
            expected_file_sha256=expected_file_sha256,
        )
        == "ABSENT"
    )
    path.write_bytes(b"{}\n")
    assert (
        cohort._spend_publication_state(
            path,
            expected=spend,
            expected_file_sha256=expected_file_sha256,
        )
        == "AMBIGUOUS"
    )
    path.unlink()
    cohort._write_exclusive_json(path, spend)
    assert (
        cohort._spend_publication_state(
            path,
            expected=spend,
            expected_file_sha256=expected_file_sha256,
        )
        == "EXACT"
    )


def test_exact_spend_without_a_guard_retains_result_and_publishes_no_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    ready = _synthetic_ready()
    spend_path = tmp_path / "synthetic-spent.json"
    result_path = tmp_path / "synthetic-result.json"
    prepared = SimpleNamespace(
        vault=SimpleNamespace(
            receipt=SimpleNamespace(sha256=ready.vault_receipt_sha256),
        )
    )
    guard_calls = 0
    terminal_writes = 0

    def fail_guard(*_args, **_kwargs):
        nonlocal guard_calls
        guard_calls += 1
        raise OSError("synthetic guard acquisition failure")

    def count_terminal_write(*_args, **_kwargs):
        nonlocal terminal_writes
        terminal_writes += 1
        raise AssertionError("terminal writer must not be reached")

    monkeypatch.setattr(cohort, "_assert_canonical_artifact_paths", lambda: None)
    monkeypatch.setattr(cohort, "load_ready_unspent_v1", lambda: ready)
    monkeypatch.setattr(cohort, "_verify_protocol_scope", lambda *_args: None)
    monkeypatch.setattr(cohort, "make_ready_unspent_v1", lambda: ready)
    monkeypatch.setattr(cohort, "_load_fresh_batch", lambda: ())
    monkeypatch.setattr(
        cohort,
        "_prepare_continual_rust_iterator_cohort_v1",
        lambda: prepared,
    )
    monkeypatch.setattr(cohort, "_training_observation", lambda _prepared: {})
    monkeypatch.setattr(cohort, "_vault_observation", lambda _prepared: {})
    monkeypatch.setattr(cohort, "_CANONICAL_SPEND_PATH", spend_path)
    monkeypatch.setattr(cohort, "_CANONICAL_RESULT_PATH", result_path)
    monkeypatch.setattr(cohort, "_acquire_spend_marker_guard", fail_guard)
    monkeypatch.setattr(cohort, "_write_reserved_result_json", count_terminal_write)

    with pytest.raises(OSError, match="synthetic guard acquisition failure"):
        cohort.run_one_shot(protocol_commit="9" * 40)

    assert guard_calls == 2
    assert terminal_writes == 0
    assert result_path.read_bytes() == b""
    assert spend_path.exists()


def test_held_spend_marker_denies_replacement_until_release(tmp_path) -> None:
    path = tmp_path / "synthetic-spent.json"
    ready = _synthetic_ready()
    spend = cohort._make_spend_record("c" * 40, ready)
    file_sha256 = cohort._write_exclusive_json(path, spend)
    original = path.read_bytes()
    guard = cohort._acquire_spend_marker_guard(
        path,
        expected=spend,
        expected_file_sha256=file_sha256,
    )
    try:
        if guard.windows:
            with pytest.raises(OSError):
                path.write_bytes(b"{}\n")
            assert path.read_bytes() == original
    finally:
        cohort._release_spend_marker_guard(guard)

    path.write_bytes(b"{}\n")
    assert path.read_bytes() == b"{}\n"


def test_public_artifact_path_redirection_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cohort, "SPEND_PATH", tmp_path / "alternate-spent.json")

    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="artifact paths were redirected",
    ):
        cohort.load_terminal_result_v1()


def test_terminal_loader_requires_the_current_exact_spend_marker(
    tmp_path,
    monkeypatch,
) -> None:
    ready = _synthetic_ready()
    spend_path = tmp_path / "synthetic-spent.json"
    result_path = tmp_path / "synthetic-result.json"
    result_path.write_bytes(b"{}\n")
    spend = cohort._make_spend_record("e" * 40, ready)
    spend_file_sha256 = sha256(cohort._canonical_json(spend) + b"\n").hexdigest()
    structural_result = {
        "ready_sha256": ready.sha256,
        "protocol_commit": "e" * 40,
        "spend_file_sha256": spend_file_sha256,
    }
    assert not inspect.signature(cohort.load_terminal_result_v1).parameters
    monkeypatch.setattr(cohort, "RESULT_PATH", result_path)
    monkeypatch.setattr(cohort, "_CANONICAL_RESULT_PATH", result_path)
    monkeypatch.setattr(cohort, "SPEND_PATH", spend_path)
    monkeypatch.setattr(cohort, "_CANONICAL_SPEND_PATH", spend_path)
    monkeypatch.setattr(cohort, "load_ready_unspent_v1", lambda: ready)
    monkeypatch.setattr(
        cohort,
        "_authenticate_terminal_result_structure_v1",
        lambda _payload: structural_result,
    )

    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="current spend marker could not be read",
    ):
        cohort.load_terminal_result_v1()

    spend_path.write_bytes(b"{}\n")
    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="current spend marker authentication failed",
    ):
        cohort.load_terminal_result_v1()

    spend_path.unlink()
    cohort._write_exclusive_json(spend_path, spend)
    assert cohort.load_terminal_result_v1() == structural_result


def test_forged_empty_pass_is_rejected_even_with_recomputed_terminal_root() -> None:
    ready = _synthetic_ready()
    spend = cohort._make_spend_record("b" * 40, ready)
    spend_file_sha256 = sha256(cohort._canonical_json(spend) + b"\n").hexdigest()
    work = cohort._initial_work()
    work.update(
        {
            "evaluation_vault_recall_attempt_count": 5,
            "evaluation_vault_recall_count": 5,
            "evaluation_control_recall_count": 3,
            "evaluation_executor_preparation_attempt_count": 2,
            "evaluation_executor_preparation_count": 2,
            "evaluation_executor_call_count": 2,
            "evaluation_release_attempt_count": 2,
            "evaluation_authenticated_release_count": 2,
            "evaluation_authenticated_compile_invocation_count": 2,
            "evaluation_authenticated_run_invocation_count": 2,
            "oracle_open_attempt_count": 1,
            "oracle_read_count": 1,
        }
    )
    forged = cohort._terminal_record(
        status="PASS",
        protocol_commit="b" * 40,
        ready=ready,
        spend_sha256=spend["sha256"],
        spend_file_sha256=spend_file_sha256,
        events=list(cohort.PASS_EVENTS),
        work=work,
        observations={},
        error=None,
    )

    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="base observations are incomplete",
    ):
        cohort._authenticate_terminal_result_structure_v1(forged)


def test_failure_receipt_binds_observations_and_work_to_event_phase() -> None:
    prepared = cohort._prepare_continual_rust_iterator_cohort_v1()
    ready = _synthetic_protocol_ready(prepared)
    ready.authenticate_structure()
    spend = cohort._make_spend_record("d" * 40, ready)
    spend_file_sha256 = sha256(cohort._canonical_json(spend) + b"\n").hexdigest()
    observations = {
        "ready": ready.record(),
        "training": cohort._training_observation(prepared),
        "vault": cohort._vault_observation(prepared),
    }
    work = cohort._initial_work()
    work["evaluation_vault_recall_attempt_count"] = 1
    failure = cohort._terminal_record(
        status="STOP_EXECUTION_PROTOCOL_FAILURE",
        protocol_commit="d" * 40,
        ready=ready,
        spend_sha256=spend["sha256"],
        spend_file_sha256=spend_file_sha256,
        events=list(cohort.PASS_EVENTS[:4]),
        work=work,
        observations=observations,
        error={"type": "RuntimeError", "message": "synthetic recall failure"},
    )
    assert cohort._authenticate_terminal_result_structure_v1(failure)["status"].startswith("STOP_")

    extra_observation = cohort._terminal_record(
        status="STOP_EXECUTION_PROTOCOL_FAILURE",
        protocol_commit="d" * 40,
        ready=ready,
        spend_sha256=spend["sha256"],
        spend_file_sha256=spend_file_sha256,
        events=list(cohort.PASS_EVENTS[:4]),
        work=work,
        observations={**observations, "executions": ()},
        error={"type": "RuntimeError", "message": "synthetic recall failure"},
    )
    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="failure observations",
    ):
        cohort._authenticate_terminal_result_structure_v1(extra_observation)

    impossible_work = dict(work)
    impossible_work["evaluation_vault_recall_count"] = 1
    impossible_work_receipt = cohort._terminal_record(
        status="STOP_EXECUTION_PROTOCOL_FAILURE",
        protocol_commit="d" * 40,
        ready=ready,
        spend_sha256=spend["sha256"],
        spend_file_sha256=spend_file_sha256,
        events=list(cohort.PASS_EVENTS[:4]),
        work=impossible_work,
        observations=observations,
        error={"type": "RuntimeError", "message": "synthetic recall failure"},
    )
    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="failure work differs",
    ):
        cohort._authenticate_terminal_result_structure_v1(impossible_work_receipt)


def test_failure_observations_require_the_authenticated_execution_prefix(
    monkeypatch,
) -> None:
    prepared = cohort._prepare_continual_rust_iterator_cohort_v1()
    ready = _synthetic_protocol_ready(prepared)
    observed_counts: list[int] = []

    monkeypatch.setattr(
        cohort,
        "_recalls_from_observation",
        lambda _value, *, prepared: {"A_prime": object(), "B_prime": object()},
    )

    def execution_prefix(value, *, expected_count, **_custody):
        records = tuple(value)
        if len(records) != expected_count:
            raise cohort.ContinualRustIteratorCohortError(
                "synthetic execution prefix count changed"
            )
        observed_counts.append(expected_count)
        return tuple(object() for _record in records)

    monkeypatch.setattr(cohort, "_execution_prefix_from_observation", execution_prefix)

    base_observations = {
        "ready": ready.record(),
        "training": {},
        "vault": {},
        "recalls": {},
    }
    work_a = cohort._initial_work()
    work_a.update(
        {
            "evaluation_vault_recall_attempt_count": 5,
            "evaluation_vault_recall_count": 5,
            "evaluation_control_recall_count": 3,
            "evaluation_executor_preparation_attempt_count": 2,
            "evaluation_executor_preparation_count": 2,
            "evaluation_executor_call_count": 1,
            "evaluation_release_attempt_count": 1,
            "evaluation_authenticated_release_count": 1,
            "evaluation_authenticated_compile_invocation_count": 1,
            "evaluation_authenticated_run_invocation_count": 1,
        }
    )
    work_pair = dict(work_a)
    work_pair.update(
        {
            "evaluation_executor_call_count": 2,
            "evaluation_release_attempt_count": 2,
            "evaluation_authenticated_release_count": 2,
            "evaluation_authenticated_compile_invocation_count": 2,
            "evaluation_authenticated_run_invocation_count": 2,
        }
    )

    cohort._authenticate_failure_observations(
        {"observations": {**base_observations, "executions": ({"slot": "A"},)}},
        events=cohort.PASS_EVENTS[:8],
        ready=ready,
        prepared=prepared,
        authority=prepared.vault.authority,
        work=work_a,
    )
    cohort._authenticate_failure_observations(
        {
            "observations": {
                **base_observations,
                "executions": ({"slot": "A"}, {"slot": "B"}),
            }
        },
        events=cohort.PASS_EVENTS[:9],
        ready=ready,
        prepared=prepared,
        authority=prepared.vault.authority,
        work=work_pair,
    )
    assert observed_counts == [1, 2]

    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="failure observations",
    ):
        cohort._authenticate_failure_observations(
            {"observations": base_observations},
            events=cohort.PASS_EVENTS[:8],
            ready=ready,
            prepared=prepared,
            authority=prepared.vault.authority,
            work=work_a,
        )


def test_rehydrators_round_trip_noncanonical_control_recall_and_vault() -> None:
    prepared = cohort._prepare_continual_rust_iterator_cohort_v1()
    task = cohort.knowledge_action.project_knowledge_action_task_v1(
        "Reverse the letters in this unrelated synthetic control."
    )
    recall = prepared.vault.recall(task, cell_gate=False)

    serialized_recall = json.loads(json.dumps(recall.record()))
    serialized_receipt = json.loads(json.dumps(asdict(prepared.vault.receipt)))
    serialized_authority = json.loads(json.dumps(asdict(prepared.vault.authority)))

    assert cohort._recall_from_record(serialized_recall) == recall
    assert cohort._vault_receipt_from_record(serialized_receipt) == prepared.vault.receipt
    assert cohort._vault_authority_from_record(serialized_authority) == prepared.vault.authority


def test_execution_prefix_requires_exact_order_without_running_rust(monkeypatch) -> None:
    record_a = {"slot": "A"}
    record_b = {"slot": "B"}
    selections = {"A_prime": object(), "B_prime": object()}
    authority = type("SyntheticAuthority", (), {"sha256": _root("authority")})()
    rehydrated: list[tuple[object, str]] = []
    authenticated: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
    released: list[int] = []

    def rehydrate(record, *, expected_source):
        rehydrated.append((record, expected_source))
        return (record["slot"], expected_source)

    def authenticate_prefix(executions, selected, *, ready, authority_sha256):
        assert ready == _synthetic_ready()
        assert authority_sha256 == authority.sha256
        authenticated.append((executions, selected))

    def authenticate_release(**values):
        released.append(values["action_index"])

    monkeypatch.setattr(cohort, "_execution_from_record", rehydrate)
    monkeypatch.setattr(cohort, "_authenticate_execution_prefix", authenticate_prefix)
    monkeypatch.setattr(cohort, "_authenticate_release_root", authenticate_release)

    one = cohort._execution_prefix_from_observation(
        (record_a,),
        expected_count=1,
        recalls=selections,
        ready=_synthetic_ready(),
        prepared=object(),
        authority=authority,
    )
    pair = cohort._execution_prefix_from_observation(
        (record_a, record_b),
        expected_count=2,
        recalls=selections,
        ready=_synthetic_ready(),
        prepared=object(),
        authority=authority,
    )

    assert one == (("A", cohort.MAP_RUST_SOURCE),)
    assert pair == (
        ("A", cohort.MAP_RUST_SOURCE),
        ("B", cohort.FILTER_RUST_SOURCE),
    )
    assert rehydrated == [
        (record_a, cohort.MAP_RUST_SOURCE),
        (record_a, cohort.MAP_RUST_SOURCE),
        (record_b, cohort.FILTER_RUST_SOURCE),
    ]
    assert authenticated == [
        (one, (selections["A_prime"],)),
        (pair, (selections["A_prime"], selections["B_prime"])),
    ]
    assert released == [0, 0, 1]

    with pytest.raises(
        cohort.ContinualRustIteratorCohortError,
        match="execution count changed",
    ):
        cohort._execution_prefix_from_observation(
            (record_a, record_b),
            expected_count=1,
            recalls=selections,
            ready=_synthetic_ready(),
            prepared=object(),
            authority=authority,
        )


def test_canonical_runner_orders_all_recalls_before_execution_and_oracle_last() -> None:
    signature = inspect.signature(cohort.run_one_shot)
    assert tuple(signature.parameters) == ("protocol_commit",)
    assert signature.parameters["protocol_commit"].kind is inspect.Parameter.KEYWORD_ONLY

    source = inspect.getsource(cohort.run_one_shot)
    result_reservation = source.index("_reserve_result_path(_CANONICAL_RESULT_PATH)")
    spend_publication = source.index("_write_exclusive_json(_CANONICAL_SPEND_PATH, spend)")
    last_recall = source.index('recalls["B_prime"] = prepared.vault.recall')
    first_executor = source.index(
        "rust_selected_action_executor.prepare_rust_selected_action_executor_v2"
    )
    record_a = source.index('observations["executions"] = (execution_record_a,)')
    event_a = source.index('events.append("A_PRIME_EXECUTION_AUTHENTICATED")')
    record_pair = source.index(
        'observations["executions"] = (execution_record_a, execution_record_b)'
    )
    second_execution_authenticated = source.index(
        'events.append("B_PRIME_EXECUTION_AUTHENTICATED")'
    )
    oracle_gate = source.index("_authenticate_oracle_gate(")
    oracle_call = source.index("raw_oracle = _CANONICAL_ORACLE_PATH.read_bytes()")

    assert result_reservation < spend_publication < last_recall < first_executor
    assert record_a < event_a < record_pair < second_execution_authenticated
    assert second_execution_authenticated < oracle_gate < oracle_call
    assert source.count("_CANONICAL_ORACLE_PATH.read_bytes()") == 1
    assert source.count(".execute_selected_once(") == 2
    assert "--result" not in inspect.getsource(cohort._main)
