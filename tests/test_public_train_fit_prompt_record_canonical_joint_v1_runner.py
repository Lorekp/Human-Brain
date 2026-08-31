from __future__ import annotations

from hashlib import sha256
from typing import NamedTuple

import pytest
from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as occurrence_static,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v1_protocol as protocol,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v1_runner as runner,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as prompt_producer,
)
from experiments.r3_pilot import (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1,
    typed_semantic_packed_public_train_fit_prompt_record_authorized_plan_frontier_v1,
)

packed_converter = (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1
)
packed_frontier = (
    typed_semantic_packed_public_train_fit_prompt_record_authorized_plan_frontier_v1
)


def test_prespend_runner_binds_exact_payloads_and_uses_only_stub_frontiers() -> None:
    assert runner.PROTOCOL_BANK_COMMIT == (
        "c9ae0fdad7534a5b9ed44e6cab8d057afcd1f7f0"
    )
    assert runner.PROTOCOL_SOURCE_SHA256 == (
        "a149b3468a980517c30d7322a4247b9f7506306fc80f2ced4d0ff9ab479a97a0"
    )
    assert runner.PRESPEND_CONTRACT_SHA256 == (
        "03f8e0d944970c8532c765fd63c4e6ae61b38d87b2060fa9b992db06b51e31e1"
    )
    assert runner.RUN_COMMAND == (
        r".\.venv\Scripts\python.exe -m "
        "experiments.r3_pilot."
        "public_train_fit_prompt_record_canonical_joint_v1_runner"
    )
    assert not runner.RESULT_PATH.exists()
    assert not runner.SPENT_PATH.exists()

    def digest(label: str) -> str:
        return sha256(label.encode("ascii")).hexdigest()

    selector_calls: list[tuple[object, float]] = []
    real_selector = prompt_producer._is_selected
    selector_event = object()

    def selector_probe(event: object, *, cell_gain: float = 1.0) -> bool:
        selector_calls.append((event, cell_gain))
        return cell_gain > 0.0

    prompt_producer._is_selected = selector_probe
    try:
        with runner._real_intervention(runner.MODE_PRODUCER_OFF):
            assert not prompt_producer._is_selected(selector_event, cell_gain=7.0)
    finally:
        prompt_producer._is_selected = real_selector
    assert selector_calls == [(selector_event, 0.0)]

    class StaticReceipt(NamedTuple):
        status: str
        sha256: str

    class NoIteration:
        def __iter__(self) -> object:
            raise AssertionError("converter-off iterated packed inputs")

    incoming = object.__new__(occurrence_static.OccurrenceFactorStaticCertificate)
    incoming._receipt = StaticReceipt(occurrence_static.READY, digest("static-ready"))
    converter_type = (
        packed_converter.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1
    )
    cases = NoIteration()
    literals = NoIteration()
    with runner._real_intervention(runner.MODE_CONVERTER_OFF):
        converter_stopped = converter_type.build(
            object(),
            cases,
            literals,
            0,
            incoming,
        )
    assert converter_stopped.status == packed_converter.STOP_SOURCE_CERTIFICATE
    assert converter_stopped.state.plan_build_attempts == 0
    assert converter_stopped._source_certificate is not incoming
    assert type(converter_stopped._source_certificate) is type(incoming)
    assert (
        occurrence_static._ISSUED_CERTIFICATES.get(
            id(converter_stopped._source_certificate)
        )
        is None
    )

    class State(NamedTuple):
        status: str
        source_event_status: str | None
        occurrence_status: str | None
        source_candidate_event_count: int
        source_event_count: int
        occurrence_count: int
        producer_build_attempts: int
        occurrence_inference_attempts: int
        plan_build_attempts: int
        sha256: str

    class Relation(NamedTuple):
        completed_units: int

    class Work(NamedTuple):
        relation: Relation
        physical_predictions: int
        physical_races: int
        concrete_proof_witnesses: int
        sha256: str

    class Selection(NamedTuple):
        status: str
        build_state_sha256: str
        exact_proof_count: int
        witnesses: tuple[object, ...]
        selected_proof_forest_sha256: str | None
        work: Work

    issued: dict[int, object] = {}
    build_log: list[tuple[str, str, object]] = []
    selection_log: list[tuple[str, str, tuple[int, ...]]] = []
    occurrence_by_payload = {
        receipt.build_payload_sha256: expected
        for receipt, expected in zip(
            protocol.PROVENANCE_RECEIPTS,
            (9, 8),
            strict=True,
        )
    }

    class Frontier:
        __slots__ = ("mode", "payload_sha256", "sha256", "state", "work")

        def __init__(self, mode: str, payload_sha256: str) -> None:
            occurrences = occurrence_by_payload[payload_sha256]
            if mode == runner.MODE_PRODUCER_OFF:
                status = packed_frontier.STOP_SOURCE_EVENT_CERTIFICATE
                source_status = prompt_producer.STOP_ZERO
                occurrence_status = None
                source_count = occurrence_count = occurrence_attempts = plan_attempts = 0
            elif mode == runner.MODE_STATIC_OFF:
                status = packed_frontier.STOP_SOURCE_EVENT_CERTIFICATE
                source_status = prompt_producer.STOP_UNSUPPORTED
                occurrence_status = None
                source_count = occurrence_count = occurrence_attempts = plan_attempts = 0
            elif mode == runner.MODE_CONVERTER_OFF:
                status = packed_converter.STOP_SOURCE_CERTIFICATE
                source_status = prompt_producer.READY
                occurrence_status = occurrence_static.READY
                source_count = occurrence_count = occurrences
                occurrence_attempts = 1
                plan_attempts = 0
            else:
                status = packed_frontier.READY
                source_status = prompt_producer.READY
                occurrence_status = occurrence_static.READY
                source_count = occurrence_count = occurrences
                occurrence_attempts = plan_attempts = 1
            self.mode = mode
            self.payload_sha256 = payload_sha256
            self.sha256 = digest(f"frontier:{mode}:{payload_sha256}")
            self.state = State(
                status,
                source_status,
                occurrence_status,
                protocol.PRODUCER_CANDIDATES_PER_ACTIVATION,
                source_count,
                occurrence_count,
                1,
                occurrence_attempts,
                plan_attempts,
                self.sha256,
            )
            self.work = (
                Work(Relation(100), 20, 5, 0, digest(f"build-work:{self.sha256}"))
                if status == packed_frontier.READY
                else None
            )
            issued[id(self)] = self

        @property
        def status(self) -> str:
            return self.state.status

    def build(
        payload: protocol.FrozenJointBuildPayload,
        mode: str,
    ) -> runner._BuildProduct:
        assert not hasattr(payload, "desired")
        frontier = Frontier(mode, payload.sha256)
        build_log.append((mode, payload.sha256, payload))
        return runner._BuildProduct(
            payload,
            frontier,
            runner._payload_field_sha256(payload),
        )

    def authenticate(frontier: object) -> None:
        assert issued.get(id(frontier)) is frontier

    def select(frontier: object, desired: tuple[int, ...]) -> Selection:
        assert type(frontier) is Frontier
        authenticate(frontier)
        selection_log.append((frontier.mode, frontier.payload_sha256, desired))
        row_ordinal = tuple(occurrence_by_payload).index(frontier.payload_sha256)
        if frontier.mode == runner.MODE_WRONG_DESIRED:
            status = (
                packed_converter.plan_slice.STOP_AUTHORIZED_ECLASS_MISMATCH
                if row_ordinal == 0
                else packed_converter.plan_slice.STOP_NO_EXACT
            )
            proof_count = 0
            witnesses: tuple[object, ...] = ()
            forest = None
        elif row_ordinal == 0:
            status = packed_converter.plan_slice.STOP_NO_EXACT
            proof_count = 0
            witnesses = ()
            forest = None
        else:
            status = packed_frontier.SELECTED
            proof_count = 1
            witnesses = (object(),)
            forest = digest(f"forest:{frontier.payload_sha256}")
        work = Work(
            Relation(120),
            21,
            6,
            len(witnesses),
            digest(f"selection-work:{frontier.mode}:{frontier.payload_sha256}"),
        )
        return Selection(
            status,
            frontier.sha256,
            proof_count,
            witnesses,
            forest,
            work,
        )

    backend = runner._Backend(
        "authority-preserving-focused-stub",
        False,
        build,
        authenticate,
        select,
    )
    prepared = runner._prepare_pre_spend()
    authority = runner._issue_stub_test_authority(prepared, backend)
    execution = runner._execute_prepared(prepared, authority, backend)

    assert execution.verdict == protocol.ONE_SHOT_STOP
    assert execution.stop_reason == "normal row 0 violated its frozen outcome"
    assert tuple((item.mode, item.row_ordinal) for item in execution.treatment_outcomes) == tuple(
        (mode, row) for mode in runner.TREATMENT_ORDER for row in (0, 1)
    )
    assert execution.work.producer_query_count == protocol.PRODUCER_QUERY_CAP
    assert execution.work.static_query_count == protocol.STATIC_QUERY_CAP
    assert execution.work.fitted_glif_step_count == protocol.FITTED_GLIF_STEP_CAP
    assert execution.work.plan_build_count == 4
    assert execution.proof_materialization_count == 1
    producer_off_outcomes = tuple(
        item
        for item in execution.treatment_outcomes
        if item.mode == runner.MODE_PRODUCER_OFF
    )
    assert tuple(
        item.source_candidate_event_count * occurrence_static.OUTPUT_STEPS
        for item in producer_off_outcomes
    ) == (16_860, 16_860)
    assert all(item.static_query_count == 0 for item in producer_off_outcomes)
    assert len(build_log) == 10
    assert len({id(item[2]) for item in build_log}) == 10
    assert [item[0] for item in selection_log] == [
        runner.MODE_WRONG_DESIRED,
        runner.MODE_WRONG_DESIRED,
        runner.MODE_NORMAL,
        runner.MODE_NORMAL,
    ]
    normal_outcomes = tuple(
        item for item in execution.treatment_outcomes if item.mode == runner.MODE_NORMAL
    )
    assert tuple((item.row_ordinal, item.public_status) for item in normal_outcomes) == (
        (0, packed_converter.plan_slice.STOP_NO_EXACT),
        (1, packed_frontier.SELECTED),
    )
    normal_payloads = tuple(
        payload
        for mode, payload, _desired in selection_log
        if mode == runner.MODE_NORMAL
    )
    assert normal_payloads == tuple(occurrence_by_payload)
    for payload_sha256 in occurrence_by_payload:
        wrong = next(
            desired
            for mode, payload, desired in selection_log
            if mode == runner.MODE_WRONG_DESIRED and payload == payload_sha256
        )
        normal = next(
            desired
            for mode, payload, desired in selection_log
            if mode == runner.MODE_NORMAL and payload == payload_sha256
        )
        assert wrong != normal

    prepared_receipt = runner._lookup_prepared(prepared)

    def splice(
        payload: protocol.FrozenJointBuildPayload,
        mode: str,
    ) -> runner._BuildProduct:
        frontier = Frontier(mode, payload.sha256)
        return runner._BuildProduct(
            object(),  # type: ignore[arg-type]
            frontier,
            runner._payload_field_sha256(payload),
        )

    splice_backend = runner._Backend(
        "payload-splice-stub",
        False,
        splice,
        authenticate,
        select,
    )
    with pytest.raises(runner.RunnerStop) as stopped:
        runner._build_bound(
            prepared_receipt.exact_preflight,
            protocol.ROWS[0],
            runner.MODE_NORMAL,
            splice_backend,
        )
    assert stopped.value.status == protocol.STOP_PROMPT_IO_PAIRING_MISMATCH

    assert not runner.RESULT_PATH.exists()
    assert not runner.SPENT_PATH.exists()
