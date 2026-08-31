from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Iterable
from types import ModuleType

from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as source,
)
from experiments.r3_pilot import (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1 as integrated,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

CASES = (
    ([],),
    ([0, 1, 2, 3],),
    ([1, 2, 10],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
LITERALS = (-3, -2, 0)
DESIRED = (0, 2, 2, 1, 3)
FORBIDDEN_RUNTIME_MODULES = (
    "experiments.r3_pilot.public_train_fit_occurrence_factor_learner_v1",
    "experiments.r3_pilot.public_train_fit_occurrence_factor_annotation_capsule_v2",
    "experiments.r3_pilot.prompt_occurrence_factor_relays_v6",
    "experiments.r3_pilot.typed_semantic_packed_occurrence_factor_authorized_plan_slice_v1",
)


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "opaque",
        (ProgramParameter("items", ProgramType.List(integer)),),
        integer,
    )


def _neutral_bundle(
    learner: ModuleType,
    events: Iterable[object],
) -> tuple[source.RawPhysicalSourceEvent | source.LiteralPhysicalSourceEvent, ...]:
    converted: list[
        source.RawPhysicalSourceEvent | source.LiteralPhysicalSourceEvent
    ] = []
    for event in events:
        if type(event) is learner.RawPhysicalSourceEvent:
            converted.append(
                source.RawPhysicalSourceEvent(
                    event.assembly_cells,
                    event.direction,
                    event.cue_slot,
                    event.selected_slot,
                )
            )
        elif type(event) is learner.LiteralPhysicalSourceEvent:
            converted.append(
                source.LiteralPhysicalSourceEvent(
                    event.typed_value_event_cells,
                    event.typed_value_type_cells,
                )
            )
        else:
            raise AssertionError("learner fixture exposed a non-physical source event")
    return tuple(converted)


class _Explodes:
    def __iter__(self):
        raise AssertionError("a stopped build or selection iterated sealed evidence")


def test_static_learned_relays_privately_authorize_retained_plan_slices() -> None:
    assert all(name not in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
    assert not hasattr(source, "FIT_SOURCE_BUNDLES")
    assert source.NORMAL_RUNTIME_TEACHER_TARGET_READ_COUNT == 0
    assert source.NORMAL_RUNTIME_LEARNER_IMPORT_COUNT == 0
    assert source.NORMAL_RUNTIME_ANNOTATION_IMPORT_COUNT == 0
    assert integrated.V2_TARGET_READ_COUNT == 0
    assert integrated.LEARNER_PROTOCOL_READ_COUNT == 0
    assert integrated.LEARNER_FIXTURE_READ_COUNT == 0
    assert integrated.CANONICAL_PANEL_SPEND_COUNT == 0
    assert integrated.PROTECTED_INPUT_COUNT == 0

    build_parameters = inspect.signature(
        integrated.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build
    ).parameters
    assert tuple(build_parameters) == (
        "signature",
        "cases",
        "literals",
        "max_cost",
        "source_certificate",
    )
    assert "desired" not in build_parameters
    assert integrated.WORK_CAPS == (30_000, 30_000, 80_000)

    learner = importlib.import_module(
        "experiments.r3_pilot.public_train_fit_occurrence_factor_learner_v1"
    )
    representatives: dict[
        int,
        tuple[source.RawPhysicalSourceEvent | source.LiteralPhysicalSourceEvent, ...],
    ] = {}
    for fixture in learner.FIT_SOURCE_BUNDLES:
        representatives.setdefault(len(fixture), _neutral_bundle(learner, fixture))
    assert set(representatives) == {8, 9}

    bundle = representatives[8]
    certificate = source.infer_occurrence_factors(reversed(bundle))
    assert certificate.status == source.READY
    frontier = integrated.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build(
        _signature(), CASES, LITERALS, 9, certificate
    )
    assert frontier.status == integrated.READY
    assert frontier.state.occurrence_count == 8
    assert frontier.state.structural_freeze_attempts == 1
    assert frontier.state.plan_build_attempts == 1
    assert frontier.state.desired_reads_before_select == 0
    assert frontier.state.proof_witnesses_before_select == 0
    assert frontier.state.source_certificate_sha256 == certificate.sha256
    assert frontier.state.source_checkpoint_sha256 == source.BANKED_CHECKPOINT_SHA256
    assert frontier.state.source_protocol_sha256 == source.BANKED_PROTOCOL_SHA256
    assert frontier.state.converter_receipt is not None
    assert frontier.plan_slice_receipt is not None
    assert frontier.work is not None
    assert frontier.work.relation.completed_units <= integrated.WORK_CAPS[0]
    assert frontier.work.physical_predictions <= integrated.WORK_CAPS[1]
    assert frontier.work.physical_races <= integrated.WORK_CAPS[2]
    assert not hasattr(frontier, "structural_certificate")

    delegated = frontier.select(DESIRED)
    assert delegated.plan_slice_selection_sha256 is not None
    assert delegated.build_state_sha256 == frontier.sha256
    assert delegated.work is not None

    forged = integrated.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build(
        _signature(),
        _Explodes(),  # type: ignore[arg-type]
        _Explodes(),  # type: ignore[arg-type]
        9,
        object(),  # type: ignore[arg-type]
    )
    assert forged.status == integrated.STOP_SOURCE_CERTIFICATE
    assert forged.state.structural_freeze_attempts == 0
    assert forged.state.plan_build_attempts == 0
    assert forged.select(_Explodes()).status == integrated.STOP_SOURCE_CERTIFICATE  # type: ignore[arg-type]

    repeated = bundle
    ambiguous_certificate = source.infer_occurrence_factors((*repeated, repeated[0]))
    assert ambiguous_certificate.status == source.STOP_AMBIGUOUS
    ambiguous = (
        integrated.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build(
            _signature(),
            _Explodes(),  # type: ignore[arg-type]
            _Explodes(),  # type: ignore[arg-type]
            9,
            ambiguous_certificate,
        )
    )
    assert ambiguous.status == integrated.STOP_RELAY_AMBIGUOUS
    assert ambiguous.state.plan_build_attempts == 0

    wrong_signature = TypedSignature(
        "opaque",
        (ProgramParameter("value", ProgramType.Int()),),
        ProgramType.Int(),
    )
    mismatched = (
        integrated.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build(
            wrong_signature,
            _Explodes(),  # type: ignore[arg-type]
            _Explodes(),  # type: ignore[arg-type]
            9,
            certificate,
        )
    )
    assert mismatched.status == integrated.STOP_SIGNATURE_MISMATCH
    assert mismatched.state.structural_freeze_attempts == 0
    assert mismatched.state.plan_build_attempts == 0
