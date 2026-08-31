from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import sys
from pathlib import Path

import pytest
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

RUNTIME_MODULE = (
    "experiments.r3_pilot."
    "typed_semantic_packed_public_train_fit_prompt_record_authorized_plan_frontier_v1"
)
INPUT_MODULE = (
    "experiments.r3_pilot.public_train_list_int_to_int_prompt_input_capsule_v1"
)
ANNOTATION_V1_MODULE = (
    "experiments.r3_pilot.public_train_fit_occurrence_factor_annotation_capsule_v1"
)
FORBIDDEN_RUNTIME_MODULES = (
    INPUT_MODULE,
    ANNOTATION_V1_MODULE,
    "experiments.r3_pilot.public_train_fit_occurrence_factor_annotation_capsule_v2",
    "experiments.r3_pilot.public_train_fit_occurrence_factor_learner_v1",
    "experiments.r3_pilot.export_public_train_fit_prompt_record_source_event_producer_v1",
    "experiments.r3_pilot.prompt_occurrence_factor_relays_v6",
    "experiments.r3_pilot."
    "typed_semantic_packed_occurrence_factor_authorized_plan_slice_v1",
)

CASES = (
    ([],),
    ([0, 1, 2, 3],),
    ([1, 2, 10],),
    ([0, 0, 5],),
    ([2, 2, 2],),
)
LITERALS = (-3, -2, 0)
DESIRED = (0, 2, 2, 1, 3)


def _signature() -> TypedSignature:
    integer = ProgramType.Int()
    return TypedSignature(
        "opaque",
        (ProgramParameter("items", ProgramType.List(integer)),),
        integer,
    )


class _Explodes:
    def __iter__(self):
        raise AssertionError("a stopped seam iterated sealed desired/frontier evidence")


def test_actual_prompt_records_reach_private_desired_free_packed_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (RUNTIME_MODULE, *FORBIDDEN_RUNTIME_MODULES):
        sys.modules.pop(name, None)
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if any(
            name == forbidden or name.startswith(f"{forbidden}.")
            for forbidden in FORBIDDEN_RUNTIME_MODULES
        ):
            raise AssertionError(f"forbidden runtime import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as import_guard:
        import_guard.setattr(builtins, "__import__", guarded_import)
        runtime = importlib.import_module(RUNTIME_MODULE)

    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
    assert runtime.DESIRED_BUILD_INPUT_COUNT == 0
    assert runtime.CANONICAL_PANEL_SPEND_COUNT == 0
    assert runtime.PROTECTED_INPUT_COUNT == 0

    runtime_path = Path(runtime.__file__)
    runtime_source = runtime_path.read_text(encoding="utf-8")
    runtime_tree = ast.parse(runtime_source)
    runtime_imports = {
        alias.name
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not set(FORBIDDEN_RUNTIME_MODULES) & runtime_imports
    assert "FIT_SOURCE_BUNDLES" not in runtime_source
    assert "input_root" not in runtime_source

    build_parameters = inspect.signature(
        runtime.PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1.build
    ).parameters
    assert tuple(build_parameters) == (
        "span_records",
        "literal_records",
        "signature_role_record",
        "signature",
        "cases",
        "literals",
        "max_cost",
    )
    assert "desired" not in build_parameters
    assert tuple(
        inspect.signature(
            runtime.PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1.select
        ).parameters
    ) == ("self", "desired")

    inputs = importlib.import_module(INPUT_MODULE)
    annotation_v1 = importlib.import_module(ANNOTATION_V1_MODULE)
    input_state = inputs.load_input_capsule()
    audit = annotation_v1.CAPSULE.fit_identity_audits[6]
    records = input_state.lookup(audit.input_root_sha256).neural_records()

    frontier = (
        runtime.PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1.build(
            records[0],
            records[1],
            records[2],
            _signature(),
            CASES,
            LITERALS,
            9,
        )
    )
    assert frontier.status == runtime.READY
    assert frontier.state.source_event_status == runtime.prompt_producer.READY
    assert frontier.state.source_event_count == 8
    assert frontier.state.occurrence_status == runtime.occurrence_static.READY
    assert frontier.state.occurrence_count == 8
    assert frontier.state.producer_build_attempts == 1
    assert frontier.state.occurrence_inference_attempts == 1
    assert frontier.state.plan_build_attempts == 1
    assert frontier.state.desired_reads_before_select == 0
    assert frontier.state.proof_witnesses_before_select == 0
    assert frontier.plan_slice_receipt is not None
    assert frontier.work is not None
    assert not hasattr(frontier, "structural_certificate")
    assert not hasattr(frontier, "desired")

    selection = frontier.select(DESIRED)
    assert selection.build_state_sha256 == frontier.sha256
    assert selection.delegated_selection_sha256 is not None
    assert selection.work is not None

    with monkeypatch.context() as forged_source_patch:
        forged_source_patch.setattr(
            runtime.prompt_producer,
            "produce_source_event_bundle",
            lambda *_args, **_kwargs: object(),
        )
        forged_source = (
            runtime.PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1.build(
                records[0],
                records[1],
                records[2],
                _signature(),
                _Explodes(),  # type: ignore[arg-type]
                _Explodes(),  # type: ignore[arg-type]
                9,
            )
        )
    assert forged_source.status == runtime.STOP_SOURCE_EVENT_CERTIFICATE
    assert forged_source.state.occurrence_inference_attempts == 0
    assert forged_source.state.plan_build_attempts == 0
    assert (
        forged_source.select(_Explodes()).status  # type: ignore[arg-type]
        == runtime.STOP_SOURCE_EVENT_CERTIFICATE
    )

    real_occurrence_inference = runtime.occurrence_static.infer_occurrence_factors
    occurrence_calls = 0

    def forge_second_occurrence_certificate(events: object) -> object:
        nonlocal occurrence_calls
        occurrence_calls += 1
        if occurrence_calls == 1:
            return real_occurrence_inference(events)
        return object()

    with monkeypatch.context() as forged_occurrence_patch:
        forged_occurrence_patch.setattr(
            runtime.occurrence_static,
            "infer_occurrence_factors",
            forge_second_occurrence_certificate,
        )
        forged_occurrence = (
            runtime.PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1.build(
                records[0],
                records[1],
                records[2],
                _signature(),
                _Explodes(),  # type: ignore[arg-type]
                _Explodes(),  # type: ignore[arg-type]
                9,
            )
        )
    assert forged_occurrence.status == runtime.STOP_OCCURRENCE_CERTIFICATE
    assert forged_occurrence.state.occurrence_inference_attempts == 1
    assert forged_occurrence.state.plan_build_attempts == 0
    assert (
        forged_occurrence.select(_Explodes()).status  # type: ignore[arg-type]
        == runtime.STOP_OCCURRENCE_CERTIFICATE
    )

    forged_frontier = object.__new__(
        runtime.PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1
    )
    with pytest.raises(TypeError, match="issued exact object"):
        forged_frontier.select(_Explodes())  # type: ignore[arg-type]
