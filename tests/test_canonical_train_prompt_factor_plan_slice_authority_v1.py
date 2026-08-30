from __future__ import annotations

import ast
import inspect
import json
import sys
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import canonical_train_prompt_factor_attention_v1 as attention
from experiments.r3_pilot import (
    canonical_train_prompt_factor_plan_slice_authority_v1 as authority,
)
from experiments.r3_pilot.public_train_prompt_semantic_bridge import (
    observe_public_train_prompt,
)
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

RAW_CAPSULE = Path(
    "experiments/r3_pilot/public_train_prompt_signature_capsule_v1.json"
)
RAW_CAPSULE_SHA256 = (
    "0f34d5c1e4b9e90f76f098bf3296af546495d8b6053586b1620ecd88ba5b3289"
)
TARGET_INPUT_SHA256 = (
    "fc3c43134eeaea4c920f1f31f15e04da1081b3ea5fb29c492d3058961c518224"
)

FORBIDDEN_RUNTIME_MODULES = (
    "experiments.r3_pilot.public_train_prompt_selection_v4",
    "experiments.r3_pilot.public_train_prompt_selection_v4_protocol",
    "experiments.r3_pilot.public_train_structural_annotation_capsule_v1",
    "experiments.r3_pilot.public_train_factor_workspace_v4",
    "experiments.r3_pilot.public_train_factor_workspace_v4_protocol",
    "experiments.r3_pilot.public_typed_semantic_canonical_train_freeze_v3",
)


def _target_free_list_int_fixture() -> tuple[
    TypedSignature,
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    raw = RAW_CAPSULE.read_bytes()
    assert sha256(raw).hexdigest() == RAW_CAPSULE_SHA256
    capsule = json.loads(raw)
    assert set(capsule) == {"schema", "scope", "records"}
    assert capsule["schema"] == "public-train-prompt-signature-capsule-v1"
    assert capsule["scope"] == "PUBLIC_TRAIN_PROMPTS_AND_SIGNATURES_ONLY"

    integer = ProgramType.Int()
    list_integer = ProgramType.List(integer)
    matches = []
    for row in capsule["records"]:
        assert set(row) == {"prompt", "signature"}
        signature = TypedSignature.from_record(row["signature"])
        if (
            len(signature.parameters) == 1
            and signature.parameters[0].type == list_integer
            and signature.result == integer
        ):
            observed = observe_public_train_prompt(row["prompt"], signature)
            if observed.sha256 == TARGET_INPUT_SHA256:
                matches.append((signature, observed))
    assert len(matches) == 1
    signature, observed = matches[0]
    spans = tuple(
        {
            "start_token": item.start_token,
            "stop_token": item.stop_token,
            "first_spike_step": item.first_spike_step,
            "assembly_cells": item.assembly_cells,
            "spike_cells": item.spike_cells,
        }
        for item in observed.spans
    )
    literals = tuple(
        {
            "canonical_token_index": item.canonical_token_index,
            "typed_output_active_cells": item.typed_output_active_cells,
        }
        for item in observed.signed_literals
    )
    signature_role = {
        "parameters": tuple(
            {
                "role": f"ARG{len(signature.parameters) - index - 1}",
                "type": parameter.type.to_record(),
            }
            for index, parameter in enumerate(signature.parameters)
        ),
        "result": signature.result.to_record(),
        "role_binding": "reverse-declaration-depth",
        "entrypoint_spelling_present": False,
        "parameter_spelling_present": False,
    }
    assert len(spans) == 70
    assert len(literals) == 2
    return signature, spans, literals, signature_role


class _ExplodingDesired:
    def __iter__(self) -> object:
        raise AssertionError("STOP authority read desired values")

    def __len__(self) -> int:
        raise AssertionError("STOP authority sized desired values")

    def __getitem__(self, _index: object) -> object:
        raise AssertionError("STOP authority indexed desired values")


def test_private_attention_to_plan_slice_authority_stops_at_exact_missing_join() -> None:
    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
    signature, spans, prompt_literals, signature_role = (
        _target_free_list_int_fixture()
    )
    cases = (([],), ([-6, -2, 0, 8],), ([1, 2, 3],), ([-10, 10],))
    build_literals = (-2, 1, 2)

    handoff = authority.issue_canonical_train_plan_slice_authority(
        spans,
        prompt_literals,
        signature_role,
        signature,
        cases,
        build_literals,
        9,
    )
    state = handoff.state
    assert state.status == authority.STOP_OCCURRENCE_LOCAL_CARRIER_JOIN
    assert state.attention_status == attention.ATTENTION_READY
    assert state.signature_role_agreement
    assert state.prompt_factor_agreement
    assert state.factor_membership
    assert state.attention_inference_attempt_count == 1
    assert state.prompt_recovery_attempt_count == 1
    assert state.factor_inference_attempt_count > 0
    assert state.occurrence_join.ordered_edge_binding_count > 0
    assert state.occurrence_join.reused_edge_endpoint_count == 0
    assert state.occurrence_join.non_edge_role_slot_copy_event_count > 0
    assert state.occurrence_join.bound_non_edge_role_slot_copy_event_count == 0
    assert state.occurrence_join.typed_literal_event_count == 2
    assert state.occurrence_join.occurrence_bound_typed_literal_event_count == 0
    assert state.occurrence_join.ready_factor_observation_count > 0
    assert state.occurrence_join.retained_factor_observation_identity_count == 0
    assert state.occurrence_join.packed_factor_slot_schema_count == 0
    assert state.occurrence_join.missing_relays == authority.MISSING_OCCURRENCE_RELAYS
    assert state.structural_conversion_candidate_count == 0
    assert state.structural_certificate_freeze_attempt_count == 0
    assert state.plan_slice_build_attempt_count == 0
    assert state.plan_slice_receipt_sha256 is None
    assert state.authenticated_eclass_id is None
    assert not state.eclass_receipt_identity_verified
    assert (
        state.desired_input_count_during_build,
        state.desired_value_read_count_during_build,
        state.caller_factor_observation_count,
        state.caller_structural_certificate_count,
        state.target_input_count,
        state.expected_input_count,
        state.program_input_count,
        state.task_input_count,
        state.family_input_count,
        state.heldout_input_count,
        state.protected_input_count,
    ) == (0,) * 11

    stopped_selection = handoff.select(_ExplodingDesired())
    assert stopped_selection.status == state.status
    assert stopped_selection.packed_selection is None
    assert stopped_selection.desired_value_read_count == 0

    wrong_signature = TypedSignature(
        signature.entry_point,
        (ProgramParameter(signature.parameters[0].name, ProgramType.Int()),),
        ProgramType.Int(),
    )
    mismatch = authority.issue_canonical_train_plan_slice_authority(
        spans,
        prompt_literals,
        signature_role,
        wrong_signature,
        cases,
        build_literals,
        9,
    ).state
    assert mismatch.status == authority.STOP_SIGNATURE_ROLE
    assert not mismatch.signature_role_agreement
    assert mismatch.raw_prompt_input_root_sha256 == state.raw_prompt_input_root_sha256
    assert mismatch.packed_build_input_root_sha256 != state.packed_build_input_root_sha256
    assert mismatch.joint_input_root_sha256 != state.joint_input_root_sha256
    assert mismatch.plan_slice_build_attempt_count == 0

    parameters = inspect.signature(
        authority.issue_canonical_train_plan_slice_authority
    ).parameters
    assert tuple(parameters) == (
        "span_records",
        "prompt_literal_records",
        "signature_role_record",
        "signature",
        "cases",
        "literals",
        "max_cost",
    )
    assert not {
        "desired",
        "target",
        "expected",
        "program",
        "task",
        "family",
        "factor_observations",
        "structural_certificate",
    }.intersection(parameters)
    tree = ast.parse(Path(authority.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "freeze_structural_factor_certificate" not in calls
    assert not set(FORBIDDEN_RUNTIME_MODULES).intersection(imports)
    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
