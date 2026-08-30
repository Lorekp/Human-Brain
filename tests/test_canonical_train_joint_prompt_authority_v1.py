from __future__ import annotations

import sys
from dataclasses import replace

import pytest
from experiments.r3_pilot import canonical_train_joint_prompt_authority_v1 as joint
from experiments.r3_pilot import (
    public_train_factor_workspace_v4_static_checkpoint_capsule_v1 as factor_checkpoint,
)
from experiments.r3_pilot import (
    public_train_prompt_selection_v4_static_checkpoint_capsule_v1 as prompt_checkpoint,
)

FORBIDDEN_RUNTIME_MODULES = (
    "experiments.r3_pilot.public_train_prompt_selection_v4",
    "experiments.r3_pilot.public_train_prompt_selection_v4_protocol",
    "experiments.r3_pilot.public_train_structural_annotation_capsule_v1",
    "experiments.r3_pilot.public_train_factor_workspace_v4",
    "experiments.r3_pilot.public_train_factor_workspace_v4_protocol",
)


def test_prompt_authority_binds_same_input_and_stops_before_unbound_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
    spans, literals, signature_role, expected_actual_winner = (
        prompt_checkpoint._load_reproduction_fixture()
    )

    def forbid_factor_inference(*args: object) -> None:
        raise AssertionError("unbound factor inference must not run")

    actual_certificates: list[prompt_checkpoint.PromptStructuralCertificate] = []
    infer_prompt = prompt_checkpoint.infer_prompt

    def capture_prompt(*args: object) -> prompt_checkpoint.PromptStructuralCertificate:
        certificate = infer_prompt(*args)  # type: ignore[arg-type]
        actual_certificates.append(certificate)
        return certificate

    monkeypatch.setattr(factor_checkpoint, "infer_factors", forbid_factor_inference)
    monkeypatch.setattr(prompt_checkpoint, "infer_prompt", capture_prompt)
    handoff = joint.infer_prompt_authority(spans, literals, signature_role)
    certificate = actual_certificates[0]

    assert handoff.state.prompt_status == joint.PROMPT_AUTHORITY_READY
    assert handoff.status == joint.STOP_JOINT_FACTOR_SELECTOR_BINDING
    assert handoff.state.stop_reason == joint.MISSING_SELECTOR_RELAY_REASON
    assert handoff.state.prompt_actual_winner_sha256 == expected_actual_winner
    assert handoff.state.prompt_checkpoint_sha256 == (
        prompt_checkpoint.BANKED_CHECKPOINT_SHA256
    )
    assert handoff.state.factor_checkpoint_sha256 == (
        factor_checkpoint.BANKED_CHECKPOINT_SHA256
    )
    assert handoff.state.factor_artifact_sha256 == factor_checkpoint.ARTIFACT_SHA256
    assert handoff.state.factor_selector_relay_status == joint.MISSING_SELECTOR_RELAY
    assert handoff.state.factor_carrier_assembly_status == (
        joint.UNRESOLVED_FACTOR_CARRIER
    )
    assert handoff.state.factor_inference_attempts == 0
    assert certificate.status == prompt_checkpoint.STOP
    assert len(handoff.state.span_record_sha256s) == len(spans)
    assert len(handoff.state.literal_record_sha256s) == len(literals)
    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)

    with pytest.raises(RuntimeError, match="direction/cue-slot/selected-slot relay"):
        joint.require_joint_authority(handoff)
    with pytest.raises(TypeError, match="private inference receipt"):
        joint.CanonicalTrainJointPromptAuthorityHandoff(  # type: ignore[arg-type]
            object()
        )

    capped = replace(
        certificate,
        status=prompt_checkpoint.CAP,
        terminal_winner=None,
    )
    monkeypatch.setattr(prompt_checkpoint, "infer_prompt", lambda *args: capped)
    prompt_stopped = joint.infer_prompt_authority(spans, literals, signature_role)

    assert prompt_stopped.status == joint.STOP_PROMPT_AUTHORITY
    assert prompt_stopped.state.prompt_status == joint.STOP_PROMPT_AUTHORITY
    assert prompt_stopped.state.factor_inference_attempts == 0
    assert prompt_stopped.state.stop_reason == (
        "physical prompt inference terminated as CAP"
    )
    with pytest.raises(RuntimeError, match="terminated as CAP"):
        joint.require_joint_authority(prompt_stopped)
