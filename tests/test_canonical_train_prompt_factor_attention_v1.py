from __future__ import annotations

import ast
import inspect
import sys
from dataclasses import replace
from pathlib import Path

from experiments.r3_pilot import canonical_train_prompt_factor_attention_v1 as attention
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


def test_exhaustive_same_prompt_attention_suppresses_distractors_causally() -> None:
    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
    spans, literals, signature_role, _ = prompt_checkpoint._load_reproduction_fixture()

    handoff = attention.infer_prompt_factor_attention(spans, literals, signature_role)
    state = attention.require_prompt_factor_attention(handoff)

    assert state.status == attention.ATTENTION_READY
    assert state.factor_membership == ("minimum", "maximum")
    assert state.membership_count == 2
    assert tuple(
        factor
        for factor, supported in zip(
            factor_checkpoint.FACTOR_REGISTRY,
            state.prompt_physical_support,
            strict=True,
        )
        if supported
    ) == ("minimum", "maximum")
    assert state.scan.status == attention.SCAN_READY
    assert state.scan.raw_span_count == 62
    assert state.scan.unique_assembly_count == 60
    assert state.scan.selector_geometry_count == 8
    assert state.scan.factor_inference_attempts == 480
    assert state.scan.ready_observation_count == 20
    assert state.scan.zero_observation_count == 460
    assert state.scan.cap_observation_count == 0
    assert state.scan.collision_observation_count == 0
    assert state.scan.duplicate_equivalent_collapses == 15
    assert sum(state.scan.physical_membership) == 9
    assert len(state.physical_coincidence_spikes) == factor_checkpoint.FACTOR_COUNT
    assert sum(state.physical_coincidence_spikes) == 2
    assert state.generic_ascii_class_support is False
    assert state.host_class_label_selection_count == 0
    connector = attention._prompt_factor_connector()
    assert attention._authenticate_prompt_factor_connector(connector)
    assert state.prompt_factor_connector_sha256 == connector.sha256
    assert state.prompt_output_registry_sha256 == connector.output_registry_sha256
    assert state.prompt_factor_connector_cell_count == len(connector.relay_targets) == 134
    assert state.runtime_winner_relay_parse_count == 0
    assert (
        state.caller_factor_observation_count,
        state.target_input_count,
        state.expected_input_count,
        state.desired_input_count,
        state.program_input_count,
        state.task_input_count,
        state.family_input_count,
        state.heldout_input_count,
        state.protected_input_count,
        state.executable_learner_import_count,
        state.target_protocol_import_count,
    ) == (0,) * 11

    deranged = attention._run_attention(
        spans,
        literals,
        signature_role,
        attention._DERANGED,
    ).state
    assert deranged.status == attention.STOP_DERANGEMENT
    assert deranged.factor_membership == ()
    assert sum(deranged.physical_coincidence_spikes) == 0
    assert deranged.scan.factor_inference_attempts == 480

    prompt = prompt_checkpoint.infer_prompt(spans, literals, signature_role)
    baseline = attention._prompt_support(prompt, connector)
    winners = list(prompt.occurrences[0].branch_winners)
    winners[0] = replace(
        winners[0],
        relay_id="root-production-family|root-production|production|maximum",
    )
    occurrences = list(prompt.occurrences)
    occurrences[0] = replace(occurrences[0], branch_winners=tuple(winners))
    altered = attention._prompt_support(
        replace(prompt, occurrences=tuple(occurrences)),
        connector,
    )
    minimum = factor_checkpoint.FACTOR_REGISTRY.index("minimum")
    maximum = factor_checkpoint.FACTOR_REGISTRY.index("maximum")
    assert altered[0][minimum] == baseline[0][minimum] - 1
    assert altered[0][maximum] == baseline[0][maximum]
    assert altered[-1] == 1
    assert "winner.relay_id.split" not in inspect.getsource(attention._prompt_support)
    assert "winner.selected_cell_id" in inspect.getsource(attention._prompt_support)

    assert list(inspect.signature(attention.infer_prompt_factor_attention).parameters) == [
        "span_records",
        "literal_records",
        "signature_role_record",
    ]
    tree = ast.parse(Path(attention.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not set(FORBIDDEN_RUNTIME_MODULES).intersection(imports)
    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
