from __future__ import annotations

import json
from hashlib import sha256

import pytest
from experiments.r3_pilot import (
    public_train_list_int_to_int_prompt_input_capsule_v1 as capsule_module,
)


def test_static_list_int_to_int_inputs_are_target_free_and_root_addressed(
    tmp_path,
) -> None:
    capsule = capsule_module.load_input_capsule()
    artifact_raw = capsule_module.ARTIFACT_PATH.read_bytes()
    artifact = json.loads(artifact_raw)

    assert sha256(artifact_raw).hexdigest() == capsule_module.ARTIFACT_SHA256
    assert capsule.artifact_sha256 == capsule_module.ARTIFACT_SHA256
    assert capsule.content_root == capsule_module.CONTENT_ROOT
    assert (capsule.appearance_count, capsule.input_count) == (20, 15)
    assert tuple(row.input_root for row in capsule.iter_inputs()) == (
        capsule_module.EXPECTED_INPUT_ROOTS
    )
    assert tuple((row.input_root, row.multiplicity) for row in capsule.iter_inputs()) == (
        capsule_module.EXPECTED_INPUT_ROOT_MULTIPLICITIES
    )
    appearances = tuple(capsule.iter_appearances())
    assert len(appearances) == 20
    assert tuple(row.input_root for row in appearances) == tuple(
        artifact["appearance_input_roots"]
    )

    for row in capsule.iter_inputs():
        assert capsule.lookup(row.input_root) is row
        spans, literals, signature_role = row.neural_records()
        assert spans and all(
            set(record)
            == {
                "start_token",
                "stop_token",
                "first_spike_step",
                "assembly_cells",
                "spike_cells",
            }
            for record in spans
        )
        assert all(
            set(record)
            == {"canonical_token_index", "typed_output_active_cells"}
            for record in literals
        )
        assert signature_role["role_binding"] == "reverse-declaration-depth"
        assert signature_role["entrypoint_spelling_present"] is False
        assert signature_role["parameter_spelling_present"] is False

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert keys(artifact).isdisjoint(
        {
            "ordinal",
            "task",
            "family",
            "split",
            "target",
            "desired",
            "program",
            "property",
            "expected",
            "teacher",
            "dev",
            "protected",
            "entry_point",
            "parameter_name",
        }
    )
    runtime_source = __import__("inspect").getsource(capsule_module)
    assert "public_train_prompt_selection_v4" not in runtime_source
    assert "public_train_factor_workspace" not in runtime_source
    assert "public_train_prompt_signature_capsule_v1.json" not in runtime_source

    changed = tmp_path / "changed-input-capsule.json"
    changed.write_bytes(artifact_raw[:-2] + b" \n")
    with pytest.raises(ValueError, match="artifact identity changed"):
        capsule_module.load_input_capsule(changed)
