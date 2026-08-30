from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from experiments.r3_pilot import (
    public_train_factor_workspace_v4_static_checkpoint_capsule_v1 as capsule,
)


def _observation(
    direction: str,
    cue_slot: int,
    selected_slot: int,
    cells: tuple[int, ...],
) -> capsule.RawSpanObservation:
    return capsule.RawSpanObservation(
        direction=direction,
        cue_slot=cue_slot,
        selected_slot=selected_slot,
        assembly_cells=cells,
    )


_MINIMUM_MAXIMUM = _observation(
    "NEXT",
    0,
    1,
    (66, 73, 138, 162, 163, 215, 231, 261, 294, 321, 348, 436),
)
_FIVE_SINGLETONS = (
    _observation(
        "PREV",
        0,
        3,
        (66, 73, 138, 162, 163, 215, 231, 261, 294, 321, 348, 436),
    ),
    _observation(
        "NEXT",
        0,
        1,
        (90, 114, 128, 202, 212, 269, 308, 354, 377, 400, 450, 504),
    ),
    _observation(
        "PREV",
        0,
        3,
        (90, 114, 128, 202, 212, 269, 308, 354, 377, 400, 450, 504),
    ),
    _observation(
        "NEXT",
        0,
        1,
        (104, 115, 159, 163, 168, 207, 231, 250, 260, 366, 370, 485),
    ),
    _observation(
        "PREV",
        0,
        3,
        (104, 115, 159, 163, 168, 207, 231, 250, 260, 366, 370, 485),
    ),
)


def test_static_artifact_authenticates_the_banked_checkpoint() -> None:
    artifact = capsule.ARTIFACT_PATH.read_bytes()
    checkpoint = capsule.STATIC_CHECKPOINT

    assert sha256(artifact).hexdigest() == capsule.ARTIFACT_SHA256
    assert checkpoint.checkpoint_sha256 == capsule.BANKED_CHECKPOINT_SHA256
    assert checkpoint.protocol_sha256 == capsule.BANKED_PROTOCOL_SHA256
    assert checkpoint.result_sha256 == capsule.BANKED_RESULT_SHA256
    assert capsule.BANKED_COMMIT == "2bee9d0120cf4d957e9aa426410490a333180139"
    for state in (checkpoint.excitatory_ticks, checkpoint.inhibitory_ticks):
        assert state.shape == (24, 5_632)
        assert state.dtype == np.uint8
        assert not state.flags.writeable
        with pytest.raises(ValueError):
            state.setflags(write=True)


def test_target_free_raw_observation_matches_banked_actual_winners_post_hoc() -> None:
    certificate = capsule.infer_factors((_MINIMUM_MAXIMUM,))

    assert certificate.status == "READY_FACTOR_SET"
    assert certificate.factor_membership == ("minimum", "maximum")
    assert certificate.span_decisions[0].factor_membership == (
        "minimum",
        "maximum",
    )
    assert certificate.membership_count == 2


def test_duplicate_and_fifth_factor_controls_stop() -> None:
    zero = capsule.infer_factors(())
    duplicate = capsule.infer_factors((_MINIMUM_MAXIMUM, _MINIMUM_MAXIMUM))
    fifth = capsule.infer_factors(_FIVE_SINGLETONS)

    assert zero.status == "STOP_ZERO"
    assert zero.factor_membership == ()
    assert duplicate.status == "STOP_DUPLICATE_FACTOR"
    assert duplicate.factor_membership == ("minimum", "maximum")
    assert tuple(count for count in duplicate.event_counts if count) == (2, 2)
    assert fifth.status == "STOP_EVENT_CAPACITY_4"
    assert fifth.factor_membership == (
        "add",
        "absolute",
        "less-than",
        "map",
        "sum",
    )
    assert fifth.membership_count == 5


def test_runtime_ast_has_no_executable_or_answer_bearing_seam() -> None:
    source_path = Path(capsule.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } | {
        node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)
    }

    assert "experiments.r3_pilot.public_train_factor_workspace_v4" not in imports
    assert "experiments.r3_pilot.public_train_factor_workspace_v4_protocol" not in imports
    assert "importlib" not in imports
    assert not {"candidate_factors", "candidate_exact", "target_factors", "lesson"} & identifiers
    assert "candidate_factors" not in source
    assert "candidate_exact" not in source
    assert "target_factors" not in source
