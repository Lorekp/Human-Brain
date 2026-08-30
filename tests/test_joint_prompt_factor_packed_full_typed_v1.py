from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace

import pytest
from experiments.r3_pilot import joint_prompt_factor_packed_full_typed_v1 as joint
from experiments.r3_pilot import teacher_gated_factor_workspace_v5 as factor_v5
from experiments.r3_pilot import teacher_gated_factor_workspace_v5_protocol as factor_protocol


@pytest.fixture(scope="module")
def bridge() -> factor_v5.FrozenTeacherGatedFactorBridge:
    return factor_v5.build_frozen_teacher_gated_factor_bridge(factor_protocol.PROTOCOL)


@pytest.fixture(scope="module")
def count_treatment(
    bridge: factor_v5.FrozenTeacherGatedFactorBridge,
) -> tuple[joint.FrozenJointPromptFactorFrontier, joint.JointSelection]:
    frontier = joint.build_frozen_joint_frontier(
        bridge,
        joint._self_posed_batch(5),
        signature=joint._self_posed_signature(),
        case_arguments=joint._self_posed_cases(),
        literals=joint.SELF_POSED_LITERALS,
        max_ast_cost=6,
    )
    assert frontier.status == joint.FRONTIER_READY
    assert frontier.state.packed_frontier_sha256 is not None
    assert frontier.state.build_concrete_proof_witnesses == 0
    selection = frontier.select(joint._COUNT_ITEM_LT_DESIRED)
    return frontier, selection


def test_two_operation_seam_and_banked_dependency_receipt(
    count_treatment: tuple[joint.FrozenJointPromptFactorFrontier, joint.JointSelection],
) -> None:
    public_functions = tuple(
        sorted(
            name
            for name, value in vars(joint).items()
            if inspect.isfunction(value)
            and value.__module__ == joint.__name__
            and not name.startswith("_")
        )
    )
    public_methods = tuple(
        sorted(
            name
            for name, value in vars(joint.FrozenJointPromptFactorFrontier).items()
            if inspect.isfunction(value) and not name.startswith("_")
        )
    )
    build_parameters = inspect.signature(joint.build_frozen_joint_frontier).parameters

    assert public_functions == ("build_frozen_joint_frontier",)
    assert public_methods == ("select",)
    assert "desired" not in " ".join(build_parameters)
    assert joint.FULL_TYPED_SOURCE_COMMIT == "b0e95e8"
    assert joint.V5_SOURCE_COMMIT == "5cb01cc"
    assert (
        joint.FULL_TYPED_SOURCE_SHA256
        == "e488cc1a461b148b96c854a66ff7ffacc39384bcbe59f61787021b72eb85704f"
    )
    assert (
        joint.V5_SOURCE_SHA256 == "c76292164b43586b9400f8277f1c2aa71b2a136f113f36bc3f89712f0ca6895d"
    )

    frontier, _selection = count_treatment
    assert frontier.state.full_typed_source_commit == joint.FULL_TYPED_SOURCE_COMMIT
    assert frontier.state.full_typed_source_sha256 == joint.FULL_TYPED_SOURCE_SHA256
    assert frontier.state.v5_source_commit == joint.V5_SOURCE_COMMIT
    assert (
        frontier.state.public_input_count,
        frontier.state.canonical_input_count,
        frontier.state.protected_input_count,
    ) == (0, 0, 0)


class _ExplodingIterable:
    def __iter__(self):
        raise AssertionError("causal STOP touched a downstream input")


def test_factor_controls_stop_before_engine_build_and_desired_access(
    bridge: factor_v5.FrozenTeacherGatedFactorBridge,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_build(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("causal prompt STOP built the packed engine")

    monkeypatch.setattr(joint.full_typed.PackedFullTypedFrontier, "build", forbidden_build)
    fixtures = (
        (joint._self_posed_batch(5), joint._FACTOR_OFF, (8, 5), ((), ())),
        (joint._self_posed_batch(5), joint._DERANGED, (8, 5), ((9,), (6,))),
        (joint._self_posed_batch(3), joint._TREATMENT, (5, 8), ((5,), (8,))),
    )
    for batch, intervention, raw_cells, projected_cells in fixtures:
        frontier = joint._build_intervened_joint_frontier(
            bridge,
            batch,
            signature=joint._self_posed_signature(),
            case_arguments=_ExplodingIterable(),
            literals=_ExplodingIterable(),
            max_ast_cost=6,
            intervention=intervention,
        )
        assert frontier.status == joint.STOP_FACTOR_ADMISSION
        assert frontier.state.packed_frontier_sha256 is None
        assert frontier.root_admission is None
        assert frontier.prompt_admission.raw_relay_cells == raw_cells
        assert frontier.prompt_admission.projected_relay_cells == projected_cells
        selection = frontier.select(_ExplodingIterable())
        assert selection.status == joint.STOP_FACTOR_ADMISSION
        assert selection.joint_race is None
        assert selection.packed_selection_sha256 is None
        assert selection.witnesses == ()


def test_authoritative_joint_winner_licenses_selected_only_public_proof(
    count_treatment: tuple[joint.FrozenJointPromptFactorFrontier, joint.JointSelection],
) -> None:
    frontier, selection = count_treatment
    assert selection.status == joint.SELECTED
    assert selection.joint_race is not None
    assert selection.joint_race.status == joint.SELECTED
    assert selection.winner_identity_equal is True
    assert (
        selection.joint_winner_key
        == selection.joint_race.winner_key
        == selection.packed_winner_key
        == selection.packed_selected_eclass_id
        == selection.selected_eclass_id
    )
    assert selection.packed_race_sha256 is not None
    assert len(selection.witnesses) == 1
    assert selection.witnesses[0].source == joint._COUNT_ITEM_LT_SOURCE
    assert selection.packed_concrete_proof_witnesses == 1
    assert frontier.prompt_admission.raw_relay_cells == (8, 5)
    assert frontier.prompt_admission.selected_lineage == (8, 5)

    relay_source = textwrap.dedent(inspect.getsource(joint._relay_trace))
    attributes = {
        node.attr for node in ast.walk(ast.parse(relay_source)) if isinstance(node, ast.Attribute)
    }
    assert "factor" not in attributes


def test_public_surface_seals_live_engine_and_rerooted_build_receipt(
    count_treatment: tuple[joint.FrozenJointPromptFactorFrontier, joint.JointSelection],
) -> None:
    frontier, _selection = count_treatment
    assert not hasattr(frontier, "packed_frontier")
    for value in (
        frontier.state,
        frontier.status,
        frontier.sha256,
        frontier.prompt_admission,
        frontier.root_admission,
    ):
        assert not callable(getattr(value, "select", None))

    prompt = replace(
        frontier.prompt_admission, selected_lineage=(joint._SUM_RELAY_CELL,), sha256=""
    )
    prompt = replace(
        prompt,
        sha256=joint._digest("prompt-admission", prompt.record(include_sha256=False)),
    )
    receipt = frontier._construction_receipt._replace(prompt_admission=prompt, sha256="")
    receipt = receipt._replace(
        sha256=joint._digest("joint-build-certificate", joint._receipt_payload(receipt))
    )
    with pytest.raises(TypeError, match="private executed-build certificate"):
        joint.FrozenJointPromptFactorFrontier(receipt)


def test_exact_seven_row_self_posed_matrix(
    bridge: factor_v5.FrozenTeacherGatedFactorBridge,
) -> None:
    result = joint._run_self_posed_matrix(bridge)
    assert result.verdict == joint.PASS_MATRIX
    assert (result.treatment_pass_count, result.control_stop_count) == (3, 4)
    assert (
        result.public_input_count,
        result.canonical_input_count,
        result.protected_input_count,
    ) == (0, 0, 0)
    assert tuple(row.row_id for row in result.rows) == (
        "treatment-sum",
        "treatment-count-item-lt",
        "treatment-count-literal-lt",
        "control-factor-off",
        "control-deranged-relays",
        "control-event-order",
        "control-visible-io-match-off",
    )
    assert tuple(row.selection_status for row in result.rows) == (
        joint.SELECTED,
        joint.SELECTED,
        joint.SELECTED,
        joint.STOP_FACTOR_ADMISSION,
        joint.STOP_FACTOR_ADMISSION,
        joint.STOP_FACTOR_ADMISSION,
        joint.STOP_NO_EXACT,
    )
    assert tuple(row.source for row in result.rows) == (
        joint._SUM_SOURCE,
        joint._COUNT_ITEM_LT_SOURCE,
        joint._COUNT_LITERAL_LT_SOURCE,
        None,
        None,
        None,
        None,
    )
    assert result.rows[1].prompt_admission_sha256 == result.rows[2].prompt_admission_sha256
    assert result.rows[6].build_status == joint.FRONTIER_READY
