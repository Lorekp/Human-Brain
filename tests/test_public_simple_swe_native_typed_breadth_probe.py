from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO_ROOT / "experiments" / "r3_pilot"
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, PROBE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_native_typed_breadth_probe as probe
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalRuntimeError,
    NativeTemporalRuntimeStatus,
    TemporalStepMode,
)


def _selected() -> tuple[object, tuple[object, ...], tuple[object, ...]]:
    return probe._select_tasks()


def _cohort_record(
    *, exact: bool, eos_full_pass: bool = False
) -> dict[str, object]:
    return {
        "classification": {
            "structural_exact": exact,
            "pre_completion_structural_failure": not exact,
            "ordinary_eos_full_pass": eos_full_pass,
        }
    }


def test_ordered_visible_train_cohort_is_native_typed_fresh_not_globally_untouched() -> None:
    _protocol, donors, targets = _selected()
    assert tuple((task.public.family, task.public.ordinal) for task in donors) == probe.TRAIN_KEYS
    assert tuple((task.public.family, task.public.ordinal) for task in targets) == probe.TARGET_KEYS
    assert all(task.public.partition == "train" for task in (*donors, *targets))
    assert not set(probe.TARGET_KEYS) & probe.KNOWN_NATIVE_TYPED_SPENT_TARGET_KEYS
    assert "historical prompt-only baseline" in (probe.__doc__ or "")
    assert [task.public.task_sha256 for task in donors] == [
        probe.SOURCE_RECORDS[key]["task_sha256"] for key in probe.TRAIN_KEYS
    ]
    assert [task.public.task_sha256 for task in targets] == [
        probe.SOURCE_RECORDS[key]["task_sha256"] for key in probe.TARGET_KEYS
    ]


def test_frozen_programs_accounting_and_all_target_prompt_codebook_union() -> None:
    _protocol, donors, targets = _selected()
    assert [len(probe.EXPECTED_TARGET_TOKENS[key]) for key in probe.TARGET_KEYS] == [
        9,
        6,
        27,
        27,
    ]
    for key, task in zip(probe.TARGET_KEYS, targets, strict=True):
        assert probe._expected_program_tokens(task.public) == probe.EXPECTED_TARGET_TOKENS[key]
        assert len(probe._tokenized_public_prompt(task.public)) == probe.EXPECTED_PROMPT_TOKEN_COUNTS[key]

    target_inputs = tuple(probe._target_runtime_input(task.public) for task in targets)
    definition, accounting, target_union = probe._definition(donors, target_inputs)
    defined = {token_id for token_id, _columns in definition.token_codes}
    assert set(target_union) == set().union(*(set(item.prompt_tokens) for item in target_inputs))
    assert all(set(item.prompt_tokens) <= defined for item in target_inputs)
    assert accounting["base_learning_arrivals"] == 642
    assert accounting["constructor_pointer_cue_arrivals"] == 15
    assert accounting["total_learning_arrivals"] == 657
    assert definition.config.segment_capacity == 7_370


def test_frozen_definition_can_train_first_donor_without_unknown_private_cues() -> None:
    _protocol, donors, targets = _selected()
    target_inputs = tuple(probe._target_runtime_input(task.public) for task in targets)
    definition, _accounting, _target_union = probe._definition(donors, target_inputs)
    key = probe.TRAIN_KEYS[0]
    task = donors[0]

    with probe.base._public_development_authority():
        with probe._configured_shared():
            with NativeTemporalCircuit.fresh(definition) as circuit:
                circuit.advance_block(
                    probe._tokenized_public_prompt(task.public),
                    mode=TemporalStepMode.LEARN,
                    reset_before=True,
                )
                circuit.begin_typed_session(
                    probe.shared._workspace_vocabulary(),
                    parameter_type=probe.shared._typed_value(
                        task.public.signature.parameters[0].type
                    ),
                    result_type=probe.shared._typed_value(task.public.signature.result),
                    mode=TemporalStepMode.LEARN,
                )
                teacher_steps = []
                for position, token_id in enumerate(probe._internal_teacher_tokens(key)):
                    try:
                        teacher_steps.append(circuit.teach_typed_token(token_id))
                    except NativeTemporalRuntimeError as error:
                        pytest.fail(
                            f"teacher_position={position} token_id={token_id}: {error}",
                            pytrace=False,
                        )

    assert len(teacher_steps) == probe.EXPECTED_TEACHER_CALL_COUNTS[key]


def test_stop_rule_needs_three_structural_exacts_and_one_full_eos_pass() -> None:
    three_pre_eos_exact = [_cohort_record(exact=True) for _ in range(3)]
    assert probe._stop_decision(three_pre_eos_exact) == (
        "STOP_THREE_STRUCTURAL_EXACT_WITHOUT_EOS_LICENSE"
    )
    assert probe._stop_decision(
        [
            _cohort_record(exact=True, eos_full_pass=True),
            _cohort_record(exact=True),
            _cohort_record(exact=True),
        ]
    ) == "LICENSE_CLOSED_LOOP_EDIT"
    assert probe._stop_decision(
        [_cohort_record(exact=False), _cohort_record(exact=False)]
    ) == "STOP_TWO_STRUCTURAL_FAILURES"
    assert probe._stop_decision(
        [_cohort_record(exact=True), _cohort_record(exact=True), _cohort_record(exact=False)]
    ) == "CONTINUE"


def test_target_markers_have_stable_names_and_exclusive_durable_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_names = [
        "public_simple_swe_native_typed_breadth_probe.t1_list_map_add-005.spent.json",
        "public_simple_swe_native_typed_breadth_probe.t0_scalar_offset-009.spent.json",
        "public_simple_swe_native_typed_breadth_probe.t0_scalar_clamp-009.spent.json",
        "public_simple_swe_native_typed_breadth_probe.t2_clamp_then_sum-007.spent.json",
    ]
    assert [probe._marker_name(key) for key in probe.TARGET_KEYS] == expected_names
    marker_paths = {
        key: tmp_path / probe._marker_name(key) for key in probe.TARGET_KEYS
    }
    monkeypatch.setattr(probe, "SPEND_MARKER_PATHS", marker_paths)
    monkeypatch.setattr(probe, "OUTPUT_PATH", tmp_path / "cohort.json")
    record = probe._create_spend_marker(
        probe.TARGET_KEYS[0], attempt_index=1, checkpoint_sha256="ab" * 32
    )
    assert record["exclusive_create"] is True
    assert record["flushed_and_fsynced_before_generate"] is True
    assert marker_paths[probe.TARGET_KEYS[0]].exists()
    with pytest.raises(RuntimeError, match="spend marker exists"):
        probe._assert_one_shot_paths_absent()
    with pytest.raises(RuntimeError, match="spend marker exists"):
        probe._create_spend_marker(
            probe.TARGET_KEYS[0], attempt_index=1, checkpoint_sha256="ab" * 32
        )


def test_complete_pre_eos_error_prefix_is_retained_and_scores_structural_exact() -> None:
    key = probe.TARGET_KEYS[0]
    expected = probe.EXPECTED_TARGET_TOKENS[key]
    retained = tuple(object() for _ in expected)
    error = NativeTemporalRuntimeError(
        NativeTemporalRuntimeStatus.SCIENTIFIC_STOP,
        "EOS race stopped",
        completed_prefix=len(retained),
        completed_generations=retained,
        poison_code=7,
        stage_code=3,
    )
    assert probe._completed_generations(error) == retained

    record = {
        "generations": [
            {"surface_token_id": probe.PRIMITIVE_TOKEN_IDS[token]}
            for token in expected
        ],
        "copy_trace_exact": True,
        "fitted_cell_races_exact": True,
        "terminated_by_ordinary_eos": False,
        "first_failure": {"gate": "GENERATION_STOP"},
    }
    classification = probe._classify_structural_record(record, expected)
    assert classification == {
        "surface_ids_exact_through_complete_ast": True,
        "physical_copy_control_exact": True,
        "fitted_cell_races_exact": True,
        "structural_exact": True,
        "ordinary_eos": False,
        "end_to_end_pass": False,
        "ordinary_eos_full_pass": False,
        "pre_completion_structural_failure": False,
    }
    record["copy_trace_exact"] = False
    assert probe._classify_structural_record(record, expected)["structural_exact"] is False


def test_checkpoint_restore_and_marker_immediately_guard_each_no_argument_call() -> None:
    source = inspect.getsource(probe.main)
    checkpoint = source.index("checkpoint = circuit.checkpoint()")
    restore = source.index("NativeTemporalCircuit.from_checkpoint(definition, checkpoint)")
    marker = source.index("spend_marker = _create_spend_marker(")
    generation = source.index("raw_generations = restored.generate_typed()")
    assert checkpoint < restore < marker < generation
    assert source.count(".generate_typed()") == 1
    between = source[marker:generation]
    assert "except" not in between
    assert "generate_typed(" not in between
