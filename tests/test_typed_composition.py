from __future__ import annotations

import io
import json
import re
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest
from humanbrain.typed_composition import (
    CompletionRequest,
    derive_opaque_identifier,
    load_composition_protocol,
)

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "benchmarks" / "typed_composition_protocol_v1.json"
EXPECTED_PROTOCOL_SHA256 = "2d41d366611c6217dd8a14411b68bc03be1cafb4572e41f3c5a77da326cda412"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_protocol_loader_caps_the_only_read_before_hashing_or_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = PROTOCOL_PATH.read_bytes()
    read_sizes: list[int] = []
    original_open = Path.open

    class RecordingBytes(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return super().read(size)

    def recording_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == PROTOCOL_PATH:
            mode = args[0] if args else kwargs.pop("mode", None)
            assert mode == "rb"
            assert len(args) <= 1
            assert not kwargs
            return RecordingBytes(raw)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    protocol = load_composition_protocol(PROTOCOL_PATH)

    assert protocol.raw_sha256 == EXPECTED_PROTOCOL_SHA256
    assert read_sizes == [32 * 1024 + 1]


def test_frozen_protocol_mechanically_derives_exact_corpus_arithmetic() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    accounting = protocol.accounting

    assert protocol.raw_sha256 == sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    assert protocol.raw_sha256 == EXPECTED_PROTOCOL_SHA256
    assert tuple(lane.lane for lane in protocol.lanes) == ("A0", "A1", "A2")
    assert tuple(lane.eligible for lane in protocol.lanes) == (False, False, True)
    assert accounting.train_task_count == 42
    assert accounting.calibration_task_count == 18
    assert accounting.train_case_count == 126
    assert accounting.calibration_case_count == 54
    assert accounting.teacher_targets_per_pass == 272
    assert accounting.framing_tokens_per_pass == 8_484
    assert accounting.observations_per_pass == 8_714
    assert accounting.corpus_passes == 3
    assert accounting.a2_strengthen_calls == 816
    assert accounting.a0_a1_strengthen_calls == 26_142
    assert accounting.observation_calls_per_lane == 26_142


def test_every_signature_identifier_is_domain_derived_opaque_and_semantically_empty() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    tasks = (*protocol.train_tasks, *protocol.calibration_tasks)
    entry_points: set[str] = set()

    for task in tasks:
        signature = task.signature
        signature_text = _canonical_json(signature.to_record()).decode("utf-8").lower()
        assert signature.entry_point == derive_opaque_identifier(
            protocol.identifier_domain,
            task.opaque_ordinal,
        )
        assert re.fullmatch(r"q_[bcdfghjklmnpqrstvwxyz]{9}", signature.entry_point)
        assert signature.entry_point not in entry_points
        entry_points.add(signature.entry_point)
        assert tuple(signature.to_record()) == ("entry_point", "parameters", "result")
        assert tuple(parameter.name for parameter in signature.parameters) == ("datum",)

        forbidden = {
            task.task_id.lower(),
            task.family_id.lower(),
            task.template_id.lower(),
            "teacher",
            "add",
            "plus",
            "minus",
            "increase",
            "decrease",
            "identity",
            "scalar",
            "list_add",
            "reverse",
            "subtract",
            "variable-0",
            "integer",
            "bind",
            "map",
        }
        if task.literal is not None:
            forbidden.add(str(abs(task.literal)))
        assert all(atom not in signature_text for atom in forbidden)


def test_task_metadata_and_labels_are_removed_at_both_model_boundaries() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    for task in (*protocol.train_tasks, *protocol.calibration_tasks):
        request = task.completion_request()
        assert isinstance(request, CompletionRequest)
        assert tuple(asdict(request)) == ("prompt", "signature")
        assert request.prompt == task.prompt
        assert request.signature == task.signature
        assert "task_id" not in asdict(request)
        assert "family_id" not in asdict(request)
        assert "template_id" not in asdict(request)
        assert "teacher_tokens" not in asdict(request)
        assert "cases" not in asdict(request)

    assert len(protocol.training_examples()) == 42
    with pytest.raises(ValueError, match="TRAIN tasks only"):
        protocol.calibration_tasks[0].training_example()


def test_a0_a1_share_exact_components_and_a1_a2_share_exact_observation_streams() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)

    for task in protocol.train_tasks:
        components = protocol.framing_components(task)
        assert components["A0"].prompt_bytes == components["A1"].prompt_bytes
        assert components["A0"].signature_bytes == components["A1"].signature_bytes
        assert components["A0"].vocabulary_sha256 == components["A1"].vocabulary_sha256
        assert components["A1"].token_ids == components["A2"].token_ids
        assert components["A1"].state_observation_ids == (components["A2"].state_observation_ids)
        assert components["A1"].strengthen_positions != (components["A2"].strengthen_positions)


def test_future_cohort_is_committed_but_has_no_seed_or_generated_tasks() -> None:
    protocol = load_composition_protocol(PROTOCOL_PATH)
    future = protocol.future_cohort

    assert future.seed is None
    assert future.generated is False
    assert future.task_count == 16
    assert future.factor_counts == {
        "map_input_minus_literal": 6,
        "map_literal_minus_input": 6,
        "scalar_subtract_controls": 2,
        "map_add_controls": 2,
    }
    assert future.forbidden_sources == (
        "DEV-v1",
        "hidden",
        "canonical",
        "Fly-Brain",
    )


def test_protocol_spends_the_cohort_once_and_reserves_eligibility_for_execution() -> None:
    record = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert record["execution"]["resume_policy"] == (
        "no resume and no second attempt of this protocol/cohort under any run "
        "identifier after success or failure"
    )
    assert "independent of metadata run ID" in record["execution"]["evidence_mode"]
    assert ".typed_composition_a012_v1.claimed" in (
        record["execution"]["evidence_mode"]
    )
    assert ".typed_composition_a012_v1.claiming" in (
        record["execution"]["evidence_mode"]
    )
    assert "public sealing/scoring and every preterminal evidence record" in (
        record["execution"]["selection_policy"]
    )
    assert record["eligibility_gates"]["mandatory_common"][-2:] == [
        "exclusive-protocol-cohort-attempt",
        "terminal-evidence-envelope-integrity",
    ]
    assert "sole eligibility authority" in (
        record["execution"]["selection_policy"]
    )
