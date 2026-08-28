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

import public_simple_swe_role_chunk_layer0_action_probe as probe


@pytest.fixture(scope="module")
def prepared():
    return probe._prepare_probe()


def test_parent_license_sources_and_unspent_target_are_exact() -> None:
    parent = probe._assert_parent_license()
    probe._assert_frozen_sources()
    inventory = probe._durable_marker_inventory()

    assert parent["outcome"] == "LICENSE_CONCEPT_REPETITION_CONDITIONED_ROUTE"
    assert parent["gate"]["selected_representation"] == "concept"
    assert (
        parent["train"]["concept_true"]["checkpoint_sha256"]
        == probe.EXPECTED_CHECKPOINT_SHA256
    )
    assert probe._sha256_file(probe.PARENT_ARTIFACT_PATH) == (
        probe.EXPECTED_PARENT_ARTIFACT_SHA256
    )
    assert all(probe.TARGET_TASK_ID not in path["path"] for path in inventory)


def test_rebuilt_checkpoint_and_encoded_009_match_the_license(prepared) -> None:
    (
        _protocol,
        target,
        episode,
        definition,
        checkpoint,
        _identity,
        train_record,
        capacity,
    ) = prepared

    assert target.public.task_id == probe.TARGET_TASK_ID
    assert target.public.partition == "train"
    assert episode.structural_sha256 == probe.TARGET_STRUCTURAL_SHA256
    assert probe._sha256_json(tuple(episode.token_ids)) == (
        probe.TARGET_CONCEPT_STREAM_SHA256
    )
    assert probe._sha256_file_bytes(checkpoint) == probe.EXPECTED_CHECKPOINT_SHA256
    assert train_record["checkpoint_sha256"] == probe.EXPECTED_CHECKPOINT_SHA256
    assert definition.config.column_count == 4_096
    assert definition.config.theta_match == 16
    assert definition.config.theta_active == 18
    assert capacity["segment_capacity"] == 11_242


def test_observed_route_alone_dispatches_one_existing_layer0_action(prepared) -> None:
    _protocol, target, episode, *_rest = prepared

    action = probe._dispatch_action(probe.EXPECTED_ROUTE_TOKEN_ID)
    assert action == ("MAX", "ZERO", "ARG0", "END")
    assert probe._dispatch_action(1537) is None
    assert probe._dispatch_action(None) is None

    compiled = probe._compile_action(action, episode, target.public)
    assert compiled["source"] == (
        "def solve_58866c1699d816b6(n):\n"
        "    return max(0, n)\n"
    )
    assert compiled["max_nodes"] == 3
    assert compiled["complete_function_exact"] is True
    assert compiled["compiler_received_prompt"] is False
    assert compiled["compiler_received_private_cases"] is False


def test_spend_marker_is_exclusive_and_durable_before_one_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared
) -> None:
    _protocol, target, *_rest = prepared
    marker_path = tmp_path / "target.spent.json"
    monkeypatch.setattr(probe, "SPEND_MARKER_PATH", marker_path)
    monkeypatch.setattr(probe, "OUTPUT_PATH", tmp_path / "artifact.json")
    monkeypatch.setattr(probe, "_display_path", lambda path: Path(path).name)

    record = probe._create_spend_marker(target.public)
    assert record["exclusive_create"] is True
    assert record["flushed_and_fsynced_before_single_generate_call"] is True
    assert marker_path.exists()
    with pytest.raises(RuntimeError, match="already spent"):
        probe._create_spend_marker(target.public)


def test_source_orders_prompt_marker_generate_dispatch_compile_then_sandbox() -> None:
    source = inspect.getsource(probe.main)
    prompt = source.index("prompt_steps = circuit.advance_block(")
    marker = source.index("marker = _create_spend_marker(target.public)")
    generation = source.index("raw_generation = route_probe._generate_once(circuit)")
    dispatch = source.index("action = _dispatch_action(observed_route)")
    compile_action = source.index("_compile_action(action, episode, target.public)")
    sandbox = source.index("sandbox = _sandbox_record(")

    assert prompt < marker < generation < dispatch < compile_action < sandbox
    assert source.count("route_probe._generate_once(") == 1
    assert probe.ACTUATOR_BY_ROUTE_TOKEN == {
        1536: ("MAX", "ZERO", "ARG0", "END")
    }
