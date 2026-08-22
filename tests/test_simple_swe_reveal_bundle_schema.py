from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from humanbrain.simple_swe import reveal
from humanbrain.simple_swe.freeze import FreezeIntegrityError
from humanbrain.simple_swe.tasks import canonical_json_bytes, canonical_sha256


def _valid_bundle_payload() -> dict[str, object]:
    signature = {
        "entry_point": "solve_synthetic",
        "parameters": [
            {
                "name": "value",
                "type": {"constructor": "Int", "arguments": []},
            }
        ],
        "result": {"constructor": "Int", "arguments": []},
    }
    regression_cases = [
        {"args": [1], "expected": 2, "distribution": "fixed-regression"}
    ]
    property_cases = [
        {"args": [7], "expected": 8, "distribution": "random-small"}
    ]
    private_commitment = {
        "task_id": "synthetic/eval/000",
        "signature": signature,
        "regression_cases": regression_cases,
        "property_cases": property_cases,
    }
    test_sha256 = canonical_sha256(private_commitment)
    public_commitment = {
        "task_id": "synthetic/eval/000",
        "partition": "eval",
        "tier": 0,
        "family": "synthetic",
        "ordinal": 0,
        "signature": signature,
        "prompt": "Return one synthetic function.",
        "hidden_test_sha256": test_sha256,
    }
    task_sha256 = canonical_sha256(public_commitment)
    seed = bytes(range(32))
    return {
        "schema": "humanbrain.simple-swe-private-eval-bundle.v3",
        "protocol_id": "synthetic-protocol",
        "protocol_sha256": "1" * 64,
        "freeze_manifest_sha256": "2" * 64,
        "checkpoint_sha256": "3" * 64,
        "checkpoint_state_sha256": "4" * 64,
        "source_head": "5" * 40,
        "eval_seed_hex": seed.hex(),
        "eval_seed_sha256": sha256(seed).hexdigest(),
        "tasks": [
            {
                "public": {**public_commitment, "task_sha256": task_sha256},
                "private": {
                    "task_id": "synthetic/eval/000",
                    "public_task_sha256": task_sha256,
                    "regression_cases": regression_cases,
                    "property_cases": property_cases,
                    "test_sha256": test_sha256,
                },
            }
        ],
    }


def _write_bundle(path: Path, payload: dict[str, object]) -> Path:
    path.write_bytes(canonical_json_bytes(payload))
    return path


def _nested_record(
    payload: dict[str, object], path: tuple[str | int, ...]
) -> dict[str, object]:
    value: object = payload
    for part in path:
        if isinstance(part, int):
            assert isinstance(value, list)
            value = value[part]
        else:
            assert isinstance(value, dict)
            value = value[part]
    assert isinstance(value, dict)
    return value


def test_bundle_accepts_the_exact_v3_record_shape(tmp_path: Path) -> None:
    bundle = reveal.load_evaluation_bundle(
        _write_bundle(tmp_path / "bundle.json", _valid_bundle_payload())
    )

    assert bundle.tasks[0].public.task_id == "synthetic/eval/000"


def test_bundle_rejects_an_extra_top_level_field(tmp_path: Path) -> None:
    payload = _valid_bundle_payload()
    payload["unexpected"] = True

    with pytest.raises(FreezeIntegrityError, match="bundle fields"):
        reveal.load_evaluation_bundle(_write_bundle(tmp_path / "bundle.json", payload))


def test_bundle_rejects_an_extra_task_wrapper_field(tmp_path: Path) -> None:
    payload = _valid_bundle_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    wrapper = tasks[0]
    assert isinstance(wrapper, dict)
    wrapper["unexpected"] = True

    with pytest.raises(FreezeIntegrityError, match="task wrapper fields"):
        reveal.load_evaluation_bundle(_write_bundle(tmp_path / "bundle.json", payload))


def test_bundle_rejects_an_extra_public_task_field(tmp_path: Path) -> None:
    payload = _valid_bundle_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    wrapper = tasks[0]
    assert isinstance(wrapper, dict)
    public = wrapper["public"]
    assert isinstance(public, dict)
    public["unexpected"] = True

    with pytest.raises(FreezeIntegrityError, match="public task fields"):
        reveal.load_evaluation_bundle(_write_bundle(tmp_path / "bundle.json", payload))


def test_bundle_rejects_an_extra_private_task_field(tmp_path: Path) -> None:
    payload = _valid_bundle_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    wrapper = tasks[0]
    assert isinstance(wrapper, dict)
    private = wrapper["private"]
    assert isinstance(private, dict)
    private["unexpected"] = True

    with pytest.raises(FreezeIntegrityError, match="private task fields"):
        reveal.load_evaluation_bundle(_write_bundle(tmp_path / "bundle.json", payload))


def test_bundle_rejects_an_extra_hidden_case_field(tmp_path: Path) -> None:
    payload = _valid_bundle_payload()
    tasks = payload["tasks"]
    assert isinstance(tasks, list)
    wrapper = tasks[0]
    assert isinstance(wrapper, dict)
    private = wrapper["private"]
    assert isinstance(private, dict)
    regression_cases = private["regression_cases"]
    assert isinstance(regression_cases, list)
    case = regression_cases[0]
    assert isinstance(case, dict)
    case["unexpected"] = True

    with pytest.raises(FreezeIntegrityError, match="hidden case fields"):
        reveal.load_evaluation_bundle(_write_bundle(tmp_path / "bundle.json", payload))


@pytest.mark.parametrize(
    ("path", "field", "message"),
    (
        ((), "schema", "bundle fields"),
        (("tasks", 0), "private", "task wrapper fields"),
        (("tasks", 0, "public"), "task_id", "public task fields"),
        (("tasks", 0, "private"), "test_sha256", "private task fields"),
        (
            ("tasks", 0, "private", "property_cases", 0),
            "args",
            "hidden case fields",
        ),
    ),
)
def test_bundle_rejects_a_missing_field_at_every_record_boundary(
    tmp_path: Path,
    path: tuple[str | int, ...],
    field: str,
    message: str,
) -> None:
    payload = _valid_bundle_payload()
    del _nested_record(payload, path)[field]

    with pytest.raises(FreezeIntegrityError, match=message):
        reveal.load_evaluation_bundle(_write_bundle(tmp_path / "bundle.json", payload))
