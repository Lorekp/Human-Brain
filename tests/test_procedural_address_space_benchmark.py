from hashlib import sha256
import json

import pytest

from benchmarks import procedural_address_space as benchmark
from benchmarks.procedural_address_space import _query_topology


def test_address_query_result_uses_truthful_scale_accounting() -> None:
    result = _query_topology(
        cell_count=8,
        fanout=6,
        topology="uniform_no_self",
        seed=71,
        valid_gids=(0, 7),
        invalid_gids=(-1, 8, 9, 1 << 64),
    )

    assert result["valid_cell_queries"] == 2
    assert result["valid_edge_row_queries"] == 2
    assert result["invalid_cell_queries_rejected"] == 4
    assert result["invalid_edge_queries_rejected"] == 4
    assert result["all_invariants_pass"] is True
    accounting = result["scale_accounting"]
    assert accounting["cells"] == {
        "configured": 8,
        "addressable": 8,
        "resident_peak": 0,
        "resident_distinct": 0,
        "updated_distinct": 0,
        "spiking_distinct": 0,
    }
    assert accounting["edges"] == {
        "configured": 48,
        "resident": 0,
        "mutable": 0,
        "generated": 12,
    }
    assert accounting["work"] == {
        "cell_step_evaluations": 0,
        "spike_events": 0,
        "analytic_catchups": 0,
        "analytic_step_equivalents": 0,
    }
    assert accounting["events"] == {
        "delivered": 0,
        "queue_peak": 0,
        "sent_bytes_by_shard_pair": [],
    }
    assert accounting["interval"]["step_count"] == 0
    assert accounting["interval"]["biological_step_seconds"] is None
    assert "no resident or updated cell simulation" in accounting["claim"]["text"]
    assert "tokens_per_second" not in result


def _unexpected_execution(*_: object, **__: object) -> None:
    raise AssertionError("execution crossed a preregistration guard")


def test_cli_defaults_to_a_plan_without_worker_or_artifact_write(
    tmp_path, monkeypatch, capsys
) -> None:
    artifact = tmp_path / "evidence.json"
    monkeypatch.setattr(benchmark, "run", _unexpected_execution)
    monkeypatch.setattr(benchmark, "_worker", _unexpected_execution)

    benchmark.main(["--artifact", str(artifact)])

    plan = json.loads(capsys.readouterr().out)
    assert plan["will_execute"] is False
    assert plan["protocol_sha256"] == sha256(
        benchmark.PROTOCOL_PATH.read_bytes()
    ).hexdigest()
    assert not artifact.exists()


def test_execute_requires_an_exact_raw_protocol_sha_before_subprocess_or_write(
    tmp_path, monkeypatch
) -> None:
    artifact = tmp_path / "evidence.json"
    monkeypatch.setattr(benchmark.subprocess, "check_output", _unexpected_execution)
    monkeypatch.setattr(benchmark.subprocess, "run", _unexpected_execution)

    with pytest.raises(ValueError, match="protocol SHA-256 confirmation"):
        benchmark.run(artifact, confirmed_protocol_sha="0" * 64)

    assert not artifact.exists()


def test_execute_without_confirmation_is_rejected_before_run(
    tmp_path, monkeypatch
) -> None:
    artifact = tmp_path / "evidence.json"
    monkeypatch.setattr(benchmark, "run", _unexpected_execution)

    with pytest.raises(SystemExit):
        benchmark.main(["--execute", "--artifact", str(artifact)])

    assert not artifact.exists()


def test_execute_refuses_an_existing_artifact_before_subprocess_or_write(
    tmp_path, monkeypatch
) -> None:
    artifact = tmp_path / "evidence.json"
    artifact.write_text("sentinel", encoding="utf-8")
    protocol_sha = sha256(benchmark.PROTOCOL_PATH.read_bytes()).hexdigest()
    monkeypatch.setattr(benchmark.subprocess, "check_output", _unexpected_execution)
    monkeypatch.setattr(benchmark.subprocess, "run", _unexpected_execution)

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        benchmark.run(artifact, confirmed_protocol_sha=protocol_sha)

    assert artifact.read_text(encoding="utf-8") == "sentinel"


def test_internal_worker_requires_the_same_protocol_confirmation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(benchmark, "_worker", _unexpected_execution)

    with pytest.raises(ValueError, match="protocol SHA-256 confirmation"):
        benchmark.main(
            [
                "--worker",
                "1000000",
                "--confirm-protocol-sha",
                "0" * 64,
            ]
        )
