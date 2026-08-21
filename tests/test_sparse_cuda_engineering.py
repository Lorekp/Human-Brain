from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "sparse_cuda_engineering.py"
SPEC = importlib.util.spec_from_file_location("sparse_cuda_engineering", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_registered_200k_configuration_has_no_dormant_padding() -> None:
    protocol = json.loads(BENCHMARK.PROTOCOL_PATH.read_text(encoding="utf-8"))
    config = BENCHMARK._config(protocol)

    assert config.reservoir_cells == 200_000
    assert config.active_reservoir_cells == 200_000
    assert config.readout_learning_rule == "sparse_three_factor"
    assert config.readout_fan_in == 64
    assert protocol["data_boundary"]["source"] == "synthetic_token_ids_only"


def test_execution_is_hash_locked_and_plan_mode_does_not_construct_cells(
    tmp_path: Path,
) -> None:
    plan = BENCHMARK.run(tmp_path / "unused.json", execute=False)
    assert plan["status"] == "plan-only"
    assert plan["cells"] == 200_000

    with pytest.raises(ValueError, match="protocol SHA-256"):
        BENCHMARK.run(
            tmp_path / "unused.json",
            execute=True,
            expected_protocol_sha256="0" * 64,
        )


def test_report_persistence_uses_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "report.json"
    replacements: list[tuple[Path, Path]] = []
    real_replace = BENCHMARK.os.replace

    def replace(source, destination) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(BENCHMARK.os, "replace", replace)
    BENCHMARK._write_report(artifact, {"status": "running", "repeats": []})

    assert len(replacements) == 1
    assert replacements[0][1] == artifact
    assert json.loads(artifact.read_text(encoding="utf-8"))["status"] == "running"
    assert not replacements[0][0].exists()


def test_execution_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    artifact = tmp_path / "existing.json"
    artifact.write_text("preserve me", encoding="utf-8")
    protocol_sha256 = BENCHMARK.run(artifact, execute=False)["protocol_sha256"]

    with pytest.raises(FileExistsError, match="already exists"):
        BENCHMARK.run(
            artifact,
            execute=True,
            expected_protocol_sha256=protocol_sha256,
        )

    assert artifact.read_text(encoding="utf-8") == "preserve me"
