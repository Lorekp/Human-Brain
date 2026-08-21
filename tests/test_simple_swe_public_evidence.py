from __future__ import annotations

from hashlib import sha256
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / "benchmarks" / "publish_simple_swe_public_baseline.py"
RUNNER = ROOT / "benchmarks" / "simple_swe_public_baseline.py"
if str(RUNNER.parent) not in sys.path:
    sys.path.insert(0, str(RUNNER.parent))
baseline = importlib.import_module("simple_swe_public_baseline")
ABSOLUTE_WORKSTATION_PATH = re.compile(
    rb"(?:[A-Za-z]:[\\/]|/(?:Users|home)/)", re.IGNORECASE
)


def _write_raw_fixture(path: Path) -> bytes:
    windows_separator = "\\"
    repository = windows_separator.join(("C:", "Users", "alice", "repo"))
    interpreter = windows_separator.join(
        ("C:", "Users", "alice", "venv", "Scripts", "python.exe")
    )
    payload = {
        "schema": "human-cell-simple-swe-public-development-baseline-result-v1",
        "status": "noncanonical-public-development-only",
        "checkpoint": {
            "path": windows_separator.join(
                (
                    repository,
                    ".runtime",
                    "simple-swe-public-baseline-v1",
                    "selected_prompt_only_cells.npz",
                )
            ),
            "sha256": "1" * 64,
        },
        "command": {
            "argv": [
                interpreter,
                windows_separator.join(
                    ("benchmarks", "simple_swe_public_baseline.py")
                ),
            ],
            "cwd": repository,
            "windows_command_line": "private workstation command",
        },
        "experiment_protocol": {
            "path": windows_separator.join(
                (
                    repository,
                    "benchmarks",
                    "simple_swe_public_baseline_protocol_v1.json",
                )
            ),
            "sha256": "2" * 64,
        },
        "runtime": {
            "executable": interpreter,
            "executable_sha256": "3" * 64,
            "implementation": "CPython",
            "python_version": "3.11.15",
            "worker_path": windows_separator.join(
                (repository, "src", "humanbrain", "simple_swe", "_worker.py")
            ),
            "worker_sha256": "4" * 64,
        },
        "task_protocol": {
            "path": windows_separator.join(
                (repository, "benchmarks", "simple_swe_protocol_v1.json")
            ),
            "sha256": "5" * 64,
        },
        "development": {
            "aggregate": {
                "passed": 0,
                "inference_seconds": 0.7083915000548586,
            }
        },
    }
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return raw


def test_publisher_sanitizes_paths_without_changing_results(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.json"
    published_path = tmp_path / "published.json"
    record_path = tmp_path / "publication.json"
    archive_path = tmp_path / "archive" / "raw.json"
    raw = _write_raw_fixture(raw_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(PUBLISHER),
            "--raw",
            str(raw_path),
            "--published",
            str(published_path),
            "--record",
            str(record_path),
            "--archive-raw",
            str(archive_path),
            "--expected-raw-sha256",
            sha256(raw).hexdigest(),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    published_bytes = published_path.read_bytes()
    published = json.loads(published_bytes)
    record = json.loads(record_path.read_bytes())
    assert published["development"]["aggregate"] == {
        "inference_seconds": 0.7083915000548586,
        "passed": 0,
    }
    assert published["checkpoint"]["path"] == (
        ".runtime/simple-swe-public-baseline-v1/selected_prompt_only_cells.npz"
    )
    assert published["experiment_protocol"]["path"] == (
        "benchmarks/simple_swe_public_baseline_protocol_v1.json"
    )
    assert published["task_protocol"]["path"] == (
        "benchmarks/simple_swe_protocol_v1.json"
    )
    assert published["runtime"] == {
        "executable_basename": "python.exe",
        "executable_sha256": "3" * 64,
        "implementation": "CPython",
        "python_version": "3.11.15",
        "worker_path": "src/humanbrain/simple_swe/_worker.py",
        "worker_sha256": "4" * 64,
    }
    assert published["command"] == {
        "argv": ["python", "benchmarks/simple_swe_public_baseline.py"],
        "cwd": ".",
        "portable_command": "python benchmarks/simple_swe_public_baseline.py",
    }
    combined = published_bytes + record_path.read_bytes()
    assert ABSOLUTE_WORKSTATION_PATH.search(combined) is None
    assert archive_path.read_bytes() == raw
    assert record["raw_artifact_sha256"] == sha256(raw).hexdigest()
    assert record["published_artifact_sha256"] == sha256(published_bytes).hexdigest()
    assert record["metrics_and_results_unchanged"] is True
    assert record["portable_payload_sha256"] == (
        published["publication"]["portable_payload_sha256"]
    )


def test_publisher_is_byte_deterministic_for_the_same_frozen_raw_input(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.json"
    raw = _write_raw_fixture(raw_path)
    outputs: list[tuple[bytes, bytes]] = []
    for ordinal in (1, 2):
        published_path = tmp_path / f"published-{ordinal}.json"
        record_path = tmp_path / f"publication-{ordinal}.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(PUBLISHER),
                "--raw",
                str(raw_path),
                "--published",
                str(published_path),
                "--record",
                str(record_path),
                "--archive-raw",
                str(tmp_path / f"archive-{ordinal}.json"),
                "--expected-raw-sha256",
                sha256(raw).hexdigest(),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append((published_path.read_bytes(), record_path.read_bytes()))

    assert outputs[0] == outputs[1]


def test_runner_plan_defaults_to_ignored_raw_output_without_touching_evidence() -> None:
    evidence = ROOT / "artifacts" / "simple_swe_public_baseline_v1.json"
    evidence_before = evidence.read_bytes()

    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--plan"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "checked_in_evidence": "artifacts/simple_swe_public_baseline_v1.json",
        "raw_output": ".runtime/simple-swe-public-baseline-v1/raw-result.json",
        "would_overwrite_checked_in_evidence": False,
    }
    assert evidence.read_bytes() == evidence_before


def test_runner_rejects_checked_in_evidence_as_an_output() -> None:
    evidence = ROOT / "artifacts" / "simple_swe_public_baseline_v1.json"
    evidence_before = evidence.read_bytes()

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--plan",
            "--output",
            str(evidence),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "cannot overwrite checked-in evidence" in completed.stderr
    assert evidence.read_bytes() == evidence_before


def _registered_candidates() -> list[dict[str, object]]:
    protocol = json.loads(
        (ROOT / "benchmarks" / "simple_swe_public_baseline_protocol_v1.json").read_bytes()
    )
    return protocol["candidates"]


def _commit_protocol_repository(repository: Path, protocol_bytes: bytes) -> Path:
    protocol = repository / "benchmarks" / "simple_swe_public_baseline_protocol_v1.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes(protocol_bytes)
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "benchmarks"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Baseline Test",
            "-c",
            "user.email=baseline@example.invalid",
            "commit",
            "-q",
            "-m",
            "protocol",
        ],
        cwd=repository,
        check=True,
    )
    return protocol


def test_protocol_validation_rejects_dirty_bytes_that_differ_from_git_blob(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    clean = (
        ROOT / "benchmarks" / "simple_swe_public_baseline_protocol_v1.json"
    ).read_bytes()
    protocol = _commit_protocol_repository(repository, clean)
    protocol.write_bytes(clean + b" ")

    with pytest.raises(ValueError, match="differs from its Git blob"):
        baseline.load_frozen_experiment_protocol(
            protocol, repository_root=repository
        )


def test_protocol_validation_rejects_clean_unregistered_git_blob(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    clean = (
        ROOT / "benchmarks" / "simple_swe_public_baseline_protocol_v1.json"
    ).read_bytes()
    unregistered = clean.replace(b'"max_order": 64', b'"max_order": 63')
    protocol = _commit_protocol_repository(repository, unregistered)

    with pytest.raises(ValueError, match="SHA-256 is not the registered value"):
        baseline.load_frozen_experiment_protocol(
            protocol, repository_root=repository
        )


def test_registered_candidate_validation_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="exactly two registered candidates"):
        baseline.validate_registered_candidates(_registered_candidates()[:1])


def test_registered_candidate_validation_rejects_wrong_id() -> None:
    candidates = _registered_candidates()
    candidates[0]["candidate_id"] = "unregistered"

    with pytest.raises(ValueError, match="IDs or configurations differ"):
        baseline.validate_registered_candidates(candidates)


def test_registered_candidate_validation_rejects_wrong_config() -> None:
    candidates = _registered_candidates()
    candidates[1]["model_config"]["max_order"] = 63

    with pytest.raises(ValueError, match="IDs or configurations differ"):
        baseline.validate_registered_candidates(candidates)


def test_checked_in_evidence_and_report_are_portable_and_precise() -> None:
    evidence_path = ROOT / "artifacts" / "simple_swe_public_baseline_v1.json"
    record_path = (
        ROOT / "artifacts" / "simple_swe_public_baseline_v1.publication.json"
    )
    report_path = ROOT / "docs" / "simple-swe-public-baseline-v1.md"
    evidence_bytes = evidence_path.read_bytes()
    record_bytes = record_path.read_bytes()
    report_bytes = report_path.read_bytes()
    combined = evidence_bytes + record_bytes + report_bytes

    assert ABSOLUTE_WORKSTATION_PATH.search(combined) is None
    evidence = json.loads(evidence_bytes)
    record = json.loads(record_bytes)
    assert evidence["schema"].endswith("publication-v1")
    assert evidence["development"]["aggregate"]["passed"] == 0
    assert evidence["development"]["aggregate"]["inference_seconds"] == (
        0.7083915000548586
    )
    assert record["raw_artifact_sha256"] == (
        "bbfa23cde2ce7d8bbf01119e7b253cd1d28282d9fada9b3c551edcb2c4e44900"
    )
    assert record["metrics_and_results_unchanged"] is True
    report = report_bytes.decode("utf-8")
    assert "python benchmarks/simple_swe_public_baseline.py" in report
    assert "measured and environment-dependent" in report
    assert "behavior reproducibility" in report
    assert "benchmark-only instrumentation debt" in report
