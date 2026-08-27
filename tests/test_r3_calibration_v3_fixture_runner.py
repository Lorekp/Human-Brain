from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "experiments" / "r3_pilot"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import calibration_v3_protocol as governing_protocol
import run_calibration_v3_fixtures as runner


class _SyntheticProtocolError(RuntimeError):
    pass


def _synthetic_governing(calls: list[str]) -> SimpleNamespace:
    binding: dict[str, object] = {"files": {}, "runnable": True}
    protocol = {"source_binding": binding}
    repository_root = Path("synthetic-repository-root")

    def build_protocol() -> dict[str, object]:
        calls.append("build_protocol")
        return protocol

    def validate_protocol(value: object, *, require_runnable: bool) -> None:
        assert value is protocol
        assert require_runnable is True
        calls.append("validate_protocol:runnable")

    def validate_committed_source_blobs(root: Path, value: object) -> None:
        assert root == repository_root
        assert value is binding
        calls.append("validate_committed_source_blobs")

    def validate_source_bytes(root: Path, value: object) -> None:
        assert root == repository_root
        assert value is binding
        calls.append("validate_source_bytes")

    def protocol_sha256() -> str:
        calls.append("protocol_sha256")
        return "11" * 32

    return SimpleNamespace(
        ProtocolError=_SyntheticProtocolError,
        REPO_ROOT=repository_root,
        build_protocol=build_protocol,
        validate_protocol=validate_protocol,
        validate_committed_source_blobs=validate_committed_source_blobs,
        validate_source_bytes=validate_source_bytes,
        protocol_sha256=protocol_sha256,
    )


def test_import_is_transport_only_and_does_not_load_execution_dependencies() -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import humanbrain.native as native; "
                "assert native._LIBRARY is None; "
                "import run_calibration_v3_fixtures; "
                "assert native._LIBRARY is None; "
                "import sys; "
                "forbidden = ("
                "'humanbrain.native_temporal_runtime',"
                "'humanbrain.tokenizer',"
                "'humanbrain.concept_tokens',"
                "'experiments.r3_pilot.run_c1_cohort',"
                "'run_c1_cohort'"
                "); "
                "assert all(name not in sys.modules for name in forbidden)"
            ),
        ],
        cwd=RUNNER_DIR,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_authority_gate_order_is_exact_and_retained() -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls)

    def source_bundle_identity(root: Path) -> str:
        assert root == governing.REPO_ROOT
        calls.append("source_bundle_identity")
        return "22" * 32

    authority = runner._authorize_frozen_plan(
        governing=governing,
        source_bundle_identity=source_bundle_identity,
    )

    assert calls == [
        "build_protocol",
        "validate_protocol:runnable",
        "validate_committed_source_blobs",
        "validate_source_bytes",
        "protocol_sha256",
        "source_bundle_identity",
    ]
    assert authority.repository_root == governing.REPO_ROOT
    assert authority.protocol["source_binding"] is not None
    assert authority.protocol_sha256 == "11" * 32
    assert authority.source_bundle_identity_sha256 == "22" * 32


def test_current_non_runnable_protocol_stops_before_any_execution_or_file_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reached: list[str] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "_governing_protocol", lambda: governing_protocol)
    monkeypatch.setattr(
        runner,
        "_source_bundle_identity",
        lambda _root: reached.append("source-bundle") or "22" * 32,
    )
    monkeypatch.setattr(
        runner,
        "_execute_frozen_plan",
        lambda _authority: reached.append("execution") or runner.EXIT_OK,
    )

    def forbidden_path_operation(*_args: object, **_kwargs: object) -> None:
        reached.append("filesystem")
        raise AssertionError("file mutation/open must not be reached")

    monkeypatch.setattr(Path, "open", forbidden_path_operation)
    monkeypatch.setattr(Path, "mkdir", forbidden_path_operation)
    monkeypatch.setattr(Path, "write_text", forbidden_path_operation)
    monkeypatch.setattr(Path, "write_bytes", forbidden_path_operation)

    code = runner.main(["--run-frozen-plan"])

    assert code == runner.EXIT_STOP
    assert reached == []
    assert list(tmp_path.iterdir()) == []
    error = capsys.readouterr().err
    assert "STOP" in error
    assert "protocol is not runnable" in error


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--output-dir", "x"],
        ["--force"],
        ["--seed", "17"],
        ["--arm", "theta-m-10_theta-a-13"],
        ["--screen", "r3-f0"],
        ["--held-out"],
        ["--resume", "checkpoint"],
        ["--run-frozen-plan", "--force"],
        ["--run-frozen-plan", "unexpected"],
    ],
)
def test_arbitrary_and_forbidden_cli_knobs_stop_before_authority(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_authorize_frozen_plan",
        lambda: pytest.fail("authority must not run for rejected arguments"),
    )

    assert runner.main(arguments) == runner.EXIT_STOP


def test_failed_authority_has_no_execution_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached: list[str] = []

    def stop() -> runner.FrozenPlanAuthority:
        reached.append("authority")
        raise runner.FixtureRunnerStop("synthetic authority STOP")

    monkeypatch.setattr(runner, "_authorize_frozen_plan", stop)
    monkeypatch.setattr(
        runner,
        "_execute_frozen_plan",
        lambda _authority: reached.append("execution") or runner.EXIT_OK,
    )

    assert runner.main(["--run-frozen-plan"]) == runner.EXIT_STOP
    assert reached == ["authority"]


def test_even_a_synthetic_future_authority_reaches_only_the_disabled_seam(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authority = runner.FrozenPlanAuthority(
        repository_root=ROOT,
        protocol={"source_binding": {"runnable": True}},
        protocol_sha256="11" * 32,
        source_bundle_identity_sha256="22" * 32,
    )
    monkeypatch.setattr(runner, "_authorize_frozen_plan", lambda: authority)

    assert runner.main(["--run-frozen-plan"]) == runner.EXIT_STOP
    assert "execution remains disabled" in capsys.readouterr().err
