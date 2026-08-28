from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "experiments" / "r3_pilot"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import calibration_v3_authority as authority_contract  # noqa: E402
import calibration_v3_gate3_authority as gate3_authority  # noqa: E402
import calibration_v3_protocol as governing_protocol  # noqa: E402
import run_calibration_v3_fixtures as runner  # noqa: E402

GATE3_PROTOCOL_PATH = "benchmarks/procedural_runtime_gate3_slice_protocol_v1.json"
GATE3_SOURCE_FRAMING = (
    "humanbrain-procedural-runtime-source-v1: ordered UTF-8 path length/path "
    "plus uint64 byte length/raw bytes"
)
SYNTHETIC_SOURCE_MANIFEST = ({"path": "src/synthetic.rs", "size": 3, "sha256": "44" * 32},)


class _SyntheticProtocolError(RuntimeError):
    pass


def _synthetic_source_row(payload: bytes) -> dict[str, str]:
    return {
        "commit": "aa" * 20,
        "sha256": hashlib.sha256(governing_protocol.canonical_source_bytes(payload)).hexdigest(),
    }


def _rebind_synthetic_gate3(governing: SimpleNamespace) -> None:
    gate3_path = governing.REPO_ROOT.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    governing.source_binding["files"][GATE3_PROTOCOL_PATH] = _synthetic_source_row(
        gate3_path.read_bytes()
    )


def _synthetic_governing(
    calls: list[str],
    repository_root: Path,
    *,
    declared_bundle_sha256: str = "22" * 32,
    after_validate_source_bytes: Callable[[], None] | None = None,
) -> SimpleNamespace:
    gate3_path = repository_root.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    gate3_path.parent.mkdir(parents=True)
    gate3_bytes = json.dumps(
        {
            "source_binding": {
                "framing": GATE3_SOURCE_FRAMING,
                "files": list(SYNTHETIC_SOURCE_MANIFEST),
                "bundle_sha256": declared_bundle_sha256,
                "limits": {
                    "maximum_file_bytes": 1 << 20,
                    "maximum_aggregate_bytes": 32 << 20,
                    "actual_aggregate_bytes": 3,
                },
            }
        },
        sort_keys=True,
    ).encode("utf-8")
    gate3_path.write_bytes(gate3_bytes)
    binding: dict[str, object] = {
        "files": {GATE3_PROTOCOL_PATH: _synthetic_source_row(gate3_bytes)},
        "governing_gate3_path": GATE3_PROTOCOL_PATH,
        "runnable": True,
    }
    protocol = {"source_binding": binding}

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
        if after_validate_source_bytes is not None:
            after_validate_source_bytes()

    def protocol_sha256() -> str:
        calls.append("protocol_sha256")
        return hashlib.sha256(governing_protocol.canonical_json_bytes(protocol)).hexdigest()

    return SimpleNamespace(
        ProtocolError=_SyntheticProtocolError,
        REPO_ROOT=repository_root,
        build_protocol=build_protocol,
        validate_protocol=validate_protocol,
        validate_committed_source_blobs=validate_committed_source_blobs,
        validate_source_bytes=validate_source_bytes,
        protocol_sha256=protocol_sha256,
        protocol=protocol,
        source_binding=binding,
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{\r\n  "source_binding": {}\r\n}\r\n',
        b"line one\r\nline two\n",
    ],
)
def test_gate3_canonical_source_rule_has_literal_protocol_parity(raw: bytes) -> None:
    assert gate3_authority._canonical_source_bytes(
        raw
    ) == governing_protocol.canonical_source_bytes(raw)


def test_gate3_canonical_source_rule_matches_protocol_lone_cr_rejection() -> None:
    raw = b'{"source_binding": {}}\r'

    with pytest.raises(gate3_authority.Gate3AuthorityError, match="lone CR"):
        gate3_authority._canonical_source_bytes(raw)
    with pytest.raises(governing_protocol.ProtocolError, match="lone CR"):
        governing_protocol.canonical_source_bytes(raw)


def test_authority_accepts_crlf_gate3_bytes_bound_by_the_protocol_rule(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    document = json.loads(gate3_path.read_text(encoding="utf-8"))
    gate3_path.write_bytes(
        json.dumps(document, indent=2, sort_keys=True).replace("\n", "\r\n").encode("utf-8")
    )
    _rebind_synthetic_gate3(governing)

    authority = runner._authorize_frozen_plan(
        governing=governing,
        source_manifest=lambda _root: SYNTHETIC_SOURCE_MANIFEST,
        source_bundle_identity=lambda _root: "22" * 32,
    )

    assert authority.source_bundle_identity_sha256 == "22" * 32


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


def test_authority_gate_order_is_exact_and_retained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    original_open = Path.open

    def observed_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == gate3_path:
            calls.append("gate3_read")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)

    def source_bundle_identity(root: Path) -> str:
        assert root == governing.REPO_ROOT
        calls.append("source_bundle_identity")
        return "22" * 32

    def source_manifest(root: Path) -> tuple[dict[str, str | int], ...]:
        assert root == governing.REPO_ROOT
        calls.append("source_manifest")
        return SYNTHETIC_SOURCE_MANIFEST

    authority = runner._authorize_frozen_plan(
        governing=governing,
        source_manifest=source_manifest,
        source_bundle_identity=source_bundle_identity,
    )

    assert type(authority) is authority_contract.FrozenPlanAuthority
    assert calls == [
        "build_protocol",
        "validate_protocol:runnable",
        "validate_committed_source_blobs",
        "validate_source_bytes",
        "protocol_sha256",
        "gate3_read",
        "source_manifest",
        "source_bundle_identity",
    ]
    assert authority.repository_root == governing.REPO_ROOT
    assert authority.protocol["source_binding"] is not None
    assert (
        authority.protocol_sha256
        == hashlib.sha256(governing_protocol.canonical_json_bytes(governing.protocol)).hexdigest()
    )
    assert authority.source_bundle_identity_sha256 == "22" * 32


def test_atomic_authority_snapshot_failure_stops_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    reached: list[str] = []

    def mutate_after_protocol_digest(root: Path) -> str:
        assert root == governing.REPO_ROOT
        governing.protocol["late_nested_mutation"] = {"split": True}
        return "22" * 32

    monkeypatch.setattr(runner, "_governing_protocol", lambda: governing)
    monkeypatch.setattr(runner, "_source_manifest", lambda _root: SYNTHETIC_SOURCE_MANIFEST)
    monkeypatch.setattr(runner, "_source_bundle_identity", mutate_after_protocol_digest)
    monkeypatch.setattr(
        runner,
        "_execute_frozen_plan",
        lambda _authority: reached.append("execution") or runner.EXIT_OK,
    )

    assert runner.main(["--run-frozen-plan"]) == runner.EXIT_STOP
    assert reached == []
    assert "canonical snapshot" in capsys.readouterr().err


def test_authority_binds_the_exact_gate3_bytes_read_after_live_source_validation(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))

    def replace_gate3_after_validation() -> None:
        replacement_path = gate3_path.with_suffix(".replacement.json")
        replacement_path.write_text(
            json.dumps(
                {
                    "replacement": True,
                    "source_binding": {
                        "framing": GATE3_SOURCE_FRAMING,
                        "files": list(SYNTHETIC_SOURCE_MANIFEST),
                        "bundle_sha256": "22" * 32,
                        "limits": {
                            "maximum_file_bytes": 1 << 20,
                            "maximum_aggregate_bytes": 32 << 20,
                            "actual_aggregate_bytes": 3,
                        },
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        replacement_path.replace(gate3_path)
        calls.append("replace_gate3")

    governing = _synthetic_governing(
        calls,
        tmp_path,
        after_validate_source_bytes=replace_gate3_after_validation,
    )

    def source_manifest(_root: Path) -> tuple[dict[str, str | int], ...]:
        calls.append("source_manifest")
        return SYNTHETIC_SOURCE_MANIFEST

    def source_bundle_identity(_root: Path) -> str:
        calls.append("source_bundle_identity")
        return "22" * 32

    with pytest.raises(runner.FixtureRunnerStop, match="outer source binding"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=source_manifest,
            source_bundle_identity=source_bundle_identity,
        )

    assert calls == [
        "build_protocol",
        "validate_protocol:runnable",
        "validate_committed_source_blobs",
        "validate_source_bytes",
        "replace_gate3",
        "protocol_sha256",
    ]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ([], "row shape"),
        ({"sha256": "aa" * 32}, "row shape"),
        (
            {"commit": "aa" * 20, "sha256": "aa" * 32, "size": 1},
            "row shape",
        ),
        (
            {"commit": "AA" * 20, "sha256": "aa" * 32},
            "Git commit identity",
        ),
        (
            {"commit": "aa" * 20, "sha256": "AA" * 32},
            "lowercase SHA-256",
        ),
    ],
)
def test_authority_rejects_nonexact_outer_gate3_source_binding_row(
    row: object,
    message: str,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    governing.source_binding["files"][GATE3_PROTOCOL_PATH] = row

    with pytest.raises(runner.FixtureRunnerStop, match=message):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after an invalid outer row"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after an invalid outer row"
            ),
        )


def test_authority_rejects_lone_cr_in_bound_gate3_bytes_before_json_parse(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    payload = gate3_path.read_bytes() + b"\r"
    gate3_path.write_bytes(payload)
    governing.source_binding["files"][GATE3_PROTOCOL_PATH] = {
        "commit": "aa" * 20,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    with pytest.raises(runner.FixtureRunnerStop, match="lone CR"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after a lone CR"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after a lone CR"
            ),
        )


def test_authority_rejects_live_bundle_not_approved_by_nested_gate3(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(
        calls,
        tmp_path,
        declared_bundle_sha256="33" * 32,
    )

    with pytest.raises(runner.FixtureRunnerStop, match=r"Gate3.*live source-bundle"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: SYNTHETIC_SOURCE_MANIFEST,
            source_bundle_identity=lambda _root: "22" * 32,
        )


def test_authority_rejects_live_manifest_before_computing_bundle_digest(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    mismatched_manifest = ({"path": "src/synthetic.rs", "size": 3, "sha256": "55" * 32},)

    with pytest.raises(runner.FixtureRunnerStop, match="Gate3 source manifest"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: mismatched_manifest,
            source_bundle_identity=lambda _root: pytest.fail(
                "bundle digest must not run after a manifest mismatch"
            ),
        )


def test_authority_rejects_duplicate_keys_in_bound_gate3_json(tmp_path: Path) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    manifest_json = json.dumps(list(SYNTHETIC_SOURCE_MANIFEST), separators=(",", ":"))
    gate3_path.write_text(
        "{"
        '"source_binding":{'
        f'"framing":"{GATE3_SOURCE_FRAMING}",'
        f'"files":{manifest_json},'
        f'"bundle_sha256":"{"22" * 32}",'
        f'"bundle_sha256":"{"22" * 32}",'
        '"limits":{'
        '"maximum_file_bytes":1048576,'
        '"maximum_aggregate_bytes":33554432,'
        '"actual_aggregate_bytes":3'
        "}}}",
        encoding="utf-8",
    )
    _rebind_synthetic_gate3(governing)

    with pytest.raises(runner.FixtureRunnerStop, match="duplicate JSON key"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: SYNTHETIC_SOURCE_MANIFEST,
            source_bundle_identity=lambda _root: "22" * 32,
        )


def test_authority_rejects_nonstandard_json_constants_anywhere_in_gate3(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    original = gate3_path.read_text(encoding="utf-8")
    gate3_path.write_text('{"invalid":NaN,' + original[1:], encoding="utf-8")
    _rebind_synthetic_gate3(governing)

    with pytest.raises(runner.FixtureRunnerStop, match="nonstandard JSON constant"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after invalid JSON"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after invalid JSON"
            ),
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff", "strict UTF-8 JSON"),
        (b"{", "strict UTF-8 JSON"),
    ],
)
def test_authority_rejects_malformed_gate3_before_live_source_computation(
    payload: bytes,
    message: str,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    gate3_path.write_bytes(payload)
    _rebind_synthetic_gate3(governing)

    with pytest.raises(runner.FixtureRunnerStop, match=message):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after malformed Gate3 JSON"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after malformed Gate3 JSON"
            ),
        )


def test_authority_rejects_noncanonical_nested_gate3_digest_before_live_source(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    document = json.loads(gate3_path.read_text(encoding="utf-8"))
    document["source_binding"]["bundle_sha256"] = "AA" * 32
    gate3_path.write_text(json.dumps(document), encoding="utf-8")
    _rebind_synthetic_gate3(governing)

    with pytest.raises(runner.FixtureRunnerStop, match="lowercase SHA-256"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after a noncanonical digest"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after a noncanonical digest"
            ),
        )


def test_authority_rejects_incomplete_nested_gate3_source_binding(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    document = json.loads(gate3_path.read_text(encoding="utf-8"))
    del document["source_binding"]["limits"]
    gate3_path.write_text(json.dumps(document), encoding="utf-8")
    _rebind_synthetic_gate3(governing)

    with pytest.raises(runner.FixtureRunnerStop, match="source_binding shape"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: SYNTHETIC_SOURCE_MANIFEST,
            source_bundle_identity=lambda _root: "22" * 32,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("framing", "changed-framing", "source framing"),
        ("maximum_file_bytes", (1 << 20) - 1, "source limits"),
        ("maximum_aggregate_bytes", (32 << 20) - 1, "source limits"),
        ("actual_aggregate_bytes", 4, "aggregate bytes"),
    ],
)
def test_authority_rejects_redefined_gate3_framing_or_limits(
    field: str,
    value: str | int,
    message: str,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    document = json.loads(gate3_path.read_text(encoding="utf-8"))
    if field == "framing":
        document["source_binding"][field] = value
    else:
        document["source_binding"]["limits"][field] = value
    gate3_path.write_text(json.dumps(document), encoding="utf-8")
    _rebind_synthetic_gate3(governing)

    with pytest.raises(runner.FixtureRunnerStop, match=message):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after a redefined Gate3 envelope"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after a redefined Gate3 envelope"
            ),
        )


def test_authority_rejects_boolean_manifest_size_even_when_live_shape_matches(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    document = json.loads(gate3_path.read_text(encoding="utf-8"))
    malformed_manifest = ({"path": "src/synthetic.rs", "size": True, "sha256": "44" * 32},)
    document["source_binding"]["files"] = list(malformed_manifest)
    document["source_binding"]["limits"] = {
        "maximum_file_bytes": 1 << 20,
        "maximum_aggregate_bytes": 32 << 20,
        "actual_aggregate_bytes": 1,
    }
    gate3_path.write_text(json.dumps(document), encoding="utf-8")
    _rebind_synthetic_gate3(governing)

    with pytest.raises(runner.FixtureRunnerStop, match=r"manifest.*size"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: malformed_manifest,
            source_bundle_identity=lambda _root: "22" * 32,
        )


def test_authority_rejects_duplicate_nested_manifest_paths_before_live_source(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    document = json.loads(gate3_path.read_text(encoding="utf-8"))
    document["source_binding"]["files"] *= 2
    document["source_binding"]["limits"]["actual_aggregate_bytes"] = 6
    gate3_path.write_text(json.dumps(document), encoding="utf-8")
    _rebind_synthetic_gate3(governing)

    with pytest.raises(runner.FixtureRunnerStop, match="manifest path"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after duplicate nested paths"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after duplicate nested paths"
            ),
        )


def test_authority_rejects_unknown_nested_manifest_record_keys(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    document = json.loads(gate3_path.read_text(encoding="utf-8"))
    document["source_binding"]["files"][0]["unexpected"] = 0
    gate3_path.write_text(json.dumps(document), encoding="utf-8")
    _rebind_synthetic_gate3(governing)

    with pytest.raises(runner.FixtureRunnerStop, match="manifest row 0 shape"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after an unknown record key"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after an unknown record key"
            ),
        )


@pytest.mark.parametrize("mode", ["missing", "unbound", "escape"])
def test_authority_requires_one_exact_bound_governing_gate3_path(
    mode: str,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    protocol = governing.build_protocol()
    calls.clear()
    binding = protocol["source_binding"]
    if mode == "missing":
        del binding["governing_gate3_path"]
    elif mode == "unbound":
        binding["governing_gate3_path"] = "benchmarks/unbound.json"
    else:
        binding["governing_gate3_path"] = "../escape.json"
        binding["files"]["../escape.json"] = {
            "commit": "aa" * 20,
            "sha256": "aa" * 32,
        }

    with pytest.raises(runner.FixtureRunnerStop, match="governing Gate3 path"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after an invalid Gate3 path"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after an invalid Gate3 path"
            ),
        )


def test_authority_rejects_final_gate3_symlink_or_reparse_before_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    original_lstat = os.lstat

    def reparse_lstat(path: os.PathLike[str] | str) -> os.stat_result:
        metadata = original_lstat(path)
        if Path(path) == gate3_path:
            fields = list(metadata)
            fields[0] = stat.S_IFLNK | 0o777
            return os.stat_result(fields)
        return metadata

    monkeypatch.setattr(os, "lstat", reparse_lstat)

    with pytest.raises(runner.FixtureRunnerStop, match="symlink or reparse point"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after a reparse Gate3 path"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after a reparse Gate3 path"
            ),
        )


def test_authority_rejects_windows_reparse_attribute_before_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    monkeypatch.setattr(gate3_authority, "_path_is_reparse", lambda _metadata: True)

    with pytest.raises(runner.FixtureRunnerStop, match="symlink or reparse point"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after a reparse Gate3 path"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after a reparse Gate3 path"
            ),
        )


def test_authority_rejects_oversized_gate3_before_live_source_computation(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    governing = _synthetic_governing(calls, tmp_path)
    gate3_path = tmp_path.joinpath(*GATE3_PROTOCOL_PATH.split("/"))
    gate3_path.write_bytes(b" " * ((1 << 20) + 1))

    with pytest.raises(runner.FixtureRunnerStop, match="Gate3 file size is outside"):
        runner._authorize_frozen_plan(
            governing=governing,
            source_manifest=lambda _root: pytest.fail(
                "live source manifest must not run after an oversized Gate3 file"
            ),
            source_bundle_identity=lambda _root: pytest.fail(
                "live source digest must not run after an oversized Gate3 file"
            ),
        )


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
    monkeypatch.setattr(
        runner,
        "_gate3_authority_module",
        lambda: (
            reached.append("gate3-authority")
            or pytest.fail("Gate3 authority must not import before runnable validation")
        ),
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

    def stop() -> authority_contract.FrozenPlanAuthority:
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
    protocol = {"source_binding": {"runnable": True}}
    authority = authority_contract.FrozenPlanAuthority(
        repository_root=ROOT,
        protocol=protocol,
        protocol_sha256=hashlib.sha256(
            governing_protocol.canonical_json_bytes(protocol)
        ).hexdigest(),
        source_bundle_identity_sha256="22" * 32,
    )
    monkeypatch.setattr(runner, "_authorize_frozen_plan", lambda: authority)
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "_run_electrical_v3_prerequisite",
        lambda: calls.append("electrical"),
    )
    monkeypatch.setattr(
        runner,
        "_run_source_bound_q4_vertical_slice",
        lambda value: calls.append("q4") or {"authority": value.protocol_sha256},
    )
    assert runner.main(["--run-frozen-plan"]) == runner.EXIT_STOP
    assert calls == ["electrical", "q4"]
    assert "remaining calibration-v3 execution" in capsys.readouterr().err


def test_real_native_q4_vertical_slice_reaches_the_frozen_reserve_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import humanbrain.native_temporal_runtime as native

    protocol = governing_protocol.build_protocol()
    protocol_sha256 = hashlib.sha256(
        governing_protocol.canonical_json_bytes(protocol)
    ).hexdigest()
    authority = authority_contract.FrozenPlanAuthority(
        repository_root=ROOT,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        source_bundle_identity_sha256="22" * 32,
    )
    native_authority = native._FreshAuthority(
        bytes.fromhex(protocol_sha256),
        bytes.fromhex(authority.source_bundle_identity_sha256),
    )
    monkeypatch.setattr(native, "_fresh_authority", lambda: native_authority)

    result = runner._run_source_bound_q4_vertical_slice(authority)

    assert result == {
        "circuit_identity_sha256": (
            "1e15d708c81a07c3949097babf94c2c5d7922c3545dac487f8e53e84a3b92220"
        ),
        "cohort_count": 44,
        "next_reserve_cell": 33_120,
        "segment_count": 1_419,
    }
