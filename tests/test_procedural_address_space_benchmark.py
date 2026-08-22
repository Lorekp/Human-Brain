from copy import deepcopy
from hashlib import sha256
import json

import pytest

from benchmarks import procedural_address_space as benchmark
from benchmarks.procedural_address_space import _query_topology
from humanbrain.procedural_topology import topology_implementation_contract
from humanbrain.simple_swe.tasks import canonical_sha256


_EXPECTED_IMPLEMENTATION_SOURCE_PATHS = [
    "benchmarks/procedural_address_space.py",
    "src/humanbrain/human_cell_identity.py",
    "src/humanbrain/procedural_topology.py",
    "src/humanbrain/scale_accounting.py",
    "src/humanbrain/simple_swe/artifacts.py",
    "src/humanbrain/simple_swe/tasks.py",
]
_EXPECTED_V2_PROTOCOL_SHA256 = (
    "cb3c66402c9d1b88a4a6bf804bf232d7cba2741d4bc469af32a97bad53f40139"
)


def _leaf_paths(value: object, prefix: tuple[object, ...] = ()):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaf_paths(value[key], prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, prefix + (index,))
    else:
        yield prefix


def _replace_path(root: object, path: tuple[object, ...], value: object) -> None:
    cursor = root
    for step in path[:-1]:
        cursor = cursor[step]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]


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


def test_mixed_query_reports_the_sampled_membership_distinctness_claim() -> None:
    result = _query_topology(
        cell_count=8,
        fanout=6,
        topology="mixed_membership",
        seed=71,
        valid_gids=(0, 3, 7),
        invalid_gids=(-1, 8),
    )

    assert result["mixed_membership_distinct"] == {
        "evaluated": True,
        "sampled_descriptors": 3,
        "passed": True,
    }
    assert result["all_invariants_pass"] is True


def test_v2_protocol_matches_the_complete_runtime_implementation_contract() -> None:
    protocol = json.loads(benchmark.PROTOCOL_PATH.read_bytes())

    assert benchmark.PROTOCOL_PATH.name == "procedural_address_space_protocol_v2.json"
    assert sha256(benchmark.PROTOCOL_PATH.read_bytes()).hexdigest() == (
        _EXPECTED_V2_PROTOCOL_SHA256
    )
    assert protocol["name"] == "procedural-address-space-v2"
    assert protocol["benchmark_protocol_version"] == 2
    assert protocol["implementation_contract"] == topology_implementation_contract()
    assert protocol["implementation_sources"] == benchmark.implementation_source_bundle()
    assert protocol["implementation_sources"]["bundle_format"] == (
        "sha256-canonical-json-path-size-raw-sha256-v1"
    )
    assert [
        file_record["path"]
        for file_record in protocol["implementation_sources"]["files"]
    ] == _EXPECTED_IMPLEMENTATION_SOURCE_PATHS
    benchmark.validate_protocol(protocol)


def test_every_duplicated_protocol_leaf_is_validated_against_implementation() -> None:
    protocol = json.loads(benchmark.PROTOCOL_PATH.read_bytes())
    contract = protocol["implementation_contract"]

    for path in _leaf_paths(contract):
        drifted = deepcopy(protocol)
        _replace_path(drifted["implementation_contract"], path, None)
        with pytest.raises(ValueError, match="implementation contract"):
            benchmark.validate_protocol(drifted)


def test_every_declared_implementation_source_leaf_is_validated() -> None:
    protocol = json.loads(benchmark.PROTOCOL_PATH.read_bytes())
    implementation_sources = protocol["implementation_sources"]

    for path in _leaf_paths(implementation_sources):
        drifted = deepcopy(protocol)
        _replace_path(drifted["implementation_sources"], path, None)
        with pytest.raises(ValueError, match="implementation source bundle"):
            benchmark.validate_protocol(drifted)


@pytest.mark.parametrize(
    ("path", "error"),
    [
        (("name",), "name"),
        (("benchmark_protocol_version",), "version"),
        (("publication", "artifact_kind"), "publication contract"),
        (("publication", "artifact_schema"), "publication contract"),
        (("publication", "policy"), "publication contract"),
    ],
)
def test_every_runner_protocol_duplicate_is_validated(
    path: tuple[object, ...],
    error: str,
) -> None:
    protocol = json.loads(benchmark.PROTOCOL_PATH.read_bytes())
    _replace_path(protocol, path, None)

    with pytest.raises(ValueError, match=error):
        benchmark.validate_protocol(protocol)


def test_protocol_rejects_any_other_evidence_scope() -> None:
    protocol = json.loads(benchmark.PROTOCOL_PATH.read_bytes())
    protocol["evidence_scope"] = "CPU scalar address and capability evidence"

    with pytest.raises(ValueError, match="evidence scope"):
        benchmark.validate_protocol(protocol)


def test_preflight_executes_and_records_kats_reference_and_source_bundle() -> None:
    preflight = benchmark.topology_implementation_preflight()

    assert all(row["passed"] for row in preflight["philox_known_answers"])
    assert preflight["small_materialized_reference"] == {
        "cell_count": 17,
        "fanout": 4,
        "seed": 71,
        "byte_order": "little",
        "sha256": "ba55ded147218d524bf4d004d97a97f42d3471c3079ad110fa4a0451f4f92b60",
        "passed": True,
    }
    assert preflight["implementation_sources"] == benchmark.implementation_source_bundle()


def test_preflight_failure_stops_before_git_workers_or_publication(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "evidence.json"
    protocol_sha = sha256(benchmark.PROTOCOL_PATH.read_bytes()).hexdigest()

    def fail_preflight() -> None:
        raise RuntimeError("preflight stopped")

    monkeypatch.setattr(benchmark, "topology_implementation_preflight", fail_preflight)
    monkeypatch.setattr(benchmark.subprocess, "check_output", _unexpected_execution)
    monkeypatch.setattr(benchmark.subprocess, "run", _unexpected_execution)

    with pytest.raises(RuntimeError, match="preflight stopped"):
        benchmark.run(artifact, confirmed_protocol_sha=protocol_sha)

    assert not artifact.exists()


def test_runner_rejects_source_bundle_drift_before_preflight_or_workers(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "evidence.json"
    drifted_path = tmp_path / "protocol.json"
    protocol = json.loads(benchmark.PROTOCOL_PATH.read_bytes())
    protocol["implementation_sources"]["bundle_sha256"] = "0" * 64
    drifted_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = sha256(drifted_path.read_bytes()).hexdigest()

    monkeypatch.setattr(benchmark, "PROTOCOL_PATH", drifted_path)
    monkeypatch.setattr(
        benchmark, "topology_implementation_preflight", _unexpected_execution
    )
    monkeypatch.setattr(benchmark.subprocess, "check_output", _unexpected_execution)
    monkeypatch.setattr(benchmark.subprocess, "run", _unexpected_execution)

    with pytest.raises(ValueError, match="implementation source bundle"):
        benchmark.run(artifact, confirmed_protocol_sha=protocol_sha)

    assert not artifact.exists()


@pytest.mark.parametrize("mixed_membership_passed", [True, False])
def test_future_run_publishes_one_atomic_v2_envelope_with_durable_sha(
    tmp_path,
    monkeypatch,
    mixed_membership_passed: bool,
) -> None:
    artifact = tmp_path / "evidence.json"
    protocol = json.loads(benchmark.PROTOCOL_PATH.read_bytes())
    protocol_sha = sha256(benchmark.PROTOCOL_PATH.read_bytes()).hexdigest()
    source_sha = protocol["implementation_sources"]["bundle_sha256"]
    topologies = protocol["implementation_contract"]["topology_names"]

    def fake_check_output(args, **_: object) -> str:
        if args[:2] == ["git", "status"]:
            return ""
        if args[:2] == ["git", "rev-parse"]:
            return "a" * 40 + "\n"
        raise AssertionError(f"unexpected command: {args}")

    def fake_worker(args, **_: object):
        assert args[args.index("--confirm-source-sha") + 1] == source_sha
        cell_count = int(args[args.index("--worker") + 1])
        rows = [
            {
                "topology": topology,
                "cell_count": cell_count,
                "query_median_ns": 100,
                "tracemalloc_peak_bytes": 200,
                "all_invariants_pass": True,
                "mixed_membership_distinct": {
                    "evaluated": topology == "mixed_membership",
                    "sampled_descriptors": (
                        12 if topology == "mixed_membership" else 0
                    ),
                    "passed": (
                        mixed_membership_passed
                        if topology == "mixed_membership"
                        else None
                    ),
                },
            }
            for topology in topologies
        ]
        return benchmark.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(
                {
                    "cell_count": cell_count,
                    "implementation_source_bundle_sha256": source_sha,
                    "results": rows,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(benchmark.subprocess, "run", fake_worker)

    envelope = benchmark.run(artifact, confirmed_protocol_sha=protocol_sha)

    assert json.loads(artifact.read_bytes()) == envelope
    assert envelope["schema"] == "humanbrain.canonical-json-artifact.v1"
    assert envelope["kind"] == "procedural-address-space-evidence"
    assert envelope["payload_sha256"] == canonical_sha256(envelope["payload"])
    assert envelope["payload"]["benchmark_protocol_version"] == 2
    assert envelope["payload"]["preflight"]["small_materialized_reference"][
        "passed"
    ] is True
    assert envelope["payload"]["claim_evaluations"][
        "mixed-membership-distinct"
    ] == {
        "evaluated_topology_rows": 3,
        "sampled_descriptors": 36,
        "passed": mixed_membership_passed,
    }
    assert envelope["payload"]["gate_2_pass"] is mixed_membership_passed


def test_run_rejects_worker_source_attestation_mismatch_before_publication(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "evidence.json"
    protocol = json.loads(benchmark.PROTOCOL_PATH.read_bytes())
    protocol_sha = sha256(benchmark.PROTOCOL_PATH.read_bytes()).hexdigest()
    topologies = protocol["implementation_contract"]["topology_names"]

    def fake_check_output(args, **_: object) -> str:
        if args[:2] == ["git", "status"]:
            return ""
        if args[:2] == ["git", "rev-parse"]:
            return "a" * 40 + "\n"
        raise AssertionError(f"unexpected command: {args}")

    def unattested_worker(args, **_: object):
        cell_count = int(args[args.index("--worker") + 1])
        rows = [
            {
                "topology": topology,
                "cell_count": cell_count,
                "query_median_ns": 100,
                "tracemalloc_peak_bytes": 200,
                "all_invariants_pass": True,
            }
            for topology in topologies
        ]
        return benchmark.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(
                {
                    "cell_count": cell_count,
                    "implementation_source_bundle_sha256": "0" * 64,
                    "results": rows,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(benchmark.subprocess, "run", unattested_worker)

    with pytest.raises(RuntimeError, match="worker source bundle attestation"):
        benchmark.run(artifact, confirmed_protocol_sha=protocol_sha)

    assert not artifact.exists()


@pytest.mark.parametrize("drift_kind", ["dirty", "head", "source"])
def test_run_rechecks_the_starting_repository_snapshot_before_publication(
    tmp_path,
    monkeypatch,
    drift_kind: str,
) -> None:
    artifact = tmp_path / "evidence.json"
    protocol = json.loads(benchmark.PROTOCOL_PATH.read_bytes())
    protocol_sha = sha256(benchmark.PROTOCOL_PATH.read_bytes()).hexdigest()
    source_sha = protocol["implementation_sources"]["bundle_sha256"]
    topologies = protocol["implementation_contract"]["topology_names"]
    source_root = tmp_path / "source"
    for file_record in protocol["implementation_sources"]["files"]:
        relative_path = file_record["path"]
        destination = source_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((benchmark.ROOT / relative_path).read_bytes())
    monkeypatch.setattr(benchmark, "ROOT", source_root)

    status_calls = 0
    head_calls = 0

    def changing_check_output(args, **_: object) -> str:
        nonlocal status_calls, head_calls
        if args[:2] == ["git", "status"]:
            status_calls += 1
            if drift_kind == "dirty" and status_calls == 2:
                return " M src/humanbrain/procedural_topology.py\n"
            return ""
        if args[:2] == ["git", "rev-parse"]:
            head_calls += 1
            if drift_kind == "head" and head_calls == 2:
                return "b" * 40 + "\n"
            return "a" * 40 + "\n"
        raise AssertionError(f"unexpected command: {args}")

    worker_calls = 0

    def drifting_worker(args, **_: object):
        nonlocal worker_calls
        worker_calls += 1
        if drift_kind == "source" and worker_calls == 1:
            source_path = source_root / "src/humanbrain/procedural_topology.py"
            source_path.write_bytes(source_path.read_bytes() + b"\n# drift\n")
        cell_count = int(args[args.index("--worker") + 1])
        rows = [
            {
                "topology": topology,
                "cell_count": cell_count,
                "query_median_ns": 100,
                "tracemalloc_peak_bytes": 200,
                "all_invariants_pass": True,
                "mixed_membership_distinct": {
                    "evaluated": topology == "mixed_membership",
                    "sampled_descriptors": (
                        12 if topology == "mixed_membership" else 0
                    ),
                    "passed": True if topology == "mixed_membership" else None,
                },
            }
            for topology in topologies
        ]
        return benchmark.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(
                {
                    "cell_count": cell_count,
                    "implementation_source_bundle_sha256": source_sha,
                    "results": rows,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "check_output", changing_check_output)
    monkeypatch.setattr(benchmark.subprocess, "run", drifting_worker)

    with pytest.raises(RuntimeError):
        benchmark.run(artifact, confirmed_protocol_sha=protocol_sha)

    assert not artifact.exists()


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


def test_internal_worker_requires_the_confirmed_source_bundle(
    monkeypatch,
) -> None:
    monkeypatch.setattr(benchmark, "_worker", _unexpected_execution)
    protocol_sha = sha256(benchmark.PROTOCOL_PATH.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="source bundle SHA-256 confirmation"):
        benchmark.main(
            [
                "--worker",
                "1000000",
                "--confirm-protocol-sha",
                protocol_sha,
                "--confirm-source-sha",
                "0" * 64,
            ]
        )
