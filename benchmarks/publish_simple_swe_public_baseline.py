from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_SCHEMA = "human-cell-simple-swe-public-development-baseline-result-v1"
EXPECTED_STATUS = "noncanonical-public-development-only"
PUBLISHED_SCHEMA = (
    "human-cell-simple-swe-public-development-baseline-publication-v1"
)
RECORD_SCHEMA = (
    "human-cell-simple-swe-public-development-baseline-publication-record-v1"
)
TRANSFORM_ID = "portable-public-evidence-v1"
CHECKPOINT_PATH = (
    ".runtime/simple-swe-public-baseline-v1/selected_prompt_only_cells.npz"
)
EXPERIMENT_PROTOCOL_PATH = (
    "benchmarks/simple_swe_public_baseline_protocol_v1.json"
)
TASK_PROTOCOL_PATH = "benchmarks/simple_swe_protocol_v1.json"
WORKER_PATH = "src/humanbrain/simple_swe/_worker.py"
RUNNER_PATH = "benchmarks/simple_swe_public_baseline.py"
CHANGED_FIELDS = (
    "checkpoint.path",
    "command.argv",
    "command.cwd",
    "command.portable_command",
    "command.windows_command_line",
    "experiment_protocol.path",
    "publication",
    "runtime.executable",
    "runtime.executable_basename",
    "runtime.worker_path",
    "schema",
    "task_protocol.path",
)
ABSOLUTE_WORKSTATION_PATH = re.compile(
    rb"(?:[A-Za-z]:[\\/]|/(?:Users|home)/)", re.IGNORECASE
)


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _portable_payload(value: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(value)
    payload.pop("schema", None)
    payload.pop("publication", None)
    payload.pop("command", None)
    payload["checkpoint"].pop("path", None)
    payload["experiment_protocol"].pop("path", None)
    payload["task_protocol"].pop("path", None)
    runtime = payload["runtime"]
    runtime.pop("executable", None)
    runtime.pop("executable_basename", None)
    runtime.pop("worker_path", None)
    return payload


def _portable_payload_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        _portable_payload(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _basename(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _require_path_suffix(value: object, suffix: str, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a path string")
    normalized = value.replace("\\", "/")
    if not normalized.endswith(suffix):
        raise ValueError(f"{field} does not identify {suffix}")


def publish_evidence(
    *,
    raw_path: Path,
    published_path: Path,
    record_path: Path,
    archive_raw_path: Path,
    expected_raw_sha256: str,
) -> tuple[str, str]:
    raw_bytes = raw_path.read_bytes()
    raw_sha256 = _sha256_bytes(raw_bytes)
    if raw_sha256 != expected_raw_sha256:
        raise ValueError("raw artifact SHA-256 differs from the frozen input")
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or raw.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("raw artifact schema is unsupported")
    if raw.get("status") != EXPECTED_STATUS:
        raise ValueError("raw artifact is not public-development evidence")

    published: dict[str, Any] = deepcopy(raw)
    _require_path_suffix(
        published["checkpoint"]["path"], CHECKPOINT_PATH, "checkpoint.path"
    )
    published["checkpoint"]["path"] = CHECKPOINT_PATH
    _require_path_suffix(
        published["experiment_protocol"]["path"],
        EXPERIMENT_PROTOCOL_PATH,
        "experiment_protocol.path",
    )
    published["experiment_protocol"]["path"] = EXPERIMENT_PROTOCOL_PATH
    _require_path_suffix(
        published["task_protocol"]["path"], TASK_PROTOCOL_PATH, "task_protocol.path"
    )
    published["task_protocol"]["path"] = TASK_PROTOCOL_PATH

    runtime = published["runtime"]
    _require_path_suffix(runtime["worker_path"], WORKER_PATH, "runtime.worker_path")
    runtime["worker_path"] = WORKER_PATH
    runtime["executable_basename"] = _basename(
        runtime.pop("executable"), "runtime.executable"
    )

    command = published["command"]
    argv = command.get("argv")
    if not isinstance(argv, list) or len(argv) != 2:
        raise ValueError("recorded command must contain Python and the runner")
    _require_path_suffix(argv[1], RUNNER_PATH, "command.argv[1]")
    command.clear()
    command.update(
        {
            "argv": ["python", RUNNER_PATH],
            "cwd": ".",
            "portable_command": f"python {RUNNER_PATH}",
        }
    )

    raw_payload_bytes = _portable_payload_bytes(raw)
    published_payload_bytes = _portable_payload_bytes(published)
    if published_payload_bytes != raw_payload_bytes:
        raise ValueError("publication changed a metric, result, or portable identity")
    portable_payload_sha256 = _sha256_bytes(raw_payload_bytes)

    publisher_sha256 = _sha256_file(Path(__file__).resolve())
    published["schema"] = PUBLISHED_SCHEMA
    published["publication"] = {
        "changed_fields": list(CHANGED_FIELDS),
        "metrics_and_results_unchanged": True,
        "portable_payload_sha256": portable_payload_sha256,
        "publisher_sha256": publisher_sha256,
        "raw_artifact_sha256": raw_sha256,
        "timings": "measured and environment-dependent; values preserved verbatim",
        "transform_id": TRANSFORM_ID,
    }
    published_bytes = _json_bytes(published)
    if ABSOLUTE_WORKSTATION_PATH.search(published_bytes):
        raise ValueError("published evidence still contains an absolute workstation path")
    published_sha256 = _sha256_bytes(published_bytes)
    record = {
        "schema": RECORD_SCHEMA,
        "transform_id": TRANSFORM_ID,
        "raw_artifact_sha256": raw_sha256,
        "raw_archive_sha256": raw_sha256,
        "published_artifact_sha256": published_sha256,
        "publisher_sha256": publisher_sha256,
        "changed_fields": list(CHANGED_FIELDS),
        "metrics_and_results_unchanged": True,
        "portable_payload_sha256": portable_payload_sha256,
        "timings": "measured and environment-dependent; values preserved verbatim",
    }
    record_bytes = _json_bytes(record)
    if ABSOLUTE_WORKSTATION_PATH.search(record_bytes):
        raise ValueError("publication record contains an absolute workstation path")

    archive_raw_path.parent.mkdir(parents=True, exist_ok=True)
    published_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    archive_raw_path.write_bytes(raw_bytes)
    published_path.write_bytes(published_bytes)
    record_path.write_bytes(record_bytes)
    return published_sha256, raw_sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish portable evidence from a frozen raw public baseline result."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--archive-raw", type=Path, required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    args = parser.parse_args()
    published_sha256, raw_sha256 = publish_evidence(
        raw_path=args.raw.resolve(),
        published_path=args.published.resolve(),
        record_path=args.record.resolve(),
        archive_raw_path=args.archive_raw.resolve(),
        expected_raw_sha256=args.expected_raw_sha256,
    )
    print(
        json.dumps(
            {
                "published_artifact_sha256": published_sha256,
                "raw_artifact_sha256": raw_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
