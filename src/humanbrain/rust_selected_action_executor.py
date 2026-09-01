"""Compile and run exactly one materialized Rust action.

The module is a trusted-host adapter around a concrete ``rustc`` executable.  A
successful result commits to the selected source, the fixed harness, the exact
compiler executable, rustc dep-info, the deterministic batch bytes, and the
bounded child-process output.  The resource controls are containment for
audited experiment code; they are not a hostile-code security sandbox.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from secrets import token_bytes
from threading import Event, Lock, Thread
from time import monotonic
from typing import Protocol, TypeVar

from experiments.r3_pilot import common_knowledge_action_memory_v1 as action_memory

from humanbrain.rust_action_registry import (
    PreparedRustActionRegistryV1,
    RustActionMaterializationV1,
    RustActionRegistryError,
)

SCHEMA = "humanbrain.rust-selected-action-executor.v1"
CLAIM_BOUNDARY = (
    "Trusted-host compilation and resource-bounded execution of exactly one "
    "authenticated selected Rust source with a fixed harness; not hostile-code "
    "sandboxing, filesystem or network isolation, semantic understanding, or "
    "coding-transfer evidence."
)
V2_SCHEMA = "humanbrain.rust-selected-source-executor.v2"
V2_CLAIM_BOUNDARY = (
    "Trusted-host compilation and resource-bounded execution of exactly one "
    "source released by an authenticated selected-source authority; not source "
    "selection, hostile-code sandboxing, semantic understanding, or coding-"
    "transfer evidence."
)

RUST_EDITION = "2021"
HARNESS_FILENAME = "harness.rs"
SELECTED_FILENAME = "selected.rs"
DEP_INFO_FILENAME = "selected_action.d"
CRATE_NAME = "humanbrain_selected_action"

MAX_SOURCE_BYTES = 64 * 1024
MAX_CASES = 64
MAX_VALUES_PER_CASE = 4_096
MAX_TOTAL_INPUT_VALUES = 16_384
MAX_VALUES_PER_OUTPUT = 16_384
MAX_TOTAL_OUTPUT_VALUES = 65_536
MAX_DEP_INFO_BYTES = 64 * 1024
MAX_BINARY_BYTES = 16 * 1024 * 1024
MAX_TOOLCHAIN_FILES = 2_048
MAX_TOOLCHAIN_FILE_BYTES = 512 * 1024 * 1024
MAX_TOOLCHAIN_DIRECTORY_BYTES = 2 * 1024 * 1024 * 1024
PROCESS_CLEANUP_SECONDS = 5.0

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EMPTY_SHA256 = sha256(b"").hexdigest()

_IDENTITY_NAMESPACE = b"rust-selected-action-rustc-identity-v1"
_DIRECTORY_MANIFEST_NAMESPACE = b"rust-selected-action-directory-manifest-v1"
_LIMITS_NAMESPACE = b"rust-selected-action-process-limits-v1"
_ENVIRONMENT_NAMESPACE = b"rust-selected-action-environment-v1"
_POLICY_NAMESPACE = b"rust-selected-action-policy-v1"
_PREPARATION_NAMESPACE = b"rust-selected-action-preparation-v1"
_DEPENDENCY_NAMESPACE = b"rust-selected-action-dependency-receipt-v1"
_COMPILE_NAMESPACE = b"rust-selected-action-compile-receipt-v1"
_RUN_NAMESPACE = b"rust-selected-action-run-receipt-v1"
_EXECUTION_NAMESPACE = b"rust-selected-action-execution-v1"
_BATCH_NAMESPACE = b"rust-selected-action-batch-v1"
_PROTOCOL_NAMESPACE = b"rust-selected-action-batch-protocol-v1"
_V2_SELECTED_SOURCE_NAMESPACE = b"rust-selected-source-grant-v2"
_V2_REGISTRY_AUTHORITY_NAMESPACE = b"rust-action-registry-authority-v2"
_V2_POLICY_NAMESPACE = b"rust-selected-source-policy-v2"
_V2_PREPARATION_NAMESPACE = b"rust-selected-source-preparation-v2"
_V2_DEPENDENCY_NAMESPACE = b"rust-selected-source-dependency-receipt-v2"
_V2_COMPILE_NAMESPACE = b"rust-selected-source-compile-receipt-v2"
_V2_RUN_NAMESPACE = b"rust-selected-source-run-receipt-v2"
_V2_EXECUTION_NAMESPACE = b"rust-selected-source-execution-v2"
_V2_BATCH_NAMESPACE = b"rust-selected-source-batch-v2"
_V2_PROTOCOL_NAMESPACE = b"rust-selected-source-batch-protocol-v2"
_V2_ACTION_HANDLE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,127}")

_HARNESS_BYTES = b"""#[path = \"selected.rs\"]
mod selected;

use std::fmt::Write as _;
use std::io::{self, Read};

fn next_usize<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> usize {
    tokens.next().expect("missing unsigned integer").parse().expect("invalid unsigned integer")
}

fn next_i64<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> i64 {
    tokens.next().expect("missing signed integer").parse().expect("invalid signed integer")
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("stdin read failed");
    let mut tokens = input.split_ascii_whitespace();
    let case_count = next_usize(&mut tokens);
    let mut rendered = String::new();
    writeln!(&mut rendered, "{case_count}").expect("render failed");
    for _ in 0..case_count {
        let value_count = next_usize(&mut tokens);
        let mut values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            values.push(next_i64(&mut tokens));
        }
        let output = selected::solve(&values);
        write!(&mut rendered, "{}", output.len()).expect("render failed");
        for value in output {
            write!(&mut rendered, " {value}").expect("render failed");
        }
        rendered.push('\\n');
    }
    assert!(tokens.next().is_none(), "unexpected trailing input");
    print!("{rendered}");
}
"""
HARNESS_SHA256 = sha256(_HARNESS_BYTES).hexdigest()
BATCH_PROTOCOL_SHA256 = sha256(
    _PROTOCOL_NAMESPACE
    + b"\0"
    + b"ascii-lf;first-line=case-count;case-line=value-count values;i64-decimal"
).hexdigest()
V2_BATCH_PROTOCOL_SHA256 = sha256(
    _V2_PROTOCOL_NAMESPACE
    + b"\0"
    + b"ascii-lf;first-line=case-count;case-line=value-count values;i64-decimal"
).hexdigest()


class RustSelectedActionExecutionError(ValueError):
    """Selected-source authentication, compilation, or execution failed."""


class _RustSelectedSourceView(Protocol):
    sha256: str
    source_sha256: str
    source: str


class _RustOperationalPreparationView(Protocol):
    compile_environment: tuple[tuple[str, str], ...]
    run_environment: tuple[tuple[str, str], ...]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _binary_filename() -> str:
    return f"{CRATE_NAME}.exe" if os.name == "nt" else CRATE_NAME


_TEMPORARY_ROOT_PLACEHOLDER = "<temporary-root>"
_CONSTRUCTION_TOKEN = object()
_T = TypeVar("_T")


def _discovery_environment(temporary_root: Path) -> dict[str, str]:
    environment = {
        "RUST_BACKTRACE": "0",
        "TEMP": str(temporary_root),
        "TMP": str(temporary_root),
    }
    for name in (
        "SYSTEMROOT",
        "WINDIR",
        "PATH",
        "PATHEXT",
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "LANG",
        "LC_ALL",
    ):
        if value := os.environ.get(name):
            environment[name] = value
    return environment


def _frozen_environments(
    rustc: RustcIdentityV1,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    shared = {
        "RUST_BACKTRACE": "0",
        "TEMP": _TEMPORARY_ROOT_PLACEHOLDER,
        "TMP": _TEMPORARY_ROOT_PLACEHOLDER,
    }
    for name in ("SYSTEMROOT", "WINDIR"):
        if value := os.environ.get(name):
            shared[name] = value
    run_environment = tuple(sorted(shared.items()))
    compile_values = dict(shared)
    if rustc.link_search_paths:
        compile_values["LIB"] = os.pathsep.join(rustc.link_search_paths)
    compile_environment = tuple(sorted(compile_values.items()))
    return compile_environment, run_environment


def _environment_sha256(kind: str, environment: tuple[tuple[str, str], ...]) -> str:
    return _digest(
        _ENVIRONMENT_NAMESPACE,
        {"kind": kind, "environment": environment},
    )


def _materialize_environment(
    environment: tuple[tuple[str, str], ...],
    temporary_root: Path,
) -> dict[str, str]:
    root = str(temporary_root)
    return {
        name: root if value == _TEMPORARY_ROOT_PLACEHOLDER else value for name, value in environment
    }


@dataclass(frozen=True, slots=True)
class RustcIdentityV1:
    path: str
    version_verbose: str
    host_triple: str
    executable_sha256: str
    linker_path: str
    linker_executable_sha256: str
    target_libdir_path: str
    target_libdir_manifest_sha256: str
    target_libdir_file_count: int
    target_libdir_total_bytes: int
    link_search_paths: tuple[str, ...]
    link_search_manifest_sha256s: tuple[str, ...]
    link_search_file_counts: tuple[int, ...]
    link_search_total_bytes: tuple[int, ...]
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        if (
            type(self.path) is not str
            or not Path(self.path).is_absolute()
            or type(self.version_verbose) is not str
            or not self.version_verbose.startswith("rustc ")
            or type(self.host_triple) is not str
            or not self.host_triple
            or not _valid_sha256(self.executable_sha256)
            or type(self.linker_path) is not str
            or not Path(self.linker_path).is_absolute()
            or not _valid_sha256(self.linker_executable_sha256)
            or type(self.target_libdir_path) is not str
            or not Path(self.target_libdir_path).is_absolute()
            or not _valid_sha256(self.target_libdir_manifest_sha256)
            or type(self.target_libdir_file_count) is not int
            or self.target_libdir_file_count < 1
            or type(self.target_libdir_total_bytes) is not int
            or self.target_libdir_total_bytes < 1
            or type(self.link_search_paths) is not tuple
            or any(
                type(value) is not str or not Path(value).is_absolute()
                for value in self.link_search_paths
            )
            or (os.name == "nt" and len(self.link_search_paths) != 3)
            or type(self.link_search_manifest_sha256s) is not tuple
            or len(self.link_search_manifest_sha256s) != len(self.link_search_paths)
            or any(not _valid_sha256(value) for value in self.link_search_manifest_sha256s)
            or type(self.link_search_file_counts) is not tuple
            or len(self.link_search_file_counts) != len(self.link_search_paths)
            or any(type(value) is not int or value < 1 for value in self.link_search_file_counts)
            or type(self.link_search_total_bytes) is not tuple
            or len(self.link_search_total_bytes) != len(self.link_search_paths)
            or any(type(value) is not int or value < 1 for value in self.link_search_total_bytes)
            or self.sha256 != _digest(_IDENTITY_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError("rustc identity authentication failed")


@dataclass(frozen=True, slots=True)
class RustProcessLimitsV1:
    wall_seconds: float
    cpu_seconds: int
    memory_bytes: int
    active_processes: int
    output_bytes: int
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        if (
            type(self.wall_seconds) is not float
            or not 0.0 < self.wall_seconds <= 120.0
            or type(self.cpu_seconds) is not int
            or not 0 < self.cpu_seconds <= 120
            or type(self.memory_bytes) is not int
            or not 16 * 1024 * 1024 <= self.memory_bytes <= 2 * 1024 * 1024 * 1024
            or type(self.active_processes) is not int
            or not 1 <= self.active_processes <= 16
            or type(self.output_bytes) is not int
            or not 1_024 <= self.output_bytes <= 1024 * 1024
            or self.sha256 != _digest(_LIMITS_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError("process limits authentication failed")


def _make_limits(
    *,
    wall_seconds: float,
    cpu_seconds: int,
    memory_bytes: int,
    active_processes: int,
    output_bytes: int,
) -> RustProcessLimitsV1:
    payload = {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "memory_bytes": memory_bytes,
        "active_processes": active_processes,
        "output_bytes": output_bytes,
    }
    limits = RustProcessLimitsV1(
        **payload,
        sha256=_digest(_LIMITS_NAMESPACE, payload),
    )
    limits.authenticate_structure()
    return limits


COMPILE_LIMITS = _make_limits(
    wall_seconds=45.0,
    cpu_seconds=30,
    memory_bytes=1024 * 1024 * 1024,
    active_processes=8,
    output_bytes=256 * 1024,
)
RUN_LIMITS = _make_limits(
    wall_seconds=2.0,
    cpu_seconds=1,
    memory_bytes=256 * 1024 * 1024,
    active_processes=1,
    output_bytes=64 * 1024,
)


def _compile_arguments() -> tuple[str, ...]:
    return (
        "--edition",
        RUST_EDITION,
        "--crate-name",
        CRATE_NAME,
        "--crate-type",
        "bin",
        "--color",
        "never",
        "-C",
        "debuginfo=0",
        "-C",
        "linker=<rooted-rust-lld>",
        "--emit",
        f"link={_binary_filename()},dep-info={DEP_INFO_FILENAME}",
        HARNESS_FILENAME,
    )


def _actual_compile_arguments(rustc: RustcIdentityV1) -> tuple[str, ...]:
    return tuple(
        f"linker={rustc.linker_path}" if value == "linker=<rooted-rust-lld>" else value
        for value in _compile_arguments()
    )


COMPILER_POLICY_SHA256 = _digest(
    _POLICY_NAMESPACE,
    {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "edition": RUST_EDITION,
        "harness_filename": HARNESS_FILENAME,
        "harness_sha256": HARNESS_SHA256,
        "selected_filename": SELECTED_FILENAME,
        "dep_info_filename": DEP_INFO_FILENAME,
        "binary_filename": _binary_filename(),
        "compile_arguments": _compile_arguments(),
        "compile_limits_sha256": COMPILE_LIMITS.sha256,
        "run_limits_sha256": RUN_LIMITS.sha256,
        "batch_protocol_sha256": BATCH_PROTOCOL_SHA256,
    },
)

V2_COMPILER_POLICY_SHA256 = _digest(
    _V2_POLICY_NAMESPACE,
    {
        "schema": V2_SCHEMA,
        "claim_boundary": V2_CLAIM_BOUNDARY,
        "edition": RUST_EDITION,
        "harness_filename": HARNESS_FILENAME,
        "harness_sha256": HARNESS_SHA256,
        "selected_filename": SELECTED_FILENAME,
        "dep_info_filename": DEP_INFO_FILENAME,
        "binary_filename": _binary_filename(),
        "compile_arguments": _compile_arguments(),
        "compile_limits_sha256": COMPILE_LIMITS.sha256,
        "run_limits_sha256": RUN_LIMITS.sha256,
        "batch_protocol_sha256": V2_BATCH_PROTOCOL_SHA256,
    },
)


@dataclass(frozen=True, slots=True)
class RustExecutorPreparationV1:
    schema: str
    claim_boundary: str
    compiler_policy_sha256: str
    rustc_identity_sha256: str
    expected_registry_sha256: str
    harness_sha256: str
    batch_protocol_sha256: str
    batch_sha256: str
    stdin_sha256: str
    compile_environment: tuple[tuple[str, str], ...]
    compile_environment_sha256: str
    run_environment: tuple[tuple[str, str], ...]
    run_environment_sha256: str
    case_count: int
    compile_limits_sha256: str
    run_limits_sha256: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        environments = (self.compile_environment, self.run_environment)
        if any(
            type(environment) is not tuple
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not str
                for pair in environment
            )
            for environment in environments
        ):
            raise RustSelectedActionExecutionError("executor preparation authentication failed")
        if any(
            len({name for name, _value in environment}) != len(environment)
            for environment in environments
        ):
            raise RustSelectedActionExecutionError("executor preparation authentication failed")
        roots = (
            self.rustc_identity_sha256,
            self.expected_registry_sha256,
            self.harness_sha256,
            self.batch_protocol_sha256,
            self.batch_sha256,
            self.stdin_sha256,
            self.compile_environment_sha256,
            self.run_environment_sha256,
            self.compile_limits_sha256,
            self.run_limits_sha256,
        )
        required_environment = {
            "RUST_BACKTRACE": "0",
            "TEMP": _TEMPORARY_ROOT_PLACEHOLDER,
            "TMP": _TEMPORARY_ROOT_PLACEHOLDER,
        }
        compile_values = dict(self.compile_environment)
        run_values = dict(self.run_environment)
        allowed_run_keys = {*required_environment, "SYSTEMROOT", "WINDIR"}
        expected_compile_keys = set(run_values)
        if os.name == "nt":
            expected_compile_keys.add("LIB")
        if (
            self.schema != SCHEMA
            or self.claim_boundary != CLAIM_BOUNDARY
            or self.compiler_policy_sha256 != COMPILER_POLICY_SHA256
            or any(not _valid_sha256(value) for value in roots)
            or self.harness_sha256 != HARNESS_SHA256
            or self.batch_protocol_sha256 != BATCH_PROTOCOL_SHA256
            or self.compile_environment != tuple(sorted(set(self.compile_environment)))
            or self.run_environment != tuple(sorted(set(self.run_environment)))
            or not required_environment.items() <= run_values.items()
            or not set(run_values) <= allowed_run_keys
            or set(compile_values) != expected_compile_keys
            or any(compile_values[name] != value for name, value in run_values.items())
            or (os.name == "nt" and not compile_values.get("LIB"))
            or self.compile_environment_sha256
            != _environment_sha256("compile", self.compile_environment)
            or self.run_environment_sha256 != _environment_sha256("run", self.run_environment)
            or type(self.case_count) is not int
            or not 1 <= self.case_count <= MAX_CASES
            or self.compile_limits_sha256 != COMPILE_LIMITS.sha256
            or self.run_limits_sha256 != RUN_LIMITS.sha256
            or self.sha256 != _digest(_PREPARATION_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError("executor preparation authentication failed")


@dataclass(frozen=True, slots=True)
class RustDependencyReceiptV1:
    schema: str
    materialization_sha256: str
    harness_sha256: str
    selected_source_sha256: str
    dep_info_sha256: str
    dependency_paths: tuple[str, str]
    dependency_content_sha256s: tuple[tuple[str, str], tuple[str, str]]
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        expected_paths = (HARNESS_FILENAME, SELECTED_FILENAME)
        expected_content = (
            (HARNESS_FILENAME, HARNESS_SHA256),
            (SELECTED_FILENAME, self.selected_source_sha256),
        )
        if (
            self.schema != SCHEMA
            or not _valid_sha256(self.materialization_sha256)
            or self.harness_sha256 != HARNESS_SHA256
            or not _valid_sha256(self.selected_source_sha256)
            or not _valid_sha256(self.dep_info_sha256)
            or self.dependency_paths != expected_paths
            or self.dependency_content_sha256s != expected_content
            or self.sha256 != _digest(_DEPENDENCY_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError("dependency receipt authentication failed")


@dataclass(frozen=True, slots=True)
class RustCompileReceiptV1:
    schema: str
    compiler_policy_sha256: str
    rustc_identity_sha256: str
    preparation_sha256: str
    materialization_sha256: str
    selected_source_sha256: str
    harness_sha256: str
    edition: str
    arguments: tuple[str, ...]
    compile_limits_sha256: str
    compile_environment_sha256: str
    compile_invocations: int
    dependency_receipt_sha256: str
    binary_sha256: str
    binary_size_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        roots = (
            self.rustc_identity_sha256,
            self.preparation_sha256,
            self.materialization_sha256,
            self.selected_source_sha256,
            self.harness_sha256,
            self.compile_limits_sha256,
            self.compile_environment_sha256,
            self.dependency_receipt_sha256,
            self.binary_sha256,
            self.stdout_sha256,
            self.stderr_sha256,
        )
        if (
            self.schema != SCHEMA
            or self.compiler_policy_sha256 != COMPILER_POLICY_SHA256
            or any(not _valid_sha256(value) for value in roots)
            or self.harness_sha256 != HARNESS_SHA256
            or self.edition != RUST_EDITION
            or self.arguments != _compile_arguments()
            or self.compile_limits_sha256 != COMPILE_LIMITS.sha256
            or type(self.compile_invocations) is not int
            or self.compile_invocations != 1
            or type(self.binary_size_bytes) is not int
            or not 1 <= self.binary_size_bytes <= MAX_BINARY_BYTES
            or self.stdout_sha256 != _EMPTY_SHA256
            or self.stderr_sha256 != _EMPTY_SHA256
            or self.sha256 != _digest(_COMPILE_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError("compile receipt authentication failed")


@dataclass(frozen=True, slots=True)
class RustRunReceiptV1:
    schema: str
    compiler_policy_sha256: str
    preparation_sha256: str
    compile_receipt_sha256: str
    materialization_sha256: str
    batch_protocol_sha256: str
    batch_sha256: str
    stdin_sha256: str
    stdout_sha256: str
    stderr_sha256: str
    output_sha256: str
    run_limits_sha256: str
    run_environment_sha256: str
    run_invocations: int
    case_count: int
    returncode: int
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        roots = (
            self.compile_receipt_sha256,
            self.preparation_sha256,
            self.materialization_sha256,
            self.batch_protocol_sha256,
            self.batch_sha256,
            self.stdin_sha256,
            self.stdout_sha256,
            self.stderr_sha256,
            self.output_sha256,
            self.run_limits_sha256,
            self.run_environment_sha256,
        )
        if (
            self.schema != SCHEMA
            or self.compiler_policy_sha256 != COMPILER_POLICY_SHA256
            or any(not _valid_sha256(value) for value in roots)
            or self.batch_protocol_sha256 != BATCH_PROTOCOL_SHA256
            or self.run_limits_sha256 != RUN_LIMITS.sha256
            or type(self.run_invocations) is not int
            or self.run_invocations != 1
            or type(self.case_count) is not int
            or not 1 <= self.case_count <= MAX_CASES
            or type(self.returncode) is not int
            or self.returncode != 0
            or self.stderr_sha256 != _EMPTY_SHA256
            or self.sha256 != _digest(_RUN_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError("run receipt authentication failed")


@dataclass(frozen=True, slots=True)
class RustSelectedActionExecutionV1:
    materialization: RustActionMaterializationV1
    rustc_identity: RustcIdentityV1
    preparation: RustExecutorPreparationV1
    dependency_receipt: RustDependencyReceiptV1
    compile_receipt: RustCompileReceiptV1
    run_receipt: RustRunReceiptV1
    outputs: tuple[tuple[int, ...], ...]
    sha256: str

    def authenticate_structure(self) -> None:
        if type(self.materialization) is not RustActionMaterializationV1:
            raise RustSelectedActionExecutionError("execution materialization changed")
        try:
            self.materialization.authenticate_structure()
        except RustActionRegistryError as exc:
            raise RustSelectedActionExecutionError(
                "execution materialization authentication failed"
            ) from exc
        self.rustc_identity.authenticate_structure()
        self.preparation.authenticate_structure()
        self.dependency_receipt.authenticate_structure()
        self.compile_receipt.authenticate_structure()
        self.run_receipt.authenticate_structure()
        outputs = _validate_outputs(self.outputs, expected_cases=self.run_receipt.case_count)
        output_sha256 = _digest(_BATCH_NAMESPACE, outputs)
        payload = {
            "materialization_sha256": self.materialization.sha256,
            "rustc_identity_sha256": self.rustc_identity.sha256,
            "preparation_sha256": self.preparation.sha256,
            "dependency_receipt_sha256": self.dependency_receipt.sha256,
            "compile_receipt_sha256": self.compile_receipt.sha256,
            "run_receipt_sha256": self.run_receipt.sha256,
            "outputs": outputs,
        }
        if (
            self.dependency_receipt.materialization_sha256 != self.materialization.sha256
            or self.preparation.rustc_identity_sha256 != self.rustc_identity.sha256
            or self.dependency_receipt.selected_source_sha256 != self.materialization.source_sha256
            or self.compile_receipt.rustc_identity_sha256 != self.rustc_identity.sha256
            or self.compile_receipt.preparation_sha256 != self.preparation.sha256
            or self.compile_receipt.compile_environment_sha256
            != self.preparation.compile_environment_sha256
            or self.compile_receipt.materialization_sha256 != self.materialization.sha256
            or self.compile_receipt.selected_source_sha256 != self.materialization.source_sha256
            or self.compile_receipt.dependency_receipt_sha256 != self.dependency_receipt.sha256
            or self.run_receipt.compile_receipt_sha256 != self.compile_receipt.sha256
            or self.run_receipt.preparation_sha256 != self.preparation.sha256
            or self.run_receipt.run_environment_sha256 != self.preparation.run_environment_sha256
            or self.run_receipt.materialization_sha256 != self.materialization.sha256
            or self.run_receipt.output_sha256 != output_sha256
            or self.run_receipt.case_count != self.preparation.case_count
            or len(outputs) != self.preparation.case_count
            or self.run_receipt.stdout_sha256 != sha256(_encode_batch(outputs)).hexdigest()
            or self.run_receipt.stderr_sha256 != _EMPTY_SHA256
            or self.preparation.expected_registry_sha256 != self.materialization.registry_sha256
            or self.preparation.batch_sha256 != self.run_receipt.batch_sha256
            or self.preparation.stdin_sha256 != self.run_receipt.stdin_sha256
            or self.sha256 != _digest(_EXECUTION_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError("execution receipt authentication failed")


@dataclass(frozen=True, slots=True)
class RustSelectedSourceV2:
    """Authority-neutral one-shot grant for exact Rust source bytes."""

    authority_sha256: str
    selection_sha256: str
    release_sha256: str
    action_handle: str
    source_sha256: str
    source: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        try:
            source_bytes = self.source.encode("ascii", errors="strict")
        except (AttributeError, UnicodeEncodeError) as exc:
            raise RustSelectedActionExecutionError(
                "selected-source grant authentication failed"
            ) from exc
        if (
            not _valid_sha256(self.authority_sha256)
            or not _valid_sha256(self.selection_sha256)
            or not _valid_sha256(self.release_sha256)
            or type(self.action_handle) is not str
            or _V2_ACTION_HANDLE.fullmatch(self.action_handle) is None
            or not source_bytes
            or len(source_bytes) > MAX_SOURCE_BYTES
            or self.source_sha256 != sha256(source_bytes).hexdigest()
            or self.sha256 != _digest(_V2_SELECTED_SOURCE_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError("selected-source grant authentication failed")


def make_rust_selected_source_v2(
    *,
    authority_sha256: str,
    selection_sha256: str,
    release_sha256: str,
    action_handle: str,
    source: str,
) -> RustSelectedSourceV2:
    """Seal exact authority-released Rust bytes into a V2 source grant."""

    try:
        source_sha256 = sha256(source.encode("ascii", errors="strict")).hexdigest()
    except (AttributeError, UnicodeEncodeError) as exc:
        raise RustSelectedActionExecutionError("selected Rust source is not ASCII") from exc
    payload = {
        "authority_sha256": authority_sha256,
        "selection_sha256": selection_sha256,
        "release_sha256": release_sha256,
        "action_handle": action_handle,
        "source_sha256": source_sha256,
        "source": source,
    }
    result = RustSelectedSourceV2(
        **payload,
        sha256=_digest(_V2_SELECTED_SOURCE_NAMESPACE, payload),
    )
    result.authenticate_structure()
    return result


@dataclass(frozen=True, slots=True)
class RustSelectedSourcePreparationV2:
    schema: str
    claim_boundary: str
    compiler_policy_sha256: str
    rustc_identity_sha256: str
    expected_authority_sha256: str
    harness_sha256: str
    batch_protocol_sha256: str
    batch_sha256: str
    stdin_sha256: str
    compile_environment: tuple[tuple[str, str], ...]
    compile_environment_sha256: str
    run_environment: tuple[tuple[str, str], ...]
    run_environment_sha256: str
    case_count: int
    compile_limits_sha256: str
    run_limits_sha256: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        environments = (self.compile_environment, self.run_environment)
        if any(
            type(environment) is not tuple
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not str
                for pair in environment
            )
            for environment in environments
        ):
            raise RustSelectedActionExecutionError(
                "selected-source preparation authentication failed"
            )
        if any(
            len({name for name, _value in environment}) != len(environment)
            for environment in environments
        ):
            raise RustSelectedActionExecutionError(
                "selected-source preparation authentication failed"
            )
        roots = (
            self.rustc_identity_sha256,
            self.expected_authority_sha256,
            self.harness_sha256,
            self.batch_protocol_sha256,
            self.batch_sha256,
            self.stdin_sha256,
            self.compile_environment_sha256,
            self.run_environment_sha256,
            self.compile_limits_sha256,
            self.run_limits_sha256,
        )
        required_environment = {
            "RUST_BACKTRACE": "0",
            "TEMP": _TEMPORARY_ROOT_PLACEHOLDER,
            "TMP": _TEMPORARY_ROOT_PLACEHOLDER,
        }
        compile_values = dict(self.compile_environment)
        run_values = dict(self.run_environment)
        allowed_run_keys = {*required_environment, "SYSTEMROOT", "WINDIR"}
        expected_compile_keys = set(run_values)
        if os.name == "nt":
            expected_compile_keys.add("LIB")
        if (
            self.schema != V2_SCHEMA
            or self.claim_boundary != V2_CLAIM_BOUNDARY
            or self.compiler_policy_sha256 != V2_COMPILER_POLICY_SHA256
            or any(not _valid_sha256(value) for value in roots)
            or self.harness_sha256 != HARNESS_SHA256
            or self.batch_protocol_sha256 != V2_BATCH_PROTOCOL_SHA256
            or self.compile_environment != tuple(sorted(set(self.compile_environment)))
            or self.run_environment != tuple(sorted(set(self.run_environment)))
            or not required_environment.items() <= run_values.items()
            or not set(run_values) <= allowed_run_keys
            or set(compile_values) != expected_compile_keys
            or any(compile_values[name] != value for name, value in run_values.items())
            or (os.name == "nt" and not compile_values.get("LIB"))
            or self.compile_environment_sha256
            != _environment_sha256("compile", self.compile_environment)
            or self.run_environment_sha256 != _environment_sha256("run", self.run_environment)
            or type(self.case_count) is not int
            or not 1 <= self.case_count <= MAX_CASES
            or self.compile_limits_sha256 != COMPILE_LIMITS.sha256
            or self.run_limits_sha256 != RUN_LIMITS.sha256
            or self.sha256 != _digest(_V2_PREPARATION_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError(
                "selected-source preparation authentication failed"
            )


@dataclass(frozen=True, slots=True)
class RustSelectedSourceDependencyReceiptV2:
    schema: str
    selected_source_grant_sha256: str
    harness_sha256: str
    selected_source_sha256: str
    dep_info_sha256: str
    dependency_paths: tuple[str, str]
    dependency_content_sha256s: tuple[tuple[str, str], tuple[str, str]]
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        expected_paths = (HARNESS_FILENAME, SELECTED_FILENAME)
        expected_content = (
            (HARNESS_FILENAME, HARNESS_SHA256),
            (SELECTED_FILENAME, self.selected_source_sha256),
        )
        if (
            self.schema != V2_SCHEMA
            or not _valid_sha256(self.selected_source_grant_sha256)
            or self.harness_sha256 != HARNESS_SHA256
            or not _valid_sha256(self.selected_source_sha256)
            or not _valid_sha256(self.dep_info_sha256)
            or self.dependency_paths != expected_paths
            or self.dependency_content_sha256s != expected_content
            or self.sha256 != _digest(_V2_DEPENDENCY_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError(
                "selected-source dependency receipt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class RustSelectedSourceCompileReceiptV2:
    schema: str
    compiler_policy_sha256: str
    rustc_identity_sha256: str
    preparation_sha256: str
    selected_source_grant_sha256: str
    selected_source_sha256: str
    harness_sha256: str
    edition: str
    arguments: tuple[str, ...]
    compile_limits_sha256: str
    compile_environment_sha256: str
    compile_invocations: int
    dependency_receipt_sha256: str
    binary_sha256: str
    binary_size_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        roots = (
            self.rustc_identity_sha256,
            self.preparation_sha256,
            self.selected_source_grant_sha256,
            self.selected_source_sha256,
            self.harness_sha256,
            self.compile_limits_sha256,
            self.compile_environment_sha256,
            self.dependency_receipt_sha256,
            self.binary_sha256,
            self.stdout_sha256,
            self.stderr_sha256,
        )
        if (
            self.schema != V2_SCHEMA
            or self.compiler_policy_sha256 != V2_COMPILER_POLICY_SHA256
            or any(not _valid_sha256(value) for value in roots)
            or self.harness_sha256 != HARNESS_SHA256
            or self.edition != RUST_EDITION
            or self.arguments != _compile_arguments()
            or self.compile_limits_sha256 != COMPILE_LIMITS.sha256
            or type(self.compile_invocations) is not int
            or self.compile_invocations != 1
            or type(self.binary_size_bytes) is not int
            or not 1 <= self.binary_size_bytes <= MAX_BINARY_BYTES
            or self.stdout_sha256 != _EMPTY_SHA256
            or self.stderr_sha256 != _EMPTY_SHA256
            or self.sha256 != _digest(_V2_COMPILE_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError(
                "selected-source compile receipt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class RustSelectedSourceRunReceiptV2:
    schema: str
    compiler_policy_sha256: str
    preparation_sha256: str
    compile_receipt_sha256: str
    selected_source_grant_sha256: str
    batch_protocol_sha256: str
    batch_sha256: str
    stdin_sha256: str
    stdout_sha256: str
    stderr_sha256: str
    output_sha256: str
    run_limits_sha256: str
    run_environment_sha256: str
    run_invocations: int
    case_count: int
    returncode: int
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        roots = (
            self.compile_receipt_sha256,
            self.preparation_sha256,
            self.selected_source_grant_sha256,
            self.batch_protocol_sha256,
            self.batch_sha256,
            self.stdin_sha256,
            self.stdout_sha256,
            self.stderr_sha256,
            self.output_sha256,
            self.run_limits_sha256,
            self.run_environment_sha256,
        )
        if (
            self.schema != V2_SCHEMA
            or self.compiler_policy_sha256 != V2_COMPILER_POLICY_SHA256
            or any(not _valid_sha256(value) for value in roots)
            or self.batch_protocol_sha256 != V2_BATCH_PROTOCOL_SHA256
            or self.run_limits_sha256 != RUN_LIMITS.sha256
            or type(self.run_invocations) is not int
            or self.run_invocations != 1
            or type(self.case_count) is not int
            or not 1 <= self.case_count <= MAX_CASES
            or type(self.returncode) is not int
            or self.returncode != 0
            or self.stderr_sha256 != _EMPTY_SHA256
            or self.sha256 != _digest(_V2_RUN_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError(
                "selected-source run receipt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class RustSelectedActionExecutionV2:
    selected_source: RustSelectedSourceV2
    rustc_identity: RustcIdentityV1
    preparation: RustSelectedSourcePreparationV2
    dependency_receipt: RustSelectedSourceDependencyReceiptV2
    compile_receipt: RustSelectedSourceCompileReceiptV2
    run_receipt: RustSelectedSourceRunReceiptV2
    outputs: tuple[tuple[int, ...], ...]
    sha256: str

    def authenticate_structure(self) -> None:
        if type(self.selected_source) is not RustSelectedSourceV2:
            raise RustSelectedActionExecutionError("execution selected source changed")
        if type(self.rustc_identity) is not RustcIdentityV1:
            raise RustSelectedActionExecutionError("execution rustc identity changed")
        if type(self.preparation) is not RustSelectedSourcePreparationV2:
            raise RustSelectedActionExecutionError("execution preparation changed")
        if type(self.dependency_receipt) is not RustSelectedSourceDependencyReceiptV2:
            raise RustSelectedActionExecutionError("execution dependency receipt changed")
        if type(self.compile_receipt) is not RustSelectedSourceCompileReceiptV2:
            raise RustSelectedActionExecutionError("execution compile receipt changed")
        if type(self.run_receipt) is not RustSelectedSourceRunReceiptV2:
            raise RustSelectedActionExecutionError("execution run receipt changed")
        self.selected_source.authenticate_structure()
        self.rustc_identity.authenticate_structure()
        self.preparation.authenticate_structure()
        self.dependency_receipt.authenticate_structure()
        self.compile_receipt.authenticate_structure()
        self.run_receipt.authenticate_structure()
        outputs = _validate_outputs(self.outputs, expected_cases=self.run_receipt.case_count)
        output_sha256 = _digest(_V2_BATCH_NAMESPACE, outputs)
        payload = {
            "selected_source_grant_sha256": self.selected_source.sha256,
            "rustc_identity_sha256": self.rustc_identity.sha256,
            "preparation_sha256": self.preparation.sha256,
            "dependency_receipt_sha256": self.dependency_receipt.sha256,
            "compile_receipt_sha256": self.compile_receipt.sha256,
            "run_receipt_sha256": self.run_receipt.sha256,
            "outputs": outputs,
        }
        if (
            self.dependency_receipt.selected_source_grant_sha256 != self.selected_source.sha256
            or self.preparation.rustc_identity_sha256 != self.rustc_identity.sha256
            or self.dependency_receipt.selected_source_sha256 != self.selected_source.source_sha256
            or self.compile_receipt.rustc_identity_sha256 != self.rustc_identity.sha256
            or self.compile_receipt.preparation_sha256 != self.preparation.sha256
            or self.compile_receipt.compile_environment_sha256
            != self.preparation.compile_environment_sha256
            or self.compile_receipt.selected_source_grant_sha256 != self.selected_source.sha256
            or self.compile_receipt.selected_source_sha256 != self.selected_source.source_sha256
            or self.compile_receipt.dependency_receipt_sha256 != self.dependency_receipt.sha256
            or self.run_receipt.compile_receipt_sha256 != self.compile_receipt.sha256
            or self.run_receipt.preparation_sha256 != self.preparation.sha256
            or self.run_receipt.run_environment_sha256 != self.preparation.run_environment_sha256
            or self.run_receipt.selected_source_grant_sha256 != self.selected_source.sha256
            or self.run_receipt.output_sha256 != output_sha256
            or self.run_receipt.case_count != self.preparation.case_count
            or len(outputs) != self.preparation.case_count
            or self.run_receipt.stdout_sha256 != sha256(_encode_batch(outputs)).hexdigest()
            or self.run_receipt.stderr_sha256 != _EMPTY_SHA256
            or self.preparation.expected_authority_sha256 != self.selected_source.authority_sha256
            or self.preparation.batch_sha256 != self.run_receipt.batch_sha256
            or self.preparation.stdin_sha256 != self.run_receipt.stdin_sha256
            or self.sha256 != _digest(_V2_EXECUTION_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError(
                "selected-source execution receipt authentication failed"
            )


@dataclass(frozen=True, slots=True)
class _ProcessObservationV1:
    stdout: bytes
    stderr: bytes
    returncode: int
    timed_out: bool
    output_exceeded: bool


class _WindowsJobV1:
    _JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
    _JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    _JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    _EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self, limits: RustProcessLimitsV1) -> None:
        from ctypes import wintypes

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        self._handle: int | None = int(handle)
        self._kernel32 = kernel32
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_JOB_TIME
            | self._JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | self._JOB_OBJECT_LIMIT_JOB_MEMORY
            | self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | self._JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
        )
        information.BasicLimitInformation.PerJobUserTimeLimit = limits.cpu_seconds * 10_000_000
        information.BasicLimitInformation.ActiveProcessLimit = limits.active_processes
        information.JobMemoryLimit = limits.memory_bytes
        if not kernel32.SetInformationJobObject(
            wintypes.HANDLE(self._handle),
            self._EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            self.close()
            raise OSError(error, "SetInformationJobObject failed")

    def attach_suspended(self, process: subprocess.Popen[bytes]) -> None:
        from ctypes import wintypes

        if self._handle is None:
            raise OSError("job is closed")
        self._kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
        if not self._kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(self._handle), process_handle
        ):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    @staticmethod
    def resume(process: subprocess.Popen[bytes]) -> None:
        """Resume the already job-bound process via Windows NtResumeProcess."""
        from ctypes import wintypes

        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        status = int(ntdll.NtResumeProcess(wintypes.HANDLE(int(process._handle))))  # type: ignore[attr-defined]
        if status != 0:
            raise OSError(status, "NtResumeProcess failed")

    def terminate(self, exit_code: int) -> None:
        from ctypes import wintypes

        if self._handle is None:
            return
        self._kernel32.TerminateJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.UINT,
        ]
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        if not self._kernel32.TerminateJobObject(wintypes.HANDLE(self._handle), exit_code):
            error = ctypes.get_last_error()
            if error:
                raise OSError(error, "TerminateJobObject failed")

    def close(self) -> None:
        from ctypes import wintypes

        if self._handle is None:
            return
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32.CloseHandle(wintypes.HANDLE(self._handle))
        self._handle = None


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    job: _WindowsJobV1,
    *,
    exit_code: int,
) -> None:
    try:
        job.terminate(exit_code)
    except OSError:
        if process.poll() is None:
            process.kill()


def _remaining_seconds(deadline: float) -> float:
    return max(0.001, deadline - monotonic())


def _run_capped_process(
    command: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str],
    input_bytes: bytes,
    limits: RustProcessLimitsV1,
) -> _ProcessObservationV1:
    limits.authenticate_structure()
    if os.name != "nt":
        raise RustSelectedActionExecutionError(
            "bounded Rust process execution requires Windows Job Objects"
        )
    started = monotonic()
    execution_deadline = started + limits.wall_seconds
    cleanup_deadline = execution_deadline + PROCESS_CLEANUP_SECONDS
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | 0x00000004
    process: subprocess.Popen[bytes] | None = None
    job: _WindowsJobV1 | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )
        job = _WindowsJobV1(limits)
        job.attach_suspended(process)
    except (OSError, ValueError) as exc:
        if process is not None and process.poll() is None:
            process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=_remaining_seconds(cleanup_deadline))
        if job is not None:
            job.close()
        raise RustSelectedActionExecutionError("bounded process could not start") from exc

    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    output_exceeded = Event()
    io_error = Event()
    termination_lock = Lock()
    output_lock = Lock()
    output_count = 0

    assert job is not None

    def terminate_for_output() -> None:
        with termination_lock:
            _terminate_process_tree(process, job, exit_code=253)

    def pump(stream, sink: bytearray) -> None:  # type: ignore[no-untyped-def]
        nonlocal output_count
        try:
            while chunk := stream.read(8_192):
                exceeded = False
                with output_lock:
                    remaining = limits.output_bytes - output_count
                    accepted = min(len(chunk), max(0, remaining))
                    if accepted:
                        sink.extend(chunk[:accepted])
                        output_count += accepted
                    if accepted < len(chunk):
                        output_exceeded.set()
                        exceeded = True
                if exceeded:
                    terminate_for_output()
                    return
        except OSError:
            io_error.set()

    def write_input() -> None:
        try:
            process.stdin.write(input_bytes)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    readers = (
        Thread(target=pump, args=(process.stdout, stdout), daemon=True),
        Thread(target=pump, args=(process.stderr, stderr), daemon=True),
    )
    writer = Thread(target=write_input, daemon=True)
    for thread in (*readers, writer):
        thread.start()
    timed_out = False
    control_error: BaseException | None = None
    try:
        job.resume(process)
        try:
            process.wait(timeout=_remaining_seconds(execution_deadline))
        except subprocess.TimeoutExpired:
            timed_out = True
            with termination_lock:
                _terminate_process_tree(process, job, exit_code=252)
    except (OSError, subprocess.SubprocessError) as exc:
        control_error = exc
        with termination_lock:
            _terminate_process_tree(process, job, exit_code=251)
    finally:
        with termination_lock:
            if process.poll() is None:
                _terminate_process_tree(process, job, exit_code=250)
            try:
                process.wait(timeout=_remaining_seconds(cleanup_deadline))
            except subprocess.TimeoutExpired:
                control_error = control_error or RuntimeError(
                    "bounded process cleanup deadline exceeded"
                )
            job.close()
        for thread in (*readers, writer):
            thread.join(timeout=_remaining_seconds(cleanup_deadline))
        process.stdout.close()
        process.stderr.close()
        if not process.stdin.closed:
            process.stdin.close()
    if control_error is not None:
        raise RustSelectedActionExecutionError("bounded process control failed") from control_error
    if (
        any(thread.is_alive() for thread in (*readers, writer))
        or (io_error.is_set() and not (timed_out or output_exceeded.is_set()))
        or process.returncode is None
    ):
        raise RustSelectedActionExecutionError("bounded process I/O failed")
    return _ProcessObservationV1(
        stdout=bytes(stdout),
        stderr=bytes(stderr),
        returncode=process.returncode,
        timed_out=timed_out,
        output_exceeded=output_exceeded.is_set(),
    )


def _windows_link_search_paths() -> tuple[str, ...]:
    if os.name != "nt":
        return ()
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)")
    if not program_files_x86:
        raise RustSelectedActionExecutionError("Windows toolchain root is unavailable")
    base = Path(program_files_x86)
    vswhere = base / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    try:
        completed = subprocess.run(
            (
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=10,
        )
        installation = completed.stdout.decode("utf-8", errors="strict").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise RustSelectedActionExecutionError(
            "Visual C++ toolchain could not be resolved"
        ) from exc
    if completed.returncode != 0 or not installation or completed.stderr:
        raise RustSelectedActionExecutionError("Visual C++ toolchain could not be resolved")
    installation_path = Path(installation)
    try:
        tools_version = (
            (
                installation_path
                / "VC"
                / "Auxiliary"
                / "Build"
                / "Microsoft.VCToolsVersion.default.txt"
            )
            .read_text(encoding="ascii")
            .strip()
        )
        vc_lib = (
            installation_path / "VC" / "Tools" / "MSVC" / tools_version / "lib" / "x64"
        ).resolve(strict=True)
        sdk_versions = sorted(
            path
            for path in (base / "Windows Kits" / "10" / "Lib").iterdir()
            if path.is_dir()
            and (path / "um" / "x64" / "kernel32.lib").is_file()
            and (path / "ucrt" / "x64" / "ucrt.lib").is_file()
        )
        if not sdk_versions:
            raise OSError("Windows SDK libraries are missing")
        sdk = sdk_versions[-1]
        sdk_um = (sdk / "um" / "x64").resolve(strict=True)
        sdk_ucrt = (sdk / "ucrt" / "x64").resolve(strict=True)
    except (OSError, UnicodeError) as exc:
        raise RustSelectedActionExecutionError(
            "Windows link libraries could not be resolved"
        ) from exc
    return (str(vc_lib), str(sdk_um), str(sdk_ucrt))


def _is_reparse_point(metadata: os.stat_result) -> bool:
    return bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _stream_hash_regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[str, int]:
    digest = sha256()
    byte_count = 0
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse_point(metadata)
            or not 0 < metadata.st_size <= maximum_bytes
        ):
            raise RustSelectedActionExecutionError(f"{label} is not a bounded regular file")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != metadata.st_size:
                raise RustSelectedActionExecutionError(f"{label} changed before reading")
            while chunk := handle.read(1024 * 1024):
                byte_count += len(chunk)
                if byte_count > maximum_bytes:
                    raise RustSelectedActionExecutionError(f"{label} exceeds its byte limit")
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            if after.st_size != opened.st_size or byte_count != opened.st_size:
                raise RustSelectedActionExecutionError(f"{label} changed while reading")
    except OSError as exc:
        raise RustSelectedActionExecutionError(f"{label} could not be authenticated") from exc
    return digest.hexdigest(), byte_count


def _read_bounded_regular_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or _is_reparse_point(metadata)
            or not 0 < metadata.st_size <= maximum_bytes
        ):
            raise RustSelectedActionExecutionError(f"{label} is not a bounded regular file")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != metadata.st_size:
                raise RustSelectedActionExecutionError(f"{label} changed before reading")
            payload = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise RustSelectedActionExecutionError(f"{label} could not be read") from exc
    if (
        len(payload) > maximum_bytes
        or len(payload) != opened.st_size
        or after.st_size != opened.st_size
    ):
        raise RustSelectedActionExecutionError(f"{label} changed while reading")
    return payload


def _directory_manifest(path: Path) -> tuple[str, int, int]:
    try:
        resolved = path.resolve(strict=True)
        root_metadata = resolved.lstat()
        if not stat.S_ISDIR(root_metadata.st_mode) or _is_reparse_point(root_metadata):
            raise RustSelectedActionExecutionError(
                "Rust toolchain manifest root is not a concrete directory"
            )
        files: list[Path] = []
        for current, directory_names, file_names in os.walk(
            resolved,
            topdown=True,
            followlinks=False,
        ):
            current_path = Path(current)
            directory_names.sort()
            file_names.sort()
            for directory_name in directory_names:
                metadata = (current_path / directory_name).lstat()
                if _is_reparse_point(metadata) or not stat.S_ISDIR(metadata.st_mode):
                    raise RustSelectedActionExecutionError(
                        "Rust toolchain manifest contains a reparse directory"
                    )
            for file_name in file_names:
                files.append(current_path / file_name)
                if len(files) > MAX_TOOLCHAIN_FILES:
                    raise RustSelectedActionExecutionError(
                        "Rust toolchain manifest exceeds its file-count limit"
                    )
        files.sort(key=lambda entry: entry.relative_to(resolved).as_posix())
        stated_total = sum(entry.lstat().st_size for entry in files)
    except OSError as exc:
        raise RustSelectedActionExecutionError(
            "Rust toolchain manifest could not be enumerated"
        ) from exc
    if not files or not 0 < stated_total <= MAX_TOOLCHAIN_DIRECTORY_BYTES:
        raise RustSelectedActionExecutionError(
            "Rust toolchain manifest exceeds its aggregate byte limit"
        )
    entries: list[tuple[str, int, str]] = []
    total_bytes = 0
    for entry in files:
        digest, size = _stream_hash_regular_file(
            entry,
            label="Rust toolchain input",
            maximum_bytes=MAX_TOOLCHAIN_FILE_BYTES,
        )
        total_bytes += size
        if total_bytes > MAX_TOOLCHAIN_DIRECTORY_BYTES:
            raise RustSelectedActionExecutionError(
                "Rust toolchain manifest exceeds its aggregate byte limit"
            )
        entries.append((entry.relative_to(resolved).as_posix(), size, digest))
    return (
        _digest(_DIRECTORY_MANIFEST_NAMESPACE, entries),
        len(entries),
        total_bytes,
    )


def _read_rustc_identity(path: Path) -> RustcIdentityV1:
    try:
        resolved = path.resolve(strict=True)
        executable_sha256, _executable_bytes = _stream_hash_regular_file(
            resolved,
            label="rustc executable",
            maximum_bytes=MAX_TOOLCHAIN_FILE_BYTES,
        )
        with tempfile.TemporaryDirectory(prefix="humanbrain-rustc-version-") as root:
            completed = subprocess.run(
                (str(resolved), "-vV"),
                cwd=root,
                env=_discovery_environment(Path(root)),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=10,
            )
        version = completed.stdout.decode("utf-8", errors="strict").strip()
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise RustSelectedActionExecutionError("rustc identity could not be authenticated") from exc
    if completed.returncode != 0 or not version or completed.stderr:
        raise RustSelectedActionExecutionError("rustc identity could not be authenticated")
    host_lines = [line[6:] for line in version.splitlines() if line.startswith("host: ")]
    if len(host_lines) != 1 or not host_lines[0]:
        raise RustSelectedActionExecutionError("rustc host identity is missing")
    host_triple = host_lines[0]
    linker_name = "rust-lld.exe" if os.name == "nt" else "rust-lld"
    rustlib = resolved.parent.parent / "lib" / "rustlib" / host_triple
    linker = rustlib / "bin" / linker_name
    try:
        linker = linker.resolve(strict=True)
        linker_executable_sha256, _linker_bytes = _stream_hash_regular_file(
            linker,
            label="rust-lld executable",
            maximum_bytes=MAX_TOOLCHAIN_FILE_BYTES,
        )
    except OSError as exc:
        raise RustSelectedActionExecutionError(
            "bundled rust-lld identity could not be authenticated"
        ) from exc
    target_libdir = (rustlib / "lib").resolve(strict=True)
    target_manifest, target_count, target_bytes = _directory_manifest(target_libdir)
    link_search_paths = _windows_link_search_paths()
    link_manifests = tuple(_directory_manifest(Path(value)) for value in link_search_paths)
    payload = {
        "path": str(resolved),
        "version_verbose": version,
        "host_triple": host_triple,
        "executable_sha256": executable_sha256,
        "linker_path": str(linker),
        "linker_executable_sha256": linker_executable_sha256,
        "target_libdir_path": str(target_libdir),
        "target_libdir_manifest_sha256": target_manifest,
        "target_libdir_file_count": target_count,
        "target_libdir_total_bytes": target_bytes,
        "link_search_paths": link_search_paths,
        "link_search_manifest_sha256s": tuple(value[0] for value in link_manifests),
        "link_search_file_counts": tuple(value[1] for value in link_manifests),
        "link_search_total_bytes": tuple(value[2] for value in link_manifests),
    }
    identity = RustcIdentityV1(
        **payload,
        sha256=_digest(_IDENTITY_NAMESPACE, payload),
    )
    identity.authenticate_structure()
    return identity


def _discover_rustc() -> RustcIdentityV1:
    rustc = shutil.which("rustc")
    if rustc is None:
        raise RustSelectedActionExecutionError("one installed rustc executable is required")
    concrete = Path(rustc)
    if rustup := shutil.which("rustup"):
        try:
            with tempfile.TemporaryDirectory(prefix="humanbrain-rustc-discovery-") as root:
                completed = subprocess.run(
                    (rustup, "which", "rustc"),
                    cwd=root,
                    env=_discovery_environment(Path(root)),
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            output = completed.stdout.decode("utf-8", errors="strict").strip()
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise RustSelectedActionExecutionError(
                "concrete rustc executable could not be resolved"
            ) from exc
        if completed.returncode != 0 or not output or completed.stderr:
            raise RustSelectedActionExecutionError(
                "concrete rustc executable could not be resolved"
            )
        concrete = Path(output)
    return _read_rustc_identity(concrete)


def _validate_materialization(value: object) -> RustActionMaterializationV1:
    if type(value) is not RustActionMaterializationV1:
        raise RustSelectedActionExecutionError(
            "exactly one Rust action materialization is required"
        )
    try:
        value.authenticate_structure()
        source_bytes = value.source.encode("ascii", errors="strict")
    except (RustActionRegistryError, UnicodeError) as exc:
        raise RustSelectedActionExecutionError(
            "Rust action materialization authentication failed"
        ) from exc
    if not source_bytes or len(source_bytes) > MAX_SOURCE_BYTES:
        raise RustSelectedActionExecutionError("selected Rust source size is invalid")
    return value


def _validate_selected_source_v2(value: object) -> RustSelectedSourceV2:
    if type(value) is not RustSelectedSourceV2:
        raise RustSelectedActionExecutionError("exactly one V2 selected-source grant is required")
    value.authenticate_structure()
    return value


def _validate_batch(value: object) -> tuple[tuple[int, ...], ...]:
    if type(value) is not tuple or not 1 <= len(value) <= MAX_CASES:
        raise RustSelectedActionExecutionError("batch must be one bounded tuple of cases")
    cases: list[tuple[int, ...]] = []
    total = 0
    for case in value:
        if type(case) is not tuple or len(case) > MAX_VALUES_PER_CASE:
            raise RustSelectedActionExecutionError("batch case is not a bounded tuple")
        checked: list[int] = []
        for item in case:
            if type(item) is not int or not _I64_MIN <= item <= _I64_MAX:
                raise RustSelectedActionExecutionError("batch values must be signed i64")
            checked.append(item)
        total += len(checked)
        if total > MAX_TOTAL_INPUT_VALUES:
            raise RustSelectedActionExecutionError("batch has too many input values")
        cases.append(tuple(checked))
    return tuple(cases)


def _encode_batch(cases: tuple[tuple[int, ...], ...]) -> bytes:
    lines = [str(len(cases))]
    for case in cases:
        values = " ".join(str(value) for value in case)
        lines.append(f"{len(case)} {values}".rstrip())
    return ("\n".join(lines) + "\n").encode("ascii")


def _validate_outputs(
    value: object,
    *,
    expected_cases: int,
) -> tuple[tuple[int, ...], ...]:
    if type(value) is not tuple or len(value) != expected_cases:
        raise RustSelectedActionExecutionError("Rust batch output case count changed")
    total = 0
    result: list[tuple[int, ...]] = []
    for case in value:
        if type(case) is not tuple or len(case) > MAX_VALUES_PER_OUTPUT:
            raise RustSelectedActionExecutionError("Rust batch output is not bounded")
        checked: list[int] = []
        for item in case:
            if type(item) is not int or not _I64_MIN <= item <= _I64_MAX:
                raise RustSelectedActionExecutionError("Rust batch output is not signed i64")
            checked.append(item)
        total += len(checked)
        if total > MAX_TOTAL_OUTPUT_VALUES:
            raise RustSelectedActionExecutionError("Rust batch output is too large")
        result.append(tuple(checked))
    return tuple(result)


def _decode_batch(stdout: bytes, *, expected_cases: int) -> tuple[tuple[int, ...], ...]:
    try:
        text = stdout.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise RustSelectedActionExecutionError("Rust batch output was not ASCII") from exc
    lines = text.splitlines(keepends=True)
    if (
        len(lines) != expected_cases + 1
        or not all(line.endswith("\n") and not line.endswith("\r\n") for line in lines)
        or lines[0] != f"{expected_cases}\n"
    ):
        raise RustSelectedActionExecutionError("Rust batch output protocol mismatch")
    outputs: list[tuple[int, ...]] = []
    for line in lines[1:]:
        body = line[:-1]
        tokens = body.split(" ")
        if not tokens or any(not token for token in tokens):
            raise RustSelectedActionExecutionError("Rust batch output protocol mismatch")
        try:
            count = int(tokens[0])
            values = tuple(int(token) for token in tokens[1:])
        except ValueError as exc:
            raise RustSelectedActionExecutionError("Rust batch output protocol mismatch") from exc
        if (
            str(count) != tokens[0]
            or count != len(values)
            or any(str(value) != token for value, token in zip(values, tokens[1:], strict=True))
        ):
            raise RustSelectedActionExecutionError("Rust batch output protocol mismatch")
        outputs.append(values)
    return _validate_outputs(tuple(outputs), expected_cases=expected_cases)


def _parse_dependency_paths(dep_info: bytes) -> tuple[str, str]:
    if not dep_info or len(dep_info) > MAX_DEP_INFO_BYTES:
        raise RustSelectedActionExecutionError("rustc dep-info is missing or oversized")
    try:
        text = dep_info.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise RustSelectedActionExecutionError("rustc dep-info is not UTF-8") from exc
    if "\\\n" in text or "\r" in text:
        raise RustSelectedActionExecutionError("rustc dep-info is not canonical")
    dependencies: set[str] = set()
    for line in text.splitlines():
        if not line:
            continue
        if ": " in line:
            _target, raw_dependencies = line.split(": ", 1)
            tokens = raw_dependencies.split(" ")
            if any(not token or "\\" in token for token in tokens):
                raise RustSelectedActionExecutionError("rustc dep-info path is not canonical")
            dependencies.update(tokens)
        elif not line.endswith(":"):
            raise RustSelectedActionExecutionError("rustc dep-info rule is malformed")
    expected = (HARNESS_FILENAME, SELECTED_FILENAME)
    if dependencies != set(expected):
        raise RustSelectedActionExecutionError(
            "rustc loaded a dependency other than the harness and selected source"
        )
    return expected


@dataclass(frozen=True, slots=True)
class _RustOperationalObservationV1:
    dep_info: bytes
    binary_sha256: str
    binary_size_bytes: int
    compile_stdout: bytes
    compile_stderr: bytes
    run_stdout: bytes
    run_stderr: bytes
    run_returncode: int
    outputs: tuple[tuple[int, ...], ...]


def _execute_authenticated_source_observation_once(
    selected: _RustSelectedSourceView,
    *,
    rustc: RustcIdentityV1,
    preparation: _RustOperationalPreparationView,
    cases: tuple[tuple[int, ...], ...],
    stdin: bytes,
    authenticate_current: Callable[[], None],
) -> _RustOperationalObservationV1:
    """Compile and run once, returning raw facts without an authority schema."""

    authenticate_current()
    try:
        with tempfile.TemporaryDirectory(prefix="humanbrain-rust-selected-") as root:
            temporary_root = Path(root)
            harness_path = temporary_root / HARNESS_FILENAME
            selected_path = temporary_root / SELECTED_FILENAME
            dep_info_path = temporary_root / DEP_INFO_FILENAME
            binary_path = temporary_root / _binary_filename()
            try:
                harness_path.write_bytes(_HARNESS_BYTES)
                selected_path.write_bytes(selected.source.encode("ascii", errors="strict"))
            except (OSError, UnicodeError) as exc:
                raise RustSelectedActionExecutionError(
                    "selected Rust sources could not be staged"
                ) from exc
            compile_observation = _run_capped_process(
                (rustc.path, *_actual_compile_arguments(rustc)),
                cwd=temporary_root,
                environment=_materialize_environment(
                    preparation.compile_environment,
                    temporary_root,
                ),
                input_bytes=b"",
                limits=COMPILE_LIMITS,
            )
            if compile_observation.timed_out:
                raise RustSelectedActionExecutionError("rustc compile wall time exceeded")
            if compile_observation.output_exceeded:
                raise RustSelectedActionExecutionError("rustc compile output limit exceeded")
            if compile_observation.returncode != 0:
                detail = compile_observation.stderr.decode("utf-8", errors="replace").strip()
                raise RustSelectedActionExecutionError(
                    f"selected Rust source failed compilation: {detail[:2_048]}"
                )
            if compile_observation.stdout or compile_observation.stderr:
                raise RustSelectedActionExecutionError(
                    "successful rustc compile emitted unexpected output"
                )
            dep_info = _read_bounded_regular_file(
                dep_info_path,
                label="rustc dep-info",
                maximum_bytes=MAX_DEP_INFO_BYTES,
            )
            _parse_dependency_paths(dep_info)
            binary_sha256, binary_size_bytes = _stream_hash_regular_file(
                binary_path,
                label="selected Rust binary",
                maximum_bytes=MAX_BINARY_BYTES,
            )
            authenticate_current()
            expected_binary = (binary_sha256, binary_size_bytes)
            prelaunch_binary = _stream_hash_regular_file(
                binary_path,
                label="selected Rust binary",
                maximum_bytes=MAX_BINARY_BYTES,
            )
            if prelaunch_binary != expected_binary:
                raise RustSelectedActionExecutionError("selected Rust binary changed before launch")
            try:
                run_observation = _run_capped_process(
                    (str(binary_path),),
                    cwd=temporary_root,
                    environment=_materialize_environment(
                        preparation.run_environment,
                        temporary_root,
                    ),
                    input_bytes=stdin,
                    limits=RUN_LIMITS,
                )
            finally:
                postrun_binary = _stream_hash_regular_file(
                    binary_path,
                    label="selected Rust binary",
                    maximum_bytes=MAX_BINARY_BYTES,
                )
                if postrun_binary != expected_binary:
                    raise RustSelectedActionExecutionError(
                        "selected Rust binary changed during execution"
                    )
    finally:
        authenticate_current()
    if run_observation.timed_out:
        raise RustSelectedActionExecutionError("selected Rust process timed out")
    if run_observation.output_exceeded:
        raise RustSelectedActionExecutionError("selected Rust process output limit exceeded")
    if run_observation.returncode != 0:
        raise RustSelectedActionExecutionError("selected Rust process failed")
    if run_observation.stderr:
        raise RustSelectedActionExecutionError("selected Rust process emitted stderr")
    outputs = _decode_batch(run_observation.stdout, expected_cases=len(cases))
    return _RustOperationalObservationV1(
        dep_info=dep_info,
        binary_sha256=binary_sha256,
        binary_size_bytes=binary_size_bytes,
        compile_stdout=compile_observation.stdout,
        compile_stderr=compile_observation.stderr,
        run_stdout=run_observation.stdout,
        run_stderr=run_observation.stderr,
        run_returncode=run_observation.returncode,
        outputs=outputs,
    )


def _make_v1_execution_receipts(
    selected: RustActionMaterializationV1,
    *,
    rustc: RustcIdentityV1,
    preparation: RustExecutorPreparationV1,
    cases: tuple[tuple[int, ...], ...],
    stdin: bytes,
    observation: _RustOperationalObservationV1,
) -> tuple[RustDependencyReceiptV1, RustCompileReceiptV1, RustRunReceiptV1]:
    dependency_payload = {
        "schema": SCHEMA,
        "materialization_sha256": selected.sha256,
        "harness_sha256": HARNESS_SHA256,
        "selected_source_sha256": selected.source_sha256,
        "dep_info_sha256": sha256(observation.dep_info).hexdigest(),
        "dependency_paths": _parse_dependency_paths(observation.dep_info),
        "dependency_content_sha256s": (
            (HARNESS_FILENAME, HARNESS_SHA256),
            (SELECTED_FILENAME, selected.source_sha256),
        ),
    }
    dependency_receipt = RustDependencyReceiptV1(
        **dependency_payload,
        sha256=_digest(_DEPENDENCY_NAMESPACE, dependency_payload),
    )
    dependency_receipt.authenticate_structure()
    compile_payload = {
        "schema": SCHEMA,
        "compiler_policy_sha256": COMPILER_POLICY_SHA256,
        "rustc_identity_sha256": rustc.sha256,
        "preparation_sha256": preparation.sha256,
        "materialization_sha256": selected.sha256,
        "selected_source_sha256": selected.source_sha256,
        "harness_sha256": HARNESS_SHA256,
        "edition": RUST_EDITION,
        "arguments": _compile_arguments(),
        "compile_limits_sha256": COMPILE_LIMITS.sha256,
        "compile_environment_sha256": preparation.compile_environment_sha256,
        "compile_invocations": 1,
        "dependency_receipt_sha256": dependency_receipt.sha256,
        "binary_sha256": observation.binary_sha256,
        "binary_size_bytes": observation.binary_size_bytes,
        "stdout_sha256": sha256(observation.compile_stdout).hexdigest(),
        "stderr_sha256": sha256(observation.compile_stderr).hexdigest(),
    }
    compile_receipt = RustCompileReceiptV1(
        **compile_payload,
        sha256=_digest(_COMPILE_NAMESPACE, compile_payload),
    )
    compile_receipt.authenticate_structure()
    run_payload = {
        "schema": SCHEMA,
        "compiler_policy_sha256": COMPILER_POLICY_SHA256,
        "preparation_sha256": preparation.sha256,
        "compile_receipt_sha256": compile_receipt.sha256,
        "materialization_sha256": selected.sha256,
        "batch_protocol_sha256": BATCH_PROTOCOL_SHA256,
        "batch_sha256": preparation.batch_sha256,
        "stdin_sha256": sha256(stdin).hexdigest(),
        "stdout_sha256": sha256(observation.run_stdout).hexdigest(),
        "stderr_sha256": sha256(observation.run_stderr).hexdigest(),
        "output_sha256": _digest(_BATCH_NAMESPACE, observation.outputs),
        "run_limits_sha256": RUN_LIMITS.sha256,
        "run_environment_sha256": preparation.run_environment_sha256,
        "run_invocations": 1,
        "case_count": len(cases),
        "returncode": observation.run_returncode,
    }
    run_receipt = RustRunReceiptV1(
        **run_payload,
        sha256=_digest(_RUN_NAMESPACE, run_payload),
    )
    run_receipt.authenticate_structure()
    return dependency_receipt, compile_receipt, run_receipt


def _make_v2_execution_receipts(
    selected: RustSelectedSourceV2,
    *,
    rustc: RustcIdentityV1,
    preparation: RustSelectedSourcePreparationV2,
    cases: tuple[tuple[int, ...], ...],
    stdin: bytes,
    observation: _RustOperationalObservationV1,
) -> tuple[
    RustSelectedSourceDependencyReceiptV2,
    RustSelectedSourceCompileReceiptV2,
    RustSelectedSourceRunReceiptV2,
]:
    dependency_payload = {
        "schema": V2_SCHEMA,
        "selected_source_grant_sha256": selected.sha256,
        "harness_sha256": HARNESS_SHA256,
        "selected_source_sha256": selected.source_sha256,
        "dep_info_sha256": sha256(observation.dep_info).hexdigest(),
        "dependency_paths": _parse_dependency_paths(observation.dep_info),
        "dependency_content_sha256s": (
            (HARNESS_FILENAME, HARNESS_SHA256),
            (SELECTED_FILENAME, selected.source_sha256),
        ),
    }
    dependency_receipt = RustSelectedSourceDependencyReceiptV2(
        **dependency_payload,
        sha256=_digest(_V2_DEPENDENCY_NAMESPACE, dependency_payload),
    )
    dependency_receipt.authenticate_structure()
    compile_payload = {
        "schema": V2_SCHEMA,
        "compiler_policy_sha256": V2_COMPILER_POLICY_SHA256,
        "rustc_identity_sha256": rustc.sha256,
        "preparation_sha256": preparation.sha256,
        "selected_source_grant_sha256": selected.sha256,
        "selected_source_sha256": selected.source_sha256,
        "harness_sha256": HARNESS_SHA256,
        "edition": RUST_EDITION,
        "arguments": _compile_arguments(),
        "compile_limits_sha256": COMPILE_LIMITS.sha256,
        "compile_environment_sha256": preparation.compile_environment_sha256,
        "compile_invocations": 1,
        "dependency_receipt_sha256": dependency_receipt.sha256,
        "binary_sha256": observation.binary_sha256,
        "binary_size_bytes": observation.binary_size_bytes,
        "stdout_sha256": sha256(observation.compile_stdout).hexdigest(),
        "stderr_sha256": sha256(observation.compile_stderr).hexdigest(),
    }
    compile_receipt = RustSelectedSourceCompileReceiptV2(
        **compile_payload,
        sha256=_digest(_V2_COMPILE_NAMESPACE, compile_payload),
    )
    compile_receipt.authenticate_structure()
    run_payload = {
        "schema": V2_SCHEMA,
        "compiler_policy_sha256": V2_COMPILER_POLICY_SHA256,
        "preparation_sha256": preparation.sha256,
        "compile_receipt_sha256": compile_receipt.sha256,
        "selected_source_grant_sha256": selected.sha256,
        "batch_protocol_sha256": V2_BATCH_PROTOCOL_SHA256,
        "batch_sha256": preparation.batch_sha256,
        "stdin_sha256": sha256(stdin).hexdigest(),
        "stdout_sha256": sha256(observation.run_stdout).hexdigest(),
        "stderr_sha256": sha256(observation.run_stderr).hexdigest(),
        "output_sha256": _digest(_V2_BATCH_NAMESPACE, observation.outputs),
        "run_limits_sha256": RUN_LIMITS.sha256,
        "run_environment_sha256": preparation.run_environment_sha256,
        "run_invocations": 1,
        "case_count": len(cases),
        "returncode": observation.run_returncode,
    }
    run_receipt = RustSelectedSourceRunReceiptV2(
        **run_payload,
        sha256=_digest(_V2_RUN_NAMESPACE, run_payload),
    )
    run_receipt.authenticate_structure()
    return dependency_receipt, compile_receipt, run_receipt


def _authenticate_rust_toolchain_current(rustc: RustcIdentityV1) -> None:
    rustc.authenticate_structure()
    try:
        path = Path(rustc.path).resolve(strict=True)
        current_sha256, _rustc_bytes = _stream_hash_regular_file(
            path,
            label="rustc executable",
            maximum_bytes=MAX_TOOLCHAIN_FILE_BYTES,
        )
        linker_path = Path(rustc.linker_path).resolve(strict=True)
        linker_sha256, _linker_bytes = _stream_hash_regular_file(
            linker_path,
            label="rust-lld executable",
            maximum_bytes=MAX_TOOLCHAIN_FILE_BYTES,
        )
        link_search_paths = tuple(
            str(Path(value).resolve(strict=True)) for value in rustc.link_search_paths
        )
        target_manifest = _directory_manifest(Path(rustc.target_libdir_path))
        link_manifests = tuple(
            _directory_manifest(Path(value)) for value in rustc.link_search_paths
        )
    except OSError as exc:
        raise RustSelectedActionExecutionError("Rust toolchain executable changed") from exc
    if (
        str(path) != rustc.path
        or current_sha256 != rustc.executable_sha256
        or str(linker_path) != rustc.linker_path
        or linker_sha256 != rustc.linker_executable_sha256
        or link_search_paths != rustc.link_search_paths
        or target_manifest
        != (
            rustc.target_libdir_manifest_sha256,
            rustc.target_libdir_file_count,
            rustc.target_libdir_total_bytes,
        )
        or tuple(value[0] for value in link_manifests) != rustc.link_search_manifest_sha256s
        or tuple(value[1] for value in link_manifests) != rustc.link_search_file_counts
        or tuple(value[2] for value in link_manifests) != rustc.link_search_total_bytes
    ):
        raise RustSelectedActionExecutionError("Rust toolchain executable changed")


class PreparedRustSelectedActionExecutorV1:
    """One deep seam for authenticated selected-source compilation and execution."""

    __slots__ = (
        "_callback_invoked",
        "_cases",
        "_consumed",
        "_lock",
        "_preparation",
        "_registry",
        "_rustc",
        "_stdin",
    )

    def __init__(
        self,
        *,
        rustc: RustcIdentityV1,
        preparation: RustExecutorPreparationV1,
        registry: PreparedRustActionRegistryV1,
        cases: tuple[tuple[int, ...], ...],
        stdin: bytes,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("prepared Rust executors are factory-owned")
        if (
            type(rustc) is not RustcIdentityV1
            or type(preparation) is not RustExecutorPreparationV1
            or type(registry) is not PreparedRustActionRegistryV1
        ):
            raise TypeError("executor preparation identities must be exact")
        rustc.authenticate_structure()
        preparation.authenticate_structure()
        checked_cases = _validate_batch(cases)
        expected_stdin = _encode_batch(checked_cases)
        if (
            type(stdin) is not bytes
            or stdin != expected_stdin
            or preparation.rustc_identity_sha256 != rustc.sha256
            or preparation.expected_registry_sha256 != registry.receipt.sha256
            or preparation.batch_sha256 != _digest(_BATCH_NAMESPACE, checked_cases)
            or preparation.stdin_sha256 != sha256(stdin).hexdigest()
            or preparation.case_count != len(checked_cases)
        ):
            raise RustSelectedActionExecutionError("executor preparation derivation changed")
        self._rustc = rustc
        self._preparation = preparation
        self._registry = registry
        self._cases = checked_cases
        self._stdin = stdin
        self._lock = Lock()
        self._consumed = False
        self._callback_invoked = False
        self._authenticate_current()

    @property
    def rustc_identity(self) -> RustcIdentityV1:
        self._authenticate_current()
        return self._rustc

    @property
    def preparation(self) -> RustExecutorPreparationV1:
        self._preparation.authenticate_structure()
        return self._preparation

    def _authenticate_current(self) -> None:
        _authenticate_rust_toolchain_current(self._rustc)
        self._preparation.authenticate_structure()
        if (
            self._preparation.rustc_identity_sha256 != self._rustc.sha256
            or self._registry.receipt.sha256 != self._preparation.expected_registry_sha256
        ):
            raise RustSelectedActionExecutionError("executor authority changed")

    def execute_selected_once(
        self,
        recall: action_memory.KnowledgeActionRecallV1,
    ) -> RustSelectedActionExecutionV1:
        """Consume this executor and execute its registry-selected action once."""
        with self._lock:
            if self._consumed:
                raise RustSelectedActionExecutionError("prepared Rust executor was consumed")
            self._consumed = True
        return self._registry.execute_selected(
            recall,
            self._execute_materialization_once,
        )

    def _execute_materialization_once(
        self,
        materialization: RustActionMaterializationV1,
    ) -> RustSelectedActionExecutionV1:
        with self._lock:
            if self._callback_invoked:
                raise RustSelectedActionExecutionError(
                    "registry materialization callback was already consumed"
                )
            self._callback_invoked = True
        selected = _validate_materialization(materialization)
        if selected.registry_sha256 != self._preparation.expected_registry_sha256:
            raise RustSelectedActionExecutionError(
                "materialization registry differs from executor preparation"
            )
        observation = _execute_authenticated_source_observation_once(
            selected,
            rustc=self._rustc,
            preparation=self._preparation,
            cases=self._cases,
            stdin=self._stdin,
            authenticate_current=self._authenticate_current,
        )
        dependency_receipt, compile_receipt, run_receipt = _make_v1_execution_receipts(
            selected,
            rustc=self._rustc,
            preparation=self._preparation,
            cases=self._cases,
            stdin=self._stdin,
            observation=observation,
        )
        outputs = observation.outputs
        execution_payload = {
            "materialization_sha256": selected.sha256,
            "rustc_identity_sha256": self._rustc.sha256,
            "preparation_sha256": self._preparation.sha256,
            "dependency_receipt_sha256": dependency_receipt.sha256,
            "compile_receipt_sha256": compile_receipt.sha256,
            "run_receipt_sha256": run_receipt.sha256,
            "outputs": outputs,
        }
        execution = RustSelectedActionExecutionV1(
            materialization=selected,
            rustc_identity=self._rustc,
            preparation=self._preparation,
            dependency_receipt=dependency_receipt,
            compile_receipt=compile_receipt,
            run_receipt=run_receipt,
            outputs=outputs,
            sha256=_digest(_EXECUTION_NAMESPACE, execution_payload),
        )
        execution.authenticate_structure()
        return execution


def prepare_rust_selected_action_executor_v1(
    registry: PreparedRustActionRegistryV1,
    batch: tuple[tuple[int, ...], ...],
) -> PreparedRustSelectedActionExecutorV1:
    """Freeze registry, harness, batch bytes, limits, and one concrete toolchain."""
    if os.name != "nt":
        raise RustSelectedActionExecutionError(
            "Rust selected-action execution v1 requires Windows Job Objects"
        )
    if type(registry) is not PreparedRustActionRegistryV1:
        raise TypeError("one exact prepared Rust action registry is required")
    registry_sha256 = registry.receipt.sha256
    cases = _validate_batch(batch)
    stdin = _encode_batch(cases)
    identity = _discover_rustc()
    compile_environment, run_environment = _frozen_environments(identity)
    payload = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "compiler_policy_sha256": COMPILER_POLICY_SHA256,
        "rustc_identity_sha256": identity.sha256,
        "expected_registry_sha256": registry_sha256,
        "harness_sha256": HARNESS_SHA256,
        "batch_protocol_sha256": BATCH_PROTOCOL_SHA256,
        "batch_sha256": _digest(_BATCH_NAMESPACE, cases),
        "stdin_sha256": sha256(stdin).hexdigest(),
        "compile_environment": compile_environment,
        "compile_environment_sha256": _environment_sha256("compile", compile_environment),
        "run_environment": run_environment,
        "run_environment_sha256": _environment_sha256("run", run_environment),
        "case_count": len(cases),
        "compile_limits_sha256": COMPILE_LIMITS.sha256,
        "run_limits_sha256": RUN_LIMITS.sha256,
    }
    preparation = RustExecutorPreparationV1(
        **payload,
        sha256=_digest(_PREPARATION_NAMESPACE, payload),
    )
    preparation.authenticate_structure()
    return PreparedRustSelectedActionExecutorV1(
        rustc=identity,
        preparation=preparation,
        registry=registry,
        cases=cases,
        stdin=stdin,
        construction_token=_CONSTRUCTION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class RustActionRegistryAuthorityReceiptV2:
    schema: str
    registry_receipt_sha256: str
    instance_nonce_sha256: str
    sha256: str

    def authenticate_structure(self) -> None:
        payload = asdict(self)
        payload.pop("sha256")
        if (
            self.schema != V2_SCHEMA
            or not _valid_sha256(self.registry_receipt_sha256)
            or not _valid_sha256(self.instance_nonce_sha256)
            or self.sha256 != _digest(_V2_REGISTRY_AUTHORITY_NAMESPACE, payload)
        ):
            raise RustSelectedActionExecutionError(
                "registry authority receipt authentication failed"
            )


def _make_registry_authority_receipt_v2(
    registry: PreparedRustActionRegistryV1,
) -> RustActionRegistryAuthorityReceiptV2:
    registry_receipt = registry.receipt
    registry_receipt.authenticate_structure()
    payload = {
        "schema": V2_SCHEMA,
        "registry_receipt_sha256": registry_receipt.sha256,
        "instance_nonce_sha256": sha256(token_bytes(32)).hexdigest(),
    }
    result = RustActionRegistryAuthorityReceiptV2(
        **payload,
        sha256=_digest(_V2_REGISTRY_AUTHORITY_NAMESPACE, payload),
    )
    result.authenticate_structure()
    return result


class PreparedRustActionRegistryAuthorityV2:
    """Exact V2 authority adapter over the existing source-owning registry."""

    __slots__ = ("_authority_receipt", "_registry")

    def __init__(
        self,
        *,
        registry: PreparedRustActionRegistryV1,
        authority_receipt: RustActionRegistryAuthorityReceiptV2,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("selected-source authorities are factory-owned")
        if type(registry) is not PreparedRustActionRegistryV1:
            raise TypeError("one exact prepared Rust action registry is required")
        if type(authority_receipt) is not RustActionRegistryAuthorityReceiptV2:
            raise TypeError("registry authority receipt must be exact")
        registry_receipt = registry.receipt
        registry_receipt.authenticate_structure()
        authority_receipt.authenticate_structure()
        if authority_receipt.registry_receipt_sha256 != registry_receipt.sha256:
            raise RustSelectedActionExecutionError("registry authority receipt changed")
        self._registry = registry
        self._authority_receipt = authority_receipt

    @property
    def receipt(self) -> RustActionRegistryAuthorityReceiptV2:
        registry_receipt = self._registry.receipt
        self._authority_receipt.authenticate_structure()
        if self._authority_receipt.registry_receipt_sha256 != registry_receipt.sha256:
            raise RustSelectedActionExecutionError("registry authority receipt changed")
        return self._authority_receipt

    @property
    def authority_sha256(self) -> str:
        return self.receipt.sha256

    def execute_selected(
        self,
        selection: object,
        execute: Callable[[RustSelectedSourceV2], _T],
    ) -> _T:
        if type(selection) is not action_memory.KnowledgeActionRecallV1:
            raise RustSelectedActionExecutionError(
                "registry authority requires one exact knowledge-action recall"
            )
        if not callable(execute):
            raise TypeError("execute must be one callable selected-source adapter")

        def release(materialization: RustActionMaterializationV1) -> _T:
            selected = _validate_materialization(materialization)
            grant = make_rust_selected_source_v2(
                authority_sha256=self.authority_sha256,
                selection_sha256=selected.recall_sha256,
                release_sha256=selected.sha256,
                action_handle=selected.action_handle,
                source=selected.source,
            )
            if type(grant) is not RustSelectedSourceV2:
                raise RustSelectedActionExecutionError(
                    "registry selected-source factory returned an inexact grant"
                )
            grant.authenticate_structure()
            if (
                grant.source_sha256 != selected.source_sha256
                or selected.registry_sha256 != self.receipt.registry_receipt_sha256
                or grant.authority_sha256 != self.authority_sha256
                or grant.selection_sha256 != selected.recall_sha256
                or grant.release_sha256 != selected.sha256
                or grant.action_handle != selected.action_handle
                or grant.source != selected.source
            ):
                raise RustSelectedActionExecutionError("registry selected-source authority changed")
            return execute(grant)

        try:
            return self._registry.execute_selected(selection, release)
        except RustSelectedActionExecutionError:
            raise
        except (RustActionRegistryError, TypeError) as exc:
            raise RustSelectedActionExecutionError(
                "registry selected-source release failed"
            ) from exc


def adapt_rust_action_registry_authority_v2(
    registry: PreparedRustActionRegistryV1,
) -> PreparedRustActionRegistryAuthorityV2:
    """Expose one exact prepared registry through the V2 authority contract."""

    authority_receipt = _make_registry_authority_receipt_v2(registry)
    return PreparedRustActionRegistryAuthorityV2(
        registry=registry,
        authority_receipt=authority_receipt,
        construction_token=_CONSTRUCTION_TOKEN,
    )


def _v2_authority_sha256(authority: object) -> str:
    if type(authority) is PreparedRustActionRegistryAuthorityV2:
        root = authority.authority_sha256
    else:
        try:
            from experiments.r3_pilot.common_promoted_rust_action_vault_v1 import (
                PreparedPromotedRustActionVaultV1,
            )
        except ImportError as exc:
            raise TypeError("one exact supported selected-source authority is required") from exc
        if type(authority) is not PreparedPromotedRustActionVaultV1:
            raise TypeError("one exact supported selected-source authority is required")
        root = authority.authority_sha256
    if not _valid_sha256(root):
        raise RustSelectedActionExecutionError("selected-source authority root is invalid")
    return root


def _execute_v2_authority(
    authority: object,
    selection: object,
    execute: Callable[[RustSelectedSourceV2], _T],
) -> _T:
    _v2_authority_sha256(authority)
    if type(authority) is PreparedRustActionRegistryAuthorityV2:
        return authority.execute_selected(selection, execute)
    from experiments.r3_pilot.common_promoted_rust_action_vault_v1 import (
        PreparedPromotedRustActionVaultV1,
    )

    if type(authority) is not PreparedPromotedRustActionVaultV1:
        raise TypeError("one exact supported selected-source authority is required")
    return authority.execute_selected(selection, execute)


class PreparedRustSelectedActionExecutorV2:
    """One-shot trusted-host execution for an exact selected-source authority."""

    __slots__ = (
        "_authority",
        "_callback_invoked",
        "_cases",
        "_consumed",
        "_expected_selection_sha256",
        "_lock",
        "_preparation",
        "_rustc",
        "_stdin",
    )

    def __init__(
        self,
        *,
        rustc: RustcIdentityV1,
        preparation: RustSelectedSourcePreparationV2,
        authority: object,
        cases: tuple[tuple[int, ...], ...],
        stdin: bytes,
        construction_token: object,
    ) -> None:
        if construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("prepared Rust executors are factory-owned")
        if type(rustc) is not RustcIdentityV1:
            raise TypeError("executor rustc identity must be exact")
        if type(preparation) is not RustSelectedSourcePreparationV2:
            raise TypeError("executor V2 preparation must be exact")
        rustc.authenticate_structure()
        preparation.authenticate_structure()
        authority_sha256 = _v2_authority_sha256(authority)
        checked_cases = _validate_batch(cases)
        expected_stdin = _encode_batch(checked_cases)
        if (
            type(stdin) is not bytes
            or stdin != expected_stdin
            or preparation.rustc_identity_sha256 != rustc.sha256
            or preparation.expected_authority_sha256 != authority_sha256
            or preparation.batch_sha256 != _digest(_V2_BATCH_NAMESPACE, checked_cases)
            or preparation.stdin_sha256 != sha256(stdin).hexdigest()
            or preparation.case_count != len(checked_cases)
        ):
            raise RustSelectedActionExecutionError(
                "selected-source executor preparation derivation changed"
            )
        self._rustc = rustc
        self._preparation = preparation
        self._authority = authority
        self._cases = checked_cases
        self._stdin = stdin
        self._lock = Lock()
        self._consumed = False
        self._callback_invoked = False
        self._expected_selection_sha256: str | None = None
        self._authenticate_current()

    @property
    def rustc_identity(self) -> RustcIdentityV1:
        self._authenticate_current()
        return self._rustc

    @property
    def preparation(self) -> RustSelectedSourcePreparationV2:
        self._preparation.authenticate_structure()
        return self._preparation

    def _authenticate_current(self) -> None:
        _authenticate_rust_toolchain_current(self._rustc)
        self._preparation.authenticate_structure()
        if (
            self._preparation.rustc_identity_sha256 != self._rustc.sha256
            or _v2_authority_sha256(self._authority) != self._preparation.expected_authority_sha256
        ):
            raise RustSelectedActionExecutionError("selected-source executor authority changed")

    def execute_selected_once(self, selection: object) -> RustSelectedActionExecutionV2:
        """Consume the executor before asking its exact authority to release source."""

        with self._lock:
            if self._consumed:
                raise RustSelectedActionExecutionError("prepared Rust executor was consumed")
            self._consumed = True
        selection_sha256 = getattr(selection, "sha256", None)
        if not _valid_sha256(selection_sha256):
            raise RustSelectedActionExecutionError("selection root is invalid")
        self._expected_selection_sha256 = selection_sha256
        try:
            return _execute_v2_authority(
                self._authority,
                selection,
                self._execute_selected_source_once,
            )
        except RustSelectedActionExecutionError:
            raise
        except (TypeError, ValueError) as exc:
            raise RustSelectedActionExecutionError(
                "selected-source authority release failed"
            ) from exc

    def _execute_selected_source_once(
        self,
        selected_source: RustSelectedSourceV2,
    ) -> RustSelectedActionExecutionV2:
        with self._lock:
            if self._callback_invoked:
                raise RustSelectedActionExecutionError(
                    "selected-source authority callback was already consumed"
                )
            self._callback_invoked = True
        selected = _validate_selected_source_v2(selected_source)
        if (
            selected.authority_sha256 != self._preparation.expected_authority_sha256
            or selected.selection_sha256 != self._expected_selection_sha256
        ):
            raise RustSelectedActionExecutionError(
                "selected-source grant differs from executor preparation"
            )
        observation = _execute_authenticated_source_observation_once(
            selected,
            rustc=self._rustc,
            preparation=self._preparation,
            cases=self._cases,
            stdin=self._stdin,
            authenticate_current=self._authenticate_current,
        )
        dependency_receipt, compile_receipt, run_receipt = _make_v2_execution_receipts(
            selected,
            rustc=self._rustc,
            preparation=self._preparation,
            cases=self._cases,
            stdin=self._stdin,
            observation=observation,
        )
        outputs = observation.outputs
        execution_payload = {
            "selected_source_grant_sha256": selected.sha256,
            "rustc_identity_sha256": self._rustc.sha256,
            "preparation_sha256": self._preparation.sha256,
            "dependency_receipt_sha256": dependency_receipt.sha256,
            "compile_receipt_sha256": compile_receipt.sha256,
            "run_receipt_sha256": run_receipt.sha256,
            "outputs": outputs,
        }
        execution = RustSelectedActionExecutionV2(
            selected_source=selected,
            rustc_identity=self._rustc,
            preparation=self._preparation,
            dependency_receipt=dependency_receipt,
            compile_receipt=compile_receipt,
            run_receipt=run_receipt,
            outputs=outputs,
            sha256=_digest(_V2_EXECUTION_NAMESPACE, execution_payload),
        )
        execution.authenticate_structure()
        return execution


def prepare_rust_selected_action_executor_v2(
    authority: object,
    batch: tuple[tuple[int, ...], ...],
) -> PreparedRustSelectedActionExecutorV2:
    """Freeze one exact V2 authority, batch, policy, limits, and toolchain."""

    if os.name != "nt":
        raise RustSelectedActionExecutionError(
            "Rust selected-source execution v2 requires Windows Job Objects"
        )
    authority_sha256 = _v2_authority_sha256(authority)
    cases = _validate_batch(batch)
    stdin = _encode_batch(cases)
    identity = _discover_rustc()
    compile_environment, run_environment = _frozen_environments(identity)
    payload = {
        "schema": V2_SCHEMA,
        "claim_boundary": V2_CLAIM_BOUNDARY,
        "compiler_policy_sha256": V2_COMPILER_POLICY_SHA256,
        "rustc_identity_sha256": identity.sha256,
        "expected_authority_sha256": authority_sha256,
        "harness_sha256": HARNESS_SHA256,
        "batch_protocol_sha256": V2_BATCH_PROTOCOL_SHA256,
        "batch_sha256": _digest(_V2_BATCH_NAMESPACE, cases),
        "stdin_sha256": sha256(stdin).hexdigest(),
        "compile_environment": compile_environment,
        "compile_environment_sha256": _environment_sha256("compile", compile_environment),
        "run_environment": run_environment,
        "run_environment_sha256": _environment_sha256("run", run_environment),
        "case_count": len(cases),
        "compile_limits_sha256": COMPILE_LIMITS.sha256,
        "run_limits_sha256": RUN_LIMITS.sha256,
    }
    preparation = RustSelectedSourcePreparationV2(
        **payload,
        sha256=_digest(_V2_PREPARATION_NAMESPACE, payload),
    )
    preparation.authenticate_structure()
    return PreparedRustSelectedActionExecutorV2(
        rustc=identity,
        preparation=preparation,
        authority=authority,
        cases=cases,
        stdin=stdin,
        construction_token=_CONSTRUCTION_TOKEN,
    )


__all__ = [
    "PreparedRustActionRegistryAuthorityV2",
    "PreparedRustSelectedActionExecutorV1",
    "PreparedRustSelectedActionExecutorV2",
    "RustActionRegistryAuthorityReceiptV2",
    "RustCompileReceiptV1",
    "RustDependencyReceiptV1",
    "RustExecutorPreparationV1",
    "RustRunReceiptV1",
    "RustSelectedActionExecutionError",
    "RustSelectedActionExecutionV1",
    "RustSelectedActionExecutionV2",
    "RustSelectedSourceCompileReceiptV2",
    "RustSelectedSourceDependencyReceiptV2",
    "RustSelectedSourcePreparationV2",
    "RustSelectedSourceRunReceiptV2",
    "RustSelectedSourceV2",
    "adapt_rust_action_registry_authority_v2",
    "make_rust_selected_source_v2",
    "prepare_rust_selected_action_executor_v1",
    "prepare_rust_selected_action_executor_v2",
]
