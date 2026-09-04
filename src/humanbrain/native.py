"""Build and bind the dependency-free native human-cell simulation kernel."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from humanbrain.cells import AdaptiveGlifPopulation

_SOURCE_ROOT = Path(__file__).with_name("_native_src")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PROCEDURAL_RUNTIME_SOURCE_PATHS = (
    ".gitattributes",
    "data/allen/523000114_neuron_config.json",
    "src/humanbrain/_native_src/Cargo.lock",
    "src/humanbrain/_native_src/Cargo.toml",
    "src/humanbrain/_native_src/src/lib.rs",
    "src/humanbrain/_native_src/src/procedural.rs",
    "src/humanbrain/cells.py",
    "src/humanbrain/human_cell_identity.py",
    "src/humanbrain/native.py",
    "src/humanbrain/procedural_runtime.py",
    "src/humanbrain/procedural_topology.py",
    "src/humanbrain/scale_accounting.py",
)
_BUILD_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_PATH: Path | None = None
_LOADED_BUILD_PLAN: _NativeBuildPlan | None = None
_NATIVE_CARGO_FEATURES: tuple[str, ...] = ()
_TARGET_ROOT_PLACEHOLDER = "@HUMANBRAIN_NATIVE_TARGET_ROOT@"
_MAX_NATIVE_SOURCE_FILE_BYTES = 1024 * 1024
# The audited sha2 plus target-specific libc/windows-sys closure is 26--27 MiB.
# Keep one explicit bounded envelope for recursive wheel/sdist/build identity.
_MAX_NATIVE_SOURCE_BUNDLE_BYTES = 32 * 1024 * 1024
_MAX_TARGET_LIBRARY_FILE_BYTES = 128 * 1024 * 1024
_MAX_TARGET_LIBRARY_BUNDLE_BYTES = 512 * 1024 * 1024
_VENDORED_CARGO_CONFIG = (
    b'[source.crates-io]\nreplace-with = "vendored-sources"\n\n'
    b'[source.vendored-sources]\ndirectory = "vendor"\n'
)
_BUILD_CONTROL_ENVIRONMENT = (
    "CARGO_BUILD_RUSTFLAGS",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_TARGET",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_HOME",
    "CARGO_TARGET_DIR",
    "RUSTC",
    "RUSTFLAGS",
    "RUSTUP_TOOLCHAIN",
)
_PASSTHROUGH_BUILD_ENVIRONMENT = (
    "COMSPEC",
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
)

_F64_POINTER = ctypes.POINTER(ctypes.c_double)
_I32_POINTER = ctypes.POINTER(ctypes.c_int32)
_U32_POINTER = ctypes.POINTER(ctypes.c_uint32)
_U8_POINTER = ctypes.POINTER(ctypes.c_uint8)
_SIZE_POINTER = ctypes.POINTER(ctypes.c_size_t)
_U64_POINTER = ctypes.POINTER(ctypes.c_uint64)


class NativeBuildError(RuntimeError):
    """Raised when the required native cell kernel cannot be built or loaded."""


def _enable_temporal_profile_build() -> None:
    """Select the benchmark-only temporal profiler before native library loading."""

    global _NATIVE_CARGO_FEATURES
    with _BUILD_LOCK:
        if _NATIVE_CARGO_FEATURES == ("temporal-profile",):
            return
        if _LIBRARY is not None:
            raise NativeBuildError(
                "temporal profiling must be enabled before native library loading"
            )
        _NATIVE_CARGO_FEATURES = ("temporal-profile",)


class _GlifConfig(ctypes.Structure):
    _fields_ = [
        ("cell_count", ctypes.c_size_t),
        ("after_channel_count", ctypes.c_size_t),
        ("spike_cut_steps", ctypes.c_int32),
        ("leak_ns", ctypes.c_double),
        ("voltage_step_scale", ctypes.c_double),
        ("threshold_beta_scale", ctypes.c_double),
        ("threshold_exp_decay", ctypes.c_double),
        ("threshold_voltage_factor", ctypes.c_double),
        ("threshold_beta_factor", ctypes.c_double),
        ("spike_threshold_decay", ctypes.c_double),
        ("threshold_inf_relative_mv", ctypes.c_double),
        ("voltage_reset_scale", ctypes.c_double),
        ("voltage_reset_offset_mv", ctypes.c_double),
        ("spike_cut_threshold_decay", ctypes.c_double),
        ("threshold_spike_increment_mv", ctypes.c_double),
    ]


class _RecurrentConfig(ctypes.Structure):
    _fields_ = [
        ("token_steps", ctypes.c_size_t),
        ("pulse_steps", ctypes.c_size_t),
        ("input_fanout", ctypes.c_size_t),
        ("recurrent_fanout", ctypes.c_size_t),
        ("input_current_pa", ctypes.c_double),
        ("recurrent_decay", ctypes.c_double),
        # R0 multi-trace bank: per-row step decays, length trace_count.
        ("trace_count", ctypes.c_size_t),
        ("eligibility_decays", _F64_POINTER),
        # R0-b family A: u32 (cells,) per-token spike-count accumulator
        # maintained by the kernel at the mask-write sites; may be NULL.
        ("token_spike_counts", _U32_POINTER),
    ]


@dataclass(frozen=True)
class NativeBuildInfo:
    source_digest: str
    library_path: Path
    build_plan: _NativeBuildPlan


@dataclass(frozen=True, slots=True)
class NativeLibraryIdentity:
    """Portable identity of the exact attested native bytes in this process."""

    source_digest: str
    binary_size: int
    binary_sha256: str

    def to_record(self) -> dict[str, int | str]:
        return {
            "binary_sha256": self.binary_sha256,
            "binary_size": self.binary_size,
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True, slots=True)
class _NativeToolchain:
    cargo_path: str
    cargo_binary_sha256: str
    cargo_version: str
    cargo_discovery_path: str
    cargo_discovery_sha256: str
    rustc_path: str
    rustc_binary_sha256: str
    rustc_version: str
    rustc_discovery_path: str
    rustc_discovery_sha256: str
    sysroot: str
    target_triple: str
    target_library_directory: str
    target_library_manifest_sha256: str
    native_cfg: str
    target_cpus: str
    linker_path: str
    linker_binary_sha256: str
    sdk_identity_sha256: str
    sdk_environment: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _NativeBuildPlan:
    source_digest: str
    command: tuple[str, ...]
    cwd: Path
    environment: tuple[tuple[str, str], ...]
    target_triple: str
    library_relative_path: Path
    toolchain_identity_sha256: str
    sdk_identity_sha256: str

    def materialize_environment(self, target_root: Path) -> dict[str, str]:
        environment = dict(self.environment)
        if environment.get("CARGO_TARGET_DIR") != _TARGET_ROOT_PLACEHOLDER:
            raise NativeBuildError("native build plan target placeholder differs")
        environment["CARGO_TARGET_DIR"] = str(target_root.resolve())
        return environment


def _framed_sha256(namespace: str, entries: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    namespace_bytes = namespace.encode("utf-8")
    digest.update(len(namespace_bytes).to_bytes(8, "big"))
    digest.update(namespace_bytes)
    for name, payload in entries:
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _native_build_input_paths(
    source_root: Path | None = None,
) -> tuple[tuple[str, Path], ...]:
    """Return normalized POSIX-relative paths in byte-stable sorted order, unread."""
    root = (_SOURCE_ROOT if source_root is None else source_root).resolve()
    paths = [
        root / "Cargo.toml",
        root / "Cargo.lock",
        root / ".cargo" / "config.toml",
    ]
    build_script = root / "build.rs"
    if build_script.is_file():
        paths.append(build_script)
    for tree in (root / "src", root / "vendor"):
        if not tree.is_dir():
            raise NativeBuildError(f"native build input tree is missing: {tree.name}")
        paths.extend(path for path in tree.rglob("*") if path.is_file())
    ordered = sorted(paths, key=lambda path: path.relative_to(root).as_posix())
    records: list[tuple[str, Path]] = []
    for path in ordered:
        if path.is_symlink():
            raise NativeBuildError("native build inputs cannot be symbolic links")
        resolved = path.resolve()
        try:
            relative_path = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise NativeBuildError("native build input escapes its source root") from error
        records.append((relative_path, resolved))
    return tuple(records)


def _bounded_file_payload(path: Path, label: str, maximum_bytes: int) -> bytes:
    try:
        stated_size = path.stat().st_size
        if stated_size > maximum_bytes:
            raise NativeBuildError(f"{label} exceeds the identity byte limit")
        with path.open("rb") as handle:
            payload = handle.read(maximum_bytes + 1)
    except OSError as error:
        raise NativeBuildError(f"{label} cannot be read for identity") from error
    if len(payload) > maximum_bytes:
        raise NativeBuildError(f"{label} exceeds the identity byte limit")
    return payload


def _native_build_input_manifest(
    source_root: Path | None = None,
) -> tuple[tuple[str, bytes], ...]:
    """Read the complete Cargo source manifest under strict byte bounds."""
    paths = _native_build_input_paths(source_root)
    try:
        stated_total = sum(path.stat().st_size for _name, path in paths)
    except OSError as error:
        raise NativeBuildError("native build input cannot be sized") from error
    if stated_total > _MAX_NATIVE_SOURCE_BUNDLE_BYTES:
        raise NativeBuildError("native build inputs exceed the aggregate byte limit")
    manifest: list[tuple[str, bytes]] = []
    actual_total = 0
    for name, path in paths:
        payload = _bounded_file_payload(
            path,
            "native build input",
            _MAX_NATIVE_SOURCE_FILE_BYTES,
        )
        actual_total += len(payload)
        if actual_total > _MAX_NATIVE_SOURCE_BUNDLE_BYTES:
            raise NativeBuildError("native build inputs exceed the aggregate byte limit")
        manifest.append((name, payload))
    return tuple(manifest)


def _sha256_file_with_size(
    path: Path,
    label: str,
    *,
    maximum_bytes: int = 512 * 1024 * 1024,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > maximum_bytes:
                    raise NativeBuildError(f"{label} exceeds the identity byte limit")
                digest.update(chunk)
    except OSError as error:
        raise NativeBuildError(f"{label} cannot be read for identity") from error
    if byte_count == 0:
        raise NativeBuildError(f"{label} is empty")
    return digest.hexdigest(), byte_count


def _sha256_file(path: Path, label: str) -> str:
    return _sha256_file_with_size(path, label)[0]


def _target_library_manifest_sha256(target_library_directory: Path) -> str:
    """Hash every concrete regular file in rustc's selected target libdir."""
    root = target_library_directory.resolve()
    if not root.is_dir():
        raise NativeBuildError("rustc target library directory cannot be discovered")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise NativeBuildError("rustc target library directory is empty")
    try:
        stated_total = sum(path.stat().st_size for path in paths)
    except OSError as error:
        raise NativeBuildError("rustc target library input cannot be sized") from error
    if stated_total > _MAX_TARGET_LIBRARY_BUNDLE_BYTES:
        raise NativeBuildError("rustc target libraries exceed the aggregate byte limit")
    entries: list[tuple[str, bytes]] = []
    actual_total = 0
    for path in paths:
        digest, byte_count = _sha256_file_with_size(
            path,
            "rustc target library",
            maximum_bytes=_MAX_TARGET_LIBRARY_FILE_BYTES,
        )
        actual_total += byte_count
        if actual_total > _MAX_TARGET_LIBRARY_BUNDLE_BYTES:
            raise NativeBuildError("rustc target libraries exceed the aggregate byte limit")
        entries.append(
            (
                path.relative_to(root).as_posix(),
                json.dumps(
                    {"byte_count": byte_count, "sha256": digest},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii"),
            )
        )
    return _framed_sha256("humanbrain-rust-target-libraries-v1", entries)


def _probe_tool(
    command: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=environment,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NativeBuildError("native toolchain identity probe failed") from error
    output = completed.stdout
    if completed.returncode != 0 or not output.strip():
        detail = completed.stderr.strip() or output.strip() or "no probe output"
        raise NativeBuildError(f"native toolchain identity probe failed: {detail}")
    if len(output.encode("utf-8")) > 256 * 1024:
        raise NativeBuildError("native toolchain identity probe output is oversized")
    return output.strip()


def _ambient_value(name: str) -> str | None:
    for key, value in os.environ.items():
        if key.upper() == name:
            return value
    return None


def _base_tool_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in _PASSTHROUGH_BUILD_ENVIRONMENT:
        value = _ambient_value(name)
        if value is not None:
            environment[name] = value
    return environment


def _reject_build_control_environment() -> None:
    present = tuple(
        name for name in _BUILD_CONTROL_ENVIRONMENT if _ambient_value(name) is not None
    )
    if present:
        raise NativeBuildError(
            "ambient native build control is forbidden: " + ", ".join(present)
        )


def _cargo_home() -> Path:
    return (Path.home() / ".cargo").resolve()


def _reject_cargo_configuration(source_root: Path, cargo_home: Path) -> None:
    candidates: list[Path] = []
    source_config = source_root / ".cargo" / "config.toml"
    if _bounded_file_payload(
        source_config,
        "vendored Cargo configuration",
        len(_VENDORED_CARGO_CONFIG),
    ) != _VENDORED_CARGO_CONFIG:
        raise NativeBuildError("vendored Cargo configuration differs")
    if (source_root / ".cargo" / "config").exists():
        raise NativeBuildError("legacy source-root Cargo configuration is forbidden")
    for ancestor in source_root.resolve().parents:
        candidates.extend((ancestor / ".cargo" / "config", ancestor / ".cargo" / "config.toml"))
    candidates.extend((cargo_home / "config", cargo_home / "config.toml"))
    if any(path.is_file() for path in candidates):
        raise NativeBuildError(
            "hierarchical Cargo configuration is forbidden for the native cache"
        )


def _windows_sdk_identity() -> tuple[str, tuple[tuple[str, str], ...]]:
    if os.name != "nt":
        digest = _framed_sha256(
            "humanbrain-native-platform-sdk-v1",
            (("platform", platform.platform().encode("utf-8")),),
        )
        return digest, ()
    program_files_x86 = _ambient_value("PROGRAMFILES(X86)") or r"C:\Program Files (x86)"
    sdk_root = Path(program_files_x86) / "Windows Kits" / "10"
    library_root = sdk_root / "Lib"
    versions = sorted(
        (
            path
            for path in library_root.iterdir()
            if path.is_dir()
            and (path / "um" / "x64" / "kernel32.lib").is_file()
            and (path / "ucrt" / "x64" / "ucrt.lib").is_file()
        ),
        key=lambda path: path.name,
    )
    if not versions:
        raise NativeBuildError("Windows SDK identity cannot be discovered")
    sdk_version_root = versions[-1]
    selected_libraries = (
        sdk_version_root / "um" / "x64" / "kernel32.lib",
        sdk_version_root / "um" / "x64" / "advapi32.lib",
        sdk_version_root / "um" / "x64" / "bcrypt.lib",
        sdk_version_root / "um" / "x64" / "userenv.lib",
        sdk_version_root / "um" / "x64" / "ws2_32.lib",
        sdk_version_root / "ucrt" / "x64" / "ucrt.lib",
    )
    entries: list[tuple[str, bytes]] = [
        ("sdk-root", str(sdk_root.resolve()).encode("utf-8")),
        ("sdk-version", sdk_version_root.name.encode("ascii")),
    ]
    for library in selected_libraries:
        entries.append(
            (
                library.relative_to(sdk_root).as_posix(),
                _sha256_file(library, "Windows SDK library").encode("ascii"),
            )
        )
    identity = _framed_sha256("humanbrain-native-windows-sdk-v1", entries)
    environment = (
        ("WindowsSdkDir", str(sdk_root.resolve()) + os.sep),
        ("WindowsSDKVersion", sdk_version_root.name + os.sep),
    )
    return identity, environment


def _discover_native_toolchain(source_root: Path) -> _NativeToolchain:
    # Rustup discovery needs its user-profile state. Its selected proxy bytes,
    # sysroot, concrete tool bytes, and every probe output are bound below; the
    # inherited discovery environment is not passed to the Cargo build.
    probe_environment = os.environ.copy()
    rustc_discovery = shutil.which("rustc", path=probe_environment.get("PATH"))
    cargo_discovery = shutil.which("cargo", path=probe_environment.get("PATH"))
    if rustc_discovery is None or cargo_discovery is None:
        raise NativeBuildError("Cargo and rustc are required for the native kernel")
    rustc_discovery_path = Path(rustc_discovery).resolve()
    cargo_discovery_path = Path(cargo_discovery).resolve()
    sysroot = Path(
        _probe_tool(
            (str(rustc_discovery_path), "--print", "sysroot"),
            cwd=source_root,
            environment=probe_environment,
        )
    ).resolve()
    executable_suffix = ".exe" if os.name == "nt" else ""
    rustc_path = (sysroot / "bin" / f"rustc{executable_suffix}").resolve()
    cargo_path = (sysroot / "bin" / f"cargo{executable_suffix}").resolve()
    if not rustc_path.is_file():
        rustc_path = rustc_discovery_path
    if not cargo_path.is_file():
        cargo_path = cargo_discovery_path
    target_triple = _probe_tool(
        (str(rustc_path), "--print", "host-tuple"),
        cwd=source_root,
        environment=probe_environment,
    )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", target_triple):
        raise NativeBuildError("rustc host target identity is invalid")
    linker_path = (
        sysroot
        / "lib"
        / "rustlib"
        / target_triple
        / "bin"
        / f"rust-lld{executable_suffix}"
    ).resolve()
    if not linker_path.is_file():
        raise NativeBuildError("the pinned rust-lld linker cannot be discovered")
    target_library_directory = Path(
        _probe_tool(
            (
                str(rustc_path),
                "--print",
                "target-libdir",
                "--target",
                target_triple,
            ),
            cwd=source_root,
            environment=probe_environment,
        )
    ).resolve()
    try:
        target_library_directory.relative_to(sysroot)
    except ValueError as error:
        raise NativeBuildError("rustc target library directory is outside its sysroot") from error
    target_library_manifest = _target_library_manifest_sha256(
        target_library_directory
    )
    sdk_identity, sdk_environment = _windows_sdk_identity()
    return _NativeToolchain(
        cargo_path=str(cargo_path),
        cargo_binary_sha256=_sha256_file(cargo_path, "Cargo executable"),
        cargo_version=_probe_tool(
            (str(cargo_path), "-vV"), cwd=source_root, environment=probe_environment
        ),
        cargo_discovery_path=str(cargo_discovery_path),
        cargo_discovery_sha256=_sha256_file(
            cargo_discovery_path, "Cargo discovery executable"
        ),
        rustc_path=str(rustc_path),
        rustc_binary_sha256=_sha256_file(rustc_path, "rustc executable"),
        rustc_version=_probe_tool(
            (str(rustc_path), "-vV"), cwd=source_root, environment=probe_environment
        ),
        rustc_discovery_path=str(rustc_discovery_path),
        rustc_discovery_sha256=_sha256_file(
            rustc_discovery_path, "rustc discovery executable"
        ),
        sysroot=str(sysroot),
        target_triple=target_triple,
        target_library_directory=str(target_library_directory),
        target_library_manifest_sha256=target_library_manifest,
        native_cfg=_probe_tool(
            (str(rustc_path), "--print", "cfg", "-Ctarget-cpu=native"),
            cwd=source_root,
            environment=probe_environment,
        ),
        target_cpus=_probe_tool(
            (str(rustc_path), "--print", "target-cpus"),
            cwd=source_root,
            environment=probe_environment,
        ),
        linker_path=str(linker_path),
        linker_binary_sha256=_sha256_file(linker_path, "rust-lld executable"),
        sdk_identity_sha256=sdk_identity,
        sdk_environment=sdk_environment,
    )


def _toolchain_entries(toolchain: _NativeToolchain) -> tuple[tuple[str, bytes], ...]:
    record = {
        field: getattr(toolchain, field)
        for field in toolchain.__dataclass_fields__
        if field != "sdk_environment"
    }
    record["sdk_environment"] = list(toolchain.sdk_environment)
    return (("@toolchain", json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")),)


def _native_build_plan(source_root: Path | None = None) -> _NativeBuildPlan:
    _reject_build_control_environment()
    root = (_SOURCE_ROOT if source_root is None else source_root).resolve()
    native_build_inputs = _native_build_input_manifest(root)
    cargo_home = _cargo_home()
    _reject_cargo_configuration(root, cargo_home)
    toolchain = _discover_native_toolchain(root)
    feature_arguments = (
        ()
        if not _NATIVE_CARGO_FEATURES
        else ("--features", ",".join(_NATIVE_CARGO_FEATURES))
    )
    command = (
        toolchain.cargo_path,
        "build",
        "--release",
        "--locked",
        "--offline",
        *feature_arguments,
        "--manifest-path",
        str(root / "Cargo.toml"),
        "--target",
        toolchain.target_triple,
    )
    environment = _base_tool_environment()
    environment.update(dict(toolchain.sdk_environment))
    environment.update(
        {
            "CARGO_BUILD_RUSTC": toolchain.rustc_path,
            "CARGO_ENCODED_RUSTFLAGS": "\x1f".join(
                (
                    "-Ctarget-cpu=native",
                    f"-Clinker={toolchain.linker_path}",
                    f"--remap-path-prefix={root}=/humanbrain/native-src",
                )
            ),
            "CARGO_HOME": str(cargo_home),
            "CARGO_INCREMENTAL": "0",
            "CARGO_NET_OFFLINE": "true",
            "CARGO_TARGET_DIR": _TARGET_ROOT_PLACEHOLDER,
            "CARGO_TERM_COLOR": "never",
            "RUSTC": toolchain.rustc_path,
            "RUSTC_WRAPPER": "",
            "RUSTC_WORKSPACE_WRAPPER": "",
            "SOURCE_DATE_EPOCH": "0",
        }
    )
    ordered_environment = tuple(sorted(environment.items()))
    command_bytes = json.dumps(command, separators=(",", ":")).encode("utf-8")
    environment_bytes = json.dumps(
        ordered_environment, separators=(",", ":")
    ).encode("utf-8")
    toolchain_entries = _toolchain_entries(toolchain)
    toolchain_identity = _framed_sha256(
        "humanbrain-native-toolchain-v2", toolchain_entries
    )
    digest = _framed_sha256(
        "humanbrain-native-cache-v5-vendored-target-libraries",
        (
            *native_build_inputs,
            ("@plan/cwd", str(root).encode("utf-8")),
            ("@plan/command", command_bytes),
            ("@plan/environment", environment_bytes),
            *toolchain_entries,
        ),
    )
    return _NativeBuildPlan(
        source_digest=digest,
        command=command,
        cwd=root,
        environment=ordered_environment,
        target_triple=toolchain.target_triple,
        library_relative_path=(
            Path(toolchain.target_triple) / "release" / _library_filename()
        ),
        toolchain_identity_sha256=toolchain_identity,
        sdk_identity_sha256=toolchain.sdk_identity_sha256,
    )


def _source_digest(source_root: Path | None = None) -> str:
    return _native_build_plan(source_root).source_digest


def procedural_runtime_source_manifest(
    repository_root: Path = _REPOSITORY_ROOT,
) -> tuple[dict[str, str | int], ...]:
    payloads = _procedural_runtime_source_payloads(repository_root)
    records: list[dict[str, str | int]] = []
    for relative_path, payload in payloads:
        records.append(
            {
                "path": relative_path,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return tuple(records)


def procedural_runtime_source_bundle_sha256(
    repository_root: Path = _REPOSITORY_ROOT,
) -> str:
    return _framed_sha256(
        "humanbrain-procedural-runtime-source-v1",
        _procedural_runtime_source_payloads(repository_root),
    )


def _procedural_runtime_source_paths(repository_root: Path) -> tuple[str, ...]:
    native_root = repository_root / "src" / "humanbrain" / "_native_src"
    native_paths = (
        f"src/humanbrain/_native_src/{relative_path}"
        for relative_path, _path in _native_build_input_paths(native_root)
    )
    return tuple(dict.fromkeys((*_PROCEDURAL_RUNTIME_SOURCE_PATHS, *native_paths)))


def _procedural_runtime_source_payloads(
    repository_root: Path,
) -> tuple[tuple[str, bytes], ...]:
    payloads: list[tuple[str, bytes]] = []
    aggregate = 0
    for relative_path in _procedural_runtime_source_paths(repository_root):
        payload = _bounded_file_payload(
            repository_root / relative_path,
            "procedural runtime source",
            _MAX_NATIVE_SOURCE_FILE_BYTES,
        )
        aggregate += len(payload)
        if aggregate > _MAX_NATIVE_SOURCE_BUNDLE_BYTES:
            raise NativeBuildError("procedural runtime sources exceed the aggregate byte limit")
        payloads.append((relative_path, payload))
    return tuple(payloads)


def _library_filename() -> str:
    if os.name == "nt":
        return "humanbrain_kernel.dll"
    if platform.system() == "Darwin":
        return "libhumanbrain_kernel.dylib"
    return "libhumanbrain_kernel.so"


def _default_cache_root() -> Path:
    override = os.environ.get("HUMANBRAIN_NATIVE_CACHE")
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "humanbrain-native"


def _identity_manifest_path(library_path: Path) -> Path:
    return library_path.with_name(library_path.name + ".identity.json")


def _hash_native_library(
    library_path: Path,
    source_digest: str,
) -> NativeLibraryIdentity:
    digest = hashlib.sha256()
    binary_size = 0
    try:
        with library_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                binary_size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise NativeBuildError("native library bytes cannot be read") from error
    if binary_size == 0:
        raise NativeBuildError("native library is empty")
    return NativeLibraryIdentity(
        source_digest=source_digest,
        binary_size=binary_size,
        binary_sha256=digest.hexdigest(),
    )


def _identity_bytes(identity: NativeLibraryIdentity) -> bytes:
    return json.dumps(
        identity.to_record(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _publish_identity_manifest(
    library_path: Path,
    identity: NativeLibraryIdentity,
) -> None:
    manifest = _identity_manifest_path(library_path)
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_identity_bytes(identity))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest)
    except OSError as error:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise NativeBuildError("native identity manifest cannot be published") from error


def _attested_library_identity(
    library_path: Path,
    source_digest: str,
) -> NativeLibraryIdentity:
    resolved = library_path.resolve()
    if (
        len(resolved.parents) < 3
        or resolved.parent.name != "release"
        or resolved.parents[2].name != source_digest
    ):
        raise NativeBuildError("native library is outside its source-addressed cache")
    manifest_path = _identity_manifest_path(resolved)
    try:
        with manifest_path.open("rb") as handle:
            raw_manifest = handle.read(4097)
    except OSError as error:
        raise NativeBuildError("native library has no readable integrity manifest") from error
    if len(raw_manifest) > 4096:
        raise NativeBuildError("native library integrity manifest is oversized")
    try:
        record = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeBuildError("native library integrity manifest is invalid") from error
    if not isinstance(record, dict) or frozenset(record) != frozenset(
        {"source_digest", "binary_size", "binary_sha256"}
    ):
        raise NativeBuildError("native library integrity manifest fields differ")
    if (
        not isinstance(record["source_digest"], str)
        or record["source_digest"] != source_digest
        or type(record["binary_size"]) is not int
        or record["binary_size"] <= 0
        or not isinstance(record["binary_sha256"], str)
        or len(record["binary_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in record["binary_sha256"])
    ):
        raise NativeBuildError("native library integrity manifest values differ")
    expected = NativeLibraryIdentity(
        source_digest=record["source_digest"],
        binary_size=record["binary_size"],
        binary_sha256=record["binary_sha256"],
    )
    if raw_manifest != _identity_bytes(expected):
        raise NativeBuildError("native library integrity manifest is not canonical")
    actual = _hash_native_library(resolved, source_digest)
    if actual != expected:
        raise NativeBuildError("native library integrity check failed")
    return actual


def build_native(cache_root: str | Path | None = None, *, force: bool = False) -> NativeBuildInfo:
    """Build the Rust kernel into a content-addressed cache and return its path."""
    plan = _native_build_plan()
    source_digest = plan.source_digest
    root = (
        Path(cache_root) if cache_root is not None else _default_cache_root()
    ).resolve()
    target_root = root / source_digest
    library_path = target_root / plan.library_relative_path
    if library_path.is_file() and not force:
        _attested_library_identity(library_path, source_digest)
        return NativeBuildInfo(source_digest, library_path, plan)

    target_root.mkdir(parents=True, exist_ok=True)
    environment = plan.materialize_environment(target_root)
    try:
        completed = subprocess.run(
            list(plan.command),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            cwd=plan.cwd,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise NativeBuildError("Rust Cargo is required to build the human-cell kernel") from error
    if completed.returncode != 0 or not library_path.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip() or "Cargo produced no library"
        raise NativeBuildError(f"native human-cell kernel build failed: {detail}")
    identity = _hash_native_library(library_path, source_digest)
    _publish_identity_manifest(library_path, identity)
    _attested_library_identity(library_path, source_digest)
    return NativeBuildInfo(source_digest, library_path, plan)


def _load_library() -> ctypes.CDLL:
    global _LIBRARY, _LIBRARY_PATH, _LOADED_BUILD_PLAN
    if _LIBRARY is not None:
        return _LIBRARY
    with _BUILD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        build = build_native()
        identity_before_load = _attested_library_identity(
            build.library_path,
            build.source_digest,
        )
        try:
            library = ctypes.CDLL(str(build.library_path))
        except OSError as error:
            message = f"could not load native cell kernel: {build.library_path}"
            raise NativeBuildError(message) from error

        library.humanbrain_kernel_abi_version.argtypes = []
        library.humanbrain_kernel_abi_version.restype = ctypes.c_uint32
        if library.humanbrain_kernel_abi_version() != 5:
            raise NativeBuildError("native human-cell kernel ABI mismatch")

        pointer_args = [
            ctypes.POINTER(_GlifConfig),
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _U8_POINTER,
        ]
        library.humanbrain_glif_step.argtypes = pointer_args
        library.humanbrain_glif_step.restype = ctypes.c_size_t
        recurrent_args = [
            ctypes.POINTER(_GlifConfig),
            ctypes.POINTER(_RecurrentConfig),
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _U8_POINTER,
            _I32_POINTER,
            _U8_POINTER,
            _SIZE_POINTER,
            _U64_POINTER,
            _U64_POINTER,
            _U64_POINTER,
        ]
        for name in (
            "humanbrain_observe_token",
            "humanbrain_observe_token_serial",
            "humanbrain_observe_token_parallel",
        ):
            function = getattr(library, name)
            function.argtypes = recurrent_args
            function.restype = ctypes.c_size_t
        library.humanbrain_observe_token_reference_v1.argtypes = [
            *recurrent_args[:-8],
            _F64_POINTER,
            _I32_POINTER,
        ]
        library.humanbrain_observe_token_reference_v1.restype = ctypes.c_size_t
        library.humanbrain_three_factor_update.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
        ]
        library.humanbrain_three_factor_update.restype = ctypes.c_size_t
        library.humanbrain_sparse_readout_current.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            _F64_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            _F64_POINTER,
        ]
        library.humanbrain_sparse_readout_current.restype = None
        library.humanbrain_sparse_three_factor_update.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            _F64_POINTER,  # per-candidate apposition decays (output_count*candidates)
            ctypes.c_double,
            ctypes.c_double,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _U32_POINTER,
            _F64_POINTER,
            _F64_POINTER,
            ctypes.c_double,
        ]
        library.humanbrain_sparse_three_factor_update.restype = ctypes.c_size_t
        library.humanbrain_sparse_structural_turnover.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            _I32_POINTER,
            _F64_POINTER,
            _I32_POINTER,
            _F64_POINTER,
            _U32_POINTER,
            _F64_POINTER,
        ]
        library.humanbrain_sparse_structural_turnover.restype = ctypes.c_uint32
        if (
            _attested_library_identity(build.library_path, build.source_digest)
            != identity_before_load
        ):
            raise NativeBuildError("native library bytes changed while loading")
        _LIBRARY = library
        _LIBRARY_PATH = build.library_path
        _LOADED_BUILD_PLAN = build.build_plan
        return library


def loaded_build_info() -> NativeBuildInfo:
    """Return the source digest and loaded library path, building if necessary."""
    loaded_library_identity()
    assert _LIBRARY_PATH is not None and _LOADED_BUILD_PLAN is not None
    return NativeBuildInfo(
        _LOADED_BUILD_PLAN.source_digest,
        _LIBRARY_PATH,
        _LOADED_BUILD_PLAN,
    )


def loaded_library_identity() -> NativeLibraryIdentity:
    """Rehash and verify the exact source-addressed library loaded by this process."""
    _load_library()
    if _LIBRARY_PATH is None or type(_LOADED_BUILD_PLAN) is not _NativeBuildPlan:
        raise NativeBuildError("loaded native build plan identity is unavailable")
    return _attested_library_identity(
        _LIBRARY_PATH,
        _LOADED_BUILD_PLAN.source_digest,
    )


def _f64(array: np.ndarray) -> _F64_POINTER:
    return array.ctypes.data_as(_F64_POINTER)


def _i32(array: np.ndarray) -> _I32_POINTER:
    return array.ctypes.data_as(_I32_POINTER)


def _u32(array: np.ndarray) -> _U32_POINTER:
    return array.ctypes.data_as(_U32_POINTER)


def _require_native_array(
    array: np.ndarray,
    *,
    dtype: np.dtype[object],
    shape: tuple[int, ...],
    name: str,
) -> None:
    if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be a contiguous {dtype} array with shape {shape}")


def _finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_scalar(name: str, value: float) -> float:
    scalar = _finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _unit_interval(name: str, value: float, *, open_interval: bool = False) -> float:
    scalar = _finite_scalar(name, value)
    valid = 0.0 < scalar < 1.0 if open_interval else 0.0 <= scalar <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return scalar


def _glif_config(population: AdaptiveGlifPopulation) -> _GlifConfig:
    parameters = population.parameters
    constants = population.constants
    return _GlifConfig(
        population.cell_count,
        population.state.after_spike_currents_pa.shape[1],
        parameters.spike_cut_steps,
        parameters.leak_ns,
        constants.voltage_step_scale,
        constants.threshold_beta_scale,
        constants.threshold_exp_decay,
        constants.threshold_voltage_factor,
        constants.threshold_beta_factor,
        constants.spike_threshold_decay,
        parameters.threshold_inf_relative_mv,
        parameters.voltage_reset_scale,
        parameters.voltage_reset_offset_mv,
        constants.spike_cut_threshold_decay,
        parameters.threshold_spike_increment_mv,
    )


class NativeGlifKernel:
    """Native step binding over one population's checkpointable NumPy state."""

    def __init__(self, population: AdaptiveGlifPopulation) -> None:
        self._library = _load_library()
        self._population = population
        self._config = _glif_config(population)
        self._spike_mask = np.empty(population.cell_count, dtype=np.bool_)

    @property
    def spike_mask(self) -> np.ndarray:
        return self._spike_mask

    def step(self, current: np.ndarray) -> np.ndarray:
        population = self._population
        state = population.state
        self._library.humanbrain_glif_step(
            ctypes.byref(self._config),
            _f64(population.constants.after_spike_decay),
            _f64(population.constants.after_spike_cut_decay),
            _f64(population.constants.after_spike_amplitudes_pa),
            _f64(current),
            _f64(state.voltage_relative_mv),
            _f64(state.after_spike_currents_pa),
            _f64(state.threshold_spike_mv),
            _f64(state.threshold_voltage_mv),
            _i32(state.refractory_steps),
            self._spike_mask.ctypes.data_as(_U8_POINTER),
        )
        return self._spike_mask


class NativeRecurrentKernel:
    """One-call native token simulation over sparse recurrent connectivity."""

    backend = "cpu"

    def __init__(
        self,
        population: AdaptiveGlifPopulation,
        *,
        token_steps: int,
        pulse_steps: int,
        input_fanout: int,
        recurrent_fanout: int,
        input_current_pa: float,
        recurrent_decay: float,
        eligibility_decays: np.ndarray,
        afferent_targets: np.ndarray,
        recurrent_targets: np.ndarray,
        recurrent_weights_pa: np.ndarray,
        synaptic_current_pa: np.ndarray,
        eligibility: np.ndarray,
        host_mirror: str = "full",
        token_spike_counts: np.ndarray | None = None,
        mirror_token_spike_counts: bool = False,
        mirror_boundary_voltage: bool = False,
    ) -> None:
        if host_mirror not in {"full", "eligibility"}:
            raise ValueError("host mirror must be 'full' or 'eligibility'")
        self._library = _load_library()
        self._population = population
        cells = population.cell_count
        # R0-b family A: the u32 accumulator is host-live kernel state on CPU;
        # it is cleared and folded inside every observe call. None (legacy
        # callers) leaves the struct pointer null and skips accumulation.
        if token_spike_counts is not None:
            _require_native_array(
                token_spike_counts,
                dtype=np.dtype(np.uint32),
                shape=(cells,),
                name="token spike counts",
            )
        self._token_spike_counts = token_spike_counts
        decays = np.ascontiguousarray(eligibility_decays, dtype=np.float64)
        if decays.ndim != 1 or decays.size < 1:
            raise ValueError("eligibility decays must be a nonempty 1-D array")
        if not np.all(np.isfinite(decays)):
            raise ValueError("eligibility decays must be finite")
        trace_count = int(decays.size)
        expected_afferents = (afferent_targets.shape[0], input_fanout)
        _require_native_array(
            afferent_targets,
            dtype=np.dtype(np.int32),
            shape=expected_afferents,
            name="afferent targets",
        )
        _require_native_array(
            recurrent_targets,
            dtype=np.dtype(np.int32),
            shape=(cells, recurrent_fanout),
            name="recurrent targets",
        )
        _require_native_array(
            recurrent_weights_pa,
            dtype=np.dtype(np.float64),
            shape=(cells, recurrent_fanout),
            name="recurrent weights",
        )
        for array, name in (
            (synaptic_current_pa, "synaptic current"),
        ):
            _require_native_array(
                array,
                dtype=np.dtype(np.float64),
                shape=(cells,),
                name=name,
            )
        # The eligibility buffer is the FULL trace bank: trace_count rows of
        # cells, C-order (row k = trace k), delivered contiguously.
        _require_native_array(
            eligibility,
            dtype=np.dtype(np.float64),
            shape=(trace_count * cells,),
            name="eligibility",
        )
        if np.any(afferent_targets < 0) or np.any(afferent_targets >= cells):
            raise ValueError("afferent targets must index reservoir cells")
        if np.any(recurrent_targets < 0) or np.any(recurrent_targets >= cells):
            raise ValueError("recurrent targets must index reservoir cells")
        self._glif_config = _glif_config(population)
        # Anchor the decay array's lifetime: the config stores its address.
        self._eligibility_decays = decays
        self._recurrent_config = _RecurrentConfig(
            token_steps,
            pulse_steps,
            input_fanout,
            recurrent_fanout,
            input_current_pa,
            recurrent_decay,
            trace_count,
            _f64(decays),
            (
                _u32(token_spike_counts)
                if token_spike_counts is not None
                else ctypes.cast(None, _U32_POINTER)
            ),
        )
        self._afferent_targets = afferent_targets
        self._recurrent_targets = recurrent_targets
        self._recurrent_weights_pa = recurrent_weights_pa
        self._synaptic_current_pa = synaptic_current_pa
        self._eligibility = eligibility
        self._input_scratch = np.empty(input_fanout, dtype=np.float64)
        self._spike_mask = population._native_kernel.spike_mask
        self._active_indices = np.empty(cells, dtype=np.int32)
        self._active_mask = np.empty(cells, dtype=np.bool_)
        self._active_count = ctypes.c_size_t()
        self._scan_nanoseconds = ctypes.c_uint64()
        self._frontier_build_nanoseconds = ctypes.c_uint64()
        self._step_nanoseconds = ctypes.c_uint64()
        self._reference_step_current: np.ndarray | None = None
        self._reference_spike_indices: np.ndarray | None = None

    @property
    def device_name(self) -> str:
        return platform.processor() or platform.machine()

    def synchronize_from_host(self) -> None:
        """CPU arrays are the live native state, so no synchronization is needed."""

    def synchronize_to_host(self) -> None:
        """CPU arrays are already the canonical mutable state."""

    def synchronize_structure_from_host(self) -> None:
        """CPU topology and weight arrays are already live native state."""

    def observe(self, token_id: int) -> int:
        return self._observe_with(self._library.humanbrain_observe_token, token_id)

    def _observe_serial(self, token_id: int) -> int:
        return self._observe_with(self._library.humanbrain_observe_token_serial, token_id)

    def _observe_parallel(self, token_id: int) -> int:
        return self._observe_with(self._library.humanbrain_observe_token_parallel, token_id)

    @property
    def last_active_cells(self) -> int:
        return int(self._active_count.value)

    @property
    def last_frontier_profile_ns(self) -> tuple[int, int, int]:
        return (
            int(self._scan_nanoseconds.value),
            int(self._frontier_build_nanoseconds.value),
            int(self._step_nanoseconds.value),
        )

    def _observe_with(self, function: object, token_id: int) -> int:
        population = self._population
        state = population.state
        afferents = self._afferent_targets[token_id]
        return int(
            function(
                ctypes.byref(self._glif_config),
                ctypes.byref(self._recurrent_config),
                _f64(population.constants.after_spike_decay),
                _f64(population.constants.after_spike_cut_decay),
                _f64(population.constants.after_spike_amplitudes_pa),
                _i32(afferents),
                _i32(self._recurrent_targets),
                _f64(self._recurrent_weights_pa),
                _f64(state.voltage_relative_mv),
                _f64(state.after_spike_currents_pa),
                _f64(state.threshold_spike_mv),
                _f64(state.threshold_voltage_mv),
                _i32(state.refractory_steps),
                _f64(self._synaptic_current_pa),
                _f64(self._eligibility),
                _f64(self._input_scratch),
                self._spike_mask.ctypes.data_as(_U8_POINTER),
                _i32(self._active_indices),
                self._active_mask.ctypes.data_as(_U8_POINTER),
                ctypes.byref(self._active_count),
                ctypes.byref(self._scan_nanoseconds),
                ctypes.byref(self._frontier_build_nanoseconds),
                ctypes.byref(self._step_nanoseconds),
            )
        )

    def _observe_reference_v1(self, token_id: int) -> int:
        population = self._population
        state = population.state
        afferents = self._afferent_targets[token_id]
        cells = population.cell_count
        if self._reference_step_current is None:
            self._reference_step_current = np.empty(cells, dtype=np.float64)
            self._reference_spike_indices = np.empty(cells, dtype=np.int32)
        assert self._reference_spike_indices is not None
        return int(
            self._library.humanbrain_observe_token_reference_v1(
                ctypes.byref(self._glif_config),
                ctypes.byref(self._recurrent_config),
                _f64(population.constants.after_spike_decay),
                _f64(population.constants.after_spike_cut_decay),
                _f64(population.constants.after_spike_amplitudes_pa),
                _i32(afferents),
                _i32(self._recurrent_targets),
                _f64(self._recurrent_weights_pa),
                _f64(state.voltage_relative_mv),
                _f64(state.after_spike_currents_pa),
                _f64(state.threshold_spike_mv),
                _f64(state.threshold_voltage_mv),
                _i32(state.refractory_steps),
                _f64(self._synaptic_current_pa),
                _f64(self._eligibility),
                _f64(self._reference_step_current),
                _i32(self._reference_spike_indices),
            )
        )


class NativeThreeFactorLearner:
    """Allocation-free native plasticity over checkpointable synapse arrays."""

    def __init__(
        self,
        *,
        eligibility: np.ndarray,
        output_heterogeneity_pa: np.ndarray,
        readout_synapses_pa: np.ndarray,
        pacemaker_synapses_pa: np.ndarray,
    ) -> None:
        self._library = _load_library()
        output_cells, reservoir_cells = readout_synapses_pa.shape
        _require_native_array(
            eligibility,
            dtype=np.dtype(np.float64),
            shape=(reservoir_cells,),
            name="eligibility",
        )
        _require_native_array(
            output_heterogeneity_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="output heterogeneity",
        )
        _require_native_array(
            readout_synapses_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells, reservoir_cells),
            name="readout synapses",
        )
        _require_native_array(
            pacemaker_synapses_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="pacemaker synapses",
        )
        self._eligibility = eligibility
        self._output_heterogeneity_pa = output_heterogeneity_pa
        self._readout_synapses_pa = readout_synapses_pa
        self._pacemaker_synapses_pa = pacemaker_synapses_pa
        self._output_cells = output_cells
        self._reservoir_cells = reservoir_cells

    def update(
        self,
        target_token_id: int,
        learning_rate_pa: float,
        competitive_margin_pa: float,
    ) -> int | None:
        """Update a margin-violating competitor, or return ``None`` when satisfied."""
        target = int(target_token_id)
        if not 0 <= target < self._output_cells:
            raise ValueError("target token outside output-cell population")
        if learning_rate_pa <= 0.0:
            raise ValueError("local learning rate must be positive")
        if competitive_margin_pa <= 0.0:
            raise ValueError("competitive margin must be positive")
        winner = int(
            self._library.humanbrain_three_factor_update(
                self._reservoir_cells,
                self._output_cells,
                target,
                learning_rate_pa,
                competitive_margin_pa,
                _f64(self._eligibility),
                _f64(self._output_heterogeneity_pa),
                _f64(self._readout_synapses_pa),
                _f64(self._pacemaker_synapses_pa),
            )
        )
        if winner == self._output_cells:
            return None
        if winner > self._output_cells:
            raise RuntimeError("native local learner rejected its array dimensions")
        return winner


class NativeSparseCellReadout:
    """Native current summation over explicit bounded output-cell contacts."""

    def __init__(
        self,
        *,
        eligibility: np.ndarray,
        output_heterogeneity_pa: np.ndarray,
        potential_sources: np.ndarray,
        candidate_decays: np.ndarray,
        apposition_eligibility: np.ndarray,
        contact_sources: np.ndarray,
        contact_weights_pa: np.ndarray,
        contact_ages: np.ndarray,
        pacemaker_synapses_pa: np.ndarray,
        output_activity_ema: np.ndarray,
    ) -> None:
        self._library = _load_library()
        output_cells, fan_in = contact_sources.shape
        if potential_sources.shape[0] != output_cells:
            raise ValueError("potential appositions need one row per output cell")
        candidate_count = potential_sources.shape[1]
        if apposition_eligibility.shape != (output_cells, candidate_count):
            raise ValueError("potential sources and apposition traces must have equal shape")
        if contact_weights_pa.shape != (output_cells, fan_in):
            raise ValueError("sparse contact sources and weights must have equal shape")
        reservoir_cells = eligibility.shape[0]
        _require_native_array(
            eligibility,
            dtype=np.dtype(np.float64),
            shape=(reservoir_cells,),
            name="eligibility",
        )
        _require_native_array(
            output_heterogeneity_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="output heterogeneity",
        )
        _require_native_array(
            potential_sources,
            dtype=np.dtype(np.int32),
            shape=(output_cells, candidate_count),
            name="potential apposition sources",
        )
        _require_native_array(
            candidate_decays,
            dtype=np.dtype(np.float64),
            shape=(output_cells, candidate_count),
            name="candidate apposition decays",
        )
        _require_native_array(
            apposition_eligibility,
            dtype=np.dtype(np.float64),
            shape=(output_cells, candidate_count),
            name="apposition eligibility",
        )
        _require_native_array(
            contact_sources,
            dtype=np.dtype(np.int32),
            shape=(output_cells, fan_in),
            name="sparse contact sources",
        )
        _require_native_array(
            contact_weights_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells, fan_in),
            name="sparse contact weights",
        )
        _require_native_array(
            contact_ages,
            dtype=np.dtype(np.uint32),
            shape=(output_cells, fan_in),
            name="sparse contact ages",
        )
        _require_native_array(
            output_activity_ema,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="output activity",
        )
        _require_native_array(
            pacemaker_synapses_pa,
            dtype=np.dtype(np.float64),
            shape=(output_cells,),
            name="pacemaker synapses",
        )
        if np.any(potential_sources < 0) or np.any(potential_sources >= reservoir_cells):
            raise ValueError("potential appositions must index reservoir cells")
        if np.any(contact_sources < 0) or np.any(contact_sources >= reservoir_cells):
            raise ValueError("sparse contacts must index reservoir cells")
        self._eligibility = eligibility
        self._output_heterogeneity_pa = output_heterogeneity_pa
        self._potential_sources = potential_sources
        self._candidate_decays = candidate_decays
        self._apposition_eligibility = apposition_eligibility
        self._contact_sources = contact_sources
        self._contact_weights_pa = contact_weights_pa
        self._contact_ages = contact_ages
        self._pacemaker_synapses_pa = pacemaker_synapses_pa
        self._output_activity_ema = output_activity_ema
        self._current_pa = np.empty(output_cells, dtype=np.float64)
        self._output_cells = output_cells
        self._reservoir_cells = reservoir_cells
        self._fan_in = fan_in
        self._candidate_count = candidate_count

    def current(self, base_current_pa: float) -> np.ndarray:
        base_current_pa = _finite_scalar("base current", base_current_pa)
        self._library.humanbrain_sparse_readout_current(
            self._reservoir_cells,
            self._output_cells,
            self._fan_in,
            base_current_pa,
            _f64(self._eligibility),
            _f64(self._output_heterogeneity_pa),
            _i32(self._contact_sources),
            _f64(self._contact_weights_pa),
            _f64(self._pacemaker_synapses_pa),
            _f64(self._current_pa),
        )
        return self._current_pa.copy()

    def update(
        self,
        *,
        target: int,
        winner: int,
        learning_rate_pa: float,
        maximum_weight_pa: float,
        target_post_gate: float,
        winner_post_gate: float,
        activity_decay: float,
    ) -> int | None:
        target = int(target)
        winner = int(winner)
        if not 0 <= target < self._output_cells:
            raise ValueError("target token outside output-cell population")
        if not 0 <= winner < self._output_cells:
            raise ValueError("winner outside output-cell population")
        learning_rate_pa = _positive_scalar("learning rate", learning_rate_pa)
        maximum_weight_pa = _positive_scalar("maximum weight", maximum_weight_pa)
        target_post_gate = _unit_interval("target post gate", target_post_gate)
        winner_post_gate = _unit_interval("winner post gate", winner_post_gate)
        activity_decay = _unit_interval("activity decay", activity_decay, open_interval=True)
        result = int(
            self._library.humanbrain_sparse_three_factor_update(
                self._reservoir_cells,
                self._output_cells,
                self._candidate_count,
                self._fan_in,
                target,
                winner,
                learning_rate_pa,
                maximum_weight_pa,
                _f64(self._candidate_decays),
                target_post_gate,
                winner_post_gate,
                _f64(self._eligibility),
                _i32(self._potential_sources),
                _f64(self._apposition_eligibility),
                _i32(self._contact_sources),
                _f64(self._contact_weights_pa),
                _u32(self._contact_ages),
                _f64(self._pacemaker_synapses_pa),
                _f64(self._output_activity_ema),
                activity_decay,
            )
        )
        if result == self._output_cells:
            return None
        if result > self._output_cells:
            raise RuntimeError("native sparse learner rejected its dimensions")
        return result

    def turnover(
        self,
        *,
        output: int,
        structural_event: int,
        random_seed: int,
        minimum_age: int,
        turnover_probability: float,
        learning_rate_pa: float,
        maximum_weight_pa: float,
        homeostatic_rate: float,
    ) -> tuple[bool, bool]:
        output = int(output)
        structural_event = int(structural_event)
        minimum_age = int(minimum_age)
        if not 0 <= output < self._output_cells:
            raise ValueError("turnover output outside output-cell population")
        if structural_event < 0:
            raise ValueError("structural event must be nonnegative")
        if minimum_age < 0 or minimum_age > np.iinfo(np.uint32).max:
            raise ValueError("minimum age must fit uint32")
        turnover_probability = _unit_interval("turnover probability", turnover_probability)
        learning_rate_pa = _positive_scalar("learning rate", learning_rate_pa)
        maximum_weight_pa = _positive_scalar("maximum weight", maximum_weight_pa)
        homeostatic_rate = _finite_scalar("homeostatic rate", homeostatic_rate)
        if homeostatic_rate < 0.0:
            raise ValueError("homeostatic rate must be nonnegative")
        result = int(
            self._library.humanbrain_sparse_structural_turnover(
                self._output_cells,
                self._candidate_count,
                self._fan_in,
                output,
                structural_event,
                int(random_seed) & ((1 << 64) - 1),
                minimum_age,
                turnover_probability,
                learning_rate_pa,
                maximum_weight_pa,
                homeostatic_rate,
                _i32(self._potential_sources),
                _f64(self._apposition_eligibility),
                _i32(self._contact_sources),
                _f64(self._contact_weights_pa),
                _u32(self._contact_ages),
                _f64(self._output_activity_ema),
            )
        )
        if result > 3:
            raise RuntimeError("native sparse turnover rejected its dimensions")
        return bool(result & 1), bool(result & 2)


def recurrent_kernel(*args: object, **kwargs: object) -> NativeRecurrentKernel:
    """Create the fastest exact backend suitable for this circuit size."""
    population = args[0] if args else kwargs.get("population")
    if population is None:
        raise TypeError("recurrent kernel requires a cell population")
    from humanbrain.cuda import CudaRecurrentKernel, should_use_cuda

    if should_use_cuda(population.cell_count):
        return CudaRecurrentKernel(*args, **kwargs)  # type: ignore[return-value]
    return NativeRecurrentKernel(*args, **kwargs)
