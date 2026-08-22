from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from humanbrain import native


def _write_build_tree(root: Path) -> None:
    (root / "src" / "nested").mkdir(parents=True)
    (root / "target" / "release").mkdir(parents=True)
    (root / "Cargo.toml").write_bytes(b"[package]\nname='identity-test'\n")
    (root / "Cargo.lock").write_bytes(b"# frozen lock\n")
    (root / "build.rs").write_bytes(b"fn main() {}\n")
    (root / "src" / "lib.rs").write_bytes(b"mod nested;\n")
    (root / "src" / "nested" / "procedural.rs").write_bytes(
        b"pub fn frontier() -> u64 { 1 }\n"
    )
    (root / "target" / "release" / "cached.rs").write_bytes(b"not a build input\n")


def test_native_source_digest_covers_every_recursive_cargo_input(tmp_path: Path) -> None:
    source_root = tmp_path / "native-source"
    _write_build_tree(source_root)

    manifest = native._native_build_input_manifest(source_root)
    first_digest = native._source_digest(source_root)

    assert tuple(name for name, _payload in manifest) == (
        "Cargo.lock",
        "Cargo.toml",
        "build.rs",
        "src/lib.rs",
        "src/nested/procedural.rs",
    )
    assert len(first_digest) == 64

    (source_root / "src" / "nested" / "procedural.rs").write_bytes(
        b"pub fn frontier() -> u64 { 2 }\n"
    )
    second_digest = native._source_digest(source_root)

    assert second_digest != first_digest
    assert (tmp_path / first_digest) != (tmp_path / second_digest)
    assert native.NativeLibraryIdentity(
        source_digest=second_digest,
        binary_size=1,
        binary_sha256="0" * 64,
    ).source_digest == second_digest


def test_native_digest_framing_separates_path_and_payload_boundaries() -> None:
    namespace = "native-boundary-regression"

    assert native._framed_sha256(namespace, (("ab", b"c"),)) != native._framed_sha256(
        namespace,
        (("a", b"bc"),),
    )
    assert native._framed_sha256(namespace, (("a", b"bc"),)) != native._framed_sha256(
        namespace,
        (("a", b"b"), ("c", b"")),
    )


def test_loaded_native_identity_uses_the_complete_build_input_digest() -> None:
    identity = native.loaded_library_identity()

    assert identity.source_digest == native._source_digest()


def test_native_build_plan_rejects_ambient_encoded_flags_before_cache_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-feature=-avx2")
    monkeypatch.setenv("RUSTFLAGS", "-Ctarget-feature=+avx2")

    with pytest.raises(native.NativeBuildError, match="CARGO_ENCODED_RUSTFLAGS"):
        native._native_build_plan()


def test_native_build_plan_rejects_hierarchical_cargo_configuration(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "native-source"
    _write_build_tree(source_root)
    (source_root / ".cargo").mkdir()
    (source_root / ".cargo" / "config.toml").write_text(
        '[build]\ntarget = "wasm32-unknown-unknown"\n',
        encoding="utf-8",
    )

    with pytest.raises(native.NativeBuildError, match="Cargo configuration"):
        native._native_build_plan(source_root)


def test_native_build_plan_freezes_target_flags_tools_linker_and_environment() -> None:
    plan = native._native_build_plan()
    environment = dict(plan.environment)

    assert plan.cwd == native._SOURCE_ROOT.resolve()
    assert plan.command[-2:] == ("--target", plan.target_triple)
    assert plan.library_relative_path == (
        Path(plan.target_triple) / "release" / native._library_filename()
    )
    assert Path(plan.command[0]).is_absolute()
    assert Path(environment["RUSTC"]).is_absolute()
    flags = environment["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert flags[0] == "-Ctarget-cpu=native"
    assert any(flag.startswith("-Clinker=") for flag in flags)
    assert "RUSTFLAGS" not in environment
    assert environment["RUSTC_WRAPPER"] == ""
    assert environment["RUSTC_WORKSPACE_WRAPPER"] == ""
    assert environment["CARGO_TARGET_DIR"] == native._TARGET_ROOT_PLACEHOLDER
    assert len(plan.toolchain_identity_sha256) == 64
    assert len(plan.sdk_identity_sha256) == 64


def test_tool_target_cpu_linker_and_sdk_drift_rotate_the_native_cache_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "native-source"
    _write_build_tree(source_root)
    toolchain = native._discover_native_toolchain(source_root)
    monkeypatch.setattr(native, "_discover_native_toolchain", lambda _root: toolchain)
    first = native._native_build_plan(source_root)
    changes = (
        {"cargo_binary_sha256": "0" * 64},
        {"rustc_binary_sha256": "0" * 64},
        {"target_triple": toolchain.target_triple + "-changed"},
        {"native_cfg": toolchain.native_cfg + "\ntarget_feature=\"synthetic\""},
        {"linker_binary_sha256": "0" * 64},
        {"sdk_identity_sha256": "0" * 64},
    )
    for fields in changes:
        changed = replace(toolchain, **fields)
        monkeypatch.setattr(native, "_discover_native_toolchain", lambda _root: changed)
        assert native._native_build_plan(source_root).source_digest != first.source_digest


def test_ambient_wrappers_are_neutralized_instead_of_changing_the_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = native._native_build_plan()
    monkeypatch.setenv("RUSTC_WRAPPER", r"C:\untrusted\wrapper.exe")
    monkeypatch.setenv("RUSTC_WORKSPACE_WRAPPER", r"C:\untrusted\workspace.exe")

    second = native._native_build_plan()

    assert second.source_digest == first.source_digest
    assert dict(second.environment)["RUSTC_WRAPPER"] == ""
    assert dict(second.environment)["RUSTC_WORKSPACE_WRAPPER"] == ""


def test_build_uses_the_same_frozen_plan_for_cache_address_and_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = replace(native._native_build_plan(), source_digest="a" * 64)
    calls: list[dict[str, object]] = []

    def fake_run(command: object, **kwargs: object) -> object:
        calls.append({"command": command, **kwargs})
        target_root = tmp_path / plan.source_digest
        library_path = target_root / plan.library_relative_path
        library_path.parent.mkdir(parents=True, exist_ok=True)
        library_path.write_bytes(b"synthetic-native-binary")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(native, "_native_build_plan", lambda: plan)
    monkeypatch.setattr(native.subprocess, "run", fake_run)

    built = native.build_native(tmp_path, force=True)

    expected_environment = plan.materialize_environment(tmp_path / plan.source_digest)
    assert built.source_digest == plan.source_digest
    assert built.library_path == tmp_path / plan.source_digest / plan.library_relative_path
    assert calls == [
        {
            "command": list(plan.command),
            "check": False,
            "capture_output": True,
            "text": True,
            "env": expected_environment,
            "cwd": plan.cwd,
            "timeout": 600,
        }
    ]


def test_loaded_identity_reuses_the_loaded_plan_without_tool_probes_and_rehashes_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = replace(native._native_build_plan(), source_digest="b" * 64)
    library_path = tmp_path / plan.source_digest / plan.library_relative_path
    library_path.parent.mkdir(parents=True)
    library_path.write_bytes(b"loaded-native-bytes")
    identity = native._hash_native_library(library_path, plan.source_digest)
    native._publish_identity_manifest(library_path, identity)
    monkeypatch.setattr(native, "_LIBRARY", object())
    monkeypatch.setattr(native, "_LIBRARY_PATH", library_path)
    monkeypatch.setattr(native, "_LOADED_BUILD_PLAN", plan)

    def forbidden_plan(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("loaded identity must not probe or rebuild the native plan")

    monkeypatch.setattr(native, "_native_build_plan", forbidden_plan)

    assert native.loaded_library_identity() == identity
    assert native.loaded_library_identity() == identity

    library_path.write_bytes(b"changed-native-bytes")
    with pytest.raises(native.NativeBuildError, match="integrity"):
        native.loaded_library_identity()
