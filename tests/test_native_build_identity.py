from __future__ import annotations

from pathlib import Path

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
