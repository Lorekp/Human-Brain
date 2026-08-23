"""Refresh the frozen Gate-3 slice protocol's source binding.

The Gate-3 protocol JSON pins, among other things:

- ``source_binding.files``: the exact path/size/sha256 manifest of every
  production source file (Rust kernel, Python driver, vendored crates);
- ``source_binding.bundle_sha256``: the framed digest over those payloads;
- ``source_binding.limits.actual_aggregate_bytes``: total manifest size.

``tests/test_procedural_runtime.py::test_gate3_slice_protocol_keeps_material_timing_frozen``
asserts all three against live recomputation, so ANY change to a bound source
file (e.g. optimizing the Rust kernel) requires regenerating this binding.
Everything else in the protocol — material-timing freeze, forbidden runs,
determinism/exactness gate text — is untouched by this script.

After refreshing the JSON, the test's pinned ``sha256(protocol_bytes)``
constant must be updated with the new whole-file digest, which this script
prints. Run from the repository/worktree root:

    python tools/refresh_gate3_source_binding.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROTOCOL_PATH = ROOT / "benchmarks" / "procedural_runtime_gate3_slice_protocol_v1.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the refreshed protocol JSON (default: dry run)",
    )
    args = parser.parse_args(argv)

    import humanbrain.native as native_module

    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    old_bundle = protocol["source_binding"]["bundle_sha256"]
    old_file_sha = hashlib.sha256(protocol_bytes).hexdigest()

    manifest = native_module.procedural_runtime_source_manifest()
    bundle_sha = native_module.procedural_runtime_source_bundle_sha256()
    aggregate = sum(int(item["size"]) for item in manifest)

    limits = protocol["source_binding"]["limits"]
    if aggregate != limits["maximum_aggregate_bytes"]:
        if aggregate > int(limits["maximum_aggregate_bytes"]):
            print(
                f"REFUSING: aggregate {aggregate} exceeds declared maximum "
                f"{limits['maximum_aggregate_bytes']}",
                file=sys.stderr,
            )
            return 1
    protocol["source_binding"]["files"] = list(manifest)
    protocol["source_binding"]["bundle_sha256"] = bundle_sha
    limits["actual_aggregate_bytes"] = aggregate

    new_bytes = (
        json.dumps(protocol, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
        if protocol_bytes.endswith(b"\n")
        else json.dumps(protocol, indent=2, ensure_ascii=False).encode("utf-8")
    )
    new_file_sha = hashlib.sha256(new_bytes).hexdigest()

    print(f"bundle_sha256: {old_bundle} -> {bundle_sha}")
    print(f"protocol sha256: {old_file_sha} -> {new_file_sha}")
    print(f"aggregate bytes: {sum(int(i['size']) for i in json.loads(protocol_bytes)['source_binding']['files'])} -> {aggregate}")

    if args.apply:
        PROTOCOL_PATH.write_bytes(new_bytes)
        print(f"wrote {PROTOCOL_PATH}")
        print(
            "Update tests/test_procedural_runtime.py "
            "test_gate3_slice_protocol_keeps_material_timing_frozen to pin "
            f'{new_file_sha}'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
