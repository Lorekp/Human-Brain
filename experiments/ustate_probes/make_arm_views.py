#!/usr/bin/env python3
"""Build per-arm VIEW artifacts for parallel probe execution (Change Note 1 §3).

The canonical logged_states.npz files are NEVER modified. For every frozen arm
this script writes experiments/<artifact>/arm_<arm>/logged_states.npz holding
ONLY that arm's arrays (bit-identical bytes, same dtype/shape, sha256s matching
the canonical manifest) plus a filtered copy of the manifest. The UNCHANGED
r0c probes.py is then invoked once per view directory; results are merged
verbatim by merge_results.py. Purely an execution-topology change: identical
arrays, seeds, decoders, controls.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_views(artifact_dir: Path) -> list[Path]:
    manifest_path = artifact_dir / "logged_states_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    z = np.load(artifact_dir / "logged_states.npz")
    made = []
    for arm_meta in manifest["arms"]:
        arm = arm_meta["arm"]
        prefix = f"arm_{arm}_n"
        keys = [k for k in z.files if k.startswith(prefix)]
        assert keys, f"no arrays for {arm}"
        arrays = {k: z[k] for k in sorted(keys)}
        sub = artifact_dir / f"view_{arm}"
        sub.mkdir(parents=True, exist_ok=True)
        npz_path = sub / "logged_states.npz"
        tmp = npz_path.with_name("tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(npz_path)
        sub_manifest = dict(manifest)
        sub_manifest["arms"] = [arm_meta]
        sub_manifest["npz_arrays"] = {
            k: v for k, v in manifest["npz_arrays"].items() if k.startswith(prefix)
        }
        # verify bit-identity against canonical hashes
        for k, arr in arrays.items():
            h = sha256_bytes(np.ascontiguousarray(arr).tobytes())
            assert h == manifest["npz_arrays"][k]["sha256"], f"hash mismatch {k}"
        (sub / "logged_states_manifest.json").write_bytes(
            json.dumps(sub_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        made.append(sub)
        print(f"built view {sub} ({len(arrays)} arrays)")
    return made


def main() -> None:
    targets = [
        REPO / "experiments" / "ustate_probes",
        REPO / "experiments" / "ustate_probes_concept",
    ]
    views: list[Path] = []
    for t in targets:
        if (t / "logged_states.npz").exists():
            views.extend(make_views(t))
    print(json.dumps([str(v) for v in views], indent=1))


if __name__ == "__main__":
    main()
