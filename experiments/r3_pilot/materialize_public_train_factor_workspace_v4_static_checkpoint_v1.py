"""Offline recovery tool for the banked factor-workspace V4 checkpoint.

This is not an experiment and produces no new scientific result.  It executes
the already-frozen learner bytes, accepts only the already-published checkpoint
identity, and writes the otherwise-unavailable learned arrays into a static
artifact.  Runtime code and tests must use the static capsule instead of
importing this module or the executable learner.
"""

from __future__ import annotations

import importlib
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOURCE_PATH = REPO_ROOT / "experiments/r3_pilot/public_train_factor_workspace_v4.py"
TEST_PATH = REPO_ROOT / "tests/test_public_train_factor_workspace_v4.py"
ARTIFACT_PATH = (
    REPO_ROOT
    / "experiments/r3_pilot/public_train_factor_workspace_v4_static_checkpoint_v1.npz"
)

BANKED_COMMIT = "2bee9d0120cf4d957e9aa426410490a333180139"
EXPECTED_SOURCE_SHA256 = "f91a51db6b2b236571647c6b5027a713167d22f68d0463ee55afdcd9fda92ecf"
EXPECTED_TEST_SHA256 = "23fbd16abe55d5ac808bd548ce6e5c9fe893cdbc1ff4fc5147c355761773e9f0"
EXPECTED_PROTOCOL_COMMIT = "c9da529"
EXPECTED_PROTOCOL_SHA256 = "83335e57fb896e9ba509a3f7f34e1bba39e7394d700c80813a702c4b34cc0f12"
EXPECTED_PROTOCOL_SOURCE_SHA256 = (
    "9df77077bbce8ba206fc6dcf8a1df4d282005e983199020be94b5b979db533b3"
)
EXPECTED_PROTOCOL_TEST_SHA256 = (
    "8980d6b8750a46fb9397dd29263dc513fa9e7a6e746f3afbf79e2a7fb896a1b4"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "63fd83a72f23aeb759bd72d032bb9b7c44adfcdead8557992602279b99847578"
)
EXPECTED_RESULT_SHA256 = (
    "96ab4c088382271e2208bbc71d8f308497cd5a560d601529e124938dda73cc4e"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_record(array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": tuple(int(value) for value in contiguous.shape),
        "sha256": sha256(memoryview(contiguous).cast("B")).hexdigest(),
    }


def main() -> None:
    if _file_sha256(SOURCE_PATH) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("banked factor-workspace V4 source changed")
    if _file_sha256(TEST_PATH) != EXPECTED_TEST_SHA256:
        raise RuntimeError("banked factor-workspace V4 test changed")
    if ARTIFACT_PATH.exists():
        raise RuntimeError("static factor checkpoint already exists")

    learner = importlib.import_module(
        "experiments.r3_pilot.public_train_factor_workspace_v4"
    )
    if (
        learner.RESULT.status != learner.PASS_STATUS
        or learner.RESULT.result_sha256 != EXPECTED_RESULT_SHA256
        or learner.RESULT.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256
        or learner.V4_PROTOCOL_COMMIT != EXPECTED_PROTOCOL_COMMIT
        or learner.V4_PROTOCOL_SHA256 != EXPECTED_PROTOCOL_SHA256
        or learner.V4_PROTOCOL_SOURCE_SHA256 != EXPECTED_PROTOCOL_SOURCE_SHA256
        or learner.V4_PROTOCOL_TEST_SHA256 != EXPECTED_PROTOCOL_TEST_SHA256
    ):
        raise RuntimeError("banked factor-workspace V4 result binding changed")

    # The banked module did not retain its local checkpoint object.  Reproduce
    # the exact frozen pass only as artifact recovery, then require the already
    # published checkpoint identity before any bytes are written.
    learner._STATE_ARRAY_ALLOCATION_COUNT = 0
    excitatory, inhibitory = learner._allocate_treatment_state(
        learner.ELECTRICAL_PREFLIGHT
    )
    telemetry = learner._fit_once(excitatory, inhibitory)
    checkpoint = learner._freeze_checkpoint(excitatory, inhibitory, telemetry)
    if checkpoint.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("recovered checkpoint differs from the banked identity")

    arrays = {
        "excitatory_ticks": np.ascontiguousarray(checkpoint._excitatory, dtype=np.uint8),
        "inhibitory_ticks": np.ascontiguousarray(checkpoint._inhibitory, dtype=np.uint8),
    }
    metadata = {
        "format": "public-train-factor-workspace-v4-static-checkpoint-v1",
        "purpose": "artifact-recovery-only-not-new-scientific-evidence",
        "banked_commit": BANKED_COMMIT,
        "banked_source_sha256": EXPECTED_SOURCE_SHA256,
        "banked_test_sha256": EXPECTED_TEST_SHA256,
        "banked_protocol_commit": EXPECTED_PROTOCOL_COMMIT,
        "banked_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "banked_protocol_source_sha256": EXPECTED_PROTOCOL_SOURCE_SHA256,
        "banked_protocol_test_sha256": EXPECTED_PROTOCOL_TEST_SHA256,
        "banked_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "banked_result_sha256": EXPECTED_RESULT_SHA256,
        "banked_checkpoint_metadata": checkpoint._metadata,
        "arrays": {name: _array_record(array) for name, array in sorted(arrays.items())},
        "array_count": 2,
        "state_bytes": sum(array.nbytes for array in arrays.values()),
        "normal_runtime_imports_executable_learner": False,
        "normal_runtime_imports_target_protocol": False,
    }
    temporary = ARTIFACT_PATH.with_suffix(".tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(_canonical_json(metadata).decode("utf-8")),
            **arrays,
        )
    temporary.replace(ARTIFACT_PATH)
    print(
        json.dumps(
            {
                "artifact_sha256": _file_sha256(ARTIFACT_PATH),
                "artifact_bytes": ARTIFACT_PATH.stat().st_size,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
