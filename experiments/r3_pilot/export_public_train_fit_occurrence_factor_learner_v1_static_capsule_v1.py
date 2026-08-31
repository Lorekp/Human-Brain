"""One-time recovery of the banked occurrence-factor learner checkpoint.

Importing this module is inert.  Only an explicit invocation imports the
target-bearing banked learner, authenticates its published roots, and writes
the immutable static payload consumed by the teacher-free runtime capsule.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

BANKED_COMMIT = "3ac16dc22caae8ce2ac14b82f8579494c0b30dc0"
BANKED_SOURCE_SHA256 = (
    "e656dfdf4799800c6d6c633e5a7474d9414a22a1d69dc6ca08b847cf883e929e"
)
BANKED_TEST_SHA256 = (
    "0d63e23bff61e40860d16227186d6497a0f2ba40c49855dc682e43c79c8956e8"
)
BANKED_PROTOCOL_SHA256 = (
    "792b517f364f075c46fef51671ea26b8117eb0f78d37d3b1d0cdee6867c26e61"
)
BANKED_CHECKPOINT_SHA256 = (
    "aee5d7da81f11a2d71bd1a6cb37383f8364a5e5427fe5055ef15c11790dba290"
)
BANKED_SUMMARY_SHA256 = (
    "dce5727421810f9759d3d04a7d6501123eeccd8fe97bb3bc4db83cc2977ff4b5"
)
BANKED_PHYSICAL_WORK_SHA256 = (
    "a4bda9c6593ba7dd86b2e5b5bb8301c2d1e42c1e4011f6cf51ff11d5bb20f9c2"
)

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = (
    _ROOT / "experiments/r3_pilot/public_train_fit_occurrence_factor_learner_v1.py"
)
_TEST_PATH = _ROOT / "tests/test_public_train_fit_occurrence_factor_learner_v1.py"
ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_fit_occurrence_factor_static_checkpoint_capsule_v1.npz"
)
_FORMAT = "public-train-fit-occurrence-factor-learner-v1-static-capsule-v1"
_CONTENT_NAMESPACE = b"public-train-fit-occurrence-factor-static-v1-content\0"


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def reproduce_artifact(path: Path = ARTIFACT_PATH) -> tuple[str, str, int]:
    """Recover exact banked bytes once; this performs no new experiment."""

    from experiments.r3_pilot import (
        public_train_fit_occurrence_factor_learner_v1 as banked,
    )

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("static occurrence-factor artifact already exists")
    if (
        _file_sha256(_SOURCE_PATH) != BANKED_SOURCE_SHA256
        or _file_sha256(_TEST_PATH) != BANKED_TEST_SHA256
        or banked.PROTOCOL.protocol_sha256 != BANKED_PROTOCOL_SHA256
        or banked.BUILD_SUMMARY.checkpoint_sha256 != BANKED_CHECKPOINT_SHA256
        or banked.BUILD_SUMMARY.sha256 != BANKED_SUMMARY_SHA256
        or banked.BUILD_SUMMARY.physical_work_receipt_sha256
        != BANKED_PHYSICAL_WORK_SHA256
        or banked.STATIC_CONSUMER.checkpoint_sha256 != BANKED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("banked occurrence-factor authority changed")

    excitatory = banked.STATIC_CONSUMER._excitatory
    inhibitory = banked.STATIC_CONSUMER._inhibitory
    checkpoint_metadata = dict(banked.STATIC_CONSUMER._metadata)
    views = tuple(
        banked.require_public_train_fit_occurrence_factors_v1(
            banked.infer_public_train_fit_occurrence_factors_v1(bundle)
        )
        for bundle in banked.FIT_SOURCE_BUNDLES
    )
    certificates = tuple(
        banked.infer_public_train_fit_occurrence_factors_v1(bundle)
        for bundle in banked.FIT_SOURCE_BUNDLES
    )
    aggregate_counts = (
        sum(len(view.source_relays) for view in views),
        sum(len(view.edges) for view in views),
        sum(len(view.factor_slots) for view in views),
        sum(len(view.relations) for view in views),
        sum(len(view.domains) for view in views),
        sum(len(view.literal_copy_groups) for view in views),
        sum(len(view.signature_relays) for view in views),
    )
    if aggregate_counts != (77, 68, 164, 9, 18, 18, 18):
        raise RuntimeError("banked occurrence-factor parity cohort changed")

    metadata: dict[str, object] = {
        "format": _FORMAT,
        "purpose": "artifact-recovery-only-not-new-scientific-evidence",
        "banked_commit": BANKED_COMMIT,
        "banked_source_sha256": BANKED_SOURCE_SHA256,
        "banked_test_sha256": BANKED_TEST_SHA256,
        "banked_protocol_sha256": BANKED_PROTOCOL_SHA256,
        "banked_checkpoint_sha256": BANKED_CHECKPOINT_SHA256,
        "banked_summary_sha256": BANKED_SUMMARY_SHA256,
        "banked_physical_work_sha256": BANKED_PHYSICAL_WORK_SHA256,
        "checkpoint_metadata": checkpoint_metadata,
        "array_shape": excitatory.shape,
        "array_dtype": excitatory.dtype.str,
        "excitatory_sha256": sha256(excitatory.tobytes(order="C")).hexdigest(),
        "inhibitory_sha256": sha256(inhibitory.tobytes(order="C")).hexdigest(),
        "tick_cap": banked.TICK_CAP,
        "output_cell_count": banked.OUTPUT_CELL_COUNT,
        "input_cell_count": banked.INPUT_CELL_COUNT,
        "score_threshold_numerator": banked.SCORE_THRESHOLD_NUMERATOR,
        "score_threshold_denominator": banked.SCORE_THRESHOLD_DENOMINATOR,
        "output_current_pa_hex": banked.OUTPUT_CURRENT_PA.hex(),
        "output_steps": banked.OUTPUT_STEPS,
        "carrier_registry": banked.PROTOCOL.carrier_registry,
        "copy_group_registry": banked.PROTOCOL.copy_group_registry,
        "type_registry": banked.PROTOCOL.type_registry,
        "value_event_registry": banked.PROTOCOL.value_event_registry,
        "feature_registry": banked.PROTOCOL.feature_registry,
        "occupancy_registry": banked.OCCUPANCY_REGISTRY,
        "banked_view_sha256s": tuple(view.sha256 for view in views),
        "banked_certificate_sha256s": tuple(row.sha256 for row in certificates),
        "banked_aggregate_view_counts": aggregate_counts,
        "normal_runtime_imports_banked_learner": False,
        "normal_runtime_imports_annotation_v2": False,
        "normal_runtime_teacher_target_read_count": 0,
        "normal_runtime_fit_count": 0,
        "fit_source_bundles_embedded": False,
    }
    metadata["artifact_content_root"] = sha256(
        _CONTENT_NAMESPACE + _canonical_json(metadata)
    ).hexdigest()
    stream = BytesIO()
    np.savez_compressed(
        stream,
        metadata=np.asarray(_canonical_json(metadata).decode("ascii")),
        excitatory=np.ascontiguousarray(excitatory, dtype=np.uint8),
        inhibitory=np.ascontiguousarray(inhibitory, dtype=np.uint8),
    )
    payload = stream.getvalue()
    path.write_bytes(payload)
    return (
        sha256(payload).hexdigest(),
        str(metadata["artifact_content_root"]),
        len(payload),
    )


if __name__ == "__main__":
    print(reproduce_artifact())
