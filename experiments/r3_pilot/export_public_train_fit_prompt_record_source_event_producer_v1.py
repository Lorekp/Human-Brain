"""Offline-only exporter for the FIT prompt-record source-event selector.

The banked V2 annotation is imported only inside ``reproduce_artifact``.  The
resulting artifact contains a physical candidate vocabulary and frozen neural
state, never the V2 target objects, input roots, or preassembled source bundles.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_fit_prompt_record_source_event_producer_v1.npz"
)

INPUT_BANK_COMMIT = "eae4944c22a748300c1210ce4b4201d21cdb0d55"
INPUT_SOURCE_SHA256 = (
    "cef67a95765e268425e23cc9c270f8cdc161723584135c7b29e208b8ec9a4433"
)
INPUT_ARTIFACT_SHA256 = (
    "0e90318d1c1d3bd365f9812fb3c43aa5f0806dbfa5a1e81cb527c8e7f2c20dcf"
)
INPUT_CONTENT_ROOT = (
    "a7ff53818cf4e83c823e6b23f5547e0e7ee1fd30ed73dd136e3672063ea7aca8"
)
V1_BANK_COMMIT = "e2cc14a05354c7f80b8dea34234d45c8596f3209"
V1_SOURCE_SHA256 = (
    "86fa1e23e2d9c2bdab3751353932168ec95e561af64f35535221eec5b0f7bbc8"
)
V1_CAPSULE_SHA256 = (
    "f0e1bf69b3943c7c09fa7f7322d5a91d3246db651e8ce898030b54ae6393235b"
)
V1_STOP_SHA256 = (
    "39f02d04249bbb6cb52c2d36f76a72f705ce5871f4899568377784243325e7de"
)
V2_BANK_COMMIT = "56b9c73cc8515bd9b5377b26effe758d75857750"
V2_SOURCE_SHA256 = (
    "f41f5b904edbe6aa4ecf29f45117eb86aa7dfd135a11c721199ecb0a6542f92c"
)
V2_CAPSULE_SHA256 = (
    "75d9fcd9abcfeba051fe1c0caade3fabdcbbb140b956350d3e2dd5d82ec0fc9f"
)
V2_TARGET_SET_SHA256 = (
    "be3c88adc730d58ece81cddab97f90484f346194f2621ec8dbe18c3c0475d02b"
)
STATIC_BANK_COMMIT = "d2376c7dc75314ae285ffb98edeae3e704b63d49"
STATIC_SOURCE_SHA256 = (
    "0af68609e46afd65e8dc50c656f7ad22919a08329045d3068f8a457813870c89"
)
STATIC_ARTIFACT_SHA256 = (
    "42ca5251b7a045ca98da4e2b008a82d8c89d42f229f6b3eb670ade7a30919711"
)
STATIC_PROTOCOL_SHA256 = (
    "792b517f364f075c46fef51671ea26b8117eb0f78d37d3b1d0cdee6867c26e61"
)
STATIC_CHECKPOINT_SHA256 = (
    "aee5d7da81f11a2d71bd1a6cb37383f8364a5e5427fe5055ef15c11790dba290"
)

_FORMAT = "public-train-fit-prompt-record-source-event-producer-v1"
_NAMESPACE = b"public-train-fit-prompt-record-source-event-producer-v1\0"
_CONTENT_NAMESPACE = _NAMESPACE + b"artifact-content\0"
_PROTOCOL_NAMESPACE = _NAMESPACE + b"protocol\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_BUNDLE_NAMESPACE = _NAMESPACE + b"source-bundle\0"


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
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


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + _canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_frame(name: str, array: np.ndarray) -> bytes:
    value = np.ascontiguousarray(array)
    return b"".join(
        (
            name.encode("ascii"),
            b"\0",
            value.dtype.str.encode("ascii"),
            b"\0",
            _canonical_json(value.shape),
            b"\0",
            value.tobytes(order="C"),
        )
    )


def _checkpoint_payload(
    header: Mapping[str, object],
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> bytes:
    return b"".join(
        (
            _canonical_json(dict(header)),
            b"\0",
            _array_frame("excitatory", excitatory),
            _array_frame("inhibitory", inhibitory),
        )
    )


def reproduce_artifact(path: Path = ARTIFACT_PATH) -> tuple[str, str, int]:
    """Materialize one authenticated artifact from banked offline authorities."""

    from experiments.r3_pilot import (
        public_train_fit_occurrence_factor_annotation_capsule_v1 as annotation_v1,
    )
    from experiments.r3_pilot import (
        public_train_fit_occurrence_factor_annotation_capsule_v2 as annotation_v2,
    )
    from experiments.r3_pilot import (
        public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as static,
    )
    from experiments.r3_pilot import (
        public_train_list_int_to_int_prompt_input_capsule_v1 as inputs,
    )
    from humanbrain.resting_glif_race import CompiledRestingGlifRace

    base = Path(__file__).resolve().parent
    checks = (
        _file_sha256(base / "public_train_list_int_to_int_prompt_input_capsule_v1.py")
        == INPUT_SOURCE_SHA256,
        inputs.ARTIFACT_SHA256 == INPUT_ARTIFACT_SHA256,
        inputs.CONTENT_ROOT == INPUT_CONTENT_ROOT,
        _file_sha256(
            base / "public_train_fit_occurrence_factor_annotation_capsule_v1.py"
        )
        == V1_SOURCE_SHA256,
        annotation_v1.CAPSULE.capsule_sha256 == V1_CAPSULE_SHA256,
        annotation_v1.STOP_LEDGER.sha256 == V1_STOP_SHA256,
        _file_sha256(
            base / "public_train_fit_occurrence_factor_annotation_capsule_v2.py"
        )
        == V2_SOURCE_SHA256,
        annotation_v2.CAPSULE.capsule_sha256 == V2_CAPSULE_SHA256,
        annotation_v2.CAPSULE.target_set_sha256 == V2_TARGET_SET_SHA256,
        _file_sha256(
            base
            / "public_train_fit_occurrence_factor_static_checkpoint_capsule_v1.py"
        )
        == STATIC_SOURCE_SHA256,
        static.ARTIFACT_SHA256 == STATIC_ARTIFACT_SHA256,
        static.BANKED_PROTOCOL_SHA256 == STATIC_PROTOCOL_SHA256,
        static.BANKED_CHECKPOINT_SHA256 == STATIC_CHECKPOINT_SHA256,
    )
    if not all(checks):
        raise RuntimeError("source-event producer dependency changed")

    def source_feature(source: object) -> tuple[object, ...]:
        if isinstance(source, annotation_v2.RawSourceTarget):
            observation = source.observation
            return (
                "raw",
                observation.assembly_cells,
                observation.direction,
                observation.cue_slot,
                observation.selected_slot,
            )
        if isinstance(source, annotation_v2.LiteralSourceTarget):
            return (
                "literal",
                source.typed_value_event_cells,
                source.typed_value_type_cells,
            )
        raise TypeError("unknown V2 source target")

    literal_type_cells = tuple(static.STATIC_CHECKPOINT._protocol.type_registry[0])

    def candidate_features(frozen: object) -> tuple[tuple[object, ...], ...]:
        assemblies = {
            tuple(int(cell) for cell in row["assembly_cells"])
            for row in frozen.raw_span_records
        }
        candidates: set[tuple[object, ...]] = set()
        for assembly in assemblies:
            for direction in ("NEXT", "PREV"):
                for cue_slot in range(4):
                    selected_slot = (
                        (cue_slot + 1) % 4
                        if direction == "NEXT"
                        else (cue_slot - 1) % 4
                    )
                    candidates.add(
                        (
                            "raw",
                            assembly,
                            direction,
                            cue_slot,
                            selected_slot,
                        )
                    )
        candidates.update(
            (
                "literal",
                tuple(int(cell) for cell in row["typed_output_active_cells"]),
                literal_type_cells,
            )
            for row in frozen.typed_literal_records
        )
        return tuple(sorted(candidates, key=_canonical_json))

    positives = {
        source_feature(source)
        for target in annotation_v2.CAPSULE.targets
        for source in target.sources
    }
    if (
        len(positives) != 26
        or sum(feature[0] == "raw" for feature in positives) != 13
        or sum(feature[0] == "literal" for feature in positives) != 13
    ):
        raise RuntimeError("positive physical source vocabulary changed")

    state = inputs.load_input_capsule()
    label_sets: dict[tuple[object, ...], set[bool]] = {}
    candidate_appearance_count = 0
    selected_appearance_count = 0
    per_input_candidates: list[tuple[tuple[object, ...], ...]] = []
    for audit in annotation_v1.CAPSULE.fit_identity_audits:
        frozen = state.lookup(audit.input_root_sha256)
        candidates = candidate_features(frozen)
        per_input_candidates.append(candidates)
        candidate_appearance_count += len(candidates)
        selected_appearance_count += sum(feature in positives for feature in candidates)
        for feature in candidates:
            label_sets.setdefault(feature, set()).add(feature in positives)
    conflict_count = sum(len(labels) != 1 for labels in label_sets.values())
    missing_positive_count = len(positives - set(label_sets))
    if (
        len(per_input_candidates) != 9
        or candidate_appearance_count != 4_986
        or selected_appearance_count != 77
        or len(label_sets) != 941
        or conflict_count != 0
        or missing_positive_count != 0
    ):
        raise RuntimeError("physical source candidate identifiability changed")

    feature_registry = tuple(sorted(label_sets, key=_canonical_json))
    labels = tuple(next(iter(label_sets[feature])) for feature in feature_registry)
    excitatory = np.zeros((1, len(feature_registry)), dtype=np.uint8)
    inhibitory = np.zeros_like(excitatory)
    for index, selected in enumerate(labels):
        if selected:
            excitatory[0, index] = 4
        else:
            inhibitory[0, index] = 4

    kernel = CompiledRestingGlifRace.compile(step_count=30)

    def learned_selected(index: int, *, cell_gain: float = 1.0) -> bool:
        score = int(excitatory[0, index]) - int(inhibitory[0, index])
        current = np.asarray([5_000.0 * cell_gain if score >= 1 else 0.0])
        return bool(kernel.accumulated_spike_mask(current)[0])

    fit_exact_count = sum(
        learned_selected(index) == label for index, label in enumerate(labels)
    )
    teacher_off_ready_count = sum(
        bool(kernel.accumulated_spike_mask(np.asarray([0.0]))[0])
        for _feature in feature_registry
    )
    cell_off_ready_count = sum(
        learned_selected(index, cell_gain=0.0)
        for index, label in enumerate(labels)
        if label
    )
    if (
        fit_exact_count != len(feature_registry)
        or teacher_off_ready_count != 0
        or cell_off_ready_count != 0
    ):
        raise RuntimeError("physical source selector controls failed")

    protocol_record = {
        "candidate_rule": "unordered-unique-raw-assembly-x-8-plus-typed-literals",
        "feature_registry": feature_registry,
        "literal_type_cells": literal_type_cells,
        "fit_identity_count": 9,
        "candidate_appearance_count": candidate_appearance_count,
        "unique_feature_count": len(feature_registry),
        "positive_feature_count": sum(labels),
        "negative_feature_count": len(labels) - sum(labels),
        "query_target_conflict_count": conflict_count,
        "missing_positive_feature_count": missing_positive_count,
        "input_roots_are_neural_evidence": False,
        "record_order_is_neural_evidence": False,
        "teacher_targets_are_runtime_evidence": False,
    }
    protocol_sha = _digest(_PROTOCOL_NAMESPACE, protocol_record)
    checkpoint_header = {
        "schema": _FORMAT,
        "protocol_sha256": protocol_sha,
        "array_shape": excitatory.shape,
        "array_dtype": excitatory.dtype.str,
        "tick_cap": 4,
        "output_current_pa_hex": 5_000.0.hex(),
        "output_steps": 30,
        "fit_pass_count": 1,
        "direct_physical_candidate_encoder": True,
        "task_field_count": 0,
        "family_field_count": 0,
        "program_field_count": 0,
        "desired_field_count": 0,
        "input_root_field_count": 0,
        "source_order_field_count": 0,
        "heldout_input_count": 0,
        "dev_input_count": 0,
        "m0_input_count": 0,
        "c1_input_count": 0,
        "path2_input_count": 0,
        "g0_input_count": 0,
        "protected_input_count": 0,
        "panel_spend_count": 0,
    }
    checkpoint_sha = sha256(
        _checkpoint_payload(checkpoint_header, excitatory, inhibitory)
    ).hexdigest()

    feature_index = {feature: index for index, feature in enumerate(feature_registry)}
    bundle_roots: list[str] = []
    static_view_roots: list[str] = []
    static_certificate_roots: list[str] = []
    selected_total = 0
    for candidates in per_input_candidates:
        selected = tuple(
            feature
            for feature in candidates
            if learned_selected(feature_index[feature])
        )
        events = tuple(
            static.RawPhysicalSourceEvent(
                feature[1], feature[2], feature[3], feature[4]
            )
            if feature[0] == "raw"
            else static.LiteralPhysicalSourceEvent(feature[1], feature[2])
            for feature in selected
        )
        certificate = static.infer_occurrence_factors(events)
        view = static.require_occurrence_factors(certificate)
        selected_total += len(events)
        bundle_roots.append(
            _digest(
                _BUNDLE_NAMESPACE,
                tuple(event.neural_record() for event in events),
            )
        )
        static_view_roots.append(view.sha256)
        static_certificate_roots.append(certificate.sha256)
    if selected_total != 77:
        raise RuntimeError("selected source-event replay count changed")

    metadata: dict[str, object] = {
        "format": _FORMAT,
        "purpose": "same-fit-physical-source-event-learning-only",
        "input_bank_commit": INPUT_BANK_COMMIT,
        "input_source_sha256": INPUT_SOURCE_SHA256,
        "input_artifact_sha256": INPUT_ARTIFACT_SHA256,
        "input_content_root": INPUT_CONTENT_ROOT,
        "v1_bank_commit": V1_BANK_COMMIT,
        "v1_source_sha256": V1_SOURCE_SHA256,
        "v1_capsule_sha256": V1_CAPSULE_SHA256,
        "v1_stop_sha256": V1_STOP_SHA256,
        "v2_bank_commit": V2_BANK_COMMIT,
        "v2_source_sha256": V2_SOURCE_SHA256,
        "v2_capsule_sha256": V2_CAPSULE_SHA256,
        "v2_target_set_sha256": V2_TARGET_SET_SHA256,
        "static_bank_commit": STATIC_BANK_COMMIT,
        "static_source_sha256": STATIC_SOURCE_SHA256,
        "static_artifact_sha256": STATIC_ARTIFACT_SHA256,
        "static_protocol_sha256": STATIC_PROTOCOL_SHA256,
        "static_checkpoint_sha256": STATIC_CHECKPOINT_SHA256,
        "protocol_sha256": protocol_sha,
        "checkpoint_header": checkpoint_header,
        "checkpoint_sha256": checkpoint_sha,
        "array_shape": excitatory.shape,
        "array_dtype": excitatory.dtype.str,
        "excitatory_sha256": sha256(excitatory.tobytes(order="C")).hexdigest(),
        "inhibitory_sha256": sha256(inhibitory.tobytes(order="C")).hexdigest(),
        "feature_registry": feature_registry,
        "literal_type_cells": literal_type_cells,
        "fit_identity_count": 9,
        "fit_appearance_count": 12,
        "candidate_appearance_count": candidate_appearance_count,
        "unique_feature_count": len(feature_registry),
        "positive_feature_count": sum(labels),
        "raw_positive_feature_count": 13,
        "literal_positive_feature_count": 13,
        "negative_feature_count": len(labels) - sum(labels),
        "selected_source_appearance_count": selected_appearance_count,
        "query_target_conflict_count": conflict_count,
        "missing_positive_feature_count": missing_positive_count,
        "fit_exact_count": fit_exact_count,
        "teacher_off_ready_count": teacher_off_ready_count,
        "cell_off_ready_count": cell_off_ready_count,
        "committed_excitatory_ticks": int(excitatory.sum()),
        "committed_inhibitory_ticks": int(inhibitory.sum()),
        "selector_physical_query_count": (
            len(feature_registry) * 2 + candidate_appearance_count
        ),
        "banked_source_bundle_sha256s": tuple(bundle_roots),
        "banked_static_view_sha256s": tuple(static_view_roots),
        "banked_static_certificate_sha256s": tuple(static_certificate_roots),
        "normal_runtime_imports_input_capsule": False,
        "normal_runtime_imports_annotation_v1": False,
        "normal_runtime_imports_annotation_v2": False,
        "normal_runtime_imports_executable_learner": False,
        "normal_runtime_teacher_target_read_count": 0,
        "normal_runtime_fit_count": 0,
        "input_roots_embedded": False,
        "fit_source_bundles_embedded": False,
        "record_order_used_for_selection": False,
        "heldout_input_count": 0,
        "dev_input_count": 0,
        "m0_input_count": 0,
        "c1_input_count": 0,
        "path2_input_count": 0,
        "g0_input_count": 0,
        "protected_input_count": 0,
        "panel_spend_count": 0,
    }
    metadata["artifact_content_root"] = _digest(_CONTENT_NAMESPACE, metadata)
    stream = BytesIO()
    np.savez_compressed(
        stream,
        metadata=np.asarray(_canonical_json(metadata).decode("ascii")),
        excitatory=np.ascontiguousarray(excitatory),
        inhibitory=np.ascontiguousarray(inhibitory),
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
