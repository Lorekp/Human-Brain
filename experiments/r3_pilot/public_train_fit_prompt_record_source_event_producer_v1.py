"""Target-free prompt-record to occurrence-factor source-event production.

The runtime enumerates generic selector geometries from actual physical prompt
records, classifies each candidate through one frozen GLIF membership head, and
passes the selected neutral event tuple through the banked static
occurrence-factor consumer.  No input root, record order, V2 target, or
preassembled FIT source bundle participates in inference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as occurrence_static,
)
from humanbrain.resting_glif_race import CompiledRestingGlifRace

SCHEMA = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_SOURCE_EVENT_PRODUCER_V1"
READY = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_SOURCE_EVENT_PRODUCER_V1_READY"
STOP_ZERO = "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_SOURCE_EVENT_PRODUCER_V1_ZERO"
STOP_AMBIGUOUS = (
    "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_SOURCE_EVENT_PRODUCER_V1_AMBIGUOUS"
)
STOP_UNSUPPORTED = (
    "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_SOURCE_EVENT_PRODUCER_V1_UNSUPPORTED"
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

ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_fit_prompt_record_source_event_producer_v1.npz"
)
ARTIFACT_SHA256 = "6945aae4edb456462163c5428fd303a461a9bfde5abafedac0c47c6d073e868c"
ARTIFACT_CONTENT_ROOT = (
    "a604e316ebd59a6c813408cfaa8245666d5d3387083e0c337142f7ffc45c0caf"
)
ARTIFACT_BYTES = 13_306
PROTOCOL_SHA256 = "023a2ab777abbd7159de9fa7c54cc29d31d53f2e2e1e9e4d0e36579a8c81ba8c"
CHECKPOINT_SHA256 = (
    "de6192dc2de627c19d9078a5c887cf84cf692aa17537539cdb36944d1ab9797e"
)

NORMAL_RUNTIME_IMPORTS_INPUT_CAPSULE = 0
NORMAL_RUNTIME_IMPORTS_ANNOTATION_V1 = 0
NORMAL_RUNTIME_IMPORTS_ANNOTATION_V2 = 0
NORMAL_RUNTIME_IMPORTS_EXECUTABLE_LEARNER = 0
NORMAL_RUNTIME_TEACHER_TARGET_READ_COUNT = 0
NORMAL_RUNTIME_FIT_COUNT = 0
HELDOUT_INPUT_COUNT = 0
DEV_INPUT_COUNT = 0
M0_INPUT_COUNT = 0
C1_INPUT_COUNT = 0
PATH2_INPUT_COUNT = 0
G0_INPUT_COUNT = 0
PROTECTED_INPUT_COUNT = 0
PANEL_SPEND_COUNT = 0

RawPhysicalSourceEvent = occurrence_static.RawPhysicalSourceEvent
LiteralPhysicalSourceEvent = occurrence_static.LiteralPhysicalSourceEvent
PhysicalSourceEvent = RawPhysicalSourceEvent | LiteralPhysicalSourceEvent
PhysicalFeature = tuple[object, ...]

_FORMAT = "public-train-fit-prompt-record-source-event-producer-v1"
_ARTIFACT_KEYS = {"metadata", "excitatory", "inhibitory"}
_NAMESPACE = b"public-train-fit-prompt-record-source-event-producer-v1\0"
_CONTENT_NAMESPACE = _NAMESPACE + b"artifact-content\0"
_CHECKPOINT_NAMESPACE = _NAMESPACE + b"checkpoint\0"
_BUNDLE_NAMESPACE = _NAMESPACE + b"source-bundle\0"
_CERTIFICATE_NAMESPACE = _NAMESPACE + b"certificate\0"
_REQUEST_NAMESPACE = _NAMESPACE + b"request\0"
_WORK_NAMESPACE = _NAMESPACE + b"physical-work\0"

_OUTPUT_CURRENT_PA = 5_000.0
_OUTPUT_STEPS = 30
_TICK_CAP = 4
_TICK_DTYPE = np.dtype(np.uint8)
_SPIKE_KERNEL = CompiledRestingGlifRace.compile(step_count=_OUTPUT_STEPS)

_SPAN_KEYS = {
    "start_token",
    "stop_token",
    "first_spike_step",
    "assembly_cells",
    "spike_cells",
}
_LITERAL_KEYS = {"canonical_token_index", "typed_output_active_cells"}
_SIGNATURE_KEYS = {
    "parameters",
    "result",
    "role_binding",
    "entrypoint_spelling_present",
    "parameter_spelling_present",
}
_EXPECTED_SIGNATURE = {
    "parameters": [
        {
            "role": "ARG0",
            "type": {
                "constructor": "List",
                "arguments": [{"constructor": "Int", "arguments": []}],
            },
        }
    ],
    "result": {"constructor": "Int", "arguments": []},
    "role_binding": "reverse-declaration-depth",
    "entrypoint_spelling_present": False,
    "parameter_spelling_present": False,
}


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


def _deep_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_deep_tuple(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _deep_tuple(item) for key, item in value.items()}
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _integer_cells(
    value: object,
    label: str,
    *,
    width: int,
    count: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a physical cell sequence")
    cells = tuple(_integer(cell, label) for cell in value)
    if (
        tuple(sorted(set(cells))) != cells
        or any(not 0 <= cell < width for cell in cells)
        or (count is not None and len(cells) != count)
        or not cells
    ):
        raise ValueError(f"{label} changed")
    return cells


def _normalize_prompt_records(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    if isinstance(span_records, (str, bytes)) or not isinstance(span_records, Sequence):
        raise TypeError("span records must be a sequence")
    if isinstance(literal_records, (str, bytes)) or not isinstance(
        literal_records, Sequence
    ):
        raise TypeError("literal records must be a sequence")
    assemblies: set[tuple[int, ...]] = set()
    for row in span_records:
        if not isinstance(row, Mapping) or set(row) != _SPAN_KEYS:
            raise ValueError("physical span record schema changed")
        start = _integer(row["start_token"], "span start")
        stop = _integer(row["stop_token"], "span stop")
        first_spike = _integer(row["first_spike_step"], "span first spike")
        assembly = _integer_cells(
            row["assembly_cells"], "span assembly", width=512, count=12
        )
        spikes = _integer_cells(row["spike_cells"], "span spikes", width=512, count=12)
        if start < 0 or stop <= start or first_spike < 0 or spikes != assembly:
            raise ValueError("physical span event changed")
        assemblies.add(assembly)
    if not assemblies:
        raise ValueError("physical prompt input has no span records")

    literal_cells: set[tuple[int, ...]] = set()
    for row in literal_records:
        if not isinstance(row, Mapping) or set(row) != _LITERAL_KEYS:
            raise ValueError("typed literal record schema changed")
        token_index = _integer(row["canonical_token_index"], "literal token index")
        active = _integer_cells(
            row["typed_output_active_cells"], "typed literal cells", width=96
        )
        if token_index < 0:
            raise ValueError("typed literal token index changed")
        literal_cells.add(active)

    if (
        not isinstance(signature_role_record, Mapping)
        or set(signature_role_record) != _SIGNATURE_KEYS
        or _plain(signature_role_record) != _EXPECTED_SIGNATURE
    ):
        raise ValueError("name-free List[Int]-to-Int signature changed")
    return tuple(sorted(assemblies)), tuple(sorted(literal_cells))


def _event_feature(event: PhysicalSourceEvent) -> PhysicalFeature:
    if type(event) is RawPhysicalSourceEvent:
        return (
            "raw",
            event.assembly_cells,
            event.direction,
            event.cue_slot,
            event.selected_slot,
        )
    if type(event) is LiteralPhysicalSourceEvent:
        return (
            "literal",
            event.typed_value_event_cells,
            event.typed_value_type_cells,
        )
    raise TypeError("candidate must be a neutral physical source event")


@dataclass(frozen=True, slots=True)
class FrozenSourceEventSelector:
    artifact_sha256: str
    content_root: str
    protocol_sha256: str
    checkpoint_sha256: str
    literal_type_cells: tuple[int, ...]
    _checkpoint_header: Mapping[str, object]
    _feature_registry: tuple[PhysicalFeature, ...]
    _excitatory: np.ndarray
    _inhibitory: np.ndarray
    _banked_source_bundle_sha256s: tuple[str, ...]
    _banked_static_view_sha256s: tuple[str, ...]

    def authenticate(self) -> None:
        if (
            self.artifact_sha256 != ARTIFACT_SHA256
            or self.content_root != ARTIFACT_CONTENT_ROOT
            or self.protocol_sha256 != PROTOCOL_SHA256
            or self.checkpoint_sha256 != CHECKPOINT_SHA256
            or len(self._feature_registry) != 941
            or len(set(self._feature_registry)) != 941
            or self._excitatory.shape != (1, 941)
            or self._inhibitory.shape != self._excitatory.shape
            or self._excitatory.dtype != _TICK_DTYPE
            or self._inhibitory.dtype != _TICK_DTYPE
            or self._excitatory.flags.writeable
            or self._inhibitory.flags.writeable
            or np.any(self._excitatory > _TICK_CAP)
            or np.any(self._inhibitory > _TICK_CAP)
            or int(self._excitatory.sum()) != 104
            or int(self._inhibitory.sum()) != 3_660
            or sha256(
                _checkpoint_payload(
                    self._checkpoint_header,
                    self._excitatory,
                    self._inhibitory,
                )
            ).hexdigest()
            != CHECKPOINT_SHA256
        ):
            raise RuntimeError("frozen prompt-record source-event selector changed")


def _immutable_array(array: np.ndarray) -> np.ndarray:
    result = np.frombuffer(array.tobytes(order="C"), dtype=_TICK_DTYPE).reshape(
        (1, 941)
    )
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def load_source_event_selector(path: Path = ARTIFACT_PATH) -> FrozenSourceEventSelector:
    path = Path(path).resolve()
    payload = path.read_bytes()
    if len(payload) != ARTIFACT_BYTES or sha256(payload).hexdigest() != ARTIFACT_SHA256:
        raise RuntimeError("source-event selector artifact bytes changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _ARTIFACT_KEYS:
            raise RuntimeError("source-event selector artifact members changed")
        metadata_array = archive["metadata"]
        excitatory = np.array(archive["excitatory"], copy=True, order="C")
        inhibitory = np.array(archive["inhibitory"], copy=True, order="C")
    if metadata_array.shape != () or metadata_array.dtype.kind != "U":
        raise RuntimeError("source-event selector metadata framing changed")
    metadata_text = str(metadata_array.item())
    metadata = json.loads(metadata_text)
    if _canonical_json(metadata).decode("ascii") != metadata_text:
        raise RuntimeError("source-event selector metadata is not canonical")
    required = {
        "array_dtype",
        "array_shape",
        "artifact_content_root",
        "banked_source_bundle_sha256s",
        "banked_static_certificate_sha256s",
        "banked_static_view_sha256s",
        "c1_input_count",
        "candidate_appearance_count",
        "cell_off_ready_count",
        "checkpoint_header",
        "checkpoint_sha256",
        "committed_excitatory_ticks",
        "committed_inhibitory_ticks",
        "dev_input_count",
        "excitatory_sha256",
        "feature_registry",
        "fit_appearance_count",
        "fit_exact_count",
        "fit_identity_count",
        "fit_source_bundles_embedded",
        "format",
        "g0_input_count",
        "heldout_input_count",
        "inhibitory_sha256",
        "input_artifact_sha256",
        "input_bank_commit",
        "input_content_root",
        "input_roots_embedded",
        "input_source_sha256",
        "literal_positive_feature_count",
        "literal_type_cells",
        "m0_input_count",
        "missing_positive_feature_count",
        "negative_feature_count",
        "normal_runtime_fit_count",
        "normal_runtime_imports_annotation_v1",
        "normal_runtime_imports_annotation_v2",
        "normal_runtime_imports_executable_learner",
        "normal_runtime_imports_input_capsule",
        "normal_runtime_teacher_target_read_count",
        "panel_spend_count",
        "path2_input_count",
        "positive_feature_count",
        "protected_input_count",
        "protocol_sha256",
        "purpose",
        "query_target_conflict_count",
        "raw_positive_feature_count",
        "record_order_used_for_selection",
        "selected_source_appearance_count",
        "selector_physical_query_count",
        "static_artifact_sha256",
        "static_bank_commit",
        "static_checkpoint_sha256",
        "static_protocol_sha256",
        "static_source_sha256",
        "teacher_off_ready_count",
        "unique_feature_count",
        "v1_bank_commit",
        "v1_capsule_sha256",
        "v1_source_sha256",
        "v1_stop_sha256",
        "v2_bank_commit",
        "v2_capsule_sha256",
        "v2_source_sha256",
        "v2_target_set_sha256",
    }
    if set(metadata) != required:
        raise RuntimeError("source-event selector metadata keys changed")
    content_record = dict(metadata)
    content_root = str(content_record.pop("artifact_content_root"))
    if (
        content_root != ARTIFACT_CONTENT_ROOT
        or _digest(_CONTENT_NAMESPACE, content_record) != content_root
    ):
        raise RuntimeError("source-event selector content root changed")
    expected_authority = {
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
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "array_shape": [1, 941],
        "array_dtype": "|u1",
        "fit_identity_count": 9,
        "fit_appearance_count": 12,
        "candidate_appearance_count": 4_986,
        "unique_feature_count": 941,
        "positive_feature_count": 26,
        "raw_positive_feature_count": 13,
        "literal_positive_feature_count": 13,
        "negative_feature_count": 915,
        "selected_source_appearance_count": 77,
        "query_target_conflict_count": 0,
        "missing_positive_feature_count": 0,
        "fit_exact_count": 941,
        "teacher_off_ready_count": 0,
        "cell_off_ready_count": 0,
        "committed_excitatory_ticks": 104,
        "committed_inhibitory_ticks": 3_660,
        "selector_physical_query_count": 6_868,
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
    if any(metadata.get(key) != value for key, value in expected_authority.items()):
        raise RuntimeError("source-event selector authority changed")
    if (
        excitatory.shape != (1, 941)
        or inhibitory.shape != excitatory.shape
        or excitatory.dtype != _TICK_DTYPE
        or inhibitory.dtype != _TICK_DTYPE
        or sha256(excitatory.tobytes(order="C")).hexdigest()
        != metadata["excitatory_sha256"]
        or sha256(inhibitory.tobytes(order="C")).hexdigest()
        != metadata["inhibitory_sha256"]
    ):
        raise RuntimeError("source-event selector arrays changed")
    features = tuple(_deep_tuple(row) for row in metadata["feature_registry"])
    literal_type_cells = tuple(int(cell) for cell in metadata["literal_type_cells"])
    if (
        len(features) != 941
        or len(set(features)) != 941
        or literal_type_cells
        != occurrence_static.STATIC_CHECKPOINT._protocol.type_registry[0]
        or len(metadata["banked_source_bundle_sha256s"]) != 9
        or len(metadata["banked_static_view_sha256s"]) != 9
        or len(metadata["banked_static_certificate_sha256s"]) != 9
    ):
        raise RuntimeError("source-event selector vocabulary changed")
    selector = FrozenSourceEventSelector(
        ARTIFACT_SHA256,
        content_root,
        PROTOCOL_SHA256,
        CHECKPOINT_SHA256,
        literal_type_cells,
        MappingProxyType(dict(metadata["checkpoint_header"])),
        features,
        _immutable_array(excitatory),
        _immutable_array(inhibitory),
        tuple(str(value) for value in metadata["banked_source_bundle_sha256s"]),
        tuple(str(value) for value in metadata["banked_static_view_sha256s"]),
    )
    selector.authenticate()
    return selector


STATIC_SELECTOR = load_source_event_selector()
_FEATURE_INDEX = {
    feature: index for index, feature in enumerate(STATIC_SELECTOR._feature_registry)
}


def _candidate_events(
    assemblies: tuple[tuple[int, ...], ...],
    literal_cells: tuple[tuple[int, ...], ...],
) -> tuple[PhysicalSourceEvent, ...]:
    events: list[PhysicalSourceEvent] = []
    for assembly in assemblies:
        for direction in ("NEXT", "PREV"):
            for cue_slot in range(4):
                selected_slot = (
                    (cue_slot + 1) % 4
                    if direction == "NEXT"
                    else (cue_slot - 1) % 4
                )
                events.append(
                    RawPhysicalSourceEvent(
                        assembly, direction, cue_slot, selected_slot
                    )
                )
    events.extend(
        LiteralPhysicalSourceEvent(cells, STATIC_SELECTOR.literal_type_cells)
        for cells in literal_cells
    )
    return tuple(sorted(events, key=lambda event: _canonical_json(event.neural_record())))


def _is_selected(event: PhysicalSourceEvent, *, cell_gain: float = 1.0) -> bool:
    index = _FEATURE_INDEX.get(_event_feature(event))
    if index is None:
        current = 0.0
    else:
        score = int(STATIC_SELECTOR._excitatory[0, index]) - int(
            STATIC_SELECTOR._inhibitory[0, index]
        )
        current = _OUTPUT_CURRENT_PA if score >= 1 else 0.0
    spike_mask = _SPIKE_KERNEL.accumulated_spike_mask(
        np.asarray([current * cell_gain], dtype=np.float64)
    )
    return bool(spike_mask[0])


class _CertificateReceipt(NamedTuple):
    authority: object
    status: str
    stop_reason: str | None
    request_sha256: str
    candidate_event_count: int
    selected_raw_event_count: int
    selected_literal_event_count: int
    events: tuple[PhysicalSourceEvent, ...]
    bundle_sha256: str | None
    downstream_certificate: occurrence_static.OccurrenceFactorStaticCertificate | None
    downstream_view_sha256: str | None
    physical_work_receipt_sha256: str
    sha256: str


_CERTIFICATE_AUTHORITY = object()
_ISSUED_CERTIFICATES: dict[int, PromptRecordSourceEventCertificate] = {}


class PromptRecordSourceEventCertificate:
    """Opaque exact-object result for one target-free prompt record bundle."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _CertificateReceipt) -> None:
        if receipt.authority is not _CERTIFICATE_AUTHORITY:
            raise TypeError("source-event certificate requires private authority")
        self._receipt = receipt

    @property
    def status(self) -> str:
        return self._receipt.status

    @property
    def stop_reason(self) -> str | None:
        return self._receipt.stop_reason

    @property
    def candidate_event_count(self) -> int:
        return self._receipt.candidate_event_count

    @property
    def selected_event_count(self) -> int:
        return len(self._receipt.events)

    @property
    def selected_raw_event_count(self) -> int:
        return self._receipt.selected_raw_event_count

    @property
    def selected_literal_event_count(self) -> int:
        return self._receipt.selected_literal_event_count

    @property
    def checkpoint_sha256(self) -> str:
        return CHECKPOINT_SHA256

    @property
    def bundle_sha256(self) -> str | None:
        return self._receipt.bundle_sha256

    @property
    def sha256(self) -> str:
        return self._receipt.sha256


def _certificate_payload(receipt: _CertificateReceipt) -> dict[str, object]:
    return {
        "status": receipt.status,
        "stop_reason": receipt.stop_reason,
        "request_sha256": receipt.request_sha256,
        "candidate_event_count": receipt.candidate_event_count,
        "selected_raw_event_count": receipt.selected_raw_event_count,
        "selected_literal_event_count": receipt.selected_literal_event_count,
        "bundle_sha256": receipt.bundle_sha256,
        "selector_checkpoint_sha256": CHECKPOINT_SHA256,
        "downstream_checkpoint_sha256": STATIC_CHECKPOINT_SHA256,
        "downstream_certificate_sha256": (
            receipt.downstream_certificate.sha256
            if receipt.downstream_certificate is not None
            else None
        ),
        "downstream_view_sha256": receipt.downstream_view_sha256,
        "physical_work_receipt_sha256": receipt.physical_work_receipt_sha256,
    }


def _issue_certificate(
    *,
    status: str,
    stop_reason: str | None,
    request_sha256: str,
    candidate_event_count: int,
    events: tuple[PhysicalSourceEvent, ...],
    downstream_certificate: occurrence_static.OccurrenceFactorStaticCertificate | None,
    downstream_view_sha256: str | None,
) -> PromptRecordSourceEventCertificate:
    selected_raw = sum(type(event) is RawPhysicalSourceEvent for event in events)
    selected_literal = sum(type(event) is LiteralPhysicalSourceEvent for event in events)
    bundle_sha = (
        _digest(
            _BUNDLE_NAMESPACE,
            tuple(
                event.neural_record()
                for event in sorted(
                    events, key=lambda item: _canonical_json(_event_feature(item))
                )
            ),
        )
        if events
        else None
    )
    work_receipt = _digest(
        _WORK_NAMESPACE,
        {
            "request_sha256": request_sha256,
            "selector_candidate_query_count": candidate_event_count,
            "selector_output_cell_count": 1,
            "selector_fitted_glif_step_count": candidate_event_count * _OUTPUT_STEPS,
            "downstream_certificate_sha256": (
                downstream_certificate.sha256
                if downstream_certificate is not None
                else None
            ),
        },
    )
    partial = _CertificateReceipt(
        _CERTIFICATE_AUTHORITY,
        status,
        stop_reason,
        request_sha256,
        candidate_event_count,
        selected_raw,
        selected_literal,
        events,
        bundle_sha,
        downstream_certificate,
        downstream_view_sha256,
        work_receipt,
        "",
    )
    receipt = partial._replace(
        sha256=_digest(_CERTIFICATE_NAMESPACE, _certificate_payload(partial))
    )
    certificate = PromptRecordSourceEventCertificate(receipt)
    _ISSUED_CERTIFICATES[id(certificate)] = certificate
    return certificate


def _authenticate_certificate(
    certificate: PromptRecordSourceEventCertificate,
) -> _CertificateReceipt:
    if (
        type(certificate) is not PromptRecordSourceEventCertificate
        or _ISSUED_CERTIFICATES.get(id(certificate)) is not certificate
        or certificate._receipt.authority is not _CERTIFICATE_AUTHORITY
    ):
        raise TypeError("source-event certificate is not an issued exact object")
    receipt = certificate._receipt
    if receipt.sha256 != _digest(
        _CERTIFICATE_NAMESPACE, _certificate_payload(receipt)
    ):
        raise RuntimeError("source-event certificate receipt changed")
    return receipt


def produce_source_event_bundle(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
) -> PromptRecordSourceEventCertificate:
    """Produce a neutral same-FIT source bundle from actual prompt records."""

    STATIC_SELECTOR.authenticate()
    occurrence_static.STATIC_CHECKPOINT.authenticate()
    assemblies, literal_cells = _normalize_prompt_records(
        span_records, literal_records, signature_role_record
    )
    request_record = {
        "raw_assemblies": assemblies,
        "literal_event_cells": literal_cells,
        "signature_role_record": _plain(signature_role_record),
    }
    request_sha = _digest(_REQUEST_NAMESPACE, request_record)
    candidates = _candidate_events(assemblies, literal_cells)
    selected = tuple(event for event in candidates if _is_selected(event))
    raw_count = sum(type(event) is RawPhysicalSourceEvent for event in selected)
    literal_count = sum(
        type(event) is LiteralPhysicalSourceEvent for event in selected
    )
    if not selected:
        return _issue_certificate(
            status=STOP_ZERO,
            stop_reason="no learned physical source event was selected",
            request_sha256=request_sha,
            candidate_event_count=len(candidates),
            events=(),
            downstream_certificate=None,
            downstream_view_sha256=None,
        )
    if raw_count not in (6, 7) or literal_count != 2 or len(selected) not in (8, 9):
        return _issue_certificate(
            status=STOP_UNSUPPORTED,
            stop_reason="selected source events are not one retained 8/9-node FIT shape",
            request_sha256=request_sha,
            candidate_event_count=len(candidates),
            events=selected,
            downstream_certificate=None,
            downstream_view_sha256=None,
        )
    downstream = occurrence_static.infer_occurrence_factors(selected)
    if downstream.status != occurrence_static.READY:
        return _issue_certificate(
            status=STOP_UNSUPPORTED,
            stop_reason=(
                "selected source events have no authenticated static occurrence-factor view"
            ),
            request_sha256=request_sha,
            candidate_event_count=len(candidates),
            events=selected,
            downstream_certificate=downstream,
            downstream_view_sha256=None,
        )
    view = occurrence_static.require_occurrence_factors(downstream)
    STATIC_SELECTOR.authenticate()
    return _issue_certificate(
        status=READY,
        stop_reason=None,
        request_sha256=request_sha,
        candidate_event_count=len(candidates),
        events=selected,
        downstream_certificate=downstream,
        downstream_view_sha256=view.sha256,
    )


def require_source_event_bundle(
    certificate: PromptRecordSourceEventCertificate,
) -> tuple[PhysicalSourceEvent, ...]:
    """Authenticate a READY exact object and return only neutral events."""

    receipt = _authenticate_certificate(certificate)
    STATIC_SELECTOR.authenticate()
    if receipt.status != READY or receipt.downstream_certificate is None:
        raise RuntimeError(receipt.stop_reason or receipt.status)
    view = occurrence_static.require_occurrence_factors(
        receipt.downstream_certificate
    )
    if receipt.downstream_view_sha256 != view.sha256:
        raise RuntimeError("downstream occurrence-factor view changed")
    return receipt.events


CLAIM_BOUNDARY = (
    "public-TRAIN same-FIT physical candidate membership and prompt-record to "
    "neutral occurrence-factor source-event replay only; selector assemblies "
    "are finite-cohort learned cues, and unseen prompt transfer, semantic "
    "geometry induction, canonical plan selection, DEV, protected, M0, C1, "
    "Path-2, G0, coding benchmarks, and broad capability remain unproven"
)


__all__ = [
    "ARTIFACT_BYTES",
    "ARTIFACT_CONTENT_ROOT",
    "ARTIFACT_PATH",
    "ARTIFACT_SHA256",
    "CHECKPOINT_SHA256",
    "CLAIM_BOUNDARY",
    "PROTOCOL_SHA256",
    "READY",
    "STATIC_SELECTOR",
    "STOP_AMBIGUOUS",
    "STOP_UNSUPPORTED",
    "STOP_ZERO",
    "LiteralPhysicalSourceEvent",
    "PromptRecordSourceEventCertificate",
    "RawPhysicalSourceEvent",
    "load_source_event_selector",
    "produce_source_event_bundle",
    "require_source_event_bundle",
]
