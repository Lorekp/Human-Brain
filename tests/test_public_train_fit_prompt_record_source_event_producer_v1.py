from __future__ import annotations

import ast
import builtins
import importlib
import inspect
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

RUNTIME_MODULE = (
    "experiments.r3_pilot.public_train_fit_prompt_record_source_event_producer_v1"
)
INPUT_MODULE = (
    "experiments.r3_pilot.public_train_list_int_to_int_prompt_input_capsule_v1"
)
V1_MODULE = (
    "experiments.r3_pilot."
    "public_train_fit_occurrence_factor_annotation_capsule_v1"
)
V2_MODULE = (
    "experiments.r3_pilot."
    "public_train_fit_occurrence_factor_annotation_capsule_v2"
)
LEARNER_MODULE = (
    "experiments.r3_pilot.public_train_fit_occurrence_factor_learner_v1"
)
FORBIDDEN_RUNTIME_MODULES = (INPUT_MODULE, V1_MODULE, V2_MODULE, LEARNER_MODULE)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=list,
    ).encode("ascii")


def test_target_free_static_source_event_producer_exact_fit_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (RUNTIME_MODULE, *FORBIDDEN_RUNTIME_MODULES):
        sys.modules.pop(name, None)
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if any(
            name == forbidden or name.startswith(f"{forbidden}.")
            for forbidden in FORBIDDEN_RUNTIME_MODULES
        ):
            raise AssertionError(f"forbidden runtime import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    producer = importlib.import_module(RUNTIME_MODULE)
    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
    assert (
        producer.NORMAL_RUNTIME_IMPORTS_INPUT_CAPSULE
        == producer.NORMAL_RUNTIME_IMPORTS_ANNOTATION_V1
        == producer.NORMAL_RUNTIME_IMPORTS_ANNOTATION_V2
        == producer.NORMAL_RUNTIME_IMPORTS_EXECUTABLE_LEARNER
        == producer.NORMAL_RUNTIME_TEACHER_TARGET_READ_COUNT
        == producer.NORMAL_RUNTIME_FIT_COUNT
        == 0
    )
    monkeypatch.setattr(builtins, "__import__", real_import)

    runtime_path = Path(producer.__file__)
    runtime_source = runtime_path.read_text(encoding="utf-8")
    runtime_tree = ast.parse(runtime_source)
    runtime_imports = {
        alias.name
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(runtime_tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not set(FORBIDDEN_RUNTIME_MODULES) & runtime_imports
    assert "FIT_SOURCE_BUNDLES" not in runtime_source
    assert tuple(inspect.signature(producer.produce_source_event_bundle).parameters) == (
        "span_records",
        "literal_records",
        "signature_role_record",
    )

    exporter_path = runtime_path.with_name(
        "export_public_train_fit_prompt_record_source_event_producer_v1.py"
    )
    exporter_tree = ast.parse(exporter_path.read_text(encoding="utf-8"))
    top_level_imports = {
        node.module or ""
        for node in exporter_tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert not set(FORBIDDEN_RUNTIME_MODULES) & top_level_imports

    selector = producer.STATIC_SELECTOR
    assert len(producer.ARTIFACT_PATH.read_bytes()) == producer.ARTIFACT_BYTES
    assert sha256(producer.ARTIFACT_PATH.read_bytes()).hexdigest() == (
        producer.ARTIFACT_SHA256
    )
    assert selector.protocol_sha256 == producer.PROTOCOL_SHA256
    assert selector.checkpoint_sha256 == producer.CHECKPOINT_SHA256
    assert len(selector._feature_registry) == 941
    for state in (selector._excitatory, selector._inhibitory):
        assert state.shape == (1, 941)
        assert state.dtype == np.uint8
        assert not state.flags.writeable
        with pytest.raises(ValueError):
            state.setflags(write=True)

    inputs = importlib.import_module(INPUT_MODULE)
    annotation_v1 = importlib.import_module(V1_MODULE)
    annotation_v2 = importlib.import_module(V2_MODULE)
    state = inputs.load_input_capsule()

    expected_bundle_records: set[tuple[bytes, ...]] = set()
    for target in annotation_v2.CAPSULE.targets:
        expected_events = tuple(
            producer.RawPhysicalSourceEvent(
                source.observation.assembly_cells,
                source.observation.direction,
                source.observation.cue_slot,
                source.observation.selected_slot,
            )
            if isinstance(source, annotation_v2.RawSourceTarget)
            else producer.LiteralPhysicalSourceEvent(
                source.typed_value_event_cells,
                source.typed_value_type_cells,
            )
            for source in target.sources
        )
        expected_bundle_records.add(
            tuple(sorted(_canonical(event.neural_record()) for event in expected_events))
        )

    actual_bundle_records: set[tuple[bytes, ...]] = set()
    certificates = []
    selected_count = 0
    candidate_count = 0
    first_records = None
    for index, audit in enumerate(annotation_v1.CAPSULE.fit_identity_audits):
        frozen = state.lookup(audit.input_root_sha256)
        records = frozen.neural_records()
        if first_records is None:
            first_records = records
        certificate = producer.produce_source_event_bundle(*records)
        events = producer.require_source_event_bundle(certificate)
        certificates.append(certificate)
        selected_count += len(events)
        candidate_count += certificate.candidate_event_count

        assert certificate.status == producer.READY
        assert certificate.candidate_event_count == 554
        assert certificate.selected_literal_event_count == 2
        assert certificate.selected_raw_event_count in (6, 7)
        assert certificate.bundle_sha256 == (
            selector._banked_source_bundle_sha256s[index]
        )
        downstream = producer.occurrence_static.infer_occurrence_factors(events)
        view = producer.occurrence_static.require_occurrence_factors(downstream)
        assert view.sha256 == selector._banked_static_view_sha256s[index]
        actual_bundle_records.add(
            tuple(sorted(_canonical(event.neural_record()) for event in events))
        )

    assert candidate_count == 4_986
    assert selected_count == 77
    assert len(actual_bundle_records) == len(expected_bundle_records) == 9
    assert actual_bundle_records == expected_bundle_records

    assert first_records is not None
    reversed_certificate = producer.produce_source_event_bundle(
        tuple(reversed(first_records[0])),
        tuple(reversed(first_records[1])),
        first_records[2],
    )
    assert reversed_certificate.sha256 == certificates[0].sha256
    assert producer.require_source_event_bundle(reversed_certificate) == (
        producer.require_source_event_bundle(certificates[0])
    )
    with pytest.raises(TypeError, match="issued exact object"):
        producer.require_source_event_bundle(object())
