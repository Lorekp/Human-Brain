from __future__ import annotations

import ast
import builtins
import importlib
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

STATIC_MODULE = (
    "experiments.r3_pilot."
    "public_train_fit_occurrence_factor_static_checkpoint_capsule_v1"
)
LEARNER_MODULE = (
    "experiments.r3_pilot.public_train_fit_occurrence_factor_learner_v1"
)
ANNOTATION_MODULE = (
    "experiments.r3_pilot."
    "public_train_fit_occurrence_factor_annotation_capsule_v2"
)
FORBIDDEN_RUNTIME_MODULES = (LEARNER_MODULE, ANNOTATION_MODULE)


def test_static_capsule_isolated_import_and_nine_bundle_exact_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (STATIC_MODULE, *FORBIDDEN_RUNTIME_MODULES):
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
    capsule = importlib.import_module(STATIC_MODULE)
    assert not any(name in sys.modules for name in FORBIDDEN_RUNTIME_MODULES)
    assert (
        capsule.NORMAL_RUNTIME_TEACHER_TARGET_READ_COUNT
        == capsule.NORMAL_RUNTIME_FIT_COUNT
        == capsule.NORMAL_RUNTIME_LEARNER_IMPORT_COUNT
        == capsule.NORMAL_RUNTIME_ANNOTATION_IMPORT_COUNT
        == 0
    )
    monkeypatch.setattr(builtins, "__import__", real_import)

    source_path = Path(capsule.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not set(FORBIDDEN_RUNTIME_MODULES) & imports
    assert "importlib" not in imports
    assert "FIT_SOURCE_BUNDLES" not in source

    exporter_path = source_path.with_name(
        "export_public_train_fit_occurrence_factor_learner_v1_static_capsule_v1.py"
    )
    exporter_tree = ast.parse(exporter_path.read_text(encoding="utf-8"))
    top_level_imports = {
        node.module or ""
        for node in exporter_tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert LEARNER_MODULE not in top_level_imports

    checkpoint = capsule.STATIC_CHECKPOINT
    assert len(capsule.ARTIFACT_PATH.read_bytes()) == capsule.ARTIFACT_BYTES
    assert sha256(capsule.ARTIFACT_PATH.read_bytes()).hexdigest() == (
        capsule.ARTIFACT_SHA256
    )
    assert capsule.BANKED_COMMIT == "3ac16dc22caae8ce2ac14b82f8579494c0b30dc0"
    assert checkpoint.checkpoint_sha256 == capsule.BANKED_CHECKPOINT_SHA256
    assert checkpoint.protocol_sha256 == capsule.BANKED_PROTOCOL_SHA256
    for state in (checkpoint._excitatory, checkpoint._inhibitory):
        assert state.shape == (499, 372)
        assert state.dtype == np.uint8
        assert not state.flags.writeable
        with pytest.raises(ValueError):
            state.setflags(write=True)

    learner = importlib.import_module(LEARNER_MODULE)
    view_roots: list[str] = []
    certificate_roots: list[str] = []
    aggregate = [0] * 7
    static_bundles: list[tuple[object, ...]] = []
    for banked_bundle in learner.FIT_SOURCE_BUNDLES:
        static_bundle = tuple(
            capsule.RawPhysicalSourceEvent(
                event.assembly_cells,
                event.direction,
                event.cue_slot,
                event.selected_slot,
            )
            if type(event) is learner.RawPhysicalSourceEvent
            else capsule.LiteralPhysicalSourceEvent(
                event.typed_value_event_cells,
                event.typed_value_type_cells,
            )
            for event in banked_bundle
        )
        static_bundles.append(static_bundle)
        banked_certificate = learner.infer_public_train_fit_occurrence_factors_v1(
            banked_bundle
        )
        banked_view = learner.require_public_train_fit_occurrence_factors_v1(
            banked_certificate
        )
        certificate = capsule.infer_occurrence_factors(static_bundle)
        view = capsule.require_occurrence_factors(certificate)

        assert certificate.status == capsule.READY
        assert certificate.checkpoint_sha256 == capsule.BANKED_CHECKPOINT_SHA256
        assert certificate.sha256 == banked_certificate.sha256
        assert view.sha256 == banked_view.sha256
        assert view.checkpoint_sha256 == capsule.BANKED_CHECKPOINT_SHA256
        assert view.protocol_sha256 == capsule.BANKED_PROTOCOL_SHA256
        view_roots.append(view.sha256)
        certificate_roots.append(certificate.sha256)
        counts = (
            len(view.source_relays),
            len(view.edges),
            len(view.factor_slots),
            len(view.relations),
            len(view.domains),
            len(view.literal_copy_groups),
            len(view.signature_relays),
        )
        aggregate = [left + right for left, right in zip(aggregate, counts, strict=True)]

    assert tuple(view_roots) == checkpoint._banked_view_sha256s
    assert tuple(certificate_roots) == checkpoint._banked_certificate_sha256s
    assert tuple(aggregate) == (77, 68, 164, 9, 18, 18, 18)

    reverse = capsule.infer_occurrence_factors(reversed(static_bundles[0]))
    assert capsule.require_occurrence_factors(reverse).sha256 == view_roots[0]
    duplicate = capsule.infer_occurrence_factors(
        (*static_bundles[0], static_bundles[0][0])
    )
    assert duplicate.status == capsule.STOP_AMBIGUOUS
    with pytest.raises(RuntimeError, match="duplicated"):
        capsule.require_occurrence_factors(duplicate)
    with pytest.raises(TypeError, match="issued exact object"):
        capsule.require_occurrence_factors(object())
