from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from hashlib import sha256
from pathlib import Path

import pytest

from experiments.r3_pilot import public_train_prompt_factor_manifest_v1 as manifest


def _synthetic_manifest_record() -> dict[str, object]:
    lessons: list[dict[str, object]] = []
    for index in range(90):
        targets = [
            {
                "start_token": 0,
                "stop_token": 1,
                "span_identity_sha256": f"{index + 1000:064x}",
                "factor": "add",
            }
        ]
        if index >= 87:
            targets.append(
                {
                    "start_token": 0,
                    "stop_token": 1,
                    "span_identity_sha256": f"{index + 1000:064x}",
                    "factor": "absolute",
                }
            )
        lessons.append(
            {
                "prompt_input_sha256": f"{index:064x}",
                "targets": targets,
            }
        )
    return {
        "schema": manifest.LESSON_SCHEMA,
        "scope": manifest.LESSON_SCOPE,
        "input_projection_sha256": manifest.INPUT_PROJECTION_SHA256,
        "factor_registry_sha256": manifest.FACTOR_REGISTRY_SHA256,
        "lessons": lessons,
    }


def test_registry_capsule_and_lesson_hashes_are_frozen() -> None:
    assert len(manifest.FACTOR_REGISTRY) == 24
    assert len(set(manifest.FACTOR_REGISTRY)) == 24
    assert manifest.FACTOR_REGISTRY_SHA256 == (
        "f7df546c7c89c20d48c79c737dab8fcb101d9c861cb041c0c39654268ecb28c6"
    )
    assert manifest.INPUT_PROJECTION_SHA256 == (
        "2108f4864cfbbcd3f44be96ad7b55d95a89d8d4899c824ec913204b5318320d6"
    )
    assert manifest.CAPSULE_RAW_SHA256 == (
        "0f34d5c1e4b9e90f76f098bf3296af546495d8b6053586b1620ecd88ba5b3289"
    )
    assert manifest.CAPSULE_PAYLOAD_SHA256 == (
        "d64c61452d8a4fcd2e67a4deb4848c07cd93a757d36da2cfd2b90f81fa00a617"
    )
    assert manifest.EXPECTED_LESSON_MANIFEST_SHA256 == (
        "139e678a6001f5264d8e6ba44609e2b97082cb8b7c443462326c370d4fc5e774"
    )
    assert manifest.EXPECTED_PREFLIGHT_PAYLOAD_SHA256 == (
        "c4dd1d14b57c9359691f8077742f41b2fe4277e6cca414b5b2f60f94871459cd"
    )


def test_capsule_records_expose_only_prompt_and_typed_signature() -> None:
    path = Path(manifest.CAPSULE_PATH)
    raw = path.read_bytes()
    capsule = json.loads(raw)

    assert set(capsule) == {"schema", "scope", "records"}
    assert len(capsule["records"]) == 90
    assert all(set(record) == {"prompt", "signature"} for record in capsule["records"])
    assert all(
        set(record["signature"]) == {"entry_point", "parameters", "result"}
        for record in capsule["records"]
    )
    assert manifest.canonical_json_bytes(capsule) + b"\n" == raw


def test_gate_source_and_public_interface_exclude_forbidden_import_surfaces() -> None:
    source = Path(inspect.getsourcefile(manifest) or "").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    blocked_imports = (
        "humanbrain.simple_swe.tasks",
        "humanbrain.simple_swe.sandbox",
        "humanbrain.simple_swe.runner",
        "humanbrain.simple_swe.artifacts",
    )
    assert not any(
        name == blocked or name.startswith(blocked + ".")
        for name in imported
        for blocked in blocked_imports
    )

    forbidden = manifest.FORBIDDEN_FIELD_FRAGMENTS
    assert not any(
        fragment in name.casefold()
        for name in imported
        for fragment in forbidden
    )
    assert not any(
        fragment in exported.casefold()
        for exported in manifest.__all__
        for fragment in forbidden
    )
    public_callables = (
        manifest.authenticate_lesson_targets,
        manifest.build_lesson_manifest_record,
        manifest.construct_split_manifest,
        manifest.load_public_train_prompt_inputs,
        manifest.parse_lesson_manifest,
        manifest.preflight_prompt_factor_manifest,
    )
    assert not any(
        fragment in parameter.casefold()
        for function in public_callables
        for parameter in inspect.signature(function).parameters
        for fragment in forbidden
    )
    record_types = (
        manifest.FactorTarget,
        manifest.FactorLesson,
        manifest.LessonManifest,
        manifest.SplitManifest,
        manifest.ManifestPreflight,
    )
    assert not any(
        fragment in field.name.casefold()
        for record_type in record_types
        for field in dataclasses.fields(record_type)
        for fragment in forbidden
    )


def test_strict_lesson_parser_rejects_extra_and_forbidden_fields() -> None:
    record = _synthetic_manifest_record()
    parsed = manifest.parse_lesson_manifest(record)
    assert len(parsed.lessons) == 90

    extra = json.loads(json.dumps(record))
    extra["notes"] = "not frozen"
    with pytest.raises(manifest.ManifestValidationError, match="fields differ"):
        manifest.parse_lesson_manifest(extra)

    forbidden = json.loads(json.dumps(record))
    forbidden["lessons"][0]["family"] = "forbidden"
    with pytest.raises(manifest.ManifestValidationError, match="forbidden field"):
        manifest.parse_lesson_manifest(forbidden)

    nested = json.loads(json.dumps(record))
    nested["lessons"][0]["targets"][0]["property"] = []
    with pytest.raises(manifest.ManifestValidationError, match="forbidden field"):
        manifest.parse_lesson_manifest(nested)

    unknown_factor = json.loads(json.dumps(record))
    unknown_factor["lessons"][0]["targets"][0]["factor"] = "unknown"
    with pytest.raises(manifest.ManifestValidationError, match="outside the registry"):
        manifest.parse_lesson_manifest(unknown_factor)


def test_targets_authenticate_against_physical_prompt_spans() -> None:
    prompt_inputs = manifest.load_public_train_prompt_inputs()
    record = manifest.build_lesson_manifest_record(prompt_inputs)
    parsed = manifest.parse_lesson_manifest(record)
    manifest.authenticate_lesson_targets(parsed, prompt_inputs)

    tampered = json.loads(json.dumps(record))
    tampered_identity = tampered["lessons"][0]["prompt_input_sha256"]
    for lesson in tampered["lessons"]:
        if lesson["prompt_input_sha256"] == tampered_identity:
            lesson["targets"][0]["span_identity_sha256"] = "0" * 64
    parsed_tampered = manifest.parse_lesson_manifest(tampered)
    with pytest.raises(manifest.ManifestValidationError, match="does not authenticate"):
        manifest.authenticate_lesson_targets(parsed_tampered, prompt_inputs)

    with pytest.raises(TypeError, match="PromptSpanInput"):
        manifest.authenticate_lesson_targets(parsed, (object(),))


def test_duplicate_semantic_inputs_cannot_receive_conflicting_targets() -> None:
    prompt_inputs = manifest.load_public_train_prompt_inputs()
    record = manifest.build_lesson_manifest_record(prompt_inputs)
    identities: dict[str, list[int]] = {}
    for index, lesson in enumerate(record["lessons"]):
        identities.setdefault(lesson["prompt_input_sha256"], []).append(index)
    duplicate = next(indices for indices in identities.values() if len(indices) > 1)
    altered = json.loads(json.dumps(record))
    first_target = altered["lessons"][duplicate[1]]["targets"][0]
    replacement = "subtract" if first_target["factor"] == "add" else "add"
    first_target["factor"] = replacement
    altered["lessons"][duplicate[1]]["targets"].sort(
        key=lambda target: (
            target["start_token"],
            target["stop_token"],
            manifest.FACTOR_REGISTRY.index(target["factor"]),
        )
    )
    parsed = manifest.parse_lesson_manifest(altered)
    with pytest.raises(manifest.ManifestValidationError, match="contradictory"):
        manifest.authenticate_lesson_targets(parsed, prompt_inputs)


def test_split_algorithm_deterministically_constructs_72_and_18_when_legal() -> None:
    parsed = manifest.parse_lesson_manifest(_synthetic_manifest_record())
    first = manifest.construct_split_manifest(parsed)
    second = manifest.construct_split_manifest(parsed)

    assert first == second
    assert len(first.fit_prompt_input_sha256s) == 72
    assert len(first.generalization_prompt_input_sha256s) == 18
    assert not (
        set(first.fit_prompt_input_sha256s)
        & set(first.generalization_prompt_input_sha256s)
    )
    absolute_inputs = {f"{index:064x}" for index in range(87, 90)}
    assert set(first.generalization_prompt_input_sha256s) & absolute_inputs
    assert len(set(first.fit_prompt_input_sha256s) & absolute_inputs) >= 2
    assert first.sha256 == second.sha256


def test_canonical_public_train_preflight_stops_before_split_or_learning() -> None:
    first = manifest.canonical_public_train_preflight()
    second = manifest.canonical_public_train_preflight()
    assert first == second
    assert first.status == manifest.STOP_IDENTIFIABILITY
    assert first.prompt_appearance_count == 90
    assert first.unique_prompt_input_count == 42
    assert first.duplicate_prompt_appearance_count == 48
    assert first.lesson_manifest_sha256 == (
        "139e678a6001f5264d8e6ba44609e2b97082cb8b7c443462326c370d4fc5e774"
    )
    assert first.split_manifest_sha256 is None
    assert manifest.LEARNING_UPDATES == 0
    assert manifest.LEARNER_ARRAY_BYTES == 0

    appearances = dict(first.factor_appearance_counts)
    distinct = dict(first.factor_distinct_input_counts)
    assert appearances == {
        "add": 30,
        "subtract": 0,
        "minimum": 20,
        "maximum": 30,
        "absolute": 10,
        "less-than": 10,
        "map": 40,
        "sum": 10,
        "count": 10,
        "contains": 0,
        "filter": 20,
        "reverse": 10,
        "class:ascii-alphanumeric": 0,
        "class:ascii-decimal-digits": 5,
        "class:ascii-hex-digits": 0,
        "class:ascii-letters": 0,
        "class:ascii-lowercase": 6,
        "class:ascii-octal-digits": 0,
        "class:ascii-printable": 0,
        "class:ascii-punctuation": 0,
        "class:ascii-uppercase": 0,
        "class:ascii-whitespace": 0,
        "class:english-consonants": 0,
        "class:english-vowels": 9,
    }
    assert distinct == {
        "add": 22,
        "subtract": 0,
        "minimum": 12,
        "maximum": 13,
        "absolute": 1,
        "less-than": 8,
        "map": 21,
        "sum": 7,
        "count": 8,
        "contains": 0,
        "filter": 6,
        "reverse": 3,
        "class:ascii-alphanumeric": 0,
        "class:ascii-decimal-digits": 2,
        "class:ascii-hex-digits": 0,
        "class:ascii-letters": 0,
        "class:ascii-lowercase": 2,
        "class:ascii-octal-digits": 0,
        "class:ascii-printable": 0,
        "class:ascii-punctuation": 0,
        "class:ascii-uppercase": 0,
        "class:ascii-whitespace": 0,
        "class:english-consonants": 0,
        "class:english-vowels": 2,
    }
    assert first.reasons == (
        "expected 90 unique prompt-input identities; observed 42 across 90 appearances",
        "active factor absolute occurs in 1 distinct prompt inputs; requires at least 3",
        "active factor class:ascii-decimal-digits occurs in 2 distinct prompt inputs; requires at least 3",
        "active factor class:ascii-lowercase occurs in 2 distinct prompt inputs; requires at least 3",
        "active factor class:english-vowels occurs in 2 distinct prompt inputs; requires at least 3",
    )
    record = first.to_record()
    assert (
        sha256(manifest.canonical_json_bytes(record)).hexdigest()
        == manifest.EXPECTED_PREFLIGHT_PAYLOAD_SHA256
    )
    assert record["checkpoint_created"] is False
    assert record["learner_array_bytes"] == 0
    assert record["learning_updates"] == 0
    assert record["split_constructed"] is False


def test_same_physical_span_identity_never_has_conflicting_label_status() -> None:
    prompt_inputs = manifest.load_public_train_prompt_inputs()
    record = manifest.build_lesson_manifest_record(prompt_inputs)
    by_input = {
        lesson["prompt_input_sha256"]: lesson for lesson in record["lessons"]
    }
    status_by_span_and_factor: dict[tuple[str, str], set[bool]] = {}
    for prompt_input_sha, lesson in by_input.items():
        prompt_input = next(
            value for value in prompt_inputs if value.sha256 == prompt_input_sha
        )
        labelled = {
            (
                target["start_token"],
                target["stop_token"],
                target["factor"],
            )
            for target in lesson["targets"]
        }
        for span in prompt_input.spans:
            for factor in manifest.FACTOR_REGISTRY:
                key = (span.identity_sha256, factor)
                status_by_span_and_factor.setdefault(key, set()).add(
                    (span.start_token, span.stop_token, factor) in labelled
                )
    assert all(len(statuses) == 1 for statuses in status_by_span_and_factor.values())
