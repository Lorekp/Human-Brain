"""One-time exporter for the public-TRAIN ``List[Int] -> Int`` input capsule.

The module is inert on import.  An explicit invocation authenticates the frozen
prompt/signature source, calls the target-free prompt observer, and writes only
the three physical input channels consumed by prompt V4.  Prompt text, declared
names, source order, semantic annotations, and all target-bearing surfaces are
discarded before serialization.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import public_train_prompt_semantic_bridge as prompt_input
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "public-train-list-int-to-int-prompt-input-capsule-v1"
SCOPE = "PUBLIC_TRAIN_LIST_INT_TO_INT_PROMPT_INPUTS_ONLY"
SOURCE_SCHEMA = "public-train-prompt-signature-capsule-v1"
SOURCE_SCOPE = "PUBLIC_TRAIN_PROMPTS_AND_SIGNATURES_ONLY"

APPEARANCE_COUNT = 20
INPUT_COUNT = 15
SOURCE_RECORD_COUNT = 90

SOURCE_PATH = Path(__file__).with_name("public_train_prompt_signature_capsule_v1.json")
OBSERVER_PATH = Path(prompt_input.__file__).resolve()
ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_list_int_to_int_prompt_input_capsule_v1.json"
)

SOURCE_RAW_SHA256 = "0f34d5c1e4b9e90f76f098bf3296af546495d8b6053586b1620ecd88ba5b3289"
OBSERVER_SOURCE_SHA256 = (
    "74a123d88707525b63ff50021b3c7e5638f93ed8ed3dd3b82d20724bb3f1d662"
)

_SIGNATURE_ROLE_NAMESPACE = b"public-train-prompt-selection-v1-signature-role\0"
_SPAN_EVENTS_NAMESPACE = b"public-train-prompt-selection-v1-span-events\0"
_LITERAL_EVENTS_NAMESPACE = b"public-train-prompt-selection-v1-literal-events\0"
_CAUSAL_KEY_NAMESPACE = b"public-train-prompt-selection-v1-causal-key\0"
_CONTENT_ROOT_NAMESPACE = b"public-train-list-int-to-int-prompt-input-content-v1\0"

_CAUSAL_SOURCES = (
    "ordered-physical-span-events",
    "ordered-typed-literal-cell-events",
    "ordered-name-free-signature-role-events",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(value))
    return digest.hexdigest()


def _strict_record(
    value: object,
    keys: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"{label} schema changed")
    return value


def _strict_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeError(f"{label} must be a sequence")
    return value


def _is_list_int_to_int(signature_record: Mapping[str, object]) -> bool:
    parameters = _strict_sequence(signature_record["parameters"], "signature parameters")
    if len(parameters) != 1:
        return False
    parameter = _strict_record(
        parameters[0], frozenset({"name", "type"}), "signature parameter"
    )
    parameter_type = _strict_record(
        parameter["type"], frozenset({"constructor", "arguments"}), "parameter type"
    )
    arguments = _strict_sequence(parameter_type["arguments"], "parameter type arguments")
    if len(arguments) != 1:
        return False
    item_type = _strict_record(
        arguments[0], frozenset({"constructor", "arguments"}), "item type"
    )
    result_type = _strict_record(
        signature_record["result"],
        frozenset({"constructor", "arguments"}),
        "result type",
    )
    return (
        parameter_type["constructor"] == "List"
        and item_type == {"constructor": "Int", "arguments": []}
        and result_type == {"constructor": "Int", "arguments": []}
    )


def _signature_role_record(signature: TypedSignature) -> dict[str, object]:
    parameter_count = len(signature.parameters)
    return {
        "parameters": [
            {
                "role": f"ARG{parameter_count - index - 1}",
                "type": parameter.type.to_record(),
            }
            for index, parameter in enumerate(signature.parameters)
        ],
        "result": signature.result.to_record(),
        "role_binding": "reverse-declaration-depth",
        "entrypoint_spelling_present": False,
        "parameter_spelling_present": False,
    }


def _span_records(
    observed: prompt_input.PromptSpanInput,
) -> list[dict[str, object]]:
    return [
        {
            "start_token": span.start_token,
            "stop_token": span.stop_token,
            "first_spike_step": span.first_spike_step,
            "assembly_cells": list(span.assembly_cells),
            "spike_cells": list(span.spike_cells),
        }
        for span in observed.spans
    ]


def _literal_records(
    observed: prompt_input.PromptSpanInput,
) -> list[dict[str, object]]:
    return [
        {
            "canonical_token_index": literal.canonical_token_index,
            "typed_output_active_cells": list(literal.typed_output_active_cells),
        }
        for literal in observed.signed_literals
    ]


def _physical_input(
    observed: prompt_input.PromptSpanInput,
    signature: TypedSignature,
) -> dict[str, object]:
    spans = _span_records(observed)
    literals = _literal_records(observed)
    role_record = _signature_role_record(signature)
    span_root = _namespaced_sha256(_SPAN_EVENTS_NAMESPACE, spans)
    literal_root = _namespaced_sha256(_LITERAL_EVENTS_NAMESPACE, literals)
    role_root = _namespaced_sha256(_SIGNATURE_ROLE_NAMESPACE, role_record)
    causal_record = {
        "physical_span_events_sha256": span_root,
        "typed_literal_events_sha256": literal_root,
        "signature_role_events_sha256": role_root,
        "causal_sources": list(_CAUSAL_SOURCES),
    }
    return {
        "input_root": _namespaced_sha256(_CAUSAL_KEY_NAMESPACE, causal_record),
        "physical_span_events_sha256": span_root,
        "typed_literal_events_sha256": literal_root,
        "signature_role_events_sha256": role_root,
        "raw_span_records": spans,
        "typed_literal_records": literals,
        "signature_role_record": role_record,
    }


def export_artifact(path: Path = ARTIFACT_PATH) -> tuple[str, str]:
    """Materialize the immutable input artifact once and return its two roots."""

    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("public-TRAIN input capsule already exists")
    source_raw = SOURCE_PATH.read_bytes()
    if sha256(source_raw).hexdigest() != SOURCE_RAW_SHA256:
        raise RuntimeError("prompt/signature source identity changed")
    if sha256(OBSERVER_PATH.read_bytes()).hexdigest() != OBSERVER_SOURCE_SHA256:
        raise RuntimeError("target-free prompt observer identity changed")
    source = _strict_record(
        json.loads(source_raw),
        frozenset({"records", "schema", "scope"}),
        "prompt/signature source",
    )
    records = _strict_sequence(source["records"], "prompt/signature records")
    if (
        source["schema"] != SOURCE_SCHEMA
        or source["scope"] != SOURCE_SCOPE
        or len(records) != SOURCE_RECORD_COUNT
    ):
        raise RuntimeError("prompt/signature source contract changed")

    appearances: list[dict[str, object]] = []
    for raw_record in records:
        record = _strict_record(
            raw_record, frozenset({"prompt", "signature"}), "prompt/signature record"
        )
        signature_record = _strict_record(
            record["signature"],
            frozenset({"entry_point", "parameters", "result"}),
            "typed signature",
        )
        if not _is_list_int_to_int(signature_record):
            continue
        if type(record["prompt"]) is not str:
            raise RuntimeError("prompt text schema changed")
        signature = TypedSignature.from_record(signature_record)
        observed = prompt_input.observe_public_train_prompt(record["prompt"], signature)
        appearances.append(_physical_input(observed, signature))

    if len(appearances) != APPEARANCE_COUNT:
        raise RuntimeError("List[Int] -> Int appearance count changed")
    by_root: dict[str, dict[str, object]] = {}
    for appearance in appearances:
        input_root = str(appearance["input_root"])
        previous = by_root.setdefault(input_root, appearance)
        if previous != appearance:
            raise RuntimeError("one physical input root maps to multiple records")
    if len(by_root) != INPUT_COUNT:
        raise RuntimeError("List[Int] -> Int physical input identity count changed")

    multiplicities = Counter(str(row["input_root"]) for row in appearances)
    inputs = []
    for input_root in sorted(by_root):
        row = dict(by_root[input_root])
        row["multiplicity"] = multiplicities[input_root]
        inputs.append(row)
    appearance_roots = sorted(str(row["input_root"]) for row in appearances)
    artifact_without_root: dict[str, object] = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "source_authority": {
            "prompt_signature_schema": SOURCE_SCHEMA,
            "prompt_signature_scope": SOURCE_SCOPE,
            "prompt_signature_raw_sha256": SOURCE_RAW_SHA256,
            "prompt_signature_record_count": SOURCE_RECORD_COUNT,
            "prompt_observer_source_sha256": OBSERVER_SOURCE_SHA256,
            "prompt_observer_tokenizer_sha256": prompt_input.TOKENIZER_SHA256,
            "prompt_observer_assembly_projection_sha256": (
                prompt_input.ASSEMBLY_PROJECTION_SHA256
            ),
        },
        "appearance_count": APPEARANCE_COUNT,
        "input_count": INPUT_COUNT,
        "appearance_input_roots": appearance_roots,
        "inputs": inputs,
    }
    content_root = _namespaced_sha256(_CONTENT_ROOT_NAMESPACE, artifact_without_root)
    artifact = {**artifact_without_root, "content_root": content_root}
    payload = _canonical_json(artifact) + b"\n"
    path.write_bytes(payload)
    return sha256(payload).hexdigest(), content_root


if __name__ == "__main__":
    print(export_artifact())
