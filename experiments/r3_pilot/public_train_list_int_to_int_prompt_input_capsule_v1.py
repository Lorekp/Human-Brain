"""Pure loader for frozen public-TRAIN ``List[Int] -> Int`` prompt inputs.

The immutable artifact contains only the physical records accepted by prompt
V4: ordered span events, ordered typed-literal events, and a name-free typed
signature-role event.  Input roots are lookup/authentication handles and are
never added to the neural event stream.  This runtime imports no prompt
learner, factor learner, annotation, manifest, task, or protocol module.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType

SCHEMA = "public-train-list-int-to-int-prompt-input-capsule-v1"
SCOPE = "PUBLIC_TRAIN_LIST_INT_TO_INT_PROMPT_INPUTS_ONLY"
SOURCE_SCHEMA = "public-train-prompt-signature-capsule-v1"
SOURCE_SCOPE = "PUBLIC_TRAIN_PROMPTS_AND_SIGNATURES_ONLY"

APPEARANCE_COUNT = 20
INPUT_COUNT = 15
SOURCE_RECORD_COUNT = 90

ARTIFACT_PATH = Path(__file__).with_name(
    "public_train_list_int_to_int_prompt_input_capsule_v1.json"
)
ARTIFACT_SHA256 = "0e90318d1c1d3bd365f9812fb3c43aa5f0806dbfa5a1e81cb527c8e7f2c20dcf"
CONTENT_ROOT = "a7ff53818cf4e83c823e6b23f5547e0e7ee1fd30ed73dd136e3672063ea7aca8"
SOURCE_RAW_SHA256 = "0f34d5c1e4b9e90f76f098bf3296af546495d8b6053586b1620ecd88ba5b3289"
OBSERVER_SOURCE_SHA256 = (
    "74a123d88707525b63ff50021b3c7e5638f93ed8ed3dd3b82d20724bb3f1d662"
)
OBSERVER_TOKENIZER_SHA256 = (
    "73777898bca9c1b009a3fb652de1993464f2bc6b0e14dafca5d706a0519fe62e"
)
OBSERVER_ASSEMBLY_PROJECTION_SHA256 = (
    "05181981735d4dbe6f7e609e26b26bc105bc289d8e77a9a108bbd50f17fcc89c"
)

EXPECTED_INPUT_ROOT_MULTIPLICITIES = (
    ("1c27890d66e7c640fb6eda85e5abc89b076d1de5da2aff7c4dc53238ca74512a", 2),
    ("44c0ced4cf0fc1e68e753c13467563b92a6dcc150732be29e093d96b637c3da6", 1),
    ("6ab68eead6b880d900baa9a94b981d937976cb2abf11fd515b2d078172bb49fc", 1),
    ("6ce57bcff03faaee3204c0806de890bdb37d87dc80ebce31dbf069c2ceabb8d6", 2),
    ("736507d11bb0330acd8e4d097eb718e47cc5116259d9fa46456b33ff27d5c8b8", 1),
    ("781be7355d7b577006361245a80d38817bde4b71dd970236124d15378421f577", 1),
    ("82249fd2801be46db6369c11d7e00727cd7b4a10264eb9c8460bc6a17e4f067d", 1),
    ("a37ad078846ce08b015ae18cec4f9b0fe364315be1e20512b80a4a886b661066", 1),
    ("c052a4977ab84722b5d951ea5ebfbcb8e2ef4f03f93425f1974ed7471dfefa69", 1),
    ("c7f9add62599364abb2e1577e5b3445cd899bf9bf52cd93c663dd6ff328497c1", 2),
    ("c95e6812371bb9c385172538cad4a324f46318dd44eaee0022db8596f306e149", 1),
    ("d56f3f0d2f5b5f26ddd5061927f750ad071fa60fef4174cb0fd10828c08686e2", 3),
    ("ec3f6089c4566d49f30f0b3e7ed266897481cceae2d596125da85c291ffd7560", 1),
    ("faa62b1bca6ac6de900c2ff5f820c83115d8dfcf27543d462c56e7ca835f9339", 1),
    ("fb63e9413feeb3d64774d9175668fbc40ce3fe664021e70c6e9288d191e11330", 1),
)
EXPECTED_INPUT_ROOTS = tuple(row[0] for row in EXPECTED_INPUT_ROOT_MULTIPLICITIES)

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
_HEX = frozenset("0123456789abcdef")
_SPAN_KEYS = frozenset(
    {
        "start_token",
        "stop_token",
        "first_spike_step",
        "assembly_cells",
        "spike_cells",
    }
)
_LITERAL_KEYS = frozenset({"canonical_token_index", "typed_output_active_cells"})
_ROLE_KEYS = frozenset(
    {
        "parameters",
        "result",
        "role_binding",
        "entrypoint_spelling_present",
        "parameter_spelling_present",
    }
)
_EXPECTED_ROLE_RECORD = {
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
        raise ValueError(f"{label} schema changed")
    return value


def _strict_sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, label) for item in _strict_sequence(value, label)
    )


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FrozenPromptInput:
    """One unique physical input; ``multiplicity`` is audit-only provenance."""

    input_root: str
    multiplicity: int
    physical_span_events_sha256: str
    typed_literal_events_sha256: str
    signature_role_events_sha256: str
    raw_span_records: tuple[Mapping[str, object], ...]
    typed_literal_records: tuple[Mapping[str, object], ...]
    signature_role_record: Mapping[str, object]

    def neural_records(
        self,
    ) -> tuple[
        tuple[Mapping[str, object], ...],
        tuple[Mapping[str, object], ...],
        Mapping[str, object],
    ]:
        """Return only causal records, excluding roots and multiplicity."""

        return (
            self.raw_span_records,
            self.typed_literal_records,
            self.signature_role_record,
        )


@dataclass(frozen=True, slots=True)
class PromptInputCapsule:
    """Authenticated iterable and root-addressed lookup for frozen inputs."""

    artifact_sha256: str
    content_root: str
    source_raw_sha256: str
    appearance_count: int
    input_count: int
    inputs: tuple[FrozenPromptInput, ...]
    appearance_input_roots: tuple[str, ...]
    _by_root: Mapping[str, FrozenPromptInput]

    def iter_inputs(self) -> Iterator[FrozenPromptInput]:
        return iter(self.inputs)

    def lookup(self, input_root: str) -> FrozenPromptInput:
        """Resolve one authenticated causal input root, never a family label."""

        if type(input_root) is not str:
            raise TypeError("input root must be a string")
        try:
            return self._by_root[input_root]
        except KeyError as error:
            raise KeyError("unknown authenticated prompt input root") from error

    def iter_appearances(self) -> Iterator[FrozenPromptInput]:
        """Yield the canonical multiset without exposing source ordinals."""

        return (self._by_root[input_root] for input_root in self.appearance_input_roots)


def _normalize_spans(value: object) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for raw in _strict_sequence(value, "raw span records"):
        row = _strict_record(raw, _SPAN_KEYS, "raw span record")
        start = _integer(row["start_token"], "span start")
        stop = _integer(row["stop_token"], "span stop")
        first_spike = _integer(row["first_spike_step"], "span first spike")
        assembly = _integer_tuple(row["assembly_cells"], "span assembly cells")
        spikes = _integer_tuple(row["spike_cells"], "span spike cells")
        if start < 0 or stop <= start or first_spike < 0 or not assembly or assembly != spikes:
            raise ValueError("raw span physical event changed")
        rows.append(
            {
                "start_token": start,
                "stop_token": stop,
                "first_spike_step": first_spike,
                "assembly_cells": list(assembly),
                "spike_cells": list(spikes),
            }
        )
    if not rows:
        raise ValueError("physical prompt input has no span records")
    return tuple(rows)


def _normalize_literals(value: object) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for raw in _strict_sequence(value, "typed literal records"):
        row = _strict_record(raw, _LITERAL_KEYS, "typed literal record")
        token_index = _integer(row["canonical_token_index"], "literal token index")
        active = _integer_tuple(
            row["typed_output_active_cells"], "literal typed-output cells"
        )
        if token_index < 0 or not active:
            raise ValueError("typed literal physical event changed")
        rows.append(
            {
                "canonical_token_index": token_index,
                "typed_output_active_cells": list(active),
            }
        )
    return tuple(rows)


def _normalize_role(value: object) -> dict[str, object]:
    row = _strict_record(value, _ROLE_KEYS, "signature role record")
    plain = _plain(row)
    if plain != _EXPECTED_ROLE_RECORD:
        raise ValueError("name-free reverse-declaration signature role changed")
    return dict(plain)


def _normalize_input(value: object) -> FrozenPromptInput:
    row = _strict_record(
        value,
        frozenset(
            {
                "input_root",
                "multiplicity",
                "physical_span_events_sha256",
                "typed_literal_events_sha256",
                "signature_role_events_sha256",
                "raw_span_records",
                "typed_literal_records",
                "signature_role_record",
            }
        ),
        "physical prompt input",
    )
    input_root = _sha(row["input_root"], "input root")
    multiplicity = _integer(row["multiplicity"], "input multiplicity")
    if multiplicity <= 0:
        raise ValueError("input multiplicity must be positive")
    spans = _normalize_spans(row["raw_span_records"])
    literals = _normalize_literals(row["typed_literal_records"])
    role = _normalize_role(row["signature_role_record"])
    span_root = _namespaced_sha256(_SPAN_EVENTS_NAMESPACE, spans)
    literal_root = _namespaced_sha256(_LITERAL_EVENTS_NAMESPACE, literals)
    role_root = _namespaced_sha256(_SIGNATURE_ROLE_NAMESPACE, role)
    if (
        _sha(row["physical_span_events_sha256"], "span event root") != span_root
        or _sha(row["typed_literal_events_sha256"], "literal event root")
        != literal_root
        or _sha(row["signature_role_events_sha256"], "signature role root")
        != role_root
    ):
        raise ValueError("physical input channel root changed")
    causal_record = {
        "physical_span_events_sha256": span_root,
        "typed_literal_events_sha256": literal_root,
        "signature_role_events_sha256": role_root,
        "causal_sources": list(_CAUSAL_SOURCES),
    }
    if _namespaced_sha256(_CAUSAL_KEY_NAMESPACE, causal_record) != input_root:
        raise ValueError("physical causal input root changed")
    frozen_spans = tuple(_freeze(item) for item in spans)
    frozen_literals = tuple(_freeze(item) for item in literals)
    return FrozenPromptInput(
        input_root=input_root,
        multiplicity=multiplicity,
        physical_span_events_sha256=span_root,
        typed_literal_events_sha256=literal_root,
        signature_role_events_sha256=role_root,
        raw_span_records=frozen_spans,
        typed_literal_records=frozen_literals,
        signature_role_record=_freeze(role),
    )


@lru_cache(maxsize=4)
def _load_cached(path_text: str) -> PromptInputCapsule:
    path = Path(path_text)
    payload = path.read_bytes()
    artifact_sha = sha256(payload).hexdigest()
    if artifact_sha != ARTIFACT_SHA256:
        raise ValueError("public-TRAIN input artifact identity changed")
    artifact = _strict_record(
        json.loads(payload),
        frozenset(
            {
                "schema",
                "scope",
                "source_authority",
                "appearance_count",
                "input_count",
                "appearance_input_roots",
                "inputs",
                "content_root",
            }
        ),
        "public-TRAIN input artifact",
    )
    if artifact["schema"] != SCHEMA or artifact["scope"] != SCOPE:
        raise ValueError("public-TRAIN input artifact contract changed")
    source_authority = _strict_record(
        artifact["source_authority"],
        frozenset(
            {
                "prompt_signature_schema",
                "prompt_signature_scope",
                "prompt_signature_raw_sha256",
                "prompt_signature_record_count",
                "prompt_observer_source_sha256",
                "prompt_observer_tokenizer_sha256",
                "prompt_observer_assembly_projection_sha256",
            }
        ),
        "source authority",
    )
    expected_authority = {
        "prompt_signature_schema": SOURCE_SCHEMA,
        "prompt_signature_scope": SOURCE_SCOPE,
        "prompt_signature_raw_sha256": SOURCE_RAW_SHA256,
        "prompt_signature_record_count": SOURCE_RECORD_COUNT,
        "prompt_observer_source_sha256": OBSERVER_SOURCE_SHA256,
        "prompt_observer_tokenizer_sha256": OBSERVER_TOKENIZER_SHA256,
        "prompt_observer_assembly_projection_sha256": (
            OBSERVER_ASSEMBLY_PROJECTION_SHA256
        ),
    }
    if source_authority != expected_authority:
        raise ValueError("public-TRAIN input source authority changed")
    appearance_count = _integer(artifact["appearance_count"], "appearance count")
    input_count = _integer(artifact["input_count"], "input count")
    if appearance_count != APPEARANCE_COUNT or input_count != INPUT_COUNT:
        raise ValueError("public-TRAIN input counts changed")

    raw_inputs = _strict_sequence(artifact["inputs"], "physical prompt inputs")
    inputs = tuple(_normalize_input(row) for row in raw_inputs)
    roots = tuple(row.input_root for row in inputs)
    multiplicities = tuple((row.input_root, row.multiplicity) for row in inputs)
    if roots != EXPECTED_INPUT_ROOTS or multiplicities != EXPECTED_INPUT_ROOT_MULTIPLICITIES:
        raise ValueError("authenticated per-input roots or multiplicities changed")
    by_root = {row.input_root: row for row in inputs}
    if len(by_root) != INPUT_COUNT:
        raise ValueError("physical causal input roots are not one-to-one")
    appearance_roots = tuple(
        _sha(item, "appearance input root")
        for item in _strict_sequence(
            artifact["appearance_input_roots"], "appearance input roots"
        )
    )
    expected_appearances = tuple(
        root
        for root, multiplicity in EXPECTED_INPUT_ROOT_MULTIPLICITIES
        for _ in range(multiplicity)
    )
    if (
        appearance_roots != expected_appearances
        or Counter(appearance_roots)
        != Counter(dict(EXPECTED_INPUT_ROOT_MULTIPLICITIES))
    ):
        raise ValueError("source-order-free appearance multiset changed")

    content_root = _sha(artifact["content_root"], "artifact content root")
    without_root = dict(artifact)
    del without_root["content_root"]
    if (
        content_root != CONTENT_ROOT
        or _namespaced_sha256(_CONTENT_ROOT_NAMESPACE, without_root) != content_root
    ):
        raise ValueError("public-TRAIN input artifact content root changed")
    return PromptInputCapsule(
        artifact_sha256=artifact_sha,
        content_root=content_root,
        source_raw_sha256=SOURCE_RAW_SHA256,
        appearance_count=appearance_count,
        input_count=input_count,
        inputs=inputs,
        appearance_input_roots=appearance_roots,
        _by_root=MappingProxyType(by_root),
    )


def load_input_capsule(path: Path = ARTIFACT_PATH) -> PromptInputCapsule:
    """Load and deeply authenticate the immutable physical input capsule."""

    return _load_cached(str(Path(path).resolve()))


__all__ = [
    "APPEARANCE_COUNT",
    "ARTIFACT_PATH",
    "ARTIFACT_SHA256",
    "CONTENT_ROOT",
    "EXPECTED_INPUT_ROOTS",
    "EXPECTED_INPUT_ROOT_MULTIPLICITIES",
    "INPUT_COUNT",
    "SOURCE_RAW_SHA256",
    "FrozenPromptInput",
    "PromptInputCapsule",
    "load_input_capsule",
]
