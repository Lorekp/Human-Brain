"""Strict, checkpoint-free public-TRAIN prompt-factor manifest preflight.

This module implements only the first frozen layer of the prompt-to-factor
bridge.  It loads a source-bound capsule whose records contain exactly public
prompt text and typed signatures, projects those records into the banked
``PromptSpanInput`` representation, authenticates atomic factor targets, and
constructs the preregistered 72/18 split when identifiable.

The module has no learner arrays and performs no update.  It deliberately has
no dependency on Simple-SWE task generation, hidden cases, sandboxes, result
records, or program candidates.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.r3_pilot import public_train_prompt_semantic_bridge as prompt_spans
from experiments.r3_pilot import public_typed_output_cells as output_cells


CAPSULE_PATH = (
    REPO_ROOT
    / "experiments"
    / "r3_pilot"
    / "public_train_prompt_signature_capsule_v1.json"
)

CAPSULE_SCHEMA = "public-train-prompt-signature-capsule-v1"
CAPSULE_SCOPE = "PUBLIC_TRAIN_PROMPTS_AND_SIGNATURES_ONLY"
LESSON_SCHEMA = "public-train-prompt-factor-lessons-v1"
LESSON_SCOPE = "PUBLIC_TRAIN_PROMPT_FACTORS_ONLY"
SPLIT_SCHEMA = "public-train-prompt-factor-split-v1"

STOP_IDENTIFIABILITY = "STOP_PUBLIC_TRAIN_PROMPT_FACTOR_IDENTIFIABILITY_V1"
PREFLIGHT_READY = "PREFLIGHT_READY_NOT_RUN"

EXPECTED_PROMPT_APPEARANCES = 90
FIT_PROMPT_COUNT = 72
GENERALIZATION_PROMPT_COUNT = 18
MIN_ACTIVE_FACTOR_PROMPTS = 3
MIN_FIT_FACTOR_PROMPTS = 2
LEARNING_UPDATES = 0
LEARNER_ARRAY_BYTES = 0

CAPSULE_RAW_SHA256 = (
    "0f34d5c1e4b9e90f76f098bf3296af546495d8b6053586b1620ecd88ba5b3289"
)
CAPSULE_PAYLOAD_SHA256 = (
    "d64c61452d8a4fcd2e67a4deb4848c07cd93a757d36da2cfd2b90f81fa00a617"
)

FACTOR_REGISTRY = (
    "add",
    "subtract",
    "minimum",
    "maximum",
    "absolute",
    "less-than",
    "map",
    "sum",
    "count",
    "contains",
    "filter",
    "reverse",
    "class:ascii-alphanumeric",
    "class:ascii-decimal-digits",
    "class:ascii-hex-digits",
    "class:ascii-letters",
    "class:ascii-lowercase",
    "class:ascii-octal-digits",
    "class:ascii-printable",
    "class:ascii-punctuation",
    "class:ascii-uppercase",
    "class:ascii-whitespace",
    "class:english-consonants",
    "class:english-vowels",
)
_FACTOR_INDEX = MappingProxyType(
    {factor: index for index, factor in enumerate(FACTOR_REGISTRY)}
)

FORBIDDEN_FIELD_FRAGMENTS = (
    "task",
    "family",
    "ordinal",
    "desired",
    "output_case",
    "regression",
    "property",
    "candidate",
    "program",
    "ast_path",
    "dev",
    "eval",
    "m0",
    "heldout",
    "c1",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _namespaced_sha256(namespace: bytes, value: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


FACTOR_REGISTRY_SHA256 = _namespaced_sha256(
    b"public-train-prompt-factor-registry-v1\0",
    list(FACTOR_REGISTRY),
)
INPUT_PROJECTION_SHA256 = _namespaced_sha256(
    b"public-train-prompt-factor-input-projection-v1\0",
    {
        "scope": prompt_spans.PUBLIC_TRAIN_INPUT_SCOPE,
        "tokenizer_sha256": prompt_spans.TOKENIZER_SHA256,
        "assembly_projection_sha256": prompt_spans.ASSEMBLY_PROJECTION_SHA256,
        "typed_output_projection_sha256": (
            output_cells.TypedOutputCellBank.projection_sha256()
        ),
    },
)

EXPECTED_FACTOR_REGISTRY_SHA256 = (
    "f7df546c7c89c20d48c79c737dab8fcb101d9c861cb041c0c39654268ecb28c6"
)
EXPECTED_INPUT_PROJECTION_SHA256 = (
    "2108f4864cfbbcd3f44be96ad7b55d95a89d8d4899c824ec913204b5318320d6"
)
EXPECTED_LESSON_MANIFEST_SHA256 = (
    "139e678a6001f5264d8e6ba44609e2b97082cb8b7c443462326c370d4fc5e774"
)
EXPECTED_PREFLIGHT_PAYLOAD_SHA256 = (
    "c4dd1d14b57c9359691f8077742f41b2fe4277e6cca414b5b2f60f94871459cd"
)

if FACTOR_REGISTRY_SHA256 != EXPECTED_FACTOR_REGISTRY_SHA256:
    raise RuntimeError("frozen prompt-factor registry identity changed")
if INPUT_PROJECTION_SHA256 != EXPECTED_INPUT_PROJECTION_SHA256:
    raise RuntimeError("frozen prompt input projection identity changed")


class ManifestValidationError(ValueError):
    """A serialized record crossed or weakened the frozen interface."""


class IdentifiabilityStop(RuntimeError):
    """The exact preregistered split cannot be constructed."""


@dataclass(frozen=True, slots=True)
class FactorTarget:
    start_token: int
    stop_token: int
    span_identity_sha256: str
    factor: str

    def to_record(self) -> dict[str, object]:
        return {
            "start_token": self.start_token,
            "stop_token": self.stop_token,
            "span_identity_sha256": self.span_identity_sha256,
            "factor": self.factor,
        }


@dataclass(frozen=True, slots=True)
class FactorLesson:
    prompt_input_sha256: str
    targets: tuple[FactorTarget, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "prompt_input_sha256": self.prompt_input_sha256,
            "targets": [target.to_record() for target in self.targets],
        }


@dataclass(frozen=True, slots=True)
class LessonManifest:
    lessons: tuple[FactorLesson, ...]
    sha256: str

    def to_record(self) -> dict[str, object]:
        return {
            "schema": LESSON_SCHEMA,
            "scope": LESSON_SCOPE,
            "input_projection_sha256": INPUT_PROJECTION_SHA256,
            "factor_registry_sha256": FACTOR_REGISTRY_SHA256,
            "lessons": [lesson.to_record() for lesson in self.lessons],
        }


@dataclass(frozen=True, slots=True)
class SplitManifest:
    fit_prompt_input_sha256s: tuple[str, ...]
    generalization_prompt_input_sha256s: tuple[str, ...]
    sha256: str

    def to_record(self, lesson_manifest_sha256: str) -> dict[str, object]:
        return {
            "schema": SPLIT_SCHEMA,
            "scope": LESSON_SCOPE,
            "lesson_manifest_sha256": lesson_manifest_sha256,
            "algorithm": {
                "prompt_appearance_count": EXPECTED_PROMPT_APPEARANCES,
                "fit_prompt_count": FIT_PROMPT_COUNT,
                "generalization_prompt_count": GENERALIZATION_PROMPT_COUNT,
                "minimum_active_factor_prompts": MIN_ACTIVE_FACTOR_PROMPTS,
                "minimum_fit_factor_prompts": MIN_FIT_FACTOR_PROMPTS,
                "coverage_reservation_order": "factor-registry-order",
                "fill_order": "lowercase-prompt-input-sha256",
            },
            "fit_prompt_input_sha256s": list(self.fit_prompt_input_sha256s),
            "generalization_prompt_input_sha256s": list(
                self.generalization_prompt_input_sha256s
            ),
        }


@dataclass(frozen=True, slots=True)
class ManifestPreflight:
    status: str
    lesson_manifest_sha256: str
    split_manifest_sha256: str | None
    prompt_appearance_count: int
    unique_prompt_input_count: int
    duplicate_prompt_appearance_count: int
    factor_appearance_counts: tuple[tuple[str, int], ...]
    factor_distinct_input_counts: tuple[tuple[str, int], ...]
    reasons: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "schema": "public-train-prompt-factor-manifest-preflight-v1",
            "scope": LESSON_SCOPE,
            "status": self.status,
            "capsule_raw_sha256": CAPSULE_RAW_SHA256,
            "capsule_payload_sha256": CAPSULE_PAYLOAD_SHA256,
            "input_projection_sha256": INPUT_PROJECTION_SHA256,
            "factor_registry_sha256": FACTOR_REGISTRY_SHA256,
            "lesson_manifest_sha256": self.lesson_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "prompt_appearance_count": self.prompt_appearance_count,
            "unique_prompt_input_count": self.unique_prompt_input_count,
            "duplicate_prompt_appearance_count": (
                self.duplicate_prompt_appearance_count
            ),
            "factor_appearance_counts": dict(self.factor_appearance_counts),
            "factor_distinct_input_counts": dict(
                self.factor_distinct_input_counts
            ),
            "reasons": list(self.reasons),
            "lesson_manifest_authenticated": True,
            "prompt_span_targets_authenticated": True,
            "split_constructed": self.split_manifest_sha256 is not None,
            "checkpoint_created": False,
            "learner_array_bytes": LEARNER_ARRAY_BYTES,
            "learning_updates": LEARNING_UPDATES,
        }


def _is_sha256(value: object) -> bool:
    if type(value) is not str or len(value) != 64 or value != value.lower():
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _reject_forbidden_fields(value: object, location: str = "record") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ManifestValidationError(f"{location} has a non-string field")
            lowered = key.casefold()
            if any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
                raise ManifestValidationError(
                    f"{location} contains forbidden field {key!r}"
                )
            _reject_forbidden_fields(child, f"{location}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{location}[{index}]")


def _strict_record(
    value: object,
    expected_fields: frozenset[str],
    location: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ManifestValidationError(f"{location} must be an object")
    _reject_forbidden_fields(value, location)
    if set(value) != expected_fields:
        raise ManifestValidationError(f"{location} fields differ from the freeze")
    return value


def _target_sort_key(target: FactorTarget) -> tuple[int, int, int]:
    return (
        target.start_token,
        target.stop_token,
        _FACTOR_INDEX[target.factor],
    )


def _parse_target(value: object, location: str) -> FactorTarget:
    record = _strict_record(
        value,
        frozenset(
            {"start_token", "stop_token", "span_identity_sha256", "factor"}
        ),
        location,
    )
    start = record["start_token"]
    stop = record["stop_token"]
    factor = record["factor"]
    identity = record["span_identity_sha256"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
        or stop - start > prompt_spans.MAX_SPAN_TOKENS
    ):
        raise ManifestValidationError(f"{location} token interval is invalid")
    if factor not in _FACTOR_INDEX:
        raise ManifestValidationError(f"{location} factor is outside the registry")
    if not _is_sha256(identity):
        raise ManifestValidationError(f"{location} span identity is invalid")
    return FactorTarget(
        start_token=start,
        stop_token=stop,
        span_identity_sha256=identity,
        factor=factor,
    )


def parse_lesson_manifest(value: object) -> LessonManifest:
    """Strictly parse the frozen serialized lesson interface."""

    record = _strict_record(
        value,
        frozenset(
            {
                "schema",
                "scope",
                "input_projection_sha256",
                "factor_registry_sha256",
                "lessons",
            }
        ),
        "lesson manifest",
    )
    if record["schema"] != LESSON_SCHEMA or record["scope"] != LESSON_SCOPE:
        raise ManifestValidationError("lesson manifest schema or scope changed")
    if record["input_projection_sha256"] != INPUT_PROJECTION_SHA256:
        raise ManifestValidationError("lesson manifest input projection changed")
    if record["factor_registry_sha256"] != FACTOR_REGISTRY_SHA256:
        raise ManifestValidationError("lesson manifest factor registry changed")
    raw_lessons = record["lessons"]
    if type(raw_lessons) is not list or not raw_lessons:
        raise ManifestValidationError("lesson manifest requires a non-empty list")

    lessons: list[FactorLesson] = []
    for lesson_index, raw_lesson in enumerate(raw_lessons):
        lesson_record = _strict_record(
            raw_lesson,
            frozenset({"prompt_input_sha256", "targets"}),
            f"lesson[{lesson_index}]",
        )
        prompt_input_sha = lesson_record["prompt_input_sha256"]
        if not _is_sha256(prompt_input_sha):
            raise ManifestValidationError(
                f"lesson[{lesson_index}] prompt input identity is invalid"
            )
        raw_targets = lesson_record["targets"]
        if type(raw_targets) is not list or not raw_targets:
            raise ManifestValidationError(
                f"lesson[{lesson_index}] requires atomic targets"
            )
        targets = tuple(
            _parse_target(target, f"lesson[{lesson_index}].targets[{target_index}]")
            for target_index, target in enumerate(raw_targets)
        )
        if targets != tuple(sorted(targets, key=_target_sort_key)):
            raise ManifestValidationError(
                f"lesson[{lesson_index}] targets are not canonically ordered"
            )
        if len(targets) != len(set(targets)):
            raise ManifestValidationError(
                f"lesson[{lesson_index}] contains duplicate targets"
            )
        lessons.append(FactorLesson(prompt_input_sha, targets))

    canonical_record = {
        "schema": LESSON_SCHEMA,
        "scope": LESSON_SCOPE,
        "input_projection_sha256": INPUT_PROJECTION_SHA256,
        "factor_registry_sha256": FACTOR_REGISTRY_SHA256,
        "lessons": [lesson.to_record() for lesson in lessons],
    }
    manifest_sha = sha256(canonical_json_bytes(canonical_record)).hexdigest()
    return LessonManifest(tuple(lessons), manifest_sha)


def _load_capsule_record(path: str | Path = CAPSULE_PATH) -> dict[str, object]:
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    if sha256(raw).hexdigest() != CAPSULE_RAW_SHA256:
        raise ManifestValidationError("public prompt capsule raw identity changed")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ManifestValidationError("public prompt capsule is not JSON") from error
    if sha256(canonical_json_bytes(value)).hexdigest() != CAPSULE_PAYLOAD_SHA256:
        raise ManifestValidationError("public prompt capsule payload identity changed")
    return _strict_record(
        value,
        frozenset({"schema", "scope", "records"}),
        "public prompt capsule",
    )


def load_public_train_prompt_inputs(
    path: str | Path = CAPSULE_PATH,
) -> tuple[prompt_spans.PromptSpanInput, ...]:
    """Parse the strict external capsule and return only physical inputs."""

    capsule = _load_capsule_record(path)
    if capsule["schema"] != CAPSULE_SCHEMA or capsule["scope"] != CAPSULE_SCOPE:
        raise ManifestValidationError("public prompt capsule schema or scope changed")
    raw_records = capsule["records"]
    if type(raw_records) is not list or len(raw_records) != EXPECTED_PROMPT_APPEARANCES:
        raise ManifestValidationError("public prompt capsule appearance count changed")

    observed: list[prompt_spans.PromptSpanInput] = []
    for index, raw_record in enumerate(raw_records):
        record = _strict_record(
            raw_record,
            frozenset({"prompt", "signature"}),
            f"public prompt capsule records[{index}]",
        )
        prompt = record["prompt"]
        if type(prompt) is not str or not prompt:
            raise ManifestValidationError("public prompt capsule prompt is invalid")
        try:
            signature = prompt_spans.TypedSignature.from_record(record["signature"])
        except (TypeError, ValueError) as error:
            raise ManifestValidationError(
                "public prompt capsule typed signature is invalid"
            ) from error
        observed.append(prompt_spans.observe_public_train_prompt(prompt, signature))
    return tuple(observed)


@dataclass(frozen=True, slots=True)
class _TargetSpec:
    start_token: int
    stop_token: int
    span_identity_sha256: str
    factor: str

    def target(self) -> FactorTarget:
        return FactorTarget(
            self.start_token,
            self.stop_token,
            self.span_identity_sha256,
            self.factor,
        )


def _spec(
    factor: str,
    start_token: int,
    stop_token: int,
    span_identity_sha256: str,
) -> _TargetSpec:
    return _TargetSpec(
        start_token=start_token,
        stop_token=stop_token,
        span_identity_sha256=span_identity_sha256,
        factor=factor,
    )


_SCALAR_CLAMP = (
    _spec("minimum", 8, 12, "bf72cd5d37d2b59fb1a830f921221d507bb1a2d4d15c1ccd62f0fab3d3c5f214"),
    _spec("maximum", 8, 12, "bf72cd5d37d2b59fb1a830f921221d507bb1a2d4d15c1ccd62f0fab3d3c5f214"),
)
_SCALAR_ADD = (
    _spec("add", 8, 12, "a4147c76eed7bce453f000c4fcb991bc80be0df15a701b3d6264fa7847ca9eac"),
)
_SCALAR_NONNEGATIVE = (
    _spec("maximum", 11, 12, "c67cdc2b7aec3ab6d63257d9c5999da06c714ee6098fd5b5f6005a4bcf2d4909"),
)
_LIST_MAP_ADD = (
    _spec("add", 12, 16, "e08d64f75992b81ef58e54ba2d4bf51df863e3636307b13ca4eea0aa2aa77236"),
    _spec("map", 14, 16, "478d27879ed45783338669e00ab9d3cf4ec9670bd73ee35f9c80af5715abda65"),
)
_LIST_MAP_ABSOLUTE = (
    _spec("absolute", 12, 16, "af7971cc565cc4b27e4812c3dfd7c8c030ef524fe240ca181db7bf34a6dbfab5"),
    _spec("map", 15, 17, "ffa0729974ba5c52670bd910332eabf2dc250dd9cbf4fdf15f40b21dff7d6fd7"),
)
_CLAMP_THEN_SUM = (
    _spec("minimum", 6, 10, "f41a2c3ef24cfacf12c8b2a14aa203daac3a3d42fcd208c7b9fa050a02147e95"),
    _spec("maximum", 6, 10, "f41a2c3ef24cfacf12c8b2a14aa203daac3a3d42fcd208c7b9fa050a02147e95"),
    _spec("map", 7, 9, "478d27879ed45783338669e00ab9d3cf4ec9670bd73ee35f9c80af5715abda65"),
    _spec("sum", 15, 18, "d3a63ed53f038b87a65202bd5d21b5d02fa06b16d56059423d9b1263401dccc8"),
)
_MAP_THEN_COUNT = (
    _spec("add", 6, 10, "255faf9892026d7850becb6156a4069d898b9da907b2359155e1088dbe1db9b7"),
    _spec("map", 9, 11, "ffa0729974ba5c52670bd910332eabf2dc250dd9cbf4fdf15f40b21dff7d6fd7"),
    _spec("count", 12, 14, "af8b9568f27a201399d1028b21c0ddacbcab9758ff3484c7319297c8599e8bbe"),
    _spec("less-than", 14, 17, "8d71231c9cfb399092b1f20ce4cdedf0334ed0414bcb966c1d48392d6c556ff5"),
)
_FILTER_DECIMAL = (
    _spec("filter", 6, 8, "8fb3306e7280b623d902ce69e03747c5c1760e3e9f0dda2ad79b3f9bdeb0726f"),
    _spec("class:ascii-decimal-digits", 8, 11, "6a017054daf00dd0036af326f9e4c1ea67faae753c3e2580535c513f20ea72b5"),
)
_FILTER_LOWERCASE = (
    _spec("filter", 6, 8, "8fb3306e7280b623d902ce69e03747c5c1760e3e9f0dda2ad79b3f9bdeb0726f"),
    _spec("class:ascii-lowercase", 8, 11, "7e3df3de6ced2f5dd05e4336a5c8a8cca1acdef23371d054d7cbc304dc8feffb"),
)
_FILTER_VOWELS = (
    _spec("filter", 6, 8, "8fb3306e7280b623d902ce69e03747c5c1760e3e9f0dda2ad79b3f9bdeb0726f"),
    _spec("class:english-vowels", 8, 10, "ef293a0d14a7c84e41ab268d9b56bfe680c048b6a6b554f520881ccf89d4f4bd"),
)
_FILTER_REVERSE_DECIMAL = (
    *_FILTER_DECIMAL,
    _spec("reverse", 14, 18, "301fd9b2d43f09666fb9e7f5845714cf70eb80587b32b94e137c230fb34e31da"),
)
_FILTER_REVERSE_LOWERCASE = (
    *_FILTER_LOWERCASE,
    _spec("reverse", 14, 18, "301fd9b2d43f09666fb9e7f5845714cf70eb80587b32b94e137c230fb34e31da"),
)
_FILTER_REVERSE_VOWELS = (
    *_FILTER_VOWELS,
    _spec("reverse", 13, 17, "301fd9b2d43f09666fb9e7f5845714cf70eb80587b32b94e137c230fb34e31da"),
)


def _annotation_map() -> Mapping[str, tuple[_TargetSpec, ...]]:
    groups: tuple[tuple[tuple[str, ...], tuple[_TargetSpec, ...]], ...] = (
        ((
            "0b1b35530cf1cd008dbf1f06c9b12b5bdb4f7f21116800eb41f145565e7218db",
            "44d4833bf08a64132d580a3b4177eb7a519a5e332437276c4cfcaa90acdaa6c6",
            "819f05c7981087f36c02d8fb77aaa262f814bbf3f3f054ffed470053fa081613",
            "92574fe36c00ab573f262c9e2e7fb6efe8b70334a63f2febb7a6e3513a97412c",
            "c1822ecc1eed61b2fd345a3da91d8ece36e8df89fe9ddef05c7ec0cadac8f416",
        ), _SCALAR_CLAMP),
        ((
            "1ea63f6a9e51f0ba7c53332f6bb6afa257c1e0c9e6ac8d0d7a2c6c3105a0350a",
            "3c6d723ca3674df3a00a2bb3209ac91c532d6556154a302d133dd790d58b3417",
            "93b50c88e382800341b4df95eb2c266d21e049033e7a124d4aa355bc9f1e3830",
            "9c37e4ec14e765097f2ed3b9ce639758611b401d128112b478ef1ed8e17d5db1",
            "a69c93d115f0c21b0fd15393205ab280f3d6a94d71a1e24ca2fc10d9aa69de70",
            "ada96e9f27df9673cf5403da639d2c00dc7601404eb521d1e75c7a3e1f727981",
            "b0e11966bc4d8401403bdd8911f88d5ca2fca5cd089cb632a4f2dca3412b1057",
            "c13c6983dcc883f18221f1bb498804b3bfd328aab5e9f801553f2d0735b36f5f",
            "ef74c4faa42df9fe13830226617d93ea9a0ce426b57657babfd57aad5c1c34e8",
        ), _SCALAR_ADD),
        (("bb19ef6d812310fc149c7674e202255f4deec260ae6815b455849183b0c241b9",), _SCALAR_NONNEGATIVE),
        ((
            "8ae6fe3144d91c4eba12deaae79953a391399c21a2612d1edca63d840c4f2b3c",
            "a5384e5aaf19583709419e4d18a1c9e36f0b41fdcd5257f6b8ade3c545a7d453",
            "ca5b99de021d754bd33999d7f59b18a2e2fd4132afb1559404ad9b8b890e750b",
            "cce640ee89e2c02c5b3749e1bdde528d822b825fad77d65d964c0d041a927628",
            "f81c12c1435d040d5fdd6977ca98269f184ac6ba3f0513c6c715d12b371835f1",
        ), _LIST_MAP_ADD),
        (("e673f87f51571c9b3b11d4e81aac52a993e11a6ca9d7155e57e43b129c41c1f5",), _LIST_MAP_ABSOLUTE),
        ((
            "2b98f16e11011af3a4a43e467fd032b38191eed195ee4dc397e462848b12686c",
            "449c4e8e502b9e0eb9454c4f85d6618cdcd768ad03976d3d28486727b05cb596",
            "89772cfe0acde474c4bd561aff4620d70d2f99a29b7ad595bfd38f50d01178ee",
            "90c8125972221dd26bfcda51f78ad12bfa2678a7ef8e2712959ad837eaeddbff",
            "9dc96b833e02ced1f9bd681ecdd3c2d5e982eec8fcee427a772569dcb419b2e0",
            "e8d84713fea7c2d0057368bea1e75aa3dd2eeaa1c685e9dc675de41428dd288d",
            "fc3c43134eeaea4c920f1f31f15e04da1081b3ea5fb29c492d3058961c518224",
        ), _CLAMP_THEN_SUM),
        ((
            "0eb123c8a8a5d799a359dd1a5cd8a0f36f8a30cffbccaf2b1c9fa676035a3a1c",
            "17bffc18f60edf862f6ac6b141fbdbc8522b30d3b78a699fe229f6d13fafca48",
            "53b970f9f2bc8cb840b6d1d13e71977c62b9cb203888374cc9fb170bac05a796",
            "78e892eb3ea2d16eca25293096ab61793a051cbbc8d30a72c250e5b2b753f0dc",
            "b54c3f40cf170b70da6fb073b2bd825bcca12b9508dcb12900a29ae8ef2c295d",
            "d67e923ca43060115634a1b26ea63b0c745723b963e45d48ab320848f1dd27c6",
            "e99695895a675fd842c92a151fd8a06f257dfd1a8410f5f04178cc6a84187684",
            "ead6b56f7281b339824ae963e9c7de1764d743b862fbb7a98bac84d094b35ef6",
        ), _MAP_THEN_COUNT),
        (("fccd2b5c2b1538c0f6374c4d26faff3f75722773fd9ad9ebd7922a5f3381e0dd",), _FILTER_DECIMAL),
        (("c11291cc43f2cd38d2491b6ec4a4da3f45fe97f8b792fe01029d15835002b0e8",), _FILTER_LOWERCASE),
        (("915f0c59431bced6f285bd3e276776652d8fa375f6ed6c3b5fd30a4a6753b6ff",), _FILTER_VOWELS),
        (("86a085f40de0125e2e9246fc1f0e50ca1dfcc15cabbaa2cf05dbf2eb2f91348b",), _FILTER_REVERSE_DECIMAL),
        (("88a0aab410711043014c050ec9a4f1fb31628d5d7838a81c5169dbb2bf307bbd",), _FILTER_REVERSE_LOWERCASE),
        (("cf798e16f3e6e7ef6dc6466c3e915cb6587e4c1c04c695800050a7ac3d8dffed",), _FILTER_REVERSE_VOWELS),
    )
    annotations: dict[str, tuple[_TargetSpec, ...]] = {}
    for identities, targets in groups:
        for identity in identities:
            if identity in annotations:
                raise RuntimeError("frozen prompt annotation identity is duplicated")
            annotations[identity] = targets
    if len(annotations) != 42:
        raise RuntimeError("frozen prompt annotation registry changed")
    return MappingProxyType(annotations)


_TARGETS_BY_PROMPT_INPUT_SHA256 = _annotation_map()


def build_lesson_manifest_record(
    prompt_inputs: Sequence[prompt_spans.PromptSpanInput],
) -> dict[str, object]:
    """Build the source-frozen 90-appearance atomic annotation record."""

    lessons: list[dict[str, object]] = []
    for prompt_input in prompt_inputs:
        if not isinstance(prompt_input, prompt_spans.PromptSpanInput):
            raise TypeError("lesson manifest accepts PromptSpanInput values only")
        specs = _TARGETS_BY_PROMPT_INPUT_SHA256.get(prompt_input.sha256)
        if specs is None:
            raise IdentifiabilityStop(
                "atomic public-prompt annotation is absent from the freeze"
            )
        targets = tuple(spec.target() for spec in specs)
        lessons.append(
            FactorLesson(prompt_input.sha256, targets).to_record()
        )
    return {
        "schema": LESSON_SCHEMA,
        "scope": LESSON_SCOPE,
        "input_projection_sha256": INPUT_PROJECTION_SHA256,
        "factor_registry_sha256": FACTOR_REGISTRY_SHA256,
        "lessons": lessons,
    }


def _semantic_input_identity(prompt_input: prompt_spans.PromptSpanInput) -> object:
    return (
        prompt_input.scope,
        prompt_input.trainable_state_bytes,
        prompt_input.tokenizer_sha256,
        prompt_input.assembly_projection_sha256,
        prompt_input.typed_output_projection_sha256,
        prompt_input.tokens,
        tuple(literal.semantic_sha256 for literal in prompt_input.signed_literals),
        tuple(
            (
                span.start_token,
                span.stop_token,
                span.identity_sha256,
                span.assembly_cells,
                span.physical_drive_sha256,
            )
            for span in prompt_input.spans
        ),
        prompt_input.lexical_sha256,
        prompt_input.literal_channel_sha256,
        prompt_input.sha256,
    )


def authenticate_lesson_targets(
    manifest: LessonManifest,
    prompt_inputs: Sequence[prompt_spans.PromptSpanInput],
) -> None:
    """Authenticate every target against already-observed physical inputs."""

    inputs = tuple(prompt_inputs)
    if not all(isinstance(value, prompt_spans.PromptSpanInput) for value in inputs):
        raise TypeError("lesson authentication accepts PromptSpanInput values only")
    if any(
        value.scope != prompt_spans.PUBLIC_TRAIN_INPUT_SCOPE
        or value.trainable_state_bytes != 0
        or value.tokenizer_sha256 != prompt_spans.TOKENIZER_SHA256
        or value.assembly_projection_sha256
        != prompt_spans.ASSEMBLY_PROJECTION_SHA256
        or value.typed_output_projection_sha256
        != output_cells.TypedOutputCellBank.projection_sha256()
        for value in inputs
    ):
        raise ManifestValidationError("prompt input projection changed")

    lesson_multiplicity = Counter(
        lesson.prompt_input_sha256 for lesson in manifest.lessons
    )
    input_multiplicity = Counter(value.sha256 for value in inputs)
    if lesson_multiplicity != input_multiplicity:
        raise ManifestValidationError("lesson and prompt-input multiplicities differ")

    grouped_inputs: dict[str, list[prompt_spans.PromptSpanInput]] = defaultdict(list)
    for value in inputs:
        grouped_inputs[value.sha256].append(value)
    for identity, values in grouped_inputs.items():
        semantic = _semantic_input_identity(values[0])
        if any(_semantic_input_identity(value) != semantic for value in values[1:]):
            raise ManifestValidationError(
                f"prompt input {identity} has a semantic identity collision"
            )

    grouped_lessons: dict[str, list[FactorLesson]] = defaultdict(list)
    for lesson in manifest.lessons:
        grouped_lessons[lesson.prompt_input_sha256].append(lesson)
    for identity, lessons in grouped_lessons.items():
        if any(lesson.targets != lessons[0].targets for lesson in lessons[1:]):
            raise ManifestValidationError(
                f"prompt input {identity} has contradictory atomic targets"
            )
        representative = grouped_inputs[identity][0]
        spans = {
            (span.start_token, span.stop_token): span
            for span in representative.spans
        }
        for target in lessons[0].targets:
            span = spans.get((target.start_token, target.stop_token))
            if span is None or span.identity_sha256 != target.span_identity_sha256:
                raise ManifestValidationError(
                    f"prompt input {identity} target span does not authenticate"
                )


def _factor_counts(
    manifest: LessonManifest,
) -> tuple[Counter[str], Counter[str]]:
    appearances: Counter[str] = Counter()
    factors_by_input: dict[str, set[str]] = defaultdict(set)
    for lesson in manifest.lessons:
        lesson_factors = {target.factor for target in lesson.targets}
        appearances.update(lesson_factors)
        factors_by_input[lesson.prompt_input_sha256].update(lesson_factors)
    distinct: Counter[str] = Counter()
    for factors in factors_by_input.values():
        distinct.update(factors)
    return appearances, distinct


def construct_split_manifest(manifest: LessonManifest) -> SplitManifest:
    """Apply the frozen coverage-first deterministic 72/18 algorithm."""

    identities = [lesson.prompt_input_sha256 for lesson in manifest.lessons]
    if len(identities) != EXPECTED_PROMPT_APPEARANCES:
        raise IdentifiabilityStop("the lesson manifest does not contain 90 prompts")
    if len(set(identities)) != EXPECTED_PROMPT_APPEARANCES:
        raise IdentifiabilityStop("the 90 lesson prompt identities are not unique")

    factors_by_input = {
        lesson.prompt_input_sha256: {target.factor for target in lesson.targets}
        for lesson in manifest.lessons
    }
    active = tuple(
        factor
        for factor in FACTOR_REGISTRY
        if any(factor in factors for factors in factors_by_input.values())
    )
    totals = {
        factor: sum(factor in factors for factors in factors_by_input.values())
        for factor in active
    }
    if any(total < MIN_ACTIVE_FACTOR_PROMPTS for total in totals.values()):
        raise IdentifiabilityStop("an active factor has fewer than three prompts")

    ordered = tuple(sorted(identities, key=str.lower))
    generalization: set[str] = set()

    def fit_count_after(selection: str, factor: str) -> int:
        removed = generalization | {selection}
        return sum(
            factor in factors_by_input[identity]
            for identity in ordered
            if identity not in removed
        )

    def admissible(selection: str) -> bool:
        return all(
            fit_count_after(selection, factor) >= MIN_FIT_FACTOR_PROMPTS
            for factor in factors_by_input[selection]
            if factor in active
        )

    for factor in active:
        if any(
            factor in factors_by_input[identity]
            for identity in generalization
        ):
            continue
        selection = next(
            (
                identity
                for identity in ordered
                if identity not in generalization
                and factor in factors_by_input[identity]
                and admissible(identity)
            ),
            None,
        )
        if selection is None:
            raise IdentifiabilityStop(
                f"factor {factor} cannot reserve a generalization prompt"
            )
        generalization.add(selection)
        if len(generalization) > GENERALIZATION_PROMPT_COUNT:
            raise IdentifiabilityStop("coverage reservations exceed 18 prompts")

    for identity in ordered:
        if len(generalization) == GENERALIZATION_PROMPT_COUNT:
            break
        if identity not in generalization and admissible(identity):
            generalization.add(identity)
    if len(generalization) != GENERALIZATION_PROMPT_COUNT:
        raise IdentifiabilityStop("the deterministic split cannot select 18 prompts")

    generalization_ids = tuple(
        identity for identity in ordered if identity in generalization
    )
    fit_ids = tuple(identity for identity in ordered if identity not in generalization)
    if len(fit_ids) != FIT_PROMPT_COUNT:
        raise IdentifiabilityStop("the deterministic split does not leave 72 prompts")

    record = {
        "schema": SPLIT_SCHEMA,
        "scope": LESSON_SCOPE,
        "lesson_manifest_sha256": manifest.sha256,
        "algorithm": {
            "prompt_appearance_count": EXPECTED_PROMPT_APPEARANCES,
            "fit_prompt_count": FIT_PROMPT_COUNT,
            "generalization_prompt_count": GENERALIZATION_PROMPT_COUNT,
            "minimum_active_factor_prompts": MIN_ACTIVE_FACTOR_PROMPTS,
            "minimum_fit_factor_prompts": MIN_FIT_FACTOR_PROMPTS,
            "coverage_reservation_order": "factor-registry-order",
            "fill_order": "lowercase-prompt-input-sha256",
        },
        "fit_prompt_input_sha256s": list(fit_ids),
        "generalization_prompt_input_sha256s": list(generalization_ids),
    }
    split_sha = sha256(canonical_json_bytes(record)).hexdigest()
    return SplitManifest(fit_ids, generalization_ids, split_sha)


def preflight_prompt_factor_manifest(
    value: object,
    prompt_inputs: Sequence[prompt_spans.PromptSpanInput],
) -> ManifestPreflight:
    """Authenticate the manifest and stop before any learner allocation."""

    manifest = parse_lesson_manifest(value)
    authenticate_lesson_targets(manifest, prompt_inputs)
    identities = tuple(
        lesson.prompt_input_sha256 for lesson in manifest.lessons
    )
    appearances, distinct = _factor_counts(manifest)

    reasons: list[str] = []
    if len(identities) != EXPECTED_PROMPT_APPEARANCES:
        reasons.append(
            f"expected 90 prompt appearances; observed {len(identities)}"
        )
    unique_count = len(set(identities))
    if unique_count != EXPECTED_PROMPT_APPEARANCES:
        reasons.append(
            "expected 90 unique prompt-input identities; "
            f"observed {unique_count} across {len(identities)} appearances"
        )
    for factor in FACTOR_REGISTRY:
        if 0 < distinct[factor] < MIN_ACTIVE_FACTOR_PROMPTS:
            reasons.append(
                f"active factor {factor} occurs in {distinct[factor]} distinct "
                "prompt inputs; requires at least 3"
            )

    split: SplitManifest | None = None
    if not reasons:
        try:
            split = construct_split_manifest(manifest)
        except IdentifiabilityStop as error:
            reasons.append(str(error))

    return ManifestPreflight(
        status=STOP_IDENTIFIABILITY if reasons else PREFLIGHT_READY,
        lesson_manifest_sha256=manifest.sha256,
        split_manifest_sha256=None if split is None else split.sha256,
        prompt_appearance_count=len(identities),
        unique_prompt_input_count=unique_count,
        duplicate_prompt_appearance_count=len(identities) - unique_count,
        factor_appearance_counts=tuple(
            (factor, appearances[factor]) for factor in FACTOR_REGISTRY
        ),
        factor_distinct_input_counts=tuple(
            (factor, distinct[factor]) for factor in FACTOR_REGISTRY
        ),
        reasons=tuple(reasons),
    )


def canonical_public_train_preflight() -> ManifestPreflight:
    """Run the sole canonical capsule-to-identifiability preflight."""

    prompt_inputs = load_public_train_prompt_inputs()
    record = build_lesson_manifest_record(prompt_inputs)
    manifest_sha = sha256(canonical_json_bytes(record)).hexdigest()
    if manifest_sha != EXPECTED_LESSON_MANIFEST_SHA256:
        raise ManifestValidationError("lesson manifest identity changed")
    result = preflight_prompt_factor_manifest(record, prompt_inputs)
    if result.lesson_manifest_sha256 != EXPECTED_LESSON_MANIFEST_SHA256:
        raise ManifestValidationError("parsed lesson manifest identity changed")
    if (
        sha256(canonical_json_bytes(result.to_record())).hexdigest()
        != EXPECTED_PREFLIGHT_PAYLOAD_SHA256
    ):
        raise ManifestValidationError("canonical preflight identity changed")
    return result


def main() -> int:
    result = canonical_public_train_preflight()
    print(canonical_json_bytes(result.to_record()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPSULE_PAYLOAD_SHA256",
    "CAPSULE_RAW_SHA256",
    "EXPECTED_LESSON_MANIFEST_SHA256",
    "EXPECTED_PREFLIGHT_PAYLOAD_SHA256",
    "FACTOR_REGISTRY",
    "FACTOR_REGISTRY_SHA256",
    "FIT_PROMPT_COUNT",
    "GENERALIZATION_PROMPT_COUNT",
    "INPUT_PROJECTION_SHA256",
    "LESSON_SCHEMA",
    "LESSON_SCOPE",
    "PREFLIGHT_READY",
    "STOP_IDENTIFIABILITY",
    "FactorLesson",
    "FactorTarget",
    "IdentifiabilityStop",
    "LessonManifest",
    "ManifestPreflight",
    "ManifestValidationError",
    "SplitManifest",
    "authenticate_lesson_targets",
    "build_lesson_manifest_record",
    "canonical_json_bytes",
    "canonical_public_train_preflight",
    "construct_split_manifest",
    "load_public_train_prompt_inputs",
    "parse_lesson_manifest",
    "preflight_prompt_factor_manifest",
]
