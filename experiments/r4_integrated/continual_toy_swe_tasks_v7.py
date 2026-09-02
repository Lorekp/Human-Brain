"""Frozen public-TRAIN-only tasks for integrated continual coding V7.

V7 is a fresh, deterministic pressure stream authored after the spent V6
session.  It is deliberately larger than a mechanism microfixture while staying
small enough for repeated sandbox execution.  The stream presents ordinary
public software tasks, neutral repository sources, variable-cardinality typed
interfaces, later recombinations, order-sensitive revision opportunities,
delayed paraphrases, and evidence-limited requests that require STOP.

The learner projection contains only a sealed task commitment, repository path,
typed signature, natural-language request, and inert starter source.  Oracles,
hidden cases, stage labels, grouping, retry policy, feedback pressure, retention
anchors, and terminal-action expectations are evaluator-private.  No target
span, candidate patch, repaired source, answer-bearing diagnostic class, or
task-specific learner ontology is supplied.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

from humanbrain.simple_swe.tasks import GeneratedTask, JsonValue, make_task
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

SCHEMA = "continual-toy-swe-tasks-v7"
PARTITION = "train"
TASK_COUNT = 16
MAX_ATTEMPTS = 3
EXECUTION_REGRESSION_CASES = 5
EXECUTION_PROPERTY_CASES = 9

_FAMILY = "F7"
_EXECUTE = "execute"
_STOP = "stop"

_ACQUISITION = "public-family-acquisition"
_RECOMBINATION = "cross-family-recombination"
_FEEDBACK = "execution-feedback-revision"
_RETENTION = "delayed-retention-recheck"
_INSUFFICIENT = "insufficient-evidence"
_STAGES = (_ACQUISITION, _RECOMBINATION, _FEEDBACK, _RETENTION, _INSUFFICIENT)

_GROUPS = tuple(f"g7-{ordinal:02d}" for ordinal in range(TASK_COUNT))
_GROUP_STAGE = {
    **{_GROUPS[index]: _ACQUISITION for index in range(0, 4)},
    **{_GROUPS[index]: _RECOMBINATION for index in range(4, 8)},
    **{_GROUPS[index]: _FEEDBACK for index in range(8, 12)},
    **{_GROUPS[index]: _RETENTION for index in range(12, 14)},
    **{_GROUPS[index]: _INSUFFICIENT for index in range(14, 16)},
}
_ALL_GROUPS = frozenset(_GROUP_STAGE)

_INT = ProgramType.Int()
_STRING = ProgramType.String()
_LIST_INT = ProgramType.List(_INT)

_SOURCE_MASK_BEGIN = b"# " + b"v7-sealed-hash-values-begin"
_SOURCE_MASK_END = b"# " + b"v7-sealed-hash-values-end"


class ContinualToySweTaskV7Error(ValueError):
    """A V7 task, observation, environment, or stream is unauthenticated."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class LearnerVisibleToySweObservationV7:
    """The complete V7 record allowed into learner input."""

    task_commitment_sha256: str
    relative_path: str
    signature: TypedSignature
    prompt: str
    starter_source: str
    observation_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_commitment_sha256": self.task_commitment_sha256,
            "relative_path": self.relative_path,
            "signature": self.signature.to_record(),
            "prompt": self.prompt,
            "starter_source": self.starter_source,
        }

    def authenticate(self) -> None:
        if not _is_sha256(self.task_commitment_sha256):
            raise ContinualToySweTaskV7Error("public task commitment is not SHA-256")
        if type(self.signature) is not TypedSignature:
            raise TypeError("learner-visible signature must be one TypedSignature")
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".py"
            or self.relative_path != path.as_posix()
        ):
            raise ContinualToySweTaskV7Error("learner-visible path is not canonical")
        if not self.prompt or not self.starter_source:
            raise ContinualToySweTaskV7Error("learner-visible prompt or starter is empty")
        expected = _digest(b"continual-toy-swe-observation-v7", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskV7Error("learner-visible observation changed")


@dataclass(frozen=True, slots=True)
class ToySweEnvironmentPrivateV7:
    """Evaluator-only cases, grouping, retry policy, and retention custody."""

    generated_task: GeneratedTask
    capability_group: str
    stage: str
    expected_action: str
    attempt_budget: int
    feedback_pressure: str
    retention_anchor_ordinal: int | None
    environment_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_task_sha256": self.generated_task.public.task_sha256,
            "private_test_sha256": self.generated_task.private.test_sha256,
            "capability_group": self.capability_group,
            "stage": self.stage,
            "expected_action": self.expected_action,
            "attempt_budget": self.attempt_budget,
            "feedback_pressure": self.feedback_pressure,
            "retention_anchor_ordinal": self.retention_anchor_ordinal,
        }

    def authenticate(self) -> None:
        if type(self.generated_task) is not GeneratedTask:
            raise TypeError("evaluator task must be one exact GeneratedTask")
        public = self.generated_task.public
        private = self.generated_task.private
        if (
            public.partition != PARTITION
            or public.family != _FAMILY
            or public.hidden_test_sha256 != private.test_sha256
            or private.public_task_sha256 != public.task_sha256
        ):
            raise ContinualToySweTaskV7Error("evaluator task commitments changed")
        if (
            self.capability_group not in _ALL_GROUPS
            or self.stage not in _STAGES
            or _GROUP_STAGE[self.capability_group] != self.stage
            or self.expected_action not in {_EXECUTE, _STOP}
            or type(self.attempt_budget) is not int
            or not 1 <= self.attempt_budget <= MAX_ATTEMPTS
            or not self.feedback_pressure
        ):
            raise ContinualToySweTaskV7Error("evaluator policy classification changed")
        expected_counts = (
            (0, 0)
            if self.expected_action == _STOP
            else (EXECUTION_REGRESSION_CASES, EXECUTION_PROPERTY_CASES)
        )
        actual_counts = (len(private.regression_cases), len(private.property_cases))
        if actual_counts != expected_counts:
            raise ContinualToySweTaskV7Error("evaluator hidden-case counts changed")
        if self.expected_action == _STOP:
            if self.stage != _INSUFFICIENT or self.attempt_budget != 1:
                raise ContinualToySweTaskV7Error(
                    "STOP policy must be immediate and evidence-limited"
                )
        elif self.stage == _INSUFFICIENT:
            raise ContinualToySweTaskV7Error(
                "insufficient-evidence task became executable"
            )
        if self.stage in {_RECOMBINATION, _FEEDBACK} and self.attempt_budget != 3:
            raise ContinualToySweTaskV7Error("revision pressure lost its retry budget")
        if self.stage == _RETENTION:
            anchor = self.retention_anchor_ordinal
            if type(anchor) is not int or not 0 <= anchor < public.ordinal:
                raise ContinualToySweTaskV7Error("retention anchor is not earlier")
        elif self.retention_anchor_ordinal is not None:
            raise ContinualToySweTaskV7Error("non-retention task gained an anchor")
        expected = _digest(b"continual-toy-swe-environment-v7", self.record())
        if self.environment_sha256 != expected:
            raise ContinualToySweTaskV7Error("evaluator environment changed")


@dataclass(frozen=True, slots=True)
class ContinualToySweTrainTaskV7:
    public: LearnerVisibleToySweObservationV7
    private: ToySweEnvironmentPrivateV7
    task_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_observation_sha256": self.public.observation_sha256,
            "private_environment_sha256": self.private.environment_sha256,
        }

    def authenticate(self) -> None:
        if type(self.public) is not LearnerVisibleToySweObservationV7:
            raise TypeError("task public record must be one exact V7 observation")
        if type(self.private) is not ToySweEnvironmentPrivateV7:
            raise TypeError("task private record must be one exact V7 environment")
        self.public.authenticate()
        self.private.authenticate()
        generated = self.private.generated_task.public
        if not 0 <= generated.ordinal < len(_SPECS):
            raise ContinualToySweTaskV7Error("generated ordinal is outside V7 stream")
        spec = _SPECS[generated.ordinal]
        if (
            self.public.task_commitment_sha256 != generated.task_sha256
            or self.public.signature != generated.signature
            or self.public.prompt != generated.prompt
            or self.public.relative_path != f"workbench/v7_continual_{generated.ordinal:02d}.py"
            or self.public.starter_source
            != _starter_source(generated.ordinal, generated.signature)
            or self.private.capability_group != spec.capability_group
            or self.private.stage != spec.stage
        ):
            raise ContinualToySweTaskV7Error(
                "public observation is not bound to evaluator"
            )
        expected = _digest(b"continual-toy-swe-task-v7", self.record())
        if self.task_sha256 != expected:
            raise ContinualToySweTaskV7Error("continual V7 task changed")


_Oracle = Callable[..., JsonValue]
_PropertyArguments = Callable[[int, int], tuple[JsonValue, ...]]


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    description: str
    capability_group: str
    stage: str
    seed: int
    entry_point: str
    parameters: tuple[tuple[str, ProgramType], ...]
    result_type: ProgramType
    oracle: _Oracle
    regression_inputs: tuple[tuple[JsonValue, ...], ...]
    property_arguments: _PropertyArguments
    expected_action: str
    attempt_budget: int
    feedback_pressure: str
    retention_anchor_ordinal: int | None = None


def _stable_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _stable_characters(text: str) -> str:
    seen: set[str] = set()
    result = []
    for character in text:
        if character not in seen:
            seen.add(character)
            result.append(character)
    return "".join(result)


def _rotate_text(text: str, steps: int) -> str:
    if not text:
        return ""
    offset = steps % len(text)
    return text[offset:] + text[:offset]


def _closed_range(values: list[int], lower: int, upper: int) -> list[int]:
    if lower > upper:
        return []
    return [value for value in values if lower <= value <= upper]


def _affine_clamp(
    values: list[int],
    gain: int,
    bias: int,
    lower: int,
    upper: int,
) -> list[int]:
    if lower > upper:
        return []
    return [min(upper, max(lower, value * gain + bias)) for value in values]


def _sum_affine_clamp(
    values: list[int],
    gain: int,
    bias: int,
    lower: int,
    upper: int,
) -> int:
    return sum(_affine_clamp(values, gain, bias, lower, upper))


def _unique_middle(values: list[int]) -> list[int]:
    return _stable_unique(values)[1:-1]


def _unique_text_then_rotate(text: str, steps: int) -> str:
    return _rotate_text(_stable_characters(text), steps)


def _range_then_unique(values: list[int], lower: int, upper: int) -> list[int]:
    return _stable_unique(_closed_range(values, lower, upper))


def _transform_then_unique(
    values: list[int],
    gain: int,
    bias: int,
    lower: int,
    upper: int,
) -> int:
    return sum(_stable_unique(_affine_clamp(values, gain, bias, lower, upper)))


def _trim_then_unique(values: list[int]) -> list[int]:
    return _stable_unique(values[1:-1])


def _rotate_text_then_unique(text: str, steps: int) -> str:
    return _stable_characters(_rotate_text(text, steps))


def _range_then_trim(values: list[int], lower: int, upper: int) -> list[int]:
    return _closed_range(values, lower, upper)[1:-1]


def _unique_then_transform(
    values: list[int],
    gain: int,
    bias: int,
    lower: int,
    upper: int,
) -> int:
    return sum(_affine_clamp(_stable_unique(values), gain, bias, lower, upper))


def _retained_unique(values: list[int]) -> list[int]:
    return _stable_unique(values)


def _retained_text_rotation(text: str, distance: int) -> str:
    return _rotate_text(text, distance)


def _unreachable_stop_oracle(*args: JsonValue) -> JsonValue:
    del args
    raise RuntimeError("STOP task has no executable target oracle")


def _integer(seed: int, index: int, channel: int = 0) -> int:
    return ((seed * 31 + index * 17 + channel * 13) % 15) - 7


def _integer_list(seed: int, index: int) -> list[int]:
    length = (seed + index * 5) % 11
    return [_integer(seed, index * 7 + item, item % 4) for item in range(length)]


def _mixed_text(seed: int, index: int) -> str:
    alphabet = "abacabadxyzz"
    length = (seed + index * 5) % 12
    start = (seed * 7 + index * 3) % len(alphabet)
    return "".join(
        alphabet[(start + offset) % len(alphabet)] for offset in range(length)
    )


def _values_arguments(seed: int, index: int) -> tuple[JsonValue, ...]:
    return (_integer_list(seed, index),)


def _text_rotation_arguments(seed: int, index: int) -> tuple[JsonValue, ...]:
    return (_mixed_text(seed, index), _integer(seed, index, 4))


def _range_arguments(seed: int, index: int) -> tuple[JsonValue, ...]:
    return (
        _integer_list(seed, index),
        _integer(seed, index, 5),
        _integer(seed, index, 6),
    )


def _affine_arguments(seed: int, index: int) -> tuple[JsonValue, ...]:
    return (
        _integer_list(seed, index),
        _integer(seed, index, 1),
        _integer(seed, index, 2),
        _integer(seed, index, 5),
        _integer(seed, index, 6),
    )


_SPECS = (
    _TaskSpec(
        "Return a new list that keeps only the first occurrence of each integer "
        "in readings. Preserve input order and do not mutate readings.",
        _GROUPS[0],
        _ACQUISITION,
        821,
        "keep_first_readings",
        (("readings", _LIST_INT),),
        _LIST_INT,
        _stable_unique,
        (
            ([],),
            ([4],),
            ([1, 1, 2],),
            ([2, 1, 2, 3, 1],),
            ([0, -1, 0, -1, 2],),
        ),
        _values_arguments,
        _EXECUTE,
        2,
        "initial-public-family-acquisition",
    ),
    _TaskSpec(
        "Return a new string formed by rotating text left by steps characters. "
        "For non-empty text normalize positive or negative steps modulo its "
        "length; return the empty string for empty text.",
        _GROUPS[1],
        _ACQUISITION,
        823,
        "shift_text_left",
        (("text", _STRING), ("steps", _INT)),
        _STRING,
        _rotate_text,
        (
            ("", 3),
            ("q", -8),
            ("abcd", 1),
            ("abca", -1),
            ("aabbcc", 8),
        ),
        _text_rotation_arguments,
        _EXECUTE,
        2,
        "initial-public-family-acquisition",
    ),
    _TaskSpec(
        "Keep the integers from values that satisfy lower <= value <= upper, "
        "preserving their original order. Return an empty list when lower is "
        "greater than upper.",
        _GROUPS[2],
        _ACQUISITION,
        827,
        "select_closed_band",
        (("values", _LIST_INT), ("lower", _INT), ("upper", _INT)),
        _LIST_INT,
        _closed_range,
        (
            ([], 0, 1),
            ([1, 2, 3], 2, 2),
            ([-3, -1, 0, 2, 4], -1, 2),
            ([1, 2, 3], 4, 1),
            ([5, 1, 5, 3], 3, 5),
        ),
        _range_arguments,
        _EXECUTE,
        2,
        "initial-public-family-acquisition",
    ),
    _TaskSpec(
        "For every integer value, compute value * gain + bias and clamp that "
        "result to the inclusive interval [lower, upper]. Return the sum of the "
        "clamped results, or 0 when lower is greater than upper.",
        _GROUPS[3],
        _ACQUISITION,
        829,
        "calibrate_and_bound",
        (
            ("values", _LIST_INT),
            ("gain", _INT),
            ("bias", _INT),
            ("lower", _INT),
            ("upper", _INT),
        ),
        _INT,
        _sum_affine_clamp,
        (
            ([], 2, 1, 0, 5),
            ([1, -2, 4], 2, 1, 0, 5),
            ([3, 1, -1], -2, 3, -2, 4),
            ([0, 1, 2], 1, 0, 0, 1),
            ([1, 2], 3, 4, 9, 2),
        ),
        _affine_arguments,
        _EXECUTE,
        2,
        "initial-public-family-acquisition",
    ),
    _TaskSpec(
        "First remove duplicate integers while keeping first appearances. Then "
        "drop the first and last distinct values. Return an empty list when fewer "
        "than three distinct values remain before dropping the ends.",
        _GROUPS[4],
        _RECOMBINATION,
        839,
        "middle_distinct_readings",
        (("readings", _LIST_INT),),
        _LIST_INT,
        _unique_middle,
        (
            ([],),
            ([1, 1],),
            ([1, 2, 1, 3],),
            ([4, 4, 2, 3, 2, 5],),
            ([0, -1, 0, 2, -1, 3],),
        ),
        _values_arguments,
        _EXECUTE,
        3,
        "multi-attempt-execution-feedback",
    ),
    _TaskSpec(
        "Remove repeated characters from text while preserving first appearances, "
        "then rotate that distinct-character string left by steps. Normalize steps "
        "modulo the distinct string length; empty text stays empty.",
        _GROUPS[5],
        _RECOMBINATION,
        853,
        "rotate_distinct_characters",
        (("text", _STRING), ("steps", _INT)),
        _STRING,
        _unique_text_then_rotate,
        (
            ("", 4),
            ("aabc", 1),
            ("abac", -1),
            ("aabbccde", 6),
            ("xyxzy", -7),
        ),
        _text_rotation_arguments,
        _EXECUTE,
        3,
        "multi-attempt-execution-feedback",
    ),
    _TaskSpec(
        "First keep values inside the inclusive interval [lower, upper]. Then "
        "remove duplicates from that filtered sequence while preserving first "
        "appearances. Return an empty list for reversed bounds.",
        _GROUPS[6],
        _RECOMBINATION,
        857,
        "distinct_values_in_band",
        (("values", _LIST_INT), ("lower", _INT), ("upper", _INT)),
        _LIST_INT,
        _range_then_unique,
        (
            ([], 0, 2),
            ([1, 1, 2, 3], 1, 2),
            ([3, -1, 3, 0, -1, 4], -1, 3),
            ([1, 2, 3], 5, 1),
            ([5, 2, 5, 4, 2, 6], 2, 5),
        ),
        _range_arguments,
        _EXECUTE,
        3,
        "multi-attempt-execution-feedback",
    ),
    _TaskSpec(
        "Transform and clamp every value using value * gain + bias and the "
        "inclusive [lower, upper] bounds. Afterward, remove duplicate transformed "
        "outputs while preserving first appearances and return their sum. Reversed "
        "bounds yield 0.",
        _GROUPS[7],
        _RECOMBINATION,
        859,
        "distinct_bounded_calibrations",
        (
            ("values", _LIST_INT),
            ("gain", _INT),
            ("bias", _INT),
            ("lower", _INT),
            ("upper", _INT),
        ),
        _INT,
        _transform_then_unique,
        (
            ([], 2, 1, 0, 5),
            ([0, 1, 2], 1, 0, 0, 1),
            ([3, 1, 3, -2], 2, -1, 0, 4),
            ([-3, -1, 0, 2], -2, 1, -2, 3),
            ([1, 2], 3, 4, 8, 2),
        ),
        _affine_arguments,
        _EXECUTE,
        3,
        "multi-attempt-execution-feedback",
    ),
    _TaskSpec(
        "Drop the first and last elements from the original values first. Then "
        "remove duplicates from the remaining sequence, keeping first appearances. "
        "Inputs shorter than three elements produce an empty list.",
        _GROUPS[8],
        _FEEDBACK,
        863,
        "distinct_after_trimming",
        (("values", _LIST_INT),),
        _LIST_INT,
        _trim_then_unique,
        (
            ([],),
            ([1, 2],),
            ([1, 2, 1, 3],),
            ([4, 4, 2, 3, 2, 5],),
            ([0, -1, 0, 2, -1, 3],),
        ),
        _values_arguments,
        _EXECUTE,
        3,
        "multi-attempt-order-correction",
    ),
    _TaskSpec(
        "Rotate the original text left by steps first, normalizing positive or "
        "negative steps modulo its length. Then remove repeated characters from the "
        "rotated text while keeping first appearances; empty text stays empty.",
        _GROUPS[9],
        _FEEDBACK,
        877,
        "distinct_after_text_shift",
        (("text", _STRING), ("steps", _INT)),
        _STRING,
        _rotate_text_then_unique,
        (
            ("", 4),
            ("aabc", 1),
            ("abac", -1),
            ("aabbccde", 6),
            ("xyxzy", -7),
        ),
        _text_rotation_arguments,
        _EXECUTE,
        3,
        "multi-attempt-order-correction",
    ),
    _TaskSpec(
        "Select all values inside the inclusive interval [lower, upper] first. "
        "Then drop the first and last qualifying values, retaining duplicates and "
        "order in the interior. Reversed bounds or fewer than three matches yield [].",
        _GROUPS[10],
        _FEEDBACK,
        881,
        "interior_values_in_band",
        (("values", _LIST_INT), ("lower", _INT), ("upper", _INT)),
        _LIST_INT,
        _range_then_trim,
        (
            ([], 0, 2),
            ([9, 1, 2, 3, 8], 1, 3),
            ([1, 1, 2, 3, 3], 1, 3),
            ([-5, -2, 0, 2, 7], -2, 2),
            ([1, 2, 3], 5, 1),
        ),
        _range_arguments,
        _EXECUTE,
        3,
        "multi-attempt-order-correction",
    ),
    _TaskSpec(
        "Remove duplicate input integers first, preserving first appearances. "
        "Then transform every remaining value with value * gain + bias, clamp to "
        "[lower, upper], and return the sum. Count repeated outputs created by "
        "clamping separately. Return 0 for reversed bounds.",
        _GROUPS[11],
        _FEEDBACK,
        883,
        "calibrate_distinct_inputs",
        (
            ("values", _LIST_INT),
            ("gain", _INT),
            ("bias", _INT),
            ("lower", _INT),
            ("upper", _INT),
        ),
        _INT,
        _unique_then_transform,
        (
            ([], 2, 1, 0, 5),
            ([0, 1, 2], 1, 0, 0, 1),
            ([3, 1, 3, -2], 2, -1, 0, 4),
            ([-3, -1, -3, 2], -2, 1, -2, 3),
            ([1, 2], 3, 4, 8, 2),
        ),
        _affine_arguments,
        _EXECUTE,
        3,
        "multi-attempt-order-correction",
    ),
    _TaskSpec(
        "Build a compact event sequence by keeping the earliest appearance of "
        "every integer event code and discarding later repeats. Keep encounter order.",
        _GROUPS[12],
        _RETENTION,
        887,
        "compact_event_codes",
        (("event_codes", _LIST_INT),),
        _LIST_INT,
        _retained_unique,
        (
            ([],),
            ([7],),
            ([2, 2, 1],),
            ([3, 1, 3, 2, 1],),
            ([0, -2, 0, -2, 4],),
        ),
        _values_arguments,
        _EXECUTE,
        2,
        "delayed-paraphrase-retention",
        0,
    ),
    _TaskSpec(
        "Move the first distance characters of message to its end and return the "
        "new string. Interpret any positive or negative distance cyclically; an "
        "empty message remains empty.",
        _GROUPS[13],
        _RETENTION,
        907,
        "cycle_message_prefix",
        (("message", _STRING), ("distance", _INT)),
        _STRING,
        _retained_text_rotation,
        (
            ("", -3),
            ("z", 9),
            ("wxyz", 2),
            ("abca", -2),
            ("aabbcc", 10),
        ),
        _text_rotation_arguments,
        _EXECUTE,
        2,
        "delayed-paraphrase-retention",
        1,
    ),
    _TaskSpec(
        "Select values between lower and upper, then order the selection according "
        "to the active deployment tie policy. The policy may preserve, reverse, or "
        "rank the values, but it is absent from the request and inputs. Do not guess.",
        _GROUPS[14],
        _INSUFFICIENT,
        911,
        "apply_deployment_tie_policy",
        (("values", _LIST_INT), ("lower", _INT), ("upper", _INT)),
        _LIST_INT,
        _unreachable_stop_oracle,
        (),
        _range_arguments,
        _STOP,
        1,
        "missing-external-policy",
    ),
    _TaskSpec(
        "Combine north, south, east, west, and center using the externally selected "
        "integer coefficient vector. No coefficients or profile identifier are "
        "provided, so multiple incompatible results fit the request. Do not guess.",
        _GROUPS[15],
        _INSUFFICIENT,
        919,
        "combine_profiled_channels",
        (
            ("north", _INT),
            ("south", _INT),
            ("east", _INT),
            ("west", _INT),
            ("center", _INT),
        ),
        _INT,
        _unreachable_stop_oracle,
        (),
        _affine_arguments,
        _STOP,
        1,
        "missing-external-policy",
    ),
)


def _signature(spec: _TaskSpec) -> TypedSignature:
    return TypedSignature(
        entry_point=spec.entry_point,
        parameters=tuple(
            ProgramParameter(name, parameter_type)
            for name, parameter_type in spec.parameters
        ),
        result=spec.result_type,
    )


def _neutral_helper(name: str) -> str:
    return f"def {name}(value):\n    return value\n"


def _starter_source(ordinal: int, signature: TypedSignature) -> str:
    parameters = ", ".join(parameter.name for parameter in signature.parameters)
    return "\n".join(
        (
            _neutral_helper(f"workspace_{ordinal:02d}_before"),
            (
                f"def {signature.entry_point}({parameters}):\n"
                "    result_value = None\n"
                "    del result_value\n"
                '    raise NotImplementedError("public TRAIN v7 starter")\n'
            ),
            _neutral_helper(f"workspace_{ordinal:02d}_after"),
        )
    )


def _generated_properties(
    spec: _TaskSpec,
) -> tuple[tuple[tuple[JsonValue, ...], str], ...]:
    if spec.expected_action == _STOP:
        return ()
    return tuple(
        (
            spec.property_arguments(spec.seed, index),
            "v7-deterministic-property",
        )
        for index in range(EXECUTION_PROPERTY_CASES)
    )


def _build_task(spec: _TaskSpec, ordinal: int) -> ContinualToySweTrainTaskV7:
    signature = _signature(spec)
    generated = make_task(
        partition=PARTITION,
        family=_FAMILY,
        ordinal=ordinal,
        signature=signature,
        description=spec.description,
        oracle=spec.oracle,
        regression_inputs=spec.regression_inputs,
        property_inputs=_generated_properties(spec),
    )
    public_placeholder = LearnerVisibleToySweObservationV7(
        task_commitment_sha256=generated.public.task_sha256,
        relative_path=f"workbench/v7_continual_{ordinal:02d}.py",
        signature=signature,
        prompt=generated.public.prompt,
        starter_source=_starter_source(ordinal, signature),
        observation_sha256="",
    )
    public = LearnerVisibleToySweObservationV7(
        task_commitment_sha256=public_placeholder.task_commitment_sha256,
        relative_path=public_placeholder.relative_path,
        signature=public_placeholder.signature,
        prompt=public_placeholder.prompt,
        starter_source=public_placeholder.starter_source,
        observation_sha256=_digest(
            b"continual-toy-swe-observation-v7",
            public_placeholder.record(),
        ),
    )
    private_placeholder = ToySweEnvironmentPrivateV7(
        generated_task=generated,
        capability_group=spec.capability_group,
        stage=spec.stage,
        expected_action=spec.expected_action,
        attempt_budget=spec.attempt_budget,
        feedback_pressure=spec.feedback_pressure,
        retention_anchor_ordinal=spec.retention_anchor_ordinal,
        environment_sha256="",
    )
    private = ToySweEnvironmentPrivateV7(
        generated_task=generated,
        capability_group=private_placeholder.capability_group,
        stage=private_placeholder.stage,
        expected_action=private_placeholder.expected_action,
        attempt_budget=private_placeholder.attempt_budget,
        feedback_pressure=private_placeholder.feedback_pressure,
        retention_anchor_ordinal=private_placeholder.retention_anchor_ordinal,
        environment_sha256=_digest(
            b"continual-toy-swe-environment-v7",
            private_placeholder.record(),
        ),
    )
    task_placeholder = ContinualToySweTrainTaskV7(
        public=public,
        private=private,
        task_sha256="",
    )
    task = ContinualToySweTrainTaskV7(
        public=public,
        private=private,
        task_sha256=_digest(
            b"continual-toy-swe-task-v7",
            task_placeholder.record(),
        ),
    )
    task.authenticate()
    return task


_TRAIN_STREAM = tuple(_build_task(spec, ordinal) for ordinal, spec in enumerate(_SPECS))
_EXPECTED_STAGE_COUNTS = Counter(
    {
        _ACQUISITION: 4,
        _RECOMBINATION: 4,
        _FEEDBACK: 4,
        _RETENTION: 2,
        _INSUFFICIENT: 2,
    }
)
_EXPECTED_ACTION_COUNTS = Counter({_EXECUTE: 14, _STOP: 2})
_EXPECTED_BUDGET_COUNTS = Counter({1: 2, 2: 6, 3: 8})
_EXPECTED_ARITY_COUNTS = Counter({1: 4, 2: 4, 3: 4, 5: 4})
_EXPECTED_RETENTION_ANCHORS = {12: 0, 13: 1}


def _authenticate_stream(tasks: tuple[ContinualToySweTrainTaskV7, ...]) -> None:
    if len(tasks) != TASK_COUNT:
        raise ContinualToySweTaskV7Error("V7 TRAIN stream count changed")
    for task in tasks:
        task.authenticate()
    if len({task.task_sha256 for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV7Error("V7 task commitments are not unique")
    if len({task.public.signature.entry_point for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV7Error("V7 entry points are not unique")
    if len({task.public.prompt for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV7Error("V7 prompts are not unique")
    if len({task.public.starter_source for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV7Error("V7 starter sources are not unique")
    if Counter(task.private.stage for task in tasks) != _EXPECTED_STAGE_COUNTS:
        raise ContinualToySweTaskV7Error("V7 stage counts changed")
    if Counter(task.private.capability_group for task in tasks) != Counter(
        {group: 1 for group in _ALL_GROUPS}
    ):
        raise ContinualToySweTaskV7Error("V7 evaluator group counts changed")
    if Counter(task.private.expected_action for task in tasks) != _EXPECTED_ACTION_COUNTS:
        raise ContinualToySweTaskV7Error("V7 terminal-action counts changed")
    if Counter(task.private.attempt_budget for task in tasks) != _EXPECTED_BUDGET_COUNTS:
        raise ContinualToySweTaskV7Error("V7 attempt-budget counts changed")
    if Counter(len(task.public.signature.parameters) for task in tasks) != (
        _EXPECTED_ARITY_COUNTS
    ):
        raise ContinualToySweTaskV7Error("V7 interface-cardinality balance changed")
    anchors = {
        index: task.private.retention_anchor_ordinal
        for index, task in enumerate(tasks)
        if task.private.retention_anchor_ordinal is not None
    }
    if anchors != _EXPECTED_RETENTION_ANCHORS:
        raise ContinualToySweTaskV7Error("V7 retention anchors changed")
    expected_stage_order = (
        *(_ACQUISITION for _ in range(4)),
        *(_RECOMBINATION for _ in range(4)),
        *(_FEEDBACK for _ in range(4)),
        *(_RETENTION for _ in range(2)),
        *(_INSUFFICIENT for _ in range(2)),
    )
    if tuple(task.private.stage for task in tasks) != expected_stage_order:
        raise ContinualToySweTaskV7Error("V7 developmental stage order changed")


def continual_toy_swe_train_stream_v7() -> tuple[ContinualToySweTrainTaskV7, ...]:
    """Return the authenticated immutable ordered V7 public-TRAIN stream."""

    _authenticate_stream(_TRAIN_STREAM)
    return _TRAIN_STREAM


def learner_visible_observation_v7(
    task: ContinualToySweTrainTaskV7,
) -> LearnerVisibleToySweObservationV7:
    """Project exactly the task-family-free learner record."""

    if type(task) is not ContinualToySweTrainTaskV7:
        raise TypeError("learner projection requires one exact V7 task")
    task.authenticate()
    return task.public


def train_stream_sha256_v7(
    tasks: tuple[ContinualToySweTrainTaskV7, ...] | None = None,
) -> str:
    selected = _TRAIN_STREAM if tasks is None else tasks
    if not isinstance(selected, tuple) or not all(
        type(task) is ContinualToySweTrainTaskV7 for task in selected
    ):
        raise TypeError("V7 TRAIN stream must contain exact continual tasks")
    _authenticate_stream(selected)
    return _digest(
        b"continual-toy-swe-stream-v7",
        {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_sha256s": tuple(task.task_sha256 for task in selected),
        },
    )


def _masked_task_source() -> bytes:
    output: list[bytes] = []
    masking = False
    masked_blocks = 0
    for line in Path(__file__).read_bytes().splitlines(keepends=True):
        content = line.rstrip(b"\r\n")
        newline = line[len(content) :]
        if content == _SOURCE_MASK_BEGIN:
            if masking:
                raise ContinualToySweTaskV7Error("nested source-hash mask")
            masking = True
            masked_blocks += 1
            output.append(line)
            output.append(b"# v7-sealed-hash-values-masked" + newline)
        elif content == _SOURCE_MASK_END:
            if not masking:
                raise ContinualToySweTaskV7Error("unpaired source-hash mask")
            masking = False
            output.append(line)
        elif not masking:
            output.append(line)
    if masking or masked_blocks != 2:
        raise ContinualToySweTaskV7Error("source-hash mask count changed")
    return b"".join(output)


def task_source_sha256_v7() -> str:
    """Hash the full source while masking only its three self-seal literals."""

    return sha256(_masked_task_source()).hexdigest()


# v7-sealed-hash-values-begin
TRAIN_STREAM_SHA256 = (
    "63227e99f894654a32873b100682f30baa786d3a5e06018d2083f1b58820a117"
)
TASK_SOURCE_SHA256 = (
    "a73bdc1fb15fceccdc2419ca18a279de0d023a57e4ce4c39497117f637c8f2cc"
)
# v7-sealed-hash-values-end

if train_stream_sha256_v7() != TRAIN_STREAM_SHA256:
    raise ContinualToySweTaskV7Error("frozen V7 TRAIN stream root changed")
if task_source_sha256_v7() != TASK_SOURCE_SHA256:
    raise ContinualToySweTaskV7Error("frozen V7 task source changed")

_STREAM_CUSTODY = {
    "schema": SCHEMA,
    "partition": PARTITION,
    "task_count": TASK_COUNT,
    "task_sha256s": tuple(task.task_sha256 for task in _TRAIN_STREAM),
    "train_stream_sha256": TRAIN_STREAM_SHA256,
    "task_source_sha256": TASK_SOURCE_SHA256,
    "stage_counts": dict(sorted(_EXPECTED_STAGE_COUNTS.items())),
    "expected_action_counts": dict(sorted(_EXPECTED_ACTION_COUNTS.items())),
    "attempt_budget_counts": dict(sorted(_EXPECTED_BUDGET_COUNTS.items())),
    "runner_policy_max_attempts": MAX_ATTEMPTS,
    "interface_arity_counts": dict(sorted(_EXPECTED_ARITY_COUNTS.items())),
    "retention_anchors": dict(sorted(_EXPECTED_RETENTION_ANCHORS.items())),
    "learner_visible_fields": (
        "task_commitment_sha256",
        "relative_path",
        "signature",
        "prompt",
        "starter_source",
    ),
    "executable_hidden_cases_per_task": (
        EXECUTION_REGRESSION_CASES + EXECUTION_PROPERTY_CASES
    ),
    "three_attempt_feedback_tasks": 8,
    "later_attempt_opportunities": 22,
    "public_train_only": True,
    "authored_after_spent_v6": True,
    "learner_or_model_imports_during_authoring": 0,
    "learner_or_model_outputs_used_during_authoring": 0,
    "learner_runs_during_authoring": 0,
    "benchmark_artifacts_created": 0,
    "successful_target_sources_stored": 0,
    "candidate_patches_stored": 0,
    "target_spans_stored": 0,
    "answer_bearing_diagnostic_classes_stored": 0,
    "task_specific_learner_ontologies_stored": 0,
    "protected_dev_or_evaluation_inputs_used": 0,
    "exact_v6_task_commitments_reused": 0,
}

# v7-sealed-hash-values-begin
TRAIN_STREAM_CUSTODY_SHA256 = (
    "25d7db8cb294f5aec1f5953a336b1ada1bd36917fe0e7052081b86bfb352f35e"
)
# v7-sealed-hash-values-end

if (
    _digest(b"continual-toy-swe-stream-custody-v7", _STREAM_CUSTODY)
    != TRAIN_STREAM_CUSTODY_SHA256
):
    raise ContinualToySweTaskV7Error("frozen V7 TRAIN custody root changed")


def train_stream_custody_v7() -> dict[str, object]:
    """Return a detached authenticated custody record for the frozen V7 stream."""

    if train_stream_sha256_v7() != TRAIN_STREAM_SHA256:
        raise ContinualToySweTaskV7Error("V7 TRAIN stream root changed")
    if task_source_sha256_v7() != TASK_SOURCE_SHA256:
        raise ContinualToySweTaskV7Error("V7 task source changed")
    if (
        _digest(b"continual-toy-swe-stream-custody-v7", _STREAM_CUSTODY)
        != TRAIN_STREAM_CUSTODY_SHA256
    ):
        raise ContinualToySweTaskV7Error("V7 stream custody changed")
    return json.loads(_canonical_json(_STREAM_CUSTODY))


__all__ = [
    "EXECUTION_PROPERTY_CASES",
    "EXECUTION_REGRESSION_CASES",
    "MAX_ATTEMPTS",
    "PARTITION",
    "SCHEMA",
    "TASK_COUNT",
    "TASK_SOURCE_SHA256",
    "TRAIN_STREAM_CUSTODY_SHA256",
    "TRAIN_STREAM_SHA256",
    "ContinualToySweTaskV7Error",
    "ContinualToySweTrainTaskV7",
    "LearnerVisibleToySweObservationV7",
    "ToySweEnvironmentPrivateV7",
    "continual_toy_swe_train_stream_v7",
    "learner_visible_observation_v7",
    "task_source_sha256_v7",
    "train_stream_custody_v7",
    "train_stream_sha256_v7",
]
