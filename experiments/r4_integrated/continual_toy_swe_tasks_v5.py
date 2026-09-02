"""Fresh public-TRAIN-only tasks for the integrated continual coding V5 cohort.

V5 is authored after the spent V4 session localized rigid three-parameter
construction and after the separate variable-cardinality micro-gate.  It is
frozen before any V5 runner or integrated learner output exists.  The stream is
not an inventory-loop benchmark: variable interface width is balanced across
renamed transfer, novel composition, execution-feedback revision, delayed
retention, and honest uncertainty tasks.

Learner-visible records contain only a task commitment, repository path, typed
signature, natural-language request, and inert starter source.  Oracles, cases,
capability/stage labels, retry budgets, feedback pressure, retention anchors,
and expected terminal actions remain evaluator-private.  No successful source,
candidate patch, or answer-bearing diagnostic is stored in a public record.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath

from humanbrain.simple_swe.tasks import GeneratedTask, JsonValue, make_task
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

SCHEMA = "continual-toy-swe-tasks-v5"
PARTITION = "train"
TASK_COUNT = 16
EXECUTION_REGRESSION_CASES = 5
EXECUTION_PROPERTY_CASES = 16

_FAMILY = "F5"
_EXECUTE = "execute"
_STOP = "stop"

_TRANSFER = "renamed-transfer"
_RECOMBINATION = "novel-factor-recombination"
_FEEDBACK = "execution-feedback-revision"
_RETENTION = "delayed-retention-recheck"
_INSUFFICIENT = "insufficient-evidence"
_STAGES = (_TRANSFER, _RECOMBINATION, _FEEDBACK, _RETENTION, _INSUFFICIENT)

_DISTANCE_FIVE = "five-input-distance-transfer"
_INTERIOR_ONE = "unary-interior-transfer"
_OVERLAP_TWO = "binary-overlap-transfer"
_AFFINE_FOUR = "four-input-affine-transfer"
_NORMALIZED_INTERIOR = "normalized-interior-recombination"
_SCALED_ROUTE_COST = "scaled-route-cost-recombination"
_COMMON_NONZERO_PRODUCT = "common-nonzero-product-recombination"
_SHARED_EVEN = "shared-even-recombination"
_FIRST_NONZERO = "first-nonzero-feedback"
_MERGED_NORMALIZED = "merged-normalized-feedback"
_AFFINE_DISTANCE = "affine-distance-feedback"
_COUNT_NONZERO = "count-nonzero-feedback"
_DISTANCE_RETENTION = "distance-retention"
_SHARED_EVEN_RETENTION = "shared-even-retention"
_AMBIGUOUS_TRAVERSAL = "ambiguous-traversal-stop"
_AMBIGUOUS_AGGREGATION = "ambiguous-aggregation-stop"

_GROUP_STAGE = {
    _DISTANCE_FIVE: _TRANSFER,
    _INTERIOR_ONE: _TRANSFER,
    _OVERLAP_TWO: _TRANSFER,
    _AFFINE_FOUR: _TRANSFER,
    _NORMALIZED_INTERIOR: _RECOMBINATION,
    _SCALED_ROUTE_COST: _RECOMBINATION,
    _COMMON_NONZERO_PRODUCT: _RECOMBINATION,
    _SHARED_EVEN: _RECOMBINATION,
    _FIRST_NONZERO: _FEEDBACK,
    _MERGED_NORMALIZED: _FEEDBACK,
    _AFFINE_DISTANCE: _FEEDBACK,
    _COUNT_NONZERO: _FEEDBACK,
    _DISTANCE_RETENTION: _RETENTION,
    _SHARED_EVEN_RETENTION: _RETENTION,
    _AMBIGUOUS_TRAVERSAL: _INSUFFICIENT,
    _AMBIGUOUS_AGGREGATION: _INSUFFICIENT,
}
_ALL_GROUPS = frozenset(_GROUP_STAGE)

_INT = ProgramType.Int()
_STRING = ProgramType.String()
_LIST_INT = ProgramType.List(_INT)


class ContinualToySweTaskV5Error(ValueError):
    """A V5 task, observation, environment, or stream is unauthenticated."""


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
class LearnerVisibleToySweObservationV5:
    """The complete V5 record allowed into learner input."""

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
            raise ContinualToySweTaskV5Error("public task commitment is not SHA-256")
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
            raise ContinualToySweTaskV5Error("learner-visible path is not canonical")
        if not self.prompt or not self.starter_source:
            raise ContinualToySweTaskV5Error("learner-visible prompt or starter is empty")
        expected = _digest(b"continual-toy-swe-observation-v5", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskV5Error("learner-visible observation changed")


@dataclass(frozen=True, slots=True)
class ToySweEnvironmentPrivateV5:
    """Evaluator-only tests, labels, retry policy, and retention custody."""

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
            raise ContinualToySweTaskV5Error("evaluator task commitments changed")
        if (
            self.capability_group not in _ALL_GROUPS
            or self.stage not in _STAGES
            or _GROUP_STAGE[self.capability_group] != self.stage
            or self.expected_action not in {_EXECUTE, _STOP}
            or type(self.attempt_budget) is not int
            or not 1 <= self.attempt_budget <= 3
            or not self.feedback_pressure
        ):
            raise ContinualToySweTaskV5Error("evaluator policy classification changed")
        expected_counts = (
            (0, 0)
            if self.expected_action == _STOP
            else (EXECUTION_REGRESSION_CASES, EXECUTION_PROPERTY_CASES)
        )
        actual_counts = (len(private.regression_cases), len(private.property_cases))
        if actual_counts != expected_counts:
            raise ContinualToySweTaskV5Error("evaluator hidden-case counts changed")
        if self.expected_action == _STOP:
            if self.stage != _INSUFFICIENT or self.attempt_budget != 1:
                raise ContinualToySweTaskV5Error(
                    "STOP policy must be immediate and evidence-limited"
                )
        elif self.stage == _INSUFFICIENT:
            raise ContinualToySweTaskV5Error(
                "insufficient-evidence task became executable"
            )
        if self.stage in {_RECOMBINATION, _FEEDBACK} and self.attempt_budget != 3:
            raise ContinualToySweTaskV5Error("revision pressure lost its retry budget")
        if self.stage == _RETENTION:
            anchor = self.retention_anchor_ordinal
            if type(anchor) is not int or not 0 <= anchor < public.ordinal:
                raise ContinualToySweTaskV5Error("retention anchor is not earlier")
        elif self.retention_anchor_ordinal is not None:
            raise ContinualToySweTaskV5Error("non-retention task gained an anchor")
        expected = _digest(b"continual-toy-swe-environment-v5", self.record())
        if self.environment_sha256 != expected:
            raise ContinualToySweTaskV5Error("evaluator environment changed")


@dataclass(frozen=True, slots=True)
class ContinualToySweTrainTaskV5:
    public: LearnerVisibleToySweObservationV5
    private: ToySweEnvironmentPrivateV5
    task_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_observation_sha256": self.public.observation_sha256,
            "private_environment_sha256": self.private.environment_sha256,
        }

    def authenticate(self) -> None:
        if type(self.public) is not LearnerVisibleToySweObservationV5:
            raise TypeError("task public record must be one exact V5 observation")
        if type(self.private) is not ToySweEnvironmentPrivateV5:
            raise TypeError("task private record must be one exact V5 environment")
        self.public.authenticate()
        self.private.authenticate()
        generated = self.private.generated_task.public
        if not 0 <= generated.ordinal < len(_SPECS):
            raise ContinualToySweTaskV5Error("generated ordinal is outside V5 stream")
        spec = _SPECS[generated.ordinal]
        if (
            self.public.task_commitment_sha256 != generated.task_sha256
            or self.public.signature != generated.signature
            or self.public.prompt != generated.prompt
            or self.public.starter_source
            != _starter_source(spec, generated.signature)
        ):
            raise ContinualToySweTaskV5Error(
                "public observation is not bound to evaluator"
            )
        expected = _digest(b"continual-toy-swe-task-v5", self.record())
        if self.task_sha256 != expected:
            raise ContinualToySweTaskV5Error("continual V5 task changed")


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    description: str
    capability_group: str
    stage: str
    seed: int
    entry_point: str
    parameters: tuple[tuple[str, ProgramType], ...]
    result_type: ProgramType
    before_helpers: tuple[str, ...]
    after_helpers: tuple[str, ...]
    output_local: str
    expected_action: str
    attempt_budget: int
    feedback_pressure: str
    retention_anchor_ordinal: int | None = None


_SPECS = (
    _TaskSpec(
        "Return the non-negative distance between beacon_start and beacon_finish; "
        "phase, checksum, and channel are unused.",
        _DISTANCE_FIVE,
        _TRANSFER,
        601,
        "measure_beacon_separation",
        (
            ("beacon_start", _INT),
            ("beacon_finish", _INT),
            ("phase", _INT),
            ("checksum", _INT),
            ("channel", _INT),
        ),
        _INT,
        ("relay_beacon",),
        ("finish_beacon",),
        "separation_result",
        _EXECUTE,
        2,
        "renamed-prior-transfer",
    ),
    _TaskSpec(
        "Return readings without their first and last elements; inputs shorter "
        "than three elements produce an empty list.",
        _INTERIOR_ONE,
        _TRANSFER,
        607,
        "interior_readings",
        (("readings", _LIST_INT),),
        _LIST_INT,
        (),
        ("relay_readings",),
        "interior_result",
        _EXECUTE,
        2,
        "renamed-prior-transfer",
    ),
    _TaskSpec(
        "Return the sorted distinct codes that occur in both primary_codes and "
        "secondary_codes.",
        _OVERLAP_TWO,
        _TRANSFER,
        613,
        "shared_distinct_codes",
        (("primary_codes", _LIST_INT), ("secondary_codes", _LIST_INT)),
        _LIST_INT,
        ("hold_codes", "relay_codes"),
        (),
        "shared_codes",
        _EXECUTE,
        2,
        "renamed-prior-transfer",
    ),
    _TaskSpec(
        "Multiply reading by gain, then add bias; spare_marker is unused.",
        _AFFINE_FOUR,
        _TRANSFER,
        617,
        "calibrate_reading",
        (
            ("reading", _INT),
            ("gain", _INT),
            ("bias", _INT),
            ("spare_marker", _INT),
        ),
        _INT,
        (),
        ("finish_calibration",),
        "calibrated_value",
        _EXECUTE,
        2,
        "renamed-prior-transfer",
    ),
    _TaskSpec(
        "Remove surrounding whitespace from caption, lowercase it, then remove "
        "the first and last character; short normalized captions return an "
        "empty string.",
        _NORMALIZED_INTERIOR,
        _RECOMBINATION,
        619,
        "normalize_caption_interior",
        (("caption", _STRING),),
        _STRING,
        ("hold_caption",),
        (),
        "caption_core",
        _EXECUTE,
        3,
        "execution-guided-composition",
    ),
    _TaskSpec(
        "Compute the non-negative distance between origin and destination, "
        "multiply by scale, then add surcharge; route_tag is unused.",
        _SCALED_ROUTE_COST,
        _RECOMBINATION,
        631,
        "scaled_route_cost",
        (
            ("origin", _INT),
            ("destination", _INT),
            ("scale", _INT),
            ("surcharge", _INT),
            ("route_tag", _INT),
        ),
        _INT,
        ("relay_route",),
        ("finish_route", "echo_route"),
        "route_cost",
        _EXECUTE,
        3,
        "execution-guided-composition",
    ),
    _TaskSpec(
        "Multiply the distinct nonzero integers that occur in both left_samples "
        "and right_samples, returning 1 when none qualify.",
        _COMMON_NONZERO_PRODUCT,
        _RECOMBINATION,
        641,
        "multiply_shared_nonzero",
        (("left_samples", _LIST_INT), ("right_samples", _LIST_INT)),
        _INT,
        (),
        ("carry_product",),
        "shared_product",
        _EXECUTE,
        3,
        "execution-guided-composition",
    ),
    _TaskSpec(
        "Return the sorted distinct even integers that occur in both left_values "
        "and right_values; left_tag and right_tag are unused.",
        _SHARED_EVEN,
        _RECOMBINATION,
        643,
        "shared_even_values",
        (
            ("left_values", _LIST_INT),
            ("right_values", _LIST_INT),
            ("left_tag", _INT),
            ("right_tag", _INT),
        ),
        _LIST_INT,
        ("hold_even",),
        ("relay_even",),
        "even_overlap",
        _EXECUTE,
        3,
        "execution-guided-composition",
    ),
    _TaskSpec(
        "Return the first nonzero integer in measurements, or fallback when all "
        "measurements are zero or the list is empty.",
        _FIRST_NONZERO,
        _FEEDBACK,
        647,
        "first_nonzero_or_fallback",
        (("measurements", _LIST_INT), ("fallback", _INT)),
        _INT,
        (),
        ("relay_fallback",),
        "first_signal",
        _EXECUTE,
        3,
        "execution-feedback",
    ),
    _TaskSpec(
        "Concatenate prefix and title, then strip surrounding whitespace and "
        "lowercase the result; suffix_tag and context_tag are unused.",
        _MERGED_NORMALIZED,
        _FEEDBACK,
        653,
        "normalize_combined_title",
        (
            ("prefix", _STRING),
            ("title", _STRING),
            ("suffix_tag", _STRING),
            ("context_tag", _STRING),
        ),
        _STRING,
        ("hold_title",),
        ("finish_title",),
        "normalized_title",
        _EXECUTE,
        3,
        "execution-feedback",
    ),
    _TaskSpec(
        "Multiply value by gain, add bias, then return the non-negative distance "
        "from reference; spare_marker is unused.",
        _AFFINE_DISTANCE,
        _FEEDBACK,
        659,
        "affine_reference_distance",
        (
            ("value", _INT),
            ("gain", _INT),
            ("bias", _INT),
            ("reference", _INT),
            ("spare_marker", _INT),
        ),
        _INT,
        ("relay_signal", "hold_signal"),
        (),
        "bounded_signal",
        _EXECUTE,
        3,
        "execution-feedback",
    ),
    _TaskSpec(
        "Return the number of nonzero integers in samples.",
        _COUNT_NONZERO,
        _FEEDBACK,
        661,
        "count_active_samples",
        (("samples", _LIST_INT),),
        _INT,
        (),
        ("finish_count",),
        "active_count",
        _EXECUTE,
        3,
        "execution-feedback",
    ),
    _TaskSpec(
        "Return the non-negative separation between anchor and marker.",
        _DISTANCE_RETENTION,
        _RETENTION,
        673,
        "retained_marker_separation",
        (("anchor", _INT), ("marker", _INT)),
        _INT,
        ("hold_marker",),
        (),
        "marker_gap",
        _EXECUTE,
        2,
        "delayed-retention",
        0,
    ),
    _TaskSpec(
        "Return the sorted distinct even integers shared by primary and secondary; "
        "version and checksum are unused.",
        _SHARED_EVEN_RETENTION,
        _RETENTION,
        677,
        "retained_shared_even",
        (
            ("primary", _LIST_INT),
            ("secondary", _LIST_INT),
            ("version", _INT),
            ("checksum", _INT),
        ),
        _LIST_INT,
        (),
        ("finish_retention",),
        "retained_overlap",
        _EXECUTE,
        2,
        "delayed-retention",
        7,
    ),
    _TaskSpec(
        "Reorder checkpoints according to the active deployment traversal policy. "
        "The policy is not included in the request or inputs; do not guess.",
        _AMBIGUOUS_TRAVERSAL,
        _INSUFFICIENT,
        683,
        "apply_deployment_traversal",
        (("checkpoints", _LIST_INT),),
        _LIST_INT,
        ("hold_traversal",),
        (),
        "ordered_checkpoints",
        _STOP,
        1,
        "missing-policy",
    ),
    _TaskSpec(
        "Combine north, south, east, west, and center according to an externally "
        "selected aggregation rule. The rule is absent from both inputs and "
        "specification; do not guess.",
        _AMBIGUOUS_AGGREGATION,
        _INSUFFICIENT,
        691,
        "apply_external_aggregation",
        (
            ("north", _INT),
            ("south", _INT),
            ("east", _INT),
            ("west", _INT),
            ("center", _INT),
        ),
        _INT,
        (),
        ("relay_aggregation",),
        "aggregate_value",
        _STOP,
        1,
        "missing-policy",
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


def _helper_source(name: str) -> str:
    parameter = f"{name}_value"
    return f"def {name}({parameter}):\n    return {parameter}\n"


def _starter_source(spec: _TaskSpec, signature: TypedSignature) -> str:
    sections = [_helper_source(name) for name in spec.before_helpers]
    parameters = ", ".join(parameter.name for parameter in signature.parameters)
    sections.append(
        f"def {signature.entry_point}({parameters}):\n"
        f"    {spec.output_local} = None\n"
        f"    del {spec.output_local}\n"
        '    raise NotImplementedError("public TRAIN v5 starter")\n'
    )
    sections.extend(_helper_source(name) for name in spec.after_helpers)
    return "\n".join(sections)


def _integer(seed: int, index: int, channel: int = 0) -> int:
    return ((seed * 23 + index * 13 + channel * 17) % 43) - 21


def _integer_list(seed: int, index: int, channel: int = 0) -> list[int]:
    length = (seed + index * 5 + channel * 3) % 10
    return [
        _integer(seed + channel * 19, index * 11 + item, item + channel)
        for item in range(length)
    ]


def _mixed_text(seed: int, index: int) -> str:
    words = (" Aurora", "BASALT ", " Cinder ", "DELTA", " ember ", "FJORD ")
    count = 1 + (seed + index) % 3
    start = (seed * 5 + index * 2) % len(words)
    return "".join(words[(start + offset) % len(words)] for offset in range(count))


def _distance_five(
    start: int,
    finish: int,
    phase: int,
    checksum: int,
    channel: int,
) -> int:
    del phase, checksum, channel
    return abs(start - finish)


def _interior_one(values: list[int]) -> list[int]:
    return values[1:-1]


def _overlap_two(left: list[int], right: list[int]) -> list[int]:
    return sorted(set(left) & set(right))


def _affine_four(value: int, gain: int, bias: int, spare: int) -> int:
    del spare
    return value * gain + bias


def _normalized_interior(text: str) -> str:
    return text.strip().lower()[1:-1]


def _scaled_route_cost(
    origin: int,
    destination: int,
    scale: int,
    surcharge: int,
    route_tag: int,
) -> int:
    del route_tag
    return abs(origin - destination) * scale + surcharge


def _common_nonzero_product(left: list[int], right: list[int]) -> int:
    result = 1
    for value in set(left) & set(right):
        if value != 0:
            result *= value
    return result


def _shared_even(
    left: list[int],
    right: list[int],
    left_tag: int,
    right_tag: int,
) -> list[int]:
    del left_tag, right_tag
    return sorted(value for value in set(left) & set(right) if value % 2 == 0)


def _first_nonzero(values: list[int], fallback: int) -> int:
    return next((value for value in values if value != 0), fallback)


def _merged_normalized(
    prefix: str,
    title: str,
    suffix_tag: str,
    context: str,
) -> str:
    del suffix_tag, context
    return (prefix + title).strip().lower()


def _affine_distance(
    value: int,
    gain: int,
    bias: int,
    reference: int,
    spare: int,
) -> int:
    del spare
    return abs(value * gain + bias - reference)


def _count_nonzero(values: list[int]) -> int:
    return sum(value != 0 for value in values)


def _distance_two(left: int, right: int) -> int:
    return abs(left - right)


def _unreachable_stop_oracle(*args: JsonValue) -> JsonValue:
    del args
    raise RuntimeError("STOP task has no executable target oracle")


def _generated_properties(
    group: str,
    seed: int,
) -> tuple[tuple[tuple[JsonValue, ...], str], ...]:
    rows: list[tuple[tuple[JsonValue, ...], str]] = []
    for index in range(EXECUTION_PROPERTY_CASES):
        if group == _DISTANCE_FIVE:
            args: tuple[JsonValue, ...] = (
                _integer(seed, index),
                _integer(seed, index, 1),
                _integer(seed, index, 2),
                _integer(seed, index, 3),
                _integer(seed, index, 4),
            )
        elif group == _INTERIOR_ONE:
            args = (_integer_list(seed, index),)
        elif group == _OVERLAP_TWO:
            args = (
                _integer_list(seed, index),
                _integer_list(seed, index, 1),
            )
        elif group == _AFFINE_FOUR:
            args = (
                _integer(seed, index),
                _integer(seed, index, 1),
                _integer(seed, index, 2),
                _integer(seed, index, 3),
            )
        elif group == _NORMALIZED_INTERIOR:
            args = (_mixed_text(seed, index),)
        elif group == _SCALED_ROUTE_COST:
            args = (
                _integer(seed, index),
                _integer(seed, index, 1),
                _integer(seed, index, 2),
                _integer(seed, index, 3),
                _integer(seed, index, 4),
            )
        elif group == _COMMON_NONZERO_PRODUCT:
            args = (
                _integer_list(seed, index),
                _integer_list(seed, index, 1),
            )
        elif group in {_SHARED_EVEN, _SHARED_EVEN_RETENTION}:
            args = (
                _integer_list(seed, index),
                _integer_list(seed, index, 1),
                _integer(seed, index, 2),
                _integer(seed, index, 3),
            )
        elif group == _FIRST_NONZERO:
            args = (_integer_list(seed, index), _integer(seed, index, 2))
        elif group == _MERGED_NORMALIZED:
            args = (
                _mixed_text(seed, index),
                _mixed_text(seed + 7, index),
                f"suffix_{index}",
                f"context_{index}",
            )
        elif group == _AFFINE_DISTANCE:
            args = (
                _integer(seed, index),
                _integer(seed, index, 1),
                _integer(seed, index, 2),
                _integer(seed, index, 3),
                _integer(seed, index, 4),
            )
        elif group == _COUNT_NONZERO:
            args = (_integer_list(seed, index),)
        elif group == _DISTANCE_RETENTION:
            args = (_integer(seed, index), _integer(seed, index, 1))
        else:
            raise ValueError("unknown executable V5 capability group")
        rows.append((args, "v5-deterministic-property"))
    return tuple(rows)


def _cases(
    group: str,
    seed: int,
) -> tuple[
    Callable[..., JsonValue],
    tuple[tuple[JsonValue, ...], ...],
    tuple[tuple[tuple[JsonValue, ...], str], ...],
]:
    if group == _DISTANCE_FIVE:
        oracle = _distance_five
        regressions = (
            (0, 0, 1, 2, 3),
            (-8, 5, 0, 0, 0),
            (9, -4, 7, -2, 11),
            (-3, -12, 4, 5, 6),
            (17, 2, -1, -1, -1),
        )
    elif group == _INTERIOR_ONE:
        oracle = _interior_one
        regressions = ((), ((),), ((4,),), ((4, 5),), ((1, 2, 3, 4),))
        regressions = tuple(
            (list(arguments[0]),) if arguments else ([],)
            for arguments in regressions
        )
    elif group == _OVERLAP_TWO:
        oracle = _overlap_two
        regressions = (
            ([], []),
            ([1, 2], [3, 4]),
            ([3, 1, 3], [3, 3]),
            ([-2, 0, 4], [4, -2, 7]),
            ([8, 6, 4, 2], [2, 4, 6, 8]),
        )
    elif group == _AFFINE_FOUR:
        oracle = _affine_four
        regressions = (
            (0, 0, 0, 9),
            (2, 3, 4, 8),
            (-2, 5, 1, 7),
            (7, 0, -3, 6),
            (-4, -3, 2, 5),
        )
    elif group == _NORMALIZED_INTERIOR:
        oracle = _normalized_interior
        regressions = (("",), (" A ",), (" AB ",), (" MiXeD ",), ("  orbit  ",))
    elif group == _SCALED_ROUTE_COST:
        oracle = _scaled_route_cost
        regressions = (
            (0, 0, 3, 2, 10),
            (-3, 5, 2, 1, 30),
            (9, -1, 4, 3, 12),
            (4, 11, 0, -2, 7),
            (-6, -2, -3, 5, 9),
        )
    elif group == _COMMON_NONZERO_PRODUCT:
        oracle = _common_nonzero_product
        regressions = (
            ([], []),
            ([0, 1, 2], [0, 3, 4]),
            ([2, 2, 3], [2, 2, 5]),
            ([-2, 3, -4], [-4, -2, 9]),
            ([1, 3, 5], [5, 3, 1]),
        )
    elif group in {_SHARED_EVEN, _SHARED_EVEN_RETENTION}:
        oracle = _shared_even
        regressions = (
            ([], [], 0, 0),
            ([1, 3, 5], [1, 3, 5], 1, 2),
            ([2, 2, 4], [2, 4, 6], 3, 4),
            ([-4, -3, 0, 7], [-4, 0, 8], 5, 6),
            ([10, 8, 6, 4], [4, 6, 8, 10], 7, 8),
        )
    elif group == _FIRST_NONZERO:
        oracle = _first_nonzero
        regressions = (
            ([], 41),
            ([0, 0, 0], -7),
            ([3, 0, 4], 9),
            ([0, -8, 4], 0),
            ([0, 0, 5, 6], -1),
        )
    elif group == _MERGED_NORMALIZED:
        oracle = _merged_normalized
        regressions = (
            ("", "", "s", "x"),
            ("  ", " A ", "t", "y"),
            ("North ", " STAR", "u", "z"),
            (" MiXeD ", "CaSe ", "v", "q"),
            (" Enough", " Text ", "w", "r"),
        )
    elif group == _AFFINE_DISTANCE:
        oracle = _affine_distance
        regressions = (
            (0, 0, 0, 0, 2),
            (2, 3, 4, 20, 9),
            (5, 4, 3, 10, -1),
            (-2, 5, 1, 0, 12),
            (7, 0, -3, -2, 5),
        )
    elif group == _COUNT_NONZERO:
        oracle = _count_nonzero
        regressions = (([],), ([0],), ([1],), ([0, -2, 0, 3],), ([1, 2, 3, 4],))
    elif group == _DISTANCE_RETENTION:
        oracle = _distance_two
        regressions = ((0, 0), (-5, 7), (9, -4), (-8, -3), (17, 2))
    elif group in {_AMBIGUOUS_TRAVERSAL, _AMBIGUOUS_AGGREGATION}:
        return _unreachable_stop_oracle, (), ()
    else:
        raise ValueError("unknown V5 capability group")
    return oracle, tuple(regressions), _generated_properties(group, seed)


def _build_task(spec: _TaskSpec, ordinal: int) -> ContinualToySweTrainTaskV5:
    signature = _signature(spec)
    oracle, regressions, properties = _cases(spec.capability_group, spec.seed)
    generated = make_task(
        partition=PARTITION,
        family=_FAMILY,
        ordinal=ordinal,
        signature=signature,
        description=spec.description,
        oracle=oracle,
        regression_inputs=regressions,
        property_inputs=properties,
    )
    public_placeholder = LearnerVisibleToySweObservationV5(
        task_commitment_sha256=generated.public.task_sha256,
        relative_path=f"workbench/v5_continual_{ordinal:02d}.py",
        signature=signature,
        prompt=generated.public.prompt,
        starter_source=_starter_source(spec, signature),
        observation_sha256="",
    )
    public = LearnerVisibleToySweObservationV5(
        task_commitment_sha256=public_placeholder.task_commitment_sha256,
        relative_path=public_placeholder.relative_path,
        signature=public_placeholder.signature,
        prompt=public_placeholder.prompt,
        starter_source=public_placeholder.starter_source,
        observation_sha256=_digest(
            b"continual-toy-swe-observation-v5",
            public_placeholder.record(),
        ),
    )
    private_placeholder = ToySweEnvironmentPrivateV5(
        generated_task=generated,
        capability_group=spec.capability_group,
        stage=spec.stage,
        expected_action=spec.expected_action,
        attempt_budget=spec.attempt_budget,
        feedback_pressure=spec.feedback_pressure,
        retention_anchor_ordinal=spec.retention_anchor_ordinal,
        environment_sha256="",
    )
    private = ToySweEnvironmentPrivateV5(
        generated_task=generated,
        capability_group=private_placeholder.capability_group,
        stage=private_placeholder.stage,
        expected_action=private_placeholder.expected_action,
        attempt_budget=private_placeholder.attempt_budget,
        feedback_pressure=private_placeholder.feedback_pressure,
        retention_anchor_ordinal=private_placeholder.retention_anchor_ordinal,
        environment_sha256=_digest(
            b"continual-toy-swe-environment-v5",
            private_placeholder.record(),
        ),
    )
    task_placeholder = ContinualToySweTrainTaskV5(
        public=public,
        private=private,
        task_sha256="",
    )
    task = ContinualToySweTrainTaskV5(
        public=public,
        private=private,
        task_sha256=_digest(
            b"continual-toy-swe-task-v5",
            task_placeholder.record(),
        ),
    )
    task.authenticate()
    return task


_TRAIN_STREAM = tuple(_build_task(spec, ordinal) for ordinal, spec in enumerate(_SPECS))
_EXPECTED_STAGE_COUNTS = Counter(
    {
        _TRANSFER: 4,
        _RECOMBINATION: 4,
        _FEEDBACK: 4,
        _RETENTION: 2,
        _INSUFFICIENT: 2,
    }
)
_EXPECTED_ACTION_COUNTS = Counter({_EXECUTE: 14, _STOP: 2})
_EXPECTED_BUDGET_COUNTS = Counter({1: 2, 2: 6, 3: 8})
_EXPECTED_ARITY_COUNTS = Counter({1: 4, 2: 4, 4: 4, 5: 4})
_EXPECTED_RETENTION_ANCHORS = {12: 0, 13: 7}


def _authenticate_stream(tasks: tuple[ContinualToySweTrainTaskV5, ...]) -> None:
    if len(tasks) != TASK_COUNT:
        raise ContinualToySweTaskV5Error("V5 TRAIN stream count changed")
    for task in tasks:
        task.authenticate()
    if len({task.task_sha256 for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV5Error("V5 task commitments are not unique")
    if len({task.public.signature.entry_point for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV5Error("V5 entry points are not unique")
    if Counter(task.private.stage for task in tasks) != _EXPECTED_STAGE_COUNTS:
        raise ContinualToySweTaskV5Error("V5 stage counts changed")
    if Counter(task.private.capability_group for task in tasks) != Counter(
        {group: 1 for group in _ALL_GROUPS}
    ):
        raise ContinualToySweTaskV5Error("V5 capability group counts changed")
    if Counter(task.private.expected_action for task in tasks) != _EXPECTED_ACTION_COUNTS:
        raise ContinualToySweTaskV5Error("V5 terminal-action counts changed")
    if Counter(task.private.attempt_budget for task in tasks) != _EXPECTED_BUDGET_COUNTS:
        raise ContinualToySweTaskV5Error("V5 attempt-budget counts changed")
    if Counter(len(task.public.signature.parameters) for task in tasks) != (
        _EXPECTED_ARITY_COUNTS
    ):
        raise ContinualToySweTaskV5Error("V5 interface-cardinality balance changed")
    anchors = {
        index: task.private.retention_anchor_ordinal
        for index, task in enumerate(tasks)
        if task.private.retention_anchor_ordinal is not None
    }
    if anchors != _EXPECTED_RETENTION_ANCHORS:
        raise ContinualToySweTaskV5Error("V5 retention anchors changed")
    expected_stage_order = (
        *(_TRANSFER for _ in range(4)),
        *(_RECOMBINATION for _ in range(4)),
        *(_FEEDBACK for _ in range(4)),
        *(_RETENTION for _ in range(2)),
        *(_INSUFFICIENT for _ in range(2)),
    )
    if tuple(task.private.stage for task in tasks) != expected_stage_order:
        raise ContinualToySweTaskV5Error("V5 developmental stage order changed")


def continual_toy_swe_train_stream_v5() -> tuple[ContinualToySweTrainTaskV5, ...]:
    """Return the authenticated immutable ordered V5 public-TRAIN stream."""

    _authenticate_stream(_TRAIN_STREAM)
    return _TRAIN_STREAM


def learner_visible_observation_v5(
    task: ContinualToySweTrainTaskV5,
) -> LearnerVisibleToySweObservationV5:
    """Project exactly the task-family-free learner record."""

    if type(task) is not ContinualToySweTrainTaskV5:
        raise TypeError("learner projection requires one exact V5 task")
    task.authenticate()
    return task.public


def train_stream_sha256_v5(
    tasks: tuple[ContinualToySweTrainTaskV5, ...] | None = None,
) -> str:
    selected = _TRAIN_STREAM if tasks is None else tasks
    if not isinstance(selected, tuple) or not all(
        type(task) is ContinualToySweTrainTaskV5 for task in selected
    ):
        raise TypeError("V5 TRAIN stream must contain exact continual tasks")
    _authenticate_stream(selected)
    return _digest(
        b"continual-toy-swe-stream-v5",
        {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_sha256s": tuple(task.task_sha256 for task in selected),
        },
    )


TRAIN_STREAM_SHA256 = train_stream_sha256_v5()
_STREAM_CUSTODY = {
    "schema": SCHEMA,
    "partition": PARTITION,
    "task_count": TASK_COUNT,
    "task_sha256s": tuple(task.task_sha256 for task in _TRAIN_STREAM),
    "train_stream_sha256": TRAIN_STREAM_SHA256,
    "stage_counts": dict(sorted(_EXPECTED_STAGE_COUNTS.items())),
    "expected_action_counts": dict(sorted(_EXPECTED_ACTION_COUNTS.items())),
    "attempt_budget_counts": dict(sorted(_EXPECTED_BUDGET_COUNTS.items())),
    "interface_arity_counts": dict(sorted(_EXPECTED_ARITY_COUNTS.items())),
    "retention_anchors": dict(sorted(_EXPECTED_RETENTION_ANCHORS.items())),
    "learner_visible_fields": (
        "task_commitment_sha256",
        "relative_path",
        "signature",
        "prompt",
        "starter_source",
    ),
    "authored_after_spent_v4_localization": True,
    "authored_before_v5_runner_or_integrated_session": True,
    "v5_model_outputs_used_during_authoring": 0,
    "successful_target_sources_stored": 0,
    "candidate_patches_stored": 0,
    "answer_bearing_diagnostics_stored": 0,
    "protected_dev_or_evaluation_inputs_used": 0,
    "exact_v4_task_commitments_reused": 0,
}
TRAIN_STREAM_CUSTODY_SHA256 = _digest(
    b"continual-toy-swe-stream-custody-v5",
    _STREAM_CUSTODY,
)


def train_stream_custody_v5() -> dict[str, object]:
    """Return a detached authenticated custody record for the frozen V5 stream."""

    if train_stream_sha256_v5() != TRAIN_STREAM_SHA256:
        raise ContinualToySweTaskV5Error("V5 TRAIN stream root changed")
    if (
        _digest(b"continual-toy-swe-stream-custody-v5", _STREAM_CUSTODY)
        != TRAIN_STREAM_CUSTODY_SHA256
    ):
        raise ContinualToySweTaskV5Error("V5 stream custody changed")
    return json.loads(_canonical_json(_STREAM_CUSTODY))


__all__ = [
    "EXECUTION_PROPERTY_CASES",
    "EXECUTION_REGRESSION_CASES",
    "PARTITION",
    "SCHEMA",
    "TASK_COUNT",
    "TRAIN_STREAM_CUSTODY_SHA256",
    "TRAIN_STREAM_SHA256",
    "ContinualToySweTaskV5Error",
    "ContinualToySweTrainTaskV5",
    "LearnerVisibleToySweObservationV5",
    "ToySweEnvironmentPrivateV5",
    "continual_toy_swe_train_stream_v5",
    "learner_visible_observation_v5",
    "train_stream_custody_v5",
    "train_stream_sha256_v5",
]
