"""Fresh mechanism-blind public-TRAIN tasks for continual coding V4.

This stream was authored before any V4 factorized-pointer implementation or
runner.  Learner-visible records contain only a prompt, typed signature,
repository path, and inert source with a varied callable/symbol inventory.
Oracles, cases, capability labels, retry budgets, and expected STOP decisions
remain evaluator-private.  No V3 task content or successful target source is
stored here.
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

SCHEMA = "continual-toy-swe-tasks-v4"
PARTITION = "train"
TASK_COUNT = 12
EXECUTION_REGRESSION_CASES = 4
EXECUTION_PROPERTY_CASES = 12

_FAMILY = "F4"
_EXECUTE = "execute"
_STOP = "stop"

_TRANSFER = "renamed-transfer"
_RECOMBINATION = "factor-recombination"
_FEEDBACK = "feedback-revision"
_INSUFFICIENT = "insufficient-evidence"
_STAGES = (_TRANSFER, _RECOMBINATION, _FEEDBACK, _INSUFFICIENT)

_DISTANCE = "distance-transfer"
_EVEN_FILTER = "even-filter-transfer"
_NORMALIZE = "normalize-transfer"
_SCALED_DISTANCE = "scaled-distance-recombination"
_EVEN_PRODUCT = "even-product-recombination"
_NORMALIZED_DISTINCT = "normalized-distinct-recombination"
_CAPPED_AFFINE = "capped-affine-feedback"
_FIRST_EVEN = "first-even-feedback"
_NORMALIZE_FALLBACK = "normalize-fallback-feedback"
_AMBIGUOUS_BOUNDARY = "ambiguous-boundary-stop"
_AMBIGUOUS_LOCALE = "ambiguous-locale-stop"
_AMBIGUOUS_EXTREME = "ambiguous-extreme-stop"

_GROUP_STAGE = {
    _DISTANCE: _TRANSFER,
    _EVEN_FILTER: _TRANSFER,
    _NORMALIZE: _TRANSFER,
    _SCALED_DISTANCE: _RECOMBINATION,
    _EVEN_PRODUCT: _RECOMBINATION,
    _NORMALIZED_DISTINCT: _RECOMBINATION,
    _CAPPED_AFFINE: _FEEDBACK,
    _FIRST_EVEN: _FEEDBACK,
    _NORMALIZE_FALLBACK: _FEEDBACK,
    _AMBIGUOUS_BOUNDARY: _INSUFFICIENT,
    _AMBIGUOUS_LOCALE: _INSUFFICIENT,
    _AMBIGUOUS_EXTREME: _INSUFFICIENT,
}
_ALL_GROUPS = frozenset(_GROUP_STAGE)

_INT = ProgramType.Int()
_STRING = ProgramType.String()
_LIST_INT = ProgramType.List(_INT)


class ContinualToySweTaskError(ValueError):
    """A V4 task, observation, evaluator record, or stream is unauthenticated."""


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
class LearnerVisibleToySweObservationV4:
    """The complete V4 record allowed into learner input."""

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
            raise ContinualToySweTaskError("public task commitment is not SHA-256")
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
            raise ContinualToySweTaskError("learner-visible path is not canonical")
        if not self.prompt or not self.starter_source:
            raise ContinualToySweTaskError("learner-visible prompt or starter is empty")
        expected = _digest(b"continual-toy-swe-observation-v4", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskError("learner-visible observation changed")


@dataclass(frozen=True, slots=True)
class ToySweEnvironmentPrivateV4:
    """Evaluator-only tests, labels, retry budget, and terminal-action custody."""

    generated_task: GeneratedTask
    capability_group: str
    stage: str
    expected_action: str
    attempt_budget: int
    feedback_pressure: str
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
            raise ContinualToySweTaskError("evaluator task commitments changed")
        if (
            self.capability_group not in _ALL_GROUPS
            or self.stage not in _STAGES
            or _GROUP_STAGE[self.capability_group] != self.stage
            or self.expected_action not in {_EXECUTE, _STOP}
            or type(self.attempt_budget) is not int
            or not 1 <= self.attempt_budget <= 3
            or not self.feedback_pressure
        ):
            raise ContinualToySweTaskError("evaluator policy classification changed")
        expected_counts = (
            (0, 0)
            if self.expected_action == _STOP
            else (EXECUTION_REGRESSION_CASES, EXECUTION_PROPERTY_CASES)
        )
        actual_counts = (len(private.regression_cases), len(private.property_cases))
        if actual_counts != expected_counts:
            raise ContinualToySweTaskError("evaluator hidden-case counts changed")
        if self.expected_action == _STOP:
            if self.stage != _INSUFFICIENT or self.attempt_budget != 1:
                raise ContinualToySweTaskError("STOP policy must be immediate and uncertain")
        elif self.stage == _INSUFFICIENT:
            raise ContinualToySweTaskError("insufficient-evidence task became executable")
        if self.stage == _FEEDBACK and self.attempt_budget != 3:
            raise ContinualToySweTaskError("feedback-revision task lost its retry budget")
        expected = _digest(b"continual-toy-swe-environment-v4", self.record())
        if self.environment_sha256 != expected:
            raise ContinualToySweTaskError("evaluator environment changed")


@dataclass(frozen=True, slots=True)
class ContinualToySweTrainTaskV4:
    public: LearnerVisibleToySweObservationV4
    private: ToySweEnvironmentPrivateV4
    task_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_observation_sha256": self.public.observation_sha256,
            "private_environment_sha256": self.private.environment_sha256,
        }

    def authenticate(self) -> None:
        if type(self.public) is not LearnerVisibleToySweObservationV4:
            raise TypeError("task public record must be one exact V4 observation")
        if type(self.private) is not ToySweEnvironmentPrivateV4:
            raise TypeError("task private record must be one exact V4 environment")
        self.public.authenticate()
        self.private.authenticate()
        generated = self.private.generated_task.public
        if not 0 <= generated.ordinal < len(_SPECS):
            raise ContinualToySweTaskError("generated ordinal is outside V4 stream")
        spec = _SPECS[generated.ordinal]
        if (
            self.public.task_commitment_sha256 != generated.task_sha256
            or self.public.signature != generated.signature
            or self.public.prompt != generated.prompt
            or self.public.starter_source
            != _starter_source(spec, generated.signature)
        ):
            raise ContinualToySweTaskError("public observation is not bound to evaluator")
        expected = _digest(b"continual-toy-swe-task-v4", self.record())
        if self.task_sha256 != expected:
            raise ContinualToySweTaskError("continual V4 task changed")


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    description: str
    capability_group: str
    stage: str
    seed: int
    entry_point: str
    parameter_names: tuple[str, ...]
    before_helpers: tuple[str, ...]
    after_helpers: tuple[str, ...]
    output_local: str
    expected_action: str
    attempt_budget: int
    feedback_pressure: str


_SPECS = (
    _TaskSpec(
        "Return the non-negative distance between western_mark and eastern_mark; "
        "spare_token is unused.",
        _DISTANCE,
        _TRANSFER,
        401,
        "measure_stone_gap",
        ("western_mark", "eastern_mark", "spare_token"),
        ("echo_marker",),
        (),
        "gap_result",
        _EXECUTE,
        2,
        "direct-transfer",
    ),
    _TaskSpec(
        "Return a new list containing the even integers from readings in their "
        "original order; spare_flag is unused.",
        _EVEN_FILTER,
        _TRANSFER,
        409,
        "collect_even_readings",
        ("readings", "spare_flag"),
        (),
        ("relay_reading",),
        "selected_readings",
        _EXECUTE,
        2,
        "direct-transfer",
    ),
    _TaskSpec(
        "Return inscription with surrounding whitespace removed and every letter "
        "lowercased; spare_left and spare_right are unused.",
        _NORMALIZE,
        _TRANSFER,
        419,
        "normalize_inscription",
        ("inscription", "spare_left", "spare_right"),
        ("hold_text", "pass_text"),
        (),
        "normalized_inscription",
        _EXECUTE,
        2,
        "direct-transfer",
    ),
    _TaskSpec(
        "Return scale times the non-negative distance between start and finish.",
        _SCALED_DISTANCE,
        _RECOMBINATION,
        431,
        "scale_route_gap",
        ("start", "finish", "scale"),
        (),
        ("relay_scale", "relay_gap"),
        "scaled_gap",
        _EXECUTE,
        3,
        "regression-guided-composition",
    ),
    _TaskSpec(
        "Multiply the even nonzero integers in samples, returning 1 when no such "
        "integer is present.",
        _EVEN_PRODUCT,
        _RECOMBINATION,
        439,
        "multiply_active_even_samples",
        ("samples",),
        ("echo_sample",),
        ("carry_sample",),
        "running_product",
        _EXECUTE,
        3,
        "regression-guided-composition",
    ),
    _TaskSpec(
        "Strip surrounding whitespace from message, lowercase it, remove spaces, "
        "then return its distinct characters in sorted order.",
        _NORMALIZED_DISTINCT,
        _RECOMBINATION,
        443,
        "normalize_distinct_message",
        ("message",),
        (),
        (),
        "distinct_message",
        _EXECUTE,
        3,
        "regression-guided-composition",
    ),
    _TaskSpec(
        "Multiply value by multiplier, add increment, and return at most ceiling.",
        _CAPPED_AFFINE,
        _FEEDBACK,
        457,
        "cap_affine_signal",
        ("value", "multiplier", "increment", "ceiling"),
        ("mirror_signal", "carry_signal"),
        ("finish_signal",),
        "affine_candidate",
        _EXECUTE,
        3,
        "signature",
    ),
    _TaskSpec(
        "Return the first even integer in values, or fallback when values contains "
        "no even integer.",
        _FIRST_EVEN,
        _FEEDBACK,
        461,
        "first_even_or_fallback",
        ("values", "fallback"),
        (),
        ("echo_fallback",),
        "first_match",
        _EXECUTE,
        3,
        "runtime",
    ),
    _TaskSpec(
        "Strip surrounding whitespace from text and lowercase it; when the result "
        "is empty, return fallback instead.",
        _NORMALIZE_FALLBACK,
        _FEEDBACK,
        467,
        "normalize_or_fallback",
        ("text", "fallback"),
        ("relay_text",),
        ("echo_text", "finish_text"),
        "text_candidate",
        _EXECUTE,
        3,
        "regression",
    ),
    _TaskSpec(
        "Return the designated boundary value from items. The external policy that "
        "chooses the first versus last boundary is not included; do not guess.",
        _AMBIGUOUS_BOUNDARY,
        _INSUFFICIENT,
        479,
        "select_policy_boundary",
        ("items",),
        ("hold_boundary",),
        (),
        "boundary_candidate",
        _STOP,
        1,
        "missing-policy",
    ),
    _TaskSpec(
        "Normalize text according to the project locale. The locale and its case "
        "and accent policy are not provided; do not guess.",
        _AMBIGUOUS_LOCALE,
        _INSUFFICIENT,
        487,
        "apply_project_locale",
        ("text",),
        (),
        ("relay_locale",),
        "locale_candidate",
        _STOP,
        1,
        "missing-policy",
    ),
    _TaskSpec(
        "Return either the smaller or larger of left and right according to an "
        "external mode. The mode is absent from both inputs and specification.",
        _AMBIGUOUS_EXTREME,
        _INSUFFICIENT,
        491,
        "select_external_extreme",
        ("left", "right"),
        ("echo_extreme",),
        ("finish_extreme",),
        "extreme_candidate",
        _STOP,
        1,
        "missing-policy",
    ),
)


def _signature(spec: _TaskSpec) -> TypedSignature:
    group = spec.capability_group
    if group == _DISTANCE:
        parameter_types = (_INT, _INT, _INT)
        result = _INT
    elif group == _EVEN_FILTER:
        parameter_types = (_LIST_INT, _INT)
        result = _LIST_INT
    elif group == _NORMALIZE:
        parameter_types = (_STRING, _STRING, _STRING)
        result = _STRING
    elif group == _SCALED_DISTANCE:
        parameter_types = (_INT, _INT, _INT)
        result = _INT
    elif group == _EVEN_PRODUCT:
        parameter_types = (_LIST_INT,)
        result = _INT
    elif group in {_NORMALIZED_DISTINCT, _AMBIGUOUS_LOCALE}:
        parameter_types = (_STRING,)
        result = _STRING
    elif group == _CAPPED_AFFINE:
        parameter_types = (_INT, _INT, _INT, _INT)
        result = _INT
    elif group == _FIRST_EVEN:
        parameter_types = (_LIST_INT, _INT)
        result = _INT
    elif group == _NORMALIZE_FALLBACK:
        parameter_types = (_STRING, _STRING)
        result = _STRING
    elif group == _AMBIGUOUS_BOUNDARY:
        parameter_types = (_LIST_INT,)
        result = _INT
    elif group == _AMBIGUOUS_EXTREME:
        parameter_types = (_INT, _INT)
        result = _INT
    else:
        raise ValueError("unknown V4 capability group")
    if len(spec.parameter_names) != len(parameter_types):
        raise ValueError("V4 parameter names and types disagree")
    return TypedSignature(
        entry_point=spec.entry_point,
        parameters=tuple(
            ProgramParameter(name, parameter_type)
            for name, parameter_type in zip(
                spec.parameter_names,
                parameter_types,
                strict=True,
            )
        ),
        result=result,
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
        '    raise NotImplementedError("public TRAIN v4 starter")\n'
    )
    sections.extend(_helper_source(name) for name in spec.after_helpers)
    return "\n".join(sections)


def _integer(seed: int, index: int, channel: int = 0) -> int:
    return ((seed * 19 + index * 11 + channel * 7) % 37) - 18


def _integer_list(seed: int, index: int, channel: int = 0) -> list[int]:
    length = (seed + index * 3 + channel) % 8
    return [_integer(seed + channel * 13, index * 7 + item, item) for item in range(length)]


def _mixed_text(seed: int, index: int) -> str:
    words = (" Amber", "BIRCH ", " Cedar ", "DUNE", " elm ", "FROST ")
    count = 1 + (seed + index) % 3
    start = (seed * 3 + index) % len(words)
    return "".join(words[(start + offset) % len(words)] for offset in range(count))


def _distance(left: int, right: int, unused: int) -> int:
    del unused
    return abs(left - right)


def _even_filter(values: list[int], unused: int) -> list[int]:
    del unused
    return [value for value in values if value % 2 == 0]


def _normalize(text: str, unused_left: str, unused_right: str) -> str:
    del unused_left, unused_right
    return text.strip().lower()


def _scaled_distance(left: int, right: int, scale: int) -> int:
    return abs(left - right) * scale


def _even_product(values: list[int]) -> int:
    result = 1
    for value in values:
        if value != 0 and value % 2 == 0:
            result *= value
    return result


def _normalized_distinct(text: str) -> str:
    normalized = text.strip().lower().replace(" ", "")
    return "".join(sorted(set(normalized)))


def _capped_affine(value: int, multiplier: int, increment: int, ceiling: int) -> int:
    return min(ceiling, value * multiplier + increment)


def _first_even(values: list[int], fallback: int) -> int:
    return next((value for value in values if value % 2 == 0), fallback)


def _normalize_fallback(text: str, fallback: str) -> str:
    return text.strip().lower() or fallback


def _unreachable_stop_oracle(*args: JsonValue) -> JsonValue:
    del args
    raise RuntimeError("STOP task has no executable target oracle")


def _cases(
    group: str,
    seed: int,
) -> tuple[
    Callable[..., JsonValue],
    tuple[tuple[JsonValue, ...], ...],
    tuple[tuple[tuple[JsonValue, ...], str], ...],
]:
    if group == _DISTANCE:
        oracle = _distance
        regressions = ((0, 0, 91), (-5, 7, 0), (9, -4, -2), (-8, -3, 6))
        properties = tuple(
            (
                (_integer(seed, index), _integer(seed, index, 1), index),
                "v4-distance-generated",
            )
            for index in range(EXECUTION_PROPERTY_CASES)
        )
    elif group == _EVEN_FILTER:
        oracle = _even_filter
        regressions = (([], 0), ([1, 3, 5], 1), ([2, 4, 6], 2), ([0, -3, -2, 7], 3))
        properties = tuple(
            ((_integer_list(seed, index), index), "v4-even-filter-generated")
            for index in range(EXECUTION_PROPERTY_CASES)
        )
    elif group == _NORMALIZE:
        oracle = _normalize
        regressions = (("", "x", "y"), ("  A  ", "", ""), ("MiXeD", "l", "r"), (" x Y ", "a", "b"))
        properties = tuple(
            ((_mixed_text(seed, index), "left", "right"), "v4-normalize-generated")
            for index in range(EXECUTION_PROPERTY_CASES)
        )
    elif group == _SCALED_DISTANCE:
        oracle = _scaled_distance
        regressions = ((0, 0, 8), (-3, 5, 2), (9, -1, 0), (4, 11, -3))
        properties = tuple(
            (
                (_integer(seed, index), _integer(seed, index, 1), _integer(seed, index, 2)),
                "v4-scaled-distance-generated",
            )
            for index in range(EXECUTION_PROPERTY_CASES)
        )
    elif group == _EVEN_PRODUCT:
        oracle = _even_product
        regressions = (([],), ([1, 3, 5],), ([0, 2, 4],), ([-2, 3, -4, 0],))
        properties = tuple(
            ((_integer_list(seed, index),), "v4-even-product-generated")
            for index in range(EXECUTION_PROPERTY_CASES)
        )
    elif group == _NORMALIZED_DISTINCT:
        oracle = _normalized_distinct
        regressions = (("",), (" A a ",), ("bA b",), ("  Mixed  ",))
        properties = tuple(
            ((_mixed_text(seed, index),), "v4-normalized-distinct-generated")
            for index in range(EXECUTION_PROPERTY_CASES)
        )
    elif group == _CAPPED_AFFINE:
        oracle = _capped_affine
        regressions = ((2, 3, 4, 20), (5, 4, 3, 10), (-2, 5, 1, 0), (7, 0, -3, -1))
        properties = tuple(
            (
                (
                    _integer(seed, index),
                    _integer(seed, index, 1),
                    _integer(seed, index, 2),
                    _integer(seed, index, 3),
                ),
                "v4-capped-affine-generated",
            )
            for index in range(EXECUTION_PROPERTY_CASES)
        )
    elif group == _FIRST_EVEN:
        oracle = _first_even
        regressions = (([], 41), ([1, 3, 5], -7), ([2, 4, 6], 9), ([1, 8, 4], 0))
        properties = tuple(
            (
                (_integer_list(seed, index), _integer(seed, index, 3)),
                "v4-first-even-generated",
            )
            for index in range(EXECUTION_PROPERTY_CASES)
        )
    elif group == _NORMALIZE_FALLBACK:
        oracle = _normalize_fallback
        regressions = (("", "missing"), ("   ", "blank"), (" A ", "unused"), ("MiXeD", "x"))
        properties = tuple(
            (
                (_mixed_text(seed, index), f"fallback_{index}"),
                "v4-normalize-fallback-generated",
            )
            for index in range(EXECUTION_PROPERTY_CASES)
        )
    elif group in {_AMBIGUOUS_BOUNDARY, _AMBIGUOUS_LOCALE, _AMBIGUOUS_EXTREME}:
        return _unreachable_stop_oracle, (), ()
    else:
        raise ValueError("unknown V4 capability group")
    return oracle, regressions, properties


def _build_task(spec: _TaskSpec, ordinal: int) -> ContinualToySweTrainTaskV4:
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
    public_placeholder = LearnerVisibleToySweObservationV4(
        task_commitment_sha256=generated.public.task_sha256,
        relative_path=f"workbench/v4_pressure_{ordinal:02d}.py",
        signature=signature,
        prompt=generated.public.prompt,
        starter_source=_starter_source(spec, signature),
        observation_sha256="",
    )
    public = LearnerVisibleToySweObservationV4(
        task_commitment_sha256=public_placeholder.task_commitment_sha256,
        relative_path=public_placeholder.relative_path,
        signature=public_placeholder.signature,
        prompt=public_placeholder.prompt,
        starter_source=public_placeholder.starter_source,
        observation_sha256=_digest(
            b"continual-toy-swe-observation-v4",
            public_placeholder.record(),
        ),
    )
    private_placeholder = ToySweEnvironmentPrivateV4(
        generated_task=generated,
        capability_group=spec.capability_group,
        stage=spec.stage,
        expected_action=spec.expected_action,
        attempt_budget=spec.attempt_budget,
        feedback_pressure=spec.feedback_pressure,
        environment_sha256="",
    )
    private = ToySweEnvironmentPrivateV4(
        generated_task=generated,
        capability_group=private_placeholder.capability_group,
        stage=private_placeholder.stage,
        expected_action=private_placeholder.expected_action,
        attempt_budget=private_placeholder.attempt_budget,
        feedback_pressure=private_placeholder.feedback_pressure,
        environment_sha256=_digest(
            b"continual-toy-swe-environment-v4",
            private_placeholder.record(),
        ),
    )
    task_placeholder = ContinualToySweTrainTaskV4(
        public=public,
        private=private,
        task_sha256="",
    )
    task = ContinualToySweTrainTaskV4(
        public=public,
        private=private,
        task_sha256=_digest(
            b"continual-toy-swe-task-v4",
            task_placeholder.record(),
        ),
    )
    task.authenticate()
    return task


_TRAIN_STREAM = tuple(_build_task(spec, ordinal) for ordinal, spec in enumerate(_SPECS))
_EXPECTED_STAGE_COUNTS = Counter({stage: 3 for stage in _STAGES})
_EXPECTED_ACTION_COUNTS = Counter({_EXECUTE: 9, _STOP: 3})
_EXPECTED_BUDGET_COUNTS = Counter({1: 3, 2: 3, 3: 6})


def _authenticate_stream(tasks: tuple[ContinualToySweTrainTaskV4, ...]) -> None:
    if len(tasks) != TASK_COUNT:
        raise ContinualToySweTaskError("V4 TRAIN stream count changed")
    for task in tasks:
        task.authenticate()
    if len({task.task_sha256 for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskError("V4 task commitments are not unique")
    if len({task.public.signature.entry_point for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskError("V4 entry points are not unique")
    if Counter(task.private.stage for task in tasks) != _EXPECTED_STAGE_COUNTS:
        raise ContinualToySweTaskError("V4 stage counts changed")
    if Counter(task.private.capability_group for task in tasks) != Counter(
        {group: 1 for group in _ALL_GROUPS}
    ):
        raise ContinualToySweTaskError("V4 capability group counts changed")
    if Counter(task.private.expected_action for task in tasks) != _EXPECTED_ACTION_COUNTS:
        raise ContinualToySweTaskError("V4 terminal-action counts changed")
    if Counter(task.private.attempt_budget for task in tasks) != _EXPECTED_BUDGET_COUNTS:
        raise ContinualToySweTaskError("V4 attempt-budget counts changed")


def continual_toy_swe_train_stream_v4() -> tuple[ContinualToySweTrainTaskV4, ...]:
    """Return the authenticated immutable ordered V4 public-TRAIN stream."""

    _authenticate_stream(_TRAIN_STREAM)
    return _TRAIN_STREAM


def learner_visible_observation_v4(
    task: ContinualToySweTrainTaskV4,
) -> LearnerVisibleToySweObservationV4:
    """Project exactly the task-family-free learner record."""

    if type(task) is not ContinualToySweTrainTaskV4:
        raise TypeError("learner projection requires one exact V4 task")
    task.authenticate()
    return task.public


def train_stream_sha256_v4(
    tasks: tuple[ContinualToySweTrainTaskV4, ...] | None = None,
) -> str:
    selected = _TRAIN_STREAM if tasks is None else tasks
    if not isinstance(selected, tuple) or not all(
        type(task) is ContinualToySweTrainTaskV4 for task in selected
    ):
        raise TypeError("V4 TRAIN stream must contain exact continual tasks")
    _authenticate_stream(selected)
    return _digest(
        b"continual-toy-swe-stream-v4",
        {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_sha256s": tuple(task.task_sha256 for task in selected),
        },
    )


TRAIN_STREAM_SHA256 = train_stream_sha256_v4()
_STREAM_CUSTODY = {
    "schema": SCHEMA,
    "partition": PARTITION,
    "task_count": TASK_COUNT,
    "task_sha256s": tuple(task.task_sha256 for task in _TRAIN_STREAM),
    "train_stream_sha256": TRAIN_STREAM_SHA256,
    "stage_counts": dict(sorted(_EXPECTED_STAGE_COUNTS.items())),
    "expected_action_counts": dict(sorted(_EXPECTED_ACTION_COUNTS.items())),
    "attempt_budget_counts": dict(sorted(_EXPECTED_BUDGET_COUNTS.items())),
    "learner_visible_fields": (
        "task_commitment_sha256",
        "relative_path",
        "signature",
        "prompt",
        "starter_source",
    ),
    "authored_before_v4_implementation": True,
    "successful_target_sources_stored": 0,
    "target_teacher_sources": 0,
    "v3_task_inputs_used": 0,
}
TRAIN_STREAM_CUSTODY_SHA256 = _digest(
    b"continual-toy-swe-stream-custody-v4",
    _STREAM_CUSTODY,
)


def train_stream_custody_v4() -> dict[str, object]:
    """Return a detached authenticated custody record for the frozen V4 stream."""

    if train_stream_sha256_v4() != TRAIN_STREAM_SHA256:
        raise ContinualToySweTaskError("V4 TRAIN stream root changed")
    if (
        _digest(b"continual-toy-swe-stream-custody-v4", _STREAM_CUSTODY)
        != TRAIN_STREAM_CUSTODY_SHA256
    ):
        raise ContinualToySweTaskError("V4 stream custody changed")
    return json.loads(_canonical_json(_STREAM_CUSTODY))


__all__ = [
    "EXECUTION_PROPERTY_CASES",
    "EXECUTION_REGRESSION_CASES",
    "PARTITION",
    "SCHEMA",
    "TASK_COUNT",
    "TRAIN_STREAM_CUSTODY_SHA256",
    "TRAIN_STREAM_SHA256",
    "ContinualToySweTaskError",
    "ContinualToySweTrainTaskV4",
    "LearnerVisibleToySweObservationV4",
    "ToySweEnvironmentPrivateV4",
    "continual_toy_swe_train_stream_v4",
    "learner_visible_observation_v4",
    "train_stream_custody_v4",
    "train_stream_sha256_v4",
]
