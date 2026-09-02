"""Fresh public-TRAIN pressure tasks for replay-free continual coding.

Learner-visible records contain only an authenticated prompt, exact typed
signature, repository path, and inert starter source. Regression/property
cases and research-only capability grouping remain evaluator-only. No repaired
or successful source is constructed by this module.
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

SCHEMA = "continual-toy-swe-tasks-v2"
PARTITION = "train"
TASK_COUNT = 12
REGRESSION_CASES_PER_TASK = 4
PROPERTY_CASES_PER_TASK = 12

_FAMILY = "t0_continual_v2"
_PRIOR_ALPHA = "public-prior-alpha-transfer"
_STRUCTURAL_REPEAT = "within-stream-structural-repeat"
_UNSEEN = "unseen-behavior"
_STAGES = frozenset({_PRIOR_ALPHA, _STRUCTURAL_REPEAT, _UNSEEN})

_ABSOLUTE_GAP = "absolute-gap"
_EVEN_FILTER = "even-filter"
_NORMALIZE_TEXT = "normalize-text"
_AFFINE_SCALAR = "affine-scalar"
_PREFIX_SUMS = "prefix-sums"
_REVERSE_WORDS = "reverse-word-order"
_NONNEGATIVE_RUN = "longest-nonnegative-run"
_ALTERNATING_MERGE = "alternating-merge"
_TRANSFER_GROUPS = frozenset(
    {_ABSOLUTE_GAP, _EVEN_FILTER, _NORMALIZE_TEXT, _AFFINE_SCALAR}
)
_UNSEEN_GROUPS = frozenset(
    {_PREFIX_SUMS, _REVERSE_WORDS, _NONNEGATIVE_RUN, _ALTERNATING_MERGE}
)
_ALL_GROUPS = _TRANSFER_GROUPS | _UNSEEN_GROUPS

_INT = ProgramType.Int()
_STRING = ProgramType.String()
_LIST_INT = ProgramType.List(_INT)


class ContinualToySweTaskError(ValueError):
    """A V2 task, observation, or evaluator record failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _starter_source(signature: TypedSignature) -> str:
    parameters = ", ".join(parameter.name for parameter in signature.parameters)
    return (
        f"def {signature.entry_point}({parameters}):\n"
        '    """Implement the behavior in the public TRAIN prompt."""\n'
        '    raise NotImplementedError("public TRAIN v2 starter")\n'
    )


@dataclass(frozen=True, slots=True)
class LearnerVisibleToySweObservationV2:
    """The complete target-teacher-free record allowed into learner input."""

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
            raise ContinualToySweTaskError("public task commitment is not a SHA-256 root")
        if type(self.signature) is not TypedSignature:
            raise TypeError("learner-visible signature must be one exact TypedSignature")
        path = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".py"
            or self.relative_path != path.as_posix()
        ):
            raise ContinualToySweTaskError("learner-visible repository path is not canonical")
        if not self.prompt or self.starter_source != _starter_source(self.signature):
            raise ContinualToySweTaskError("learner-visible prompt or starter source changed")
        expected = _digest(b"continual-toy-swe-observation-v2", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskError("learner-visible observation failed authentication")


@dataclass(frozen=True, slots=True)
class ToySweEnvironmentPrivateV2:
    """Evaluator-only cases and post-run capability grouping."""

    generated_task: GeneratedTask
    capability_group: str
    stage: str
    environment_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_task_sha256": self.generated_task.public.task_sha256,
            "private_test_sha256": self.generated_task.private.test_sha256,
            "capability_group": self.capability_group,
            "stage": self.stage,
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
            or len(private.regression_cases) != REGRESSION_CASES_PER_TASK
            or len(private.property_cases) != PROPERTY_CASES_PER_TASK
        ):
            raise ContinualToySweTaskError("evaluator Simple-SWE commitments changed")
        if self.capability_group not in _ALL_GROUPS or self.stage not in _STAGES:
            raise ContinualToySweTaskError("unknown evaluator capability group or stage")
        if self.stage == _UNSEEN:
            valid_pair = self.capability_group in _UNSEEN_GROUPS
        else:
            valid_pair = self.capability_group in _TRANSFER_GROUPS
        if not valid_pair:
            raise ContinualToySweTaskError("capability group and stage disagree")
        expected = _digest(b"continual-toy-swe-environment-v2", self.record())
        if self.environment_sha256 != expected:
            raise ContinualToySweTaskError("evaluator environment failed authentication")


@dataclass(frozen=True, slots=True)
class ContinualToySweTrainTaskV2:
    public: LearnerVisibleToySweObservationV2
    private: ToySweEnvironmentPrivateV2
    task_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_observation_sha256": self.public.observation_sha256,
            "private_environment_sha256": self.private.environment_sha256,
        }

    def authenticate(self) -> None:
        if type(self.public) is not LearnerVisibleToySweObservationV2:
            raise TypeError("task public record must be one exact V2 learner observation")
        if type(self.private) is not ToySweEnvironmentPrivateV2:
            raise TypeError("task private record must be one exact V2 environment")
        self.public.authenticate()
        self.private.authenticate()
        generated = self.private.generated_task.public
        if (
            self.public.task_commitment_sha256 != generated.task_sha256
            or self.public.signature != generated.signature
            or self.public.prompt != generated.prompt
        ):
            raise ContinualToySweTaskError("public observation is not bound to evaluator task")
        expected = _digest(b"continual-toy-swe-task-v2", self.record())
        if self.task_sha256 != expected:
            raise ContinualToySweTaskError("continual V2 task failed authentication")


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    description: str
    capability_group: str
    stage: str
    seed: int
    entry_point: str
    parameter_names: tuple[str, ...]


_SPECS = (
    _TaskSpec(
        "Return the non-negative distance between west_cairn and east_cairn. "
        "unused_mist does not affect the result.",
        _ABSOLUTE_GAP,
        _PRIOR_ALPHA,
        101,
        "span_between_cairns",
        ("west_cairn", "east_cairn", "unused_mist"),
    ),
    _TaskSpec(
        "Return a new list containing only the even integers from embers, preserving "
        "their original order. unused_flint and unused_ash do not affect the result.",
        _EVEN_FILTER,
        _PRIOR_ALPHA,
        109,
        "sift_even_embers",
        ("embers", "unused_flint", "unused_ash"),
    ),
    _TaskSpec(
        "Remove whitespace from both ends of frosted_phrase and convert all remaining "
        "letters to lowercase. unused_reed and unused_dune do not affect the result.",
        _NORMALIZE_TEXT,
        _PRIOR_ALPHA,
        127,
        "settle_frosted_phrase",
        ("frosted_phrase", "unused_reed", "unused_dune"),
    ),
    _TaskSpec(
        "Multiply scale_seed by base_value, then add tail_offset.",
        _AFFINE_SCALAR,
        _PRIOR_ALPHA,
        139,
        "weave_scale_offset",
        ("scale_seed", "base_value", "tail_offset"),
    ),
    _TaskSpec(
        "Measure how far apart first_beacon and second_beacon are on the integer line "
        "and return that distance without a sign. spare_signal is ignored.",
        _ABSOLUTE_GAP,
        _STRUCTURAL_REPEAT,
        151,
        "measure_beacon_separation",
        ("first_beacon", "second_beacon", "spare_signal"),
    ),
    _TaskSpec(
        "Scan readings left to right and keep exactly those divisible by two; return "
        "them in scan order. spare_flag and spare_note are irrelevant.",
        _EVEN_FILTER,
        _STRUCTURAL_REPEAT,
        163,
        "retain_divisible_pairs",
        ("readings", "spare_flag", "spare_note"),
    ),
    _TaskSpec(
        "Canonicalize raw_label by trimming its outer whitespace before lowercasing it. "
        "spare_left and spare_right are unused.",
        _NORMALIZE_TEXT,
        _STRUCTURAL_REPEAT,
        179,
        "clean_margin_case",
        ("raw_label", "spare_left", "spare_right"),
    ),
    _TaskSpec(
        "Scale quantity by stride and then apply bias to the scaled value.",
        _AFFINE_SCALAR,
        _STRUCTURAL_REPEAT,
        193,
        "combine_stride_bias",
        ("stride", "quantity", "bias"),
    ),
    _TaskSpec(
        "Return cumulative totals for river_steps: output position i is the sum through "
        "input position i. Return an empty list for empty input.",
        _PREFIX_SUMS,
        _UNSEEN,
        211,
        "accumulate_river_steps",
        ("river_steps",),
    ),
    _TaskSpec(
        "Split constellation on arbitrary whitespace and return the words in reverse "
        "order joined by single spaces. Return an empty string when there are no words.",
        _REVERSE_WORDS,
        _UNSEEN,
        223,
        "reverse_constellation_words",
        ("constellation",),
    ),
    _TaskSpec(
        "Return the length of the longest consecutive run of values in weather_codes "
        "that are zero or positive. Return zero when no such value exists.",
        _NONNEGATIVE_RUN,
        _UNSEEN,
        239,
        "longest_clear_weather",
        ("weather_codes",),
    ),
    _TaskSpec(
        "Return a new list by alternating values from north_flow and south_flow, "
        "beginning with north_flow; after one ends, append the other remainder.",
        _ALTERNATING_MERGE,
        _UNSEEN,
        251,
        "braid_tributary_values",
        ("north_flow", "south_flow"),
    ),
)


def _signature(spec: _TaskSpec) -> TypedSignature:
    group = spec.capability_group
    if group in {_ABSOLUTE_GAP, _AFFINE_SCALAR}:
        parameter_types = (_INT, _INT, _INT)
        result = _INT
    elif group == _EVEN_FILTER:
        parameter_types = (_LIST_INT, _INT, _INT)
        result = _LIST_INT
    elif group == _NORMALIZE_TEXT:
        parameter_types = (_STRING, _STRING, _STRING)
        result = _STRING
    elif group in {_PREFIX_SUMS, _NONNEGATIVE_RUN}:
        parameter_types = (_LIST_INT,)
        result = _LIST_INT if group == _PREFIX_SUMS else _INT
    elif group == _REVERSE_WORDS:
        parameter_types = (_STRING,)
        result = _STRING
    elif group == _ALTERNATING_MERGE:
        parameter_types = (_LIST_INT, _LIST_INT)
        result = _LIST_INT
    else:
        raise ValueError("unknown V2 capability group")
    if len(spec.parameter_names) != len(parameter_types):
        raise ValueError("V2 task parameter names and types disagree")
    return TypedSignature(
        entry_point=spec.entry_point,
        parameters=tuple(
            ProgramParameter(name, parameter_type)
            for name, parameter_type in zip(
                spec.parameter_names, parameter_types, strict=True
            )
        ),
        result=result,
    )


def _integer(seed: int, index: int, channel: int) -> int:
    return ((seed * 17 + index * 23 + channel * 31) % 101) - 50


def _integer_list(seed: int, index: int, channel: int = 0) -> list[int]:
    length = (seed + index * 5 + channel * 3) % 9
    return [_integer(seed, index + position, channel + position) for position in range(length)]


def _mixed_text(seed: int, index: int) -> str:
    alphabet = " aAbBxyZ019_-\t"
    length = (seed + index * 7) % 22
    body = "".join(
        alphabet[(seed + index * 3 + position * 5) % len(alphabet)]
        for position in range(length)
    )
    return (" " * (index % 3)) + body + ("\t" * ((seed + index) % 2))


def _word_text(seed: int, index: int) -> str:
    words = ("cairn", "mist", "ember", "north", "river", "wisp")
    separators = (" ", "  ", "\t")
    count = (seed + index * 3) % 7
    selected = [words[(seed + index + position * 2) % len(words)] for position in range(count)]
    separator = separators[(seed + index) % len(separators)]
    return (" " if index % 2 else "") + separator.join(selected) + ("  " if index % 3 == 0 else "")


def _absolute_gap(first: int, second: int, unused: int) -> int:
    del unused
    return abs(first - second)


def _even_filter(values: list[int], unused_left: int, unused_right: int) -> list[int]:
    del unused_left, unused_right
    return [value for value in values if value % 2 == 0]


def _normalize_text(text: str, unused_left: str, unused_right: str) -> str:
    del unused_left, unused_right
    return text.strip().lower()


def _affine_scalar(scale: int, value: int, offset: int) -> int:
    return scale * value + offset


def _prefix_sums(values: list[int]) -> list[int]:
    total = 0
    result: list[int] = []
    for value in values:
        total += value
        result.append(total)
    return result


def _reverse_words(text: str) -> str:
    return " ".join(reversed(text.split()))


def _longest_nonnegative_run(values: list[int]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value >= 0 else 0
        best = max(best, current)
    return best


def _alternating_merge(left: list[int], right: list[int]) -> list[int]:
    result: list[int] = []
    shared = min(len(left), len(right))
    for index in range(shared):
        result.extend((left[index], right[index]))
    result.extend(left[shared:])
    result.extend(right[shared:])
    return result


def _cases(
    group: str, seed: int
) -> tuple[
    Callable[..., JsonValue],
    tuple[tuple[JsonValue, ...], ...],
    tuple[tuple[tuple[JsonValue, ...], str], ...],
]:
    if group == _ABSOLUTE_GAP:
        oracle: Callable[..., JsonValue] = _absolute_gap
        regressions: tuple[tuple[JsonValue, ...], ...] = (
            (-seed, seed + 3, 0),
            (seed, seed, 1),
            (seed + 7, -2, 2),
            (-11, 4, seed),
        )
        properties = tuple(
            (
                (_integer(seed, index, 0), _integer(seed, index, 1), index),
                "deterministic-v2-property",
            )
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _EVEN_FILTER:
        oracle = _even_filter
        regressions = (
            ([], seed, -seed),
            ([0, 1, 2, 3], 0, 1),
            ([-4, -3, -2, -1], 2, 3),
            ([seed, seed + 1, seed + 2], 4, 5),
        )
        properties = tuple(
            (
                (_integer_list(seed, index), _integer(seed, index, 1), index),
                "deterministic-v2-property",
            )
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _NORMALIZE_TEXT:
        oracle = _normalize_text
        regressions = (
            ("", "left", "right"),
            ("  FoGgY Harbor  ", "reed", "dune"),
            ("\tAlready Lower\n", "a", "b"),
            (f"  Seed {seed}  ", "unused", "unused"),
        )
        properties = tuple(
            (
                (_mixed_text(seed, index), f"left-{index}", f"right-{seed}"),
                "deterministic-v2-property",
            )
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _AFFINE_SCALAR:
        oracle = _affine_scalar
        regressions = (
            (0, seed, -seed),
            (1, seed, 3),
            (-2, 7, seed % 11),
            (seed % 9, -5, 4),
        )
        properties = tuple(
            (
                (
                    _integer(seed, index, 0),
                    _integer(seed, index, 1),
                    _integer(seed, index, 2),
                ),
                "deterministic-v2-property",
            )
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _PREFIX_SUMS:
        oracle = _prefix_sums
        regressions = (([],), ([seed],), ([1, -1, 2],), ([-3, 4, 5, -2],))
        properties = tuple(
            ((_integer_list(seed, index),), "deterministic-v2-property")
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _REVERSE_WORDS:
        oracle = _reverse_words
        regressions = (("",), ("single",), ("north south",), ("  mist\tember  river ",))
        properties = tuple(
            ((_word_text(seed, index),), "deterministic-v2-property")
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _NONNEGATIVE_RUN:
        oracle = _longest_nonnegative_run
        regressions = (([],), ([-3, -1],), ([0, 1, 2],), ([1, -1, 0, 4, -2, 5],))
        properties = tuple(
            ((_integer_list(seed, index),), "deterministic-v2-property")
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _ALTERNATING_MERGE:
        oracle = _alternating_merge
        regressions = (
            ([], []),
            ([1, 2], []),
            ([], [3, 4]),
            ([1, 2, 3], [8, 9]),
        )
        properties = tuple(
            (
                (_integer_list(seed, index, 0), _integer_list(seed, index, 1)),
                "deterministic-v2-property",
            )
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    else:
        raise ValueError("unknown V2 capability group")
    return oracle, regressions, properties


def _build_task(spec: _TaskSpec, ordinal: int) -> ContinualToySweTrainTaskV2:
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
    public_placeholder = LearnerVisibleToySweObservationV2(
        task_commitment_sha256=generated.public.task_sha256,
        relative_path=f"workbench/v2_pressure_{ordinal:02d}.py",
        signature=signature,
        prompt=generated.public.prompt,
        starter_source=_starter_source(signature),
        observation_sha256="",
    )
    public = LearnerVisibleToySweObservationV2(
        task_commitment_sha256=public_placeholder.task_commitment_sha256,
        relative_path=public_placeholder.relative_path,
        signature=public_placeholder.signature,
        prompt=public_placeholder.prompt,
        starter_source=public_placeholder.starter_source,
        observation_sha256=_digest(
            b"continual-toy-swe-observation-v2", public_placeholder.record()
        ),
    )
    private_placeholder = ToySweEnvironmentPrivateV2(
        generated_task=generated,
        capability_group=spec.capability_group,
        stage=spec.stage,
        environment_sha256="",
    )
    private = ToySweEnvironmentPrivateV2(
        generated_task=generated,
        capability_group=private_placeholder.capability_group,
        stage=private_placeholder.stage,
        environment_sha256=_digest(
            b"continual-toy-swe-environment-v2", private_placeholder.record()
        ),
    )
    task_placeholder = ContinualToySweTrainTaskV2(
        public=public,
        private=private,
        task_sha256="",
    )
    task = ContinualToySweTrainTaskV2(
        public=public,
        private=private,
        task_sha256=_digest(b"continual-toy-swe-task-v2", task_placeholder.record()),
    )
    task.authenticate()
    return task


if len(_SPECS) != TASK_COUNT:
    raise ValueError("V2 pressure stream must contain exactly twelve tasks")

_TRAIN_STREAM = tuple(_build_task(spec, ordinal) for ordinal, spec in enumerate(_SPECS))

_EXPECTED_STAGE_COUNTS = {
    _PRIOR_ALPHA: 4,
    _STRUCTURAL_REPEAT: 4,
    _UNSEEN: 4,
}
_EXPECTED_GROUP_COUNTS = {
    _ABSOLUTE_GAP: 2,
    _EVEN_FILTER: 2,
    _NORMALIZE_TEXT: 2,
    _AFFINE_SCALAR: 2,
    _PREFIX_SUMS: 1,
    _REVERSE_WORDS: 1,
    _NONNEGATIVE_RUN: 1,
    _ALTERNATING_MERGE: 1,
}


def _authenticate_stream(tasks: tuple[ContinualToySweTrainTaskV2, ...]) -> None:
    if len(tasks) != TASK_COUNT:
        raise ContinualToySweTaskError("V2 TRAIN stream count changed")
    for task in tasks:
        task.authenticate()
    if len({task.task_sha256 for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskError("V2 TRAIN task commitments are not unique")
    if len({task.public.signature.entry_point for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskError("V2 TRAIN entry points are not unique")
    if Counter(task.private.stage for task in tasks) != _EXPECTED_STAGE_COUNTS:
        raise ContinualToySweTaskError("V2 capability stage counts changed")
    if Counter(task.private.capability_group for task in tasks) != _EXPECTED_GROUP_COUNTS:
        raise ContinualToySweTaskError("V2 capability group counts changed")


def continual_toy_swe_train_stream_v2() -> tuple[ContinualToySweTrainTaskV2, ...]:
    """Return the immutable ordered stream shared by treatment and no-update control."""

    _authenticate_stream(_TRAIN_STREAM)
    return _TRAIN_STREAM


def learner_visible_observation_v2(
    task: ContinualToySweTrainTaskV2,
) -> LearnerVisibleToySweObservationV2:
    """Project exactly the target-teacher-free learner record."""

    if type(task) is not ContinualToySweTrainTaskV2:
        raise TypeError("learner observation requires one exact V2 continual task")
    task.authenticate()
    return task.public


def train_stream_sha256_v2(
    tasks: tuple[ContinualToySweTrainTaskV2, ...] | None = None,
) -> str:
    selected = _TRAIN_STREAM if tasks is None else tasks
    if not isinstance(selected, tuple) or not all(
        type(task) is ContinualToySweTrainTaskV2 for task in selected
    ):
        raise TypeError("V2 TRAIN stream must be a tuple of exact continual tasks")
    _authenticate_stream(selected)
    return _digest(
        b"continual-toy-swe-stream-v2",
        {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_sha256s": tuple(task.task_sha256 for task in selected),
        },
    )


TRAIN_STREAM_SHA256 = train_stream_sha256_v2()


__all__ = [
    "PARTITION",
    "PROPERTY_CASES_PER_TASK",
    "REGRESSION_CASES_PER_TASK",
    "SCHEMA",
    "TASK_COUNT",
    "TRAIN_STREAM_SHA256",
    "ContinualToySweTaskError",
    "ContinualToySweTrainTaskV2",
    "LearnerVisibleToySweObservationV2",
    "ToySweEnvironmentPrivateV2",
    "continual_toy_swe_train_stream_v2",
    "learner_visible_observation_v2",
    "train_stream_sha256_v2",
]
