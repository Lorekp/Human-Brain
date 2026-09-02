"""Deterministic self-posed public-TRAIN tasks for continual repository learning.

The learner-visible record contains only the prompt, typed signature, and an
inspectable starter file. Hidden regression/property cases and research-only
semantic grouping remain in the environment-private record.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath

from humanbrain.simple_swe.tasks import GeneratedTask, JsonValue, make_task
from humanbrain.typed_program_types import ProgramParameter, ProgramType, TypedSignature

SCHEMA = "continual-toy-swe-tasks-v1"
PARTITION = "train"
TASK_COUNT = 10
REGRESSION_CASES_PER_TASK = 4
PROPERTY_CASES_PER_TASK = 12

_FAMILY = "t0_integrated"
_FIRST_EXPOSURE = "first-exposure"
_FRESH_REPLAY = "fresh-replay"
_PARAPHRASED_TRANSFER = "paraphrased-transfer"
_UNSEEN = "unseen"
_STAGES = frozenset(
    {_FIRST_EXPOSURE, _FRESH_REPLAY, _PARAPHRASED_TRANSFER, _UNSEEN}
)
_CLAMPED_SUM = "clamped-sum"
_COUNT_ABOVE = "count-above"
_STABLE_UNIQUE = "stable-unique-characters"
_ROTATE_RIGHT = "rotate-right-unseen"
_KNOWN_GROUPS = frozenset({_CLAMPED_SUM, _COUNT_ABOVE, _STABLE_UNIQUE})
_ALL_GROUPS = _KNOWN_GROUPS | {_ROTATE_RIGHT}

_INT = ProgramType.Int()
_STRING = ProgramType.String()
_LIST_INT = ProgramType.List(_INT)


class ContinualToySweTaskError(ValueError):
    """A task, observation, or private environment record failed authentication."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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
        '    """Implement the behavior described in the public TRAIN prompt."""\n'
        '    raise NotImplementedError("public TRAIN starter")\n'
    )


@dataclass(frozen=True, slots=True)
class LearnerVisibleToySweObservationV1:
    """The complete target-free task record allowed into learner observations."""

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
        expected = _digest(b"continual-toy-swe-observation-v1", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskError("learner-visible observation failed authentication")


@dataclass(frozen=True, slots=True)
class ToySweEnvironmentPrivateV1:
    """Environment-only tests and post-run research grouping."""

    generated_task: GeneratedTask
    semantic_group: str
    stage: str
    environment_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_task_sha256": self.generated_task.public.task_sha256,
            "private_test_sha256": self.generated_task.private.test_sha256,
            "semantic_group": self.semantic_group,
            "stage": self.stage,
        }

    def authenticate(self) -> None:
        if type(self.generated_task) is not GeneratedTask:
            raise TypeError("environment task must be one exact GeneratedTask")
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
            raise ContinualToySweTaskError("environment Simple-SWE commitments changed")
        if self.semantic_group not in _ALL_GROUPS or self.stage not in _STAGES:
            raise ContinualToySweTaskError("unknown private group or stage")
        if (self.stage == _UNSEEN) != (self.semantic_group == _ROTATE_RIGHT):
            raise ContinualToySweTaskError("unseen stage and private semantic group disagree")
        expected = _digest(b"continual-toy-swe-environment-v1", self.record())
        if self.environment_sha256 != expected:
            raise ContinualToySweTaskError("private environment record failed authentication")


@dataclass(frozen=True, slots=True)
class ContinualToySweTrainTaskV1:
    public: LearnerVisibleToySweObservationV1
    private: ToySweEnvironmentPrivateV1
    task_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_observation_sha256": self.public.observation_sha256,
            "private_environment_sha256": self.private.environment_sha256,
        }

    def authenticate(self) -> None:
        if type(self.public) is not LearnerVisibleToySweObservationV1:
            raise TypeError("task public record must be one exact learner observation")
        if type(self.private) is not ToySweEnvironmentPrivateV1:
            raise TypeError("task private record must be one exact environment record")
        self.public.authenticate()
        self.private.authenticate()
        generated = self.private.generated_task.public
        if (
            self.public.task_commitment_sha256 != generated.task_sha256
            or self.public.signature != generated.signature
            or self.public.prompt != generated.prompt
        ):
            raise ContinualToySweTaskError("public observation is not bound to its private task")
        expected = _digest(b"continual-toy-swe-task-v1", self.record())
        if self.task_sha256 != expected:
            raise ContinualToySweTaskError("continual task failed authentication")


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    description: str
    semantic_group: str
    stage: str
    seed: int


_SPECS = (
    _TaskSpec(
        "Clamp each integer in values to the inclusive interval from -limit to limit, "
        "then return their sum. limit is nonnegative.",
        _CLAMPED_SUM,
        _FIRST_EXPOSURE,
        11,
    ),
    _TaskSpec(
        "Return how many integers in values are strictly greater than threshold.",
        _COUNT_ABOVE,
        _FIRST_EXPOSURE,
        17,
    ),
    _TaskSpec(
        "Return text with duplicate characters removed, keeping only the first "
        "occurrence of each character in original order.",
        _STABLE_UNIQUE,
        _FIRST_EXPOSURE,
        23,
    ),
    _TaskSpec(
        "For every value, use max(-limit, min(limit, value)); add the transformed "
        "values and return the total. limit is nonnegative.",
        _CLAMPED_SUM,
        _FRESH_REPLAY,
        31,
    ),
    _TaskSpec(
        "Scan values and count elements for which value > threshold; return that count.",
        _COUNT_ABOVE,
        _FRESH_REPLAY,
        37,
    ),
    _TaskSpec(
        "Build a string by scanning left to right and append a character only the "
        "first time it appears.",
        _STABLE_UNIQUE,
        _FRESH_REPLAY,
        43,
    ),
    _TaskSpec(
        "Treat limit as a magnitude cap: preserve each reading's sign, cap its "
        "magnitude, and total the capped readings.",
        _CLAMPED_SUM,
        _PARAPHRASED_TRANSFER,
        53,
    ),
    _TaskSpec(
        "Report the number of readings that clear the cutoff threshold. Equal readings "
        "do not count.",
        _COUNT_ABOVE,
        _PARAPHRASED_TRANSFER,
        59,
    ),
    _TaskSpec(
        "Collapse repeated symbols across the whole text while preserving the earliest "
        "copy and original order.",
        _STABLE_UNIQUE,
        _PARAPHRASED_TRANSFER,
        67,
    ),
    _TaskSpec(
        "Rotate values to the right by steps positions. Negative steps rotate left. "
        "Return an empty list when values is empty.",
        _ROTATE_RIGHT,
        _UNSEEN,
        71,
    ),
)


def _signature(group: str, entry_point: str) -> TypedSignature:
    if group == _CLAMPED_SUM:
        parameters = (
            ProgramParameter("values", _LIST_INT),
            ProgramParameter("limit", _INT),
        )
        result = _INT
    elif group == _COUNT_ABOVE:
        parameters = (
            ProgramParameter("values", _LIST_INT),
            ProgramParameter("threshold", _INT),
        )
        result = _INT
    elif group == _STABLE_UNIQUE:
        parameters = (ProgramParameter("text", _STRING),)
        result = _STRING
    elif group == _ROTATE_RIGHT:
        parameters = (
            ProgramParameter("values", _LIST_INT),
            ProgramParameter("steps", _INT),
        )
        result = _LIST_INT
    else:
        raise ValueError("unknown private semantic group")
    return TypedSignature(entry_point=entry_point, parameters=parameters, result=result)


def _clamped_sum(values: list[int], limit: int) -> int:
    return sum(max(-limit, min(limit, value)) for value in values)


def _count_above(values: list[int], threshold: int) -> int:
    return sum(value > threshold for value in values)


def _stable_unique(text: str) -> str:
    return "".join(dict.fromkeys(text))


def _rotate_right(values: list[int], steps: int) -> list[int]:
    if not values:
        return []
    offset = steps % len(values)
    if offset == 0:
        return list(values)
    return values[-offset:] + values[:-offset]


def _list_values(seed: int, index: int, *, maximum_length: int = 7) -> list[int]:
    length = (seed + index * 3) % (maximum_length + 1)
    return [((seed * 11 + index * 7 + position * 13) % 61) - 30 for position in range(length)]


def _cases(
    group: str, seed: int
) -> tuple[
    Callable[..., JsonValue],
    tuple[tuple[JsonValue, ...], ...],
    tuple[tuple[tuple[JsonValue, ...], str], ...],
]:
    if group == _CLAMPED_SUM:
        oracle: Callable[..., JsonValue] = _clamped_sum
        regressions: tuple[tuple[JsonValue, ...], ...] = (
            ([], seed % 5),
            ([-seed, 0, seed], 4),
            ([-20, -3, 2, 19], 6),
            ([1, 2, 3, -4], 0),
        )
        properties = tuple(
            ((_list_values(seed, index), (seed + index * 3) % 11), "deterministic-property")
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _COUNT_ABOVE:
        oracle = _count_above
        regressions = (
            ([], seed % 9),
            ([-1, 0, 1], 0),
            ([seed, seed, seed + 1], seed),
            ([-20, 4, 5, 17], 4),
        )
        properties = tuple(
            ((_list_values(seed, index), ((seed + index * 5) % 41) - 20), "deterministic-property")
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _STABLE_UNIQUE:
        oracle = _stable_unique
        regressions = (
            ("",),
            ("aaaa",),
            (f"abca{seed % 10}",),
            ("mississippi",),
        )
        alphabet = "abBA01_-xy"
        properties = tuple(
            (
                (
                    "".join(
                        alphabet[(seed + index * 3 + position * 5) % len(alphabet)]
                        for position in range((seed + index * 2) % 15)
                    ),
                ),
                "deterministic-property",
            )
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _ROTATE_RIGHT:
        oracle = _rotate_right
        regressions = (
            ([], 3),
            ([1], seed),
            ([1, 2, 3, 4], 1),
            ([1, 2, 3, 4], -1),
        )
        properties = tuple(
            ((_list_values(seed, index), ((seed + index * 7) % 25) - 12), "deterministic-property")
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    else:
        raise ValueError("unknown private semantic group")
    return oracle, regressions, properties


def _build_task(spec: _TaskSpec, ordinal: int) -> ContinualToySweTrainTaskV1:
    entry_point = f"solve_train_{ordinal:02d}"
    signature = _signature(spec.semantic_group, entry_point)
    oracle, regressions, properties = _cases(spec.semantic_group, spec.seed)
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
    starter_source = _starter_source(signature)
    relative_path = f"solutions/challenge_{ordinal:02d}.py"
    public_placeholder = LearnerVisibleToySweObservationV1(
        task_commitment_sha256=generated.public.task_sha256,
        relative_path=relative_path,
        signature=signature,
        prompt=generated.public.prompt,
        starter_source=starter_source,
        observation_sha256="",
    )
    public = LearnerVisibleToySweObservationV1(
        task_commitment_sha256=public_placeholder.task_commitment_sha256,
        relative_path=public_placeholder.relative_path,
        signature=public_placeholder.signature,
        prompt=public_placeholder.prompt,
        starter_source=public_placeholder.starter_source,
        observation_sha256=_digest(
            b"continual-toy-swe-observation-v1", public_placeholder.record()
        ),
    )
    private_placeholder = ToySweEnvironmentPrivateV1(
        generated_task=generated,
        semantic_group=spec.semantic_group,
        stage=spec.stage,
        environment_sha256="",
    )
    private = ToySweEnvironmentPrivateV1(
        generated_task=private_placeholder.generated_task,
        semantic_group=private_placeholder.semantic_group,
        stage=private_placeholder.stage,
        environment_sha256=_digest(
            b"continual-toy-swe-environment-v1", private_placeholder.record()
        ),
    )
    task_placeholder = ContinualToySweTrainTaskV1(
        public=public,
        private=private,
        task_sha256="",
    )
    task = ContinualToySweTrainTaskV1(
        public=public,
        private=private,
        task_sha256=_digest(b"continual-toy-swe-task-v1", task_placeholder.record()),
    )
    task.authenticate()
    return task


_TRAIN_STREAM = tuple(_build_task(spec, ordinal) for ordinal, spec in enumerate(_SPECS))


def continual_toy_swe_train_stream_v1() -> tuple[ContinualToySweTrainTaskV1, ...]:
    """Return the immutable, ordered, fresh public-TRAIN task stream."""

    for task in _TRAIN_STREAM:
        task.authenticate()
    return _TRAIN_STREAM


def learner_visible_observation_v1(
    task: ContinualToySweTrainTaskV1,
) -> LearnerVisibleToySweObservationV1:
    """Project exactly the target-free record allowed into an ObservationFrame."""

    if type(task) is not ContinualToySweTrainTaskV1:
        raise TypeError("learner observation requires one exact continual TRAIN task")
    task.authenticate()
    return task.public


def train_stream_sha256_v1(
    tasks: tuple[ContinualToySweTrainTaskV1, ...] | None = None,
) -> str:
    selected = _TRAIN_STREAM if tasks is None else tasks
    if not isinstance(selected, tuple) or not all(
        type(task) is ContinualToySweTrainTaskV1 for task in selected
    ):
        raise TypeError("TRAIN stream must be a tuple of exact continual tasks")
    for task in selected:
        task.authenticate()
    return _digest(
        b"continual-toy-swe-stream-v1",
        {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_sha256s": tuple(task.task_sha256 for task in selected),
        },
    )


TRAIN_STREAM_SHA256 = train_stream_sha256_v1()


__all__ = [
    "PARTITION",
    "PROPERTY_CASES_PER_TASK",
    "REGRESSION_CASES_PER_TASK",
    "SCHEMA",
    "TASK_COUNT",
    "TRAIN_STREAM_SHA256",
    "ContinualToySweTaskError",
    "ContinualToySweTrainTaskV1",
    "LearnerVisibleToySweObservationV1",
    "ToySweEnvironmentPrivateV1",
    "continual_toy_swe_train_stream_v1",
    "learner_visible_observation_v1",
    "train_stream_sha256_v1",
]
