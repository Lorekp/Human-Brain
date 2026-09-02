"""Fresh mechanism-blind public-TRAIN tasks for continual coding V3.

Learner-visible observations contain only a public prompt, typed signature,
repository path, and inert starter source. Oracles, cases, and research-only
capability labels remain evaluator-private; no successful source is stored.
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

SCHEMA = "continual-toy-swe-tasks-v3"
PARTITION = "train"
TASK_COUNT = 12
REGRESSION_CASES_PER_TASK = 4
PROPERTY_CASES_PER_TASK = 12

_FAMILY = "t0_continual_v3"
_PRIOR_TRANSFER = "prior-capability-transfer"
_RECOMBINATION = "recombined-transfer"
_UNSUPPORTED = "unsupported-algorithm"
_STAGES = frozenset({_PRIOR_TRANSFER, _RECOMBINATION, _UNSUPPORTED})

_NONZERO_PRODUCT = "nonzero-product"
_INNER_SLICE = "inner-slice"
_SORTED_OVERLAP = "sorted-overlap"
_NORMALIZE_TEXT = "normalize-text"
_EVEN_INNER = "even-inner-slice"
_NORMALIZED_WORD_OVERLAP = "normalized-word-overlap"
_AFFINE_NONZERO_PRODUCT = "affine-nonzero-product"
_INTERIOR_FALLBACK = "interior-index-fallback"
_BALANCED_BRACKETS = "balanced-brackets"
_RUN_LENGTH_ENCODING = "run-length-encoding"
_ROTATE_GRID = "rotate-grid-clockwise"
_MAXIMUM_SUBARRAY = "maximum-contiguous-sum"

_GROUP_STAGE = {
    _NONZERO_PRODUCT: _PRIOR_TRANSFER,
    _INNER_SLICE: _PRIOR_TRANSFER,
    _SORTED_OVERLAP: _PRIOR_TRANSFER,
    _NORMALIZE_TEXT: _PRIOR_TRANSFER,
    _EVEN_INNER: _RECOMBINATION,
    _NORMALIZED_WORD_OVERLAP: _RECOMBINATION,
    _AFFINE_NONZERO_PRODUCT: _RECOMBINATION,
    _INTERIOR_FALLBACK: _RECOMBINATION,
    _BALANCED_BRACKETS: _UNSUPPORTED,
    _RUN_LENGTH_ENCODING: _UNSUPPORTED,
    _ROTATE_GRID: _UNSUPPORTED,
    _MAXIMUM_SUBARRAY: _UNSUPPORTED,
}
_ALL_GROUPS = frozenset(_GROUP_STAGE)

_INT = ProgramType.Int()
_BOOL = ProgramType.Bool()
_STRING = ProgramType.String()
_LIST_INT = ProgramType.List(_INT)
_LIST_STRING = ProgramType.List(_STRING)
_GRID_INT = ProgramType.List(_LIST_INT)


class ContinualToySweTaskError(ValueError):
    """A V3 task, observation, or evaluator record failed authentication."""


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
        '    raise NotImplementedError("public TRAIN v3 starter")\n'
    )


@dataclass(frozen=True, slots=True)
class LearnerVisibleToySweObservationV3:
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
            raise ContinualToySweTaskError("learner-visible path is not canonical")
        if not self.prompt or self.starter_source != _starter_source(self.signature):
            raise ContinualToySweTaskError("learner-visible prompt or starter changed")
        expected = _digest(b"continual-toy-swe-observation-v3", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskError("learner-visible observation failed authentication")


@dataclass(frozen=True, slots=True)
class ToySweEnvironmentPrivateV3:
    """Evaluator-only cases and post-run capability classification."""

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
        if (
            self.capability_group not in _ALL_GROUPS
            or self.stage not in _STAGES
            or _GROUP_STAGE[self.capability_group] != self.stage
        ):
            raise ContinualToySweTaskError("unknown or mismatched evaluator classification")
        expected = _digest(b"continual-toy-swe-environment-v3", self.record())
        if self.environment_sha256 != expected:
            raise ContinualToySweTaskError("evaluator environment failed authentication")


@dataclass(frozen=True, slots=True)
class ContinualToySweTrainTaskV3:
    public: LearnerVisibleToySweObservationV3
    private: ToySweEnvironmentPrivateV3
    task_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_observation_sha256": self.public.observation_sha256,
            "private_environment_sha256": self.private.environment_sha256,
        }

    def authenticate(self) -> None:
        if type(self.public) is not LearnerVisibleToySweObservationV3:
            raise TypeError("task public record must be one exact V3 observation")
        if type(self.private) is not ToySweEnvironmentPrivateV3:
            raise TypeError("task private record must be one exact V3 environment")
        self.public.authenticate()
        self.private.authenticate()
        generated = self.private.generated_task.public
        if (
            self.public.task_commitment_sha256 != generated.task_sha256
            or self.public.signature != generated.signature
            or self.public.prompt != generated.prompt
        ):
            raise ContinualToySweTaskError("public observation is not bound to evaluator task")
        expected = _digest(b"continual-toy-swe-task-v3", self.record())
        if self.task_sha256 != expected:
            raise ContinualToySweTaskError("continual V3 task failed authentication")


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
        "Multiply every nonzero integer in mineral_samples. Return 1 when the list "
        "is empty or contains only zeros. spare_depth and spare_code are ignored.",
        _NONZERO_PRODUCT,
        _PRIOR_TRANSFER,
        307,
        "combine_active_minerals",
        ("mineral_samples", "spare_depth", "spare_code"),
    ),
    _TaskSpec(
        "Return a new list containing survey_marks without its first and last "
        "elements. Lists shorter than three yield an empty list; margin and flag "
        "are unused.",
        _INNER_SLICE,
        _PRIOR_TRANSFER,
        331,
        "extract_survey_interior",
        ("survey_marks", "margin", "flag"),
    ),
    _TaskSpec(
        "Return the sorted distinct integers present in both western_codes and "
        "eastern_codes. unused_stamp does not affect the result.",
        _SORTED_OVERLAP,
        _PRIOR_TRANSFER,
        353,
        "shared_route_codes",
        ("western_codes", "eastern_codes", "unused_stamp"),
    ),
    _TaskSpec(
        "Remove whitespace at both ends of archive_caption and lowercase every "
        "remaining letter. spare_prefix and spare_suffix are irrelevant.",
        _NORMALIZE_TEXT,
        _PRIOR_TRANSFER,
        379,
        "standardize_archive_caption",
        ("archive_caption", "spare_prefix", "spare_suffix"),
    ),
    _TaskSpec(
        "Discard the first and last values of lantern_readings, then return only "
        "the even values from that interior in original order. spare_a and spare_b "
        "are ignored.",
        _EVEN_INNER,
        _RECOMBINATION,
        401,
        "select_even_lantern_interior",
        ("lantern_readings", "spare_a", "spare_b"),
    ),
    _TaskSpec(
        "Lowercase and split each text on arbitrary whitespace, then return the "
        "sorted distinct words occurring in both northern_text and southern_text.",
        _NORMALIZED_WORD_OVERLAP,
        _RECOMBINATION,
        419,
        "shared_normalized_words",
        ("northern_text", "southern_text"),
    ),
    _TaskSpec(
        "Multiply the nonzero values in charge_levels, multiply that product by "
        "scale, then add offset. An empty or all-zero list has product 1.",
        _AFFINE_NONZERO_PRODUCT,
        _RECOMBINATION,
        443,
        "scale_active_charge_product",
        ("charge_levels", "scale", "offset"),
    ),
    _TaskSpec(
        "Remove the first and last elements of signal_values. Return the element at "
        "zero-based position within that interior when it exists; otherwise return "
        "fallback_value. Negative positions are out of range.",
        _INTERIOR_FALLBACK,
        _RECOMBINATION,
        467,
        "pick_interior_signal",
        ("signal_values", "position", "fallback_value"),
    ),
    _TaskSpec(
        "Return whether every (), [], and {} delimiter in notation is correctly "
        "nested and closed. Ignore all non-delimiter characters.",
        _BALANCED_BRACKETS,
        _UNSUPPORTED,
        487,
        "delimiters_are_balanced",
        ("notation",),
    ),
    _TaskSpec(
        "Run-length encode token_text, which contains only ASCII letters, digits, "
        "and underscores. Represent each consecutive run as its character followed "
        "by its decimal count, join runs with commas, and return an empty string for "
        "empty input.",
        _RUN_LENGTH_ENCODING,
        _UNSUPPORTED,
        509,
        "encode_token_runs",
        ("token_text",),
    ),
    _TaskSpec(
        "Rotate rectangular integer grid ninety degrees clockwise. Return an empty "
        "list for an empty grid.",
        _ROTATE_GRID,
        _UNSUPPORTED,
        541,
        "rotate_signal_grid",
        ("grid",),
    ),
    _TaskSpec(
        "Return the largest sum of any nonempty contiguous slice of values. Return "
        "zero when values is empty.",
        _MAXIMUM_SUBARRAY,
        _UNSUPPORTED,
        563,
        "maximum_contiguous_signal",
        ("values",),
    ),
)


def _signature(spec: _TaskSpec) -> TypedSignature:
    group = spec.capability_group
    if group in {_NONZERO_PRODUCT, _AFFINE_NONZERO_PRODUCT}:
        parameter_types = (_LIST_INT, _INT, _INT)
        result = _INT
    elif group in {_INNER_SLICE, _EVEN_INNER}:
        parameter_types = (_LIST_INT, _INT, _INT)
        result = _LIST_INT
    elif group == _SORTED_OVERLAP:
        parameter_types = (_LIST_INT, _LIST_INT, _INT)
        result = _LIST_INT
    elif group == _NORMALIZE_TEXT:
        parameter_types = (_STRING, _STRING, _STRING)
        result = _STRING
    elif group == _NORMALIZED_WORD_OVERLAP:
        parameter_types = (_STRING, _STRING)
        result = _LIST_STRING
    elif group == _INTERIOR_FALLBACK:
        parameter_types = (_LIST_INT, _INT, _INT)
        result = _INT
    elif group == _BALANCED_BRACKETS:
        parameter_types = (_STRING,)
        result = _BOOL
    elif group == _RUN_LENGTH_ENCODING:
        parameter_types = (_STRING,)
        result = _STRING
    elif group == _ROTATE_GRID:
        parameter_types = (_GRID_INT,)
        result = _GRID_INT
    elif group == _MAXIMUM_SUBARRAY:
        parameter_types = (_LIST_INT,)
        result = _INT
    else:
        raise ValueError("unknown V3 capability group")
    if len(spec.parameter_names) != len(parameter_types):
        raise ValueError("V3 task parameter names and types disagree")
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


def _integer(seed: int, index: int, channel: int = 0) -> int:
    return ((seed * 29 + index * 37 + channel * 43) % 127) - 63


def _integer_list(seed: int, index: int, channel: int = 0) -> list[int]:
    length = (seed + index * 7 + channel * 5) % 10
    return [
        _integer(seed, index + position, channel + position)
        for position in range(length)
    ]


def _mixed_text(seed: int, index: int) -> str:
    alphabet = " aAbBcCxyZ019_-\t"
    length = (seed + index * 11) % 24
    body = "".join(
        alphabet[(seed + index * 5 + position * 7) % len(alphabet)]
        for position in range(length)
    )
    return (" " * (index % 3)) + body + ("\t" * ((seed + index) % 2))


def _word_text(seed: int, index: int, channel: int) -> str:
    words = ("amber", "brook", "cedar", "delta", "fern", "granite", "harbor")
    separators = (" ", "  ", "\t")
    count = (seed + index * 3 + channel) % 8
    selected = [
        words[(seed + index + channel * 2 + position * 3) % len(words)]
        for position in range(count)
    ]
    if (index + channel) % 2:
        selected = [
            word.upper() if position % 2 else word
            for position, word in enumerate(selected)
        ]
    separator = separators[(seed + index + channel) % len(separators)]
    return (" " if channel else "") + separator.join(selected) + ("  " if index % 3 == 0 else "")


def _run_text(seed: int, index: int) -> str:
    alphabet = "abC019_"
    length = (seed + index * 5) % 20
    run_width = 1 + ((seed + index) % 4)
    return "".join(
        alphabet[(seed + position // run_width + index) % len(alphabet)]
        for position in range(length)
    )


def _grid(seed: int, index: int) -> list[list[int]]:
    rows = (seed + index) % 5
    if rows == 0:
        return []
    columns = 1 + ((seed + index * 3) % 4)
    return [
        [_integer(seed, row * columns + column, index) for column in range(columns)]
        for row in range(rows)
    ]


def _nonzero_product(values: list[int], unused_left: int, unused_right: int) -> int:
    del unused_left, unused_right
    result = 1
    for value in values:
        if value != 0:
            result *= value
    return result


def _inner_slice(values: list[int], unused_left: int, unused_right: int) -> list[int]:
    del unused_left, unused_right
    return values[1:-1]


def _sorted_overlap(left: list[int], right: list[int], unused: int) -> list[int]:
    del unused
    return sorted(set(left) & set(right))


def _normalize_text(text: str, unused_left: str, unused_right: str) -> str:
    del unused_left, unused_right
    return text.strip().lower()


def _even_inner(values: list[int], unused_left: int, unused_right: int) -> list[int]:
    del unused_left, unused_right
    return [value for value in values[1:-1] if value % 2 == 0]


def _normalized_word_overlap(left: str, right: str) -> list[str]:
    return sorted(set(left.lower().split()) & set(right.lower().split()))


def _affine_nonzero_product(values: list[int], scale: int, offset: int) -> int:
    product = _nonzero_product(values, 0, 0)
    return product * scale + offset


def _interior_fallback(values: list[int], position: int, fallback: int) -> int:
    interior = values[1:-1]
    if 0 <= position < len(interior):
        return interior[position]
    return fallback


def _balanced_brackets(text: str) -> bool:
    openings = "([{"
    closing = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in text:
        if character in openings:
            stack.append(character)
        elif character in closing and (
            not stack or stack.pop() != closing[character]
        ):
            return False
    return not stack


def _run_length_encoding(text: str) -> str:
    if not text:
        return ""
    chunks: list[str] = []
    start = 0
    for index in range(1, len(text) + 1):
        if index == len(text) or text[index] != text[start]:
            chunks.append(f"{text[start]}{index - start}")
            start = index
    return ",".join(chunks)


def _rotate_grid(grid: list[list[int]]) -> list[list[int]]:
    if not grid:
        return []
    return [list(row) for row in zip(*reversed(grid), strict=True)]


def _maximum_subarray(values: list[int]) -> int:
    if not values:
        return 0
    current = best = values[0]
    for value in values[1:]:
        current = max(value, current + value)
        best = max(best, current)
    return best


def _cases(
    group: str,
    seed: int,
) -> tuple[
    Callable[..., JsonValue],
    tuple[tuple[JsonValue, ...], ...],
    tuple[tuple[tuple[JsonValue, ...], str], ...],
]:
    label = "deterministic-v3-property"
    if group == _NONZERO_PRODUCT:
        oracle: Callable[..., JsonValue] = _nonzero_product
        regressions: tuple[tuple[JsonValue, ...], ...] = (
            ([], 1, 2),
            ([0, 0], 3, 4),
            ([2, 0, -3], 5, 6),
            ([-1, -2, -3], seed, 0),
        )
        properties = tuple(
            ((_integer_list(seed, index), index, -index), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _INNER_SLICE:
        oracle = _inner_slice
        regressions = (([], 0, 0), ([1], 1, 2), ([1, 2], 3, 4), ([1, 2, 3, 4], 5, 6))
        properties = tuple(
            ((_integer_list(seed, index), index, seed), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _SORTED_OVERLAP:
        oracle = _sorted_overlap
        regressions = (
            ([], [], 0),
            ([3, 1, 3], [2, 4], 1),
            ([5, 2, 5, -1], [5, -1, 9, 5], 2),
            ([-3, -2], [-2, -3], seed),
        )
        properties = tuple(
            ((_integer_list(seed, index, 0), _integer_list(seed, index, 1), index), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _NORMALIZE_TEXT:
        oracle = _normalize_text
        regressions = (
            ("", "left", "right"),
            ("  MiXeD Archive  ", "x", "y"),
            ("\tALREADY LOWER\n", "a", "b"),
            (f"  Record {seed}  ", "unused", "unused"),
        )
        properties = tuple(
            ((_mixed_text(seed, index), f"left-{index}", f"right-{seed}"), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _EVEN_INNER:
        oracle = _even_inner
        regressions = (
            ([], 0, 0),
            ([2, 4], 1, 2),
            ([9, 2, 3, 4, 7], 3, 4),
            ([-8, -6, -5, -4, -2], 5, 6),
        )
        properties = tuple(
            ((_integer_list(seed, index), index, -index), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _NORMALIZED_WORD_OVERLAP:
        oracle = _normalized_word_overlap
        regressions = (
            ("", ""),
            ("Amber cedar", "CEDAR brook"),
            (" one\tTWO one ", "two three"),
            ("granite harbor", "delta fern"),
        )
        properties = tuple(
            ((_word_text(seed, index, 0), _word_text(seed, index, 1)), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _AFFINE_NONZERO_PRODUCT:
        oracle = _affine_nonzero_product
        regressions = (
            ([], 3, 2),
            ([0, 0], -2, 5),
            ([2, 0, 4], 3, -1),
            ([-3, -2], -1, seed % 7),
        )
        properties = tuple(
            (
                (
                    _integer_list(seed, index),
                    _integer(seed, index, 1),
                    _integer(seed, index, 2),
                ),
                label,
            )
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _INTERIOR_FALLBACK:
        oracle = _interior_fallback
        regressions = (
            ([], 0, 9),
            ([1, 2], 0, 8),
            ([4, 5, 6, 7], 1, 0),
            ([4, 5, 6, 7], -1, -9),
        )
        properties = tuple(
            ((_integer_list(seed, index), (index % 7) - 2, _integer(seed, index, 2)), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _BALANCED_BRACKETS:
        oracle = _balanced_brackets
        regressions = (("",), ("([{}])",), ("([)]",), ("text {a[b](c)}",))
        bracket_texts = (
            "()",
            "[]{}",
            "(",
            "]",
            "{[()]}",
            "{[(])}",
            "no delimiters",
            "a(b[c]{d})e",
            "(()())",
            "([{}]))",
            "{x[y(z)]}",
            "}{",
        )
        properties = tuple(((text,), label) for text in bracket_texts)
    elif group == _RUN_LENGTH_ENCODING:
        oracle = _run_length_encoding
        regressions = (("",), ("a",), ("aaabb_",), ("1111000AA",))
        properties = tuple(
            ((_run_text(seed, index),), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _ROTATE_GRID:
        oracle = _rotate_grid
        regressions = (
            ([],),
            ([[1]],),
            ([[1, 2, 3], [4, 5, 6]],),
            ([[1], [2], [3]],),
        )
        properties = tuple(
            ((_grid(seed, index),), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    elif group == _MAXIMUM_SUBARRAY:
        oracle = _maximum_subarray
        regressions = (([],), ([-5],), ([-4, -2, -7],), ([3, -2, 5, -8, 4],))
        properties = tuple(
            ((_integer_list(seed, index),), label)
            for index in range(PROPERTY_CASES_PER_TASK)
        )
    else:
        raise ValueError("unknown V3 capability group")
    return oracle, regressions, properties


def _build_task(spec: _TaskSpec, ordinal: int) -> ContinualToySweTrainTaskV3:
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
    public_placeholder = LearnerVisibleToySweObservationV3(
        task_commitment_sha256=generated.public.task_sha256,
        relative_path=f"workbench/v3_pressure_{ordinal:02d}.py",
        signature=signature,
        prompt=generated.public.prompt,
        starter_source=_starter_source(signature),
        observation_sha256="",
    )
    public = LearnerVisibleToySweObservationV3(
        task_commitment_sha256=public_placeholder.task_commitment_sha256,
        relative_path=public_placeholder.relative_path,
        signature=public_placeholder.signature,
        prompt=public_placeholder.prompt,
        starter_source=public_placeholder.starter_source,
        observation_sha256=_digest(
            b"continual-toy-swe-observation-v3",
            public_placeholder.record(),
        ),
    )
    private_placeholder = ToySweEnvironmentPrivateV3(
        generated_task=generated,
        capability_group=spec.capability_group,
        stage=spec.stage,
        environment_sha256="",
    )
    private = ToySweEnvironmentPrivateV3(
        generated_task=generated,
        capability_group=private_placeholder.capability_group,
        stage=private_placeholder.stage,
        environment_sha256=_digest(
            b"continual-toy-swe-environment-v3",
            private_placeholder.record(),
        ),
    )
    task_placeholder = ContinualToySweTrainTaskV3(
        public=public,
        private=private,
        task_sha256="",
    )
    task = ContinualToySweTrainTaskV3(
        public=public,
        private=private,
        task_sha256=_digest(
            b"continual-toy-swe-task-v3",
            task_placeholder.record(),
        ),
    )
    task.authenticate()
    return task


if len(_SPECS) != TASK_COUNT:
    raise ValueError("V3 pressure stream must contain exactly twelve tasks")

_TRAIN_STREAM = tuple(_build_task(spec, ordinal) for ordinal, spec in enumerate(_SPECS))

_EXPECTED_STAGE_COUNTS = {
    _PRIOR_TRANSFER: 4,
    _RECOMBINATION: 4,
    _UNSUPPORTED: 4,
}
_EXPECTED_GROUP_COUNTS = {group: 1 for group in _ALL_GROUPS}


def _authenticate_stream(tasks: tuple[ContinualToySweTrainTaskV3, ...]) -> None:
    if len(tasks) != TASK_COUNT:
        raise ContinualToySweTaskError("V3 TRAIN stream count changed")
    for task in tasks:
        task.authenticate()
    if len({task.task_sha256 for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskError("V3 TRAIN task commitments are not unique")
    if len({task.public.signature.entry_point for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskError("V3 TRAIN entry points are not unique")
    if Counter(task.private.stage for task in tasks) != _EXPECTED_STAGE_COUNTS:
        raise ContinualToySweTaskError("V3 capability stage counts changed")
    if Counter(task.private.capability_group for task in tasks) != _EXPECTED_GROUP_COUNTS:
        raise ContinualToySweTaskError("V3 capability group counts changed")


def continual_toy_swe_train_stream_v3() -> tuple[ContinualToySweTrainTaskV3, ...]:
    """Return the immutable ordered V3 public-TRAIN stream."""

    _authenticate_stream(_TRAIN_STREAM)
    return _TRAIN_STREAM


def learner_visible_observation_v3(
    task: ContinualToySweTrainTaskV3,
) -> LearnerVisibleToySweObservationV3:
    """Project exactly the target-teacher-free learner record."""

    if type(task) is not ContinualToySweTrainTaskV3:
        raise TypeError("learner observation requires one exact V3 continual task")
    task.authenticate()
    return task.public


def train_stream_sha256_v3(
    tasks: tuple[ContinualToySweTrainTaskV3, ...] | None = None,
) -> str:
    selected = _TRAIN_STREAM if tasks is None else tasks
    if not isinstance(selected, tuple) or not all(
        type(task) is ContinualToySweTrainTaskV3 for task in selected
    ):
        raise TypeError("V3 TRAIN stream must be a tuple of exact continual tasks")
    _authenticate_stream(selected)
    return _digest(
        b"continual-toy-swe-stream-v3",
        {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_sha256s": tuple(task.task_sha256 for task in selected),
        },
    )


TRAIN_STREAM_SHA256 = train_stream_sha256_v3()


__all__ = [
    "PARTITION",
    "PROPERTY_CASES_PER_TASK",
    "REGRESSION_CASES_PER_TASK",
    "SCHEMA",
    "TASK_COUNT",
    "TRAIN_STREAM_SHA256",
    "ContinualToySweTaskError",
    "ContinualToySweTrainTaskV3",
    "LearnerVisibleToySweObservationV3",
    "ToySweEnvironmentPrivateV3",
    "continual_toy_swe_train_stream_v3",
    "learner_visible_observation_v3",
    "train_stream_sha256_v3",
]
