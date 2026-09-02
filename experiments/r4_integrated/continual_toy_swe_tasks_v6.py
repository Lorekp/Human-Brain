"""Fresh public-TRAIN-only tasks for the integrated continual coding V6 cohort.

V6 is authored after the spent V5 session localized hierarchical multi-fragment
composition and consequence-conditioned structural revision as the next missing
computation.  The stream is frozen before any recurrent-assembly model, gate,
runner, or integrated V6 output exists.

Learner-visible records contain only a task commitment, repository path, typed
signature, natural-language request, and inert starter source.  Oracles, cases,
capability/stage labels, retry budgets, feedback pressure, retention anchors,
and expected terminal actions remain evaluator-private.  No successful source,
candidate patch, target span, answer-bearing diagnostic, or task ontology is
stored in a public record.
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

SCHEMA = "continual-toy-swe-tasks-v6"
PARTITION = "train"
TASK_COUNT = 16
MAX_ATTEMPTS = 3
EXECUTION_REGRESSION_CASES = 5
EXECUTION_PROPERTY_CASES = 16

_FAMILY = "F6"
_EXECUTE = "execute"
_STOP = "stop"

_TRANSFER = "renamed-transfer"
_RECOMBINATION = "nested-multifragment-recombination"
_FEEDBACK = "execution-feedback-revision"
_RETENTION = "delayed-retention-recheck"
_INSUFFICIENT = "insufficient-evidence"
_STAGES = (_TRANSFER, _RECOMBINATION, _FEEDBACK, _RETENTION, _INSUFFICIENT)

_EVEN_TRANSFER = "even-filter-transfer"
_NORMALIZE_TRANSFER = "text-normalization-transfer"
_PRODUCT_TRANSFER = "nonzero-product-transfer"
_GAP_TRANSFER = "absolute-gap-transfer"
_INTERIOR_EVEN_PRODUCT = "interior-even-product-recombination"
_OVERLAP_INTERIOR = "overlap-interior-recombination"
_SHARED_EVEN_PRODUCT = "shared-even-product-recombination"
_DUAL_AFFINE_GAP = "dual-affine-gap-recombination"
_EVEN_INTERIOR_PRODUCT = "even-interior-product-revision"
_INTERIOR_OVERLAP = "interior-overlap-revision"
_PRODUCT_THEN_AFFINE = "product-then-affine-revision"
_GAP_THEN_AFFINE = "gap-then-affine-revision"
_NORMALIZE_RETENTION = "normalization-retention"
_SHARED_EVEN_PRODUCT_RETENTION = "shared-even-product-retention"
_AMBIGUOUS_IDENTIFIER_PROFILE = "ambiguous-identifier-profile-stop"
_AMBIGUOUS_CHANNEL_FUSION = "ambiguous-channel-fusion-stop"

_GROUP_STAGE = {
    _EVEN_TRANSFER: _TRANSFER,
    _NORMALIZE_TRANSFER: _TRANSFER,
    _PRODUCT_TRANSFER: _TRANSFER,
    _GAP_TRANSFER: _TRANSFER,
    _INTERIOR_EVEN_PRODUCT: _RECOMBINATION,
    _OVERLAP_INTERIOR: _RECOMBINATION,
    _SHARED_EVEN_PRODUCT: _RECOMBINATION,
    _DUAL_AFFINE_GAP: _RECOMBINATION,
    _EVEN_INTERIOR_PRODUCT: _FEEDBACK,
    _INTERIOR_OVERLAP: _FEEDBACK,
    _PRODUCT_THEN_AFFINE: _FEEDBACK,
    _GAP_THEN_AFFINE: _FEEDBACK,
    _NORMALIZE_RETENTION: _RETENTION,
    _SHARED_EVEN_PRODUCT_RETENTION: _RETENTION,
    _AMBIGUOUS_IDENTIFIER_PROFILE: _INSUFFICIENT,
    _AMBIGUOUS_CHANNEL_FUSION: _INSUFFICIENT,
}
_ALL_GROUPS = frozenset(_GROUP_STAGE)

_INT = ProgramType.Int()
_STRING = ProgramType.String()
_LIST_INT = ProgramType.List(_INT)


class ContinualToySweTaskV6Error(ValueError):
    """A V6 task, observation, environment, or stream is unauthenticated."""


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
class LearnerVisibleToySweObservationV6:
    """The complete V6 record allowed into learner input."""

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
            raise ContinualToySweTaskV6Error("public task commitment is not SHA-256")
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
            raise ContinualToySweTaskV6Error("learner-visible path is not canonical")
        if not self.prompt or not self.starter_source:
            raise ContinualToySweTaskV6Error("learner-visible prompt or starter is empty")
        expected = _digest(b"continual-toy-swe-observation-v6", self.record())
        if self.observation_sha256 != expected:
            raise ContinualToySweTaskV6Error("learner-visible observation changed")


@dataclass(frozen=True, slots=True)
class ToySweEnvironmentPrivateV6:
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
            raise ContinualToySweTaskV6Error("evaluator task commitments changed")
        if (
            self.capability_group not in _ALL_GROUPS
            or self.stage not in _STAGES
            or _GROUP_STAGE[self.capability_group] != self.stage
            or self.expected_action not in {_EXECUTE, _STOP}
            or type(self.attempt_budget) is not int
            or not 1 <= self.attempt_budget <= MAX_ATTEMPTS
            or not self.feedback_pressure
        ):
            raise ContinualToySweTaskV6Error("evaluator policy classification changed")
        expected_counts = (
            (0, 0)
            if self.expected_action == _STOP
            else (EXECUTION_REGRESSION_CASES, EXECUTION_PROPERTY_CASES)
        )
        actual_counts = (len(private.regression_cases), len(private.property_cases))
        if actual_counts != expected_counts:
            raise ContinualToySweTaskV6Error("evaluator hidden-case counts changed")
        if self.expected_action == _STOP:
            if self.stage != _INSUFFICIENT or self.attempt_budget != 1:
                raise ContinualToySweTaskV6Error(
                    "STOP policy must be immediate and evidence-limited"
                )
        elif self.stage == _INSUFFICIENT:
            raise ContinualToySweTaskV6Error(
                "insufficient-evidence task became executable"
            )
        if self.stage in {_RECOMBINATION, _FEEDBACK} and self.attempt_budget != 3:
            raise ContinualToySweTaskV6Error("revision pressure lost its retry budget")
        if self.stage == _RETENTION:
            anchor = self.retention_anchor_ordinal
            if type(anchor) is not int or not 0 <= anchor < public.ordinal:
                raise ContinualToySweTaskV6Error("retention anchor is not earlier")
        elif self.retention_anchor_ordinal is not None:
            raise ContinualToySweTaskV6Error("non-retention task gained an anchor")
        expected = _digest(b"continual-toy-swe-environment-v6", self.record())
        if self.environment_sha256 != expected:
            raise ContinualToySweTaskV6Error("evaluator environment changed")


@dataclass(frozen=True, slots=True)
class ContinualToySweTrainTaskV6:
    public: LearnerVisibleToySweObservationV6
    private: ToySweEnvironmentPrivateV6
    task_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "public_observation_sha256": self.public.observation_sha256,
            "private_environment_sha256": self.private.environment_sha256,
        }

    def authenticate(self) -> None:
        if type(self.public) is not LearnerVisibleToySweObservationV6:
            raise TypeError("task public record must be one exact V6 observation")
        if type(self.private) is not ToySweEnvironmentPrivateV6:
            raise TypeError("task private record must be one exact V6 environment")
        self.public.authenticate()
        self.private.authenticate()
        generated = self.private.generated_task.public
        if not 0 <= generated.ordinal < len(_SPECS):
            raise ContinualToySweTaskV6Error("generated ordinal is outside V6 stream")
        spec = _SPECS[generated.ordinal]
        if (
            self.public.task_commitment_sha256 != generated.task_sha256
            or self.public.signature != generated.signature
            or self.public.prompt != generated.prompt
            or self.public.starter_source != _starter_source(spec, generated.signature)
        ):
            raise ContinualToySweTaskV6Error(
                "public observation is not bound to evaluator"
            )
        expected = _digest(b"continual-toy-swe-task-v6", self.record())
        if self.task_sha256 != expected:
            raise ContinualToySweTaskV6Error("continual V6 task changed")


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
        "Return a new list containing the even integers from measurements in "
        "their original order.",
        _EVEN_TRANSFER,
        _TRANSFER,
        701,
        "select_even_measurements",
        (("measurements", _LIST_INT),),
        _LIST_INT,
        ("hold_measurements",),
        (),
        "selected_measurements",
        _EXECUTE,
        2,
        "renamed-prior-transfer",
    ),
    _TaskSpec(
        "Return heading with surrounding whitespace removed and all letters "
        "lowercased; context_marker is unused.",
        _NORMALIZE_TRANSFER,
        _TRANSFER,
        709,
        "canonicalize_heading",
        (("heading", _STRING), ("context_marker", _STRING)),
        _STRING,
        (),
        ("relay_heading",),
        "canonical_heading",
        _EXECUTE,
        2,
        "renamed-prior-transfer",
    ),
    _TaskSpec(
        "Multiply the nonzero integers in factors, returning 1 when none are "
        "present; channel, checksum, and note are unused.",
        _PRODUCT_TRANSFER,
        _TRANSFER,
        719,
        "multiply_live_factors",
        (
            ("factors", _LIST_INT),
            ("channel", _INT),
            ("checksum", _INT),
            ("note", _INT),
        ),
        _INT,
        ("hold_factors",),
        ("finish_product",),
        "live_product",
        _EXECUTE,
        2,
        "renamed-prior-transfer",
    ),
    _TaskSpec(
        "Return the non-negative distance between left_mark and right_mark; "
        "phase, version, and spare are unused.",
        _GAP_TRANSFER,
        _TRANSFER,
        727,
        "beacon_interval",
        (
            ("left_mark", _INT),
            ("right_mark", _INT),
            ("phase", _INT),
            ("version", _INT),
            ("spare", _INT),
        ),
        _INT,
        (),
        ("relay_interval",),
        "beacon_gap",
        _EXECUTE,
        2,
        "renamed-prior-transfer",
    ),
    _TaskSpec(
        "First remove the first and last elements of values. From that interior, "
        "multiply the even nonzero integers, returning 1 when none qualify.",
        _INTERIOR_EVEN_PRODUCT,
        _RECOMBINATION,
        733,
        "multiply_even_interior",
        (("values", _LIST_INT),),
        _INT,
        ("hold_interior",),
        (),
        "interior_product",
        _EXECUTE,
        3,
        "nested-multifragment-composition",
    ),
    _TaskSpec(
        "Find the sorted distinct integers shared by left and right, then remove "
        "the first and last values of that sorted overlap.",
        _OVERLAP_INTERIOR,
        _RECOMBINATION,
        739,
        "interior_shared_values",
        (("left", _LIST_INT), ("right", _LIST_INT)),
        _LIST_INT,
        (),
        ("finish_shared_values",),
        "shared_interior",
        _EXECUTE,
        3,
        "nested-multifragment-composition",
    ),
    _TaskSpec(
        "Find the distinct even nonzero integers shared by left and right, then "
        "multiply them, returning 1 when none qualify; both tags are unused.",
        _SHARED_EVEN_PRODUCT,
        _RECOMBINATION,
        743,
        "multiply_shared_even_values",
        (
            ("left", _LIST_INT),
            ("right", _LIST_INT),
            ("left_tag", _INT),
            ("right_tag", _INT),
        ),
        _INT,
        ("hold_shared_even",),
        ("relay_shared_even",),
        "shared_even_product",
        _EXECUTE,
        3,
        "nested-multifragment-composition",
    ),
    _TaskSpec(
        "Calibrate left as left * gain + left_bias and right as right * gain + "
        "right_bias, then return the non-negative distance between the two "
        "calibrated values.",
        _DUAL_AFFINE_GAP,
        _RECOMBINATION,
        751,
        "calibrated_pair_distance",
        (
            ("left", _INT),
            ("right", _INT),
            ("gain", _INT),
            ("left_bias", _INT),
            ("right_bias", _INT),
        ),
        _INT,
        (),
        ("finish_pair_distance",),
        "calibrated_distance",
        _EXECUTE,
        3,
        "nested-multifragment-composition",
    ),
    _TaskSpec(
        "First keep the even integers from values in their original order, then "
        "remove the first and last qualifying even values. Multiply the remaining "
        "nonzero integers, returning 1 when none remain.",
        _EVEN_INTERIOR_PRODUCT,
        _FEEDBACK,
        757,
        "multiply_interior_even_rank",
        (("values", _LIST_INT),),
        _INT,
        ("hold_even_rank",),
        (),
        "ranked_even_product",
        _EXECUTE,
        3,
        "execution-feedback-order-contrast",
    ),
    _TaskSpec(
        "Remove the first and last elements from left and from right separately, "
        "then return the sorted distinct integers shared by those two interiors.",
        _INTERIOR_OVERLAP,
        _FEEDBACK,
        761,
        "shared_values_of_interiors",
        (("left", _LIST_INT), ("right", _LIST_INT)),
        _LIST_INT,
        (),
        ("finish_interior_overlap",),
        "interior_overlap",
        _EXECUTE,
        3,
        "execution-feedback-order-contrast",
    ),
    _TaskSpec(
        "Multiply the nonzero integers in values first, then multiply that product "
        "by gain and add bias; spare is unused.",
        _PRODUCT_THEN_AFFINE,
        _FEEDBACK,
        769,
        "calibrate_factor_product",
        (
            ("values", _LIST_INT),
            ("gain", _INT),
            ("bias", _INT),
            ("spare", _INT),
        ),
        _INT,
        ("hold_factor_product",),
        ("relay_calibration",),
        "calibrated_product",
        _EXECUTE,
        3,
        "execution-feedback-order-contrast",
    ),
    _TaskSpec(
        "Compute the non-negative distance between left and right first, then "
        "multiply that distance by gain and add bias; spare is unused.",
        _GAP_THEN_AFFINE,
        _FEEDBACK,
        773,
        "calibrate_raw_distance",
        (
            ("left", _INT),
            ("right", _INT),
            ("gain", _INT),
            ("bias", _INT),
            ("spare", _INT),
        ),
        _INT,
        (),
        ("finish_distance_calibration",),
        "calibrated_gap",
        _EXECUTE,
        3,
        "execution-feedback-order-contrast",
    ),
    _TaskSpec(
        "Return title with surrounding whitespace removed and all letters "
        "lowercased; revision_tag is unused.",
        _NORMALIZE_RETENTION,
        _RETENTION,
        787,
        "normalize_retained_title",
        (("title", _STRING), ("revision_tag", _STRING)),
        _STRING,
        ("hold_retained_title",),
        (),
        "retained_title",
        _EXECUTE,
        2,
        "delayed-retention",
        1,
    ),
    _TaskSpec(
        "Find the distinct even nonzero integers shared by primary and secondary, "
        "then multiply them, returning 1 when none qualify; version and checksum "
        "are unused.",
        _SHARED_EVEN_PRODUCT_RETENTION,
        _RETENTION,
        797,
        "retained_shared_even_product",
        (
            ("primary", _LIST_INT),
            ("secondary", _LIST_INT),
            ("version", _INT),
            ("checksum", _INT),
        ),
        _INT,
        (),
        ("finish_retained_product",),
        "retained_product",
        _EXECUTE,
        2,
        "delayed-retention",
        6,
    ),
    _TaskSpec(
        "Rewrite identifier according to the active external identifier-style "
        "profile. The profile is absent from the request and inputs; do not guess.",
        _AMBIGUOUS_IDENTIFIER_PROFILE,
        _INSUFFICIENT,
        809,
        "apply_identifier_profile",
        (("identifier", _STRING),),
        _STRING,
        ("hold_identifier",),
        (),
        "profiled_identifier",
        _STOP,
        1,
        "missing-policy",
    ),
    _TaskSpec(
        "Combine north, south, east, west, and center according to an externally "
        "selected channel-fusion rule. The rule is absent from the request and "
        "inputs; do not guess.",
        _AMBIGUOUS_CHANNEL_FUSION,
        _INSUFFICIENT,
        811,
        "fuse_directional_channels",
        (
            ("north", _INT),
            ("south", _INT),
            ("east", _INT),
            ("west", _INT),
            ("center", _INT),
        ),
        _INT,
        (),
        ("relay_channel_fusion",),
        "fused_channel",
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
        '    raise NotImplementedError("public TRAIN v6 starter")\n'
    )
    sections.extend(_helper_source(name) for name in spec.after_helpers)
    return "\n".join(sections)


def _integer(seed: int, index: int, channel: int = 0) -> int:
    return ((seed * 29 + index * 17 + channel * 11) % 19) - 9


def _integer_list(seed: int, index: int, channel: int = 0) -> list[int]:
    length = (seed + index * 3 + channel * 5) % 9
    return [
        _integer(seed + channel * 13, index * 7 + item, item + channel)
        for item in range(length)
    ]


def _mixed_text(seed: int, index: int) -> str:
    words = (" Aurora", "BASALT ", " Cinder ", "DELTA", " ember ", "FJORD ")
    count = 1 + (seed + index) % 3
    start = (seed * 7 + index * 3) % len(words)
    return "".join(words[(start + offset) % len(words)] for offset in range(count))


def _even_transfer(values: list[int]) -> list[int]:
    return [value for value in values if value % 2 == 0]


def _normalize_transfer(text: str, context_marker: str) -> str:
    del context_marker
    return text.strip().lower()


def _nonzero_product(values: list[int]) -> int:
    result = 1
    for value in values:
        if value != 0:
            result *= value
    return result


def _product_transfer(
    values: list[int],
    channel: int,
    checksum: int,
    note: int,
) -> int:
    del channel, checksum, note
    return _nonzero_product(values)


def _gap_transfer(
    left: int,
    right: int,
    phase: int,
    version: int,
    spare: int,
) -> int:
    del phase, version, spare
    return abs(left - right)


def _interior_even_product(values: list[int]) -> int:
    return _nonzero_product(
        [value for value in values[1:-1] if value % 2 == 0]
    )


def _overlap_interior(left: list[int], right: list[int]) -> list[int]:
    return sorted(set(left) & set(right))[1:-1]


def _shared_even_product(
    left: list[int],
    right: list[int],
    left_tag: int,
    right_tag: int,
) -> int:
    del left_tag, right_tag
    shared_even = sorted(
        value
        for value in set(left) & set(right)
        if value != 0 and value % 2 == 0
    )
    return _nonzero_product(shared_even)


def _dual_affine_gap(
    left: int,
    right: int,
    gain: int,
    left_bias: int,
    right_bias: int,
) -> int:
    calibrated_left = left * gain + left_bias
    calibrated_right = right * gain + right_bias
    return abs(calibrated_left - calibrated_right)


def _even_interior_product(values: list[int]) -> int:
    even_values = [value for value in values if value % 2 == 0]
    return _nonzero_product(even_values[1:-1])


def _interior_overlap(left: list[int], right: list[int]) -> list[int]:
    return sorted(set(left[1:-1]) & set(right[1:-1]))


def _product_then_affine(
    values: list[int],
    gain: int,
    bias: int,
    spare: int,
) -> int:
    del spare
    return _nonzero_product(values) * gain + bias


def _gap_then_affine(
    left: int,
    right: int,
    gain: int,
    bias: int,
    spare: int,
) -> int:
    del spare
    return abs(left - right) * gain + bias


def _normalize_retention(text: str, revision_tag: str) -> str:
    del revision_tag
    return text.strip().lower()


def _shared_even_product_retention(
    primary: list[int],
    secondary: list[int],
    version: int,
    checksum: int,
) -> int:
    return _shared_even_product(primary, secondary, version, checksum)


def _unreachable_stop_oracle(*args: JsonValue) -> JsonValue:
    del args
    raise RuntimeError("STOP task has no executable target oracle")


def _generated_properties(
    group: str,
    seed: int,
) -> tuple[tuple[tuple[JsonValue, ...], str], ...]:
    rows: list[tuple[tuple[JsonValue, ...], str]] = []
    for index in range(EXECUTION_PROPERTY_CASES):
        if group == _EVEN_TRANSFER:
            args: tuple[JsonValue, ...] = (_integer_list(seed, index),)
        elif group in {_NORMALIZE_TRANSFER, _NORMALIZE_RETENTION}:
            args = (_mixed_text(seed, index), f"marker-{seed}-{index}")
        elif group == _PRODUCT_TRANSFER:
            args = (
                _integer_list(seed, index),
                _integer(seed, index, 1),
                _integer(seed, index, 2),
                _integer(seed, index, 3),
            )
        elif group == _GAP_TRANSFER:
            args = tuple(
                _integer(seed, index, channel) for channel in range(5)
            )
        elif group in {_INTERIOR_EVEN_PRODUCT, _EVEN_INTERIOR_PRODUCT}:
            args = (_integer_list(seed, index),)
        elif group in {_OVERLAP_INTERIOR, _INTERIOR_OVERLAP}:
            args = (
                _integer_list(seed, index),
                _integer_list(seed, index, 1),
            )
        elif group in {
            _SHARED_EVEN_PRODUCT,
            _SHARED_EVEN_PRODUCT_RETENTION,
        }:
            args = (
                _integer_list(seed, index),
                _integer_list(seed, index, 1),
                _integer(seed, index, 2),
                _integer(seed, index, 3),
            )
        elif group == _DUAL_AFFINE_GAP:
            args = tuple(
                _integer(seed, index, channel) for channel in range(5)
            )
        elif group == _PRODUCT_THEN_AFFINE:
            args = (
                _integer_list(seed, index),
                _integer(seed, index, 1),
                _integer(seed, index, 2),
                _integer(seed, index, 3),
            )
        elif group == _GAP_THEN_AFFINE:
            args = tuple(
                _integer(seed, index, channel) for channel in range(5)
            )
        else:
            raise ValueError("unknown executable V6 capability group")
        rows.append((args, "v6-deterministic-property"))
    return tuple(rows)


def _cases(
    group: str,
    seed: int,
) -> tuple[
    Callable[..., JsonValue],
    tuple[tuple[JsonValue, ...], ...],
    tuple[tuple[tuple[JsonValue, ...], str], ...],
]:
    if group == _EVEN_TRANSFER:
        oracle = _even_transfer
        regressions = (
            ([],),
            ([0],),
            ([1, 3, 5],),
            ([2, -4, 7, 8],),
            ([6, 6, 0, -3, -2],),
        )
    elif group == _NORMALIZE_TRANSFER:
        oracle = _normalize_transfer
        regressions = (
            ("", "a"),
            (" A ", "b"),
            (" MiXeD ", "c"),
            ("already", "d"),
            ("  TWO Words  ", "e"),
        )
    elif group == _PRODUCT_TRANSFER:
        oracle = _product_transfer
        regressions = (
            ([], 1, 2, 3),
            ([0, 0], 2, 3, 4),
            ([2, 0, 3], 3, 4, 5),
            ([-2, 4, 0], 4, 5, 6),
            ([1, -3, -3, 2], 5, 6, 7),
        )
    elif group == _GAP_TRANSFER:
        oracle = _gap_transfer
        regressions = (
            (0, 0, 1, 2, 3),
            (-8, 5, 2, 3, 4),
            (9, -4, 3, 4, 5),
            (-3, -12, 4, 5, 6),
            (17, 2, 5, 6, 7),
        )
    elif group == _INTERIOR_EVEN_PRODUCT:
        oracle = _interior_even_product
        regressions = (
            ([],),
            ([2, 4],),
            ([1, 2, 4, 6, 3],),
            ([2, 1, 4, 3],),
            ([9, -2, 0, 4, 7],),
        )
    elif group == _OVERLAP_INTERIOR:
        oracle = _overlap_interior
        regressions = (
            ([], []),
            ([1, 2], [1, 2]),
            ([2, 1, 3, 4], [2, 1, 3, 5]),
            ([5, -1, 3, 7], [5, -1, 3, 9]),
            ([8, 4, 6, 2, 0], [6, 8, 4, 1, 0]),
        )
    elif group == _SHARED_EVEN_PRODUCT:
        oracle = _shared_even_product
        regressions = (
            ([], [], 0, 0),
            ([1, 3, 5], [1, 3, 5], 1, 2),
            ([2, 2, 4], [2, 4, 6], 2, 3),
            ([-4, -3, 0, 8], [-4, 0, 8], 3, 4),
            ([10, 8, 6, 4], [4, 6, 8, 10], 4, 5),
        )
    elif group == _DUAL_AFFINE_GAP:
        oracle = _dual_affine_gap
        regressions = (
            (0, 0, 0, 0, 0),
            (2, 5, 3, 1, -2),
            (-2, 4, 2, 5, 1),
            (7, -1, -2, 3, -4),
            (5, 5, 4, -3, 6),
        )
    elif group == _EVEN_INTERIOR_PRODUCT:
        oracle = _even_interior_product
        regressions = (
            ([1, 2, 4, 6, 3],),
            ([2, 1, 4, 3],),
            ([2, 4, 1, 6, 3],),
            ([8, 3, 0, -2, 5, 4, 7],),
            ([-4, 1, -2, 3, 6, 5],),
        )
    elif group == _INTERIOR_OVERLAP:
        oracle = _interior_overlap
        regressions = (
            ([2, 1, 3, 4], [2, 1, 3, 5]),
            ([5, -1, 3, 7], [5, -1, 3, 9]),
            ([8, 4, 6, 2, 0], [6, 8, 4, 1, 0]),
            ([9, 2, 4, 6], [9, 4, 2, 7]),
            ([-5, 0, 2, -1], [-5, 2, 0, 8]),
        )
    elif group == _PRODUCT_THEN_AFFINE:
        oracle = _product_then_affine
        regressions = (
            ([], 3, 2, 9),
            ([0, 0], 4, -1, 8),
            ([2, 0, 3], 5, 1, 7),
            ([-2, 4], -3, 2, 6),
            ([1, -3, -3, 2], 2, -4, 5),
        )
    elif group == _GAP_THEN_AFFINE:
        oracle = _gap_then_affine
        regressions = (
            (0, 0, 3, 2, 9),
            (2, 5, 3, 1, 8),
            (-2, 4, 2, 5, 7),
            (7, -1, -2, 3, 6),
            (5, 5, 4, -3, 5),
        )
    elif group == _NORMALIZE_RETENTION:
        oracle = _normalize_retention
        regressions = (
            ("", "r0"),
            (" A ", "r1"),
            (" MiXeD ", "r2"),
            ("retained", "r3"),
            ("  TWO Words  ", "r4"),
        )
    elif group == _SHARED_EVEN_PRODUCT_RETENTION:
        oracle = _shared_even_product_retention
        regressions = (
            ([], [], 0, 0),
            ([1, 3, 5], [1, 3, 5], 1, 2),
            ([2, 2, 4], [2, 4, 6], 2, 3),
            ([-4, -3, 0, 8], [-4, 0, 8], 3, 4),
            ([10, 8, 6, 4], [4, 6, 8, 10], 4, 5),
        )
    elif group in {
        _AMBIGUOUS_IDENTIFIER_PROFILE,
        _AMBIGUOUS_CHANNEL_FUSION,
    }:
        return _unreachable_stop_oracle, (), ()
    else:
        raise ValueError("unknown V6 capability group")
    return oracle, tuple(regressions), _generated_properties(group, seed)


def _build_task(spec: _TaskSpec, ordinal: int) -> ContinualToySweTrainTaskV6:
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
    public_placeholder = LearnerVisibleToySweObservationV6(
        task_commitment_sha256=generated.public.task_sha256,
        relative_path=f"workbench/v6_continual_{ordinal:02d}.py",
        signature=signature,
        prompt=generated.public.prompt,
        starter_source=_starter_source(spec, signature),
        observation_sha256="",
    )
    public = LearnerVisibleToySweObservationV6(
        task_commitment_sha256=public_placeholder.task_commitment_sha256,
        relative_path=public_placeholder.relative_path,
        signature=public_placeholder.signature,
        prompt=public_placeholder.prompt,
        starter_source=public_placeholder.starter_source,
        observation_sha256=_digest(
            b"continual-toy-swe-observation-v6",
            public_placeholder.record(),
        ),
    )
    private_placeholder = ToySweEnvironmentPrivateV6(
        generated_task=generated,
        capability_group=spec.capability_group,
        stage=spec.stage,
        expected_action=spec.expected_action,
        attempt_budget=spec.attempt_budget,
        feedback_pressure=spec.feedback_pressure,
        retention_anchor_ordinal=spec.retention_anchor_ordinal,
        environment_sha256="",
    )
    private = ToySweEnvironmentPrivateV6(
        generated_task=generated,
        capability_group=private_placeholder.capability_group,
        stage=private_placeholder.stage,
        expected_action=private_placeholder.expected_action,
        attempt_budget=private_placeholder.attempt_budget,
        feedback_pressure=private_placeholder.feedback_pressure,
        retention_anchor_ordinal=private_placeholder.retention_anchor_ordinal,
        environment_sha256=_digest(
            b"continual-toy-swe-environment-v6",
            private_placeholder.record(),
        ),
    )
    task_placeholder = ContinualToySweTrainTaskV6(
        public=public,
        private=private,
        task_sha256="",
    )
    task = ContinualToySweTrainTaskV6(
        public=public,
        private=private,
        task_sha256=_digest(
            b"continual-toy-swe-task-v6",
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
_EXPECTED_RETENTION_ANCHORS = {12: 1, 13: 6}


def _authenticate_stream(tasks: tuple[ContinualToySweTrainTaskV6, ...]) -> None:
    if len(tasks) != TASK_COUNT:
        raise ContinualToySweTaskV6Error("V6 TRAIN stream count changed")
    for task in tasks:
        task.authenticate()
    if len({task.task_sha256 for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV6Error("V6 task commitments are not unique")
    if len({task.public.signature.entry_point for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV6Error("V6 entry points are not unique")
    if len({task.public.prompt for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV6Error("V6 prompts are not unique")
    if len({task.public.starter_source for task in tasks}) != TASK_COUNT:
        raise ContinualToySweTaskV6Error("V6 starter sources are not unique")
    if Counter(task.private.stage for task in tasks) != _EXPECTED_STAGE_COUNTS:
        raise ContinualToySweTaskV6Error("V6 stage counts changed")
    if Counter(task.private.capability_group for task in tasks) != Counter(
        {group: 1 for group in _ALL_GROUPS}
    ):
        raise ContinualToySweTaskV6Error("V6 capability group counts changed")
    if Counter(task.private.expected_action for task in tasks) != _EXPECTED_ACTION_COUNTS:
        raise ContinualToySweTaskV6Error("V6 terminal-action counts changed")
    if Counter(task.private.attempt_budget for task in tasks) != _EXPECTED_BUDGET_COUNTS:
        raise ContinualToySweTaskV6Error("V6 attempt-budget counts changed")
    if Counter(len(task.public.signature.parameters) for task in tasks) != (
        _EXPECTED_ARITY_COUNTS
    ):
        raise ContinualToySweTaskV6Error("V6 interface-cardinality balance changed")
    anchors = {
        index: task.private.retention_anchor_ordinal
        for index, task in enumerate(tasks)
        if task.private.retention_anchor_ordinal is not None
    }
    if anchors != _EXPECTED_RETENTION_ANCHORS:
        raise ContinualToySweTaskV6Error("V6 retention anchors changed")
    expected_stage_order = (
        *(_TRANSFER for _ in range(4)),
        *(_RECOMBINATION for _ in range(4)),
        *(_FEEDBACK for _ in range(4)),
        *(_RETENTION for _ in range(2)),
        *(_INSUFFICIENT for _ in range(2)),
    )
    if tuple(task.private.stage for task in tasks) != expected_stage_order:
        raise ContinualToySweTaskV6Error("V6 developmental stage order changed")


def continual_toy_swe_train_stream_v6() -> tuple[ContinualToySweTrainTaskV6, ...]:
    """Return the authenticated immutable ordered V6 public-TRAIN stream."""

    _authenticate_stream(_TRAIN_STREAM)
    return _TRAIN_STREAM


def learner_visible_observation_v6(
    task: ContinualToySweTrainTaskV6,
) -> LearnerVisibleToySweObservationV6:
    """Project exactly the task-family-free learner record."""

    if type(task) is not ContinualToySweTrainTaskV6:
        raise TypeError("learner projection requires one exact V6 task")
    task.authenticate()
    return task.public


def train_stream_sha256_v6(
    tasks: tuple[ContinualToySweTrainTaskV6, ...] | None = None,
) -> str:
    selected = _TRAIN_STREAM if tasks is None else tasks
    if not isinstance(selected, tuple) or not all(
        type(task) is ContinualToySweTrainTaskV6 for task in selected
    ):
        raise TypeError("V6 TRAIN stream must contain exact continual tasks")
    _authenticate_stream(selected)
    return _digest(
        b"continual-toy-swe-stream-v6",
        {
            "schema": SCHEMA,
            "partition": PARTITION,
            "task_sha256s": tuple(task.task_sha256 for task in selected),
        },
    )


TRAIN_STREAM_SHA256 = (
    "4291faf1c0035dbff79b40250ab8cfd0e718d0a610ffd73684481a93fed60303"
)
if train_stream_sha256_v6() != TRAIN_STREAM_SHA256:
    raise ContinualToySweTaskV6Error("frozen V6 TRAIN stream root changed")
_STREAM_CUSTODY = {
    "schema": SCHEMA,
    "partition": PARTITION,
    "task_count": TASK_COUNT,
    "task_sha256s": tuple(task.task_sha256 for task in _TRAIN_STREAM),
    "train_stream_sha256": TRAIN_STREAM_SHA256,
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
    "revision_order_contrasts": (
        "interior-then-even_vs_even-then-interior",
        "overlap-then-interior_vs-interior-then-overlap",
        "affine-each-then-product_vs_product-then-affine",
        "affine-then-gap_vs_gap-then-affine",
    ),
    "authored_after_spent_v5_localization": True,
    "authored_before_v6_assembly_model_or_gate_output": True,
    "assembly_model_imports_during_authoring": 0,
    "assembly_model_outputs_used_during_authoring": 0,
    "successful_target_sources_stored": 0,
    "candidate_patches_stored": 0,
    "target_spans_or_groups_stored": 0,
    "answer_bearing_diagnostics_stored": 0,
    "protected_dev_or_evaluation_inputs_used": 0,
    "exact_v5_task_commitments_reused": 0,
}
TRAIN_STREAM_CUSTODY_SHA256 = (
    "e19d14234b9d3bef90927e176590ede49f17c8280208712ee0a455b743cf672b"
)
if (
    _digest(b"continual-toy-swe-stream-custody-v6", _STREAM_CUSTODY)
    != TRAIN_STREAM_CUSTODY_SHA256
):
    raise ContinualToySweTaskV6Error("frozen V6 TRAIN custody root changed")


def train_stream_custody_v6() -> dict[str, object]:
    """Return a detached authenticated custody record for the frozen V6 stream."""

    if train_stream_sha256_v6() != TRAIN_STREAM_SHA256:
        raise ContinualToySweTaskV6Error("V6 TRAIN stream root changed")
    if (
        _digest(b"continual-toy-swe-stream-custody-v6", _STREAM_CUSTODY)
        != TRAIN_STREAM_CUSTODY_SHA256
    ):
        raise ContinualToySweTaskV6Error("V6 stream custody changed")
    return json.loads(_canonical_json(_STREAM_CUSTODY))


__all__ = [
    "EXECUTION_PROPERTY_CASES",
    "EXECUTION_REGRESSION_CASES",
    "MAX_ATTEMPTS",
    "PARTITION",
    "SCHEMA",
    "TASK_COUNT",
    "TRAIN_STREAM_CUSTODY_SHA256",
    "TRAIN_STREAM_SHA256",
    "ContinualToySweTaskV6Error",
    "ContinualToySweTrainTaskV6",
    "LearnerVisibleToySweObservationV6",
    "ToySweEnvironmentPrivateV6",
    "continual_toy_swe_train_stream_v6",
    "learner_visible_observation_v6",
    "train_stream_custody_v6",
    "train_stream_sha256_v6",
]
