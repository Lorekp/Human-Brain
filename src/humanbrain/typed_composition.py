"""Frozen public A0/A1/A2 typed-composition corpus and audit values.

This module constructs public TRAIN and calibration values without growing or
completing a cell circuit. Trusted task metadata is deliberately separated from
the two-field request that may cross the completion boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from humanbrain import typed_program
from humanbrain.typed_program import (
    Example,
    Layer1Protocol,
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

_IDENTIFIER_ALPHABET = "bcdfghjklmnpqrstvwxyz"
_IDENTIFIER_PREFIX = "q_"
_IDENTIFIER_BODY_LENGTH = 9
_MAX_PROTOCOL_BYTES = 32 * 1024
_EXPECTED_PROTOCOL_ID = "humanbrain-typed-composition-a012-v1"
COMPOSITION_PROTOCOL_SHA256 = "2d41d366611c6217dd8a14411b68bc03be1cafb4572e41f3c5a77da326cda412"
_FAMILY_IDS = (
    "scalar_add",
    "scalar_input_minus_literal",
    "scalar_literal_minus_input",
    "list_add",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _record(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def _integer_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or not all(type(item) is int for item in value):
        raise ValueError(f"{label} must be a non-empty integer list")
    return tuple(value)


def derive_opaque_identifier(domain: str, ordinal: int) -> str:
    """Derive a valid opaque Python identifier from a domain and ordinal."""
    if not isinstance(domain, str) or not domain:
        raise ValueError("identifier domain must be non-empty text")
    if type(ordinal) is not int or not 0 <= ordinal <= 65_535:
        raise ValueError("opaque ordinal must be an integer from 0 through 65535")
    digest = sha256(domain.encode("utf-8") + b"\x00" + ordinal.to_bytes(2, "big")).digest()
    body = "".join(
        _IDENTIFIER_ALPHABET[value % len(_IDENTIFIER_ALPHABET)]
        for value in digest[:_IDENTIFIER_BODY_LENGTH]
    )
    return _IDENTIFIER_PREFIX + body


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """The entire value allowed to cross into :func:`complete`."""

    prompt: str
    signature: TypedSignature


@dataclass(frozen=True, slots=True)
class SemanticCase:
    case_id: str
    arguments: tuple[object, ...]
    expected: object


@dataclass(frozen=True, slots=True)
class CompositionTask:
    """Trusted public evidence record; only ``completion_request`` reaches cells."""

    cohort: str
    task_id: str
    family_id: str
    template_id: str
    literal: int | None
    opaque_ordinal: int
    prompt: str
    signature: TypedSignature
    teacher_tokens: tuple[str, ...]
    cases: tuple[SemanticCase, ...]

    def completion_request(self) -> CompletionRequest:
        return CompletionRequest(prompt=self.prompt, signature=self.signature)

    def training_example(self) -> Example:
        if self.cohort != "TRAIN":
            raise ValueError("growth examples may be constructed from TRAIN tasks only")
        return Example(
            prompt=self.prompt,
            signature=self.signature,
            teacher_tokens=self.teacher_tokens,
        )


@dataclass(frozen=True, slots=True)
class LaneCommitment:
    lane: str
    diagnostic_only: bool
    eligible: bool
    framing_order: str
    supervision_scope: str


@dataclass(frozen=True, slots=True)
class FramingComponents:
    prompt_bytes: bytes
    signature_bytes: bytes
    vocabulary_sha256: str
    token_ids: tuple[int, ...]
    state_observation_ids: tuple[int, ...]
    strengthen_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CorpusAccounting:
    train_task_count: int
    calibration_task_count: int
    train_case_count: int
    calibration_case_count: int
    teacher_targets_per_pass: int
    framing_tokens_per_pass: int
    observations_per_pass: int
    corpus_passes: int
    a2_strengthen_calls: int
    a0_a1_strengthen_calls: int
    observation_calls_per_lane: int


@dataclass(frozen=True, slots=True)
class FutureCohortCommitment:
    seed: None
    generated: bool
    task_count: int
    factor_counts: dict[str, int]
    forbidden_sources: tuple[str, ...]
    seed_creation_preconditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompositionProtocol:
    protocol_id: str
    raw_sha256: str
    base_commit: str
    identifier_domain: str
    train_tasks: tuple[CompositionTask, ...]
    calibration_tasks: tuple[CompositionTask, ...]
    lanes: tuple[LaneCommitment, ...]
    accounting: CorpusAccounting
    future_cohort: FutureCohortCommitment
    source_bundle_paths: tuple[str, ...]

    def training_examples(self) -> tuple[Example, ...]:
        return tuple(task.training_example() for task in self.train_tasks)

    def framing_components(
        self,
        task: CompositionTask,
    ) -> dict[str, FramingComponents]:
        teacher_ids = typed_program._program_token_ids(task.teacher_tokens)
        result: dict[str, FramingComponents] = {}
        for lane in self.lanes:
            layer_protocol = Layer1Protocol.for_lane(lane.lane)
            input_ids = typed_program._model_input(
                task.prompt,
                task.signature,
                framing_order=layer_protocol.framing_order,
            )
            sequence_ids = (*input_ids, *teacher_ids)
            if lane.supervision_scope == "all-next-token-targets":
                strengthen_positions = tuple(range(1, len(sequence_ids)))
            else:
                strengthen_positions = tuple(range(len(input_ids), len(sequence_ids)))
            result[lane.lane] = FramingComponents(
                prompt_bytes=task.prompt.encode("utf-8"),
                signature_bytes=_canonical_json(task.signature.to_record()),
                vocabulary_sha256=layer_protocol.primitive_table_sha256,
                token_ids=input_ids,
                state_observation_ids=sequence_ids[:-1],
                strengthen_positions=strengthen_positions,
            )
        return result


def _literal_tokens(value: int) -> tuple[str, ...]:
    sign = "-" if value < 0 else "+"
    return ("integer", sign, *tuple(str(abs(value))), "end")


def _teacher(family_id: str, literal: int | None) -> tuple[str, ...]:
    if family_id in {"scalar_identity", "list_identity"}:
        return ("variable-0",)
    assert literal is not None
    literal_tokens = _literal_tokens(literal)
    if family_id == "scalar_add":
        return ("add", "variable-0", *literal_tokens)
    if family_id == "scalar_input_minus_literal":
        return ("subtract", "variable-0", *literal_tokens)
    if family_id == "scalar_literal_minus_input":
        return ("subtract", *literal_tokens, "variable-0")
    if family_id == "list_add":
        return (
            "map",
            "bind",
            "add",
            "variable-0",
            *literal_tokens,
            "variable-0",
        )
    raise ValueError("unsupported typed-composition family")


def _cases(family_id: str, literal: int | None) -> tuple[SemanticCase, ...]:
    if family_id in {"list_add", "list_identity"}:
        inputs: tuple[tuple[int, ...], ...] = ((), (-11, 0, 13), (9, -4))
        case_ids = ("empty", "mixed", "signed")
        cases: list[SemanticCase] = []
        for case_id, values in zip(case_ids, inputs, strict=True):
            argument = list(values)
            if family_id == "list_add":
                assert literal is not None
                expected: object = [value + literal for value in values]
            else:
                expected = list(values)
            cases.append(SemanticCase(case_id, (argument,), expected))
        return tuple(cases)

    inputs = (-11, 0, 13)
    case_ids = ("negative", "zero", "positive")
    scalar_cases: list[SemanticCase] = []
    for case_id, value in zip(case_ids, inputs, strict=True):
        if family_id == "scalar_identity":
            expected = value
        elif family_id == "scalar_add":
            assert literal is not None
            expected = value + literal
        elif family_id == "scalar_input_minus_literal":
            assert literal is not None
            expected = value - literal
        elif family_id == "scalar_literal_minus_input":
            assert literal is not None
            expected = literal - value
        else:
            raise ValueError("unsupported typed-composition family")
        scalar_cases.append(SemanticCase(case_id, (value,), expected))
    return tuple(scalar_cases)


def _signature(entry_point: str, family_id: str) -> TypedSignature:
    value_type = (
        ProgramType.List(ProgramType.Int())
        if family_id in {"list_add", "list_identity"}
        else ProgramType.Int()
    )
    return TypedSignature(
        entry_point=entry_point,
        parameters=(ProgramParameter("datum", value_type),),
        result=value_type,
    )


def _task_sort_key(domain: str, task_id: str) -> bytes:
    return sha256(domain.encode("utf-8") + b"\x00" + task_id.encode("utf-8")).digest()


def _task_specs(
    record: dict[str, Any],
    *,
    cohort: str,
    literals: tuple[int, ...],
) -> list[dict[str, object]]:
    templates = _record(
        record["templates"],
        frozenset(_FAMILY_IDS),
        "typed-composition templates",
    )
    specs: list[dict[str, object]] = []
    all_literals = tuple(record["literals"])
    for family_id in _FAMILY_IDS:
        pair = _record(
            templates[family_id],
            frozenset({"even", "odd"}),
            f"{family_id} templates",
        )
        for literal in literals:
            source_index = all_literals.index(literal)
            parity = source_index % 2
            if cohort == "CALIBRATION":
                parity = 1 - parity
            template_id = "even" if parity == 0 else "odd"
            prompt = _text(pair[template_id], "prompt template").format(k=literal)
            specs.append(
                {
                    "cohort": cohort,
                    "task_id": (f"{cohort.lower()}-{family_id}-{literal:+d}"),
                    "family_id": family_id,
                    "template_id": template_id,
                    "literal": literal,
                    "prompt": prompt,
                }
            )
    identity = _record(
        record["identity_prompts"],
        frozenset(
            {
                "train_scalar",
                "train_list",
                "calibration_scalar",
                "calibration_list",
            }
        ),
        "identity prompts",
    )
    identity_prefix = "train" if cohort == "TRAIN" else "calibration"
    for kind in ("scalar", "list"):
        specs.append(
            {
                "cohort": cohort,
                "task_id": f"{cohort.lower()}-{kind}-identity",
                "family_id": f"{kind}_identity",
                "template_id": "identity",
                "literal": None,
                "prompt": _text(identity[f"{identity_prefix}_{kind}"], "identity prompt"),
            }
        )
    return specs


def _materialize_tasks(
    specs: list[dict[str, object]],
    *,
    identifier_domain: str,
    order_domain: str,
    ordinal_offset: int,
) -> tuple[CompositionTask, ...]:
    ordered = sorted(
        specs,
        key=lambda item: _task_sort_key(order_domain, str(item["task_id"])),
    )
    tasks: list[CompositionTask] = []
    for index, spec in enumerate(ordered):
        ordinal = ordinal_offset + index
        family_id = str(spec["family_id"])
        literal_value = spec["literal"]
        literal = literal_value if type(literal_value) is int else None
        tasks.append(
            CompositionTask(
                cohort=str(spec["cohort"]),
                task_id=str(spec["task_id"]),
                family_id=family_id,
                template_id=str(spec["template_id"]),
                literal=literal,
                opaque_ordinal=ordinal,
                prompt=str(spec["prompt"]),
                signature=_signature(
                    derive_opaque_identifier(identifier_domain, ordinal),
                    family_id,
                ),
                teacher_tokens=_teacher(family_id, literal),
                cases=_cases(family_id, literal),
            )
        )
    return tuple(tasks)


def _accounting(
    tasks: tuple[CompositionTask, ...],
    calibration_tasks: tuple[CompositionTask, ...],
    passes: int,
) -> CorpusAccounting:
    a2 = Layer1Protocol.for_lane("A2")
    framing = sum(
        len(
            typed_program._model_input(
                task.prompt,
                task.signature,
                framing_order=a2.framing_order,
            )
        )
        for task in tasks
    )
    targets = sum(len(task.teacher_tokens) for task in tasks)
    observations = framing + targets - len(tasks)
    return CorpusAccounting(
        train_task_count=len(tasks),
        calibration_task_count=len(calibration_tasks),
        train_case_count=sum(len(task.cases) for task in tasks),
        calibration_case_count=sum(len(task.cases) for task in calibration_tasks),
        teacher_targets_per_pass=targets,
        framing_tokens_per_pass=framing,
        observations_per_pass=observations,
        corpus_passes=passes,
        a2_strengthen_calls=passes * targets,
        a0_a1_strengthen_calls=passes * observations,
        observation_calls_per_lane=passes * observations,
    )


def load_composition_protocol(path: Path) -> CompositionProtocol:
    """Validate and materialize the exact public protocol without executing it."""
    if not isinstance(path, Path):
        raise TypeError("composition protocol path must be a Path")
    with path.open("rb") as handle:
        raw = handle.read(_MAX_PROTOCOL_BYTES + 1)
    if len(raw) > _MAX_PROTOCOL_BYTES:
        raise ValueError("typed-composition protocol exceeds the 32 KiB limit")
    raw_sha256 = sha256(raw).hexdigest()
    if raw_sha256 != COMPOSITION_PROTOCOL_SHA256:
        raise ValueError("typed-composition raw protocol SHA-256 is not frozen")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("composition protocol is not UTF-8 JSON") from error
    record = _record(
        decoded,
        frozenset(
            {
                "protocol_id",
                "base_commit",
                "corpus_passes",
                "identifier",
                "task_order_domain",
                "literals",
                "calibration_literals",
                "templates",
                "identity_prompts",
                "lanes",
                "expected_accounting",
                "future_cohort",
                "source_bundle_paths",
                "diagnostics",
                "eligibility_gates",
                "execution",
            }
        ),
        "typed-composition protocol",
    )
    protocol_id = _text(record["protocol_id"], "protocol id")
    if protocol_id != _EXPECTED_PROTOCOL_ID:
        raise ValueError("unsupported typed-composition protocol id")
    identifier = _record(
        record["identifier"],
        frozenset({"algorithm", "domain", "alphabet", "prefix", "body_length"}),
        "identifier commitment",
    )
    if (
        identifier["algorithm"] != "sha256-domain-nul-u16be-mod-alphabet"
        or identifier["alphabet"] != _IDENTIFIER_ALPHABET
        or identifier["prefix"] != _IDENTIFIER_PREFIX
        or identifier["body_length"] != _IDENTIFIER_BODY_LENGTH
    ):
        raise ValueError("opaque identifier commitments differ from implementation")
    identifier_domain = _text(identifier["domain"], "identifier domain")
    literals = _integer_tuple(record["literals"], "TRAIN literals")
    calibration_literals = _integer_tuple(
        record["calibration_literals"],
        "calibration literals",
    )
    if len(literals) != 10 or len(set(literals)) != 10:
        raise ValueError("TRAIN must commit exactly ten unique literals")
    if any(value not in literals for value in calibration_literals):
        raise ValueError("calibration literals must be selected from TRAIN literals")
    order_domain = _text(record["task_order_domain"], "task order domain")
    train_tasks = _materialize_tasks(
        _task_specs(record, cohort="TRAIN", literals=literals),
        identifier_domain=identifier_domain,
        order_domain=order_domain + "/train",
        ordinal_offset=0,
    )
    calibration_tasks = _materialize_tasks(
        _task_specs(record, cohort="CALIBRATION", literals=calibration_literals),
        identifier_domain=identifier_domain,
        order_domain=order_domain + "/calibration",
        ordinal_offset=len(train_tasks),
    )

    lane_values = record["lanes"]
    if not isinstance(lane_values, list) or len(lane_values) != 3:
        raise ValueError("protocol must commit exactly A0, A1, and A2")
    lanes: list[LaneCommitment] = []
    for lane_value in lane_values:
        lane_record = _record(
            lane_value,
            frozenset(
                {"lane", "diagnostic_only", "eligible", "framing_order", "supervision_scope"}
            ),
            "lane commitment",
        )
        lane = _text(lane_record["lane"], "lane")
        layer_protocol = Layer1Protocol.for_lane(lane)
        if (
            type(lane_record["diagnostic_only"]) is not bool
            or type(lane_record["eligible"]) is not bool
        ):
            raise ValueError("lane eligibility fields must be booleans")
        commitment = LaneCommitment(
            lane=lane,
            diagnostic_only=lane_record["diagnostic_only"],
            eligible=lane_record["eligible"],
            framing_order=_text(lane_record["framing_order"], "lane framing order"),
            supervision_scope=_text(
                lane_record["supervision_scope"],
                "lane supervision scope",
            ),
        )
        if (
            commitment.framing_order != layer_protocol.framing_order
            or commitment.supervision_scope != layer_protocol.supervision_scope
        ):
            raise ValueError("lane commitments differ from Layer1Protocol")
        lanes.append(commitment)
    if (
        tuple(lane.lane for lane in lanes) != ("A0", "A1", "A2")
        or tuple(lane.diagnostic_only for lane in lanes) != (True, True, False)
        or tuple(lane.eligible for lane in lanes) != (False, False, True)
    ):
        raise ValueError("A2 must be the sole eligible lane")

    passes = record["corpus_passes"]
    if type(passes) is not int or passes != 3:
        raise ValueError("typed-composition protocol requires exactly three corpus passes")
    accounting = _accounting(train_tasks, calibration_tasks, passes)
    expected_accounting = _record(
        record["expected_accounting"],
        frozenset(CorpusAccounting.__dataclass_fields__),
        "expected accounting",
    )
    actual_accounting = {
        name: getattr(accounting, name) for name in CorpusAccounting.__dataclass_fields__
    }
    if expected_accounting != actual_accounting:
        raise ValueError("mechanical corpus accounting differs from frozen expected accounting")

    future_record = _record(
        record["future_cohort"],
        frozenset(
            {
                "generated",
                "task_count",
                "factor_counts",
                "forbidden_sources",
                "seed_creation_preconditions",
            }
        ),
        "future cohort commitment",
    )
    if future_record["generated"] is not False:
        raise ValueError("future cohort must remain ungenerated at protocol freeze")
    factor_counts = future_record["factor_counts"]
    if not isinstance(factor_counts, dict) or not all(
        isinstance(key, str) and type(value) is int for key, value in factor_counts.items()
    ):
        raise ValueError("future factor counts must be an integer record")
    forbidden_sources = future_record["forbidden_sources"]
    seed_preconditions = future_record["seed_creation_preconditions"]
    if not isinstance(forbidden_sources, list) or not all(
        isinstance(item, str) for item in forbidden_sources
    ):
        raise ValueError("future forbidden sources must be text")
    if not isinstance(seed_preconditions, list) or not all(
        isinstance(item, str) for item in seed_preconditions
    ):
        raise ValueError("future seed preconditions must be text")
    future = FutureCohortCommitment(
        seed=None,
        generated=False,
        task_count=int(future_record["task_count"]),
        factor_counts=dict(factor_counts),
        forbidden_sources=tuple(forbidden_sources),
        seed_creation_preconditions=tuple(seed_preconditions),
    )
    if future.task_count != sum(future.factor_counts.values()):
        raise ValueError("future factor counts must sum to its task count")

    source_paths = record["source_bundle_paths"]
    if (
        not isinstance(source_paths, list)
        or not source_paths
        or not all(isinstance(item, str) and item for item in source_paths)
    ):
        raise ValueError("source bundle paths must be non-empty text")
    return CompositionProtocol(
        protocol_id=protocol_id,
        raw_sha256=raw_sha256,
        base_commit=_text(record["base_commit"], "base commit"),
        identifier_domain=identifier_domain,
        train_tasks=train_tasks,
        calibration_tasks=calibration_tasks,
        lanes=tuple(lanes),
        accounting=accounting,
        future_cohort=future,
        source_bundle_paths=tuple(source_paths),
    )
