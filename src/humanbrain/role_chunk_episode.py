"""Role-normalized concept episodes for public Simple-SWE prompts.

This module is tokenizer plumbing, not a predictor.  It removes task-specific
entry-point, parameter, and numeric-literal payloads from a prompt while
retaining an exact sidecar, then encodes the resulting structural text with the
existing :class:`~humanbrain.concept_tokens.ConceptTokenizer`.

Only public TRAIN records are admitted.  The resulting immutable token tuple is
intended to become the input stream of a later cell-only native recognition
probe; no label, family name, private case, or simulator state enters it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import re

from humanbrain.concept_tokens import ConceptConfig, ConceptTokenizer
from humanbrain.simple_swe.tasks import PublicTask
from humanbrain.tokenizer import TokenizerConfig
from humanbrain.typed_program_types import ProgramType


_MARKER_PREFIX = "[[HB:"
_MARKER_SUFFIX = "]]"
_IDENTIFIER_CHARACTERS = "A-Za-z0-9_"
_NUMERIC_LITERAL = re.compile(
    rf"(?<![{_IDENTIFIER_CHARACTERS}])"
    r"(?P<value>[+-]?(?:(?:\d+\.\d+|\.\d+|\d+)(?:[eE][+-]?\d+)?))"
    rf"(?![{_IDENTIFIER_CHARACTERS}])"
)


DEFAULT_ROLE_CHUNK_CONCEPT_CONFIG = ConceptConfig(
    min_frequency=2,
    min_mutual_information_bits=5.0,
    max_concepts=64,
    min_tokens=2,
    max_tokens=12,
)


class RolePayloadKind(str, Enum):
    """Payload classes removed from the structural prompt stream."""

    ENTRY_POINT = "entry_point"
    PARAMETER = "parameter"
    INTEGER_LITERAL = "integer_literal"
    FLOAT_LITERAL = "float_literal"


@dataclass(frozen=True, slots=True)
class RolePayload:
    """One reversible payload bound to a stable structural marker."""

    kind: RolePayloadKind
    ordinal: int
    marker: str
    value: str

    def __post_init__(self) -> None:
        if type(self.kind) is not RolePayloadKind:
            raise TypeError("payload kind must be a RolePayloadKind")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("payload ordinal must be a nonnegative exact integer")
        if type(self.marker) is not str or not (
            self.marker.startswith(_MARKER_PREFIX)
            and self.marker.endswith(_MARKER_SUFFIX)
        ):
            raise ValueError("payload marker is outside the role-chunk namespace")
        if type(self.value) is not str or not self.value:
            raise ValueError("payload value must be nonempty text")


@dataclass(frozen=True, slots=True)
class RoleChunkEpisode:
    """One immutable structural prompt stream plus its reversible payloads."""

    task_id: str
    structural_text: str
    token_ids: tuple[int, ...]
    payloads: tuple[RolePayload, ...]

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            raise ValueError("episode task_id must be nonempty text")
        if type(self.structural_text) is not str or not self.structural_text:
            raise ValueError("episode structural_text must be nonempty text")
        if not isinstance(self.token_ids, tuple) or not self.token_ids:
            raise ValueError("episode token_ids must be a nonempty tuple")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.token_ids):
            raise ValueError("episode token IDs must be nonnegative exact integers")
        if not isinstance(self.payloads, tuple) or not all(
            type(payload) is RolePayload for payload in self.payloads
        ):
            raise TypeError("episode payloads must be a tuple of RolePayload values")
        markers = tuple(payload.marker for payload in self.payloads)
        if len(markers) != len(set(markers)):
            raise ValueError("episode payload markers must be unique")
        if any(marker not in self.structural_text for marker in markers):
            raise ValueError("every payload marker must occur in the structural text")

    @property
    def structural_sha256(self) -> str:
        return sha256(self.structural_text.encode("utf-8")).hexdigest()

    def payloads_for(self, kind: RolePayloadKind) -> tuple[RolePayload, ...]:
        if type(kind) is not RolePayloadKind:
            raise TypeError("kind must be a RolePayloadKind")
        return tuple(payload for payload in self.payloads if payload.kind is kind)

    def restore_prompt(self) -> str:
        """Reinsert every sidecar value and return the exact public prompt."""

        prompt = self.structural_text
        for payload in self.payloads:
            if payload.marker not in prompt:
                raise ValueError(f"payload marker disappeared: {payload.marker}")
            prompt = prompt.replace(payload.marker, payload.value)
        if _MARKER_PREFIX in prompt:
            raise ValueError("restored prompt retains an unresolved role marker")
        return prompt


class RoleChunkEpisodeEncoder:
    """Deterministic public-TRAIN prompt encoder backed by concept chunks."""

    __slots__ = ("_tokenizer",)

    def __init__(self, tokenizer: ConceptTokenizer) -> None:
        if type(tokenizer) is not ConceptTokenizer:
            raise TypeError("tokenizer must be a ConceptTokenizer")
        self._tokenizer = tokenizer

    @classmethod
    def fit_public_train(
        cls,
        tasks: Iterable[PublicTask],
        *,
        concept_config: ConceptConfig | None = None,
        base_config: TokenizerConfig | None = None,
    ) -> RoleChunkEpisodeEncoder:
        """Fit a concept bank from public TRAIN structural prompts only."""

        materialized = tuple(tasks)
        if not materialized:
            raise ValueError("at least one public TRAIN task is required")
        _validate_public_train_tasks(materialized)
        task_ids = tuple(task.task_id for task in materialized)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("concept fitting tasks must have unique task IDs")
        structural_texts = tuple(
            _normalize_public_train_task(task)[0] for task in materialized
        )
        tokenizer = ConceptTokenizer.fit(
            structural_texts,
            config=concept_config or DEFAULT_ROLE_CHUNK_CONCEPT_CONFIG,
            base_config=base_config,
        )
        return cls(tokenizer)

    @property
    def tokenizer(self) -> ConceptTokenizer:
        return self._tokenizer

    def encode(self, task: PublicTask) -> RoleChunkEpisode:
        """Normalize and concept-encode one public TRAIN task."""

        _validate_public_train_tasks((task,))
        structural_text, payloads = _normalize_public_train_task(task)
        episode = RoleChunkEpisode(
            task_id=task.task_id,
            structural_text=structural_text,
            token_ids=tuple(self._tokenizer.encode(structural_text)),
            payloads=payloads,
        )
        if episode.restore_prompt() != task.prompt:
            raise RuntimeError("role payload sidecar failed exact prompt restoration")
        return episode


def _validate_public_train_tasks(tasks: tuple[object, ...]) -> None:
    for task in tasks:
        if type(task) is not PublicTask:
            raise TypeError("role-chunk encoding accepts exact PublicTask values only")
        if task.partition != "train":
            raise ValueError("role-chunk encoding is restricted to the public TRAIN partition")


def _program_type_tag(program_type: ProgramType) -> str:
    if not program_type.arguments:
        return program_type.constructor.upper()
    return "_".join(
        (program_type.constructor.upper(),)
        + tuple(_program_type_tag(argument) for argument in program_type.arguments)
    )


def _identifier_pattern(identifier: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![{_IDENTIFIER_CHARACTERS}])"
        rf"{re.escape(identifier)}"
        rf"(?![{_IDENTIFIER_CHARACTERS}])"
    )


def _normalize_public_train_task(
    task: PublicTask,
) -> tuple[str, tuple[RolePayload, ...]]:
    if _MARKER_PREFIX in task.prompt:
        raise ValueError("public prompt collides with the role-chunk marker namespace")

    numeric_payloads: list[RolePayload] = []
    numeric_counts = {
        RolePayloadKind.INTEGER_LITERAL: 0,
        RolePayloadKind.FLOAT_LITERAL: 0,
    }

    def replace_numeric(match: re.Match[str]) -> str:
        value = match.group("value")
        kind = (
            RolePayloadKind.FLOAT_LITERAL
            if any(character in value for character in ".eE")
            else RolePayloadKind.INTEGER_LITERAL
        )
        ordinal = numeric_counts[kind]
        numeric_counts[kind] += 1
        marker = f"{_MARKER_PREFIX}{kind.name}:{ordinal}{_MARKER_SUFFIX}"
        numeric_payloads.append(RolePayload(kind, ordinal, marker, value))
        return marker

    structural_text = _NUMERIC_LITERAL.sub(replace_numeric, task.prompt)

    entry_marker = (
        f"{_MARKER_PREFIX}ENTRY_POINT:RET_"
        f"{_program_type_tag(task.signature.result)}{_MARKER_SUFFIX}"
    )
    structural_text, entry_count = _identifier_pattern(
        task.signature.entry_point
    ).subn(entry_marker, structural_text)
    if entry_count == 0:
        raise ValueError("public prompt does not contain its declared entry point")
    named_payloads: list[RolePayload] = [
        RolePayload(
            RolePayloadKind.ENTRY_POINT,
            0,
            entry_marker,
            task.signature.entry_point,
        )
    ]

    for ordinal, parameter in enumerate(task.signature.parameters):
        marker = (
            f"{_MARKER_PREFIX}PARAMETER:{ordinal}:"
            f"{_program_type_tag(parameter.type)}{_MARKER_SUFFIX}"
        )
        structural_text, parameter_count = _identifier_pattern(parameter.name).subn(
            marker, structural_text
        )
        if parameter_count == 0:
            raise ValueError(
                f"public prompt does not contain declared parameter {parameter.name!r}"
            )
        named_payloads.append(
            RolePayload(RolePayloadKind.PARAMETER, ordinal, marker, parameter.name)
        )

    return structural_text, tuple((*named_payloads, *numeric_payloads))


__all__ = [
    "DEFAULT_ROLE_CHUNK_CONCEPT_CONFIG",
    "RoleChunkEpisode",
    "RoleChunkEpisodeEncoder",
    "RolePayload",
    "RolePayloadKind",
]
