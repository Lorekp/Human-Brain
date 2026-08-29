"""Checkpoint-free public-TRAIN prompt-span input for semantic synthesis.

This module is intentionally only an input substrate.  It authenticates the
frozen Simple-SWE callable frame, removes declared identifier spelling, emits
ordered lexical spans, and projects each span into a sparse fitted-cell
assembly.  Signed integer lexemes use one lexical ``<INT>`` token while their
value and location remain available through an independent physical typed
output encoding.

There are no task/family identities, synonym tables, semantic labels, desired
outputs, candidate programs, or learned parameters here.  In particular, a
span assembly is derived from that span alone, never from a whole-prompt hash.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256

import numpy as np

from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.typed_program_types import TypedSignature


PUBLIC_TRAIN_INPUT_SCOPE = "PUBLIC_TRAIN_PROMPT_SPANS_ONLY"
TRAINABLE_STATE_BYTES = 0

FROZEN_PROMPT_PREFIX = (
    "Return exactly one complete Python function. Use only Python built-ins; "
    "do not emit Markdown, prose, tests, imports, or input/output examples.\n"
    "Implement this exact task:\n"
)
_DOCSTRING_OPEN = '    """'
_DOCSTRING_CLOSE = '"""'

MAX_PARAMETER_COUNT = 2
MAX_SPAN_TOKENS = 4
ASSEMBLY_CELL_COUNT = 512
ASSEMBLY_ACTIVE_CELL_COUNT = 12
ASSEMBLY_CURRENT_PA = 5_000.0
ASSEMBLY_DRIVE_STEPS = 30

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_SIGNED_INTEGER = r"(?<![A-Za-z0-9_])[+-]?\d+(?![A-Za-z0-9_])"
_MULTI_CHARACTER_PUNCTUATION = r"<=|>=|==|!=|//|\*\*|->"
_TOKENIZER = re.compile(
    rf"(?P<integer>{_SIGNED_INTEGER})"
    rf"|(?P<identifier>{_IDENTIFIER})"
    rf"|(?P<punctuation>{_MULTI_CHARACTER_PUNCTUATION}|[^\s])"
)

_TOKENIZER_NAMESPACE = b"public-train-prompt-span-tokenizer-v1\0"
_SPAN_IDENTITY_NAMESPACE = b"public-train-prompt-span-identity-v1\0"
_ASSEMBLY_NAMESPACE = b"public-train-prompt-span-assembly-v1\0"
_LITERAL_NAMESPACE = b"public-train-prompt-literal-copy-v1\0"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _config_sha256(namespace: bytes, record: object) -> str:
    digest = sha256()
    digest.update(namespace)
    digest.update(_canonical_json(record))
    return digest.hexdigest()


TOKENIZER_SHA256 = _config_sha256(
    _TOKENIZER_NAMESPACE,
    {
        "frozen_prompt_prefix": FROZEN_PROMPT_PREFIX,
        "docstring_open": _DOCSTRING_OPEN,
        "docstring_close": _DOCSTRING_CLOSE,
        "token_pattern": _TOKENIZER.pattern,
        "lowercase": True,
        "declared_name_binding": "reverse-declaration-depth",
        "integer_lexical_token": "<INT>",
        "max_parameter_count": MAX_PARAMETER_COUNT,
        "max_span_tokens": MAX_SPAN_TOKENS,
    },
)

ASSEMBLY_PROJECTION_SHA256 = _config_sha256(
    _ASSEMBLY_NAMESPACE,
    {
        "span_identity_namespace": _SPAN_IDENTITY_NAMESPACE.decode("ascii"),
        "cell_count": ASSEMBLY_CELL_COUNT,
        "active_cell_count": ASSEMBLY_ACTIVE_CELL_COUNT,
        "selection": "sha256-counter-modulo-with-deduplication",
        "current_pa": ASSEMBLY_CURRENT_PA,
        "drive_steps": ASSEMBLY_DRIVE_STEPS,
        "cell_model": "AdaptiveGlifPopulation",
    },
)


@dataclass(frozen=True, slots=True)
class SignedLiteralCopy:
    """One signed integer kept outside the lexical span channel."""

    lexeme: str
    value: int
    suffix_character_start: int
    suffix_character_end: int
    suffix_token_index: int
    canonical_token_index: int
    typed_output: output_cells.TypedOutputEncoding
    typed_output_active_cells: tuple[int, ...]
    semantic_sha256: str
    audit_sha256: str


@dataclass(frozen=True, slots=True)
class PhysicalSpanAssembly:
    """One ordered lexical span and its distributed fitted-cell event."""

    start_token: int
    stop_token: int
    tokens: tuple[str, ...]
    identity_sha256: str
    assembly_cells: tuple[int, ...]
    first_spike_step: int
    spike_cells: tuple[int, ...]
    physical_drive_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PromptSpanInput:
    """Immutable, checkpoint-free input record for one framed prompt."""

    scope: str
    trainable_state_bytes: int
    audit_prompt_sha256: str
    tokenizer_sha256: str
    assembly_projection_sha256: str
    typed_output_projection_sha256: str
    tokens: tuple[str, ...]
    signed_literals: tuple[SignedLiteralCopy, ...]
    spans: tuple[PhysicalSpanAssembly, ...]
    lexical_sha256: str
    literal_channel_sha256: str
    sha256: str


def _validate_signature(signature: TypedSignature) -> tuple[str, ...]:
    if not isinstance(signature, TypedSignature):
        raise TypeError("prompt-span input requires a TypedSignature")
    names = tuple(parameter.name for parameter in signature.parameters)
    if not 1 <= len(names) <= MAX_PARAMETER_COUNT:
        raise ValueError("prompt-span input supports one or two declared parameters")
    lowered = tuple(name.lower() for name in names)
    if len(lowered) != len(set(lowered)):
        raise ValueError("declared parameter names collide after lowercasing")
    if signature.entry_point.lower() in lowered:
        raise ValueError("entry point collides with a parameter after lowercasing")
    return names


def _extract_task_suffix(
    prompt: str,
    signature: TypedSignature,
    parameter_names: tuple[str, ...],
) -> str:
    if type(prompt) is not str or not prompt:
        raise ValueError("prompt-span input requires non-empty prompt text")
    parameter_frame = ", ".join(parameter_names)
    frame = (
        FROZEN_PROMPT_PREFIX
        + f"def {signature.entry_point}({parameter_frame}):\n"
        + _DOCSTRING_OPEN
    )
    if not prompt.startswith(frame) or not prompt.endswith(_DOCSTRING_CLOSE):
        raise ValueError("prompt does not match the frozen callable framing")
    suffix = prompt[len(frame) : -len(_DOCSTRING_CLOSE)]
    if (
        not suffix
        or "\n" in suffix
        or "\r" in suffix
        or _DOCSTRING_CLOSE in suffix
    ):
        raise ValueError("frozen callable framing requires one non-empty task suffix")
    return suffix


def _frame_tokens(parameter_count: int) -> tuple[str, ...]:
    tokens: list[str] = ["def", "<ENTRY>", "("]
    for index in range(parameter_count):
        if index:
            tokens.append(",")
        depth = parameter_count - index - 1
        tokens.append(f"<ARG{depth}>")
    tokens.extend((")", ":"))
    return tuple(tokens)


def _literal_copy(
    *,
    lexeme: str,
    value: int,
    suffix_character_start: int,
    suffix_character_end: int,
    suffix_token_index: int,
    canonical_token_index: int,
) -> SignedLiteralCopy:
    typed_output = output_cells.TypedOutputCellBank.encode(
        value,
        declared_type=output_cells.OutputType.INT,
    )
    if not typed_output.supported:
        raise ValueError(
            "signed integer lies outside the physical typed-output copy bank"
        )
    active_cells = tuple(int(cell) for cell in np.flatnonzero(typed_output.cells))
    semantic_record = {
        "value": value,
        "suffix_token_index": suffix_token_index,
        "canonical_token_index": canonical_token_index,
        "typed_output_sha256": typed_output.sha256,
        "typed_output_active_cells": active_cells,
    }
    semantic_sha = _config_sha256(_LITERAL_NAMESPACE, semantic_record)
    audit_record = {
        **semantic_record,
        "lexeme": lexeme,
        "suffix_character_start": suffix_character_start,
        "suffix_character_end": suffix_character_end,
    }
    return SignedLiteralCopy(
        lexeme=lexeme,
        value=value,
        suffix_character_start=suffix_character_start,
        suffix_character_end=suffix_character_end,
        suffix_token_index=suffix_token_index,
        canonical_token_index=canonical_token_index,
        typed_output=typed_output,
        typed_output_active_cells=active_cells,
        semantic_sha256=semantic_sha,
        audit_sha256=_config_sha256(_LITERAL_NAMESPACE, audit_record),
    )


def _tokenize_suffix(
    suffix: str,
    *,
    signature: TypedSignature,
    parameter_names: tuple[str, ...],
    frame_token_count: int,
) -> tuple[tuple[str, ...], tuple[SignedLiteralCopy, ...]]:
    lowered_parameters = {
        name.lower(): f"<ARG{len(parameter_names) - index - 1}>"
        for index, name in enumerate(parameter_names)
    }
    lowered_entry = signature.entry_point.lower()
    tokens: list[str] = []
    literals: list[SignedLiteralCopy] = []
    cursor = 0
    for match in _TOKENIZER.finditer(suffix):
        if suffix[cursor : match.start()].strip():
            raise ValueError("task suffix contains text outside the frozen tokenizer")
        raw = match.group(0)
        suffix_token_index = len(tokens)
        canonical_token_index = frame_token_count + suffix_token_index
        if match.lastgroup == "integer":
            tokens.append("<INT>")
            literals.append(
                _literal_copy(
                    lexeme=raw,
                    value=int(raw),
                    suffix_character_start=match.start(),
                    suffix_character_end=match.end(),
                    suffix_token_index=suffix_token_index,
                    canonical_token_index=canonical_token_index,
                )
            )
        elif match.lastgroup == "identifier":
            lowered = raw.lower()
            if lowered in lowered_parameters:
                tokens.append(lowered_parameters[lowered])
            elif lowered == lowered_entry:
                tokens.append("<ENTRY>")
            else:
                tokens.append(lowered)
        else:
            tokens.append(raw)
        cursor = match.end()
    if suffix[cursor:].strip() or not tokens:
        raise ValueError("task suffix contains text outside the frozen tokenizer")
    return tuple(tokens), tuple(literals)


def _span_identity(tokens: tuple[str, ...]) -> str:
    digest = sha256()
    digest.update(_SPAN_IDENTITY_NAMESPACE)
    digest.update(_canonical_json(tokens))
    return digest.hexdigest()


@lru_cache(maxsize=8_192)
def _assembly_cells(identity_sha256: str) -> tuple[int, ...]:
    selected: set[int] = set()
    counter = 0
    identity = bytes.fromhex(identity_sha256)
    while len(selected) < ASSEMBLY_ACTIVE_CELL_COUNT:
        digest = sha256()
        digest.update(_ASSEMBLY_NAMESPACE)
        digest.update(identity)
        digest.update(counter.to_bytes(8, "little"))
        block = digest.digest()
        for offset in range(0, len(block), 4):
            selected.add(
                int.from_bytes(block[offset : offset + 4], "little")
                % ASSEMBLY_CELL_COUNT
            )
            if len(selected) == ASSEMBLY_ACTIVE_CELL_COUNT:
                break
        counter += 1
    return tuple(sorted(selected))


@lru_cache(maxsize=8_192)
def _drive_assembly(
    assembly_cells: tuple[int, ...],
) -> tuple[int, tuple[int, ...], str]:
    if (
        len(assembly_cells) != ASSEMBLY_ACTIVE_CELL_COUNT
        or len(set(assembly_cells)) != ASSEMBLY_ACTIVE_CELL_COUNT
        or not all(0 <= cell < ASSEMBLY_CELL_COUNT for cell in assembly_cells)
    ):
        raise ValueError("invalid sparse span assembly")

    currents = np.zeros(ASSEMBLY_CELL_COUNT, dtype=np.float64)
    currents[np.asarray(assembly_cells, dtype=np.intp)] = ASSEMBLY_CURRENT_PA
    population = AdaptiveGlifPopulation(ASSEMBLY_CELL_COUNT)
    seen = np.zeros(ASSEMBLY_CELL_COUNT, dtype=bool)
    first_spike_step = -1
    trace_digest = sha256()
    trace_digest.update(b"public-train-prompt-span-physical-drive-v1\0")
    trace_digest.update(np.asarray(assembly_cells, dtype=np.int64).tobytes())
    for step in range(ASSEMBLY_DRIVE_STEPS):
        fired = np.asarray(population.step(currents), dtype=bool)
        trace_digest.update(np.packbits(fired, bitorder="little").tobytes())
        if np.any(fired):
            if first_spike_step < 0:
                first_spike_step = step
            seen |= fired
    spike_cells = tuple(int(cell) for cell in np.flatnonzero(seen))
    if first_spike_step < 0 or spike_cells != assembly_cells:
        raise RuntimeError("sparse span assembly failed its physical fitted-cell drive")
    return first_spike_step, spike_cells, trace_digest.hexdigest()


def _span_assembly(
    tokens: tuple[str, ...],
    start_token: int,
    stop_token: int,
) -> PhysicalSpanAssembly:
    span_tokens = tokens[start_token:stop_token]
    identity = _span_identity(span_tokens)
    assembly_cells = _assembly_cells(identity)
    first_step, spike_cells, drive_sha = _drive_assembly(assembly_cells)
    record = {
        "start_token": start_token,
        "stop_token": stop_token,
        "tokens": span_tokens,
        "identity_sha256": identity,
        "assembly_cells": assembly_cells,
        "first_spike_step": first_step,
        "spike_cells": spike_cells,
        "physical_drive_sha256": drive_sha,
        "assembly_projection_sha256": ASSEMBLY_PROJECTION_SHA256,
    }
    return PhysicalSpanAssembly(
        start_token=start_token,
        stop_token=stop_token,
        tokens=span_tokens,
        identity_sha256=identity,
        assembly_cells=assembly_cells,
        first_spike_step=first_step,
        spike_cells=spike_cells,
        physical_drive_sha256=drive_sha,
        sha256=sha256(_canonical_json(record)).hexdigest(),
    )


def observe_public_train_prompt(
    prompt: str,
    signature: TypedSignature,
) -> PromptSpanInput:
    """Project one exactly framed public prompt without reading task outputs."""

    parameter_names = _validate_signature(signature)
    suffix = _extract_task_suffix(prompt, signature, parameter_names)
    frame = _frame_tokens(len(parameter_names))
    suffix_tokens, literals = _tokenize_suffix(
        suffix,
        signature=signature,
        parameter_names=parameter_names,
        frame_token_count=len(frame),
    )
    tokens = (*frame, *suffix_tokens)
    spans = tuple(
        _span_assembly(tokens, start, start + width)
        for start in range(len(tokens))
        for width in range(1, MAX_SPAN_TOKENS + 1)
        if start + width <= len(tokens)
    )

    lexical_record = {
        "tokenizer_sha256": TOKENIZER_SHA256,
        "assembly_projection_sha256": ASSEMBLY_PROJECTION_SHA256,
        "tokens": tokens,
        "spans": tuple(
            {
                "start_token": span.start_token,
                "stop_token": span.stop_token,
                "identity_sha256": span.identity_sha256,
                "assembly_cells": span.assembly_cells,
                "physical_drive_sha256": span.physical_drive_sha256,
            }
            for span in spans
        ),
    }
    lexical_sha = sha256(_canonical_json(lexical_record)).hexdigest()
    typed_output_projection_sha = (
        output_cells.TypedOutputCellBank.projection_sha256()
    )
    literal_record = {
        "typed_output_projection_sha256": typed_output_projection_sha,
        "literals": tuple(literal.semantic_sha256 for literal in literals),
    }
    literal_sha = sha256(_canonical_json(literal_record)).hexdigest()
    input_record = {
        "scope": PUBLIC_TRAIN_INPUT_SCOPE,
        "trainable_state_bytes": TRAINABLE_STATE_BYTES,
        "lexical_sha256": lexical_sha,
        "literal_channel_sha256": literal_sha,
    }
    return PromptSpanInput(
        scope=PUBLIC_TRAIN_INPUT_SCOPE,
        trainable_state_bytes=TRAINABLE_STATE_BYTES,
        audit_prompt_sha256=sha256(prompt.encode("utf-8")).hexdigest(),
        tokenizer_sha256=TOKENIZER_SHA256,
        assembly_projection_sha256=ASSEMBLY_PROJECTION_SHA256,
        typed_output_projection_sha256=typed_output_projection_sha,
        tokens=tokens,
        signed_literals=literals,
        spans=spans,
        lexical_sha256=lexical_sha,
        literal_channel_sha256=literal_sha,
        sha256=sha256(_canonical_json(input_record)).hexdigest(),
    )


__all__ = [
    "ASSEMBLY_ACTIVE_CELL_COUNT",
    "ASSEMBLY_CELL_COUNT",
    "ASSEMBLY_PROJECTION_SHA256",
    "FROZEN_PROMPT_PREFIX",
    "MAX_SPAN_TOKENS",
    "PUBLIC_TRAIN_INPUT_SCOPE",
    "TOKENIZER_SHA256",
    "TRAINABLE_STATE_BYTES",
    "PhysicalSpanAssembly",
    "PromptSpanInput",
    "SignedLiteralCopy",
    "observe_public_train_prompt",
]
