"""Pointer-normalized text adapter for the recurrent assembly stack.

The recurrent assembly stack is the sole output selector.  This module does not
call ``FactorizedPointerCellModel.raw_complete`` or accept candidate programs,
target spans, task labels, tests, types, or Python syntax.  Public symbol
pointerization and lossless byte tokenization are representation adapters around
the neuronal decision; they do not choose its emitted sequence.

The module deliberately implements the small ``PointerLearnerModel`` interface
used by ``PersistentCodingLearner``.  A non-STOP completion remains pending
until exactly one authenticated consequence closes it.  A physical assembly
STOP is neutral-resolved before ``raw_complete`` returns because the persistent
learner correctly does not send a second consequence for an explicit STOP.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from hashlib import sha256

import numpy as np

from humanbrain.factorized_fragment_memory import (
    FactorizedFragmentMemory,
    FragmentMemoryConfig,
    _FragmentSuccess,
    _RoleLoopSpec,
)
from humanbrain.pointer_binding import SymbolInventory, pointerize
from humanbrain.recurrent_assembly_stack import (
    AssemblyAblation,
    AssemblyConsequenceUpdate,
    AssemblyStackConfig,
    AssemblyTrace,
    PendingAssemblyDecision,
    RecurrentAssemblyStack,
    Valence,
)
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

__all__ = [
    "AssemblyFactorizedPointerConfig",
    "AssemblyFactorizedPointerError",
    "AssemblyFactorizedPointerModel",
    "AssemblyPointerCompletionTrace",
    "AssemblyPointerLearningUpdate",
    "AssemblyPointerTrainingExample",
]


_MODEL_SCHEMA = "humanbrain-assembly-factorized-pointer-v2"
_STOP_OUTPUT = "<stop>insufficient recurrent assembly evidence</stop>"
_EXPLICIT_STOP = re.compile(r"\s*<stop>.*?</stop>\s*", re.DOTALL)
_POINTER_EVENT = re.compile(r"<@(\d+)>")
_ROLE_LOOP_MIN_SUPPORT = 2
_ROLE_LOOP_MAX_SEPARATOR_BYTES = 64
_ROLE_LOOP_CAPACITY = 256


def _default_stack_config() -> AssemblyStackConfig:
    return AssemblyStackConfig(
        max_cells=16_384,
        max_examples=256,
        max_programs=4_096,
        max_cue_tokens=4_096,
        max_sequence_tokens=8_192,
        max_depth=8,
        max_steps=65_536,
        max_output_tokens=8_192,
    )


class AssemblyFactorizedPointerError(ValueError):
    """The adapter input, pending query, or decoded neuronal output is invalid."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(namespace: bytes, value: object) -> str:
    payload = namespace + b"\0" + _canonical_json(value).encode("ascii")
    return sha256(payload).hexdigest()


def _require_inventory(inventory: SymbolInventory) -> None:
    if not isinstance(inventory, SymbolInventory):
        raise TypeError("inventory must be a SymbolInventory")
    if SymbolInventory(inventory.symbols).sha256 != inventory.sha256:
        raise AssemblyFactorizedPointerError("symbol inventory authentication failed")


def _query_sha256(
    prompt: str,
    inventory: SymbolInventory,
    cue_text: str,
) -> str:
    return _digest(
        b"humanbrain-assembly-factorized-pointer-query-v1",
        {
            "cue_text": cue_text,
            "inventory_sha256": inventory.sha256,
            "prompt": prompt,
        },
    )


@dataclass(frozen=True, slots=True)
class AssemblyPointerTrainingExample:
    """One public successful text sequence; no spans or structural labels."""

    prompt: str
    output: str
    inventory: SymbolInventory
    cue_text: str
    provenance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be one non-empty string")
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("output must be one non-empty string")
        _require_inventory(self.inventory)
        if not isinstance(self.cue_text, str) or not self.cue_text:
            raise ValueError("cue_text must be one non-empty string")
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("provenance_id must be one non-empty string")


@dataclass(frozen=True, slots=True)
class AssemblyFactorizedPointerConfig:
    """Bounded neuronal, tokenizer, consequence, and causal-ablation policy."""

    stack: AssemblyStackConfig = field(default_factory=_default_stack_config)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    fragments: FragmentMemoryConfig = field(default_factory=FragmentMemoryConfig)
    plasticity: bool = True
    ablation: AssemblyAblation = AssemblyAblation.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.stack, AssemblyStackConfig):
            raise TypeError("stack must be an AssemblyStackConfig")
        if not isinstance(self.tokenizer, TokenizerConfig):
            raise TypeError("tokenizer must be a TokenizerConfig")
        if not isinstance(self.fragments, FragmentMemoryConfig):
            raise TypeError("fragments must be a FragmentMemoryConfig")
        if (
            type(self.tokenizer.max_phrases) is not int
            or self.tokenizer.max_phrases < 0
            or type(self.tokenizer.min_frequency) is not int
            or self.tokenizer.min_frequency < 1
        ):
            raise ValueError("tokenizer bounds are invalid")
        if type(self.plasticity) is not bool:
            raise TypeError("plasticity must be a bool")
        if not isinstance(self.ablation, AssemblyAblation):
            raise TypeError("ablation must be an AssemblyAblation")


@dataclass(frozen=True, slots=True)
class AssemblyPointerCompletionTrace:
    """Custody binding one textual response to its physical assembly races."""

    query_sha256: str
    prior_training_digest: str
    ablation: AssemblyAblation
    cue_token_ids: tuple[int, ...]
    emitted_token_ids: tuple[int, ...]
    raw_output: str
    selected_stop: bool
    stack_trace: AssemblyTrace


@dataclass(frozen=True, slots=True)
class AssemblyPointerLearningUpdate:
    """One exactly-once scalar consequence and optional successful replay."""

    polarity: str
    valence: Valence
    evidence_sha256: str
    changed_edges: tuple[str, ...]
    observed_success_sequence: bool
    prior_training_digest: str
    training_digest: str


@dataclass(slots=True)
class _PendingCompletion:
    query_sha256: str
    cue_token_ids: tuple[int, ...]
    emitted_token_ids: tuple[int, ...]
    decision: PendingAssemblyDecision
    projected: bool


@dataclass(frozen=True, slots=True)
class _RoleLoopOccurrence:
    start: int
    end: int
    start_ordinal: int
    separator: str

    @property
    def key(self) -> tuple[int, str]:
        return self.start_ordinal, self.separator


def _inventory_loop_occurrences(
    pointer_output: str,
    role_count: int,
) -> tuple[_RoleLoopOccurrence, ...]:
    """Discover contiguous ordinal runs ending at the current inventory edge."""

    matches = tuple(_POINTER_EVENT.finditer(pointer_output))
    result: list[_RoleLoopOccurrence] = []
    cursor = 0
    while cursor + 1 < len(matches):
        first = matches[cursor]
        first_ordinal = int(first.group(1))
        second = matches[cursor + 1]
        separator = pointer_output[first.end() : second.start()]
        if (
            int(second.group(1)) != first_ordinal + 1
            or len(separator.encode("utf-8")) > _ROLE_LOOP_MAX_SEPARATOR_BYTES
            or not all(
                character.isspace()
                or (not character.isalnum() and character != "_")
                for character in separator
            )
        ):
            cursor += 1
            continue
        end_index = cursor + 1
        while end_index + 1 < len(matches):
            current = matches[end_index]
            following = matches[end_index + 1]
            if (
                int(following.group(1)) != int(current.group(1)) + 1
                or pointer_output[current.end() : following.start()] != separator
            ):
                break
            end_index += 1
        last = matches[end_index]
        if int(last.group(1)) == role_count - 1:
            result.append(
                _RoleLoopOccurrence(
                    start=first.start(),
                    end=last.end(),
                    start_ordinal=first_ordinal,
                    separator=separator,
                )
            )
            cursor = end_index + 1
        else:
            cursor += 1
    return tuple(result)


def _collapse_inventory_loops(
    tokenizer: BytePhraseTokenizer,
    pointer_outputs: tuple[str, ...],
    inventories: tuple[SymbolInventory, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[_RoleLoopSpec, ...]]:
    """Recruit role-loop markers solely from repeated complete output structure."""

    occurrences = tuple(
        _inventory_loop_occurrences(output, len(inventory.symbols))
        for output, inventory in zip(pointer_outputs, inventories, strict=True)
    )
    support: Counter[tuple[int, str]] = Counter()
    for rows in occurrences:
        support.update({row.key for row in rows})
    supported = tuple(
        sorted(key for key, count in support.items() if count >= _ROLE_LOOP_MIN_SUPPORT)
    )
    marker_by_key = {
        key: tokenizer.vocab_size + 1 + index for index, key in enumerate(supported)
    }
    specs = tuple(
        _RoleLoopSpec(
            marker_token=marker_by_key[key],
            start_ordinal=key[0],
            separator_tokens=tuple(tokenizer.encode(key[1])),
            max_roles=_ROLE_LOOP_CAPACITY,
        )
        for key in supported
    )
    collapsed: list[tuple[int, ...]] = []
    for output, rows in zip(pointer_outputs, occurrences, strict=True):
        encoded: list[int] = []
        cursor = 0
        for row in rows:
            marker = marker_by_key.get(row.key)
            if marker is None:
                continue
            encoded.extend(tokenizer.encode(output[cursor : row.start]))
            encoded.append(marker)
            cursor = row.end
        encoded.extend(tokenizer.encode(output[cursor:]))
        collapsed.append(tuple(encoded))
    return tuple(collapsed), specs


class AssemblyFactorizedPointerModel:
    """Deep adapter from pointer-normalized text to assembly-stack decisions."""

    def __init__(
        self,
        stack: RecurrentAssemblyStack,
        tokenizer: BytePhraseTokenizer,
        config: AssemblyFactorizedPointerConfig,
        fragment_memory: FactorizedFragmentMemory | None = None,
    ) -> None:
        if not isinstance(stack, RecurrentAssemblyStack):
            raise TypeError("stack must be a RecurrentAssemblyStack")
        if not isinstance(tokenizer, BytePhraseTokenizer):
            raise TypeError("tokenizer must be a BytePhraseTokenizer")
        if not isinstance(config, AssemblyFactorizedPointerConfig):
            raise TypeError("config must be an AssemblyFactorizedPointerConfig")
        if fragment_memory is not None and not isinstance(
            fragment_memory,
            FactorizedFragmentMemory,
        ):
            raise TypeError("fragment_memory must be a FactorizedFragmentMemory")
        self._stack = stack
        self._tokenizer = tokenizer
        self._config = config
        self._fragment_memory = fragment_memory
        self._pending: _PendingCompletion | None = None
        self._last_trace: AssemblyPointerCompletionTrace | None = None
        self._last_update: AssemblyPointerLearningUpdate | None = None
        self._training_digest = self._compute_training_digest()

    @classmethod
    def grow(
        cls,
        examples: Iterable[AssemblyPointerTrainingExample],
        config: AssemblyFactorizedPointerConfig | None = None,
    ) -> AssemblyFactorizedPointerModel:
        """Grow from complete public successes without target structure metadata."""

        rows = tuple(examples)
        if not rows:
            raise ValueError("at least one assembly pointer example is required")
        if any(not isinstance(row, AssemblyPointerTrainingExample) for row in rows):
            raise TypeError("examples must contain AssemblyPointerTrainingExample rows")
        cfg = config or AssemblyFactorizedPointerConfig()
        pointer_prompts = tuple(pointerize(row.prompt, row.inventory) for row in rows)
        pointer_cues = tuple(pointerize(row.cue_text, row.inventory) for row in rows)
        pointer_outputs = tuple(pointerize(row.output, row.inventory) for row in rows)
        tokenizer = BytePhraseTokenizer.fit(
            (
                text
                for prompt, cue, output in zip(
                    pointer_prompts,
                    pointer_cues,
                    pointer_outputs,
                    strict=True,
                )
                for text in (prompt, cue, output)
            ),
            cfg.tokenizer,
        )
        collapsed_outputs, role_loops = _collapse_inventory_loops(
            tokenizer,
            pointer_outputs,
            tuple(row.inventory for row in rows),
        )
        fragment_rows = tuple(
            _FragmentSuccess(
                observation_tokens=cls._fragment_observation_tokens(
                    tokenizer,
                    prompt,
                    cue,
                ),
                output_tokens=output_tokens,
            )
            for prompt, cue, output_tokens in zip(
                pointer_prompts,
                pointer_cues,
                collapsed_outputs,
                strict=True,
            )
        )
        fragment_memory = FactorizedFragmentMemory.grow(
            fragment_rows,
            cfg.fragments,
            transport_token_ids=cls._transport_token_ids(tokenizer),
            boundary_token_ids=(tokenizer.vocab_size,),
            role_loops=role_loops,
        )
        stack = RecurrentAssemblyStack(cfg.stack)
        if fragment_memory.has_variable_template:
            stack._register_projection_owner(
                fragment_memory.owner_digest,
                fragment_memory.projected_cell_ids,
            )
        else:
            for cue, output in zip(pointer_cues, pointer_outputs, strict=True):
                stack.observe_success(
                    tokenizer.encode(cue),
                    tokenizer.encode(output),
                )
        return cls(stack, tokenizer, cfg, fragment_memory)

    @property
    def training_digest(self) -> str:
        """Return stable learned-state identity even while inference is pending."""

        return self._training_digest

    @property
    def last_trace(self) -> AssemblyPointerCompletionTrace | None:
        return self._last_trace

    @property
    def last_update(self) -> AssemblyPointerLearningUpdate | None:
        return self._last_update

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocab_size

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        """Select one raw pointer response and leave executable output pending."""

        self._validate_query(prompt, inventory, cue_text)
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self._validate_stop_text(stop_text)
        if self._pending is not None:
            raise RuntimeError("resolve the pending assembly pointer completion first")
        self._last_update = None

        query_sha = _query_sha256(prompt, inventory, cue_text)
        pointer_prompt = pointerize(prompt, inventory)
        pointer_cue = pointerize(cue_text, inventory)
        projected = bool(
            self._fragment_memory is not None
            and self._fragment_memory.has_variable_template
        )
        cue_token_ids = (
            self._fragment_observation_tokens(
                self._tokenizer,
                pointer_prompt,
                pointer_cue,
            )
            if projected
            else tuple(self._tokenizer.encode(pointer_cue))
        )
        prior_digest = self._training_digest
        if projected:
            assert self._fragment_memory is not None
            projection = self._fragment_memory.project(
                cue_token_ids,
                query_sha256=query_sha,
            )
            decision = self._stack._respond_projection(
                projection,
                role_count=len(inventory.symbols),
                ablation=self._config.ablation,
            )
            emitted_fragment_cells = tuple(decision.tokens)
            if decision.trace.selected_stop:
                emitted = ()
            else:
                try:
                    emitted = self._fragment_memory.materialize(
                        emitted_fragment_cells,
                        cue_token_ids,
                    )
                except ValueError as error:
                    decision.resolve(Valence.NEUTRAL)
                    raise AssemblyFactorizedPointerError(
                        "fragment-cell path could not be materialized"
                    ) from error
        else:
            decision = self._stack.respond(
                cue_token_ids,
                ablation=self._config.ablation,
            )
            emitted = tuple(decision.tokens)
        if decision.trace.selected_stop:
            consequence = decision.resolve(Valence.NEUTRAL)
            self._require_neutral_identity(consequence)
            raw_output = _STOP_OUTPUT
        else:
            if len(emitted) > max_new_tokens:
                decision.resolve(Valence.NEUTRAL)
                raise AssemblyFactorizedPointerError(
                    "neuronal output exceeds caller max_new_tokens"
                )
            try:
                raw_output = self._tokenizer.decode_bytes(emitted).decode("utf-8")
            except UnicodeDecodeError as error:
                decision.resolve(Valence.NEUTRAL)
                raise AssemblyFactorizedPointerError(
                    "neuronal output is not lossless UTF-8"
                ) from error
            if _EXPLICIT_STOP.fullmatch(raw_output):
                consequence = decision.resolve(Valence.NEUTRAL)
                self._require_neutral_identity(consequence)
            else:
                self._pending = _PendingCompletion(
                    query_sha256=query_sha,
                    cue_token_ids=cue_token_ids,
                    emitted_token_ids=emitted,
                    decision=decision,
                    projected=projected,
                )

        self._last_trace = AssemblyPointerCompletionTrace(
            query_sha256=query_sha,
            prior_training_digest=prior_digest,
            ablation=self._config.ablation,
            cue_token_ids=cue_token_ids,
            emitted_token_ids=emitted,
            raw_output=raw_output,
            selected_stop=decision.trace.selected_stop,
            stack_trace=decision.trace,
        )
        return raw_output

    def complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        return self.raw_complete(
            prompt,
            inventory,
            cue_text=cue_text,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )

    def observe_failure(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        feedback_text: str,
    ) -> AssemblyPointerLearningUpdate:
        """Resolve with one negative scalar; feedback content never selects output."""

        if not isinstance(feedback_text, str) or not feedback_text:
            raise ValueError("feedback_text must be one non-empty string")
        pending = self._require_pending(prompt, inventory, cue_text)
        return self._resolve_pending(
            pending,
            polarity="failure",
            evidence_text=feedback_text,
        )

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> AssemblyPointerLearningUpdate:
        """Reinforce and observe only the already-emitted sequence that passed."""

        if not isinstance(output, str) or not output:
            raise ValueError("output must be one non-empty string")
        if not np.isfinite(strength) or strength <= 0.0:
            raise ValueError("strength must be one positive finite value")
        pending = self._require_pending(prompt, inventory, cue_text)
        return self._resolve_pending(
            pending,
            polarity="success",
            evidence_text=output,
        )

    def resolve_pending_neutral(
        self,
        prompt: str,
        inventory: SymbolInventory,
        cue_text: str,
    ) -> AssemblyPointerLearningUpdate:
        """Close one exact-query decision without plasticity or replay."""

        pending = self._require_pending(prompt, inventory, cue_text)
        prior_digest = self._training_digest
        consequence = pending.decision.resolve(Valence.NEUTRAL)
        self._pending = None
        self._require_neutral_identity(consequence)
        if self._compute_training_digest() != prior_digest:
            raise RuntimeError("neutral close changed assembly pointer state")
        update = AssemblyPointerLearningUpdate(
            polarity="neutral",
            valence=Valence.NEUTRAL,
            evidence_sha256=pending.query_sha256,
            changed_edges=(),
            observed_success_sequence=False,
            prior_training_digest=prior_digest,
            training_digest=prior_digest,
        )
        self._last_update = update
        return update

    def _resolve_pending(
        self,
        pending: _PendingCompletion,
        *,
        polarity: str,
        evidence_text: str,
    ) -> AssemblyPointerLearningUpdate:
        prior_digest = self._training_digest
        if self._config.plasticity:
            valence = Valence.POSITIVE if polarity == "success" else Valence.NEGATIVE
        else:
            valence = Valence.NEUTRAL
        consequence = pending.decision.resolve(valence)
        self._pending = None
        observed = False
        try:
            if polarity == "success" and self._config.plasticity:
                if pending.projected:
                    if self._fragment_memory is None:
                        raise RuntimeError("projected completion lost fragment ownership")
                    selected_spans = tuple(
                        edge
                        for edge in pending.decision.trace.selected_structural_edges
                        if edge.startswith("projected-span:")
                    )
                    self._fragment_memory.observe_emitted_success(
                        pending.cue_token_ids,
                        selected_spans,
                    )
                    observed = bool(selected_spans)
                else:
                    self._stack.observe_success(
                        pending.cue_token_ids,
                        pending.emitted_token_ids,
                    )
                    observed = True
        finally:
            self._training_digest = self._compute_training_digest()
        update = AssemblyPointerLearningUpdate(
            polarity=polarity,
            valence=valence,
            evidence_sha256=sha256(evidence_text.encode("utf-8")).hexdigest(),
            changed_edges=consequence.changed_edges,
            observed_success_sequence=observed,
            prior_training_digest=prior_digest,
            training_digest=self._training_digest,
        )
        self._last_update = update
        return update

    def _require_pending(
        self,
        prompt: str,
        inventory: SymbolInventory,
        cue_text: str,
    ) -> _PendingCompletion:
        self._validate_query(prompt, inventory, cue_text)
        query_sha = _query_sha256(prompt, inventory, cue_text)
        pending = self._pending
        if pending is None or pending.query_sha256 != query_sha:
            raise AssemblyFactorizedPointerError(
                "consequence is not bound to the pending neuronal query"
            )
        return pending

    @staticmethod
    def _validate_query(
        prompt: str,
        inventory: SymbolInventory,
        cue_text: str,
    ) -> None:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be one non-empty string")
        _require_inventory(inventory)
        if not isinstance(cue_text, str) or not cue_text:
            raise ValueError("cue_text must be one non-empty string")

    @staticmethod
    def _validate_stop_text(
        stop_text: str | tuple[str, ...] | None,
    ) -> None:
        if stop_text is None:
            return
        markers = (stop_text,) if isinstance(stop_text, str) else tuple(stop_text)
        if any(not isinstance(marker, str) or not marker for marker in markers):
            raise ValueError("stop markers must be non-empty strings")

    @staticmethod
    def _fragment_observation_tokens(
        tokenizer: BytePhraseTokenizer,
        pointer_prompt: str,
        pointer_cue: str,
    ) -> tuple[int, ...]:
        prompt_tokens = tuple(tokenizer.encode(pointer_prompt))
        cue_tokens = tuple(tokenizer.encode(pointer_cue))
        return (*prompt_tokens, tokenizer.vocab_size, *cue_tokens)

    @staticmethod
    def _transport_token_ids(tokenizer: BytePhraseTokenizer) -> tuple[int, ...]:
        result: list[int] = []
        for token_id in range(tokenizer.vocab_size):
            try:
                text = tokenizer.token_bytes(token_id).decode("utf-8")
            except UnicodeDecodeError:
                continue
            if text and all(
                character.isspace()
                or (not character.isalnum() and character != "_")
                for character in text
            ):
                result.append(token_id)
        return tuple(result)

    @staticmethod
    def _require_neutral_identity(update: AssemblyConsequenceUpdate) -> None:
        if (
            update.valence is not Valence.NEUTRAL
            or update.changed_edges
            or update.prior_digest != update.model_digest
        ):
            raise RuntimeError("neutral physical STOP changed assembly state")

    def _compute_training_digest(self) -> str:
        return _digest(
            b"humanbrain-assembly-factorized-pointer-model-v1",
            {
                "schema": _MODEL_SCHEMA,
                "stack_digest": self._stack.model_digest,
                "stack_config": asdict(self._config.stack),
                "fragment_config": asdict(self._config.fragments),
                "fragment_memory_digest": (
                    None
                    if self._fragment_memory is None
                    else self._fragment_memory.model_digest
                ),
                "tokenizer_phrases_hex": tuple(
                    phrase.hex() for phrase in self._tokenizer.phrases
                ),
            },
        )
