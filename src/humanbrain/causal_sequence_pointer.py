"""Causal neuronal sequence composition over pointer-normalized text.

The model stores transition synapses, not candidate programs.  Successful
episodes leave exact contextual assemblies that physically replay their learned
atoms.  Repeated structure and sensory
differences isolated by matched successful episodes can both drive a novel
context.  Generation is one strictly forward traversal.  Every emitted chunk,
learned segment, EOS, STOP, live role binding, and role-loop boundary is
selected by a fitted-cell first-spike race.

Failure applies one local NoGo tick to the newest eligible transition keyed by
observation state, predecessor state, and selected chunk.  The receipt is also
retained as sensory state so a later successful episode can associate its
language/tool consequence without the receipt selecting an answer directly.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from hashlib import sha256
from math import isfinite, log1p

from humanbrain.cortex import DendriticCortex
from humanbrain.pointer_binding import SymbolInventory, pointerize
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

__all__ = [
    "CausalSequenceAblation",
    "CausalSequenceCompletionTrace",
    "CausalSequenceConfig",
    "CausalSequenceLearningUpdate",
    "CausalSequencePointerError",
    "CausalSequencePointerModel",
    "CausalSequenceRaceStep",
    "CausalSequenceTrainingExample",
    "RoleCursorSpec",
]


_MODEL_SCHEMA = "humanbrain-causal-sequence-pointer-v3"
_STOP_OUTPUT = "<stop>insufficient causal sequence evidence</stop>"
_POINTER_EVENT = re.compile(r"<@(\d+)>")
_CURRENT_SCALE_PA = 900.0
_CANDIDATE_GAIN = 2.0
_STOP_DRIVE = 1.15
_SEMANTIC_EVIDENCE_FLOOR = 6.2
_SENSORY_ACTIVITY_CAP = 2.5
_SENSORY_COINCIDENCE_GAIN = 0.5
_RECURRENT_DRIVE_BUDGET = 3.0
_MIN_CONTRAST_SHARED_FEATURES = 2
_MAX_CONTRAST_NEIGHBORS = 8

_Feature = tuple[int, ...]
_Prefix = tuple[int, ...]
_Distribution = dict[int, float]


class CausalSequencePointerError(ValueError):
    """The neuronal composer input or pending consequence is invalid."""


class CausalSequenceAblation(Enum):
    """Focused causal controls over learned sequence mechanisms."""

    NONE = "none"
    CONTRAST_OFF = "contrast-off"
    EXACT_ASSEMBLY_OFF = "exact-assembly-off"
    RECURRENCE_OFF = "recurrence-off"
    NOGO_OFF = "nogo-off"


@dataclass(frozen=True, slots=True)
class CausalSequenceTrainingExample:
    """One successful public sequence, without spans or semantic labels."""

    prompt: str
    output: str
    inventory: SymbolInventory
    cue_text: str
    provenance_id: str

    def __post_init__(self) -> None:
        _validate_text(self.prompt, "prompt")
        _validate_text(self.output, "output")
        _require_inventory(self.inventory)
        _validate_text(self.cue_text, "cue_text")
        _validate_text(self.provenance_id, "provenance_id")


@dataclass(frozen=True, slots=True)
class CausalSequenceConfig:
    """Bounded representation, neuronal race, and plasticity policy."""

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    max_observation_order: int = 2
    max_observation_features: int = 256
    max_prefix_order: int = 3
    max_role_cursors: int = 32
    max_roles: int = 256
    max_contrast_output_bytes: int = 4096
    max_contrast_candidate_episodes: int = 256
    min_reuse_contexts: int = 2
    plasticity: bool = True
    ablation: CausalSequenceAblation = CausalSequenceAblation.NONE
    random_seed: int = 29

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, TokenizerConfig):
            raise TypeError("tokenizer must be a TokenizerConfig")
        integer_bounds = (
            ("max_observation_order", self.max_observation_order, 1),
            ("max_observation_features", self.max_observation_features, 1),
            ("max_prefix_order", self.max_prefix_order, 1),
            ("max_role_cursors", self.max_role_cursors, 1),
            ("max_roles", self.max_roles, 2),
            ("max_contrast_output_bytes", self.max_contrast_output_bytes, 1),
            (
                "max_contrast_candidate_episodes",
                self.max_contrast_candidate_episodes,
                1,
            ),
            ("min_reuse_contexts", self.min_reuse_contexts, 2),
        )
        for name, value, lower in integer_bounds:
            if type(value) is not int or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if type(self.plasticity) is not bool:
            raise TypeError("plasticity must be a bool")
        if not isinstance(self.ablation, CausalSequenceAblation):
            raise TypeError("ablation must be a CausalSequenceAblation")
        if type(self.random_seed) is not int:
            raise TypeError("random_seed must be an int")


@dataclass(frozen=True, slots=True)
class RoleCursorSpec:
    """Repeated opaque pointer run represented by one neuronal action cell."""

    action_cell: int
    start_ordinal: int
    separator: bytes


@dataclass(frozen=True, slots=True)
class CausalSequenceRaceStep:
    """Custody for one physical choice and only its earlier dependencies."""

    index: int
    phase: str
    winner_cell: int
    winner_operation: str
    spike_time_ms: float
    candidate_currents_pa: tuple[tuple[int, float], ...]
    depends_on_steps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CausalSequenceCompletionTrace:
    """One forward causal traversal from sensory state to emitted bytes."""

    query_sha256: str
    prior_training_digest: str
    ablation: CausalSequenceAblation
    observation_state: str
    observation_features: tuple[_Feature, ...]
    emitted_token_ids: tuple[int, ...]
    raw_output: str
    selected_stop: bool
    selected_eos: bool
    steps: tuple[CausalSequenceRaceStep, ...]


@dataclass(frozen=True, slots=True)
class CausalSequenceLearningUpdate:
    """One exactly-once consequence applied to one pending traversal."""

    polarity: str
    evidence_sha256: str
    changed_synapses: tuple[str, ...]
    observed_success_sequence: bool
    prior_training_digest: str
    training_digest: str


@dataclass(frozen=True, slots=True)
class _RoleRun:
    start: int
    end: int
    start_ordinal: int
    separator: str

    @property
    def key(self) -> tuple[int, str]:
        return self.start_ordinal, self.separator


@dataclass(frozen=True, slots=True)
class _PreparedExample:
    pointer_prompt: str
    pointer_output: str
    pointer_cue: str
    inventory: SymbolInventory
    provenance_id: str


@dataclass(frozen=True, slots=True)
class _ContrastEpisode:
    episode_id: str
    observation_state: str
    features: tuple[_Feature, ...]
    pointer_output: bytes
    template_cell: int


@dataclass(frozen=True, slots=True)
class _ContrastSegment:
    cell: int
    base_episode_id: str
    start: int
    stop: int
    replacement: bytes


@dataclass(frozen=True, slots=True)
class _ContrastCertificate:
    certificate_id: str
    witness_id: str
    base_episode_id: str
    variant_episode_id: str
    factor_cell: int
    positive_features: tuple[_Feature, ...]
    negative_features: tuple[_Feature, ...]
    segment_cells: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _NoGoKey:
    observation_state: str
    step: int
    predecessor: _Prefix
    selected_token: int


@dataclass(slots=True)
class _PendingCompletion:
    query_sha256: str
    observation_state: str
    observation_features: tuple[_Feature, ...]
    pointer_output: str
    semantic_tokens: tuple[int, ...]
    role_actions: tuple[tuple[int, int], ...]
    eligible_nogo: tuple[_NoGoKey, ...]


@dataclass(frozen=True, slots=True)
class _ContrastCompletion:
    raw_output: str
    emitted_tokens: tuple[int, ...]
    selected_stop: bool
    selected_eos: bool
    eligible_nogo: tuple[_NoGoKey, ...]


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be one non-empty string")


def _require_inventory(inventory: SymbolInventory) -> None:
    if not isinstance(inventory, SymbolInventory):
        raise TypeError("inventory must be a SymbolInventory")
    if SymbolInventory(inventory.symbols).sha256 != inventory.sha256:
        raise CausalSequencePointerError("symbol inventory authentication failed")


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


def _query_sha256(
    prompt: str,
    inventory: SymbolInventory,
    cue_text: str,
) -> str:
    return _digest(
        b"humanbrain-causal-sequence-query-v1",
        {
            "cue_text": cue_text,
            "inventory_sha256": inventory.sha256,
            "prompt": prompt,
        },
    )


def _observation_state(pointer_cue: str) -> str:
    return _digest(
        b"humanbrain-causal-sequence-observation-v1",
        pointer_cue,
    )


def _bounded_contrast_output(pointer_output: str, maximum_bytes: int) -> bytes:
    output_bytes = pointer_output.encode("utf-8")
    if len(output_bytes) > maximum_bytes:
        raise CausalSequencePointerError("contrast output exceeds configured byte capacity")
    return output_bytes


def _contrast_segments_overlap(
    left: _ContrastSegment,
    right: _ContrastSegment,
) -> bool:
    if left.start == right.start:
        return True
    if left.start == left.stop:
        return right.start < left.start < right.stop
    if right.start == right.stop:
        return left.start < right.start < left.stop
    return max(left.start, right.start) < min(left.stop, right.stop)


def _role_runs(pointer_output: str, role_count: int) -> tuple[_RoleRun, ...]:
    """Find ordinal pointer runs ending at the observed inventory boundary."""

    matches = tuple(_POINTER_EVENT.finditer(pointer_output))
    result: list[_RoleRun] = []
    cursor = 0
    while cursor + 1 < len(matches):
        first = matches[cursor]
        first_ordinal = int(first.group(1))
        second = matches[cursor + 1]
        separator = pointer_output[first.end() : second.start()]
        if (
            int(second.group(1)) != first_ordinal + 1
            or len(separator.encode("utf-8")) > 64
            or not all(
                character.isspace() or (not character.isalnum() and character != "_")
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
                _RoleRun(
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


class CausalSequencePointerModel:
    """Sparse continual neuronal composer behind ``PersistentCodingLearner``."""

    def __init__(
        self,
        tokenizer: BytePhraseTokenizer,
        role_keys: Sequence[tuple[int, str]],
        config: CausalSequenceConfig,
    ) -> None:
        if not isinstance(tokenizer, BytePhraseTokenizer):
            raise TypeError("tokenizer must be a BytePhraseTokenizer")
        if not isinstance(config, CausalSequenceConfig):
            raise TypeError("config must be a CausalSequenceConfig")
        if len(role_keys) > config.max_role_cursors:
            raise ValueError("role cursor capacity exceeded")

        self._tokenizer = tokenizer
        self._config = config
        self._eos_cell = tokenizer.vocab_size
        self._stop_cell = self._eos_cell + 1
        self._role_action_base = self._stop_cell + 1
        self._role_bind_base = self._role_action_base + config.max_role_cursors
        self._continue_cell = self._role_bind_base + config.max_roles
        self._exit_cell = self._continue_cell + 1
        self._cortex = DendriticCortex(
            self._exit_cell + 1,
            max_order=1,
            seed=config.random_seed,
        )
        self._role_specs = tuple(
            RoleCursorSpec(
                action_cell=self._role_action_base + index,
                start_ordinal=start,
                separator=separator.encode("utf-8"),
            )
            for index, (start, separator) in enumerate(role_keys)
        )
        self._role_spec_by_key = {
            (spec.start_ordinal, spec.separator.decode("utf-8")): spec for spec in self._role_specs
        }
        self._role_spec_by_cell = {spec.action_cell: spec for spec in self._role_specs}

        # Exact transitions are episodic.  The direct reusable path still
        # requires repeated contexts; the contrast path below is the only way
        # a singleton fragment may transfer to a novel observation.
        self._episodic: dict[tuple[str, int, _Prefix], _Distribution] = defaultdict(dict)
        self._episodic_states: set[str] = set()
        self._sensory: dict[tuple[int, _Feature], _Distribution] = defaultdict(dict)
        self._sensory_contexts: dict[tuple[int, _Feature, int], set[str]] = defaultdict(set)
        self._transitions: dict[_Prefix, _Distribution] = defaultdict(dict)
        self._transition_contexts: dict[tuple[_Prefix, int], set[str]] = defaultdict(set)
        self._positions: dict[int, _Distribution] = defaultdict(dict)
        self._position_contexts: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._role_starts: dict[int, _Distribution] = defaultdict(dict)
        self._role_start_contexts: dict[tuple[int, int], set[str]] = defaultdict(set)
        self._role_successor_offsets: _Distribution = {}
        self._role_successor_contexts: dict[int, set[str]] = defaultdict(set)
        self._role_boundaries: dict[bool, _Distribution] = defaultdict(dict)
        self._role_boundary_contexts: dict[tuple[bool, int], set[str]] = defaultdict(set)
        self._next_cell = self._exit_cell + 1
        self._contrast_episodes: dict[str, _ContrastEpisode] = {}
        self._contrast_episode_postings: dict[_Feature, dict[str, None]] = defaultdict(dict)
        self._contrast_certificates: dict[str, _ContrastCertificate] = {}
        self._contrast_certificate_postings: dict[_Feature, set[str]] = defaultdict(set)
        self._contrast_factor_cells: dict[
            tuple[tuple[_Feature, ...], tuple[_Feature, ...]], int
        ] = {}
        self._contrast_segments: dict[int, _ContrastSegment] = {}
        self._contrast_segment_cells: dict[tuple[str, int, int, bytes], int] = {}
        self._contrast_template_episodes: dict[int, str] = {}
        self._contrast_cell_operations: dict[int, str] = {}
        self._exact_episode_postings: dict[str, dict[str, float]] = defaultdict(dict)
        self._exact_episode_atoms: dict[str, tuple[int, ...]] = {}
        self._nogo: Counter[_NoGoKey] = Counter()
        self._consequences: dict[str, tuple[int, ...]] = {}
        self._pending: _PendingCompletion | None = None
        self._last_trace: CausalSequenceCompletionTrace | None = None
        self._last_update: CausalSequenceLearningUpdate | None = None
        self._training_digest = self._compute_training_digest()

    @classmethod
    def grow(
        cls,
        examples: Iterable[CausalSequenceTrainingExample],
        config: CausalSequenceConfig | None = None,
    ) -> CausalSequencePointerModel:
        """Grow transition synapses from public successes, never answer spans."""

        cfg = config or CausalSequenceConfig()
        raw_rows = tuple(examples)
        if not raw_rows:
            raise ValueError("examples must contain at least one success")
        prepared: list[_PreparedExample] = []
        tokenizer_texts: list[str] = []
        for row in raw_rows:
            prompt = getattr(row, "prompt", None)
            output = getattr(row, "output", None)
            inventory = getattr(row, "inventory", None)
            cue_text = getattr(row, "cue_text", None)
            provenance_id = getattr(row, "provenance_id", None)
            _validate_text(prompt, "prompt")
            _validate_text(output, "output")
            _require_inventory(inventory)
            _validate_text(cue_text, "cue_text")
            _validate_text(provenance_id, "provenance_id")
            pointer_prompt = pointerize(prompt, inventory)
            pointer_output = pointerize(output, inventory)
            pointer_cue = pointerize(cue_text, inventory)
            _bounded_contrast_output(
                pointer_output,
                cfg.max_contrast_output_bytes,
            )
            prepared.append(
                _PreparedExample(
                    pointer_prompt=pointer_prompt,
                    pointer_output=pointer_output,
                    pointer_cue=pointer_cue,
                    inventory=inventory,
                    provenance_id=provenance_id,
                )
            )
            tokenizer_texts.extend((pointer_prompt, pointer_output, pointer_cue))

        tokenizer = BytePhraseTokenizer.fit(tokenizer_texts, cfg.tokenizer)
        role_support: Counter[tuple[int, str]] = Counter()
        for row in prepared:
            role_support.update(
                {
                    run.key
                    for run in _role_runs(
                        row.pointer_output,
                        len(row.inventory.symbols),
                    )
                }
            )
        role_keys = tuple(
            sorted(key for key, count in role_support.items() if count >= cfg.min_reuse_contexts)[
                : cfg.max_role_cursors
            ]
        )
        model = cls(tokenizer, role_keys, cfg)
        for row in prepared:
            state = _observation_state(row.pointer_cue)
            features = model._observation_features(
                row.pointer_prompt,
                row.pointer_cue,
                state,
            )
            tokens = model._encode_output(
                row.pointer_output,
                len(row.inventory.symbols),
            )
            model._ingest_success(state, features, tokens, strength=1.0)
            for run in _role_runs(row.pointer_output, len(row.inventory.symbols)):
                spec = model._role_spec_by_key.get(run.key)
                if spec is not None:
                    model._ingest_role_cursor(
                        state,
                        spec,
                        len(row.inventory.symbols),
                        strength=1.0,
                    )
            model._register_contrast_episode(
                state,
                model._contrast_cue_features(row.pointer_cue),
                row.pointer_output,
                row.provenance_id,
                atoms=model._tokenizer.encode(row.pointer_output),
                strength=1.0,
            )
        model._training_digest = model._compute_training_digest()
        return model

    @property
    def training_digest(self) -> str:
        """Stable identity of learned synapses, excluding pending inference."""

        return self._training_digest

    @property
    def last_trace(self) -> CausalSequenceCompletionTrace | None:
        return self._last_trace

    @property
    def last_update(self) -> CausalSequenceLearningUpdate | None:
        return self._last_update

    @property
    def role_cursors(self) -> tuple[RoleCursorSpec, ...]:
        return self._role_specs

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        """Run one forward neuronal traversal and leave an action pending."""

        _validate_text(prompt, "prompt")
        _require_inventory(inventory)
        _validate_text(cue_text, "cue_text")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self._validate_stop_text(stop_text)
        if len(inventory.symbols) > self._config.max_roles:
            raise CausalSequencePointerError("live role count exceeds neuronal capacity")
        if self._pending is not None:
            raise RuntimeError("resolve the pending causal sequence completion first")

        self._last_update = None
        query_sha = _query_sha256(prompt, inventory, cue_text)
        pointer_prompt = pointerize(prompt, inventory)
        pointer_cue = pointerize(cue_text, inventory)
        state = _observation_state(pointer_cue)
        features = self._observation_features(pointer_prompt, pointer_cue, state)
        contrast_features = self._contrast_cue_features(pointer_cue)
        prior_digest = self._training_digest
        steps: list[CausalSequenceRaceStep] = []
        semantic_step_indices: list[int] = []
        semantic_tokens: list[int] = []
        output_pieces: list[bytes] = []
        role_actions: list[tuple[int, int]] = []
        eligible: list[_NoGoKey] = []
        selected_stop = False
        selected_eos = False

        completion = self._exact_complete(
            state,
            steps,
            max_new_tokens=max_new_tokens,
        )
        if completion is None and state not in self._episodic_states:
            completion = self._contrast_complete(
                state,
                contrast_features,
                steps,
                max_new_tokens=max_new_tokens,
            )
        if completion is not None:
            raw_output = completion.raw_output
            semantic_tokens = list(completion.emitted_tokens)
            selected_stop = completion.selected_stop
            selected_eos = completion.selected_eos
            eligible = list(completion.eligible_nogo)
            if not selected_stop:
                self._pending = _PendingCompletion(
                    query_sha256=query_sha,
                    observation_state=state,
                    observation_features=features,
                    pointer_output=raw_output,
                    semantic_tokens=tuple(semantic_tokens),
                    role_actions=(),
                    eligible_nogo=tuple(eligible),
                )
            self._last_trace = CausalSequenceCompletionTrace(
                query_sha256=query_sha,
                prior_training_digest=prior_digest,
                ablation=self._config.ablation,
                observation_state=state,
                observation_features=features,
                emitted_token_ids=tuple(semantic_tokens),
                raw_output=raw_output,
                selected_stop=selected_stop,
                selected_eos=selected_eos,
                steps=tuple(steps),
            )
            return raw_output

        for output_step in range(max_new_tokens + 1):
            predecessor = tuple(semantic_tokens[-self._config.max_prefix_order :])
            scores = self._candidate_scores(
                state,
                output_step,
                predecessor,
                features,
            )
            depends = tuple(semantic_step_indices[-self._config.max_prefix_order :])
            winner, positive_targets = self._semantic_race(
                scores,
                steps,
                depends_on=depends,
                phase="semantic",
            )
            semantic_step_indices.append(len(steps) - 1)
            if winner == self._stop_cell:
                selected_stop = True
                break
            if winner == self._eos_cell:
                selected_eos = True
                break
            if output_step == max_new_tokens:
                raise CausalSequencePointerError("neuronal output exceeds caller max_new_tokens")

            if len(positive_targets) > 1:
                eligible.append(
                    _NoGoKey(
                        observation_state=state,
                        step=output_step,
                        predecessor=predecessor,
                        selected_token=winner,
                    )
                )
            semantic_tokens.append(winner)
            role_spec = self._role_spec_by_cell.get(winner)
            if role_spec is None:
                output_pieces.append(self._tokenizer.token_bytes(winner))
                continue
            role_bytes = self._run_role_cursor(
                role_spec,
                len(inventory.symbols),
                steps,
                action_step=semantic_step_indices[-1],
            )
            if role_bytes is None:
                selected_stop = True
                output_pieces.clear()
                break
            role_actions.append((role_spec.action_cell, len(inventory.symbols)))
            output_pieces.append(role_bytes)
            # The role action is not complete until its live EXIT wins.
            # Subsequent semantic state therefore depends on that boundary,
            # never on a planning traversal or a future cursor state.
            semantic_step_indices[-1] = len(steps) - 1

        if selected_stop:
            raw_output = _STOP_OUTPUT
            semantic_tokens = []
        else:
            if not selected_eos:
                raise RuntimeError("forward traversal ended without a physical STOP or EOS")
            try:
                raw_output = b"".join(output_pieces).decode("utf-8")
            except UnicodeDecodeError as error:
                raise CausalSequencePointerError("neuronal output is not lossless UTF-8") from error
            if not raw_output:
                raise RuntimeError("EOS won before any neuronal output was emitted")

        if not selected_stop:
            if not eligible and semantic_tokens:
                output_step = len(semantic_tokens) - 1
                predecessor = tuple(
                    semantic_tokens[
                        max(0, output_step - self._config.max_prefix_order) : output_step
                    ]
                )
                eligible.append(
                    _NoGoKey(
                        observation_state=state,
                        step=output_step,
                        predecessor=predecessor,
                        selected_token=semantic_tokens[-1],
                    )
                )
            self._pending = _PendingCompletion(
                query_sha256=query_sha,
                observation_state=state,
                observation_features=features,
                pointer_output=raw_output,
                semantic_tokens=tuple(semantic_tokens),
                role_actions=tuple(role_actions),
                eligible_nogo=tuple(eligible),
            )

        self._last_trace = CausalSequenceCompletionTrace(
            query_sha256=query_sha,
            prior_training_digest=prior_digest,
            ablation=self._config.ablation,
            observation_state=state,
            observation_features=features,
            emitted_token_ids=tuple(semantic_tokens),
            raw_output=raw_output,
            selected_stop=selected_stop,
            selected_eos=selected_eos,
            steps=tuple(steps),
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
    ) -> CausalSequenceLearningUpdate:
        """Apply one delayed contextual NoGo and retain receipt sensory cells."""

        _validate_text(feedback_text, "feedback_text")
        pending = self._require_pending(prompt, inventory, cue_text)
        prior_digest = self._training_digest
        changed: list[str] = []
        if self._config.plasticity:
            self._consequences[pending.observation_state] = tuple(
                self._tokenizer.encode(feedback_text)
            )
            changed.append(f"consequence:{pending.observation_state[:16]}")
            if (
                self._config.ablation is not CausalSequenceAblation.NOGO_OFF
                and pending.eligible_nogo
            ):
                key = pending.eligible_nogo[-1]
                self._nogo[key] += 1
                changed.append(self._nogo_label(key))
        self._pending = None
        self._training_digest = self._compute_training_digest()
        update = CausalSequenceLearningUpdate(
            polarity="failure",
            evidence_sha256=_digest(
                b"humanbrain-causal-sequence-failure-v1",
                feedback_text,
            ),
            changed_synapses=tuple(changed),
            observed_success_sequence=False,
            prior_training_digest=prior_digest,
            training_digest=self._training_digest,
        )
        self._last_update = update
        return update

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> CausalSequenceLearningUpdate:
        """Consolidate one observed successful sequence without replaying old rows."""

        _validate_text(output, "output")
        if not isfinite(strength) or strength <= 0.0:
            raise ValueError("strength must be one positive finite value")
        pending = self._require_pending(prompt, inventory, cue_text)
        pointer_output = pointerize(output, inventory)
        if pointer_output != pending.pointer_output:
            raise CausalSequencePointerError(
                "successful output does not match pending neuronal emission"
            )
        _bounded_contrast_output(
            pointer_output,
            self._config.max_contrast_output_bytes,
        )
        prior_digest = self._training_digest
        changed: list[str] = []
        if self._config.plasticity:
            observed_tokens = self._encode_output(
                pointer_output,
                len(inventory.symbols),
            )
            self._ingest_success(
                pending.observation_state,
                pending.observation_features,
                observed_tokens,
                strength=float(strength),
            )
            for run in _role_runs(pointer_output, len(inventory.symbols)):
                spec = self._role_spec_by_key.get(run.key)
                if spec is not None:
                    self._ingest_role_cursor(
                        pending.observation_state,
                        spec,
                        len(inventory.symbols),
                        strength=float(strength),
                    )
            provenance = _digest(
                b"humanbrain-causal-sequence-online-success-v2",
                {
                    "query_sha256": pending.query_sha256,
                    "output": pointer_output,
                },
            )
            self._register_contrast_episode(
                pending.observation_state,
                self._contrast_cue_features(pointerize(cue_text, inventory)),
                pointer_output,
                provenance,
                atoms=self._tokenizer.encode(pointer_output),
                strength=float(strength),
            )
            self._consequences[pending.observation_state] = tuple(self._tokenizer.encode(output))
            changed.extend(
                (
                    f"success:{pending.observation_state[:16]}",
                    f"consequence:{pending.observation_state[:16]}",
                )
            )
        self._pending = None
        self._training_digest = self._compute_training_digest()
        update = CausalSequenceLearningUpdate(
            polarity="success",
            evidence_sha256=_digest(
                b"humanbrain-causal-sequence-success-v1",
                output,
            ),
            changed_synapses=tuple(changed),
            observed_success_sequence=bool(changed),
            prior_training_digest=prior_digest,
            training_digest=self._training_digest,
        )
        self._last_update = update
        return update

    def resolve_pending_neutral(
        self,
        prompt: str,
        inventory: SymbolInventory,
        cue_text: str,
    ) -> CausalSequenceLearningUpdate:
        """Close exactly one inference with no state mutation."""

        pending = self._require_pending(prompt, inventory, cue_text)
        del pending
        prior_digest = self._training_digest
        self._pending = None
        update = CausalSequenceLearningUpdate(
            polarity="neutral",
            evidence_sha256=_query_sha256(prompt, inventory, cue_text),
            changed_synapses=(),
            observed_success_sequence=False,
            prior_training_digest=prior_digest,
            training_digest=prior_digest,
        )
        self._last_update = update
        return update

    def _observation_features(
        self,
        pointer_prompt: str,
        pointer_cue: str,
        state: str,
    ) -> tuple[_Feature, ...]:
        ids = [
            *self._tokenizer.encode(pointer_prompt),
            0,
            *self._tokenizer.encode(pointer_cue),
        ]
        consequence = self._consequences.get(state)
        if consequence:
            ids.extend((0, *consequence))
        features = {
            tuple(ids[start : start + order])
            for order in range(1, self._config.max_observation_order + 1)
            for start in range(0, len(ids) - order + 1)
        }
        ordered = sorted(
            features,
            key=lambda feature: (
                0 if len(feature) == 1 else 1,
                _digest(b"humanbrain-observation-feature-v1", feature),
            ),
        )
        return tuple(ordered[: self._config.max_observation_features])

    def _contrast_cue_features(self, pointer_cue: str) -> tuple[_Feature, ...]:
        """Expose only sensory cue traces to matched-pair recruitment."""

        ids = self._tokenizer.encode(pointer_cue)
        features = {
            tuple(ids[start : start + order])
            for order in range(1, self._config.max_observation_order + 1)
            for start in range(0, len(ids) - order + 1)
        }
        ordered = sorted(
            features,
            key=lambda feature: (
                0 if len(feature) == 1 else 1,
                _digest(b"humanbrain-contrast-cue-feature-v1", feature),
            ),
        )
        return tuple(ordered[: self._config.max_observation_features])

    def _encode_output(self, pointer_output: str, role_count: int) -> tuple[int, ...]:
        encoded: list[int] = []
        cursor = 0
        for run in _role_runs(pointer_output, role_count):
            spec = self._role_spec_by_key.get(run.key)
            if spec is None:
                continue
            encoded.extend(self._tokenizer.encode(pointer_output[cursor : run.start]))
            encoded.append(spec.action_cell)
            cursor = run.end
        encoded.extend(self._tokenizer.encode(pointer_output[cursor:]))
        return tuple(encoded)

    def _ingest_success(
        self,
        state: str,
        features: tuple[_Feature, ...],
        tokens: Sequence[int],
        *,
        strength: float,
    ) -> None:
        self._episodic_states.add(state)
        sequence = (*tokens, self._eos_cell)
        prefix: list[int] = []
        for step, target in enumerate(sequence):
            predecessor = tuple(prefix[-self._config.max_prefix_order :])
            episodic = self._episodic[(state, step, predecessor)]
            episodic[target] = episodic.get(target, 0.0) + strength
            for feature in features:
                distribution = self._sensory[(step, feature)]
                distribution[target] = distribution.get(target, 0.0) + strength
                self._sensory_contexts[(step, feature, target)].add(state)
            transitions = self._transitions[predecessor]
            transitions[target] = transitions.get(target, 0.0) + strength
            self._transition_contexts[(predecessor, target)].add(state)
            positions = self._positions[step]
            positions[target] = positions.get(target, 0.0) + strength
            self._position_contexts[(step, target)].add(state)
            if target != self._eos_cell:
                prefix.append(target)

    def _allocate_contrast_cell(self, operation: str) -> int:
        cell = self._next_cell
        self._next_cell += 1
        self._contrast_cell_operations[cell] = operation
        return cell

    def _resize_cortex(self) -> None:
        # The cortex has no hidden learned weights.  Sparse synapse banks live
        # on this model, so recruiting cells only expands the physical race
        # substrate and preserves every learned edge.
        self._cortex = DendriticCortex(
            self._next_cell,
            max_order=1,
            seed=self._config.random_seed,
        )

    def _register_contrast_episode(
        self,
        state: str,
        features: tuple[_Feature, ...],
        pointer_output: str,
        provenance_id: str,
        *,
        atoms: Sequence[int],
        strength: float,
    ) -> None:
        del provenance_id
        ordered_features = tuple(sorted(set(features)))
        output_bytes = _bounded_contrast_output(
            pointer_output,
            self._config.max_contrast_output_bytes,
        )
        episode_id = _digest(
            b"humanbrain-causal-sequence-contrast-episode-v2",
            {
                "features": ordered_features,
                "observation_state": state,
                "pointer_output_hex": output_bytes.hex(),
            },
        )
        existing = self._contrast_episodes.get(episode_id)
        if existing is not None:
            self._register_exact_episode(state, existing, atoms, strength)
            return

        candidates = self._bounded_contrast_candidates(ordered_features)
        feature_set = set(ordered_features)
        matched: list[tuple[int, int, str]] = []
        for candidate_id in candidates:
            candidate = self._contrast_episodes[candidate_id]
            candidate_features = set(candidate.features)
            common = len(feature_set & candidate_features)
            positive = feature_set - candidate_features
            negative = candidate_features - feature_set
            if (
                common < _MIN_CONTRAST_SHARED_FEATURES
                or not positive
                or not negative
                or candidate.pointer_output == output_bytes
            ):
                continue
            matched.append((len(positive) + len(negative), -common, candidate_id))

        template_cell = self._allocate_contrast_cell(f"TEMPLATE:{episode_id[:12]}")
        episode = _ContrastEpisode(
            episode_id=episode_id,
            observation_state=state,
            features=ordered_features,
            pointer_output=output_bytes,
            template_cell=template_cell,
        )
        self._contrast_episodes[episode_id] = episode
        self._contrast_template_episodes[template_cell] = episode_id
        self._register_exact_episode(state, episode, atoms, strength)

        if matched:
            best_distance = min(row[0] for row in matched)
            nearest = sorted(row for row in matched if row[0] == best_distance)[
                :_MAX_CONTRAST_NEIGHBORS
            ]
            for _distance, _negative_common, candidate_id in nearest:
                candidate = self._contrast_episodes[candidate_id]
                self._register_contrast_direction(candidate, episode)
                self._register_contrast_direction(episode, candidate)

        for feature in ordered_features:
            posting = self._contrast_episode_postings[feature]
            posting.pop(episode_id, None)
            posting[episode_id] = None
            while len(posting) > self._config.max_contrast_candidate_episodes:
                del posting[next(iter(posting))]
        self._resize_cortex()

    def _register_exact_episode(
        self,
        state: str,
        episode: _ContrastEpisode,
        atoms: Sequence[int],
        strength: float,
    ) -> None:
        encoded = tuple(int(atom) for atom in atoms)
        known = self._exact_episode_atoms.setdefault(episode.episode_id, encoded)
        if known != encoded:
            raise RuntimeError("one learned episode cannot own two encoded sequences")
        posting = self._exact_episode_postings[state]
        posting[episode.episode_id] = posting.get(episode.episode_id, 0.0) + strength

    def _bounded_contrast_candidates(
        self,
        ordered_features: tuple[_Feature, ...],
    ) -> tuple[str, ...]:
        """Select a deterministic recent union without blocking online growth.

        Each insertion-ordered feature posting is itself recency-bounded.  Pair
        search walks sorted cue features and each posting newest-first, retaining
        at most the configured number of distinct episode candidates.
        """

        candidates: dict[str, None] = {}
        capacity = self._config.max_contrast_candidate_episodes
        for feature in ordered_features:
            for candidate_id in reversed(self._contrast_episode_postings.get(feature, {})):
                candidates[candidate_id] = None
                if len(candidates) == capacity:
                    return tuple(candidates)
        return tuple(candidates)

    def _register_contrast_direction(
        self,
        base: _ContrastEpisode,
        variant: _ContrastEpisode,
    ) -> None:
        base_features = set(base.features)
        variant_features = set(variant.features)
        positive = tuple(sorted(variant_features - base_features))
        negative = tuple(sorted(base_features - variant_features))
        if not positive or not negative:
            return

        opcodes = SequenceMatcher(
            None,
            base.pointer_output,
            variant.pointer_output,
            autojunk=False,
        ).get_opcodes()
        edits = tuple(
            (start, stop, variant.pointer_output[replacement_start:replacement_stop])
            for tag, start, stop, replacement_start, replacement_stop in opcodes
            if tag != "equal"
        )
        if not edits:
            return
        witness_id = _digest(
            b"humanbrain-causal-sequence-contrast-witness-v2",
            sorted((base.episode_id, variant.episode_id)),
        )
        certificate_id = _digest(
            b"humanbrain-causal-sequence-contrast-certificate-v2",
            {
                "base": base.episode_id,
                "edits": [[start, stop, replacement.hex()] for start, stop, replacement in edits],
                "negative": negative,
                "positive": positive,
                "variant": variant.episode_id,
                "witness": witness_id,
            },
        )
        if certificate_id in self._contrast_certificates:
            return

        factor_key = (positive, negative)
        factor_cell = self._contrast_factor_cells.get(factor_key)
        if factor_cell is None:
            factor_cell = self._allocate_contrast_cell(f"FACTOR:{certificate_id[:12]}")
            self._contrast_factor_cells[factor_key] = factor_cell

        segment_cells: list[int] = []
        for start, stop, replacement in edits:
            segment_key = (base.episode_id, start, stop, replacement)
            segment_cell = self._contrast_segment_cells.get(segment_key)
            if segment_cell is None:
                segment_cell = self._allocate_contrast_cell(
                    f"SEGMENT:{certificate_id[:12]}:{start}:{stop}"
                )
                self._contrast_segment_cells[segment_key] = segment_cell
                self._contrast_segments[segment_cell] = _ContrastSegment(
                    cell=segment_cell,
                    base_episode_id=base.episode_id,
                    start=start,
                    stop=stop,
                    replacement=replacement,
                )
            segment_cells.append(segment_cell)

        certificate = _ContrastCertificate(
            certificate_id=certificate_id,
            witness_id=witness_id,
            base_episode_id=base.episode_id,
            variant_episode_id=variant.episode_id,
            factor_cell=factor_cell,
            positive_features=positive,
            negative_features=negative,
            segment_cells=tuple(segment_cells),
        )
        self._contrast_certificates[certificate_id] = certificate
        for feature in positive:
            self._contrast_certificate_postings[feature].add(certificate_id)

    def _contrast_activity(
        self,
        features: tuple[_Feature, ...],
    ) -> dict[str, float]:
        active_features = set(features)
        candidate_ids: set[str] = set()
        for feature in active_features:
            candidate_ids.update(self._contrast_certificate_postings.get(feature, ()))

        activity: dict[str, float] = {}
        for certificate_id in candidate_ids:
            certificate = self._contrast_certificates[certificate_id]
            if not all(
                feature in active_features for feature in certificate.positive_features
            ) or any(feature in active_features for feature in certificate.negative_features):
                continue
            activity[certificate_id] = 1.0
        return activity

    def _exact_complete(
        self,
        state: str,
        steps: list[CausalSequenceRaceStep],
        *,
        max_new_tokens: int,
    ) -> _ContrastCompletion | None:
        postings = self._exact_episode_postings.get(state)
        if not postings:
            return None

        if self._config.ablation is CausalSequenceAblation.EXACT_ASSEMBLY_OFF:
            self._contrast_race(
                {},
                steps,
                depends_on=(),
                phase="exact-episode",
            )
            return self._contrast_stop()

        scores: dict[int, float] = {}
        keys: dict[int, _NoGoKey] = {}
        for episode_id, strength in postings.items():
            episode = self._contrast_episodes[episode_id]
            key = _NoGoKey(
                observation_state=state,
                step=-1,
                predecessor=(),
                selected_token=episode.template_cell,
            )
            keys[episode.template_cell] = key
            if (
                self._config.ablation is CausalSequenceAblation.NOGO_OFF
                or self._nogo.get(key, 0) == 0
            ):
                scores[episode.template_cell] = strength

        winner = self._contrast_race(
            scores,
            steps,
            depends_on=(),
            phase="exact-episode",
        )
        if winner == self._stop_cell:
            return self._contrast_stop()
        episode_id = self._contrast_template_episodes.get(winner)
        if episode_id is None:
            raise RuntimeError("exact episode race selected an unknown cell")

        atoms = self._exact_episode_atoms[episode_id]
        emitted: list[int] = []
        pieces: list[bytes] = []
        dependency = len(steps) - 1
        for atom in atoms:
            if len(emitted) >= max_new_tokens:
                raise CausalSequencePointerError("neuronal output exceeds caller max_new_tokens")
            selected = self._contrast_race(
                {atom: 1.0},
                steps,
                depends_on=(dependency,),
                phase="exact-atom",
            )
            dependency = len(steps) - 1
            if selected == self._stop_cell:
                return self._contrast_stop()
            emitted.append(selected)
            pieces.append(self._tokenizer.token_bytes(selected))

        winner_eos = self._contrast_race(
            {self._eos_cell: 1.0},
            steps,
            depends_on=(dependency,),
            phase="exact-eos",
        )
        if winner_eos != self._eos_cell:
            return self._contrast_stop()
        try:
            raw_output = b"".join(pieces).decode("utf-8")
        except UnicodeDecodeError as error:
            raise CausalSequencePointerError("neuronal output is not lossless UTF-8") from error
        if not raw_output:
            raise RuntimeError("exact EOS won before any neuronal output was emitted")
        return _ContrastCompletion(
            raw_output=raw_output,
            emitted_tokens=tuple(emitted),
            selected_stop=False,
            selected_eos=True,
            eligible_nogo=(keys[winner],),
        )

    def _contrast_complete(
        self,
        state: str,
        features: tuple[_Feature, ...],
        steps: list[CausalSequenceRaceStep],
        *,
        max_new_tokens: int,
    ) -> _ContrastCompletion | None:
        activity = self._contrast_activity(features)
        if not activity:
            return None

        if self._config.ablation is CausalSequenceAblation.CONTRAST_OFF:
            self._contrast_race(
                {},
                steps,
                depends_on=(),
                phase="contrast-template",
            )
            return self._contrast_stop()

        factor_scores: dict[int, float] = {}
        for certificate_id, score in activity.items():
            factor_cell = self._contrast_certificates[certificate_id].factor_cell
            factor_scores[factor_cell] = max(factor_scores.get(factor_cell, 0.0), score)
        factor_steps: dict[int, int] = {}
        for factor_cell, score in sorted(factor_scores.items()):
            winner = self._contrast_race(
                {factor_cell: score},
                steps,
                depends_on=(),
                phase="contrast-factor",
                normalize=False,
            )
            if winner == factor_cell:
                factor_steps[factor_cell] = len(steps) - 1

        activity = {
            certificate_id: score
            for certificate_id, score in activity.items()
            if self._contrast_certificates[certificate_id].factor_cell in factor_steps
        }
        template_scores: defaultdict[int, float] = defaultdict(float)
        for certificate_id, score in activity.items():
            certificate = self._contrast_certificates[certificate_id]
            episode = self._contrast_episodes[certificate.base_episode_id]
            template_scores[episode.template_cell] += score
        winner = self._contrast_race(
            dict(template_scores),
            steps,
            depends_on=tuple(sorted(factor_steps.values())),
            phase="contrast-template",
        )
        if winner == self._stop_cell:
            return self._contrast_stop()
        episode_id = self._contrast_template_episodes.get(winner)
        if episode_id is None:
            raise RuntimeError("contrast template race selected an unknown cell")
        return self._render_contrast_template(
            self._contrast_episodes[episode_id],
            state,
            activity,
            factor_steps,
            steps,
            max_new_tokens=max_new_tokens,
        )

    def _render_contrast_template(
        self,
        episode: _ContrastEpisode,
        state: str,
        activity: dict[str, float],
        factor_steps: dict[int, int],
        steps: list[CausalSequenceRaceStep],
        *,
        max_new_tokens: int,
    ) -> _ContrastCompletion:
        segment_scores: dict[int, dict[int, float]] = defaultdict(dict)
        segment_factor_steps: dict[int, set[int]] = defaultdict(set)
        for certificate_id, score in activity.items():
            certificate = self._contrast_certificates[certificate_id]
            if certificate.base_episode_id != episode.episode_id:
                continue
            factor_step = factor_steps[certificate.factor_cell]
            for segment_cell in certificate.segment_cells:
                segment = self._contrast_segments[segment_cell]
                by_cell = segment_scores[segment.start]
                by_cell[segment_cell] = by_cell.get(segment_cell, 0.0) + score
                segment_factor_steps[segment_cell].add(factor_step)

        active_segments = tuple(
            self._contrast_segments[cell]
            for cell in sorted(cell for scores in segment_scores.values() for cell in scores)
        )
        if any(
            _contrast_segments_overlap(left, right)
            for index, left in enumerate(active_segments)
            for right in active_segments[index + 1 :]
        ):
            self._contrast_race(
                {},
                steps,
                depends_on=tuple(
                    sorted(
                        {
                            len(steps) - 1,
                            *(
                                factor_step
                                for factor_steps in segment_factor_steps.values()
                                for factor_step in factor_steps
                            ),
                        }
                    )
                ),
                phase="contrast-conflict",
            )
            return self._contrast_stop()

        cursor = 0
        processed_starts: set[int] = set()
        emitted: list[int] = []
        eligible: list[_NoGoKey] = []
        dependency = len(steps) - 1
        base = episode.pointer_output
        while cursor <= len(base):
            if any(start < cursor for start in set(segment_scores) - processed_starts):
                self._contrast_race(
                    {},
                    steps,
                    depends_on=(dependency,),
                    phase="contrast-segment",
                )
                return self._contrast_stop()

            at_cursor = segment_scores.get(cursor)
            if at_cursor is not None and cursor not in processed_starts:
                candidate_scores: dict[int, float] = {}
                candidate_keys: dict[int, _NoGoKey] = {}
                for segment_cell, score in at_cursor.items():
                    key = _NoGoKey(
                        observation_state=state,
                        step=cursor,
                        predecessor=(episode.template_cell,),
                        selected_token=segment_cell,
                    )
                    candidate_keys[segment_cell] = key
                    if self._config.ablation is not CausalSequenceAblation.RECURRENCE_OFF and (
                        self._config.ablation is CausalSequenceAblation.NOGO_OFF
                        or self._nogo.get(key, 0) == 0
                    ):
                        candidate_scores[segment_cell] = score
                segment_dependencies = {dependency}
                for segment_cell in at_cursor:
                    segment_dependencies.update(segment_factor_steps[segment_cell])
                winner = self._contrast_race(
                    candidate_scores,
                    steps,
                    depends_on=tuple(sorted(segment_dependencies)),
                    phase="contrast-segment",
                )
                dependency = len(steps) - 1
                if winner == self._stop_cell:
                    return self._contrast_stop()
                segment = self._contrast_segments[winner]
                eligible.append(candidate_keys[winner])
                processed_starts.add(cursor)
                for byte in segment.replacement:
                    if len(emitted) >= max_new_tokens:
                        raise CausalSequencePointerError(
                            "neuronal output exceeds caller max_new_tokens"
                        )
                    winner_byte = self._contrast_race(
                        {byte: 1.0},
                        steps,
                        depends_on=(dependency,),
                        phase="contrast-segment-byte",
                    )
                    dependency = len(steps) - 1
                    if winner_byte == self._stop_cell:
                        return self._contrast_stop()
                    emitted.append(byte)
                if segment.stop > cursor:
                    cursor = segment.stop
                    continue

            if cursor == len(base):
                break
            if len(emitted) >= max_new_tokens:
                raise CausalSequencePointerError("neuronal output exceeds caller max_new_tokens")
            byte_scores = (
                {base[cursor]: 1.0}
                if self._config.ablation is not CausalSequenceAblation.RECURRENCE_OFF
                else {}
            )
            winner = self._contrast_race(
                byte_scores,
                steps,
                depends_on=(dependency,),
                phase="contrast-template-byte",
            )
            dependency = len(steps) - 1
            if winner == self._stop_cell:
                return self._contrast_stop()
            emitted.append(winner)
            cursor += 1

        winner = self._contrast_race(
            {self._eos_cell: 1.0},
            steps,
            depends_on=(dependency,),
            phase="contrast-eos",
        )
        if winner != self._eos_cell:
            return self._contrast_stop()
        try:
            raw_output = bytes(emitted).decode("utf-8")
        except UnicodeDecodeError as error:
            raise CausalSequencePointerError("neuronal output is not lossless UTF-8") from error
        if not raw_output:
            raise RuntimeError("contrast EOS won before any neuronal output was emitted")
        return _ContrastCompletion(
            raw_output=raw_output,
            emitted_tokens=tuple(emitted),
            selected_stop=False,
            selected_eos=True,
            eligible_nogo=tuple(eligible),
        )

    def _contrast_race(
        self,
        scores: dict[int, float],
        steps: list[CausalSequenceRaceStep],
        *,
        depends_on: tuple[int, ...],
        phase: str,
        normalize: bool = True,
    ) -> int:
        target_cells = tuple(sorted(cell for cell, score in scores.items() if score > 0.0))
        total = sum(scores[cell] for cell in target_cells)
        denominator = total if normalize else 1.0
        currents = (
            [
                _CURRENT_SCALE_PA * _CANDIDATE_GAIN * scores[cell] / denominator
                for cell in target_cells
            ]
            if total > 0.0
            else []
        )
        cells = (*target_cells, self._stop_cell)
        currents.append(_CURRENT_SCALE_PA * _STOP_DRIVE)
        race = self._cortex.race_currents_first_spike(cells, currents)
        self._record_step(
            steps,
            phase=phase,
            winner=race.winner,
            operation=self._operation(race.winner),
            spike_time_ms=race.spike_time_ms,
            cells=cells,
            currents=currents,
            depends_on=depends_on,
        )
        return race.winner

    @staticmethod
    def _contrast_stop() -> _ContrastCompletion:
        return _ContrastCompletion(
            raw_output=_STOP_OUTPUT,
            emitted_tokens=(),
            selected_stop=True,
            selected_eos=False,
            eligible_nogo=(),
        )

    def _candidate_scores(
        self,
        state: str,
        step: int,
        predecessor: _Prefix,
        features: tuple[_Feature, ...],
    ) -> dict[int, float]:
        scores: defaultdict[int, float] = defaultdict(float)
        episodic = self._episodic.get((state, step, predecessor), {})
        for target, strength in episodic.items():
            scores[target] += 12.0 * log1p(strength)

        position_distribution = self._positions.get(step, {})
        position_total = sum(position_distribution.values())
        sensory_scores: defaultdict[int, float] = defaultdict(float)
        for feature in features:
            distribution = self._sensory.get((step, feature), {})
            promoted = self._promoted_distribution(
                distribution,
                lambda target, feature=feature: self._sensory_contexts.get(
                    (step, feature, target),
                    set(),
                ),
            )
            feature_total = sum(distribution.values())
            for target, strength in promoted:
                conditional = strength / feature_total
                prior = (
                    position_distribution.get(target, 0.0) / position_total
                    if position_total > 0.0
                    else 0.0
                )
                # A sensory branch contributes only information above the
                # phase prior. Alternatives remain present in their own cells.
                excess = max(0.0, conditional - prior)
                sensory_scores[target] += excess * log1p(strength)
        for target, strength in sensory_scores.items():
            activity = min(_SENSORY_ACTIVITY_CAP, strength)
            scores[target] += _SENSORY_COINCIDENCE_GAIN * activity * activity

        if self._config.ablation is not CausalSequenceAblation.RECURRENCE_OFF:
            recurrent = self._promoted_distribution(
                self._transitions.get(predecessor, {}),
                lambda target: self._transition_contexts.get(
                    (predecessor, target),
                    set(),
                ),
            )
            if recurrent:
                # A bounded recurrent population broadcasts one finite drive
                # budget across its currently active learned postings.  This
                # preserves alternatives for a sensory branch to disambiguate
                # rather than letting frequency choose on the host.
                recurrent_drive = _RECURRENT_DRIVE_BUDGET / len(recurrent)
                for target, _strength in recurrent:
                    scores[target] += recurrent_drive

        positional = self._promoted_distribution(
            position_distribution,
            lambda target: self._position_contexts.get((step, target), set()),
        )
        positional_total = sum(strength for _, strength in positional)
        for target, strength in positional:
            scores[target] += strength / positional_total

        if self._config.ablation is not CausalSequenceAblation.NOGO_OFF:
            for target in tuple(scores):
                key = _NoGoKey(state, step, predecessor, target)
                if self._nogo.get(key, 0) > 0:
                    del scores[target]
        if step == 0 and not episodic:
            # Context-free frequency is useful continuation evidence, but it
            # must not choose the root of a novel sequence.  A reusable root
            # needs sensory evidence or a contrast certificate.
            sensory_targets = set(sensory_scores)
            scores = defaultdict(
                float,
                {target: score for target, score in scores.items() if target in sensory_targets},
            )
        return {target: score for target, score in scores.items() if score > 0.0}

    def _promoted_distribution(
        self,
        distribution: _Distribution,
        contexts_for: Callable[[int], set[str]],
    ) -> tuple[tuple[int, float], ...]:
        return tuple(
            (target, strength)
            for target, strength in distribution.items()
            if len(contexts_for(target)) >= self._config.min_reuse_contexts
        )

    def _semantic_race(
        self,
        scores: dict[int, float],
        steps: list[CausalSequenceRaceStep],
        *,
        depends_on: tuple[int, ...],
        phase: str,
    ) -> tuple[int, tuple[int, ...]]:
        # Sparse postings recruit only cells receiving learned current.  STOP
        # is always tonic and no host-side sort selects among recruited cells.
        target_cells = tuple(
            sorted(
                cell
                for cell, score in scores.items()
                if score > 0.0
                and (
                    0 <= cell < self._tokenizer.vocab_size
                    or cell == self._eos_cell
                    or cell in self._role_spec_by_cell
                )
            )
        )
        total = sum(scores.get(cell, 0.0) for cell in target_cells)
        denominator = max(_SEMANTIC_EVIDENCE_FLOOR, total)
        currents = [
            _CURRENT_SCALE_PA * _CANDIDATE_GAIN * scores.get(cell, 0.0) / denominator
            for cell in target_cells
        ]
        cells = (*target_cells, self._stop_cell)
        currents.append(_CURRENT_SCALE_PA * _STOP_DRIVE)
        race = self._cortex.race_currents_first_spike(cells, currents)
        winner = race.winner
        self._record_step(
            steps,
            phase=phase,
            winner=winner,
            operation=self._operation(winner),
            spike_time_ms=race.spike_time_ms,
            cells=cells,
            currents=currents,
            depends_on=depends_on,
        )
        positive = tuple(target for target in target_cells if scores.get(target, 0.0) > 0.0)
        return winner, positive

    def _ingest_role_cursor(
        self,
        state: str,
        spec: RoleCursorSpec,
        role_count: int,
        *,
        strength: float,
    ) -> None:
        start_distribution = self._role_starts[spec.action_cell]
        start_distribution[spec.start_ordinal] = (
            start_distribution.get(spec.start_ordinal, 0.0) + strength
        )
        self._role_start_contexts[(spec.action_cell, spec.start_ordinal)].add(state)
        for ordinal in range(spec.start_ordinal, role_count):
            successor_visible = ordinal + 1 < role_count
            boundary_target = self._continue_cell if successor_visible else self._exit_cell
            boundary_distribution = self._role_boundaries[successor_visible]
            boundary_distribution[boundary_target] = (
                boundary_distribution.get(boundary_target, 0.0) + strength
            )
            self._role_boundary_contexts[(successor_visible, boundary_target)].add(state)
            if successor_visible:
                offset = 1
                self._role_successor_offsets[offset] = (
                    self._role_successor_offsets.get(offset, 0.0) + strength
                )
                self._role_successor_contexts[offset].add(state)

    def _learned_role_race(
        self,
        scores: dict[int, float],
        steps: list[CausalSequenceRaceStep],
        *,
        depends_on: tuple[int, ...],
        phase: str,
    ) -> int:
        target_cells = tuple(sorted(cell for cell, score in scores.items() if score > 0.0))
        total = sum(scores[cell] for cell in target_cells)
        denominator = max(1.0, total)
        currents = [
            _CURRENT_SCALE_PA * _CANDIDATE_GAIN * scores[cell] / denominator
            for cell in target_cells
        ]
        cells = (*target_cells, self._stop_cell)
        currents.append(_CURRENT_SCALE_PA * _STOP_DRIVE)
        race = self._cortex.race_currents_first_spike(cells, currents)
        self._record_step(
            steps,
            phase=phase,
            winner=race.winner,
            operation=self._operation(race.winner),
            spike_time_ms=race.spike_time_ms,
            cells=cells,
            currents=currents,
            depends_on=depends_on,
        )
        return race.winner

    def _run_role_cursor(
        self,
        spec: RoleCursorSpec,
        role_count: int,
        steps: list[CausalSequenceRaceStep],
        *,
        action_step: int,
    ) -> bytes | None:
        pieces: list[bytes] = []
        dependency = action_step
        starts = self._promoted_distribution(
            self._role_starts.get(spec.action_cell, {}),
            lambda ordinal: self._role_start_contexts.get(
                (spec.action_cell, ordinal),
                set(),
            ),
        )
        start_scores = {
            self._role_bind_base + ordinal: strength
            for ordinal, strength in starts
            if 0 <= ordinal < role_count
        }
        winner = self._learned_role_race(
            start_scores,
            steps,
            depends_on=(dependency,),
            phase="role-bind",
        )
        dependency = len(steps) - 1
        if not self._role_bind_base <= winner < self._continue_cell:
            return None
        ordinal = winner - self._role_bind_base

        while True:
            pieces.append(f"<@{ordinal}>".encode("ascii"))

            successor_visible = ordinal + 1 < role_count
            boundaries = self._promoted_distribution(
                self._role_boundaries.get(successor_visible, {}),
                lambda target, successor_visible=successor_visible: (
                    self._role_boundary_contexts.get(
                        (successor_visible, target),
                        set(),
                    )
                ),
            )
            winner = self._learned_role_race(
                dict(boundaries),
                steps,
                depends_on=(dependency,),
                phase="role-boundary",
            )
            dependency = len(steps) - 1
            if winner == self._exit_cell:
                return b"".join(pieces)
            if winner != self._continue_cell:
                return None
            pieces.append(spec.separator)

            offsets = self._promoted_distribution(
                self._role_successor_offsets,
                lambda offset: self._role_successor_contexts.get(offset, set()),
            )
            successor_scores: defaultdict[int, float] = defaultdict(float)
            for offset, strength in offsets:
                successor = ordinal + offset
                if 0 <= successor < role_count:
                    successor_scores[self._role_bind_base + successor] += strength
            winner = self._learned_role_race(
                dict(successor_scores),
                steps,
                depends_on=(dependency,),
                phase="role-bind",
            )
            dependency = len(steps) - 1
            if not self._role_bind_base <= winner < self._continue_cell:
                return None
            ordinal = winner - self._role_bind_base

    def _record_step(
        self,
        steps: list[CausalSequenceRaceStep],
        *,
        phase: str,
        winner: int,
        operation: str,
        spike_time_ms: float,
        cells: Sequence[int],
        currents: Sequence[float],
        depends_on: tuple[int, ...],
    ) -> None:
        index = len(steps)
        if any(dependency >= index for dependency in depends_on):
            raise RuntimeError("causal sequence trace depends on a future race")
        steps.append(
            CausalSequenceRaceStep(
                index=index,
                phase=phase,
                winner_cell=winner,
                winner_operation=operation,
                spike_time_ms=spike_time_ms,
                candidate_currents_pa=tuple(
                    (int(cell), float(current))
                    for cell, current in zip(cells, currents, strict=True)
                    if current > 0.0
                ),
                depends_on_steps=depends_on,
            )
        )

    def _operation(self, cell: int) -> str:
        if cell == self._stop_cell:
            return "STOP"
        if cell == self._eos_cell:
            return "EOS"
        if cell == self._continue_cell:
            return "CONTINUE"
        if cell == self._exit_cell:
            return "EXIT"
        if cell in self._role_spec_by_cell:
            return "ROLE_CURSOR"
        if self._role_bind_base <= cell < self._continue_cell:
            return f"BIND:{cell - self._role_bind_base}"
        contrast_operation = self._contrast_cell_operations.get(cell)
        if contrast_operation is not None:
            return contrast_operation
        return f"EMIT:{cell}"

    def _pure_table_rows(
        self,
        table: dict[object, _Distribution],
    ) -> list[object]:
        return [
            [repr(key), [[target, strength] for target, strength in sorted(value.items())]]
            for key, value in sorted(table.items(), key=lambda item: repr(item[0]))
        ]

    def _compute_training_digest(self) -> str:
        return _digest(
            b"humanbrain-causal-sequence-state-v3",
            {
                "schema": _MODEL_SCHEMA,
                "config": {
                    "max_observation_features": self._config.max_observation_features,
                    "max_observation_order": self._config.max_observation_order,
                    "max_prefix_order": self._config.max_prefix_order,
                    "max_role_cursors": self._config.max_role_cursors,
                    "max_roles": self._config.max_roles,
                    "max_contrast_output_bytes": (self._config.max_contrast_output_bytes),
                    "max_contrast_candidate_episodes": (
                        self._config.max_contrast_candidate_episodes
                    ),
                    "min_reuse_contexts": self._config.min_reuse_contexts,
                    "random_seed": self._config.random_seed,
                },
                "tokenizer_phrases": [piece.hex() for piece in self._tokenizer.phrases],
                "role_specs": [
                    [spec.action_cell, spec.start_ordinal, spec.separator.hex()]
                    for spec in self._role_specs
                ],
                "episodic": self._pure_table_rows(self._episodic),
                "sensory": self._pure_table_rows(self._sensory),
                "sensory_contexts": sorted(
                    [repr(key), sorted(value)] for key, value in self._sensory_contexts.items()
                ),
                "transitions": self._pure_table_rows(self._transitions),
                "transition_contexts": sorted(
                    [repr(key), sorted(value)] for key, value in self._transition_contexts.items()
                ),
                "positions": self._pure_table_rows(self._positions),
                "position_contexts": sorted(
                    [repr(key), sorted(value)] for key, value in self._position_contexts.items()
                ),
                "role_starts": self._pure_table_rows(self._role_starts),
                "role_start_contexts": sorted(
                    [repr(key), sorted(value)] for key, value in self._role_start_contexts.items()
                ),
                "role_successor_offsets": [
                    [offset, strength]
                    for offset, strength in sorted(self._role_successor_offsets.items())
                ],
                "role_successor_contexts": sorted(
                    [offset, sorted(value)]
                    for offset, value in self._role_successor_contexts.items()
                ),
                "role_boundaries": self._pure_table_rows(self._role_boundaries),
                "role_boundary_contexts": sorted(
                    [repr(key), sorted(value)]
                    for key, value in self._role_boundary_contexts.items()
                ),
                "contrast_episodes": [
                    [
                        episode.episode_id,
                        episode.observation_state,
                        episode.features,
                        episode.pointer_output.hex(),
                        episode.template_cell,
                    ]
                    for episode in sorted(
                        self._contrast_episodes.values(),
                        key=lambda value: value.episode_id,
                    )
                ],
                "exact_episode_postings": [
                    [
                        state,
                        [
                            [episode_id, strength]
                            for episode_id, strength in sorted(posting.items())
                        ],
                    ]
                    for state, posting in sorted(self._exact_episode_postings.items())
                ],
                "exact_episode_atoms": [
                    [episode_id, list(atoms)]
                    for episode_id, atoms in sorted(self._exact_episode_atoms.items())
                ],
                "contrast_episode_postings": [
                    [feature, list(posting)]
                    for feature, posting in sorted(
                        self._contrast_episode_postings.items(),
                        key=lambda item: item[0],
                    )
                ],
                "contrast_certificates": [
                    [
                        certificate.certificate_id,
                        certificate.witness_id,
                        certificate.base_episode_id,
                        certificate.variant_episode_id,
                        certificate.factor_cell,
                        certificate.positive_features,
                        certificate.negative_features,
                        certificate.segment_cells,
                    ]
                    for certificate in sorted(
                        self._contrast_certificates.values(),
                        key=lambda value: value.certificate_id,
                    )
                ],
                "contrast_segments": [
                    [
                        segment.cell,
                        segment.base_episode_id,
                        segment.start,
                        segment.stop,
                        segment.replacement.hex(),
                    ]
                    for segment in sorted(
                        self._contrast_segments.values(),
                        key=lambda value: value.cell,
                    )
                ],
                "nogo": sorted(
                    [
                        key.observation_state,
                        key.step,
                        list(key.predecessor),
                        key.selected_token,
                        count,
                    ]
                    for key, count in self._nogo.items()
                ),
                "consequences": sorted(
                    [state, list(tokens)] for state, tokens in self._consequences.items()
                ),
            },
        )

    def _require_pending(
        self,
        prompt: str,
        inventory: SymbolInventory,
        cue_text: str,
    ) -> _PendingCompletion:
        _validate_text(prompt, "prompt")
        _require_inventory(inventory)
        _validate_text(cue_text, "cue_text")
        pending = self._pending
        if pending is None:
            raise RuntimeError("there is no pending causal sequence completion")
        if pending.query_sha256 != _query_sha256(prompt, inventory, cue_text):
            raise CausalSequencePointerError(
                "consequence does not match the pending causal sequence query"
            )
        return pending

    @staticmethod
    def _validate_stop_text(stop_text: str | tuple[str, ...] | None) -> None:
        if stop_text is None:
            return
        if isinstance(stop_text, str):
            if not stop_text:
                raise ValueError("stop_text must not be empty")
            return
        if (
            not isinstance(stop_text, tuple)
            or not stop_text
            or any(not isinstance(value, str) or not value for value in stop_text)
        ):
            raise ValueError("stop_text must be a string or non-empty string tuple")

    @staticmethod
    def _nogo_label(key: _NoGoKey) -> str:
        return (
            "nogo:"
            f"{key.observation_state[:12]}:{key.step}:"
            f"{_digest(b'humanbrain-prefix-v1', key.predecessor)[:12]}:"
            f"{key.selected_token}"
        )
