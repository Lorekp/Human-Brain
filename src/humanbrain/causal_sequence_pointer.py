"""Causal neuronal sequence composition over pointer-normalized text.

The model stores transition synapses, not candidate programs.  Successful
episodes leave exact contextual traces; only structure repeated across distinct
observation states is allowed to drive a novel context.  Generation is one
strictly forward traversal.  Every emitted chunk, EOS, STOP, live role binding,
and role-loop boundary is selected by a fitted-cell first-spike race.

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


_MODEL_SCHEMA = "humanbrain-causal-sequence-pointer-v1"
_STOP_OUTPUT = "<stop>insufficient causal sequence evidence</stop>"
_POINTER_EVENT = re.compile(r"<@(\d+)>")
_CURRENT_SCALE_PA = 900.0
_CANDIDATE_GAIN = 2.0
_STOP_DRIVE = 1.15
_SEMANTIC_EVIDENCE_FLOOR = 6.2
_SENSORY_ACTIVITY_CAP = 2.5
_SENSORY_COINCIDENCE_GAIN = 0.5
_RECURRENT_DRIVE_BUDGET = 3.0

_Feature = tuple[int, ...]
_Prefix = tuple[int, ...]
_Distribution = dict[int, float]


class CausalSequencePointerError(ValueError):
    """The neuronal composer input or pending consequence is invalid."""


class CausalSequenceAblation(Enum):
    """Focused controls over the two causal mechanisms under study."""

    NONE = "none"
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
    semantic_tokens: tuple[int, ...]
    role_actions: tuple[tuple[int, int], ...]
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

        # Exact transitions are episodic.  The other tables are reusable only
        # after their corresponding context sets cross min_reuse_contexts.
        self._episodic: dict[tuple[str, int, _Prefix], _Distribution] = defaultdict(dict)
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
        prior_digest = self._training_digest
        steps: list[CausalSequenceRaceStep] = []
        semantic_step_indices: list[int] = []
        semantic_tokens: list[int] = []
        output_pieces: list[bytes] = []
        role_actions: list[tuple[int, int]] = []
        eligible: list[_NoGoKey] = []
        selected_stop = False
        selected_eos = False

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
        """Consolidate the emitted success online, without replaying old rows."""

        _validate_text(output, "output")
        if not isfinite(strength) or strength <= 0.0:
            raise ValueError("strength must be one positive finite value")
        pending = self._require_pending(prompt, inventory, cue_text)
        prior_digest = self._training_digest
        changed: list[str] = []
        if self._config.plasticity:
            self._ingest_success(
                pending.observation_state,
                pending.observation_features,
                pending.semantic_tokens,
                strength=float(strength),
            )
            for action_cell, role_count in pending.role_actions:
                self._ingest_role_cursor(
                    pending.observation_state,
                    self._role_spec_by_cell[action_cell],
                    role_count,
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
            b"humanbrain-causal-sequence-state-v1",
            {
                "schema": _MODEL_SCHEMA,
                "config": {
                    "max_observation_features": self._config.max_observation_features,
                    "max_observation_order": self._config.max_observation_order,
                    "max_prefix_order": self._config.max_prefix_order,
                    "max_role_cursors": self._config.max_role_cursors,
                    "max_roles": self._config.max_roles,
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
