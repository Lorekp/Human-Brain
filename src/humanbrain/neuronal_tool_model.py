"""Generic persistent-tool interface over neuronal sequence generation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Literal

from humanbrain.causal_sequence_pointer import (
    CausalSequenceCompletionTrace,
    CausalSequenceConfig,
    CausalSequenceLearningUpdate,
    CausalSequencePointerModel,
    CausalSequenceTrainingExample,
)
from humanbrain.persistent_tool_learning import ModelEmission
from humanbrain.pointer_binding import SymbolInventory, materialize, pointerize

__all__ = [
    "NeuronalToolLearningUpdate",
    "NeuronalToolModel",
    "NeuronalToolModelError",
    "NeuronalToolTrace",
    "NeuronalToolTrainingExample",
]


_MODEL_SCHEMA = "humanbrain-neuronal-tool-model-v1"
_LEXICAL_POLICY = "first-occurrence-ascii-identifier-v1"
_CHANNEL_POLICY = "raw-utf8-decimal-state-v1+lexical-pointer-copy-v1"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED_POINTER = "<@"
_PHYSICAL_STOP_NOT_MATERIALIZED = "physical_stop_not_materialized"


class NeuronalToolModelError(ValueError):
    """The raw prompt or exact-once tool-model custody is invalid."""


@dataclass(frozen=True, slots=True)
class NeuronalToolTrainingExample:
    """One successful generic raw prompt-to-output experience."""

    prompt: str
    output: str
    provenance_id: str

    def __post_init__(self) -> None:
        _require_raw_text(self.prompt, "prompt")
        _require_raw_text(self.output, "output")
        _require_text(self.provenance_id, "provenance_id")


@dataclass(frozen=True, slots=True)
class NeuronalToolTrace:
    """Physical neuronal custody behind one sealed raw-string emission."""

    query_sha256: str
    selected_stop: bool
    inventory_cardinality: int
    inventory_sha256: str
    pointer_output_sha256: str
    materialized_output_sha256: str | None
    materialization_error: str | None
    underlying_trace: CausalSequenceCompletionTrace
    sha256: str

    @classmethod
    def _seal(
        cls,
        query_sha256: str,
        inventory: SymbolInventory,
        pointer_output: str,
        materialized_output: str | None,
        materialization_error: str | None,
        underlying_trace: CausalSequenceCompletionTrace,
    ) -> NeuronalToolTrace:
        pointer_output_sha256 = _digest(
            "humanbrain-neuronal-pointer-output-v1",
            pointer_output,
        )
        materialized_output_sha256 = (
            None
            if materialized_output is None
            else _digest(
                "humanbrain-neuronal-materialized-output-v1",
                materialized_output,
            )
        )
        payload = {
            "inventory_cardinality": len(inventory.symbols),
            "inventory_sha256": inventory.sha256,
            "materialization_error": materialization_error,
            "materialized_output_sha256": materialized_output_sha256,
            "pointer_output_sha256": pointer_output_sha256,
            "query_sha256": query_sha256,
            "selected_stop": underlying_trace.selected_stop,
            "underlying": _underlying_trace_payload(underlying_trace),
        }
        return cls(
            query_sha256=query_sha256,
            selected_stop=underlying_trace.selected_stop,
            inventory_cardinality=len(inventory.symbols),
            inventory_sha256=inventory.sha256,
            pointer_output_sha256=pointer_output_sha256,
            materialized_output_sha256=materialized_output_sha256,
            materialization_error=materialization_error,
            underlying_trace=underlying_trace,
            sha256=_digest("humanbrain-neuronal-tool-trace-v1", payload),
        )

    def __post_init__(self) -> None:
        payload = {
            "inventory_cardinality": self.inventory_cardinality,
            "inventory_sha256": self.inventory_sha256,
            "materialization_error": self.materialization_error,
            "materialized_output_sha256": self.materialized_output_sha256,
            "pointer_output_sha256": self.pointer_output_sha256,
            "query_sha256": self.query_sha256,
            "selected_stop": self.selected_stop,
            "underlying": _underlying_trace_payload(self.underlying_trace),
        }
        stop_status_valid = not self.selected_stop or (
            self.materialized_output_sha256 is None
            and self.materialization_error == _PHYSICAL_STOP_NOT_MATERIALIZED
        )
        emitted_status_valid = self.selected_stop or (
            (self.materialized_output_sha256 is None) != (self.materialization_error is None)
        )
        if (
            not self.query_sha256
            or type(self.inventory_cardinality) is not int
            or self.inventory_cardinality < 0
            or not self.inventory_sha256
            or self.selected_stop != self.underlying_trace.selected_stop
            or self.pointer_output_sha256
            != _digest(
                "humanbrain-neuronal-pointer-output-v1",
                self.underlying_trace.raw_output,
            )
            or not stop_status_valid
            or not emitted_status_valid
            or self.sha256 != _digest("humanbrain-neuronal-tool-trace-v1", payload)
        ):
            raise ValueError("neuronal tool trace is not authentic")


@dataclass(frozen=True, slots=True)
class NeuronalToolLearningUpdate:
    """One wrapper consequence bound to the underlying neuronal update."""

    polarity: Literal["success", "failure", "neutral"]
    query_sha256: str
    consequence_sha256: str | None
    prior_training_digest: str
    training_digest: str
    underlying_update: CausalSequenceLearningUpdate | None
    provenance_sha256: str

    @classmethod
    def _seal(
        cls,
        polarity: Literal["success", "failure", "neutral"],
        query_sha256: str,
        consequence_text: str | None,
        prior_training_digest: str,
        training_digest: str,
        underlying_update: CausalSequenceLearningUpdate | None,
    ) -> NeuronalToolLearningUpdate:
        consequence_sha256 = (
            None
            if consequence_text is None
            else _digest("humanbrain-tool-consequence-v1", consequence_text)
        )
        payload = {
            "consequence_sha256": consequence_sha256,
            "polarity": polarity,
            "prior_training_digest": prior_training_digest,
            "query_sha256": query_sha256,
            "training_digest": training_digest,
            "underlying_update": _underlying_update_payload(underlying_update),
        }
        return cls(
            polarity=polarity,
            query_sha256=query_sha256,
            consequence_sha256=consequence_sha256,
            prior_training_digest=prior_training_digest,
            training_digest=training_digest,
            underlying_update=underlying_update,
            provenance_sha256=_digest(
                "humanbrain-neuronal-tool-update-v1",
                payload,
            ),
        )

    def __post_init__(self) -> None:
        payload = {
            "consequence_sha256": self.consequence_sha256,
            "polarity": self.polarity,
            "prior_training_digest": self.prior_training_digest,
            "query_sha256": self.query_sha256,
            "training_digest": self.training_digest,
            "underlying_update": _underlying_update_payload(self.underlying_update),
        }
        if (
            self.polarity not in {"success", "failure", "neutral"}
            or not self.query_sha256
            or not self.prior_training_digest
            or not self.training_digest
            or (self.polarity == "neutral") != (self.consequence_sha256 is None)
            or self.provenance_sha256 != _digest("humanbrain-neuronal-tool-update-v1", payload)
        ):
            raise ValueError("neuronal tool learning update is not authentic")


@dataclass(frozen=True, slots=True)
class _PendingEmission:
    query_sha256: str
    inner_prompt: str
    inner_cue: str
    inventory: SymbolInventory
    pointer_output: str
    raw_output: str | None
    selected_stop: bool


class NeuronalToolModel:
    """Deep raw-string seam over pointer-bound neuronal sequence learning."""

    def __init__(
        self,
        inner: CausalSequencePointerModel,
        config: CausalSequenceConfig,
    ) -> None:
        if not isinstance(inner, CausalSequencePointerModel):
            raise TypeError("inner must be a CausalSequencePointerModel")
        if not isinstance(config, CausalSequenceConfig):
            raise TypeError("config must be a CausalSequenceConfig")
        self._inner = inner
        self._config = config
        self._pending: _PendingEmission | None = None
        self._last_trace: NeuronalToolTrace | None = None
        self._last_update: NeuronalToolLearningUpdate | None = None

    @classmethod
    def grow(
        cls,
        examples: Iterable[NeuronalToolTrainingExample],
        config: CausalSequenceConfig | None = None,
    ) -> NeuronalToolModel:
        cfg = CausalSequenceConfig() if config is None else config
        if not isinstance(cfg, CausalSequenceConfig):
            raise TypeError("config must be a CausalSequenceConfig")
        rows = tuple(examples)
        if not rows:
            raise ValueError("examples must contain at least one successful experience")
        if any(not isinstance(row, NeuronalToolTrainingExample) for row in rows):
            raise TypeError("examples must contain NeuronalToolTrainingExample values")
        prepared = tuple(
            CausalSequenceTrainingExample(
                prompt=_copy_channel(row.prompt),
                output=row.output,
                inventory=_lexical_inventory(row.prompt, cfg.max_roles),
                cue_text=_semantic_channel(row.prompt),
                provenance_id=row.provenance_id,
            )
            for row in rows
        )
        return cls(CausalSequencePointerModel.grow(prepared, cfg), cfg)

    @property
    def training_digest(self) -> str:
        return _digest(
            _MODEL_SCHEMA,
            {
                "inner_training_digest": self._inner.training_digest,
                "channel_policy": _CHANNEL_POLICY,
                "lexical_policy": _LEXICAL_POLICY,
                "max_roles": self._config.max_roles,
            },
        )

    @property
    def last_trace(self) -> NeuronalToolTrace | None:
        return self._last_trace

    @property
    def last_update(self) -> NeuronalToolLearningUpdate | None:
        return self._last_update

    def infer(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stop_text: tuple[str, ...],
    ) -> ModelEmission:
        _require_raw_text(prompt, "prompt")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            not isinstance(stop_text, tuple)
            or not stop_text
            or any(not isinstance(marker, str) or not marker for marker in stop_text)
        ):
            raise ValueError("stop_text must be a non-empty string tuple")
        if self._pending is not None:
            raise RuntimeError("resolve the pending neuronal tool emission first")

        inventory = _lexical_inventory(prompt, self._config.max_roles)
        inner_prompt = _copy_channel(prompt)
        inner_cue = _semantic_channel(prompt)
        prior_digest = self.training_digest
        self._last_update = None
        pointer_output = self._inner.raw_complete(
            inner_prompt,
            inventory,
            cue_text=inner_cue,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )
        if self.training_digest != prior_digest:
            raise RuntimeError("underlying neuronal inference mutated learned state")
        underlying_trace = self._inner.last_trace
        if underlying_trace is None or underlying_trace.raw_output != pointer_output:
            raise RuntimeError("underlying neuronal trace is missing or inconsistent")

        provisional = ModelEmission.seal(
            prompt,
            prior_digest,
            pointer_output,
            selected_stop=underlying_trace.selected_stop,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )
        self._pending = _PendingEmission(
            query_sha256=provisional.query_sha256,
            inner_prompt=inner_prompt,
            inner_cue=inner_cue,
            inventory=inventory,
            pointer_output=pointer_output,
            raw_output=pointer_output if underlying_trace.selected_stop else None,
            selected_stop=underlying_trace.selected_stop,
        )
        if underlying_trace.selected_stop:
            self._last_trace = NeuronalToolTrace._seal(
                provisional.query_sha256,
                inventory,
                pointer_output,
                None,
                _PHYSICAL_STOP_NOT_MATERIALIZED,
                underlying_trace,
            )
            return provisional

        try:
            raw_output = materialize(pointer_output, inventory)
        except (TypeError, ValueError) as error:
            self._last_trace = NeuronalToolTrace._seal(
                provisional.query_sha256,
                inventory,
                pointer_output,
                None,
                "invalid_pointer_output",
                underlying_trace,
            )
            raise NeuronalToolModelError(
                "neuronal output could not be materialized against its prompt inventory"
            ) from error
        emission = ModelEmission.seal(
            prompt,
            prior_digest,
            raw_output,
            selected_stop=False,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )
        if emission.query_sha256 != provisional.query_sha256:
            raise RuntimeError("materialization changed model-query custody")
        self._last_trace = NeuronalToolTrace._seal(
            emission.query_sha256,
            inventory,
            pointer_output,
            raw_output,
            None,
            underlying_trace,
        )
        self._pending = _PendingEmission(
            query_sha256=emission.query_sha256,
            inner_prompt=inner_prompt,
            inner_cue=inner_cue,
            inventory=inventory,
            pointer_output=pointer_output,
            raw_output=raw_output,
            selected_stop=False,
        )
        return emission

    def resolve_success(
        self,
        query_sha256: str,
        consequence_text: str,
        strength: float,
    ) -> NeuronalToolLearningUpdate:
        _require_text(consequence_text, "consequence_text")
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not isfinite(strength)
            or strength <= 0.0
        ):
            raise ValueError("strength must be one positive finite value")
        pending = self._require_pending(query_sha256)
        if pending.selected_stop:
            raise NeuronalToolModelError("physical STOP cannot be resolved as success")
        if (
            pending.raw_output is None
            or pointerize(
                pending.raw_output,
                pending.inventory,
            )
            != pending.pointer_output
        ):
            raise RuntimeError("cached neuronal emission lost pointer custody")
        prior_digest = self.training_digest
        inner_update = self._inner.learn(
            pending.inner_prompt,
            pending.raw_output,
            pending.inventory,
            cue_text=pending.inner_cue,
            strength=float(strength),
        )
        self._pending = None
        update = NeuronalToolLearningUpdate._seal(
            "success",
            query_sha256,
            consequence_text,
            prior_digest,
            self.training_digest,
            inner_update,
        )
        self._last_update = update
        return update

    def resolve_failure(
        self,
        query_sha256: str,
        consequence_text: str,
    ) -> NeuronalToolLearningUpdate:
        _require_text(consequence_text, "consequence_text")
        pending = self._require_pending(query_sha256)
        prior_digest = self.training_digest
        inner_update = (
            None
            if pending.selected_stop
            else self._inner.observe_failure(
                pending.inner_prompt,
                pending.inventory,
                cue_text=pending.inner_cue,
                feedback_text=consequence_text,
            )
        )
        self._pending = None
        update = NeuronalToolLearningUpdate._seal(
            "failure",
            query_sha256,
            consequence_text,
            prior_digest,
            self.training_digest,
            inner_update,
        )
        self._last_update = update
        return update

    def resolve_neutral(self, query_sha256: str) -> NeuronalToolLearningUpdate:
        pending = self._require_pending(query_sha256)
        prior_digest = self.training_digest
        inner_update = (
            None
            if pending.selected_stop
            else self._inner.resolve_pending_neutral(
                pending.inner_prompt,
                pending.inventory,
                pending.inner_cue,
            )
        )
        if self.training_digest != prior_digest:
            raise RuntimeError("neutral resolution mutated learned state")
        self._pending = None
        update = NeuronalToolLearningUpdate._seal(
            "neutral",
            query_sha256,
            None,
            prior_digest,
            prior_digest,
            inner_update,
        )
        self._last_update = update
        return update

    def _require_pending(self, query_sha256: str) -> _PendingEmission:
        _require_text(query_sha256, "query_sha256")
        pending = self._pending
        if pending is None:
            raise RuntimeError("there is no pending neuronal tool emission")
        if pending.query_sha256 != query_sha256:
            raise NeuronalToolModelError(
                "resolution does not match the pending neuronal tool query"
            )
        return pending


def _lexical_inventory(prompt: str, max_roles: int) -> SymbolInventory:
    """Seal first-occurrence ASCII identifier lexemes from the raw prompt."""

    _require_text(prompt, "prompt")
    symbols: list[str] = []
    seen: set[str] = set()
    for match in _IDENTIFIER.finditer(prompt):
        symbol = match.group(0)
        if symbol in seen:
            continue
        if len(symbols) == max_roles:
            raise NeuronalToolModelError(
                "prompt lexical inventory exceeds configured neuronal role capacity"
            )
        seen.add(symbol)
        symbols.append(symbol)
    return SymbolInventory(tuple(symbols))


def _copy_channel(prompt: str) -> str:
    """Tag the raw prompt for pointer-normalized equality/copy processing."""

    return f"1|{prompt}"


def _semantic_channel(prompt: str) -> str:
    """Losslessly retain raw UTF-8 identity without replacing its lexemes."""

    encoded = ".".join(str(byte) for byte in prompt.encode("utf-8"))
    return f"0|{encoded}"


def _underlying_update_payload(
    update: CausalSequenceLearningUpdate | None,
) -> object:
    if update is None:
        return None
    return {
        "changed_synapses": update.changed_synapses,
        "evidence_sha256": update.evidence_sha256,
        "observed_success_sequence": update.observed_success_sequence,
        "polarity": update.polarity,
        "prior_training_digest": update.prior_training_digest,
        "training_digest": update.training_digest,
    }


def _underlying_trace_payload(trace: CausalSequenceCompletionTrace) -> object:
    return {
        "ablation": trace.ablation.value,
        "emitted_token_ids": list(trace.emitted_token_ids),
        "observation_features": [list(feature) for feature in trace.observation_features],
        "observation_state": trace.observation_state,
        "prior_training_digest": trace.prior_training_digest,
        "query_sha256": trace.query_sha256,
        "raw_output_sha256": _digest(
            "humanbrain-underlying-pointer-output-v1",
            trace.raw_output,
        ),
        "selected_eos": trace.selected_eos,
        "selected_stop": trace.selected_stop,
        "steps": [
            {
                "candidate_currents_pa": [
                    [cell, current.hex()] for cell, current in step.candidate_currents_pa
                ],
                "depends_on_steps": list(step.depends_on_steps),
                "index": step.index,
                "phase": step.phase,
                "spike_time_ms": step.spike_time_ms.hex(),
                "winner_cell": step.winner_cell,
                "winner_operation": step.winner_operation,
            }
            for step in trace.steps
        ],
    }


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be one non-empty string")


def _require_raw_text(value: str, name: str) -> None:
    _require_text(value, name)
    if _RESERVED_POINTER in value:
        raise NeuronalToolModelError(f"{name} contains reserved neuronal pointer syntax")


def _digest(domain: str, value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return sha256(domain.encode("ascii") + b"\0" + payload).hexdigest()
