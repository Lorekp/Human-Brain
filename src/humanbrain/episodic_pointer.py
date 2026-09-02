"""Cue-routed episodic recall over pointer-bound neuronal generators."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from humanbrain.cortex import DendriticCortex
from humanbrain.language import LearningUpdate, ModelConfig
from humanbrain.pointer_binding import (
    PointerBoundCellModel,
    PointerTrainingExample,
    SymbolInventory,
    materialize,
    pointerize,
)

__all__ = [
    "CueRoutedPointerCellModel",
    "EpisodicRoutingTrace",
    "RoutedPointerTrainingExample",
    "ascii_word_cues",
]

_CUE = re.compile(r"[A-Za-z0-9]+")
_MODEL_SCHEMA = b"humanbrain-cue-routed-pointer-cell-model-v1\0"
_STOP_TONIC_DRIVE = 0.05


def ascii_word_cues(text: str) -> tuple[str, ...]:
    """Return unique lowercase ASCII-alphanumeric cues without semantic labels."""

    if not isinstance(text, str) or not text:
        raise ValueError("cue text must be one non-empty string")
    return tuple(sorted({match.group(0).lower() for match in _CUE.finditer(text)}))


def _prompt_cues(prompt: str, inventory: SymbolInventory) -> frozenset[str]:
    return frozenset(ascii_word_cues(pointerize(prompt, inventory)))


def _prompt_sha256(prompt: str, inventory: SymbolInventory) -> str:
    return sha256(pointerize(prompt, inventory).encode("utf-8")).hexdigest()


def _cue_sha256(cues: frozenset[str]) -> str:
    return sha256("\0".join(sorted(cues)).encode("ascii")).hexdigest()


def _is_stop(output: str) -> bool:
    return output.startswith("<stop>") and output.endswith("</stop>")


@dataclass(frozen=True, slots=True)
class EpisodicRoutingTrace:
    """Auditable first-spike route selection from one completion."""

    cue_sha256: str
    cue_count: int
    route_scores: tuple[float, ...]
    selected_route: int
    selected_stop: bool
    refractory_route: int | None
    spike_time_ms: float
    simulated_steps: int


@dataclass(frozen=True, slots=True)
class RoutedPointerTrainingExample:
    """One successful or STOP episode plus its mechanical retrieval evidence."""

    pointer: PointerTrainingExample
    cue_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.pointer, PointerTrainingExample):
            raise TypeError("pointer must be a PointerTrainingExample")
        if not isinstance(self.cue_text, str) or not self.cue_text:
            raise ValueError("cue_text must be one non-empty string")


@dataclass(slots=True)
class _EpisodeRoute:
    model: PointerBoundCellModel
    cues: set[str]
    stop: bool
    anchor_prompt: str
    anchor_inventory: SymbolInventory
    anchor_cue_text: str


class CueRoutedPointerCellModel:
    """Recall one learned episode by cue-driven cells, then generate through it.

    Host code extracts mechanical word cues and computes their synaptic overlap.
    A fitted-cell first-spike race chooses the learned episode or a STOP episode;
    the selected episode's pointer-bound cell model emits the actual completion.
    """

    def __init__(
        self,
        routes: list[_EpisodeRoute],
        router: DendriticCortex,
    ) -> None:
        if len(routes) < 2:
            raise ValueError("episodic routing needs at least two learned routes")
        if router.cell_count < len(routes):
            raise ValueError("router has fewer physical cells than learned routes")
        if not any(route.stop for route in routes):
            raise ValueError("episodic routing needs at least one learned STOP route")
        if all(route.stop for route in routes):
            raise ValueError("episodic routing needs at least one action route")
        self._routes = routes
        self._router = router
        self._last_route: int | None = None
        self._last_prompt_sha256: str | None = None
        self._last_cue_sha256: str | None = None
        self._last_trace: EpisodicRoutingTrace | None = None
        self._refractory_route: int | None = None
        self._failure_events = 0

    @classmethod
    def grow(
        cls,
        examples: Iterable[RoutedPointerTrainingExample],
        config: ModelConfig | None = None,
    ) -> CueRoutedPointerCellModel:
        example_tuple = tuple(examples)
        if len(example_tuple) < 2:
            raise ValueError("at least two routed training examples are required")
        if any(
            not isinstance(example, RoutedPointerTrainingExample)
            for example in example_tuple
        ):
            raise TypeError("examples must contain RoutedPointerTrainingExample values")
        cfg = config or ModelConfig()
        routes = [
            _EpisodeRoute(
                model=PointerBoundCellModel.grow((example.pointer,), cfg),
                cues=set(
                    _prompt_cues(example.cue_text, example.pointer.inventory)
                ),
                stop=_is_stop(example.pointer.output),
                anchor_prompt=example.pointer.prompt,
                anchor_inventory=example.pointer.inventory,
                anchor_cue_text=example.cue_text,
            )
            for example in example_tuple
        ]
        router = DendriticCortex(
            cell_count=max(2, len(routes)),
            max_order=1,
            parameters=cfg.cell_parameters,
            seed=cfg.random_seed,
        )
        return cls(routes, router)

    @property
    def training_digest(self) -> str:
        digest = sha256()
        digest.update(_MODEL_SCHEMA)
        digest.update(len(self._routes).to_bytes(8, "big"))
        for route in self._routes:
            digest.update(bytes((int(route.stop),)))
            digest.update(route.model.training_digest.encode("ascii"))
            digest.update(
                _prompt_sha256(
                    route.anchor_prompt,
                    route.anchor_inventory,
                ).encode("ascii")
            )
            digest.update(route.anchor_inventory.sha256.encode("ascii"))
            digest.update(len(route.cues).to_bytes(8, "big"))
            for cue in sorted(route.cues):
                raw = cue.encode("ascii")
                digest.update(len(raw).to_bytes(4, "big"))
                digest.update(raw)
        return digest.hexdigest()

    @property
    def grown_branches(self) -> int:
        return sum(route.model.grown_branches for route in self._routes)

    @property
    def route_count(self) -> int:
        return len(self._routes)

    @property
    def failure_events(self) -> int:
        return self._failure_events

    @property
    def last_routing_trace(self) -> EpisodicRoutingTrace | None:
        return self._last_trace

    def _background_cues(self) -> frozenset[str]:
        background = set(self._routes[0].cues)
        for route in self._routes[1:]:
            background.intersection_update(route.cues)
        return frozenset(background)

    def _route_scores(self, query_cues: frozenset[str]) -> tuple[float, ...]:
        background = self._background_cues()
        query = query_cues - background
        scores: list[float] = []
        for route in self._routes:
            learned = route.cues - background
            union = query | learned
            score = len(query & learned) / len(union) if union else 0.0
            if route.stop:
                score = max(score, _STOP_TONIC_DRIVE)
            scores.append(score)
        if self._refractory_route is not None:
            scores[self._refractory_route] = 0.0
        return tuple(scores)

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        query_cues = _prompt_cues(cue_text, inventory)
        scores = self._route_scores(query_cues)
        active = np.arange(len(self._routes), dtype=np.int64)
        parameters = self._router.parameters
        association = (
            np.asarray(scores, dtype=np.float64)
            * parameters.dendritic_probability_gain
        )
        refractory_route = self._refractory_route
        race = self._router.race_currents(
            active,
            association * parameters.branch_current_pa,
            fallback_current_pa=(association + 6.0) * parameters.branch_current_pa,
        )
        selected = race.winner
        self._refractory_route = None
        self._last_route = selected
        self._last_prompt_sha256 = _prompt_sha256(prompt, inventory)
        self._last_cue_sha256 = _cue_sha256(query_cues)
        self._last_trace = EpisodicRoutingTrace(
            cue_sha256=self._last_cue_sha256,
            cue_count=len(query_cues),
            route_scores=scores,
            selected_route=selected,
            selected_stop=self._routes[selected].stop,
            refractory_route=refractory_route,
            spike_time_ms=race.spike_time_ms,
            simulated_steps=race.simulated_steps,
        )
        return self._routes[selected].model.raw_complete(
            self._routes[selected].anchor_prompt,
            self._routes[selected].anchor_inventory,
            cue_text=self._routes[selected].anchor_cue_text,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
        )

    def complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        return materialize(
            self.raw_complete(
                prompt,
                inventory,
                cue_text=cue_text,
                max_new_tokens=max_new_tokens,
                stop_text=stop_text,
            ),
            inventory,
        )

    def observe_failure(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        feedback_text: str,
    ) -> None:
        """Apply one context-bound refractory exclusion to the failed winner."""

        cue_sha256 = _cue_sha256(_prompt_cues(cue_text, inventory))
        prompt_sha256 = _prompt_sha256(prompt, inventory)
        if not isinstance(feedback_text, str) or not feedback_text:
            raise ValueError("feedback_text must be one non-empty string")
        if (
            self._last_route is None
            or self._last_prompt_sha256 != prompt_sha256
            or self._last_cue_sha256 != cue_sha256
        ):
            raise ValueError("failure is not bound to the last neuronal completion")
        self._refractory_route = self._last_route
        self._failure_events += 1

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> LearningUpdate:
        """Reinforce the successful route and bind the new prompt cues to it."""

        prompt_sha256 = _prompt_sha256(prompt, inventory)
        cue_sha256 = _cue_sha256(_prompt_cues(cue_text, inventory))
        if (
            self._last_route is None
            or self._last_prompt_sha256 != prompt_sha256
            or self._last_cue_sha256 != cue_sha256
        ):
            raise ValueError("success is not bound to the last neuronal completion")
        route = self._routes[self._last_route]
        if route.stop:
            raise ValueError("a STOP route cannot receive successful action learning")
        prior_digest = self.training_digest
        update = route.model.learn(
            prompt,
            output,
            inventory,
            cue_text=cue_text,
            strength=strength,
        )
        route.cues.update(_prompt_cues(cue_text, inventory))
        route.anchor_prompt = prompt
        route.anchor_inventory = inventory
        route.anchor_cue_text = cue_text
        self._refractory_route = None
        return LearningUpdate(
            prior_training_digest=prior_digest,
            training_digest=self.training_digest,
            token_count=update.token_count,
            added_branches=update.added_branches,
            grown_transitions=update.grown_transitions,
        )
