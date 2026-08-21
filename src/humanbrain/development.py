"""Developmental recruitment and structural plasticity for recurrent cells.

The schedule is orchestration, while every value that survives a schedule
transition lives on a fitted cell or a sparse synapse.  The ordinary recurrent
circuit is left unchanged until this controller is explicitly constructed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256

import numpy as np

from humanbrain.recurrent import RecurrentCellCircuit
from humanbrain.recurrent_language import RecurrentCellLanguageModel, RecurrentModelConfig
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig
from humanbrain.topology import topology_coordinates

CELL_ORDERINGS = frozenset({"random", "layered_prefix"})
PRUNING_RULES = frozenset({"activity", "random"})


@dataclass(frozen=True)
class DevelopmentalGrowthConfig:
    """Frozen cell-recruitment and sparse structural-plasticity schedule."""

    stage_active_cells: tuple[int, ...]
    cell_ordering: str = "layered_prefix"
    initial_recurrent_fanout: int = 6
    final_recurrent_fanout: int = 6
    pruning_rule: str = "activity"
    homeostatic_scaling: bool = True
    structural_seed: int = 10_017
    turnover_edges_per_source: int = 2
    homeostatic_min_scale: float = 0.5
    homeostatic_max_scale: float = 2.0

    def validate(self, reservoir_cells: int, recurrent_capacity: int) -> None:
        if not self.stage_active_cells:
            raise ValueError("development needs at least one recruitment stage")
        if self.stage_active_cells[-1] != reservoir_cells:
            raise ValueError("the final stage must recruit every reservoir cell")
        if any(value < 8 or value > reservoir_cells for value in self.stage_active_cells):
            raise ValueError("stage cell counts must fit the reservoir")
        if any(
            right < left
            for left, right in zip(
                self.stage_active_cells,
                self.stage_active_cells[1:],
                strict=False,
            )
        ):
            raise ValueError("stage cell counts must be nondecreasing")
        if self.cell_ordering not in CELL_ORDERINGS:
            raise ValueError("unknown developmental cell ordering")
        if self.pruning_rule not in PRUNING_RULES:
            raise ValueError("unknown structural pruning rule")
        if not 1 <= self.initial_recurrent_fanout <= recurrent_capacity:
            raise ValueError("initial fanout must fit recurrent capacity")
        if not 1 <= self.final_recurrent_fanout <= recurrent_capacity:
            raise ValueError("final fanout must fit recurrent capacity")
        if self.turnover_edges_per_source < 0:
            raise ValueError("turnover edge count cannot be negative")
        if not 0.0 < self.homeostatic_min_scale <= self.homeostatic_max_scale:
            raise ValueError("homeostatic scale bounds must be positive and ordered")


@dataclass(frozen=True)
class DevelopmentalStage:
    stage: int
    active_cells: int
    recruited_cells: int
    training_examples: int
    recurrent_synapses: int
    grown_synapses: int
    turnover_pruned_synapses: int
    mean_activity: float
    homeostatic_cells_scaled: int
    homeostatic_scale_min: float
    homeostatic_scale_max: float


@dataclass(frozen=True)
class DevelopmentalTraining:
    model: RecurrentCellLanguageModel
    stages: tuple[DevelopmentalStage, ...]
    attempted_updates: int
    final_pruned_synapses: int
    final_recurrent_synapses: int
    configuration: dict[str, object]


def developmental_cell_order(cell_count: int, ordering: str, seed: int) -> np.ndarray:
    """Return a deterministic birth order over stable reservoir cell IDs."""
    if cell_count < 1:
        raise ValueError("cell count must be positive")
    if ordering == "random":
        return np.random.default_rng(seed).permutation(cell_count).astype(np.int32)
    if ordering != "layered_prefix":
        raise ValueError("unknown developmental cell ordering")

    coordinates = topology_coordinates(cell_count, "layered_sheet")
    surface_radius = np.square(coordinates[:, :2] - 0.5).sum(axis=1)
    # Recruit a compact sheet from its centre while cycling through cortical
    # depth at every surface location.  Cell IDs never change; only birth rank
    # does, so learned synapses remain stable across later recruitment.
    layer = np.arange(cell_count, dtype=np.int32) % 6
    layer_wave = np.asarray((2, 3, 1, 4, 0, 5), dtype=np.int32)
    layer_rank = np.empty(6, dtype=np.int32)
    layer_rank[layer_wave] = np.arange(6, dtype=np.int32)
    identifiers = np.arange(cell_count, dtype=np.int32)
    surface_cell = identifiers // 6
    return identifiers[
        np.lexsort((layer_rank[layer], surface_cell, surface_radius))
    ]


class DevelopmentalGrowthController:
    """Mutate sparse recurrent synapses while recruiting dormant GLIF5 cells."""

    def __init__(
        self,
        circuit: RecurrentCellCircuit,
        configuration: DevelopmentalGrowthConfig,
    ) -> None:
        if circuit.config.readout_learning_rule != "three_factor":
            raise ValueError("developmental growth requires the three-factor local learner")
        if circuit.observed_tokens or circuit._growth_examples:
            raise RuntimeError("configure development before the first observation")
        configuration.validate(
            circuit.config.reservoir_cells,
            circuit.config.recurrent_fanout,
        )
        if circuit.config.input_fanout > configuration.stage_active_cells[0]:
            raise ValueError("input fanout must fit the first recruitment stage")

        self.circuit = circuit
        self.configuration = configuration
        self._rng = np.random.default_rng(configuration.structural_seed)
        cell_count = circuit.config.reservoir_cells
        self.cell_order = developmental_cell_order(
            cell_count,
            configuration.cell_ordering,
            configuration.structural_seed,
        )
        self.active_mask = np.zeros(cell_count, dtype=np.bool_)
        self.structural_score = np.zeros_like(circuit._recurrent_weights_pa)
        self.structural_exposure = np.zeros_like(circuit._recurrent_weights_pa)
        self.cell_activity = np.zeros(cell_count, dtype=np.float64)
        self.homeostatic_gain = np.ones(cell_count, dtype=np.float64)
        self._stage_examples = 0
        self._stage_grown = 0
        self._stage_turnover_pruned = 0

        signs = np.ones(cell_count, dtype=np.float64)
        inhibitory_count = round(circuit.config.inhibitory_fraction * cell_count)
        if inhibitory_count:
            inhibitory = self._rng.choice(cell_count, inhibitory_count, replace=False)
            signs[inhibitory] = -1.0
        self._source_signs = signs

        circuit.reset_sequence()
        # Parent construction consumes a population-size-dependent number of
        # random draws before output heterogeneity. Recreate that fixed cell
        # property from an independent stream so full-start and staged
        # conditions differ only through their registered schedule.
        heterogeneity_rng = np.random.default_rng(circuit.config.random_seed + 700_001)
        circuit._output_heterogeneity_pa[:] = heterogeneity_rng.normal(
            0.0,
            0.01,
            circuit.token_cells,
        )
        circuit._afferent_targets.fill(0)
        circuit._recurrent_targets[:] = np.arange(cell_count, dtype=np.int32)[:, None]
        circuit._recurrent_weights_pa.fill(0.0)
        circuit._readout_synapses_pa.fill(0.0)
        circuit._pacemaker_synapses_pa.fill(0.0)
        first_count = configuration.stage_active_cells[0]
        self.active_mask[self.cell_order[:first_count]] = True
        circuit.active_reservoir_cells = first_count
        self._resample_afferents()
        initial_fanout = (
            circuit.config.recurrent_fanout
            if first_count == cell_count
            else configuration.initial_recurrent_fanout
        )
        self._grow_sources(self.active_cells, initial_fanout)
        self._publish_state()

    @property
    def active_cells(self) -> np.ndarray:
        return np.flatnonzero(self.active_mask).astype(np.int32, copy=False)

    @property
    def recurrent_synapse_count(self) -> int:
        return int(np.count_nonzero(self.circuit._recurrent_weights_pa))

    def _publish_state(self) -> None:
        """Keep checkpointable developmental state on cells and synapses."""
        circuit = self.circuit
        circuit._development_cell_order = self.cell_order
        circuit._development_active_mask = self.active_mask
        circuit._structural_scores = self.structural_score
        circuit._structural_exposure = self.structural_exposure
        circuit._homeostatic_gain = self.homeostatic_gain
        circuit.synchronize_structure_from_host()

    def _resample_afferents(self) -> None:
        active = self.active_cells
        for token in range(self.circuit.token_cells):
            self.circuit._afferent_targets[token] = self._rng.choice(
                active,
                self.circuit.config.input_fanout,
                replace=False,
            )

    def _sample_unique_targets(
        self,
        source: int,
        candidates: np.ndarray,
        count: int,
        excluded: set[int],
    ) -> list[int]:
        """Rejection-sample in O(count), without an all-cell candidate matrix."""
        selected: list[int] = []
        chosen = set(excluded)
        chosen.add(source)
        attempts = 0
        maximum_attempts = max(64, count * 64)
        while len(selected) < count:
            attempts += 1
            if attempts > maximum_attempts:
                raise ValueError("not enough distinct active cells for recurrent growth")
            target = int(candidates[int(self._rng.integers(len(candidates)))])
            if target in chosen:
                continue
            chosen.add(target)
            selected.append(target)
        return selected

    def _grow_sources(
        self,
        sources: np.ndarray,
        target_fanout: int,
        preferred_targets: np.ndarray | None = None,
    ) -> int:
        targets = self.circuit._recurrent_targets
        weights = self.circuit._recurrent_weights_pa
        active = self.active_cells
        grown = 0
        for source_value in sources:
            source = int(source_value)
            occupied = np.flatnonzero(weights[source] != 0.0)
            needed = target_fanout - len(occupied)
            if needed <= 0:
                continue
            existing = {int(value) for value in targets[source, occupied]}
            pool = preferred_targets if preferred_targets is not None else active
            try:
                selected = self._sample_unique_targets(source, pool, needed, existing)
            except ValueError:
                selected = self._sample_unique_targets(source, active, needed, existing)
            empty = np.flatnonzero(weights[source] == 0.0)[:needed]
            targets[source, empty] = selected
            weights[source, empty] = (
                self._source_signs[source] * self.circuit.config.recurrent_current_pa
            )
            self.structural_score[source, empty] = 0.0
            self.structural_exposure[source, empty] = 0.0
            grown += needed
        self._stage_grown += grown
        return grown

    def recruit_to(self, active_cell_count: int) -> int:
        """Recruit the next deterministic prefix and grow sparse synapses in place."""
        current = int(self.active_mask.sum())
        target_count = int(active_cell_count)
        if not current <= target_count <= self.circuit.config.reservoir_cells:
            raise ValueError("recruitment count must be monotonic and fit the reservoir")
        if target_count == current:
            return 0
        newly_active = self.cell_order[current:target_count]
        old_active = self.active_cells
        self.active_mask[newly_active] = True
        self.circuit.active_reservoir_cells = target_count
        self._resample_afferents()

        capacity = self.circuit.config.recurrent_fanout
        self._grow_sources(newly_active, capacity)
        self._grow_sources(old_active, capacity, preferred_targets=newly_active)
        turnover = min(self.configuration.turnover_edges_per_source, capacity)
        if turnover and len(newly_active) > turnover:
            weights = self.circuit._recurrent_weights_pa
            for source_value in old_active:
                source = int(source_value)
                occupied = np.flatnonzero(weights[source] != 0.0)
                if len(occupied) < 1:
                    continue
                normalized = self.structural_score[source, occupied] / np.maximum(
                    1.0,
                    self.structural_exposure[source, occupied],
                )
                ordering = np.lexsort((occupied, normalized))
                removed = occupied[ordering[: min(turnover, len(occupied))]]
                weights[source, removed] = 0.0
                self.structural_score[source, removed] = 0.0
                self.structural_exposure[source, removed] = 0.0
                self._stage_turnover_pruned += len(removed)
            self._grow_sources(old_active, capacity, preferred_targets=newly_active)
        self._publish_state()
        return len(newly_active)

    def record_plasticity(self) -> None:
        """Update local cell-rate and pre/post structural coactivity traces."""
        eligibility = self.circuit._eligibility
        active_edges = self.circuit._recurrent_weights_pa != 0.0
        target_activity = eligibility[self.circuit._recurrent_targets]
        contribution = eligibility[:, None] * target_activity
        self.structural_score[active_edges] += contribution[active_edges]
        self.structural_exposure[active_edges] += 1.0
        self.cell_activity[self.active_mask] += eligibility[self.active_mask]
        self._stage_examples += 1

    def apply_activity_homeostasis(self) -> tuple[int, float, float, float]:
        """Scale incoming recurrent synapses from a postsynaptic rate signal."""
        active = self.active_cells
        activity = self.cell_activity[active] / max(1, self._stage_examples)
        positive = activity[activity > 0.0]
        if not positive.size:
            return 0, 1.0, 1.0, 0.0
        target = float(np.median(positive))
        scale = np.clip(
            target / np.maximum(activity, np.finfo(np.float64).eps),
            self.configuration.homeostatic_min_scale,
            self.configuration.homeostatic_max_scale,
        )
        cell_scale = np.ones(self.circuit.config.reservoir_cells, dtype=np.float64)
        cell_scale[active] = scale
        edge_mask = self.circuit._recurrent_weights_pa != 0.0
        edge_scale = cell_scale[self.circuit._recurrent_targets]
        self.circuit._recurrent_weights_pa[edge_mask] *= edge_scale[edge_mask]
        self.homeostatic_gain[active] *= scale
        self._publish_state()
        return int(np.count_nonzero(np.abs(scale - 1.0) > 1e-12)), float(
            scale.min()
        ), float(scale.max()), float(activity.mean())

    def close_stage(self, stage: int, recruited_cells: int) -> DevelopmentalStage:
        mean_activity = float(
            np.mean(self.cell_activity[self.active_mask]) / max(1, self._stage_examples)
        )
        if self.configuration.homeostatic_scaling:
            scaled, minimum, maximum, mean_activity = self.apply_activity_homeostasis()
        else:
            scaled, minimum, maximum = 0, 1.0, 1.0
        record = DevelopmentalStage(
            stage=stage,
            active_cells=int(self.active_mask.sum()),
            recruited_cells=recruited_cells,
            training_examples=self._stage_examples,
            recurrent_synapses=self.recurrent_synapse_count,
            grown_synapses=self._stage_grown,
            turnover_pruned_synapses=self._stage_turnover_pruned,
            mean_activity=mean_activity,
            homeostatic_cells_scaled=scaled,
            homeostatic_scale_min=minimum,
            homeostatic_scale_max=maximum,
        )
        self.cell_activity.fill(0.0)
        self._stage_examples = 0
        self._stage_grown = 0
        self._stage_turnover_pruned = 0
        return record

    def prune_to_final_fanout(self) -> int:
        """Select mature recurrent synapses using only training-time traces."""
        target_fanout = self.configuration.final_recurrent_fanout
        weights = self.circuit._recurrent_weights_pa
        before_incoming = np.zeros(self.circuit.config.reservoir_cells, dtype=np.float64)
        edge_mask = weights != 0.0
        np.add.at(
            before_incoming,
            self.circuit._recurrent_targets[edge_mask],
            np.abs(weights[edge_mask]),
        )
        pruned = 0
        for source_value in self.active_cells:
            source = int(source_value)
            occupied = np.flatnonzero(weights[source] != 0.0)
            remove_count = len(occupied) - target_fanout
            if remove_count <= 0:
                continue
            if self.configuration.pruning_rule == "activity":
                score = self.structural_score[source, occupied] / np.maximum(
                    1.0,
                    self.structural_exposure[source, occupied],
                )
                ordering = np.lexsort((occupied, score))
                removed = occupied[ordering[:remove_count]]
            else:
                removed = self._rng.choice(occupied, remove_count, replace=False)
            weights[source, removed] = 0.0
            self.structural_score[source, removed] = 0.0
            self.structural_exposure[source, removed] = 0.0
            pruned += len(removed)

        if self.configuration.homeostatic_scaling:
            after_incoming = np.zeros_like(before_incoming)
            edge_mask = weights != 0.0
            np.add.at(
                after_incoming,
                self.circuit._recurrent_targets[edge_mask],
                np.abs(weights[edge_mask]),
            )
            ratio = np.ones_like(after_incoming)
            valid = (before_incoming > 0.0) & (after_incoming > 0.0)
            ratio[valid] = before_incoming[valid] / after_incoming[valid]
            edge_scale = ratio[self.circuit._recurrent_targets]
            weights[edge_mask] *= edge_scale[edge_mask]
            self.homeostatic_gain *= ratio
        self._publish_state()
        return pruned


def _training_examples(
    tokenizer: BytePhraseTokenizer,
    texts: Sequence[str],
    epochs: int,
) -> list[tuple[int, int, bool]]:
    examples: list[tuple[int, int, bool]] = []
    for _ in range(epochs):
        for text in texts:
            ids = tokenizer.encode(text)
            for index in range(len(ids) - 1):
                examples.append((ids[index], ids[index + 1], index == 0))
    return examples


def grow_developmentally(
    texts: Iterable[str],
    model_config: RecurrentModelConfig,
    development: DevelopmentalGrowthConfig,
) -> DevelopmentalTraining:
    """Train one cell model with a frozen, matched-example recruitment schedule."""
    corpus = tuple(str(text) for text in texts)
    if not corpus or not any(corpus):
        raise ValueError("at least one non-empty growth text is required")
    if model_config.circuit.readout_learning_rule != "three_factor":
        raise ValueError("developmental growth requires three-factor readout learning")
    development.validate(
        model_config.circuit.reservoir_cells,
        model_config.circuit.recurrent_fanout,
    )
    tokenizer = BytePhraseTokenizer.fit(
        corpus,
        TokenizerConfig(
            max_phrases=model_config.max_phrases,
            min_frequency=model_config.min_phrase_frequency,
        ),
    )
    circuit_config = replace(
        model_config.circuit,
        active_reservoir_cells=development.stage_active_cells[0],
    )
    circuit = RecurrentCellCircuit(tokenizer.vocab_size, circuit_config)
    controller = DevelopmentalGrowthController(circuit, development)
    examples = _training_examples(tokenizer, corpus, model_config.growth_epochs)
    if not examples:
        raise ValueError("growth corpus must contain at least one next-token example")
    boundaries = np.linspace(0, len(examples), len(development.stage_active_cells) + 1)
    boundaries = np.rint(boundaries).astype(np.int64)
    stages: list[DevelopmentalStage] = []
    for stage, active_cells in enumerate(development.stage_active_cells):
        recruited = controller.recruit_to(active_cells) if stage else active_cells
        circuit.reset_sequence()
        start = int(boundaries[stage])
        stop = int(boundaries[stage + 1])
        for token, target, reset_before in examples[start:stop]:
            if reset_before:
                circuit.reset_sequence()
            circuit.observe(token)
            circuit.strengthen(target)
            controller.record_plasticity()
        stages.append(controller.close_stage(stage, recruited))
    pruned = controller.prune_to_final_fanout()
    circuit.freeze_learning()
    digest = sha256()
    for text in corpus:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    model = RecurrentCellLanguageModel(tokenizer, circuit, digest.hexdigest())
    return DevelopmentalTraining(
        model=model,
        stages=tuple(stages),
        attempted_updates=len(examples),
        final_pruned_synapses=pruned,
        final_recurrent_synapses=controller.recurrent_synapse_count,
        configuration=asdict(development),
    )
