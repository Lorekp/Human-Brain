"""Model-neutral observations consumed by the diagnostics workbench.

The visualizer deliberately normalizes experiment formats into counts,
proportions, population structure, and named diagnostic channels.  UI code
does not know which neuron equations produced those observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Mapping


CURRENT_CONCURRENT_ACTIVITY_REFERENCE = 0.02
WILSON_95_Z = 1.959963984540054


@dataclass(frozen=True, slots=True)
class ProportionEstimate:
    """One observed proportion and its Wilson 95% score interval."""

    successes: int
    total: int
    fraction: float
    low: float
    high: float

    @classmethod
    def from_counts(cls, successes: int, total: int) -> ProportionEstimate:
        if total < 0 or successes < 0 or successes > total:
            raise ValueError("proportion counts must satisfy 0 <= successes <= total")
        if total == 0:
            return cls(successes=successes, total=total, fraction=0.0, low=0.0, high=1.0)
        fraction = successes / total
        z2 = WILSON_95_Z * WILSON_95_Z
        denominator = 1.0 + z2 / total
        center = (fraction + z2 / (2.0 * total)) / denominator
        half_width = (
            WILSON_95_Z
            * math.sqrt(
                fraction * (1.0 - fraction) / total + z2 / (4.0 * total * total)
            )
            / denominator
        )
        return cls(
            successes=successes,
            total=total,
            fraction=fraction,
            low=max(0.0, center - half_width),
            high=min(1.0, center + half_width),
        )


@dataclass(frozen=True, slots=True)
class DifferenceEstimate:
    """Newcombe hybrid-score interval for two independent proportions."""

    difference: float
    low: float
    high: float

    @classmethod
    def between(
        cls,
        left: ProportionEstimate,
        right: ProportionEstimate,
    ) -> DifferenceEstimate | None:
        if left.total == 0 or right.total == 0:
            return None
        difference = left.fraction - right.fraction
        low = difference - math.sqrt(
            (left.fraction - left.low) ** 2 + (right.high - right.fraction) ** 2
        )
        high = difference + math.sqrt(
            (left.high - left.fraction) ** 2 + (right.fraction - right.low) ** 2
        )
        return cls(difference=difference, low=low, high=high)


@dataclass(frozen=True, slots=True)
class PopulationSummary:
    """Model-neutral structural scale and contact estimates for one run."""

    model_family: str
    configured_neurons: int
    enabled_neurons: int
    excitatory_neurons: int
    inhibitory_neurons: int
    concurrent_activity_reference_neurons: int
    input_contacts: int
    recurrent_contacts: int
    readout_contacts: int
    recurrent_fanout: int
    input_assembly_size: int

    @property
    def total_contacts(self) -> int:
        return self.input_contacts + self.recurrent_contacts + self.readout_contacts

    @property
    def inhibitory_fraction(self) -> float:
        if self.enabled_neurons == 0:
            return 0.0
        return self.inhibitory_neurons / self.enabled_neurons

    @property
    def excitation_inhibition_ratio(self) -> float | None:
        if self.inhibitory_neurons == 0:
            return None
        return self.excitatory_neurons / self.inhibitory_neurons

    @property
    def recurrent_connection_density(self) -> float:
        if self.enabled_neurons == 0:
            return 0.0
        return self.recurrent_fanout / self.enabled_neurons

    @property
    def input_assembly_fraction(self) -> float:
        if self.enabled_neurons == 0:
            return 0.0
        return self.input_assembly_size / self.enabled_neurons


@dataclass(frozen=True, slots=True)
class RunOutcome:
    training: ProportionEstimate
    calibration: ProportionEstimate
    replay: ProportionEstimate
    generalization_gap: DifferenceEstimate | None
    margin_count: int
    margin_min_pa: float
    margin_median_pa: float
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class StateSeparation:
    """Representation distance at the first typed decision state."""

    eligibility_median_1minuscos: float | None = None
    synaptic_current_median_1minuscos: float | None = None

    @property
    def available(self) -> bool:
        return (
            self.eligibility_median_1minuscos is not None
            or self.synaptic_current_median_1minuscos is not None
        )


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """One normalized experiment observation.

    The interface exposes facts the workbench needs and keeps source-format
    details in ``configuration`` and ``extensions`` for the inspector.
    """

    source_path: Path
    experiment_id: str
    arm: str
    seed: int
    data_scale: int
    readout_rule: str
    schedule: str
    supervision: str
    codebook: str
    simulation_backend: str
    ridge_backend: str
    population: PopulationSummary
    outcome: RunOutcome
    state_separation: StateSeparation
    curriculum_sha256: str
    git_head: str
    configuration: Mapping[str, object] = field(repr=False)
    extensions: Mapping[str, object] = field(default_factory=dict, repr=False)

    @property
    def run_id(self) -> str:
        return f"{self.experiment_id}/{self.arm}"

    def metric(self, metric_id: str) -> float | None:
        """Return one plottable scalar by stable workbench identifier."""
        values: dict[str, float | None] = {
            "calibration_accuracy": self.outcome.calibration.fraction,
            "training_accuracy": self.outcome.training.fraction,
            "replay_rate": self.outcome.replay.fraction,
            "generalization_gap": (
                self.outcome.generalization_gap.difference
                if self.outcome.generalization_gap is not None
                else None
            ),
            "margin_median_pa": self.outcome.margin_median_pa,
            "margin_min_pa": self.outcome.margin_min_pa,
            "state_separation": self.state_separation.synaptic_current_median_1minuscos,
            "recurrent_density": self.population.recurrent_connection_density,
            "assembly_fraction": self.population.input_assembly_fraction,
            "wall_seconds": self.outcome.wall_seconds,
        }
        try:
            return values[metric_id]
        except KeyError as error:
            raise ValueError(f"unknown workbench metric: {metric_id}") from error

    def dimension(self, dimension_id: str) -> float:
        """Return one numeric experimental dimension for plotting."""
        values = {
            "neurons": float(self.population.configured_neurons),
            "data_scale": float(self.data_scale),
            "inhibitory_fraction": self.population.inhibitory_fraction,
            "assembly_size": float(self.population.input_assembly_size),
            "seed": float(self.seed),
        }
        try:
            return values[dimension_id]
        except KeyError as error:
            raise ValueError(f"unknown workbench dimension: {dimension_id}") from error

    def group(self, group_id: str) -> str:
        values = {
            "codebook": self.codebook,
            "readout": self.readout_rule,
            "schedule": self.schedule,
            "supervision": self.supervision,
            "backend": self.simulation_backend,
        }
        try:
            return values[group_id]
        except KeyError as error:
            raise ValueError(f"unknown workbench grouping: {group_id}") from error
