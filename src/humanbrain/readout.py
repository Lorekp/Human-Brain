"""Bounded cell-only output dendrites and their local plasticity state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from humanbrain.native import NativeSparseCellReadout


@dataclass(frozen=True)
class SparseReadoutDimensions:
    output_cells: int
    reservoir_cells: int
    fan_in: int
    candidate_count: int


class SparseCellReadout:
    """A fixed-capacity population of sparse fitted-output-cell dendrites."""

    CANDIDATE_FACTOR = 4
    MAX_RECRUITMENTS_PER_EVENT = 1
    MAX_DELETIONS_PER_EVENT = 1

    def __init__(
        self,
        *,
        dimensions: SparseReadoutDimensions,
        reservoir_eligibility: np.ndarray,
        output_heterogeneity_pa: np.ndarray,
        pacemaker_synapses_pa: np.ndarray,
        potential_sources: np.ndarray,
        apposition_eligibility: np.ndarray,
        contact_sources: np.ndarray,
        contact_weights_pa: np.ndarray,
        contact_ages: np.ndarray,
        output_activity_ema: np.ndarray,
        target_presentations: np.ndarray,
        applied_updates: int = 0,
        structural_recruitments: int = 0,
        structural_deletions: int = 0,
    ) -> None:
        self.dimensions = dimensions
        self.potential_sources = potential_sources
        self.apposition_eligibility = apposition_eligibility
        self.contact_sources = contact_sources
        self.contact_weights_pa = contact_weights_pa
        self.contact_ages = contact_ages
        self.output_activity_ema = output_activity_ema
        self.target_presentations = target_presentations
        self.applied_updates = int(applied_updates)
        self.structural_recruitments = int(structural_recruitments)
        self.structural_deletions = int(structural_deletions)
        self._validate_state()
        self._native = NativeSparseCellReadout(
            eligibility=reservoir_eligibility,
            output_heterogeneity_pa=output_heterogeneity_pa,
            potential_sources=self.potential_sources,
            apposition_eligibility=self.apposition_eligibility,
            contact_sources=self.contact_sources,
            contact_weights_pa=self.contact_weights_pa,
            contact_ages=self.contact_ages,
            pacemaker_synapses_pa=pacemaker_synapses_pa,
            output_activity_ema=self.output_activity_ema,
        )

    @classmethod
    def create(
        cls,
        *,
        output_cells: int,
        reservoir_cells: int,
        fan_in: int,
        rng: np.random.Generator,
        reservoir_eligibility: np.ndarray,
        output_heterogeneity_pa: np.ndarray,
        pacemaker_synapses_pa: np.ndarray,
    ) -> SparseCellReadout:
        candidate_count = min(reservoir_cells, cls.CANDIDATE_FACTOR * fan_in)
        potential_sources = np.empty((output_cells, candidate_count), dtype=np.int32)
        contact_sources = np.empty((output_cells, fan_in), dtype=np.int32)
        for output in range(output_cells):
            potential = cls._sample_range(reservoir_cells, candidate_count, rng)
            potential_sources[output] = potential
            contacts = cls._sample_values(potential, fan_in, rng)
            contact_sources[output] = contacts
        dimensions = SparseReadoutDimensions(
            output_cells=output_cells,
            reservoir_cells=reservoir_cells,
            fan_in=fan_in,
            candidate_count=candidate_count,
        )
        return cls(
            dimensions=dimensions,
            reservoir_eligibility=reservoir_eligibility,
            output_heterogeneity_pa=output_heterogeneity_pa,
            pacemaker_synapses_pa=pacemaker_synapses_pa,
            potential_sources=potential_sources,
            apposition_eligibility=np.zeros(
                (output_cells, candidate_count), dtype=np.float64
            ),
            contact_sources=contact_sources,
            contact_weights_pa=np.zeros((output_cells, fan_in), dtype=np.float64),
            contact_ages=np.zeros((output_cells, fan_in), dtype=np.uint32),
            output_activity_ema=np.zeros(output_cells, dtype=np.float64),
            target_presentations=np.zeros(output_cells, dtype=np.uint64),
        )

    @classmethod
    def from_checkpoint(
        cls,
        *,
        output_cells: int,
        reservoir_cells: int,
        fan_in: int,
        reservoir_eligibility: np.ndarray,
        output_heterogeneity_pa: np.ndarray,
        pacemaker_synapses_pa: np.ndarray,
        arrays: Mapping[str, np.ndarray],
        counters: Mapping[str, object],
    ) -> SparseCellReadout:
        """Restore one validated sparse anatomy and its exact structural clock."""
        candidate_count = min(reservoir_cells, cls.CANDIDATE_FACTOR * fan_in)

        def copy_array(name: str) -> np.ndarray:
            if name not in arrays:
                raise ValueError(f"sparse checkpoint is missing {name}")
            return np.array(arrays[name], copy=True)

        def count(name: str) -> int:
            value = counters.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"sparse checkpoint counter {name} must be nonnegative")
            return value

        return cls(
            dimensions=SparseReadoutDimensions(
                output_cells=output_cells,
                reservoir_cells=reservoir_cells,
                fan_in=fan_in,
                candidate_count=candidate_count,
            ),
            reservoir_eligibility=reservoir_eligibility,
            output_heterogeneity_pa=output_heterogeneity_pa,
            pacemaker_synapses_pa=pacemaker_synapses_pa,
            potential_sources=copy_array("readout_potential_sources"),
            apposition_eligibility=copy_array("readout_apposition_eligibility"),
            contact_sources=copy_array("readout_contact_sources"),
            contact_weights_pa=copy_array("readout_contact_weights_pa"),
            contact_ages=copy_array("readout_contact_ages"),
            output_activity_ema=copy_array("readout_output_activity_ema"),
            target_presentations=copy_array("readout_target_presentations"),
            applied_updates=count("applied_updates"),
            structural_recruitments=count("structural_recruitments"),
            structural_deletions=count("structural_deletions"),
        )

    @staticmethod
    def _sample_range(
        population: int,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count == population:
            return np.arange(population, dtype=np.int32)
        selected: set[int] = set()
        for upper in range(population - count, population):
            candidate = int(rng.integers(upper + 1))
            selected.add(upper if candidate in selected else candidate)
        return np.fromiter(sorted(selected), dtype=np.int32, count=count)

    @classmethod
    def _sample_values(
        cls,
        values: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count == values.size:
            return np.array(values, dtype=np.int32, copy=True)
        indices = cls._sample_range(values.size, count, rng)
        return np.asarray(values[indices], dtype=np.int32)

    def _validate_state(self) -> None:
        d = self.dimensions
        if d.output_cells < 2:
            raise ValueError("sparse readout needs at least two output cells")
        if not 1 <= d.fan_in <= d.reservoir_cells:
            raise ValueError("sparse fan-in must fit the reservoir")
        if d.candidate_count != min(
            d.reservoir_cells, self.CANDIDATE_FACTOR * d.fan_in
        ):
            raise ValueError("sparse potential-apposition count is not canonical")
        for name, value in (
            ("applied updates", self.applied_updates),
            ("structural recruitments", self.structural_recruitments),
            ("structural deletions", self.structural_deletions),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        expected = {
            "potential sources": (
                self.potential_sources,
                np.dtype(np.int32),
                (d.output_cells, d.candidate_count),
            ),
            "apposition eligibility": (
                self.apposition_eligibility,
                np.dtype(np.float64),
                (d.output_cells, d.candidate_count),
            ),
            "contact sources": (
                self.contact_sources,
                np.dtype(np.int32),
                (d.output_cells, d.fan_in),
            ),
            "contact weights": (
                self.contact_weights_pa,
                np.dtype(np.float64),
                (d.output_cells, d.fan_in),
            ),
            "contact ages": (
                self.contact_ages,
                np.dtype(np.uint32),
                (d.output_cells, d.fan_in),
            ),
            "output activity": (
                self.output_activity_ema,
                np.dtype(np.float64),
                (d.output_cells,),
            ),
            "target presentations": (
                self.target_presentations,
                np.dtype(np.uint64),
                (d.output_cells,),
            ),
        }
        for name, (array, dtype, shape) in expected.items():
            if array.dtype != dtype or array.shape != shape or not array.flags.c_contiguous:
                raise ValueError(
                    f"{name} must be a contiguous {dtype} array with shape {shape}"
                )
        if np.any(self.potential_sources < 0) or np.any(
            self.potential_sources >= d.reservoir_cells
        ):
            raise ValueError("potential appositions must index reservoir cells")
        if np.any(self.potential_sources[:, 1:] <= self.potential_sources[:, :-1]):
            raise ValueError("potential appositions must be unique and ascending")
        if np.any(self.contact_sources[:, 1:] <= self.contact_sources[:, :-1]):
            raise ValueError("contact sources must be unique and ascending")
        for output in range(d.output_cells):
            if not np.isin(
                self.contact_sources[output], self.potential_sources[output]
            ).all():
                raise ValueError("every contact source must be a potential apposition")
        if not np.isfinite(self.apposition_eligibility).all():
            raise ValueError("apposition eligibility must be finite")
        if np.any(self.apposition_eligibility < 0.0):
            raise ValueError("apposition eligibility must be nonnegative")
        if not np.isfinite(self.contact_weights_pa).all() or np.any(
            self.contact_weights_pa < 0.0
        ):
            raise ValueError("excitatory contact weights must be finite and nonnegative")
        if not np.isfinite(self.output_activity_ema).all() or np.any(
            (self.output_activity_ema < 0.0) | (self.output_activity_ema > 1.0)
        ):
            raise ValueError("output activity must be finite and in [0, 1]")

    @property
    def active_synapse_count(self) -> int:
        return int(np.count_nonzero(self.contact_weights_pa))

    @property
    def storage_bytes(self) -> int:
        arrays = (
            self.potential_sources,
            self.apposition_eligibility,
            self.contact_sources,
            self.contact_weights_pa,
            self.contact_ages,
            self.output_activity_ema,
            self.target_presentations,
        )
        return sum(array.nbytes for array in arrays)

    def current(self, base_current_pa: float) -> np.ndarray:
        return self._native.current(base_current_pa)

    def teach(
        self,
        *,
        target: int,
        winner: int,
        learning_rate_pa: float,
        maximum_weight_pa: float,
        trace_decays: np.ndarray,
        target_post_gate: float,
        winner_post_gate: float,
        activity_decay: float,
    ) -> bool:
        competitor = self._native.update(
            target=target,
            winner=winner,
            learning_rate_pa=learning_rate_pa,
            maximum_weight_pa=maximum_weight_pa,
            # Per-slot apposition decay over the source space: slot with
            # source s decays by trace_decays[s] (= exp(-T / tau_(s // R))).
            trace_decays=trace_decays,
            target_post_gate=target_post_gate,
            winner_post_gate=winner_post_gate,
            activity_decay=activity_decay,
        )
        applied = competitor is not None
        self.applied_updates += int(applied)
        return applied

    def advance_target(self, target: int) -> int:
        self.target_presentations[target] += 1
        return int(self.target_presentations[target])

    def turnover(
        self,
        *,
        output: int,
        structural_event: int,
        random_seed: int,
        minimum_age: int,
        turnover_probability: float,
        learning_rate_pa: float,
        maximum_weight_pa: float,
        homeostatic_rate: float,
    ) -> tuple[bool, bool]:
        recruited, deleted = self._native.turnover(
            output=output,
            structural_event=structural_event,
            random_seed=random_seed,
            minimum_age=minimum_age,
            turnover_probability=turnover_probability,
            learning_rate_pa=learning_rate_pa,
            maximum_weight_pa=maximum_weight_pa,
            homeostatic_rate=homeostatic_rate,
        )
        self.structural_recruitments += int(recruited)
        self.structural_deletions += int(deleted)
        return recruited, deleted

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        return {
            "readout_potential_sources": self.potential_sources,
            "readout_apposition_eligibility": self.apposition_eligibility,
            "readout_contact_sources": self.contact_sources,
            "readout_contact_weights_pa": self.contact_weights_pa,
            "readout_contact_ages": self.contact_ages,
            "readout_output_activity_ema": self.output_activity_ema,
            "readout_target_presentations": self.target_presentations,
        }

    def checkpoint_counters(self) -> dict[str, int]:
        return {
            "applied_updates": self.applied_updates,
            "structural_recruitments": self.structural_recruitments,
            "structural_deletions": self.structural_deletions,
        }
