"""Physical sparse temporal-memory fixture for the repaired R3 mainline.

The module deliberately owns the whole small vertical slice: distributed
token columns, context-conditioned distal segments, local current races,
basket-cell spike events, an injective relay twin for every winning pyramidal
cell, and plasticity.  The scored result is emitted by fitted GLIF3 membrane
crossings; Python never substitutes an integer contact threshold for a spike.

This is the fixture-scale implementation.  Novel tie-breaking is driven by
event-local procedural afferents from the previous physical spike population,
not by a whole-context lookup.  Columns commit already-materialized reserve
cells transactionally as resident segment capacity is consumed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from humanbrain.cortex import DendriticCortex


_UINT64_MASK = (1 << 64) - 1
_MIX_GAMMA = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB


def _mix_u64(value: int) -> int:
    """Counter-style uint64 mix used to define immutable physical afferents."""
    word = (int(value) + _MIX_GAMMA) & _UINT64_MASK
    word = ((word ^ (word >> 30)) * _MIX_A) & _UINT64_MASK
    word = ((word ^ (word >> 27)) * _MIX_B) & _UINT64_MASK
    return (word ^ (word >> 31)) & _UINT64_MASK


class GrowthPolicy(str, Enum):
    """Immutable source policy for one scientific arm."""

    BOUND_WINNERS = "bound_winners"
    GROWTH_OFF = "growth_off"
    GENERIC_AFFERENTS = "generic_afferents"


@dataclass(frozen=True)
class SparseTemporalConfig:
    """Frozen constants for one sparse temporal-memory circuit."""

    column_count: int
    active_columns: int = 11
    base_cells_per_column: int = 8
    max_pyramidal_cells: int = 131_072
    contacts_per_segment: int = 20
    theta_match: int = 16
    theta_active: int = 18
    connected_permanence: float = 0.50
    initial_permanence: float = 0.51
    permanence_increment: float = 0.10
    negative_decrement: float = 0.004
    max_segments_per_cell: int = 255
    novel_current_pa: float = 570.0
    novel_rank_spacing_pa: float = 47.0
    novel_race_cap: int = 8
    matching_current_pa: float = 620.0
    active_current_pa: float = 670.0
    homeostatic_current_pa: float = 20.0
    proximal_current_pa: float = 570.0
    relay_current_pa: float = 20_000.0
    basket_current_pa: float = 200_000.0
    basket_inhibition_pa: float = 500.0
    construction_seed: int = 17
    growth_policy: GrowthPolicy = GrowthPolicy.BOUND_WINNERS
    prediction_quorum: int | None = None

    def __post_init__(self) -> None:
        integer_positive = (
            self.column_count,
            self.active_columns,
            self.base_cells_per_column,
            self.max_pyramidal_cells,
            self.contacts_per_segment,
            self.theta_match,
            self.theta_active,
            self.max_segments_per_cell,
            self.novel_race_cap,
        )
        if any(type(value) is not int or value <= 0 for value in integer_positive):
            raise ValueError("population, segment, and threshold sizes must be positive ints")
        if (
            type(self.construction_seed) is not int
            or not 0 <= self.construction_seed < 2**32
        ):
            raise ValueError("construction_seed must be a uint32-compatible int")
        if self.active_columns > self.column_count:
            raise ValueError("active_columns cannot exceed column_count")
        required = self.column_count * self.base_cells_per_column
        if required > self.max_pyramidal_cells:
            raise ValueError(
                "base column population exceeds max_pyramidal_cells: "
                f"{required} > {self.max_pyramidal_cells}"
            )
        if self.base_cells_per_column > self.novel_race_cap:
            raise ValueError(
                "base_cells_per_column exceeds the calibrated novel_race_cap"
            )
        if self.contacts_per_segment > 2 * self.active_columns:
            raise ValueError(
                "contacts_per_segment cannot exceed the winner+relay context size"
            )
        if self.contacts_per_segment % 2:
            raise ValueError("contacts_per_segment must preserve source/relay pairs")
        if not 1 <= self.theta_match <= self.theta_active <= self.contacts_per_segment:
            raise ValueError("require 1 <= theta_match <= theta_active <= contacts")
        if not 0.0 < self.connected_permanence < 1.0:
            raise ValueError("connected_permanence must lie in (0, 1)")
        if not 0.0 < self.initial_permanence <= 1.0:
            raise ValueError("initial_permanence must lie in (0, 1]")
        if self.permanence_increment <= 0.0 or self.negative_decrement <= 0.0:
            raise ValueError("permanence changes must be positive")
        currents = (
            self.novel_current_pa,
            self.novel_rank_spacing_pa,
            self.matching_current_pa,
            self.active_current_pa,
            self.homeostatic_current_pa,
            self.proximal_current_pa,
            self.relay_current_pa,
            self.basket_current_pa,
            self.basket_inhibition_pa,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in currents):
            raise ValueError("physical drive currents must be finite and positive")
        lowest_novel = self.novel_current_pa - self.novel_rank_spacing_pa * (
            self.novel_race_cap - 1
        )
        if lowest_novel <= 0.0:
            raise ValueError("the calibrated novel current ladder must stay positive")
        if self.novel_rank_spacing_pa <= 2.0 * self.homeostatic_current_pa:
            raise ValueError("novel rank spacing must dominate bounded homeostasis")
        if self.matching_current_pa - self.novel_current_pa <= (
            2.0 * self.homeostatic_current_pa
        ):
            raise ValueError("matching current must be separated from novel drive")
        if self.active_current_pa - self.matching_current_pa <= (
            2.0 * self.homeostatic_current_pa
        ):
            raise ValueError("active current must be separated from matching drive")
        if not isinstance(self.growth_policy, GrowthPolicy):
            raise ValueError("growth_policy must be a GrowthPolicy")
        if (
            self.growth_policy is GrowthPolicy.GENERIC_AFFERENTS
            and self.base_cells_per_column < 2
        ):
            raise ValueError("generic-afferent control requires at least two pool cells")
        quorum = self.active_columns if self.prediction_quorum is None else self.prediction_quorum
        if type(quorum) is not int or not 1 <= quorum <= self.active_columns:
            raise ValueError("prediction_quorum must be in [1, active_columns]")

    @property
    def resolved_prediction_quorum(self) -> int:
        return (
            self.active_columns
            if self.prediction_quorum is None
            else int(self.prediction_quorum)
        )

    @property
    def growth_enabled(self) -> bool:
        return self.growth_policy is not GrowthPolicy.GROWTH_OFF


@dataclass(frozen=True)
class TemporalStep:
    """Physical result of one arriving-token event."""

    token_id: int
    predicted: bool
    predicted_columns: int
    winner_cells: tuple[int, ...]
    winner_spike_times_ms: tuple[float, ...]
    pyramidal_spikes_by_column: tuple[tuple[int, ...], ...]
    pyramidal_spike_times_by_column_ms: tuple[tuple[float, ...], ...]
    basket_spikes: tuple[int, ...]
    basket_spike_times_ms: tuple[float, ...]
    inhibition_onset_times_ms: tuple[float, ...]
    proximal_spikes: tuple[int, ...]
    proximal_spike_times_ms: tuple[float, ...]
    proximal_relay_spikes: tuple[int, ...]
    proximal_relay_spike_times_ms: tuple[float, ...]
    winner_context_spikes: tuple[int, ...]
    proximal_context_spikes: tuple[int, ...]
    relay_spikes: tuple[int, ...]
    relay_spike_times_ms: tuple[float, ...]
    context_spikes: tuple[int, ...]
    allocated_segments: int
    punished_segments: int


@dataclass(frozen=True)
class LearnedSnapshot:
    """Immutable learned cell/synapse state, excluding transient activity."""

    column_cells: tuple[tuple[int, ...], ...]
    segments: tuple[tuple[int, tuple[int, ...], tuple[float, ...]], ...]
    learning_step: int
    next_reserve_cell: int


@dataclass(frozen=True)
class CircuitSnapshot:
    """Deep deterministic state used by checkpoint-parity fixtures."""

    circuit_identity: str
    learned: LearnedSnapshot
    previous_context: tuple[int, ...]
    previous_winner_context: tuple[int, ...]
    event_step: int


@dataclass(frozen=True)
class ContextDiagnostics:
    """Read-only coincidence counts for pre-outcome false-match calibration."""

    candidate_segments: int
    matching_segments: int
    active_segments: int
    max_potential_contacts: int
    max_connected_contacts: int


class _Segment:
    __slots__ = ("owner", "contacts", "permanences")

    def __init__(
        self,
        owner: int,
        contacts: Sequence[int],
        permanence: float,
    ) -> None:
        self.owner = int(owner)
        self.contacts = [int(value) for value in contacts]
        self.permanences = [float(permanence)] * len(self.contacts)


class SparseTemporalCortex:
    """One deep neuronal seam for sparse context recognition and growth.

    Each token activates a sparse set of minicolumns.  One fitted GLIF3 cell
    wins in every active column.  A winner and its injective physical relay
    twin both spike, producing a 22-cell context when ``K=11``.  Distal
    segments sample 20 of those physical spikes in the bound-winner arm.
    Every arm also spikes one fitted generic proximal cell and relay per active
    column.  The generic-afferent control samples the same ten column positions
    from that nonrecursive one-step token code instead.

    Novel winner diversity comes from immutable procedural afferent weights
    activated only by the previous physical spikes.  A local least-loaded
    eligibility rule and calibrated current ladder turn those event sums into
    somatic drive.  Matching and active-segment bands dominate the novel band,
    and the coupled pyramidal-to-basket-to-GABA simulation must emit exactly
    one pyramidal spike per active column or the fixture stops.
    """

    def __init__(
        self,
        token_columns: Mapping[int, Sequence[int]],
        *,
        config: SparseTemporalConfig,
    ) -> None:
        if not token_columns:
            raise ValueError("token_columns must not be empty")
        self.config = config
        self._token_columns: dict[int, tuple[int, ...]] = {}
        seen_codes: dict[tuple[int, ...], int] = {}
        for raw_token, raw_columns in token_columns.items():
            token = int(raw_token)
            columns = tuple(sorted(int(value) for value in raw_columns))
            if len(columns) != config.active_columns or len(set(columns)) != len(columns):
                raise ValueError(
                    f"token {token} must activate exactly {config.active_columns} unique columns"
                )
            if any(value < 0 or value >= config.column_count for value in columns):
                raise ValueError(f"token {token} has a column outside the configured sheet")
            duplicate = seen_codes.get(columns)
            if duplicate is not None:
                raise ValueError(
                    f"tokens {duplicate} and {token} must not share one column code"
                )
            seen_codes[columns] = token
            self._token_columns[token] = columns

        identity_payload = {
            "config": asdict(config),
            "token_columns": [
                [token, list(columns)]
                for token, columns in sorted(self._token_columns.items())
            ],
        }
        self._circuit_identity = hashlib.sha256(
            json.dumps(
                identity_payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        base_total = config.column_count * config.base_cells_per_column
        self._base_pyramidal_cells = base_total
        self._column_cells: list[list[int]] = []
        for column in range(config.column_count):
            start = column * config.base_cells_per_column
            self._column_cells.append(
                list(range(start, start + config.base_cells_per_column))
            )
        self._relay_offset = config.max_pyramidal_cells
        self._basket_offset = 2 * config.max_pyramidal_cells
        self._proximal_offset = self._basket_offset + config.column_count
        self._proximal_relay_offset = self._proximal_offset + config.column_count
        physical_cells = self._proximal_relay_offset + config.column_count
        self._physics = DendriticCortex(
            cell_count=physical_cells,
            seed=config.construction_seed,
        )

        self._segments: list[_Segment] = []
        self._segments_by_cell: dict[int, list[int]] = {}
        self._contact_index: dict[int, list[tuple[int, int]]] = {}
        self._previous_context: tuple[int, ...] = ()
        self._previous_winner_context: tuple[int, ...] = ()
        self._event_step = 0
        self._learning_step = 0
        self._next_reserve_cell = base_total

    @classmethod
    def from_snapshot(
        cls,
        token_columns: Mapping[int, Sequence[int]],
        *,
        config: SparseTemporalConfig,
        snapshot: CircuitSnapshot,
    ) -> "SparseTemporalCortex":
        """Restore every learned and transient field from a deep snapshot."""
        circuit = cls(token_columns, config=config)
        if snapshot.circuit_identity != circuit._circuit_identity:
            raise ValueError("snapshot circuit identity does not match config/codebook")
        learned = snapshot.learned
        if len(learned.column_cells) != config.column_count:
            raise ValueError("snapshot column count does not match config")
        flattened = [cell for cells in learned.column_cells for cell in cells]
        if len(flattened) != len(set(flattened)):
            raise ValueError("snapshot assigns one pyramidal cell to multiple columns")
        if any(cell < 0 or cell >= config.max_pyramidal_cells for cell in flattened):
            raise ValueError("snapshot contains a pyramidal cell outside the configured sheet")
        for column, cells in enumerate(learned.column_cells):
            expected_base = tuple(
                range(
                    column * config.base_cells_per_column,
                    (column + 1) * config.base_cells_per_column,
                )
            )
            if tuple(cells[: config.base_cells_per_column]) != expected_base:
                raise ValueError("snapshot changed a column's physical base cells")
        next_reserve = int(learned.next_reserve_cell)
        if not circuit._base_pyramidal_cells <= next_reserve <= config.max_pyramidal_cells:
            raise ValueError("snapshot reserve cursor is outside the pyramidal sheet")
        recruited = sorted(
            cell for cell in flattened if cell >= circuit._base_pyramidal_cells
        )
        if recruited != list(range(circuit._base_pyramidal_cells, next_reserve)):
            raise ValueError("snapshot reserve commitment is not a contiguous prefix")
        committed = set(flattened)
        circuit._column_cells = [list(cells) for cells in learned.column_cells]
        circuit._segments = []
        circuit._segments_by_cell = {}
        for owner, contacts, permanences in learned.segments:
            if owner not in committed:
                raise ValueError("snapshot segment owner is not a committed column cell")
            if len(contacts) != len(permanences):
                raise ValueError("snapshot contact/permanence lengths differ")
            if (
                not contacts
                or len(contacts) > config.contacts_per_segment
                or len(contacts) % 2
                or len(set(contacts)) != len(contacts)
            ):
                raise ValueError("snapshot segment contacts violate grouped capacity")
            contact_set = {int(value) for value in contacts}
            source_set: set[int] = set()
            for contact in contact_set:
                try:
                    relay = circuit._relay_for_context_source(contact)
                except RuntimeError:
                    continue
                if relay in contact_set:
                    source_set.add(contact)
            expected_contacts = source_set.union(
                circuit._relay_for_context_source(source) for source in source_set
            )
            if expected_contacts != contact_set:
                raise ValueError("snapshot segment lost physical source/relay pairing")
            if any(
                not np.isfinite(value) or not 0.0 <= float(value) <= 1.0
                for value in permanences
            ):
                raise ValueError("snapshot permanence is outside [0, 1]")
            segment = _Segment(owner, contacts, config.initial_permanence)
            segment.permanences = [float(value) for value in permanences]
            segment_id = len(circuit._segments)
            circuit._segments.append(segment)
            circuit._segments_by_cell.setdefault(int(owner), []).append(segment_id)
        circuit._rebuild_contact_index()
        circuit._learning_step = int(learned.learning_step)
        circuit._next_reserve_cell = next_reserve
        circuit._previous_context = tuple(int(value) for value in snapshot.previous_context)
        if circuit._previous_context:
            if len(circuit._previous_context) != circuit.context_size:
                raise ValueError("snapshot context width does not match config")
            if len(set(circuit._previous_context)) != len(circuit._previous_context):
                raise ValueError("snapshot context contains duplicate physical cells")
            if any(
                cell < 0 or cell >= circuit.physical_cell_count
                for cell in circuit._previous_context
            ):
                raise ValueError("snapshot context contains an out-of-range cell")
        circuit._previous_winner_context = tuple(
            int(value) for value in snapshot.previous_winner_context
        )
        if circuit._previous_winner_context:
            if len(circuit._previous_winner_context) != circuit.context_size:
                raise ValueError("snapshot winner context width does not match config")
            if len(set(circuit._previous_winner_context)) != len(
                circuit._previous_winner_context
            ):
                raise ValueError("snapshot winner context contains duplicate cells")
            if any(
                cell < 0 or cell >= circuit.physical_cell_count
                for cell in circuit._previous_winner_context
            ):
                raise ValueError("snapshot winner context contains an out-of-range cell")
        circuit._event_step = int(snapshot.event_step)
        if circuit.snapshot() != snapshot:
            raise ValueError("snapshot did not round-trip into identical deep state")
        return circuit

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def physical_cell_count(self) -> int:
        """Materialized pool, relay, basket, and generic-afferent population."""
        return 2 * self.config.max_pyramidal_cells + 3 * self.config.column_count

    @property
    def committed_pyramidal_cell_count(self) -> int:
        return sum(len(cells) for cells in self._column_cells)

    @property
    def reserve_pyramidal_cells(self) -> int:
        return self.config.max_pyramidal_cells - self._next_reserve_cell

    @property
    def context_size(self) -> int:
        return 2 * self.config.active_columns

    def reset_transient(self) -> None:
        """Reset membrane-context boundary state while retaining learned synapses."""
        self._previous_context = ()
        self._previous_winner_context = ()
        self._event_step = 0

    def learned_snapshot(self) -> LearnedSnapshot:
        return LearnedSnapshot(
            column_cells=tuple(tuple(cells) for cells in self._column_cells),
            segments=tuple(
                (
                    segment.owner,
                    tuple(segment.contacts),
                    tuple(segment.permanences),
                )
                for segment in self._segments
            ),
            learning_step=self._learning_step,
            next_reserve_cell=self._next_reserve_cell,
        )

    def snapshot(self) -> CircuitSnapshot:
        return CircuitSnapshot(
            circuit_identity=self._circuit_identity,
            learned=self.learned_snapshot(),
            previous_context=self._previous_context,
            previous_winner_context=self._previous_winner_context,
            event_step=self._event_step,
        )

    def permanence_values(self) -> tuple[float, ...]:
        return tuple(
            value
            for segment in self._segments
            for value in segment.permanences
        )

    def diagnose_context(self, context_spikes: Sequence[int]) -> ContextDiagnostics:
        """Inspect segment coincidence without racing or changing any state."""
        context = tuple(int(value) for value in context_spikes)
        if len(context) != len(set(context)):
            raise ValueError("context_spikes must contain unique physical cell ids")
        if any(value < 0 or value >= self.physical_cell_count for value in context):
            raise ValueError("context_spikes contains a cell outside the circuit")
        potential, connected = self._recognition_counts(context)
        return ContextDiagnostics(
            candidate_segments=len(potential),
            matching_segments=sum(
                count >= self.config.theta_match for count in potential.values()
            ),
            active_segments=sum(
                count >= self.config.theta_active for count in connected.values()
            ),
            max_potential_contacts=max(potential.values(), default=0),
            max_connected_contacts=max(connected.values(), default=0),
        )

    def learn(self, token_id: int) -> TemporalStep:
        return self._step(int(token_id), mutate=True)

    def recognize(self, token_id: int) -> TemporalStep:
        return self._step(int(token_id), mutate=False)

    # ------------------------------------------------------------------ step

    def _step(self, token_id: int, *, mutate: bool) -> TemporalStep:
        columns = self._token_columns.get(token_id)
        if columns is None:
            raise ValueError(f"unknown token {token_id}")

        potential, connected = self._recognition_counts(self._previous_context)
        predicted_segments = {
            segment_id
            for segment_id, count in connected.items()
            if count >= self.config.theta_active
        }
        matching_segments = {
            segment_id
            for segment_id, count in potential.items()
            if count >= self.config.theta_match
        }
        best_predicted = self._best_segment_by_cell(
            predicted_segments, connected, potential
        )
        best_matching = self._best_segment_by_cell(
            matching_segments, potential, connected
        )

        winner_cells: list[int] = []
        winner_times: list[float] = []
        pyramidal_spikes_by_column: list[tuple[int, ...]] = []
        pyramidal_times_by_column: list[tuple[float, ...]] = []
        basket_cells: list[int] = []
        basket_times: list[float] = []
        inhibition_onsets: list[float] = []
        proximal_cells: list[int] = []
        proximal_times: list[float] = []
        proximal_relay_cells: list[int] = []
        proximal_relay_times: list[float] = []
        confirmed: set[int] = set()
        learnable: dict[int, int] = {}

        for column in columns:
            proximal_id = self._proximal_offset + column
            proximal = self._physics.race_currents(
                [proximal_id],
                [self.config.proximal_current_pa],
            )
            proximal_relay_id = self._proximal_relay_offset + column
            proximal_relay = self._physics.race_currents(
                [proximal_relay_id],
                [self.config.relay_current_pa],
            )
            proximal_cells.append(proximal_id)
            proximal_times.append(float(proximal.spike_time_ms))
            proximal_relay_cells.append(proximal_relay_id)
            proximal_relay_times.append(
                float(proximal.spike_time_ms + proximal_relay.spike_time_ms)
            )

        reserve_plan: dict[int, int] = {}
        if mutate and self.config.growth_enabled and self._previous_context:
            reserve_plan = self._plan_required_reserve_cells(
                columns,
                best_predicted=best_predicted,
                best_matching=best_matching,
            )

        for column in columns:
            staged = list(self._column_cells[column])
            reserved = reserve_plan.get(int(column))
            if reserved is not None:
                staged.append(reserved)
            cells = tuple(staged)
            currents = self._column_currents(
                cells,
                best_predicted,
                best_matching,
            )
            basket_id = self._basket_offset + column
            race = self._physics.race_currents_with_inhibition(
                cells,
                currents,
                basket_cell_id=basket_id,
                basket_current_pa=self.config.basket_current_pa,
                inhibition_current_pa=self.config.basket_inhibition_pa,
            )
            if len(race.pyramidal_spikes) != 1:
                raise RuntimeError(
                    "STOP: coupled minicolumn race emitted multiple pyramidal spikes"
                )
            if race.basket_spike_time_ms is None or race.inhibition_onset_ms is None:
                raise RuntimeError(
                    "STOP: coupled minicolumn race did not recruit basket inhibition"
                )
            winner = int(race.winner)
            winner_cells.append(winner)
            winner_times.append(float(race.winner_spike_time_ms))
            pyramidal_spikes_by_column.append(race.pyramidal_spikes)
            pyramidal_times_by_column.append(race.pyramidal_spike_times_ms)

            predicted_segment = best_predicted.get(winner)
            if predicted_segment is not None:
                confirmed.add(predicted_segment)
            else:
                matching_segment = best_matching.get(winner)
                if matching_segment is not None:
                    learnable[column] = matching_segment

            basket_cells.append(basket_id)
            basket_times.append(float(race.basket_spike_time_ms))
            inhibition_onsets.append(float(race.inhibition_onset_ms))

        relay_cells: list[int] = []
        relay_times: list[float] = []
        for winner, winner_time in zip(winner_cells, winner_times, strict=True):
            relay_id = self._relay_offset + winner
            relay = self._physics.race_currents(
                [relay_id],
                [self.config.relay_current_pa],
            )
            relay_cells.append(relay_id)
            relay_times.append(float(winner_time + relay.spike_time_ms))

        predicted_columns = len(confirmed)
        allocated = 0
        punished = 0
        if mutate:
            if self.config.growth_enabled:
                planned_allocations: list[int] = []
                for column, winner in zip(columns, winner_cells, strict=True):
                    if best_predicted.get(winner) in confirmed:
                        continue
                    if learnable.get(column) is None and self._previous_context:
                        planned_allocations.append(winner)
                projected = Counter(planned_allocations)
                for owner, count in projected.items():
                    if (
                        len(self._segments_by_cell.get(owner, ())) + count
                        > self.config.max_segments_per_cell
                    ):
                        raise RuntimeError(
                            "STOP: column capacity exhausted before atomic plasticity"
                        )

                self._commit_planned_reserve_cells(reserve_plan)

                winning_cells = set(winner_cells)
                for segment_id in sorted(predicted_segments):
                    if segment_id in confirmed:
                        self._reinforce(segment_id, self._previous_context)
                    elif self._segments[segment_id].owner not in winning_cells:
                        self._punish(segment_id, self._previous_context)
                        punished += 1

                for column, winner in zip(columns, winner_cells, strict=True):
                    if best_predicted.get(winner) in confirmed:
                        continue
                    matched = learnable.get(column)
                    if matched is not None:
                        self._reinforce(matched, self._previous_context)
                    elif self._previous_context:
                        self._allocate_segment(winner, self._previous_context)
                        allocated += 1
                self._learning_step += 1

        winner_context = tuple(winner_cells + relay_cells)
        proximal_context = tuple(proximal_cells + proximal_relay_cells)
        context_spikes = (
            proximal_context
            if self.config.growth_policy is GrowthPolicy.GENERIC_AFFERENTS
            else winner_context
        )
        self._previous_context = context_spikes
        self._previous_winner_context = winner_context
        self._event_step += 1
        return TemporalStep(
            token_id=token_id,
            predicted=predicted_columns >= self.config.resolved_prediction_quorum,
            predicted_columns=predicted_columns,
            winner_cells=tuple(winner_cells),
            winner_spike_times_ms=tuple(winner_times),
            pyramidal_spikes_by_column=tuple(pyramidal_spikes_by_column),
            pyramidal_spike_times_by_column_ms=tuple(pyramidal_times_by_column),
            basket_spikes=tuple(basket_cells),
            basket_spike_times_ms=tuple(basket_times),
            inhibition_onset_times_ms=tuple(inhibition_onsets),
            proximal_spikes=tuple(proximal_cells),
            proximal_spike_times_ms=tuple(proximal_times),
            proximal_relay_spikes=tuple(proximal_relay_cells),
            proximal_relay_spike_times_ms=tuple(proximal_relay_times),
            winner_context_spikes=winner_context,
            proximal_context_spikes=proximal_context,
            relay_spikes=tuple(relay_cells),
            relay_spike_times_ms=tuple(relay_times),
            context_spikes=context_spikes,
            allocated_segments=allocated,
            punished_segments=punished,
        )

    # ----------------------------------------------------------- recognition

    def _recognition_counts(
        self,
        context: Sequence[int],
    ) -> tuple[dict[int, int], dict[int, int]]:
        potential: dict[int, int] = {}
        connected: dict[int, int] = {}
        for presynaptic in context:
            for segment_id, contact_index in self._contact_index.get(
                int(presynaptic), ()
            ):
                potential[segment_id] = potential.get(segment_id, 0) + 1
                segment = self._segments[segment_id]
                if (
                    segment.permanences[contact_index]
                    >= self.config.connected_permanence
                ):
                    connected[segment_id] = connected.get(segment_id, 0) + 1
        return potential, connected

    def _best_segment_by_cell(
        self,
        segment_ids: set[int],
        primary: Mapping[int, int],
        secondary: Mapping[int, int],
    ) -> dict[int, int]:
        result: dict[int, int] = {}
        for segment_id in sorted(segment_ids):
            owner = self._segments[segment_id].owner
            incumbent = result.get(owner)
            if incumbent is None:
                result[owner] = segment_id
                continue
            candidate_key = (
                int(primary.get(segment_id, 0)),
                int(secondary.get(segment_id, 0)),
                -segment_id,
            )
            incumbent_key = (
                int(primary.get(incumbent, 0)),
                int(secondary.get(incumbent, 0)),
                -incumbent,
            )
            if candidate_key > incumbent_key:
                result[owner] = segment_id
        return result

    def _column_currents(
        self,
        cells: tuple[int, ...],
        best_predicted: Mapping[int, int],
        best_matching: Mapping[int, int],
    ) -> np.ndarray:
        cfg = self.config
        currents = np.zeros(len(cells), dtype=np.float64)
        loads = np.asarray(
            [len(self._segments_by_cell.get(cell, ())) for cell in cells],
            dtype=np.int64,
        )
        offsets = {cell: offset for offset, cell in enumerate(cells)}
        predicted_cells = tuple(cell for cell in cells if cell in best_predicted)
        matching_cells = tuple(cell for cell in cells if cell in best_matching)
        if len(predicted_cells) > 1:
            raise RuntimeError(
                "STOP: multiple active segment owners in one minicolumn"
            )
        if len(matching_cells) > 1:
            raise RuntimeError(
                "STOP: multiple matching segment owners in one minicolumn"
            )

        categorized = set(predicted_cells).union(matching_cells)
        if self._previous_context:
            novel_pool = [cell for cell in cells if cell not in categorized]
            if novel_pool:
                minimum_load = min(
                    len(self._segments_by_cell.get(cell, ())) for cell in novel_pool
                )
                novel_pool = [
                    cell
                    for cell in novel_pool
                    if len(self._segments_by_cell.get(cell, ())) == minimum_load
                ]
            ranked_novel = sorted(
                novel_pool,
                key=lambda cell: (-self._afferent_score(cell), int(cell)),
            )
        else:
            # Book/pass boundaries have no distal source.  Keep the pool-cell
            # code independent of learned load and exclude recruited reserve
            # cells so boundary output remains stable over development.
            ranked_novel = list(cells[: cfg.base_cells_per_column])

        if len(ranked_novel) > cfg.novel_race_cap:
            raise RuntimeError(
                "STOP: local novel candidate set exceeds the calibrated race cap"
            )
        for rank, cell in enumerate(ranked_novel):
            currents[offsets[cell]] = (
                cfg.novel_current_pa - rank * cfg.novel_rank_spacing_pa
            )

        for cell in matching_cells:
            currents[offsets[cell]] = cfg.matching_current_pa
        for cell in predicted_cells:
            currents[offsets[cell]] = cfg.active_current_pa

        driven = currents > 0.0
        currents[driven] += cfg.homeostatic_current_pa / (
            1.0 + loads[driven].astype(np.float64)
        )
        return currents

    def _afferent_score(self, postsynaptic: int) -> int:
        """Sum immutable event-driven synaptic words from the physical context.

        Each term is a procedural physical edge weight keyed only by the
        construction seed and the pre/post cell identities.  Summation makes
        the drive independent of spike tuple order; no context identity is
        stored or looked up.
        """
        score = 0
        post_key = (int(postsynaptic) + 1) * _MIX_B
        seed_key = (int(self.config.construction_seed) + 1) * _MIX_GAMMA
        for presynaptic in self._previous_context:
            pre_key = (int(presynaptic) + 1) * _MIX_A
            score += _mix_u64(seed_key ^ pre_key ^ post_key)
        return score

    def _plan_required_reserve_cells(
        self,
        columns: Sequence[int],
        *,
        best_predicted: Mapping[int, int],
        best_matching: Mapping[int, int],
    ) -> dict[int, int]:
        exhausted: list[int] = []
        for column in columns:
            cells = self._column_cells[column]
            if any(
                cell in best_predicted or cell in best_matching
                for cell in cells
            ):
                continue
            if all(
                len(self._segments_by_cell.get(cell, ()))
                >= self.config.max_segments_per_cell
                for cell in cells
            ):
                exhausted.append(int(column))
        available = self.config.max_pyramidal_cells - self._next_reserve_cell
        if len(exhausted) > available:
            raise RuntimeError(
                "STOP: column capacity exhausted; physical reserve exhausted "
                "before atomic plasticity"
            )
        return {
            column: self._next_reserve_cell + offset
            for offset, column in enumerate(exhausted)
        }

    def _commit_planned_reserve_cells(self, plan: Mapping[int, int]) -> None:
        for column, cell in plan.items():
            if cell != self._next_reserve_cell:
                raise RuntimeError("STOP: reserve plan lost contiguous physical order")
            self._column_cells[int(column)].append(int(cell))
            self._next_reserve_cell += 1

    # --------------------------------------------------------------- learning

    def _relay_for_context_source(self, source: int) -> int:
        if 0 <= source < self.config.max_pyramidal_cells:
            return self._relay_offset + source
        proximal_stop = self._proximal_offset + self.config.column_count
        if self._proximal_offset <= source < proximal_stop:
            return self._proximal_relay_offset + (source - self._proximal_offset)
        raise RuntimeError("STOP: context source has no injective relay twin")

    def _context_source_pairs(
        self,
        context: Sequence[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        sources = tuple(int(value) for value in context[: self.config.active_columns])
        relays = tuple(int(value) for value in context[self.config.active_columns :])
        if len(sources) != len(relays) or relays != tuple(
            self._relay_for_context_source(source) for source in sources
        ):
            raise RuntimeError("STOP: context does not contain injective source/relay pairs")
        return sources, relays

    def _allocate_segment(self, owner: int, context: Sequence[int]) -> None:
        owned = self._segments_by_cell.setdefault(int(owner), [])
        if len(owned) >= self.config.max_segments_per_cell:
            raise RuntimeError(
                "STOP: owner segment capacity exhausted after reserve preflight"
            )
        unique_context = np.unique(np.asarray(context, dtype=np.int64))
        if unique_context.size < self.config.contacts_per_segment:
            raise RuntimeError(
                "STOP: physical context is smaller than contacts_per_segment"
            )
        sources, _relays = self._context_source_pairs(context)
        # The sampled column positions are arm-independent.  A1 and A3 differ
        # only in whether those positions resolve to winner or generic source
        # cells; owner identity must not perturb contact mass or positions.
        rng = np.random.default_rng(
            [self.config.construction_seed, self._learning_step, 0xA110C]
        )
        chosen_pairs = np.sort(
            rng.choice(
                len(sources),
                size=self.config.contacts_per_segment // 2,
                replace=False,
            )
        )
        chosen_sources = [sources[int(index)] for index in chosen_pairs]
        chosen = sorted(
            chosen_sources
            + [self._relay_for_context_source(source) for source in chosen_sources]
        )
        segment_id = len(self._segments)
        segment = _Segment(owner, chosen, self.config.initial_permanence)
        self._segments.append(segment)
        owned.append(segment_id)
        for contact_index, presynaptic in enumerate(segment.contacts):
            self._contact_index.setdefault(presynaptic, []).append(
                (segment_id, contact_index)
            )

    def _reinforce(self, segment_id: int, context: Sequence[int]) -> None:
        active = set(int(value) for value in context)
        segment = self._segments[segment_id]
        for index, presynaptic in enumerate(segment.contacts):
            delta = (
                self.config.permanence_increment
                if presynaptic in active
                else -self.config.permanence_increment
            )
            segment.permanences[index] = float(
                np.clip(segment.permanences[index] + delta, 0.0, 1.0)
            )
        self._drop_zero_contacts(segment_id)
        self._regrow_freed_contacts(segment_id, context)

    def _punish(self, segment_id: int, context: Sequence[int]) -> None:
        active = set(int(value) for value in context)
        segment = self._segments[segment_id]
        for index, presynaptic in enumerate(segment.contacts):
            if (
                presynaptic in active
                and segment.permanences[index]
                >= self.config.connected_permanence
            ):
                segment.permanences[index] -= self.config.negative_decrement
        self._drop_zero_contacts(segment_id)

    def _drop_zero_contacts(self, segment_id: int) -> None:
        segment = self._segments[segment_id]
        if all(value > 0.0 for value in segment.permanences):
            return
        retained = [
            (contact, permanence)
            for contact, permanence in zip(
                segment.contacts, segment.permanences, strict=True
            )
            if permanence > 0.0
        ]
        segment.contacts = [contact for contact, _ in retained]
        segment.permanences = [permanence for _, permanence in retained]
        self._rebuild_contact_index()

    def _regrow_freed_contacts(
        self,
        segment_id: int,
        context: Sequence[int],
    ) -> None:
        segment = self._segments[segment_id]
        permanence_by_contact = dict(
            zip(segment.contacts, segment.permanences, strict=True)
        )
        complete_sources: set[int] = set()
        for contact in segment.contacts:
            try:
                relay = self._relay_for_context_source(contact)
            except RuntimeError:
                continue
            if relay in permanence_by_contact:
                complete_sources.add(contact)
        target_pairs = self.config.contacts_per_segment // 2
        missing_pairs = target_pairs - len(complete_sources)
        if missing_pairs <= 0 and len(segment.contacts) == self.config.contacts_per_segment:
            return
        context_sources, _context_relays = self._context_source_pairs(context)
        candidates = np.asarray(
            sorted(set(context_sources) - complete_sources),
            dtype=np.int64,
        )
        count = min(max(0, missing_pairs), int(candidates.size))
        rng = np.random.default_rng(
            [
                self.config.construction_seed,
                self._learning_step,
                int(segment.owner),
                int(segment_id),
            ]
        )
        selected = (
            rng.choice(candidates, size=count, replace=False)
            if count
            else np.asarray([], dtype=np.int64)
        )
        pairs: list[tuple[int, float]] = []
        for source in sorted(complete_sources):
            relay = self._relay_for_context_source(source)
            pairs.append((source, permanence_by_contact[source]))
            pairs.append((relay, permanence_by_contact[relay]))
        for raw_source in selected:
            source = int(raw_source)
            pairs.append((source, self.config.initial_permanence))
            pairs.append(
                (self._relay_for_context_source(source), self.config.initial_permanence)
            )
        pairs.sort(key=lambda item: item[0])
        segment.contacts = [contact for contact, _ in pairs]
        segment.permanences = [permanence for _, permanence in pairs]
        self._rebuild_contact_index()

    def _rebuild_contact_index(self) -> None:
        self._contact_index = {}
        for segment_id, segment in enumerate(self._segments):
            for contact_index, presynaptic in enumerate(segment.contacts):
                self._contact_index.setdefault(presynaptic, []).append(
                    (segment_id, contact_index)
                )


__all__ = [
    "CircuitSnapshot",
    "ContextDiagnostics",
    "GrowthPolicy",
    "LearnedSnapshot",
    "SparseTemporalConfig",
    "SparseTemporalCortex",
    "TemporalStep",
]
