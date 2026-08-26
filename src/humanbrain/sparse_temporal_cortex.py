"""Physical sparse temporal-memory fixture for the repaired R3 mainline.

The module deliberately owns the whole small vertical slice: distributed
token columns, context-conditioned distal segments, local current races,
basket-cell spike events, an injective relay twin for every winning pyramidal
cell, and plasticity.  The scored result is emitted by fitted GLIF3 membrane
crossings; Python never substitutes an integer contact threshold for a spike.

This is the fixture-scale implementation.  Novel tie-breaking is driven by
event-local procedural afferents from the previous physical spike population,
not by a whole-context lookup.  Columns commit already-materialized reserve
cells as full calibrated race cohorts.  Each source/relay contact pair shares
one exact integer permanence tick and all learned-state changes are atomic.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Final, Mapping, Sequence

import numpy as np

from humanbrain.cortex import DendriticCortex
from humanbrain.procedural_topology import _philox2x64_10


_UINT32_MASK = (1 << 32) - 1
_UINT64_MASK = (1 << 64) - 1
_MIX_GAMMA = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB
_INITIAL_CONTACT_DOMAIN: Final = 0x52334630494E4954  # ASCII "R3F0INIT"
_REGROWTH_CONTACT_DOMAIN: Final = 0x5233463052475257  # ASCII "R3F0RGRW"


def _require_uint32(value: object, *, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT32_MASK:
        raise ValueError(f"{label} must be a uint32-compatible int")
    return value


def contact_position_sampling_contract() -> dict[str, str]:
    """Return the frozen portable address scheme for paired contact sampling."""
    return {
        "protocol": "random123-philox2x64-10-contact-positions-v1",
        "oracle": "humanbrain.procedural_topology._philox2x64_10",
        "initial_domain_hex": f"{_INITIAL_CONTACT_DOMAIN:016x}",
        "regrowth_domain_hex": f"{_REGROWTH_CONTACT_DOMAIN:016x}",
        "bounded_mapping": "lemire-multiply-high-u64-rejection",
        "without_replacement": "partial-fisher-yates",
        "initial_address": (
            "key=seed^initial_domain,"
            "counter=(learning_step,(attempt<<32)|draw)"
        ),
        "regrowth_address": (
            "key=philox(owner,segment_id,seed^regrowth_domain)[0],"
            "counter=(learning_step,(attempt<<32)|draw)"
        ),
    }


def _bounded_philox_index(
    bound: int,
    *,
    key: int,
    learning_step: int,
    draw: int,
) -> int:
    """Map Philox words exactly uniformly into ``range(bound)``."""
    if not 1 <= bound <= 1 << 64 or not 0 <= draw < 1 << 32:
        raise RuntimeError("STOP: invalid bounded Philox address")
    rejection_threshold = ((1 << 64) - bound) % bound
    for attempt in range(1 << 32):
        counter_1 = (attempt << 32) | draw
        word, _ = _philox2x64_10(learning_step, counter_1, key)
        product = word * bound
        if (product & _UINT64_MASK) >= rejection_threshold:
            return product >> 64
    raise RuntimeError("STOP: bounded Philox rejection counter exhausted")


def _sample_contact_positions(
    population_size: int,
    sample_size: int,
    *,
    key: int,
    learning_step: int,
) -> tuple[int, ...]:
    """Sample sorted positions with Philox and partial Fisher-Yates."""
    if (
        not 0 <= sample_size <= population_size <= 1 << 64
        or sample_size >= 1 << 32
    ):
        raise RuntimeError("STOP: invalid contact-position sample dimensions")
    positions = list(range(population_size))
    for draw in range(sample_size):
        remaining = population_size - draw
        selected = draw + _bounded_philox_index(
            remaining,
            key=key,
            learning_step=learning_step,
            draw=draw,
        )
        positions[draw], positions[selected] = positions[selected], positions[draw]
    return tuple(sorted(positions[:sample_size]))


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
    theta_match: int
    theta_active: int
    active_columns: int = 11
    base_cells_per_column: int = 8
    max_pyramidal_cells: int = 131_072
    contacts_per_segment: int = 20
    connected_permanence_ticks: int = 250
    initial_permanence_ticks: int = 255
    permanence_increment_ticks: int = 50
    negative_decrement_ticks: int = 2
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
        uint32_population_fields = (
            ("column_count", self.column_count),
            ("max_pyramidal_cells", self.max_pyramidal_cells),
        )
        for name, value in uint32_population_fields:
            if type(value) is not int or not 1 <= value <= _UINT32_MASK:
                raise ValueError(f"{name} must be a positive uint32-compatible int")
        integer_positive = (
            self.active_columns,
            self.base_cells_per_column,
            self.contacts_per_segment,
            self.theta_match,
            self.theta_active,
            self.novel_race_cap,
        )
        if any(type(value) is not int or value <= 0 for value in integer_positive):
            raise ValueError("population, segment, and threshold sizes must be positive ints")
        if (
            type(self.max_segments_per_cell) is not int
            or not 1 <= self.max_segments_per_cell <= 255
        ):
            raise ValueError("max_segments_per_cell must be an int in [1, 255]")
        if (
            type(self.construction_seed) is not int
            or not 0 <= self.construction_seed <= _UINT32_MASK
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
        physical_cell_count = (
            2 * self.max_pyramidal_cells + 3 * self.column_count
        )
        if physical_cell_count > _UINT32_MASK:
            raise ValueError("physical cell identity layout must fit uint32")
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
        if (
            type(self.connected_permanence_ticks) is not int
            or not 1 <= self.connected_permanence_ticks <= 500
        ):
            raise ValueError("connected_permanence_ticks must be an int in [1, 500]")
        live_tick_fields = (
            ("initial_permanence_ticks", self.initial_permanence_ticks),
            ("permanence_increment_ticks", self.permanence_increment_ticks),
            ("negative_decrement_ticks", self.negative_decrement_ticks),
        )
        for name, value in live_tick_fields:
            if type(value) is not int or not 1 <= value <= 500:
                raise ValueError(f"{name} must be an int in [1, 500]")
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
    segments: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]
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
    __slots__ = ("owner", "sources", "permanence_ticks")

    def __init__(
        self,
        owner: int,
        sources: Sequence[int],
        permanence_ticks: int,
    ) -> None:
        self.owner = int(owner)
        self.sources = [int(value) for value in sources]
        self.permanence_ticks = [int(permanence_ticks)] * len(self.sources)


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
            token = _require_uint32(raw_token, label="token ID")
            columns = tuple(
                sorted(
                    _require_uint32(value, label="column ID")
                    for value in raw_columns
                )
            )
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
        flattened = [
            _require_uint32(cell, label="snapshot column cell ID")
            for cells in learned.column_cells
            for cell in cells
        ]
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
            recruited_cells = tuple(cells[config.base_cells_per_column :])
            if len(recruited_cells) % config.novel_race_cap:
                raise ValueError(
                    "snapshot column contains a partial reserve cohort"
                )
            for offset in range(0, len(recruited_cells), config.novel_race_cap):
                cohort = recruited_cells[offset : offset + config.novel_race_cap]
                expected_cohort = tuple(
                    range(cohort[0], cohort[0] + config.novel_race_cap)
                )
                if cohort != expected_cohort:
                    raise ValueError(
                        "snapshot column contains a noncontiguous reserve cohort"
                    )
        next_reserve = _require_uint32(
            learned.next_reserve_cell,
            label="snapshot reserve cursor",
        )
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
        for owner, sources, permanence_ticks in learned.segments:
            normalized_owner = _require_uint32(
                owner,
                label="snapshot segment owner",
            )
            if normalized_owner not in committed:
                raise ValueError("snapshot segment owner is not a committed column cell")
            if len(sources) != len(permanence_ticks):
                raise ValueError("snapshot source/permanence-tick lengths differ")
            if len(sources) > config.contacts_per_segment // 2:
                raise ValueError("snapshot segment sources violate paired capacity")
            normalized_sources = tuple(
                _require_uint32(value, label="snapshot segment source")
                for value in sources
            )
            for source in normalized_sources:
                try:
                    circuit._relay_for_context_source(source)
                except RuntimeError:
                    raise ValueError(
                        "snapshot segment contains a non-normalized source"
                    ) from None
            if any(
                normalized_sources[index - 1] >= normalized_sources[index]
                for index in range(
                    1,
                    len(normalized_sources),
                )
            ):
                raise ValueError(
                    "snapshot segment sources must be strictly increasing"
                )
            if any(
                type(value) is not int or not 1 <= value <= 500
                for value in permanence_ticks
            ):
                raise ValueError(
                    "snapshot live permanence tick is outside integer range [1, 500]"
                )
            segment = _Segment(
                normalized_owner,
                normalized_sources,
                config.initial_permanence_ticks,
            )
            segment.permanence_ticks = [int(value) for value in permanence_ticks]
            segment_id = len(circuit._segments)
            circuit._segments.append(segment)
            circuit._segments_by_cell.setdefault(normalized_owner, []).append(segment_id)
        circuit._rebuild_contact_index()
        if (
            type(learned.learning_step) is not int
            or not 0 <= learned.learning_step <= _UINT64_MASK
        ):
            raise ValueError("snapshot learning_step must be a uint64-compatible int")
        circuit._learning_step = learned.learning_step
        circuit._next_reserve_cell = next_reserve
        circuit._previous_context = tuple(
            _require_uint32(value, label="snapshot previous_context cell ID")
            for value in snapshot.previous_context
        )
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
            try:
                circuit._context_source_pairs(circuit._previous_context)
            except RuntimeError:
                raise ValueError(
                    "snapshot previous_context violates source/relay pairing"
                ) from None
        circuit._previous_winner_context = tuple(
            _require_uint32(
                value,
                label="snapshot previous_winner_context cell ID",
            )
            for value in snapshot.previous_winner_context
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
            try:
                circuit._context_source_pairs(circuit._previous_winner_context)
            except RuntimeError:
                raise ValueError(
                    "snapshot previous_winner_context violates source/relay pairing"
                ) from None
        if (
            type(snapshot.event_step) is not int
            or not 0 <= snapshot.event_step <= _UINT64_MASK
        ):
            raise ValueError("snapshot event_step must be a uint64-compatible int")
        circuit._event_step = snapshot.event_step
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
                    tuple(segment.sources),
                    tuple(segment.permanence_ticks),
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

    def permanence_tick_values(self) -> tuple[int, ...]:
        return tuple(
            value
            for segment in self._segments
            for value in segment.permanence_ticks
        )

    def diagnose_context(self, context_spikes: Sequence[int]) -> ContextDiagnostics:
        """Inspect segment coincidence without racing or changing any state."""
        context = tuple(
            _require_uint32(value, label="context_spikes cell ID")
            for value in context_spikes
        )
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
        token = _require_uint32(token_id, label="token_id")
        return self._step(token, mutate=True)

    def recognize(self, token_id: int) -> TemporalStep:
        token = _require_uint32(token_id, label="token_id")
        return self._step(token, mutate=False)

    # ------------------------------------------------------------------ step

    def _step(self, token_id: int, *, mutate: bool) -> TemporalStep:
        if self._event_step >= _UINT64_MASK:
            raise RuntimeError("STOP: event_step uint64 overflow before state change")
        if (
            mutate
            and self.config.growth_enabled
            and self._learning_step >= _UINT64_MASK
        ):
            raise RuntimeError(
                "STOP: learning_step uint64 overflow before atomic plasticity"
            )
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

        reserve_plan: dict[int, tuple[int, ...]] = {}
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
                staged.extend(reserved)
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
                    segment.permanence_ticks[contact_index]
                    >= self.config.connected_permanence_ticks
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
        if len(categorized) > cfg.novel_race_cap:
            raise RuntimeError(
                "STOP: categorized owners exceed the calibrated race cap"
            )
        novel_limit = cfg.novel_race_cap - len(categorized)
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
            )[:novel_limit]
        else:
            # Book/pass boundaries have no distal source.  Keep the pool-cell
            # code independent of learned load and exclude recruited reserve
            # cells so boundary output remains stable over development.
            ranked_novel = list(cells[: cfg.base_cells_per_column])

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
    ) -> dict[int, tuple[int, ...]]:
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
        cohort_size = self.config.novel_race_cap
        required = len(exhausted) * cohort_size
        available = self.config.max_pyramidal_cells - self._next_reserve_cell
        if required > available:
            raise RuntimeError(
                "STOP: column capacity exhausted; full reserve cohort does not fit "
                "before atomic plasticity"
            )
        plan: dict[int, tuple[int, ...]] = {}
        next_cell = self._next_reserve_cell
        for column in exhausted:
            plan[column] = tuple(range(next_cell, next_cell + cohort_size))
            next_cell += cohort_size
        return plan

    def _commit_planned_reserve_cells(
        self,
        plan: Mapping[int, Sequence[int]],
    ) -> None:
        for column, cohort in plan.items():
            if len(cohort) != self.config.novel_race_cap:
                raise RuntimeError("STOP: reserve plan lost full-cohort width")
            for cell in cohort:
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
        sources, _relays = self._context_source_pairs(context)
        pair_count = self.config.contacts_per_segment // 2
        if len(set(sources)) < pair_count:
            raise RuntimeError(
                "STOP: physical context is smaller than contacts_per_segment"
            )
        # The sampled column positions are arm-independent.  A1 and A3 differ
        # only in whether those positions resolve to winner or generic source
        # cells; owner identity must not perturb contact mass or positions.
        chosen_pairs = _sample_contact_positions(
            len(sources),
            pair_count,
            key=(
                self.config.construction_seed ^ _INITIAL_CONTACT_DOMAIN
            )
            & _UINT64_MASK,
            learning_step=self._learning_step,
        )
        chosen_sources = sorted(sources[int(index)] for index in chosen_pairs)
        segment_id = len(self._segments)
        segment = _Segment(
            owner,
            chosen_sources,
            self.config.initial_permanence_ticks,
        )
        self._segments.append(segment)
        owned.append(segment_id)
        self._index_segment(segment_id)

    def _reinforce(self, segment_id: int, context: Sequence[int]) -> None:
        active_sources = set(self._context_source_pairs(context)[0])
        segment = self._segments[segment_id]
        retained: list[tuple[int, int]] = []
        for source, ticks in zip(
            segment.sources,
            segment.permanence_ticks,
            strict=True,
        ):
            delta = (
                self.config.permanence_increment_ticks
                if source in active_sources
                else -self.config.permanence_increment_ticks
            )
            updated = min(500, max(0, ticks + delta))
            if updated > 0:
                retained.append((source, updated))
        sources, permanence_ticks = self._regrow_pairs(
            segment_id,
            retained,
            context,
        )
        segment.sources = sources
        segment.permanence_ticks = permanence_ticks
        self._rebuild_contact_index()

    def _punish(self, segment_id: int, context: Sequence[int]) -> None:
        active_sources = set(self._context_source_pairs(context)[0])
        segment = self._segments[segment_id]
        retained: list[tuple[int, int]] = []
        changed = False
        for source, ticks in zip(
            segment.sources,
            segment.permanence_ticks,
            strict=True,
        ):
            updated = ticks
            if source in active_sources and ticks >= self.config.connected_permanence_ticks:
                updated = max(0, ticks - self.config.negative_decrement_ticks)
            if updated > 0:
                retained.append((source, updated))
            changed |= updated != ticks
        if changed:
            segment.sources = [source for source, _ticks in retained]
            segment.permanence_ticks = [ticks for _source, ticks in retained]
            self._rebuild_contact_index()

    def _regrow_pairs(
        self,
        segment_id: int,
        retained: Sequence[tuple[int, int]],
        context: Sequence[int],
    ) -> tuple[list[int], list[int]]:
        segment = self._segments[segment_id]
        retained_by_source = dict(retained)
        target_pairs = self.config.contacts_per_segment // 2
        missing_pairs = target_pairs - len(retained_by_source)
        if missing_pairs <= 0:
            ordered = sorted(retained_by_source.items())
            return (
                [source for source, _ticks in ordered],
                [ticks for _source, ticks in ordered],
            )
        context_sources, _context_relays = self._context_source_pairs(context)
        candidates = sorted(set(context_sources) - retained_by_source.keys())
        count = min(max(0, missing_pairs), len(candidates))
        regrowth_key, _ = _philox2x64_10(
            int(segment.owner),
            int(segment_id),
            (
                self.config.construction_seed ^ _REGROWTH_CONTACT_DOMAIN
            )
            & _UINT64_MASK,
        )
        selected_positions = _sample_contact_positions(
            len(candidates),
            count,
            key=regrowth_key,
            learning_step=self._learning_step,
        )
        pairs = list(retained_by_source.items())
        for position in selected_positions:
            source = candidates[position]
            pairs.append((source, self.config.initial_permanence_ticks))
        pairs.sort(key=lambda item: item[0])
        return (
            [source for source, _ticks in pairs],
            [ticks for _source, ticks in pairs],
        )

    def _index_segment(self, segment_id: int) -> None:
        segment = self._segments[segment_id]
        for pair_index, source in enumerate(segment.sources):
            self._contact_index.setdefault(source, []).append(
                (segment_id, pair_index)
            )
            relay = self._relay_for_context_source(source)
            self._contact_index.setdefault(relay, []).append(
                (segment_id, pair_index)
            )

    def _rebuild_contact_index(self) -> None:
        self._contact_index = {}
        for segment_id in range(len(self._segments)):
            self._index_segment(segment_id)


__all__ = [
    "CircuitSnapshot",
    "ContextDiagnostics",
    "GrowthPolicy",
    "LearnedSnapshot",
    "SparseTemporalConfig",
    "SparseTemporalCortex",
    "TemporalStep",
    "contact_position_sampling_contract",
]
