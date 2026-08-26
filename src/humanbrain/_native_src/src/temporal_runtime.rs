//! Deep read-only temporal circuit runtime.
//!
//! This is the first physical single-step beneath the future Python-facing
//! temporal circuit. It owns token decoding, materialized column cohorts,
//! learned segment recognition, fitted-cell races, and transient physical
//! context. Its only inference input is a token ID. In particular, no caller
//! can inject winners, segment classes, currents, or a prepared mutation.
//!
//! Claim boundary: this module performs deterministic `READ_ONLY` recognition
//! only. It has no learning transaction, reserve growth, checkpoint format,
//! public ABI, Python wrapper, or outcome-evaluation path.

use crate::temporal::{TemporalOwnerDiagnostic, TemporalStore, TemporalStoreConfig};
use crate::temporal_physics::{
    coupled_first_spike_race, singleton_latency_ticks, BasketDrive, CandidateClass, CandidateDrive,
    CoupledRaceScratch, SpikeEvent, TemporalPhysicsStop,
};

const ACTIVE_COLUMNS: usize = 11;
const CONTEXT_CELLS: usize = 2 * ACTIVE_COLUMNS;
const CONTEXT_CELLS_U8: u8 = CONTEXT_CELLS as u8;
const PAIRS_PER_SEGMENT: usize = 10;
const NONE: u32 = u32::MAX;
const BASKET_CURRENT_PA: f64 = 200_000.0;
const BASKET_INHIBITION_PA: f64 = 500.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalContextPolicy {
    BoundWinners,
    GenericAfferents,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TemporalCircuitConfig {
    pub(crate) column_count: u32,
    pub(crate) base_cells_per_column: u8,
    pub(crate) max_pyramidal_cells: u32,
    pub(crate) segment_capacity: usize,
    pub(crate) novel_race_cap: u8,
    pub(crate) prediction_quorum: u8,
    pub(crate) construction_seed: u32,
    pub(crate) connected_permanence_ticks: u16,
    pub(crate) theta_match: u8,
    pub(crate) theta_active: u8,
    pub(crate) context_policy: TemporalContextPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TokenCode {
    pub(crate) token_id: u32,
    pub(crate) columns: [u32; ACTIVE_COLUMNS],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TemporalSegmentInit {
    pub(crate) owner: u32,
    pub(crate) sources: Vec<u32>,
    pub(crate) permanence_ticks: Vec<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TemporalRuntimeBuildError {
    Invalid(&'static str),
    Store(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PhysicalStage {
    Proximal,
    ProximalRelay,
    ColumnRace,
    WinnerRelay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalOverflowStage {
    ProximalRelayTick,
    WinnerRelayTick,
    EventCounter,
}

/// Stable terminal class persisted by the future canonical checkpoint.
///
/// The first failing call still returns the full [`TemporalCircuitStop`]. The
/// runtime retains only this `Copy` class so poisoning never clones or owns a
/// diagnostic allocation and all later calls return `Poisoned`.
#[repr(u16)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalPoisonCode {
    Unpoisoned = 0,
    MultipleActiveOwners = 1,
    MultipleMatchingOwners = 2,
    CategorizedRaceCapacity = 3,
    MalformedContextOrPolicy = 4,
    ProximalPhysics = 5,
    ProximalRelayPhysics = 6,
    ColumnRacePhysics = 7,
    WinnerRelayPhysics = 8,
    TickOrCounterOverflow = 9,
    StoreOrPlanInvariant = 10,
    PhiloxAddressExhaustion = 11,
    ReserveCellCapacity = 12,
    GlobalSegmentCapacity = 13,
    OwnerSegmentCapacity = 14,
    PreallocatedCapacityInvariant = 15,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OwnerConflict {
    pub(crate) first: u32,
    pub(crate) second: u32,
    pub(crate) count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalCircuitStop {
    UnknownToken(u32),
    Poisoned,
    MalformedContextOrPolicy(&'static str),
    MultipleActiveOwners {
        column: u32,
        owners: OwnerConflict,
    },
    MultipleMatchingOwners {
        column: u32,
        owners: OwnerConflict,
    },
    CategorizedRaceCapacity {
        column: u32,
        categorized_count: u8,
        race_cap: u8,
    },
    TickOrCounterOverflow {
        stage: TemporalOverflowStage,
        location: u32,
    },
    Physics {
        stage: PhysicalStage,
        location: u32,
        source: TemporalPhysicsStop,
    },
    StoreRecognition(u32),
    StoreOrPlanInvariant(&'static str),
    PreallocatedCapacityInvariant(&'static str),
}

impl TemporalCircuitStop {
    fn poison_code(&self) -> Option<TemporalPoisonCode> {
        match self {
            Self::UnknownToken(_) | Self::Poisoned => None,
            Self::MalformedContextOrPolicy(_) => Some(TemporalPoisonCode::MalformedContextOrPolicy),
            Self::MultipleActiveOwners { .. } => Some(TemporalPoisonCode::MultipleActiveOwners),
            Self::MultipleMatchingOwners { .. } => Some(TemporalPoisonCode::MultipleMatchingOwners),
            Self::CategorizedRaceCapacity { .. } => {
                Some(TemporalPoisonCode::CategorizedRaceCapacity)
            }
            Self::TickOrCounterOverflow { .. } => Some(TemporalPoisonCode::TickOrCounterOverflow),
            Self::Physics { stage, .. } => Some(match stage {
                PhysicalStage::Proximal => TemporalPoisonCode::ProximalPhysics,
                PhysicalStage::ProximalRelay => TemporalPoisonCode::ProximalRelayPhysics,
                PhysicalStage::ColumnRace => TemporalPoisonCode::ColumnRacePhysics,
                PhysicalStage::WinnerRelay => TemporalPoisonCode::WinnerRelayPhysics,
            }),
            Self::StoreRecognition(_) | Self::StoreOrPlanInvariant(_) => {
                Some(TemporalPoisonCode::StoreOrPlanInvariant)
            }
            Self::PreallocatedCapacityInvariant(_) => {
                Some(TemporalPoisonCode::PreallocatedCapacityInvariant)
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TemporalCircuitStep {
    pub(crate) token_id: u32,
    pub(crate) predicted: bool,
    pub(crate) predicted_columns: u8,
    pub(crate) winner_spikes: [SpikeEvent; ACTIVE_COLUMNS],
    pub(crate) basket_spikes: [SpikeEvent; ACTIVE_COLUMNS],
    pub(crate) inhibition_onset_ticks: [u16; ACTIVE_COLUMNS],
    pub(crate) proximal_spikes: [SpikeEvent; ACTIVE_COLUMNS],
    pub(crate) proximal_relay_spikes: [SpikeEvent; ACTIVE_COLUMNS],
    pub(crate) winner_relay_spikes: [SpikeEvent; ACTIVE_COLUMNS],
    pub(crate) column_simulated_ticks: [u16; ACTIVE_COLUMNS],
    pub(crate) winner_context: [u32; CONTEXT_CELLS],
    pub(crate) proximal_context: [u32; CONTEXT_CELLS],
    pub(crate) context: [u32; CONTEXT_CELLS],
    pub(crate) event_count: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NovelCandidate {
    cell_id: u32,
    column_offset: usize,
    load: u8,
    afferent_score: u128,
}

const EMPTY_DIAGNOSTIC: TemporalOwnerDiagnostic = TemporalOwnerDiagnostic {
    owner: 0,
    best_matching_segment: NONE,
    best_active_segment: NONE,
    matching_potential_contacts: 0,
    matching_connected_contacts: 0,
    active_potential_contacts: 0,
    active_connected_contacts: 0,
};
const EMPTY_SPIKE: SpikeEvent = SpikeEvent {
    cell_id: 0,
    tick: 0,
};
const EMPTY_DRIVE: CandidateDrive = CandidateDrive {
    cell_id: 0,
    current_pa: 0.0,
    threshold_offset_mv: 0.0,
};
const EMPTY_NOVEL_CANDIDATE: NovelCandidate = NovelCandidate {
    cell_id: 0,
    column_offset: 0,
    load: 0,
    afferent_score: 0,
};

pub(crate) struct TemporalCircuitRuntime {
    config: TemporalCircuitConfig,
    token_codebook: Vec<TokenCode>,
    column_offsets: Vec<usize>,
    column_cells: Vec<u32>,
    threshold_offsets_mv: Vec<f64>,
    store: TemporalStore,
    recognition_scratch: Vec<TemporalOwnerDiagnostic>,
    column_drive_scratch: Vec<CandidateDrive>,
    novel_candidate_scratch: Vec<NovelCandidate>,
    race_scratch: CoupledRaceScratch,
    previous_context: [u32; CONTEXT_CELLS],
    previous_context_len: u8,
    previous_winner_context: [u32; CONTEXT_CELLS],
    previous_winner_context_len: u8,
    event_count: u64,
    scientific_stop: Option<TemporalPoisonCode>,
    relay_offset: u32,
    basket_offset: u32,
    proximal_offset: u32,
    proximal_relay_offset: u32,
}

impl TemporalCircuitRuntime {
    pub(crate) fn new(
        config: TemporalCircuitConfig,
        mut token_codebook: Vec<TokenCode>,
        column_members: Vec<Vec<u32>>,
        threshold_offsets_mv: Vec<f64>,
        segments: &[TemporalSegmentInit],
    ) -> Result<Self, TemporalRuntimeBuildError> {
        let column_count = usize::try_from(config.column_count)
            .map_err(|_| TemporalRuntimeBuildError::Invalid("column_count exceeds usize"))?;
        let owner_count = usize::try_from(config.max_pyramidal_cells)
            .map_err(|_| TemporalRuntimeBuildError::Invalid("max_pyramidal_cells exceeds usize"))?;
        let base_width = usize::from(config.base_cells_per_column);
        let race_cap = usize::from(config.novel_race_cap);
        if column_count < ACTIVE_COLUMNS
            || base_width == 0
            || race_cap == 0
            || race_cap > 8
            || base_width > race_cap
            || config.segment_capacity == 0
            || config.segment_capacity < segments.len()
            || (config.context_policy == TemporalContextPolicy::GenericAfferents && base_width < 2)
            || !(1..=ACTIVE_COLUMNS as u8).contains(&config.prediction_quorum)
            || !(1..=500).contains(&config.connected_permanence_ticks)
            || !(1 <= config.theta_match
                && config.theta_match <= config.theta_active
                && config.theta_active <= 2 * PAIRS_PER_SEGMENT as u8)
        {
            return Err(TemporalRuntimeBuildError::Invalid(
                "invalid temporal circuit dimensions or thresholds",
            ));
        }
        let base_total =
            column_count
                .checked_mul(base_width)
                .ok_or(TemporalRuntimeBuildError::Invalid(
                    "base population overflows usize",
                ))?;
        if base_total > owner_count || column_members.len() != column_count {
            return Err(TemporalRuntimeBuildError::Invalid(
                "column membership does not match the configured sheet",
            ));
        }

        let relay_offset = config.max_pyramidal_cells;
        let basket_offset =
            relay_offset
                .checked_mul(2)
                .ok_or(TemporalRuntimeBuildError::Invalid(
                    "physical offsets overflow u32",
                ))?;
        let proximal_offset = basket_offset.checked_add(config.column_count).ok_or(
            TemporalRuntimeBuildError::Invalid("physical offsets overflow u32"),
        )?;
        let proximal_relay_offset = proximal_offset.checked_add(config.column_count).ok_or(
            TemporalRuntimeBuildError::Invalid("physical offsets overflow u32"),
        )?;
        let physical_cell_count = proximal_relay_offset
            .checked_add(config.column_count)
            .ok_or(TemporalRuntimeBuildError::Invalid(
                "physical offsets overflow u32",
            ))?;
        if threshold_offsets_mv.len() != physical_cell_count as usize
            || threshold_offsets_mv
                .iter()
                .any(|offset| !offset.is_finite())
        {
            return Err(TemporalRuntimeBuildError::Invalid(
                "threshold offsets must cover every physical cell with finite f64 values",
            ));
        }

        let mut column_offsets = Vec::with_capacity(column_count + 1);
        let mut column_cells = Vec::new();
        let mut committed = vec![false; owner_count];
        let mut recruited = Vec::new();
        column_offsets.push(0);
        for (column, cells) in column_members.iter().enumerate() {
            if cells.len() < base_width
                || (cells.len() - base_width) % race_cap != 0
                || cells.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "columns must contain sorted base cells followed by complete reserve cohorts",
                ));
            }
            let base_start =
                column
                    .checked_mul(base_width)
                    .ok_or(TemporalRuntimeBuildError::Invalid(
                        "base cell identity overflows usize",
                    ))?;
            for (offset, &cell) in cells.iter().enumerate() {
                let cell_index = cell as usize;
                if cell_index >= owner_count || committed[cell_index] {
                    return Err(TemporalRuntimeBuildError::Invalid(
                        "column cells must be unique in-range pyramidal identities",
                    ));
                }
                if offset < base_width {
                    if cell_index != base_start + offset {
                        return Err(TemporalRuntimeBuildError::Invalid(
                            "a column changed its deterministic base-cell prefix",
                        ));
                    }
                } else {
                    if cell_index < base_total {
                        return Err(TemporalRuntimeBuildError::Invalid(
                            "reserve cells overlap the deterministic base population",
                        ));
                    }
                    recruited.push(cell);
                }
                committed[cell_index] = true;
            }
            for cohort in cells[base_width..].chunks_exact(race_cap) {
                if cohort
                    .windows(2)
                    .any(|pair| pair[1] != pair[0].saturating_add(1))
                {
                    return Err(TemporalRuntimeBuildError::Invalid(
                        "reserve cohorts must be contiguous physical identities",
                    ));
                }
            }
            column_cells.extend_from_slice(cells);
            column_offsets.push(column_cells.len());
        }
        recruited.sort_unstable();
        if recruited
            .iter()
            .enumerate()
            .any(|(offset, &cell)| cell as usize != base_total + offset)
        {
            return Err(TemporalRuntimeBuildError::Invalid(
                "committed reserve cells must form one contiguous global prefix",
            ));
        }

        if token_codebook.is_empty() {
            return Err(TemporalRuntimeBuildError::Invalid(
                "token codebook must not be empty",
            ));
        }
        for code in &token_codebook {
            if code
                .columns
                .iter()
                .any(|&column| column >= config.column_count)
                || code.columns.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "each token must map to eleven sorted unique in-range columns",
                ));
            }
        }
        token_codebook.sort_unstable_by_key(|code| code.token_id);
        if token_codebook
            .windows(2)
            .any(|pair| pair[0].token_id == pair[1].token_id)
        {
            return Err(TemporalRuntimeBuildError::Invalid(
                "token IDs must be unique",
            ));
        }
        for left in 0..token_codebook.len() {
            if token_codebook[left + 1..]
                .iter()
                .any(|right| right.columns == token_codebook[left].columns)
            {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "two tokens must not share one unordered column code",
                ));
            }
        }

        let (source_gid_base, source_count) = match config.context_policy {
            TemporalContextPolicy::BoundWinners => (0, owner_count),
            TemporalContextPolicy::GenericAfferents => (proximal_offset, column_count),
        };
        let store_config = TemporalStoreConfig {
            source_count,
            owner_count,
            segment_capacity: config.segment_capacity,
            source_gid_base,
            connected_permanence_ticks: config.connected_permanence_ticks,
            theta_match: config.theta_match,
            theta_active: config.theta_active,
            padding_zero: 0,
        };
        let mut store = TemporalStore::new_runtime_owned(store_config)
            .map_err(TemporalRuntimeBuildError::Store)?;
        for (expected_id, segment) in segments.iter().enumerate() {
            if segment.owner as usize >= owner_count || !committed[segment.owner as usize] {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "segment owner is not a committed column cell",
                ));
            }
            let segment_id = store
                .allocate_runtime_segment(
                    segment.owner,
                    &segment.sources,
                    &segment.permanence_ticks,
                )
                .map_err(TemporalRuntimeBuildError::Store)?;
            if segment_id as usize != expected_id {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "temporal store changed segment allocation order",
                ));
            }
        }
        let max_column_occupancy = column_offsets
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .max()
            .ok_or(TemporalRuntimeBuildError::Invalid(
                "temporal circuit must contain at least one column",
            ))?;
        let race_scratch = CoupledRaceScratch::try_new(max_column_occupancy)
            .map_err(TemporalRuntimeBuildError::Invalid)?;
        let column_drive_scratch = vec![EMPTY_DRIVE; max_column_occupancy];
        let novel_candidate_scratch = vec![EMPTY_NOVEL_CANDIDATE; max_column_occupancy];

        Ok(Self {
            config,
            token_codebook,
            column_offsets,
            column_cells,
            threshold_offsets_mv,
            store,
            recognition_scratch: vec![EMPTY_DIAGNOSTIC; owner_count],
            column_drive_scratch,
            novel_candidate_scratch,
            race_scratch,
            previous_context: [0; CONTEXT_CELLS],
            previous_context_len: 0,
            previous_winner_context: [0; CONTEXT_CELLS],
            previous_winner_context_len: 0,
            event_count: 0,
            scientific_stop: None,
            relay_offset,
            basket_offset,
            proximal_offset,
            proximal_relay_offset,
        })
    }

    pub(crate) fn recognize(
        &mut self,
        token_id: u32,
    ) -> Result<TemporalCircuitStep, TemporalCircuitStop> {
        if self.scientific_stop.is_some() {
            return Err(TemporalCircuitStop::Poisoned);
        }
        let columns = match self
            .token_codebook
            .binary_search_by_key(&token_id, |code| code.token_id)
        {
            Ok(index) => self.token_codebook[index].columns,
            Err(_) => return Err(TemporalCircuitStop::UnknownToken(token_id)),
        };

        let recognition_source_count =
            match (self.previous_context_len, self.previous_winner_context_len) {
                (0, 0) => 0,
                (CONTEXT_CELLS_U8, CONTEXT_CELLS_U8) => ACTIVE_COLUMNS,
                _ => {
                    return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                        "prior contexts must both be empty or exactly 22 cells",
                    ));
                }
            };
        let recognition_sources = &self.previous_context[..recognition_source_count];
        let summary = match self
            .store
            .recognize_runtime_context(recognition_sources, &mut self.recognition_scratch)
        {
            Ok(summary) => summary,
            Err(status) => return self.fail(TemporalCircuitStop::StoreRecognition(status)),
        };
        let diagnostic_count = summary.owner_result_count;

        let mut proximal_spikes = [EMPTY_SPIKE; ACTIVE_COLUMNS];
        let mut proximal_relay_spikes = [EMPTY_SPIKE; ACTIVE_COLUMNS];
        for (index, &column) in columns.iter().enumerate() {
            let proximal_id = self.proximal_offset + column;
            let proximal_tick = match singleton_latency_ticks(
                CandidateClass::Proximal.current_pa(),
                self.threshold_offset(proximal_id),
            ) {
                Ok(tick) => tick,
                Err(source) => {
                    return self.fail(TemporalCircuitStop::Physics {
                        stage: PhysicalStage::Proximal,
                        location: proximal_id,
                        source,
                    });
                }
            };
            proximal_spikes[index] = SpikeEvent {
                cell_id: proximal_id,
                tick: proximal_tick,
            };
            let relay_id = self.proximal_relay_offset + column;
            let relay_latency = match singleton_latency_ticks(
                CandidateClass::Relay.current_pa(),
                self.threshold_offset(relay_id),
            ) {
                Ok(tick) => tick,
                Err(source) => {
                    return self.fail(TemporalCircuitStop::Physics {
                        stage: PhysicalStage::ProximalRelay,
                        location: relay_id,
                        source,
                    });
                }
            };
            let Some(relay_tick) = proximal_tick.checked_add(relay_latency) else {
                return self.fail(TemporalCircuitStop::TickOrCounterOverflow {
                    stage: TemporalOverflowStage::ProximalRelayTick,
                    location: relay_id,
                });
            };
            proximal_relay_spikes[index] = SpikeEvent {
                cell_id: relay_id,
                tick: relay_tick,
            };
        }

        let mut winner_spikes = [EMPTY_SPIKE; ACTIVE_COLUMNS];
        let mut basket_spikes = [EMPTY_SPIKE; ACTIVE_COLUMNS];
        let mut inhibition_onset_ticks = [0u16; ACTIVE_COLUMNS];
        let mut column_simulated_ticks = [0u16; ACTIVE_COLUMNS];
        let mut predicted_columns = 0u8;
        for (index, &column) in columns.iter().enumerate() {
            let candidate_count = match self.prepare_column_drives(column, diagnostic_count) {
                Ok(candidate_count) => candidate_count,
                Err(stop) => return self.fail(stop),
            };
            let basket_id = self.basket_offset + column;
            let candidates = &self.column_drive_scratch[..candidate_count];
            let race = match coupled_first_spike_race(
                candidates,
                BasketDrive {
                    cell_id: basket_id,
                    current_pa: BASKET_CURRENT_PA,
                    threshold_offset_mv: self.threshold_offset(basket_id),
                    inhibition_pa: BASKET_INHIBITION_PA,
                },
                &mut self.race_scratch,
            ) {
                Ok(race) => race,
                Err(source) => {
                    return self.fail(TemporalCircuitStop::Physics {
                        stage: PhysicalStage::ColumnRace,
                        location: column,
                        source,
                    });
                }
            };
            winner_spikes[index] = race.winner;
            basket_spikes[index] = SpikeEvent {
                cell_id: race.basket_cell_id,
                tick: race.basket_tick,
            };
            inhibition_onset_ticks[index] = race.inhibition_onset_tick;
            column_simulated_ticks[index] = race.simulated_ticks;
            if self
                .diagnostic_for_owner(race.winner.cell_id, diagnostic_count)
                .is_some_and(|diagnostic| diagnostic.best_active_segment != NONE)
            {
                predicted_columns += 1;
            }
        }

        let mut winner_relay_spikes = [EMPTY_SPIKE; ACTIVE_COLUMNS];
        for (index, winner) in winner_spikes.iter().enumerate() {
            let relay_id = self.relay_offset + winner.cell_id;
            let relay_latency = match singleton_latency_ticks(
                CandidateClass::Relay.current_pa(),
                self.threshold_offset(relay_id),
            ) {
                Ok(tick) => tick,
                Err(source) => {
                    return self.fail(TemporalCircuitStop::Physics {
                        stage: PhysicalStage::WinnerRelay,
                        location: relay_id,
                        source,
                    });
                }
            };
            let Some(relay_tick) = winner.tick.checked_add(relay_latency) else {
                return self.fail(TemporalCircuitStop::TickOrCounterOverflow {
                    stage: TemporalOverflowStage::WinnerRelayTick,
                    location: relay_id,
                });
            };
            winner_relay_spikes[index] = SpikeEvent {
                cell_id: relay_id,
                tick: relay_tick,
            };
        }

        let winner_context = std::array::from_fn(|index| {
            if index < ACTIVE_COLUMNS {
                winner_spikes[index].cell_id
            } else {
                winner_relay_spikes[index - ACTIVE_COLUMNS].cell_id
            }
        });
        let proximal_context = std::array::from_fn(|index| {
            if index < ACTIVE_COLUMNS {
                proximal_spikes[index].cell_id
            } else {
                proximal_relay_spikes[index - ACTIVE_COLUMNS].cell_id
            }
        });
        let context = match self.config.context_policy {
            TemporalContextPolicy::BoundWinners => winner_context,
            TemporalContextPolicy::GenericAfferents => proximal_context,
        };
        let Some(event_count) = self.event_count.checked_add(1) else {
            return self.fail(TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::EventCounter,
                location: 0,
            });
        };

        // The only transient commit point. All physical work and every STOP
        // condition above used staged values.
        self.previous_context = context;
        self.previous_context_len = CONTEXT_CELLS as u8;
        self.previous_winner_context = winner_context;
        self.previous_winner_context_len = CONTEXT_CELLS as u8;
        self.event_count = event_count;

        Ok(TemporalCircuitStep {
            token_id,
            predicted: predicted_columns >= self.config.prediction_quorum,
            predicted_columns,
            winner_spikes,
            basket_spikes,
            inhibition_onset_ticks,
            proximal_spikes,
            proximal_relay_spikes,
            winner_relay_spikes,
            column_simulated_ticks,
            winner_context,
            proximal_context,
            context,
            event_count,
        })
    }

    pub(crate) fn reset_transient(&mut self) {
        self.previous_context = [0; CONTEXT_CELLS];
        self.previous_context_len = 0;
        self.previous_winner_context = [0; CONTEXT_CELLS];
        self.previous_winner_context_len = 0;
        self.event_count = 0;
    }

    fn fail<T>(&mut self, stop: TemporalCircuitStop) -> Result<T, TemporalCircuitStop> {
        debug_assert!(!matches!(stop, TemporalCircuitStop::UnknownToken(_)));
        if let Some(code) = stop.poison_code() {
            self.scientific_stop = Some(code);
        } else {
            debug_assert!(false, "retryable STOP reached the poison helper");
        }
        Err(stop)
    }

    fn threshold_offset(&self, cell_id: u32) -> f64 {
        self.threshold_offsets_mv[cell_id as usize]
    }

    fn owner_load(&self, owner: u32) -> Result<u8, TemporalCircuitStop> {
        self.store
            .runtime_owner_load(owner)
            .ok_or(TemporalCircuitStop::StoreOrPlanInvariant(
                "column owner is outside the temporal store",
            ))
    }

    fn diagnostic_for_owner(
        &self,
        owner: u32,
        diagnostic_count: usize,
    ) -> Option<&TemporalOwnerDiagnostic> {
        let diagnostics = &self.recognition_scratch[..diagnostic_count];
        diagnostics
            .binary_search_by_key(&owner, |diagnostic| diagnostic.owner)
            .ok()
            .map(|index| &diagnostics[index])
    }

    fn prepare_column_drives(
        &mut self,
        column: u32,
        diagnostic_count: usize,
    ) -> Result<usize, TemporalCircuitStop> {
        if diagnostic_count > self.recognition_scratch.len() {
            return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
                "recognition output exceeds its construction-time scratch",
            ));
        }
        let column_index = column as usize;
        let Some((&cell_start, &cell_end)) = self
            .column_offsets
            .get(column_index)
            .zip(self.column_offsets.get(column_index + 1))
        else {
            return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                "token code references a missing column",
            ));
        };
        if cell_start > cell_end || cell_end > self.column_cells.len() {
            return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                "column CSR bounds are malformed",
            ));
        }
        let cells = &self.column_cells[cell_start..cell_end];
        if cells.len() > self.column_drive_scratch.len()
            || cells.len() > self.novel_candidate_scratch.len()
        {
            return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
                "column occupancy exceeds construction-time candidate scratch",
            ));
        }

        let mut active_first = NONE;
        let mut active_second = NONE;
        let mut active_count = 0u32;
        let mut active_offset = None;
        let mut matching_first = NONE;
        let mut matching_second = NONE;
        let mut matching_count = 0u32;
        let mut matching_offset = None;
        for (offset, &cell) in cells.iter().enumerate() {
            let Some(diagnostic) = self.diagnostic_for_owner(cell, diagnostic_count) else {
                continue;
            };
            if diagnostic.best_active_segment != NONE {
                active_count += 1;
                if active_first == NONE {
                    active_first = cell;
                    active_offset = Some(offset);
                } else if active_second == NONE {
                    active_second = cell;
                }
            }
            if diagnostic.best_matching_segment != NONE {
                matching_count += 1;
                if matching_first == NONE {
                    matching_first = cell;
                    matching_offset = Some(offset);
                } else if matching_second == NONE {
                    matching_second = cell;
                }
            }
        }
        if active_count > 1 {
            return Err(TemporalCircuitStop::MultipleActiveOwners {
                column,
                owners: OwnerConflict {
                    first: active_first,
                    second: active_second,
                    count: active_count,
                },
            });
        }
        if matching_count > 1 {
            return Err(TemporalCircuitStop::MultipleMatchingOwners {
                column,
                owners: OwnerConflict {
                    first: matching_first,
                    second: matching_second,
                    count: matching_count,
                },
            });
        }

        let categorized_count = usize::from(active_count != 0)
            + usize::from(matching_count != 0 && matching_first != active_first);
        if categorized_count > usize::from(self.config.novel_race_cap) {
            return Err(TemporalCircuitStop::CategorizedRaceCapacity {
                column,
                categorized_count: categorized_count as u8,
                race_cap: self.config.novel_race_cap,
            });
        }
        let novel_limit = usize::from(self.config.novel_race_cap) - categorized_count;
        for (offset, &cell_id) in cells.iter().enumerate() {
            self.column_drive_scratch[offset] = CandidateDrive {
                cell_id,
                current_pa: 0.0,
                threshold_offset_mv: self.threshold_offset(cell_id),
            };
        }

        if self.previous_context_len == 0 {
            for rank in 0..usize::from(self.config.base_cells_per_column) {
                let Some(&cell_id) = cells.get(rank) else {
                    return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                        "column lost its validated base-cell prefix",
                    ));
                };
                self.column_drive_scratch[rank].current_pa = CandidateClass::Novel {
                    rank: rank as u8,
                    load: self.owner_load(cell_id)?,
                }
                .current_pa();
            }
        } else {
            let mut novel_candidate_count = 0usize;
            for (column_offset, &cell_id) in cells.iter().enumerate() {
                if cell_id == active_first || cell_id == matching_first {
                    continue;
                }
                self.novel_candidate_scratch[novel_candidate_count] = NovelCandidate {
                    cell_id,
                    column_offset,
                    load: self.owner_load(cell_id)?,
                    afferent_score: afferent_score(
                        self.config.construction_seed,
                        &self.previous_context[..usize::from(self.previous_context_len)],
                        cell_id,
                    ),
                };
                novel_candidate_count += 1;
            }
            let mut selected = [EMPTY_NOVEL_CANDIDATE; 8];
            let selected_count = select_novel_candidates(
                &self.novel_candidate_scratch[..novel_candidate_count],
                novel_limit,
                &mut selected,
            );
            for (rank, candidate) in selected[..selected_count].iter().enumerate() {
                self.column_drive_scratch[candidate.column_offset].current_pa =
                    CandidateClass::Novel {
                        rank: rank as u8,
                        load: candidate.load,
                    }
                    .current_pa();
            }
        }

        if let Some(offset) = matching_offset {
            self.column_drive_scratch[offset].current_pa = CandidateClass::Matching {
                load: self.owner_load(matching_first)?,
            }
            .current_pa();
        }
        if let Some(offset) = active_offset {
            self.column_drive_scratch[offset].current_pa = CandidateClass::Active {
                load: self.owner_load(active_first)?,
            }
            .current_pa();
        }

        // Keep every materialized cohort cell in the physical race, including
        // exact zero-drive cells. This preserves the Python oracle's simulated
        // work and STOP surface; a later sparse elision needs its own parity
        // proof before changing this boundary.
        Ok(cells.len())
    }
}

fn select_novel_candidates(
    candidates: &[NovelCandidate],
    limit: usize,
    selected: &mut [NovelCandidate; 8],
) -> usize {
    let Some(minimum_load) = candidates.iter().map(|candidate| candidate.load).min() else {
        return 0;
    };
    let bounded_limit = limit.min(selected.len());
    let mut selected_count = 0usize;
    for &candidate in candidates {
        if candidate.load != minimum_load {
            continue;
        }
        let mut insertion = 0usize;
        while insertion < selected_count {
            let incumbent = selected[insertion];
            let candidate_is_better = candidate.afferent_score > incumbent.afferent_score
                || (candidate.afferent_score == incumbent.afferent_score
                    && candidate.cell_id < incumbent.cell_id);
            if candidate_is_better {
                break;
            }
            insertion += 1;
        }
        if insertion >= bounded_limit {
            continue;
        }
        let next_count = (selected_count + 1).min(bounded_limit);
        for destination in (insertion + 1..next_count).rev() {
            selected[destination] = selected[destination - 1];
        }
        selected[insertion] = candidate;
        selected_count = next_count;
    }
    selected_count
}

const MIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

fn mix_u64(value: u64) -> u64 {
    let mut word = value.wrapping_add(MIX_GAMMA);
    word = (word ^ (word >> 30)).wrapping_mul(MIX_A);
    word = (word ^ (word >> 27)).wrapping_mul(MIX_B);
    word ^ (word >> 31)
}

fn afferent_score(construction_seed: u32, context: &[u32], postsynaptic: u32) -> u128 {
    let post_key = u64::from(postsynaptic).wrapping_add(1).wrapping_mul(MIX_B);
    let seed_key = u64::from(construction_seed)
        .wrapping_add(1)
        .wrapping_mul(MIX_GAMMA);
    context
        .iter()
        .map(|&presynaptic| {
            let pre_key = u64::from(presynaptic).wrapping_add(1).wrapping_mul(MIX_A);
            u128::from(mix_u64(seed_key ^ pre_key ^ post_key))
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    struct ThreadCountingAllocator;

    thread_local! {
        static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    #[global_allocator]
    static TEST_ALLOCATOR: ThreadCountingAllocator = ThreadCountingAllocator;

    unsafe impl GlobalAlloc for ThreadCountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            COUNT_ALLOCATIONS.with(|enabled| {
                if enabled.get() {
                    ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
                }
            });
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            COUNT_ALLOCATIONS.with(|enabled| {
                if enabled.get() {
                    ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
                }
            });
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            COUNT_ALLOCATIONS.with(|enabled| {
                if enabled.get() {
                    ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
                }
            });
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    fn count_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
        ALLOCATION_COUNT.with(|count| count.set(0));
        COUNT_ALLOCATIONS.with(|enabled| {
            assert!(
                !enabled.replace(true),
                "allocation counter is not reentrant"
            );
        });
        let result = operation();
        COUNT_ALLOCATIONS.with(|enabled| enabled.set(false));
        let count = ALLOCATION_COUNT.with(Cell::get);
        (result, count)
    }

    fn config(
        column_count: u32,
        base_cells_per_column: u8,
        context_policy: TemporalContextPolicy,
    ) -> TemporalCircuitConfig {
        TemporalCircuitConfig {
            column_count,
            base_cells_per_column,
            max_pyramidal_cells: column_count * u32::from(base_cells_per_column),
            segment_capacity: 512,
            novel_race_cap: base_cells_per_column.max(2),
            prediction_quorum: ACTIVE_COLUMNS as u8,
            construction_seed: 17,
            connected_permanence_ticks: 250,
            theta_match: 18,
            theta_active: 20,
            context_policy,
        }
    }

    fn two_token_codebook() -> Vec<TokenCode> {
        vec![
            TokenCode {
                token_id: 0,
                columns: std::array::from_fn(|index| index as u32),
            },
            TokenCode {
                token_id: 1,
                columns: std::array::from_fn(|index| (ACTIVE_COLUMNS + index) as u32),
            },
        ]
    }

    fn base_members(config: TemporalCircuitConfig) -> Vec<Vec<u32>> {
        (0..config.column_count)
            .map(|column| {
                let start = column * u32::from(config.base_cells_per_column);
                (start..start + u32::from(config.base_cells_per_column)).collect()
            })
            .collect()
    }

    fn physical_count(config: TemporalCircuitConfig) -> usize {
        (2 * config.max_pyramidal_cells + 3 * config.column_count) as usize
    }

    fn runtime_with_offsets(
        config: TemporalCircuitConfig,
        offsets: Vec<f64>,
        segments: &[TemporalSegmentInit],
    ) -> TemporalCircuitRuntime {
        TemporalCircuitRuntime::new(
            config,
            two_token_codebook(),
            base_members(config),
            offsets,
            segments,
        )
        .expect("fixture must build")
    }

    // Literal NumPy-PCG64 threshold words from DendriticCortex(seed=17) for
    // the 110-cell Python fixture below. These are inputs, not regenerated by
    // Rust, so cross-language tick parity also pins every f64 threshold bit.
    const PYTHON_OFFSET_BITS: [u64; 110] = [
        4583592653688817618,
        4576030930489515032,
        13802307468283465602,
        13808073357696104095,
        13810358950406455269,
        4557231018237700912,
        13804870009762910525,
        13805313804273071794,
        13796419894460558748,
        13786992299829903238,
        13811735832483155504,
        4582323605153356864,
        13810835338782981376,
        4586860822989662070,
        4579664605228535162,
        13801325654783163664,
        4584999476000040768,
        4557932255061433723,
        4580635831668153032,
        4568327642814855928,
        4583558562712351556,
        4583305576240439397,
        13805567289784925088,
        13803352595420360962,
        4576100044897843905,
        13796075189508607077,
        13811765128941968780,
        4587460678776020464,
        13811369039722966092,
        13811032406386954751,
        13808130537424963939,
        4579293986056234002,
        4586631638040284703,
        13796807435924440321,
        13800161677228422111,
        4571662390600183182,
        13799812664725480616,
        4566894454460095691,
        13807243618326560130,
        4577469577013817372,
        4576811287161049486,
        13812307373865560829,
        13791872552517561302,
        4571345750894660294,
        4579241722968867061,
        13804909804700004091,
        13808317605584704501,
        13805880019304334901,
        4579051724443818328,
        13805167379992602776,
        13804535494718935840,
        4583898920975673503,
        13804694120706451934,
        13805504588720750516,
        13790777946831213183,
        4572374580538843942,
        4583134131079048594,
        13805365481970957221,
        13809182730248296744,
        13792579910572702276,
        4580295113445993455,
        13808870619801064284,
        13801547858752051281,
        4575475514871791445,
        4573308844243239380,
        4566951741187861851,
        13797426395230775114,
        13804789156520336924,
        13805717197391643338,
        4583988968876280431,
        13807079602672244480,
        4584505894666175229,
        13808270855235957054,
        4589895048604705197,
        13809795062475888504,
        4585940828345571592,
        13805493344400077202,
        13792183238068169159,
        13788572556622705772,
        13803729017865443367,
        4588189451627651436,
        4567617919711466472,
        13808292527663916497,
        4580316179427563139,
        4580790616993359927,
        13807297744957285461,
        4578872619209066296,
        4565592883975961210,
        4581568362806667387,
        4580789130626646332,
        13775290806547958557,
        13800409634867221178,
        4582217147930963638,
        4575667298000395009,
        13808898545108724069,
        13802378050981119133,
        13801490924358314330,
        13797578148193458807,
        4588390850548341848,
        13799962941724089479,
        4578032768112391430,
        4586124787974691308,
        4578412034151394858,
        4577377633538334349,
        4584837210118496356,
        4572337115587325458,
        4572010599313737577,
        4580620105647348835,
        4577092275574962161,
        13803931750901480275,
    ];

    fn python_oracle_runtime(permanence_ticks: u16) -> TemporalCircuitRuntime {
        // Oracle source hashes at fixture generation:
        // cells.py: 5ca05a64a89f930db3ea351e3e236e3242bf21652fff127747a36d9ca488d2e6
        // cortex.py: 16545b390c516b6f5817366946f0f516cec5b049bb9e1f6021184d9e156e0a8c
        // sparse_temporal_cortex.py:
        // 22e5ab6bb9e3b1bc7010e8c3ffd11d9c2dd9ea01a33ba15046a32db2228357a8
        // The Python circuit learned [token 0, token 1], then its learned
        // segments were imported into a fresh READ_ONLY circuit at a boundary.
        let mut config = config(22, 1, TemporalContextPolicy::BoundWinners);
        config.novel_race_cap = 8;
        let sources = vec![0, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let segments: Vec<TemporalSegmentInit> = (11..22)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: sources.clone(),
                permanence_ticks: vec![permanence_ticks; PAIRS_PER_SEGMENT],
            })
            .collect();
        runtime_with_offsets(
            config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &segments,
        )
    }

    fn events(ids: [u32; ACTIVE_COLUMNS], ticks: [u16; ACTIVE_COLUMNS]) -> [SpikeEvent; 11] {
        std::array::from_fn(|index| SpikeEvent {
            cell_id: ids[index],
            tick: ticks[index],
        })
    }

    fn assert_python_boundary_step(step: &TemporalCircuitStep, event_count: u64) {
        let winner_ticks = [223, 223, 222, 222, 222, 222, 222, 222, 222, 222, 222];
        let basket_ticks = [224, 224, 223, 223, 223, 223, 223, 223, 223, 223, 223];
        let proximal_ticks = [237, 237, 236, 237, 236, 237, 236, 238, 236, 238, 236];
        let proximal_relay_ticks = [242, 242, 241, 242, 241, 242, 241, 243, 241, 243, 241];
        let relay_ticks = [228, 228, 227, 227, 227, 227, 227, 227, 227, 227, 227];
        assert_eq!(step.token_id, 0);
        assert!(!step.predicted);
        assert_eq!(step.predicted_columns, 0);
        assert_eq!(
            step.winner_spikes,
            events(std::array::from_fn(|i| i as u32), winner_ticks)
        );
        assert_eq!(
            step.basket_spikes,
            events(std::array::from_fn(|i| 44 + i as u32), basket_ticks)
        );
        assert_eq!(
            step.inhibition_onset_ticks,
            basket_ticks.map(|tick| tick + 1)
        );
        assert_eq!(
            step.proximal_spikes,
            events(std::array::from_fn(|i| 66 + i as u32), proximal_ticks)
        );
        assert_eq!(
            step.proximal_relay_spikes,
            events(std::array::from_fn(|i| 88 + i as u32), proximal_relay_ticks,)
        );
        assert_eq!(
            step.winner_relay_spikes,
            events(std::array::from_fn(|i| 22 + i as u32), relay_ticks)
        );
        assert_eq!(step.column_simulated_ticks, basket_ticks);
        assert_eq!(
            step.winner_context,
            std::array::from_fn(|i| if i < 11 { i as u32 } else { 11 + i as u32 })
        );
        assert_eq!(
            step.proximal_context,
            std::array::from_fn(|i| if i < 11 { 66 + i as u32 } else { 77 + i as u32 })
        );
        assert_eq!(step.context, step.winner_context);
        assert_eq!(step.event_count, event_count);
    }

    fn assert_python_active_step(step: &TemporalCircuitStep, event_count: u64) {
        let winner_ticks = [176, 176, 177, 176, 176, 177, 176, 176, 176, 176, 176];
        let basket_ticks = [177, 177, 178, 177, 177, 178, 177, 177, 177, 177, 177];
        let proximal_ticks = [237, 237, 237, 238, 237, 236, 237, 237, 236, 237, 237];
        let proximal_relay_ticks = [242, 242, 242, 243, 242, 241, 242, 242, 241, 242, 242];
        let relay_ticks = [181, 181, 182, 181, 181, 182, 181, 181, 181, 181, 181];
        assert_eq!(step.token_id, 1);
        assert!(step.predicted);
        assert_eq!(step.predicted_columns, 11);
        assert_eq!(
            step.winner_spikes,
            events(std::array::from_fn(|i| 11 + i as u32), winner_ticks)
        );
        assert_eq!(
            step.basket_spikes,
            events(std::array::from_fn(|i| 55 + i as u32), basket_ticks)
        );
        assert_eq!(
            step.inhibition_onset_ticks,
            basket_ticks.map(|tick| tick + 1)
        );
        assert_eq!(
            step.proximal_spikes,
            events(std::array::from_fn(|i| 77 + i as u32), proximal_ticks)
        );
        assert_eq!(
            step.proximal_relay_spikes,
            events(std::array::from_fn(|i| 99 + i as u32), proximal_relay_ticks,)
        );
        assert_eq!(
            step.winner_relay_spikes,
            events(std::array::from_fn(|i| 33 + i as u32), relay_ticks)
        );
        assert_eq!(step.column_simulated_ticks, basket_ticks);
        assert_eq!(
            step.winner_context,
            std::array::from_fn(|i| if i < 11 { 11 + i as u32 } else { 22 + i as u32 })
        );
        assert_eq!(
            step.proximal_context,
            std::array::from_fn(|i| if i < 11 { 77 + i as u32 } else { 88 + i as u32 })
        );
        assert_eq!(step.context, step.winner_context);
        assert_eq!(step.event_count, event_count);
    }

    #[test]
    fn literal_python_oracle_multi_token_trace_is_bit_exact() {
        let mut runtime = python_oracle_runtime(255);
        assert_eq!(
            runtime
                .threshold_offsets_mv
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            PYTHON_OFFSET_BITS
        );

        let first = runtime.recognize(0).expect("boundary step");
        let second = runtime.recognize(1).expect("active replay step");
        let third = runtime.recognize(0).expect("novel context step");
        let fourth = runtime.recognize(1).expect("second active replay step");
        assert_python_boundary_step(&first, 1);
        assert_python_active_step(&second, 2);
        assert_python_boundary_step(&third, 3);
        assert_python_active_step(&fourth, 4);
    }

    #[test]
    fn read_only_success_and_stop_make_zero_post_construction_allocations() {
        let mut runtime = python_oracle_runtime(255);
        let (trace, success_allocations) = count_allocations(|| {
            (
                runtime.recognize(0),
                runtime.recognize(1),
                runtime.recognize(0),
                runtime.recognize(1),
            )
        });
        assert_eq!(success_allocations, 0);
        assert!(trace.0.is_ok() && trace.1.is_ok() && trace.2.is_ok() && trace.3.is_ok());

        let config = config(22, 2, TemporalContextPolicy::BoundWinners);
        let sources = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
        let segments = [
            TemporalSegmentInit {
                owner: 22,
                sources: sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 23,
                sources,
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
        ];
        let mut stopped =
            runtime_with_offsets(config, vec![0.0; physical_count(config)], &segments);
        stopped.recognize(0).unwrap();
        let (first_stop, first_stop_allocations) = count_allocations(|| stopped.recognize(1));
        assert_eq!(first_stop_allocations, 0);
        assert!(matches!(
            first_stop,
            Err(TemporalCircuitStop::MultipleActiveOwners { .. })
        ));
        let (poisoned, poisoned_allocations) = count_allocations(|| stopped.recognize(1));
        assert_eq!(poisoned_allocations, 0);
        assert_eq!(poisoned, Err(TemporalCircuitStop::Poisoned));
    }

    #[test]
    fn literal_python_matching_band_trace_is_bit_exact() {
        let mut runtime = python_oracle_runtime(205);
        runtime.recognize(0).expect("boundary step");
        let matching = runtime.recognize(1).expect("matching step");
        let winner_ticks = [199, 198, 199, 199, 199, 199, 199, 199, 199, 199, 199];
        let basket_ticks = [200, 199, 200, 200, 200, 200, 200, 200, 200, 200, 200];
        assert!(!matching.predicted);
        assert_eq!(matching.predicted_columns, 0);
        assert_eq!(
            matching.winner_spikes,
            events(std::array::from_fn(|i| 11 + i as u32), winner_ticks)
        );
        assert_eq!(
            matching.basket_spikes,
            events(std::array::from_fn(|i| 55 + i as u32), basket_ticks)
        );
        assert_eq!(
            matching.inhibition_onset_ticks,
            basket_ticks.map(|tick| tick + 1)
        );
        assert_eq!(matching.event_count, 2);
    }

    #[test]
    fn one_step_physically_classifies_active_matching_and_novel_columns() {
        let config = config(22, 1, TemporalContextPolicy::BoundWinners);
        let sources = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let segments = [
            TemporalSegmentInit {
                owner: 11,
                sources: sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 12,
                sources,
                permanence_ticks: vec![205; PAIRS_PER_SEGMENT],
            },
        ];
        let mut runtime =
            runtime_with_offsets(config, vec![0.0; physical_count(config)], &segments);
        runtime.recognize(0).expect("context step");
        let step = runtime.recognize(1).expect("mixed classification step");

        assert_eq!(step.predicted_columns, 1);
        assert!(!step.predicted);
        assert_eq!(step.winner_spikes[0].cell_id, 11);
        assert_eq!(step.winner_spikes[1].cell_id, 12);
        assert_eq!(step.winner_spikes[2].cell_id, 13);
        assert_eq!(
            step.winner_spikes[0].tick,
            singleton_latency_ticks(CandidateClass::Active { load: 1 }.current_pa(), 0.0).unwrap()
        );
        assert_eq!(
            step.winner_spikes[1].tick,
            singleton_latency_ticks(CandidateClass::Matching { load: 1 }.current_pa(), 0.0)
                .unwrap()
        );
        assert_eq!(
            step.winner_spikes[2].tick,
            singleton_latency_ticks(CandidateClass::Novel { rank: 0, load: 0 }.current_pa(), 0.0)
                .unwrap()
        );
    }

    #[test]
    fn least_loaded_filter_precedes_higher_afferent_score_in_a_real_step() {
        let config = config(22, 2, TemporalContextPolicy::BoundWinners);
        let dummy_segment = TemporalSegmentInit {
            owner: 23,
            sources: vec![22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
        };
        let mut runtime =
            runtime_with_offsets(config, vec![0.0; physical_count(config)], &[dummy_segment]);
        let source = runtime.recognize(0).expect("context step");
        assert_eq!(
            source
                .winner_spikes
                .iter()
                .map(|event| event.cell_id)
                .collect::<Vec<_>>(),
            (0..22).step_by(2).collect::<Vec<_>>()
        );
        assert_eq!(
            afferent_score(17, &source.context, 22),
            167_407_126_750_786_059_617
        );
        assert_eq!(
            afferent_score(17, &source.context, 23),
            188_988_591_163_576_260_026
        );
        let target = runtime.recognize(1).expect("novel target step");
        assert_eq!(target.winner_spikes[0].cell_id, 22);
    }

    #[test]
    fn multiple_active_and_matching_owners_stop_before_transient_commit() {
        for (ticks, expect_active) in [(255, true), (205, false)] {
            let config = config(22, 2, TemporalContextPolicy::BoundWinners);
            let sources = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
            let segments = [
                TemporalSegmentInit {
                    owner: 22,
                    sources: sources.clone(),
                    permanence_ticks: vec![ticks; PAIRS_PER_SEGMENT],
                },
                TemporalSegmentInit {
                    owner: 23,
                    sources,
                    permanence_ticks: vec![ticks; PAIRS_PER_SEGMENT],
                },
            ];
            let mut runtime =
                runtime_with_offsets(config, vec![0.0; physical_count(config)], &segments);
            runtime.recognize(0).expect("context step");
            let before_context = runtime.previous_context;
            let before_winner_context = runtime.previous_winner_context;
            let before_event = runtime.event_count;
            let stop = runtime.recognize(1).unwrap_err();
            if expect_active {
                assert_eq!(
                    stop,
                    TemporalCircuitStop::MultipleActiveOwners {
                        column: 11,
                        owners: OwnerConflict {
                            first: 22,
                            second: 23,
                            count: 2,
                        },
                    }
                );
            } else {
                assert_eq!(
                    stop,
                    TemporalCircuitStop::MultipleMatchingOwners {
                        column: 11,
                        owners: OwnerConflict {
                            first: 22,
                            second: 23,
                            count: 2,
                        },
                    }
                );
            }
            assert_eq!(runtime.previous_context, before_context);
            assert_eq!(runtime.previous_winner_context, before_winner_context);
            assert_eq!(runtime.event_count, before_event);
            assert_eq!(runtime.recognize(1), Err(TemporalCircuitStop::Poisoned));
        }
    }

    #[test]
    fn literal_python_multiple_active_owner_stop_is_atomic_and_reset_keeps_poison() {
        let config = config(22, 4, TemporalContextPolicy::BoundWinners);
        let sources = vec![0, 8, 12, 16, 20, 24, 28, 32, 36, 40];
        let segments = [
            TemporalSegmentInit {
                owner: 44,
                sources: sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 45,
                sources,
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
        ];
        let mut runtime =
            runtime_with_offsets(config, vec![0.0; physical_count(config)], &segments);
        let boundary = runtime.recognize(0).expect("boundary step");
        let expected_context: [u32; CONTEXT_CELLS] = std::array::from_fn(|index| {
            if index < ACTIVE_COLUMNS {
                (4 * index) as u32
            } else {
                (44 + 4 * index) as u32
            }
        });
        assert_eq!(boundary.context, expected_context);
        assert_eq!(runtime.event_count, 1);

        assert_eq!(
            runtime.recognize(1),
            Err(TemporalCircuitStop::MultipleActiveOwners {
                column: 11,
                owners: OwnerConflict {
                    first: 44,
                    second: 45,
                    count: 2,
                },
            })
        );
        assert_eq!(runtime.previous_context, expected_context);
        assert_eq!(runtime.previous_winner_context, expected_context);
        assert_eq!(runtime.event_count, 1);

        runtime.reset_transient();
        assert_eq!(runtime.previous_context_len, 0);
        assert_eq!(runtime.previous_winner_context_len, 0);
        assert_eq!(runtime.event_count, 0);
        assert_eq!(runtime.recognize(0), Err(TemporalCircuitStop::Poisoned));
    }

    #[test]
    fn context_policy_preserves_bound_winner_vs_generic_afferent_semantics() {
        let bound_config = config(22, 2, TemporalContextPolicy::BoundWinners);
        let generic_config = config(22, 2, TemporalContextPolicy::GenericAfferents);
        let bound_sources = vec![0, 4, 6, 8, 10, 12, 14, 16, 18, 20];
        let generic_sources = vec![110, 112, 113, 114, 115, 116, 117, 118, 119, 120];
        let bound_segments: Vec<_> = (22..44)
            .step_by(2)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: bound_sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let generic_segments: Vec<_> = (22..44)
            .step_by(2)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: generic_sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut bound = runtime_with_offsets(
            bound_config,
            vec![0.0; physical_count(bound_config)],
            &bound_segments,
        );
        let mut generic = runtime_with_offsets(
            generic_config,
            vec![0.0; physical_count(generic_config)],
            &generic_segments,
        );

        let bound_source = bound.recognize(0).unwrap();
        let generic_source = generic.recognize(0).unwrap();
        assert_eq!(bound_source.winner_spikes, generic_source.winner_spikes);
        assert_eq!(
            bound_source.winner_context,
            std::array::from_fn(|index| {
                if index < ACTIVE_COLUMNS {
                    (2 * index) as u32
                } else {
                    (22 + 2 * index) as u32
                }
            })
        );
        assert_eq!(
            generic_source.proximal_context,
            std::array::from_fn(|index| {
                if index < ACTIVE_COLUMNS {
                    110 + index as u32
                } else {
                    121 + index as u32
                }
            })
        );
        assert_eq!(bound_source.context, bound_source.winner_context);
        assert_eq!(generic_source.context, generic_source.proximal_context);
        assert_ne!(bound_source.context, generic_source.context);
        assert_eq!(
            bound.previous_winner_context,
            generic.previous_winner_context
        );
        assert_eq!(bound.recognize(1).unwrap().predicted_columns, 11);
        let generic_target = generic.recognize(1).unwrap();
        assert_eq!(generic_target.predicted_columns, 11);
        assert_eq!(
            generic_target.context,
            std::array::from_fn(|index| {
                if index < ACTIVE_COLUMNS {
                    121 + index as u32
                } else {
                    132 + index as u32
                }
            })
        );
    }

    #[test]
    fn reset_clears_only_transient_context_and_event_counter() {
        let mut runtime = python_oracle_runtime(255);
        let learned_loads: Vec<u8> = (0..22)
            .map(|owner| runtime.owner_load(owner).unwrap())
            .collect();
        runtime.recognize(0).unwrap();
        runtime.recognize(1).unwrap();
        runtime.recognize(0).unwrap();
        assert_eq!(runtime.recognize(1).unwrap().event_count, 4);
        runtime.reset_transient();

        assert_eq!(runtime.previous_context_len, 0);
        assert_eq!(runtime.previous_winner_context_len, 0);
        assert_eq!(runtime.event_count, 0);
        assert_eq!(
            (0..22)
                .map(|owner| runtime.owner_load(owner))
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            learned_loads
        );
        assert_python_boundary_step(&runtime.recognize(0).unwrap(), 1);
        assert_python_active_step(&runtime.recognize(1).unwrap(), 2);
    }

    #[test]
    fn context_storage_is_fixed_and_keeps_its_address_across_tokens_and_reset() {
        let mut runtime = python_oracle_runtime(255);
        let context_address = runtime.previous_context.as_ptr();
        let winner_context_address = runtime.previous_winner_context.as_ptr();
        assert_eq!(runtime.previous_context_len, 0);
        assert_eq!(runtime.previous_winner_context_len, 0);

        runtime.recognize(0).unwrap();
        runtime.recognize(1).unwrap();
        assert_eq!(runtime.previous_context_len, CONTEXT_CELLS as u8);
        assert_eq!(runtime.previous_winner_context_len, CONTEXT_CELLS as u8);
        assert_eq!(runtime.previous_context.as_ptr(), context_address);
        assert_eq!(
            runtime.previous_winner_context.as_ptr(),
            winner_context_address
        );

        runtime.reset_transient();
        assert_eq!(runtime.previous_context_len, 0);
        assert_eq!(runtime.previous_winner_context_len, 0);
        assert_eq!(runtime.previous_context, [0; CONTEXT_CELLS]);
        assert_eq!(runtime.previous_winner_context, [0; CONTEXT_CELLS]);
        assert_eq!(runtime.previous_context.as_ptr(), context_address);
        assert_eq!(
            runtime.previous_winner_context.as_ptr(),
            winner_context_address
        );
    }

    #[test]
    fn physical_no_spike_multiple_spikes_and_missing_basket_fail_closed() {
        let base_two = config(22, 2, TemporalContextPolicy::BoundWinners);
        let mut no_spike_offsets = vec![0.0; physical_count(base_two)];
        for offset in &mut no_spike_offsets[..22] {
            *offset = 1.0e9;
        }
        let mut no_spike = runtime_with_offsets(base_two, no_spike_offsets, &[]);
        assert!(matches!(
            no_spike.recognize(0),
            Err(TemporalCircuitStop::Physics {
                stage: PhysicalStage::ColumnRace,
                source: TemporalPhysicsStop::NoPyramidalSpike,
                ..
            })
        ));
        assert_eq!(no_spike.event_count, 0);

        let mut multiple_offsets = vec![0.0; physical_count(base_two)];
        multiple_offsets[0] = -100.0;
        multiple_offsets[1] = -100.0;
        let mut multiple = runtime_with_offsets(base_two, multiple_offsets, &[]);
        assert!(matches!(
            multiple.recognize(0),
            Err(TemporalCircuitStop::Physics {
                stage: PhysicalStage::ColumnRace,
                source: TemporalPhysicsStop::MultiplePyramidalSpikes { .. },
                ..
            })
        ));
        assert_eq!(multiple.event_count, 0);

        let base_one = config(22, 1, TemporalContextPolicy::BoundWinners);
        let mut missing_offsets = vec![0.0; physical_count(base_one)];
        missing_offsets[(2 * base_one.max_pyramidal_cells) as usize] = 1.0e9;
        let mut missing = runtime_with_offsets(base_one, missing_offsets, &[]);
        assert!(matches!(
            missing.recognize(0),
            Err(TemporalCircuitStop::Physics {
                stage: PhysicalStage::ColumnRace,
                source: TemporalPhysicsStop::BasketInhibitionMissing { .. },
                ..
            })
        ));
        assert_eq!(missing.event_count, 0);
    }

    #[test]
    fn independent_arm_runtime_physically_races_instead_of_copying_a_winner() {
        let config = config(22, 2, TemporalContextPolicy::BoundWinners);
        let mut arm_one = runtime_with_offsets(config, vec![0.0; physical_count(config)], &[]);
        let context = arm_one.recognize(0).unwrap().context;
        let first_column = [22u32, 23u32];
        let expected_arm_one = *first_column
            .iter()
            .max_by_key(|&&cell| afferent_score(17, &context, cell))
            .unwrap();
        let mut arm_two_offsets = vec![0.0; physical_count(config)];
        arm_two_offsets[expected_arm_one as usize] = 1.0e9;
        let mut arm_two = runtime_with_offsets(config, arm_two_offsets, &[]);
        arm_two.recognize(0).unwrap();

        let arm_one_winner = arm_one.recognize(1).unwrap().winner_spikes[0].cell_id;
        let arm_two_winner = arm_two.recognize(1).unwrap().winner_spikes[0].cell_id;
        assert_eq!(arm_one_winner, expected_arm_one);
        assert_ne!(arm_two_winner, arm_one_winner);
        assert!(first_column.contains(&arm_two_winner));
    }

    #[test]
    fn construction_requires_complete_compact_reserve_cohorts_and_explicit_capacity() {
        let mut reserve_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        reserve_config.max_pyramidal_cells = 24;
        reserve_config.segment_capacity = 1;
        let mut complete = base_members(reserve_config);
        complete[0].extend([22, 23]);
        let offsets = vec![0.0; physical_count(reserve_config)];
        assert!(TemporalCircuitRuntime::new(
            reserve_config,
            two_token_codebook(),
            complete.clone(),
            offsets.clone(),
            &[],
        )
        .is_ok());

        complete[0].pop();
        assert!(matches!(
            TemporalCircuitRuntime::new(
                reserve_config,
                two_token_codebook(),
                complete,
                offsets,
                &[],
            ),
            Err(TemporalRuntimeBuildError::Invalid(_))
        ));
        reserve_config.segment_capacity = 0;
        assert!(matches!(
            TemporalCircuitRuntime::new(
                reserve_config,
                two_token_codebook(),
                base_members(reserve_config),
                vec![0.0; physical_count(reserve_config)],
                &[],
            ),
            Err(TemporalRuntimeBuildError::Invalid(_))
        ));

        let generic_base_one = config(22, 1, TemporalContextPolicy::GenericAfferents);
        assert!(matches!(
            TemporalCircuitRuntime::new(
                generic_base_one,
                two_token_codebook(),
                base_members(generic_base_one),
                vec![0.0; physical_count(generic_base_one)],
                &[],
            ),
            Err(TemporalRuntimeBuildError::Invalid(_))
        ));
    }

    #[test]
    fn runtime_import_accepts_every_canonical_zero_to_ten_live_pair_width() {
        for live_pairs in 0..=PAIRS_PER_SEGMENT {
            let mut config = config(22, 1, TemporalContextPolicy::BoundWinners);
            config.segment_capacity = 1;
            config.theta_match = 1;
            let segment = TemporalSegmentInit {
                owner: 11,
                sources: (0..live_pairs as u32).collect(),
                permanence_ticks: vec![255; live_pairs],
            };
            let mut runtime =
                runtime_with_offsets(config, vec![0.0; physical_count(config)], &[segment]);
            assert_eq!(runtime.owner_load(11), Ok(1), "live_pairs={live_pairs}");
            runtime.recognize(0).unwrap();
            runtime.recognize(1).unwrap();
        }

        let config = config(22, 1, TemporalContextPolicy::BoundWinners);
        let mismatched = TemporalSegmentInit {
            owner: 11,
            sources: vec![0],
            permanence_ticks: Vec::new(),
        };
        assert!(matches!(
            TemporalCircuitRuntime::new(
                config,
                two_token_codebook(),
                base_members(config),
                vec![0.0; physical_count(config)],
                &[mismatched],
            ),
            Err(TemporalRuntimeBuildError::Store(_))
        ));
    }

    #[test]
    fn failing_token_is_atomic_and_scientific_stop_latches_poison() {
        let config = config(22, 1, TemporalContextPolicy::BoundWinners);
        let mut offsets = vec![0.0; physical_count(config)];
        let second_token_first_proximal =
            2 * config.max_pyramidal_cells + config.column_count + ACTIVE_COLUMNS as u32;
        offsets[second_token_first_proximal as usize] = 1.0e9;
        let mut runtime = runtime_with_offsets(config, offsets, &[]);

        assert_eq!(
            runtime.recognize(99),
            Err(TemporalCircuitStop::UnknownToken(99))
        );
        assert_eq!(runtime.event_count, 0);
        assert_eq!(runtime.previous_context_len, 0);
        assert!(runtime.scientific_stop.is_none());

        runtime.recognize(0).expect("the first token must succeed");
        let before_context = runtime.previous_context;
        let before_winner_context = runtime.previous_winner_context;
        let before_event = runtime.event_count;
        assert!(matches!(
            runtime.recognize(1),
            Err(TemporalCircuitStop::Physics {
                stage: PhysicalStage::Proximal,
                source: TemporalPhysicsStop::NoPyramidalSpike,
                ..
            })
        ));
        assert_eq!(runtime.previous_context, before_context);
        assert_eq!(runtime.previous_winner_context, before_winner_context);
        assert_eq!(runtime.event_count, before_event);
        assert_eq!(runtime.recognize(0), Err(TemporalCircuitStop::Poisoned));
    }

    #[test]
    fn poison_codes_match_the_canonical_checkpoint_contract() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<TemporalPoisonCode>();
        assert_copy::<TemporalCircuitStop>();

        assert_eq!(TemporalPoisonCode::Unpoisoned as u16, 0);
        assert_eq!(TemporalPoisonCode::MultipleActiveOwners as u16, 1);
        assert_eq!(TemporalPoisonCode::MultipleMatchingOwners as u16, 2);
        assert_eq!(TemporalPoisonCode::CategorizedRaceCapacity as u16, 3);
        assert_eq!(TemporalPoisonCode::MalformedContextOrPolicy as u16, 4);
        assert_eq!(TemporalPoisonCode::ProximalPhysics as u16, 5);
        assert_eq!(TemporalPoisonCode::ProximalRelayPhysics as u16, 6);
        assert_eq!(TemporalPoisonCode::ColumnRacePhysics as u16, 7);
        assert_eq!(TemporalPoisonCode::WinnerRelayPhysics as u16, 8);
        assert_eq!(TemporalPoisonCode::TickOrCounterOverflow as u16, 9);
        assert_eq!(TemporalPoisonCode::StoreOrPlanInvariant as u16, 10);
        assert_eq!(TemporalPoisonCode::PhiloxAddressExhaustion as u16, 11);
        assert_eq!(TemporalPoisonCode::ReserveCellCapacity as u16, 12);
        assert_eq!(TemporalPoisonCode::GlobalSegmentCapacity as u16, 13);
        assert_eq!(TemporalPoisonCode::OwnerSegmentCapacity as u16, 14);
        assert_eq!(TemporalPoisonCode::PreallocatedCapacityInvariant as u16, 15);

        assert_eq!(
            TemporalCircuitStop::MultipleActiveOwners {
                column: 11,
                owners: OwnerConflict {
                    first: 22,
                    second: 23,
                    count: 2,
                },
            }
            .poison_code(),
            Some(TemporalPoisonCode::MultipleActiveOwners),
        );
        let conflict = OwnerConflict {
            first: 22,
            second: 23,
            count: 2,
        };
        let mapped = [
            (
                TemporalCircuitStop::MultipleMatchingOwners {
                    column: 11,
                    owners: conflict,
                },
                TemporalPoisonCode::MultipleMatchingOwners,
            ),
            (
                TemporalCircuitStop::MalformedContextOrPolicy("literal fixture"),
                TemporalPoisonCode::MalformedContextOrPolicy,
            ),
            (
                TemporalCircuitStop::Physics {
                    stage: PhysicalStage::Proximal,
                    location: 1,
                    source: TemporalPhysicsStop::NoPyramidalSpike,
                },
                TemporalPoisonCode::ProximalPhysics,
            ),
            (
                TemporalCircuitStop::Physics {
                    stage: PhysicalStage::ProximalRelay,
                    location: 2,
                    source: TemporalPhysicsStop::NoPyramidalSpike,
                },
                TemporalPoisonCode::ProximalRelayPhysics,
            ),
            (
                TemporalCircuitStop::Physics {
                    stage: PhysicalStage::ColumnRace,
                    location: 3,
                    source: TemporalPhysicsStop::NoPyramidalSpike,
                },
                TemporalPoisonCode::ColumnRacePhysics,
            ),
            (
                TemporalCircuitStop::Physics {
                    stage: PhysicalStage::WinnerRelay,
                    location: 4,
                    source: TemporalPhysicsStop::NoPyramidalSpike,
                },
                TemporalPoisonCode::WinnerRelayPhysics,
            ),
            (
                TemporalCircuitStop::StoreRecognition(1),
                TemporalPoisonCode::StoreOrPlanInvariant,
            ),
            (
                TemporalCircuitStop::StoreOrPlanInvariant("literal fixture"),
                TemporalPoisonCode::StoreOrPlanInvariant,
            ),
            (
                TemporalCircuitStop::PreallocatedCapacityInvariant("literal fixture"),
                TemporalPoisonCode::PreallocatedCapacityInvariant,
            ),
        ];
        for (stop, code) in mapped {
            assert_eq!(stop.poison_code(), Some(code), "{stop:?}");
        }
        assert_eq!(
            TemporalCircuitStop::CategorizedRaceCapacity {
                column: 11,
                categorized_count: 2,
                race_cap: 1,
            }
            .poison_code(),
            Some(TemporalPoisonCode::CategorizedRaceCapacity),
        );
        assert_eq!(
            TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::WinnerRelayTick,
                location: 33,
            }
            .poison_code(),
            Some(TemporalPoisonCode::TickOrCounterOverflow),
        );
        assert_eq!(TemporalCircuitStop::UnknownToken(99).poison_code(), None,);
        assert_eq!(TemporalCircuitStop::Poisoned.poison_code(), None);
    }

    #[test]
    fn novel_ranking_orders_load_then_afferent_score_then_gid() {
        let candidates = [
            NovelCandidate {
                cell_id: 1,
                column_offset: 0,
                load: 1,
                afferent_score: u128::MAX,
            },
            NovelCandidate {
                cell_id: 9,
                column_offset: 1,
                load: 0,
                afferent_score: 7,
            },
            NovelCandidate {
                cell_id: 3,
                column_offset: 2,
                load: 0,
                afferent_score: 7,
            },
            NovelCandidate {
                cell_id: 4,
                column_offset: 3,
                load: 0,
                afferent_score: 8,
            },
        ];
        let mut ranked = [EMPTY_NOVEL_CANDIDATE; 8];
        let ranked_count = select_novel_candidates(&candidates, 3, &mut ranked);
        assert_eq!(
            ranked[..ranked_count]
                .iter()
                .map(|candidate| candidate.cell_id)
                .collect::<Vec<_>>(),
            vec![4, 3, 9]
        );
    }

    #[test]
    fn afferent_scores_match_literal_python_integer_oracle() {
        let context: Vec<u32> = (0..11).chain(22..33).collect();
        assert_eq!(
            afferent_score(17, &context, 11),
            221_370_631_115_127_163_049
        );
        assert_eq!(
            afferent_score(17, &context, 12),
            171_141_027_893_710_194_730
        );
        assert_eq!(
            afferent_score(17, &context, 13),
            221_269_005_864_163_898_031
        );
        assert_eq!(
            afferent_score(17, &context, 14),
            243_941_804_804_712_821_922
        );
        let reversed: Vec<u32> = context.iter().rev().copied().collect();
        assert_eq!(
            afferent_score(17, &context, 14),
            afferent_score(17, &reversed, 14)
        );
    }
}
