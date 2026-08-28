//! Deep temporal circuit runtime.
//!
//! This is the first physical single-step beneath the future Python-facing
//! temporal circuit. It owns token decoding, materialized column cohorts,
//! learned segment recognition, fitted-cell races, and transient physical
//! context. Its only inference input is a token ID. In particular, no caller
//! can inject winners, segment classes, currents, or a prepared mutation.
//!
//! Claim boundary: this module owns deterministic physical recognition and the
//! aggregate reserve-aware native learning transaction. Its crate-private
//! checkpoint-v1 child module owns identity, canonical streaming export,
//! reachability validation, and direct one-shot import, but no filesystem
//! publisher, public ABI, Python wrapper, or outcome-evaluation path.

use crate::random123::{philox2x64_10, sample_without_replacement};
use crate::temporal::{
    TemporalOwnerDiagnostic, TemporalSegmentRow, TemporalStore, TemporalStoreConfig,
    TemporalStorePreflightError,
};
use crate::temporal_physics::{
    coupled_first_spike_race, singleton_latency_ticks, BasketDrive, CandidateClass, CandidateDrive,
    CoupledRaceScratch, SpikeEvent, TemporalPhysicsStop, TEMPORAL_PHYSICS_IDENTITY,
};

pub(crate) mod checkpoint;
#[cfg(any(unix, windows))]
mod checkpoint_publisher;
mod ffi;
mod role_place_copy;

use role_place_copy::{is_internal_role_token, CopySlot, RolePlaceCopyState, COPY_END_TOKEN_ID};

#[cfg(not(any(unix, windows)))]
compile_error!("checkpoint path publication requires an audited Unix or Windows backend");

const ACTIVE_COLUMNS: usize = 11;
const CONTEXT_CELLS: usize = 2 * ACTIVE_COLUMNS;
const CONTEXT_CELLS_U8: u8 = CONTEXT_CELLS as u8;
const PAIRS_PER_SEGMENT: usize = 10;
const NONE: u32 = u32::MAX;
const INITIAL_CONTACT_DOMAIN: u64 = 0x5233_4630_494E_4954;
const REGROWTH_CONTACT_DOMAIN: u64 = 0x5233_4630_5247_5257;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalContextPolicy {
    BoundWinners,
    GenericAfferents,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalPlasticityPolicy {
    GrowthOff,
    Plastic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalStepMode {
    ReadOnly,
    Learn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TemporalCircuitConfig {
    pub(crate) column_count: u32,
    pub(crate) base_cells_per_column: u8,
    pub(crate) max_pyramidal_cells: u32,
    pub(crate) segment_capacity: u32,
    pub(crate) novel_race_cap: u8,
    pub(crate) prediction_quorum: u8,
    pub(crate) construction_seed: u32,
    pub(crate) connected_permanence_ticks: u16,
    pub(crate) initial_permanence_ticks: u16,
    pub(crate) permanence_increment_ticks: u16,
    pub(crate) negative_decrement_ticks: u16,
    pub(crate) max_segments_per_cell: u8,
    pub(crate) theta_match: u8,
    pub(crate) theta_active: u8,
    pub(crate) context_policy: TemporalContextPolicy,
    pub(crate) plasticity_policy: TemporalPlasticityPolicy,
}

#[repr(C)]
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
    LearningCounter,
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
    // Historical checkpoint terminal classes.  Physical multi-owner
    // arbitration replaced both live execution paths, so the runtime never
    // emits these values; they remain decodable evidence in checkpoint-v1.
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
pub(crate) enum TemporalCircuitStop {
    UnknownToken(u32),
    Poisoned,
    MalformedContextOrPolicy(&'static str),
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
    PhiloxAddressExhaustion {
        operation: &'static str,
        segment_id: u32,
    },
    ReserveCellCapacity {
        column: u32,
    },
    GlobalSegmentCapacity {
        required: u32,
        capacity: u32,
    },
    OwnerSegmentCapacity {
        owner: u32,
        cap: u8,
    },
    PreallocatedCapacityInvariant(&'static str),
}

impl TemporalCircuitStop {
    fn poison_code(&self) -> Option<TemporalPoisonCode> {
        match self {
            Self::UnknownToken(_) | Self::Poisoned => None,
            Self::MalformedContextOrPolicy(_) => Some(TemporalPoisonCode::MalformedContextOrPolicy),
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
            Self::PhiloxAddressExhaustion { .. } => {
                Some(TemporalPoisonCode::PhiloxAddressExhaustion)
            }
            Self::ReserveCellCapacity { .. } => Some(TemporalPoisonCode::ReserveCellCapacity),
            Self::GlobalSegmentCapacity { .. } => Some(TemporalPoisonCode::GlobalSegmentCapacity),
            Self::OwnerSegmentCapacity { .. } => Some(TemporalPoisonCode::OwnerSegmentCapacity),
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
    pub(crate) allocated_segments: u8,
    pub(crate) punished_segments: u32,
    pub(crate) candidate_segments: usize,
    pub(crate) matching_segments: usize,
    pub(crate) active_segments: usize,
    pub(crate) postings_visited: u64,
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
struct PhysicalOutputElection {
    pub(crate) token_id: u32,
    pub(crate) output_cell_id: u32,
    pub(crate) activated_columns: u32,
    pub(crate) candidate_tokens: u32,
    pub(crate) winner_input_columns: u8,
    pub(crate) winner_tick: u16,
    pub(crate) basket_tick: u16,
    pub(crate) inhibition_onset_tick: u16,
    pub(crate) simulated_ticks: u16,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalGenerationPath {
    Temporal = 0,
    EnterCopy = 1,
    Copy = 2,
    ExitCopy = 3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TemporalGeneration {
    pub(crate) surface_token_id: u32,
    pub(crate) surface_output_cell_id: u32,
    pub(crate) generation_index: u64,
    pub(crate) temporal_event_count: u64,
    pub(crate) activated_columns: u32,
    pub(crate) candidate_tokens: u32,
    pub(crate) control_token_id: u32,
    pub(crate) control_output_cell_id: u32,
    pub(crate) surface_winner_tick: u16,
    pub(crate) surface_basket_tick: u16,
    pub(crate) surface_inhibition_onset_tick: u16,
    pub(crate) total_simulated_ticks: u16,
    pub(crate) control_winner_tick: u16,
    pub(crate) control_basket_tick: u16,
    pub(crate) control_simulated_end_tick: u16,
    pub(crate) path: TemporalGenerationPath,
    pub(crate) winner_input_columns: u8,
    pub(crate) feedback_predicted_columns: u8,
    pub(crate) feedback_predicted: bool,
    pub(crate) copy_place_ordinal: u8,
    pub(crate) physical_election_count: u8,
    pub(crate) copy_slot: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NovelCandidate {
    cell_id: u32,
    column_offset: usize,
    load: u8,
    afferent_score: u128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CategorizedRankCandidate {
    owner: u32,
    column_offset: usize,
    primary_contacts: u8,
    secondary_contacts: u8,
    segment_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedReservePlan {
    cohort_columns: [u32; ACTIVE_COLUMNS],
    cohort_count: u8,
    next_reserve_cell: u32,
}

impl PreparedReservePlan {
    fn cohort_count(self) -> usize {
        usize::from(self.cohort_count)
    }

    fn recruits(self) -> bool {
        self.cohort_count != 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreparedLearnStep {
    reserve: PreparedReservePlan,
    active_segment_count: usize,
    winner_cells: [u32; ACTIVE_COLUMNS],
    winner_best_active: [u32; ACTIVE_COLUMNS],
    normalized_context_sources: [u32; ACTIVE_COLUMNS],
    reinforced_rows: [TemporalSegmentRow; ACTIVE_COLUMNS],
    append_owners: [u32; ACTIVE_COLUMNS],
    append_count: u8,
    punished_count: u32,
    initial_row: TemporalSegmentRow,
    next_learning_step: u64,
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
const EMPTY_CATEGORIZED_RANK_CANDIDATE: CategorizedRankCandidate = CategorizedRankCandidate {
    owner: NONE,
    column_offset: usize::MAX,
    primary_contacts: 0,
    secondary_contacts: 0,
    segment_id: NONE,
};
const EMPTY_RESERVE_PLAN: PreparedReservePlan = PreparedReservePlan {
    cohort_columns: [NONE; ACTIVE_COLUMNS],
    cohort_count: 0,
    next_reserve_cell: 0,
};
const EMPTY_SEGMENT_ROW: TemporalSegmentRow = TemporalSegmentRow {
    normalized_sources: [NONE; PAIRS_PER_SEGMENT],
    permanence_ticks: [0; PAIRS_PER_SEGMENT],
    live_pair_count: 0,
};

/// Append into capacity proven by the aggregate transaction preflight without
/// making an allocation-capable `Vec::push` call after commit entry.
fn append_prevalidated_slice<T: Copy>(values: &mut Vec<T>, appended: &[T]) {
    let original_len = values.len();
    debug_assert!(values.capacity() - original_len >= appended.len());
    for (destination, value) in values.spare_capacity_mut()[..appended.len()]
        .iter_mut()
        .zip(appended.iter().copied())
    {
        destination.write(value);
    }
    // SAFETY: aggregate preflight proved the spare capacity and the loop above
    // initialized exactly `appended.len()` consecutive elements.
    unsafe {
        values.set_len(original_len + appended.len());
    }
}

pub(crate) struct TemporalCircuitRuntime {
    config: TemporalCircuitConfig,
    token_codebook: Vec<TokenCode>,
    column_offsets: Vec<usize>,
    column_cells: Vec<u32>,
    alternate_column_offsets: Vec<usize>,
    alternate_column_cells: Vec<u32>,
    cohort_columns: Vec<u32>,
    next_reserve_cell: u32,
    threshold_offsets_mv: Vec<f64>,
    store: TemporalStore,
    recognition_scratch: Vec<TemporalOwnerDiagnostic>,
    active_segment_scratch: Vec<u32>,
    column_drive_scratch: Vec<CandidateDrive>,
    novel_candidate_scratch: Vec<NovelCandidate>,
    race_scratch: CoupledRaceScratch,
    previous_context: [u32; CONTEXT_CELLS],
    previous_context_len: u8,
    previous_winner_context: [u32; CONTEXT_CELLS],
    previous_winner_context_len: u8,
    event_count: u64,
    learning_step: u64,
    scientific_stop: Option<TemporalPoisonCode>,
    role_place_copy: RolePlaceCopyState,
    generation_index: u64,
    relay_offset: u32,
    basket_offset: u32,
    proximal_offset: u32,
    proximal_relay_offset: u32,
}

#[derive(Clone, Copy)]
struct TemporalTransientState {
    previous_context: [u32; CONTEXT_CELLS],
    previous_context_len: u8,
    previous_winner_context: [u32; CONTEXT_CELLS],
    previous_winner_context_len: u8,
    event_count: u64,
}

impl TemporalTransientState {
    const EMPTY: Self = Self {
        previous_context: [0; CONTEXT_CELLS],
        previous_context_len: 0,
        previous_winner_context: [0; CONTEXT_CELLS],
        previous_winner_context_len: 0,
        event_count: 0,
    };
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
        let segment_capacity = usize::try_from(config.segment_capacity)
            .map_err(|_| TemporalRuntimeBuildError::Invalid("segment_capacity exceeds usize"))?;
        if column_count < ACTIVE_COLUMNS
            || base_width == 0
            || race_cap == 0
            || race_cap > 8
            || base_width > race_cap
            || config.segment_capacity == 0
            || config.segment_capacity > u32::MAX / PAIRS_PER_SEGMENT as u32
            || segment_capacity < segments.len()
            || (config.context_policy == TemporalContextPolicy::GenericAfferents && base_width < 2)
            || !(1..=ACTIVE_COLUMNS as u8).contains(&config.prediction_quorum)
            || !(1..=500).contains(&config.connected_permanence_ticks)
            || !(1..=500).contains(&config.initial_permanence_ticks)
            || !(1..=500).contains(&config.permanence_increment_ticks)
            || !(1..=500).contains(&config.negative_decrement_ticks)
            || config.max_segments_per_cell == 0
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
        let mut column_cells = Vec::with_capacity(owner_count);
        let mut committed = vec![false; owner_count];
        let mut recruited = Vec::with_capacity(owner_count - base_total);
        let mut cohort_records = Vec::new();
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
            for (cohort_index, cohort) in cells[base_width..].chunks_exact(race_cap).enumerate() {
                if cohort
                    .windows(2)
                    .any(|pair| pair[1] != pair[0].saturating_add(1))
                {
                    return Err(TemporalRuntimeBuildError::Invalid(
                        "reserve cohorts must be contiguous physical identities",
                    ));
                }
                cohort_records.push((cohort[0], column, base_width + cohort_index * race_cap));
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
        cohort_records.sort_unstable_by_key(|record| record.0);
        let maximum_cohort_count = (owner_count - base_total) / race_cap;
        let mut cohort_columns = Vec::with_capacity(maximum_cohort_count);
        for (cohort_index, &(first_gid, column, _column_offset)) in
            cohort_records.iter().enumerate()
        {
            let expected_first = base_total
                .checked_add(cohort_index.checked_mul(race_cap).ok_or(
                    TemporalRuntimeBuildError::Invalid("reserve cohort identity overflows usize"),
                )?)
                .ok_or(TemporalRuntimeBuildError::Invalid(
                    "reserve cohort identity overflows usize",
                ))?;
            if first_gid as usize != expected_first {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "reserve cohorts do not cover the deterministic global prefix",
                ));
            }
            cohort_columns.push(column as u32);
        }
        let next_reserve_index =
            base_total
                .checked_add(recruited.len())
                .ok_or(TemporalRuntimeBuildError::Invalid(
                    "reserve cursor overflows usize",
                ))?;
        let next_reserve_cell = u32::try_from(next_reserve_index).map_err(|_| {
            TemporalRuntimeBuildError::Invalid("reserve cursor exceeds the uint32 domain")
        })?;

        if token_codebook.is_empty() {
            return Err(TemporalRuntimeBuildError::Invalid(
                "token codebook must not be empty",
            ));
        }
        for code in &token_codebook {
            if code.token_id == NONE {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "token IDs must be live uint32 values",
                ));
            }
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
            segment_capacity,
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
            if store
                .runtime_owner_load(segment.owner)
                .is_none_or(|load| load >= config.max_segments_per_cell)
            {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "initial owner load exceeds max_segments_per_cell",
                ));
            }
            if config.context_policy == TemporalContextPolicy::BoundWinners
                && segment
                    .sources
                    .iter()
                    .copied()
                    .any(|source| source as usize >= committed.len() || !committed[source as usize])
            {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "bound-winner segment source is not a committed column cell",
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
        let mut predecessor_counts = vec![base_width; column_count];
        for &(_first_gid, column, cohort_offset) in &cohort_records {
            let cells = &column_members[column];
            let predecessor_end = predecessor_counts[column];
            let cohort_end =
                cohort_offset
                    .checked_add(race_cap)
                    .ok_or(TemporalRuntimeBuildError::Invalid(
                        "reserve cohort bounds overflow usize",
                    ))?;
            if cohort_offset != predecessor_end || cohort_end > cells.len() {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "reserve cohort order does not match its target column",
                ));
            }
            if !cells[cohort_offset..cohort_end]
                .iter()
                .copied()
                .any(|owner| {
                    store
                        .runtime_owner_load(owner)
                        .is_some_and(|load| load != 0)
                })
            {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "committed reserve cohort has no owned segment",
                ));
            }
            if cells[..predecessor_end]
                .iter()
                .copied()
                .any(|owner| store.runtime_owner_load(owner) != Some(config.max_segments_per_cell))
            {
                return Err(TemporalRuntimeBuildError::Invalid(
                    "reserve cohort was recruited before its column predecessors were exhausted",
                ));
            }
            predecessor_counts[column] = cohort_end;
        }
        let max_column_occupancy = column_offsets
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .max()
            .ok_or(TemporalRuntimeBuildError::Invalid(
                "temporal circuit must contain at least one column",
            ))?;
        let remaining_full_cohort_cells = (owner_count - column_cells.len()) / race_cap * race_cap;
        let max_prospective_column_occupancy = max_column_occupancy
            .checked_add(remaining_full_cohort_cells)
            .ok_or(TemporalRuntimeBuildError::Invalid(
                "prospective column occupancy overflows usize",
            ))?;
        let race_scratch = CoupledRaceScratch::try_new(max_prospective_column_occupancy)
            .map_err(|error| TemporalRuntimeBuildError::Invalid(error.legacy_message()))?;
        let column_drive_scratch = vec![EMPTY_DRIVE; max_prospective_column_occupancy];
        let novel_candidate_scratch = vec![EMPTY_NOVEL_CANDIDATE; max_prospective_column_occupancy];
        let active_segment_scratch = match config.plasticity_policy {
            TemporalPlasticityPolicy::GrowthOff => Vec::new(),
            TemporalPlasticityPolicy::Plastic => vec![NONE; segment_capacity],
        };
        let role_place_copy = RolePlaceCopyState::new(
            config.construction_seed,
            config.column_count,
            physical_cell_count,
            &token_codebook,
        )
        .map_err(TemporalRuntimeBuildError::Invalid)?;

        Ok(Self {
            config,
            token_codebook,
            column_offsets,
            column_cells,
            alternate_column_offsets: Vec::with_capacity(column_count + 1),
            alternate_column_cells: Vec::with_capacity(owner_count),
            cohort_columns,
            next_reserve_cell,
            threshold_offsets_mv,
            store,
            recognition_scratch: vec![EMPTY_DIAGNOSTIC; owner_count],
            active_segment_scratch,
            column_drive_scratch,
            novel_candidate_scratch,
            race_scratch,
            previous_context: [0; CONTEXT_CELLS],
            previous_context_len: 0,
            previous_winner_context: [0; CONTEXT_CELLS],
            previous_winner_context_len: 0,
            event_count: 0,
            learning_step: 0,
            scientific_stop: None,
            role_place_copy,
            generation_index: 0,
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
        self.step(token_id, TemporalStepMode::ReadOnly)
    }

    pub(crate) fn step(
        &mut self,
        token_id: u32,
        mode: TemporalStepMode,
    ) -> Result<TemporalCircuitStep, TemporalCircuitStop> {
        if let Err(message) = self.role_place_copy.preflight_arrival(token_id) {
            return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(message));
        }
        let transient = if self.role_place_copy.normalizes_temporal_context(token_id) {
            TemporalTransientState::EMPTY
        } else {
            self.transient_state()
        };
        let step = self.step_with_transient(token_id, mode, transient)?;
        let columns = self
            .token_columns(token_id)
            .expect("successful temporal step resolved one token code");
        self.role_place_copy.observe_arrival(token_id, columns);
        Ok(step)
    }

    /// Elect one token directly from the current physical winner context.
    ///
    /// Active distal segments first drive their learned owner cells through
    /// the same fitted owner/basket races used by ordinary token arrival. One
    /// winning owner signal per activated column then supplies one eleventh of
    /// the calibrated active current to every token cell containing that
    /// column. A second fitted token-cell/basket race emits the result. The
    /// codebook is static wiring; neither a decoded score argmax nor an
    /// arriving candidate token participates in the decision.
    fn elect_temporal_output(&mut self) -> Result<PhysicalOutputElection, TemporalCircuitStop> {
        if self.scientific_stop.is_some() {
            return Err(TemporalCircuitStop::Poisoned);
        }
        let transient = self.transient_state();
        if transient.previous_context_len == 0 && transient.previous_winner_context_len == 0 {
            return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                "output election requires a prior physical context",
            ));
        }
        if let Err(stop) = self.normalized_prior_sources(&transient) {
            return self.fail(stop);
        }
        if transient.previous_context_len != CONTEXT_CELLS_U8
            || transient.previous_winner_context_len != CONTEXT_CELLS_U8
        {
            return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                "output election requires one complete prior physical context",
            ));
        }
        let recognition_sources = &transient.previous_context[..ACTIVE_COLUMNS];
        let summary = match self
            .store
            .recognize_runtime_context(recognition_sources, &mut self.recognition_scratch)
        {
            Ok(summary) => summary,
            Err(status) => return self.fail(TemporalCircuitStop::StoreRecognition(status)),
        };
        let diagnostic_count = summary.owner_result_count;
        let mut output_active_columns = vec![0u8; self.config.column_count as usize];
        let reserve = PreparedReservePlan {
            next_reserve_cell: self.next_reserve_cell,
            ..EMPTY_RESERVE_PLAN
        };
        let mut activated_columns = 0u32;
        for column in 0..self.config.column_count {
            let has_active_owner = match self.column_has_active_owner(column, diagnostic_count) {
                Ok(value) => value,
                Err(stop) => return self.fail(stop),
            };
            if !has_active_owner {
                continue;
            }
            let candidate_count =
                match self.prepare_column_drives(column, diagnostic_count, reserve, &transient) {
                    Ok(count) => count,
                    Err(stop) => return self.fail(stop),
                };
            let basket_id = self.basket_offset + column;
            let candidates = &self.column_drive_scratch[..candidate_count];
            let race = match coupled_first_spike_race(
                candidates,
                BasketDrive {
                    cell_id: basket_id,
                    current_pa: TEMPORAL_PHYSICS_IDENTITY.basket_current_pa,
                    threshold_offset_mv: self.threshold_offset(basket_id),
                    inhibition_pa: TEMPORAL_PHYSICS_IDENTITY.basket_inhibition_pa,
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
            if self
                .diagnostic_for_owner(race.winner.cell_id, diagnostic_count)
                .is_some_and(|diagnostic| diagnostic.best_active_segment != NONE)
            {
                output_active_columns[column as usize] = 1;
                activated_columns += 1;
            }
        }

        self.race_output_tokens(&output_active_columns, activated_columns)
    }

    fn race_output_tokens(
        &mut self,
        output_active_columns: &[u8],
        activated_columns: u32,
    ) -> Result<PhysicalOutputElection, TemporalCircuitStop> {
        let synaptic_share_pa = TEMPORAL_PHYSICS_IDENTITY.active_pa / ACTIVE_COLUMNS as f64;
        let output_offset = self
            .proximal_relay_offset
            .checked_add(self.config.column_count)
            .ok_or(TemporalCircuitStop::StoreOrPlanInvariant(
                "output physical offsets overflow u32",
            ))?;
        let output_cell_count = u32::try_from(self.token_codebook.len()).map_err(|_| {
            TemporalCircuitStop::StoreOrPlanInvariant("token output population exceeds uint32")
        })?;
        let output_basket_id = output_offset.checked_add(output_cell_count).ok_or(
            TemporalCircuitStop::StoreOrPlanInvariant("output physical offsets overflow u32"),
        )?;
        let mut output_drive_scratch = vec![EMPTY_DRIVE; self.token_codebook.len()];
        let mut candidate_tokens = 0usize;
        for (token_index, code) in self.token_codebook.iter().enumerate() {
            let input_columns = code
                .columns
                .iter()
                .filter(|&&column| output_active_columns[column as usize] != 0)
                .count();
            if input_columns == 0 {
                continue;
            }
            output_drive_scratch[candidate_tokens] = CandidateDrive {
                cell_id: output_offset + token_index as u32,
                current_pa: synaptic_share_pa * input_columns as f64,
                threshold_offset_mv: 0.0,
            };
            candidate_tokens += 1;
        }
        let mut output_race_scratch = match CoupledRaceScratch::try_new(self.token_codebook.len()) {
            Ok(scratch) => scratch,
            Err(_) => {
                return self.fail(TemporalCircuitStop::PreallocatedCapacityInvariant(
                    "output race scratch allocation failed",
                ));
            }
        };
        let race = match coupled_first_spike_race(
            &output_drive_scratch[..candidate_tokens],
            BasketDrive {
                cell_id: output_basket_id,
                current_pa: TEMPORAL_PHYSICS_IDENTITY.basket_current_pa,
                threshold_offset_mv: 0.0,
                inhibition_pa: TEMPORAL_PHYSICS_IDENTITY.basket_inhibition_pa,
            },
            &mut output_race_scratch,
        ) {
            Ok(race) => race,
            Err(source) => {
                return self.fail(TemporalCircuitStop::Physics {
                    stage: PhysicalStage::ColumnRace,
                    location: output_basket_id,
                    source,
                });
            }
        };
        let winner_index = (race.winner.cell_id - output_offset) as usize;
        let winner = self.token_codebook.get(winner_index).ok_or(
            TemporalCircuitStop::StoreOrPlanInvariant(
                "output race emitted a cell outside the token population",
            ),
        )?;
        let winner_input_columns = winner
            .columns
            .iter()
            .filter(|&&column| output_active_columns[column as usize] != 0)
            .count() as u8;
        Ok(PhysicalOutputElection {
            token_id: winner.token_id,
            output_cell_id: race.winner.cell_id,
            activated_columns,
            candidate_tokens: candidate_tokens as u32,
            winner_input_columns,
            winner_tick: race.winner.tick,
            basket_tick: race.basket_tick,
            inhibition_onset_tick: race.inhibition_onset_tick,
            simulated_ticks: race.simulated_ticks,
        })
    }

    fn elect_copy_output_at(
        &mut self,
        slot: CopySlot,
        position: u8,
    ) -> Result<PhysicalOutputElection, TemporalCircuitStop> {
        let columns = match self.role_place_copy.binding_columns_at(slot, position) {
            Ok(columns) => columns,
            Err(message) => {
                return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(message));
            }
        };
        let mut active_columns = vec![0u8; self.config.column_count as usize];
        for column in columns {
            let copy_cell = match self.role_place_copy.copy_column_cell(column) {
                Ok(cell) => cell,
                Err(message) => {
                    return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(message));
                }
            };
            if let Err(source) = singleton_latency_ticks(TEMPORAL_PHYSICS_IDENTITY.active_pa, 0.0) {
                return self.fail(TemporalCircuitStop::Physics {
                    stage: PhysicalStage::ColumnRace,
                    location: copy_cell,
                    source,
                });
            }
            active_columns[column as usize] = 1;
        }
        self.race_output_tokens(&active_columns, ACTIVE_COLUMNS as u32)
    }

    /// Elect and commit exactly one surface token through the currently active
    /// native route. Internal COPY controls and COPY_END never escape this
    /// interface.
    pub(crate) fn generate(&mut self) -> Result<TemporalGeneration, TemporalCircuitStop> {
        if self.scientific_stop.is_some() {
            return Err(TemporalCircuitStop::Poisoned);
        }
        let Some(next_generation_index) = self.generation_index.checked_add(1) else {
            return self.fail(TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::EventCounter,
                location: 0,
            });
        };
        if self.role_place_copy.is_recalling() {
            self.generate_from_copy(next_generation_index)
        } else {
            self.generate_from_temporal(next_generation_index)
        }
    }

    fn generate_from_temporal(
        &mut self,
        next_generation_index: u64,
    ) -> Result<TemporalGeneration, TemporalCircuitStop> {
        let elected = self.elect_temporal_output()?;
        if let Some(slot) = CopySlot::from_copy_token(elected.token_id) {
            if !self.role_place_copy.is_ready(slot) {
                return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                    "COPY control won without a captured value in its slot",
                ));
            }
            let copied = self.elect_copy_output_at(slot, 0)?;
            if copied.token_id == COPY_END_TOKEN_ID {
                return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                    "captured slot produced COPY_END before one surface token",
                ));
            }
            if is_internal_role_token(copied.token_id) {
                return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                    "copy binding emitted an internal role token",
                ));
            }
            let copied = match offset_election(copied, elected.simulated_ticks) {
                Ok(value) => value,
                Err(stop) => return self.fail(stop),
            };
            let feedback = self.step_with_transient(
                slot.copy_token_id(),
                TemporalStepMode::ReadOnly,
                self.transient_state(),
            )?;
            self.role_place_copy.begin_recall(slot);
            self.role_place_copy.advance_recall();
            self.generation_index = next_generation_index;
            return Ok(generation_record(
                copied,
                Some(elected),
                TemporalGenerationPath::EnterCopy,
                Some(&feedback),
                Some(slot),
                0,
                next_generation_index,
                self.event_count,
            ));
        }
        if is_internal_role_token(elected.token_id) {
            return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                "temporal generation elected an invalid role token",
            ));
        }
        let feedback = self.step_with_transient(
            elected.token_id,
            TemporalStepMode::ReadOnly,
            self.transient_state(),
        )?;
        self.generation_index = next_generation_index;
        Ok(generation_record(
            elected,
            None,
            TemporalGenerationPath::Temporal,
            Some(&feedback),
            None,
            u8::MAX,
            next_generation_index,
            self.event_count,
        ))
    }

    fn generate_from_copy(
        &mut self,
        next_generation_index: u64,
    ) -> Result<TemporalGeneration, TemporalCircuitStop> {
        let Some(slot) = self.role_place_copy.recall_slot() else {
            return self.fail(TemporalCircuitStop::StoreOrPlanInvariant(
                "copy generation lost its active slot",
            ));
        };
        let place = self.role_place_copy.recall_position();
        let copied = self.elect_copy_output_at(slot, place)?;
        let at_end = self.role_place_copy.recall_is_at_end();
        if copied.token_id == COPY_END_TOKEN_ID {
            if !at_end {
                return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                    "COPY_END won before the active slot place chain was exhausted",
                ));
            }
            let resumed = self.elect_temporal_output()?;
            if is_internal_role_token(resumed.token_id) {
                return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                    "adjacent internal generation controls are not valid in role/place copy",
                ));
            }
            let resumed = match offset_election(resumed, copied.simulated_ticks) {
                Ok(value) => value,
                Err(stop) => return self.fail(stop),
            };
            let feedback = self.step_with_transient(
                resumed.token_id,
                TemporalStepMode::ReadOnly,
                self.transient_state(),
            )?;
            self.role_place_copy.finish_recall();
            self.generation_index = next_generation_index;
            return Ok(generation_record(
                resumed,
                Some(copied),
                TemporalGenerationPath::ExitCopy,
                Some(&feedback),
                Some(slot),
                u8::MAX,
                next_generation_index,
                self.event_count,
            ));
        }
        if at_end || is_internal_role_token(copied.token_id) {
            return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                "copy binding failed to terminate at its exact slot length",
            ));
        }
        self.role_place_copy.advance_recall();
        self.generation_index = next_generation_index;
        Ok(generation_record(
            copied,
            None,
            TemporalGenerationPath::Copy,
            None,
            Some(slot),
            place,
            next_generation_index,
            self.event_count,
        ))
    }

    /// Execute one ordinary transaction while treating the transient state as
    /// empty until the transaction's existing no-fail commit point.
    ///
    /// This is deliberately not implemented as `reset_transient()` followed by
    /// rollback. A failing first token therefore cannot publish an empty
    /// context or reset event counter; only the terminal poison latch may
    /// change. The ABI child module uses this for reset-before block entry.
    fn step_after_transient_reset(
        &mut self,
        token_id: u32,
        mode: TemporalStepMode,
    ) -> Result<TemporalCircuitStep, TemporalCircuitStop> {
        if self.role_place_copy.is_end_boundary(token_id) {
            return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                "copy-slot end cannot begin a reset episode",
            ));
        }
        let step = self.step_with_transient(token_id, mode, TemporalTransientState::EMPTY)?;
        let columns = self
            .token_columns(token_id)
            .expect("successful temporal step resolved one token code");
        self.role_place_copy.reset();
        self.generation_index = 0;
        self.role_place_copy.observe_arrival(token_id, columns);
        Ok(step)
    }

    fn transient_state(&self) -> TemporalTransientState {
        TemporalTransientState {
            previous_context: self.previous_context,
            previous_context_len: self.previous_context_len,
            previous_winner_context: self.previous_winner_context,
            previous_winner_context_len: self.previous_winner_context_len,
            event_count: self.event_count,
        }
    }

    fn step_with_transient(
        &mut self,
        token_id: u32,
        mode: TemporalStepMode,
        transient: TemporalTransientState,
    ) -> Result<TemporalCircuitStep, TemporalCircuitStop> {
        if self.scientific_stop.is_some() {
            return Err(TemporalCircuitStop::Poisoned);
        }
        if token_id == NONE {
            return Err(TemporalCircuitStop::UnknownToken(token_id));
        }
        let columns = match self
            .token_codebook
            .binary_search_by_key(&token_id, |code| code.token_id)
        {
            Ok(index) => self.token_codebook[index].columns,
            Err(_) => return Err(TemporalCircuitStop::UnknownToken(token_id)),
        };
        if let Err(stop) = self.normalized_prior_sources(&transient) {
            return self.fail(stop);
        }

        let recognition_source_count = match (
            transient.previous_context_len,
            transient.previous_winner_context_len,
        ) {
            (0, 0) => 0,
            (CONTEXT_CELLS_U8, CONTEXT_CELLS_U8) => ACTIVE_COLUMNS,
            _ => {
                return self.fail(TemporalCircuitStop::MalformedContextOrPolicy(
                    "prior contexts must both be empty or exactly 22 cells",
                ));
            }
        };
        let recognition_sources = &transient.previous_context[..recognition_source_count];
        let summary = match self
            .store
            .recognize_runtime_context(recognition_sources, &mut self.recognition_scratch)
        {
            Ok(summary) => summary,
            Err(status) => return self.fail(TemporalCircuitStop::StoreRecognition(status)),
        };
        let diagnostic_count = summary.owner_result_count;
        let active_segment_count = if mode == TemporalStepMode::Learn
            && self.config.plasticity_policy == TemporalPlasticityPolicy::Plastic
        {
            match self
                .store
                .runtime_active_segments_into(&mut self.active_segment_scratch)
            {
                Ok(count) => count,
                Err(error) => {
                    return self.fail(map_store_preflight_error(
                        error,
                        &self.store,
                        0,
                        self.config.max_segments_per_cell,
                    ));
                }
            }
        } else {
            0
        };
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

        let reserve_plan = if mode == TemporalStepMode::Learn
            && self.config.plasticity_policy == TemporalPlasticityPolicy::Plastic
            && transient.previous_context_len != 0
        {
            match self.preflight_reserve_plan(&columns, diagnostic_count) {
                Ok(plan) => plan,
                Err(stop) => return self.fail(stop),
            }
        } else {
            PreparedReservePlan {
                next_reserve_cell: self.next_reserve_cell,
                ..EMPTY_RESERVE_PLAN
            }
        };

        let mut winner_spikes = [EMPTY_SPIKE; ACTIVE_COLUMNS];
        let mut basket_spikes = [EMPTY_SPIKE; ACTIVE_COLUMNS];
        let mut inhibition_onset_ticks = [0u16; ACTIVE_COLUMNS];
        let mut column_simulated_ticks = [0u16; ACTIVE_COLUMNS];
        let mut winner_best_active = [NONE; ACTIVE_COLUMNS];
        let mut predicted_columns = 0u8;
        for (index, &column) in columns.iter().enumerate() {
            let candidate_count = match self.prepare_column_drives(
                column,
                diagnostic_count,
                reserve_plan,
                &transient,
            ) {
                Ok(candidate_count) => candidate_count,
                Err(stop) => return self.fail(stop),
            };
            let basket_id = self.basket_offset + column;
            let candidates = &self.column_drive_scratch[..candidate_count];
            let race = match coupled_first_spike_race(
                candidates,
                BasketDrive {
                    cell_id: basket_id,
                    current_pa: TEMPORAL_PHYSICS_IDENTITY.basket_current_pa,
                    threshold_offset_mv: self.threshold_offset(basket_id),
                    inhibition_pa: TEMPORAL_PHYSICS_IDENTITY.basket_inhibition_pa,
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
            if let Some(diagnostic) =
                self.diagnostic_for_owner(race.winner.cell_id, diagnostic_count)
            {
                winner_best_active[index] = diagnostic.best_active_segment;
                if diagnostic.best_active_segment != NONE {
                    predicted_columns += 1;
                }
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
        let Some(event_count) = transient.event_count.checked_add(1) else {
            return self.fail(TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::EventCounter,
                location: 0,
            });
        };

        let prepared_learning = if mode == TemporalStepMode::Learn
            && self.config.plasticity_policy == TemporalPlasticityPolicy::Plastic
        {
            match self.preflight_learning(
                active_segment_count,
                winner_spikes.map(|spike| spike.cell_id),
                winner_best_active,
                reserve_plan,
                &transient,
            ) {
                Ok(prepared) => Some(prepared),
                Err(stop) => return self.fail(stop),
            }
        } else {
            None
        };
        let allocated_segments = prepared_learning.map_or(0, |prepared| prepared.append_count);
        let punished_segments = prepared_learning.map_or(0, |prepared| prepared.punished_count);

        // All physical work and every fallible mutation operation above used
        // staged values. The mutation commit is allocation-free and no-fail.
        if let Some(prepared) = prepared_learning {
            self.commit_prevalidated_learning(prepared);
        }
        self.previous_context = context;
        self.previous_context_len = CONTEXT_CELLS as u8;
        self.previous_winner_context = winner_context;
        self.previous_winner_context_len = CONTEXT_CELLS as u8;
        self.event_count = event_count;

        Ok(TemporalCircuitStep {
            token_id,
            predicted: predicted_columns >= self.config.prediction_quorum,
            predicted_columns,
            allocated_segments,
            punished_segments,
            candidate_segments: summary.candidate_segments,
            matching_segments: summary.matching_segments,
            active_segments: summary.active_segments,
            postings_visited: summary.postings_visited,
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
        self.role_place_copy.reset();
        self.generation_index = 0;
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

    fn token_columns(&self, token_id: u32) -> Option<[u32; ACTIVE_COLUMNS]> {
        self.token_codebook
            .binary_search_by_key(&token_id, |code| code.token_id)
            .ok()
            .map(|index| self.token_codebook[index].columns)
    }

    fn owner_load(&self, owner: u32) -> Result<u8, TemporalCircuitStop> {
        self.store
            .runtime_owner_load(owner)
            .ok_or(TemporalCircuitStop::StoreOrPlanInvariant(
                "column owner is outside the temporal store",
            ))
    }

    fn prospective_owner_load(
        &self,
        owner: u32,
        reserve: PreparedReservePlan,
    ) -> Result<u8, TemporalCircuitStop> {
        if reserve.recruits()
            && owner >= self.next_reserve_cell
            && owner < reserve.next_reserve_cell
        {
            return Ok(0);
        }
        self.owner_load(owner)
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

    fn column_has_active_owner(
        &self,
        column: u32,
        diagnostic_count: usize,
    ) -> Result<bool, TemporalCircuitStop> {
        let column_index = column as usize;
        let Some((&cell_start, &cell_end)) = self
            .column_offsets
            .get(column_index)
            .zip(self.column_offsets.get(column_index + 1))
        else {
            return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                "output election references a missing column",
            ));
        };
        if cell_start > cell_end || cell_end > self.column_cells.len() {
            return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                "column CSR bounds are malformed",
            ));
        }
        Ok(self.column_cells[cell_start..cell_end]
            .iter()
            .any(|&owner| {
                self.diagnostic_for_owner(owner, diagnostic_count)
                    .is_some_and(|diagnostic| diagnostic.best_active_segment != NONE)
            }))
    }

    fn prepare_column_drives(
        &mut self,
        column: u32,
        diagnostic_count: usize,
        reserve: PreparedReservePlan,
        transient: &TemporalTransientState,
    ) -> Result<usize, TemporalCircuitStop> {
        if diagnostic_count > self.recognition_scratch.len() {
            return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
                "recognition output exceeds its construction-time scratch",
            ));
        }
        let column_index = column as usize;
        let (column_offsets, column_cells) = if reserve.recruits() {
            (&self.alternate_column_offsets, &self.alternate_column_cells)
        } else {
            (&self.column_offsets, &self.column_cells)
        };
        let Some((&cell_start, &cell_end)) = column_offsets
            .get(column_index)
            .zip(column_offsets.get(column_index + 1))
        else {
            return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                "token code references a missing column",
            ));
        };
        if cell_start > cell_end || cell_end > column_cells.len() {
            return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                "column CSR bounds are malformed",
            ));
        }
        let cells = &column_cells[cell_start..cell_end];
        if cells.len() > self.column_drive_scratch.len()
            || cells.len() > self.novel_candidate_scratch.len()
        {
            return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
                "column occupancy exceeds construction-time candidate scratch",
            ));
        }

        let mut active_candidates = [EMPTY_CATEGORIZED_RANK_CANDIDATE; 8];
        let mut active_count = 0usize;
        for (offset, &cell) in cells.iter().enumerate() {
            self.column_drive_scratch[offset] = CandidateDrive {
                cell_id: cell,
                current_pa: 0.0,
                threshold_offset_mv: self.threshold_offset(cell),
            };
            let Some(diagnostic) = self.diagnostic_for_owner(cell, diagnostic_count) else {
                continue;
            };
            if diagnostic.best_active_segment != NONE {
                if active_count < active_candidates.len() {
                    active_candidates[active_count] = CategorizedRankCandidate {
                        owner: cell,
                        column_offset: offset,
                        primary_contacts: diagnostic.active_connected_contacts,
                        secondary_contacts: diagnostic.active_potential_contacts,
                        segment_id: diagnostic.best_active_segment,
                    };
                }
                active_count += 1;
            }
        }
        let categorized_count = active_count;
        if categorized_count > usize::from(self.config.novel_race_cap) {
            return Err(TemporalCircuitStop::CategorizedRaceCapacity {
                column,
                categorized_count: categorized_count as u8,
                race_cap: self.config.novel_race_cap,
            });
        }
        let novel_limit = usize::from(self.config.novel_race_cap) - categorized_count;

        // Only recall-competent active owners may bias the physical race.
        // Potential-only matches remain recognition diagnostics; allowing
        // them to own W_t would substitute a near context for the current
        // identity and suppress the novel-cell allocation path.
        active_candidates[..active_count].sort_unstable_by(|left, right| {
            right
                .primary_contacts
                .cmp(&left.primary_contacts)
                .then_with(|| right.secondary_contacts.cmp(&left.secondary_contacts))
                .then_with(|| left.segment_id.cmp(&right.segment_id))
                .then_with(|| left.owner.cmp(&right.owner))
        });
        for (rank, candidate) in active_candidates[..active_count].iter().enumerate() {
            self.column_drive_scratch[candidate.column_offset].current_pa = CandidateClass::Active {
                load: self.prospective_owner_load(candidate.owner, reserve)?,
            }
            .current_pa()
                - TEMPORAL_PHYSICS_IDENTITY.novel_rank_decrement_pa * rank as f64;
        }
        if transient.previous_context_len == 0 {
            for rank in 0..usize::from(self.config.base_cells_per_column) {
                let Some(&cell_id) = cells.get(rank) else {
                    return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                        "column lost its validated base-cell prefix",
                    ));
                };
                self.column_drive_scratch[rank].current_pa = CandidateClass::Novel {
                    rank: rank as u8,
                    load: self.prospective_owner_load(cell_id, reserve)?,
                }
                .current_pa();
            }
        } else {
            let mut novel_candidate_count = 0usize;
            for (column_offset, &cell_id) in cells.iter().enumerate() {
                if self.column_drive_scratch[column_offset].current_pa > 0.0 {
                    continue;
                }
                self.novel_candidate_scratch[novel_candidate_count] = NovelCandidate {
                    cell_id,
                    column_offset,
                    load: self.prospective_owner_load(cell_id, reserve)?,
                    afferent_score: afferent_score(
                        self.config.construction_seed,
                        &transient.previous_context[..usize::from(transient.previous_context_len)],
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

        // Keep every materialized cohort cell in the physical race, including
        // exact zero-drive cells. This preserves the Python oracle's simulated
        // work and STOP surface; a later sparse elision needs its own parity
        // proof before changing this boundary.
        Ok(cells.len())
    }

    fn preflight_reserve_plan(
        &mut self,
        columns: &[u32; ACTIVE_COLUMNS],
        diagnostic_count: usize,
    ) -> Result<PreparedReservePlan, TemporalCircuitStop> {
        if diagnostic_count > self.recognition_scratch.len() {
            return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
                "recognition output exceeds its construction-time scratch",
            ));
        }
        let column_count = usize::try_from(self.config.column_count).map_err(|_| {
            TemporalCircuitStop::PreallocatedCapacityInvariant(
                "column count exceeds the runtime usize domain",
            )
        })?;
        let expected_offset_count = column_count.checked_add(1).ok_or(
            TemporalCircuitStop::PreallocatedCapacityInvariant(
                "column-offset shape overflows usize",
            ),
        )?;
        if self.column_offsets.len() != expected_offset_count
            || self.column_offsets.first().copied() != Some(0)
            || self.column_offsets.last().copied() != Some(self.column_cells.len())
            || self.column_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || self.column_offsets.capacity() < expected_offset_count
            || self.alternate_column_offsets.capacity() < expected_offset_count
            || self.column_cells.capacity() < self.config.max_pyramidal_cells as usize
            || self.alternate_column_cells.capacity() < self.config.max_pyramidal_cells as usize
            || u32::try_from(self.column_cells.len()).ok() != Some(self.next_reserve_cell)
        {
            return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
                "column CSR backing violates its construction-time contract",
            ));
        }

        let mut plan = PreparedReservePlan {
            next_reserve_cell: self.next_reserve_cell,
            ..EMPTY_RESERVE_PLAN
        };
        for &column in columns {
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
            let has_recall_competent_owner = cells.iter().copied().any(|cell| {
                self.diagnostic_for_owner(cell, diagnostic_count)
                    .is_some_and(|diagnostic| diagnostic.best_active_segment != NONE)
            });
            if has_recall_competent_owner {
                continue;
            }
            let mut exhausted = true;
            for &cell in cells {
                if self.owner_load(cell)? < self.config.max_segments_per_cell {
                    exhausted = false;
                    break;
                }
            }
            if exhausted {
                let plan_index = plan.cohort_count();
                if plan_index >= plan.cohort_columns.len() {
                    return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
                        "one token requested more reserve cohorts than fixed plan scratch",
                    ));
                }
                plan.cohort_columns[plan_index] = column;
                plan.cohort_count += 1;
            }
        }
        if !plan.recruits() {
            return Ok(plan);
        }

        let required_cells = u64::from(plan.cohort_count)
            .checked_mul(u64::from(self.config.novel_race_cap))
            .ok_or(TemporalCircuitStop::ReserveCellCapacity {
                column: plan.cohort_columns[0],
            })?;
        let prospective_next = u64::from(self.next_reserve_cell)
            .checked_add(required_cells)
            .filter(|next| *next <= u64::from(self.config.max_pyramidal_cells))
            .ok_or(TemporalCircuitStop::ReserveCellCapacity {
                column: plan.cohort_columns[0],
            })?;
        plan.next_reserve_cell = prospective_next as u32;

        let prospective_cohort_count = self
            .cohort_columns
            .len()
            .checked_add(plan.cohort_count())
            .ok_or(
            TemporalCircuitStop::PreallocatedCapacityInvariant("cohort log length overflows usize"),
        )?;
        let required_cells_usize = usize::try_from(required_cells).map_err(|_| {
            TemporalCircuitStop::PreallocatedCapacityInvariant(
                "prospective reserve cells exceed the runtime usize domain",
            )
        })?;
        let prospective_cell_count = self
            .column_cells
            .len()
            .checked_add(required_cells_usize)
            .ok_or(TemporalCircuitStop::PreallocatedCapacityInvariant(
                "prospective column CSR length overflows usize",
            ))?;
        if prospective_cohort_count > self.cohort_columns.capacity()
            || prospective_cell_count > self.alternate_column_cells.capacity()
        {
            return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
                "reserve plan exceeds construction-time CSR or cohort-log capacity",
            ));
        }

        self.alternate_column_offsets.clear();
        self.alternate_column_cells.clear();
        self.alternate_column_offsets.push(0);
        let mut planned_index = 0usize;
        for column_index in 0..column_count {
            let cell_start = self.column_offsets[column_index];
            let cell_end = self.column_offsets[column_index + 1];
            if cell_start > cell_end || cell_end > self.column_cells.len() {
                return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                    "column CSR bounds are malformed",
                ));
            }
            self.alternate_column_cells
                .extend_from_slice(&self.column_cells[cell_start..cell_end]);
            if planned_index < plan.cohort_count()
                && plan.cohort_columns[planned_index] as usize == column_index
            {
                let cohort_start = u64::from(self.next_reserve_cell)
                    .checked_add(
                        (planned_index as u64)
                            .checked_mul(u64::from(self.config.novel_race_cap))
                            .ok_or(TemporalCircuitStop::StoreOrPlanInvariant(
                                "prospective cohort address overflows u64",
                            ))?,
                    )
                    .ok_or(TemporalCircuitStop::StoreOrPlanInvariant(
                        "prospective cohort address overflows u64",
                    ))?;
                for offset in 0..u32::from(self.config.novel_race_cap) {
                    let cell = u32::try_from(cohort_start + u64::from(offset)).map_err(|_| {
                        TemporalCircuitStop::StoreOrPlanInvariant(
                            "prospective cohort identity exceeds uint32",
                        )
                    })?;
                    self.alternate_column_cells.push(cell);
                }
                planned_index += 1;
            }
            self.alternate_column_offsets
                .push(self.alternate_column_cells.len());
        }
        if planned_index != plan.cohort_count()
            || self.alternate_column_offsets.len() != expected_offset_count
            || self.alternate_column_cells.len() != prospective_cell_count
        {
            return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                "prospective reserve CSR does not match the aggregate plan",
            ));
        }
        Ok(plan)
    }

    fn normalized_prior_sources(
        &self,
        transient: &TemporalTransientState,
    ) -> Result<([u32; ACTIVE_COLUMNS], usize), TemporalCircuitStop> {
        match (
            transient.previous_context_len,
            transient.previous_winner_context_len,
        ) {
            (0, 0) => return Ok(([0; ACTIVE_COLUMNS], 0)),
            (CONTEXT_CELLS_U8, CONTEXT_CELLS_U8) => {}
            _ => {
                return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                    "prior contexts must both be empty or exactly 22 cells",
                ));
            }
        }
        let mut normalized = [0u32; ACTIVE_COLUMNS];
        for (index, normalized_source) in normalized.iter_mut().enumerate() {
            let source = transient.previous_context[index];
            let expected_relay = match self.config.context_policy {
                TemporalContextPolicy::BoundWinners => {
                    if source >= self.config.max_pyramidal_cells {
                        return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                            "bound-winner context source is outside the pyramidal domain",
                        ));
                    }
                    *normalized_source = source;
                    self.relay_offset.checked_add(source)
                }
                TemporalContextPolicy::GenericAfferents => {
                    let Some(logical) = source.checked_sub(self.proximal_offset) else {
                        return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                            "generic context source is below the proximal domain",
                        ));
                    };
                    if logical >= self.config.column_count {
                        return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                            "generic context source is outside the proximal domain",
                        ));
                    }
                    *normalized_source = logical;
                    self.proximal_relay_offset.checked_add(logical)
                }
            }
            .ok_or(TemporalCircuitStop::MalformedContextOrPolicy(
                "context relay identity overflows u32",
            ))?;
            if transient.previous_context[ACTIVE_COLUMNS + index] != expected_relay {
                return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                    "context does not contain injective source/relay pairs",
                ));
            }

            let winner = transient.previous_winner_context[index];
            let winner_relay = self.relay_offset.checked_add(winner).ok_or(
                TemporalCircuitStop::MalformedContextOrPolicy(
                    "winner relay identity overflows u32",
                ),
            )?;
            if winner >= self.config.max_pyramidal_cells
                || transient.previous_winner_context[ACTIVE_COLUMNS + index] != winner_relay
            {
                return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                    "winner context does not contain injective source/relay pairs",
                ));
            }
        }
        if self.config.context_policy == TemporalContextPolicy::BoundWinners
            && transient.previous_context != transient.previous_winner_context
        {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "bound-winner contexts must be identical",
            ));
        }
        let mut sorted = normalized;
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "context sources must be unique",
            ));
        }
        Ok((normalized, ACTIVE_COLUMNS))
    }

    fn preflight_learning(
        &mut self,
        active_segment_count: usize,
        winner_cells: [u32; ACTIVE_COLUMNS],
        winner_best_active: [u32; ACTIVE_COLUMNS],
        reserve: PreparedReservePlan,
        transient: &TemporalTransientState,
    ) -> Result<PreparedLearnStep, TemporalCircuitStop> {
        if active_segment_count > self.active_segment_scratch.len() {
            return Err(TemporalCircuitStop::PreallocatedCapacityInvariant(
                "active-segment materialization exceeds construction scratch",
            ));
        }
        let (normalized_context_sources, context_source_count) =
            self.normalized_prior_sources(transient)?;
        let mut reinforced_rows = [EMPTY_SEGMENT_ROW; ACTIVE_COLUMNS];
        let mut append_owners = [NONE; ACTIVE_COLUMNS];
        let mut append_count = 0usize;
        let mut punished_count = 0u32;
        if context_source_count != 0 {
            for index in 0..ACTIVE_COLUMNS {
                if winner_best_active[index] == NONE {
                    append_owners[append_count] = winner_cells[index];
                    append_count += 1;
                }
            }
        }

        let append_preflight = self
            .store
            .preflight_runtime_mutation_backing(
                &append_owners[..append_count],
                self.config.max_segments_per_cell,
            )
            .map_err(|error| {
                map_store_preflight_error(
                    error,
                    &self.store,
                    append_count as u32,
                    self.config.max_segments_per_cell,
                )
            })?;
        if append_preflight.prospective_append_count() != append_count as u32
            || append_preflight.first_append_segment_id() != self.store.runtime_segment_count()
        {
            return Err(TemporalCircuitStop::StoreOrPlanInvariant(
                "store mutation preflight changed append identity",
            ));
        }

        let mut initial_row = TemporalSegmentRow {
            normalized_sources: [NONE; PAIRS_PER_SEGMENT],
            permanence_ticks: [0; PAIRS_PER_SEGMENT],
            live_pair_count: 0,
        };
        if append_count != 0 {
            let mut sample_scratch = [0u64; ACTIVE_COLUMNS];
            let Some(selected) = sample_without_replacement(
                ACTIVE_COLUMNS,
                PAIRS_PER_SEGMENT,
                u64::from(self.config.construction_seed) ^ INITIAL_CONTACT_DOMAIN,
                self.learning_step,
                &mut sample_scratch,
            ) else {
                return Err(TemporalCircuitStop::PhiloxAddressExhaustion {
                    operation: "initial-allocation",
                    segment_id: NONE,
                });
            };
            for (destination, &position) in selected.iter().enumerate() {
                initial_row.normalized_sources[destination] =
                    normalized_context_sources[position as usize];
                initial_row.permanence_ticks[destination] = self.config.initial_permanence_ticks;
            }
            initial_row.normalized_sources.sort_unstable();
            initial_row.live_pair_count = PAIRS_PER_SEGMENT as u8;
            for owner in append_owners[..append_count].iter().copied() {
                self.store
                    .preflight_runtime_canonical_append(owner, &initial_row)
                    .map_err(|error| {
                        map_store_preflight_error(
                            error,
                            &self.store,
                            0,
                            self.config.max_segments_per_cell,
                        )
                    })?;
            }
        }

        for &segment_id in &self.active_segment_scratch[..active_segment_count] {
            let row = if let Some(index) = winner_best_active
                .iter()
                .position(|candidate| *candidate == segment_id)
            {
                let row = build_reinforced_row(
                    &self.store,
                    self.config,
                    segment_id,
                    &normalized_context_sources,
                    context_source_count,
                    self.learning_step,
                )?;
                reinforced_rows[index] = row;
                row
            } else {
                let owner = self.store.runtime_segment_owner(segment_id).ok_or(
                    TemporalCircuitStop::StoreOrPlanInvariant(
                        "active segment has no authoritative owner",
                    ),
                )?;
                if winner_cells.contains(&owner) {
                    continue;
                }
                punished_count = punished_count.checked_add(1).ok_or(
                    TemporalCircuitStop::StoreOrPlanInvariant(
                        "punishment count overflowed its canonical domain",
                    ),
                )?;
                build_punished_row(
                    &self.store,
                    self.config,
                    segment_id,
                    &normalized_context_sources,
                    context_source_count,
                )?
            };
            self.store
                .preflight_runtime_canonical_rewrite(segment_id, &row)
                .map_err(|error| {
                    map_store_preflight_error(
                        error,
                        &self.store,
                        0,
                        self.config.max_segments_per_cell,
                    )
                })?;
        }
        let next_learning_step = self.learning_step.checked_add(1).ok_or(
            TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::LearningCounter,
                location: 0,
            },
        )?;
        Ok(PreparedLearnStep {
            reserve,
            active_segment_count,
            winner_cells,
            winner_best_active,
            normalized_context_sources,
            reinforced_rows,
            append_owners,
            append_count: append_count as u8,
            punished_count,
            initial_row,
            next_learning_step,
        })
    }

    fn commit_prevalidated_learning(&mut self, prepared: PreparedLearnStep) {
        debug_assert!(prepared.active_segment_count <= self.active_segment_scratch.len());
        debug_assert!(usize::from(prepared.append_count) <= ACTIVE_COLUMNS);
        if prepared.reserve.recruits() {
            debug_assert_eq!(self.next_reserve_cell, self.column_cells.len() as u32);
            debug_assert!(
                self.cohort_columns.len() + prepared.reserve.cohort_count()
                    <= self.cohort_columns.capacity()
            );
            debug_assert_eq!(
                self.alternate_column_cells.len() as u32,
                prepared.reserve.next_reserve_cell
            );
            std::mem::swap(&mut self.column_offsets, &mut self.alternate_column_offsets);
            std::mem::swap(&mut self.column_cells, &mut self.alternate_column_cells);
            self.next_reserve_cell = prepared.reserve.next_reserve_cell;
            append_prevalidated_slice(
                &mut self.cohort_columns,
                &prepared.reserve.cohort_columns[..prepared.reserve.cohort_count()],
            );
        }
        for active_index in 0..prepared.active_segment_count {
            let segment_id = self.active_segment_scratch[active_index];
            let mut reinforced = false;
            for index in 0..ACTIVE_COLUMNS {
                if prepared.winner_best_active[index] == segment_id {
                    self.store.commit_prevalidated_runtime_canonical_rewrite(
                        segment_id,
                        &prepared.reinforced_rows[index],
                    );
                    reinforced = true;
                    break;
                }
            }
            if reinforced {
                continue;
            }
            let owner = self
                .store
                .commit_prevalidated_runtime_segment_owner(segment_id);
            if prepared.winner_cells.contains(&owner) {
                continue;
            }
            let old = self
                .store
                .commit_prevalidated_runtime_segment_row(segment_id);
            let row = build_prevalidated_punished_row(
                old,
                self.config,
                &prepared.normalized_context_sources,
            );
            self.store
                .commit_prevalidated_runtime_canonical_rewrite(segment_id, &row);
        }
        for owner in prepared.append_owners[..usize::from(prepared.append_count)]
            .iter()
            .copied()
        {
            self.store
                .commit_prevalidated_runtime_canonical_append(owner, &prepared.initial_row);
        }
        self.learning_step = prepared.next_learning_step;
    }
}

fn offset_election(
    mut election: PhysicalOutputElection,
    control_simulated_end: u16,
) -> Result<PhysicalOutputElection, TemporalCircuitStop> {
    let shift =
        control_simulated_end
            .checked_add(1)
            .ok_or(TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::EventCounter,
                location: election.output_cell_id,
            })?;
    election.winner_tick = election.winner_tick.checked_add(shift).ok_or(
        TemporalCircuitStop::TickOrCounterOverflow {
            stage: TemporalOverflowStage::EventCounter,
            location: election.output_cell_id,
        },
    )?;
    election.basket_tick = election.basket_tick.checked_add(shift).ok_or(
        TemporalCircuitStop::TickOrCounterOverflow {
            stage: TemporalOverflowStage::EventCounter,
            location: election.output_cell_id,
        },
    )?;
    election.inhibition_onset_tick = election.inhibition_onset_tick.checked_add(shift).ok_or(
        TemporalCircuitStop::TickOrCounterOverflow {
            stage: TemporalOverflowStage::EventCounter,
            location: election.output_cell_id,
        },
    )?;
    election.simulated_ticks = election.simulated_ticks.checked_add(shift).ok_or(
        TemporalCircuitStop::TickOrCounterOverflow {
            stage: TemporalOverflowStage::EventCounter,
            location: election.output_cell_id,
        },
    )?;
    Ok(election)
}

fn generation_record(
    surface: PhysicalOutputElection,
    control: Option<PhysicalOutputElection>,
    path: TemporalGenerationPath,
    feedback: Option<&TemporalCircuitStep>,
    copy_slot: Option<CopySlot>,
    copy_place_ordinal: u8,
    generation_index: u64,
    temporal_event_count: u64,
) -> TemporalGeneration {
    TemporalGeneration {
        surface_token_id: surface.token_id,
        surface_output_cell_id: surface.output_cell_id,
        generation_index,
        temporal_event_count,
        activated_columns: surface.activated_columns,
        candidate_tokens: surface.candidate_tokens,
        control_token_id: control.map_or(NONE, |election| election.token_id),
        control_output_cell_id: control.map_or(NONE, |election| election.output_cell_id),
        surface_winner_tick: surface.winner_tick,
        surface_basket_tick: surface.basket_tick,
        surface_inhibition_onset_tick: surface.inhibition_onset_tick,
        total_simulated_ticks: surface.simulated_ticks,
        control_winner_tick: control.map_or(0, |election| election.winner_tick),
        control_basket_tick: control.map_or(0, |election| election.basket_tick),
        control_simulated_end_tick: control.map_or(0, |election| election.simulated_ticks),
        path,
        winner_input_columns: surface.winner_input_columns,
        feedback_predicted_columns: feedback.map_or(0, |step| step.predicted_columns),
        feedback_predicted: feedback.is_some_and(|step| step.predicted),
        copy_place_ordinal,
        physical_election_count: if control.is_some() { 2 } else { 1 },
        copy_slot: copy_slot.map_or(u8::MAX, |slot| slot as u8),
    }
}

fn build_reinforced_row(
    store: &TemporalStore,
    config: TemporalCircuitConfig,
    segment_id: u32,
    normalized_context_sources: &[u32; ACTIVE_COLUMNS],
    context_source_count: usize,
    learning_step: u64,
) -> Result<TemporalSegmentRow, TemporalCircuitStop> {
    if context_source_count != ACTIVE_COLUMNS {
        return Err(TemporalCircuitStop::StoreOrPlanInvariant(
            "reinforcement requires one complete logical source context",
        ));
    }
    let old = store.runtime_segment_row(segment_id).map_err(|error| {
        map_store_preflight_error(error, store, 0, config.max_segments_per_cell)
    })?;
    let owner = store.runtime_segment_owner(segment_id).ok_or(
        TemporalCircuitStop::StoreOrPlanInvariant("reinforced segment has no authoritative owner"),
    )?;
    let mut row = TemporalSegmentRow {
        normalized_sources: [NONE; PAIRS_PER_SEGMENT],
        permanence_ticks: [0; PAIRS_PER_SEGMENT],
        live_pair_count: 0,
    };
    let mut retained_count = 0usize;
    for slot in 0..usize::from(old.live_pair_count) {
        let source = old.normalized_sources[slot];
        let ticks = old.permanence_ticks[slot];
        let updated = if normalized_context_sources.contains(&source) {
            ticks
                .saturating_add(config.permanence_increment_ticks)
                .min(500)
        } else {
            ticks.saturating_sub(config.permanence_increment_ticks)
        };
        if updated != 0 {
            row.normalized_sources[retained_count] = source;
            row.permanence_ticks[retained_count] = updated;
            retained_count += 1;
        }
    }

    let missing_count = PAIRS_PER_SEGMENT - retained_count;
    if missing_count != 0 {
        let mut ordered_context = *normalized_context_sources;
        ordered_context.sort_unstable();
        let mut candidates = [NONE; ACTIVE_COLUMNS];
        let mut candidate_count = 0usize;
        for source in ordered_context {
            if !row.normalized_sources[..retained_count].contains(&source) {
                candidates[candidate_count] = source;
                candidate_count += 1;
            }
        }
        let sample_count = missing_count.min(candidate_count);
        let regrowth_key = philox2x64_10(
            u64::from(owner),
            u64::from(segment_id),
            u64::from(config.construction_seed) ^ REGROWTH_CONTACT_DOMAIN,
        )
        .0;
        let mut sample_scratch = [0u64; ACTIVE_COLUMNS];
        let Some(selected) = sample_without_replacement(
            candidate_count,
            sample_count,
            regrowth_key,
            learning_step,
            &mut sample_scratch,
        ) else {
            return Err(TemporalCircuitStop::PhiloxAddressExhaustion {
                operation: "reinforcement-regrowth",
                segment_id,
            });
        };
        for &position in selected {
            row.normalized_sources[retained_count] = candidates[position as usize];
            row.permanence_ticks[retained_count] = config.initial_permanence_ticks;
            retained_count += 1;
        }
    }
    row.live_pair_count = retained_count as u8;
    sort_canonical_row_prefix(&mut row);
    Ok(row)
}

fn build_punished_row(
    store: &TemporalStore,
    config: TemporalCircuitConfig,
    segment_id: u32,
    normalized_context_sources: &[u32; ACTIVE_COLUMNS],
    context_source_count: usize,
) -> Result<TemporalSegmentRow, TemporalCircuitStop> {
    if context_source_count != ACTIVE_COLUMNS {
        return Err(TemporalCircuitStop::StoreOrPlanInvariant(
            "punishment requires one complete logical source context",
        ));
    }
    let old = store.runtime_segment_row(segment_id).map_err(|error| {
        map_store_preflight_error(error, store, 0, config.max_segments_per_cell)
    })?;
    Ok(build_prevalidated_punished_row(
        old,
        config,
        normalized_context_sources,
    ))
}

fn build_prevalidated_punished_row(
    old: TemporalSegmentRow,
    config: TemporalCircuitConfig,
    normalized_context_sources: &[u32; ACTIVE_COLUMNS],
) -> TemporalSegmentRow {
    let mut row = TemporalSegmentRow {
        normalized_sources: [NONE; PAIRS_PER_SEGMENT],
        permanence_ticks: [0; PAIRS_PER_SEGMENT],
        live_pair_count: 0,
    };
    let mut retained_count = 0usize;
    for slot in 0..usize::from(old.live_pair_count) {
        let source = old.normalized_sources[slot];
        let ticks = old.permanence_ticks[slot];
        let updated = if normalized_context_sources.contains(&source)
            && ticks >= config.connected_permanence_ticks
        {
            ticks.saturating_sub(config.negative_decrement_ticks)
        } else {
            ticks
        };
        if updated != 0 {
            row.normalized_sources[retained_count] = source;
            row.permanence_ticks[retained_count] = updated;
            retained_count += 1;
        }
    }
    row.live_pair_count = retained_count as u8;
    row
}

fn sort_canonical_row_prefix(row: &mut TemporalSegmentRow) {
    let live_count = usize::from(row.live_pair_count);
    for index in 1..live_count {
        let source = row.normalized_sources[index];
        let ticks = row.permanence_ticks[index];
        let mut destination = index;
        while destination != 0 && row.normalized_sources[destination - 1] > source {
            row.normalized_sources[destination] = row.normalized_sources[destination - 1];
            row.permanence_ticks[destination] = row.permanence_ticks[destination - 1];
            destination -= 1;
        }
        row.normalized_sources[destination] = source;
        row.permanence_ticks[destination] = ticks;
    }
}

fn map_store_preflight_error(
    error: TemporalStorePreflightError,
    store: &TemporalStore,
    planned_appends: u32,
    max_segments_per_cell: u8,
) -> TemporalCircuitStop {
    match error {
        TemporalStorePreflightError::InvalidSegmentId
        | TemporalStorePreflightError::InvalidOwner
        | TemporalStorePreflightError::InvalidCanonicalRow
        | TemporalStorePreflightError::BackingInvariant
        | TemporalStorePreflightError::StorePoisoned => {
            TemporalCircuitStop::StoreOrPlanInvariant("temporal store mutation preflight failed")
        }
        TemporalStorePreflightError::OutputCapacity { .. }
        | TemporalStorePreflightError::PreallocatedCapacity
        | TemporalStorePreflightError::BackingCapacityInvariant => {
            TemporalCircuitStop::PreallocatedCapacityInvariant(
                "temporal store mutation scratch or backing capacity is insufficient",
            )
        }
        TemporalStorePreflightError::SegmentCapacity => {
            TemporalCircuitStop::GlobalSegmentCapacity {
                required: store.runtime_segment_count() + planned_appends,
                capacity: store.runtime_segment_capacity(),
            }
        }
        TemporalStorePreflightError::OwnerCapacity { owner } => {
            TemporalCircuitStop::OwnerSegmentCapacity {
                owner,
                cap: max_segments_per_cell,
            }
        }
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

    pub(super) fn count_allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
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
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: u8::MAX,
            theta_match: 18,
            theta_active: 20,
            context_policy,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
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
    fn prior_winner_context_physically_elects_the_learned_next_token() {
        let mut runtime = python_oracle_runtime(255);
        runtime.recognize(0).expect("public cue boundary");

        let election = runtime
            .generate()
            .expect("active target owners must generate one token");

        assert_eq!(election.surface_token_id, 1);
        assert_eq!(election.surface_output_cell_id, 111);
        assert_eq!(election.activated_columns, ACTIVE_COLUMNS as u32);
        assert_eq!(election.candidate_tokens, 1);
        assert_eq!(election.winner_input_columns, ACTIVE_COLUMNS as u8);
        assert!(election.surface_winner_tick < election.surface_basket_tick);
        assert_eq!(
            election.surface_inhibition_onset_tick,
            election.surface_basket_tick + 1
        );
        assert!(election.surface_basket_tick <= election.total_simulated_ticks);
        assert_eq!(election.path, TemporalGenerationPath::Temporal);
        assert_eq!(election.temporal_event_count, 2);
    }

    #[test]
    fn normalized_multi_slot_copy_emits_unseen_name_and_literal_then_resumes() {
        const EOS: u32 = 256;
        let mut train = b"Function "
            .iter()
            .map(|&byte| u32::from(byte))
            .collect::<Vec<_>>();
        train.push(role_place_copy::IDENTIFIER_START_TOKEN_ID);
        train.extend(b"inc".iter().map(|&byte| u32::from(byte)));
        train.push(role_place_copy::IDENTIFIER_END_TOKEN_ID);
        train.extend(b" adds ".iter().map(|&byte| u32::from(byte)));
        train.push(role_place_copy::LITERAL_START_TOKEN_ID);
        train.push(u32::from(b'1'));
        train.push(role_place_copy::LITERAL_END_TOKEN_ID);
        train.extend(b".\n".iter().map(|&byte| u32::from(byte)));
        train.extend(b"def ".iter().map(|&byte| u32::from(byte)));
        train.push(role_place_copy::COPY_ID_TOKEN_ID);
        train.extend(b"(x):\n    return x + ".iter().map(|&byte| u32::from(byte)));
        train.push(role_place_copy::COPY_LITERAL_TOKEN_ID);
        train.push(u32::from(b'\n'));
        train.push(EOS);

        let mut token_ids = train.clone();
        token_ids.extend(b"boost7".iter().map(|&byte| u32::from(byte)));
        token_ids.push(COPY_END_TOKEN_ID);
        token_ids.sort_unstable();
        token_ids.dedup();
        let column_count = u32::try_from(token_ids.len() * ACTIVE_COLUMNS).unwrap();
        let mut copy_config = config(column_count, 8, TemporalContextPolicy::BoundWinners);
        copy_config.segment_capacity = 2048;
        copy_config.theta_match = 16;
        copy_config.theta_active = 18;
        let codebook = token_ids
            .iter()
            .enumerate()
            .map(|(code_index, &token_id)| TokenCode {
                token_id,
                columns: std::array::from_fn(|offset| {
                    (code_index * ACTIVE_COLUMNS + offset) as u32
                }),
            })
            .collect::<Vec<_>>();
        let physical_count = 2 * copy_config.max_pyramidal_cells + 3 * column_count;
        let mut runtime = TemporalCircuitRuntime::new(
            copy_config,
            codebook,
            base_members(copy_config),
            vec![0.0; physical_count as usize],
            &[],
        )
        .expect("copy runtime");

        for token_id in train {
            runtime
                .step(token_id, TemporalStepMode::Learn)
                .expect("TRAIN transition");
        }
        runtime.reset_transient();
        let mut dev = b"Function "
            .iter()
            .map(|&byte| u32::from(byte))
            .collect::<Vec<_>>();
        dev.push(role_place_copy::IDENTIFIER_START_TOKEN_ID);
        dev.extend(b"boost".iter().map(|&byte| u32::from(byte)));
        dev.push(role_place_copy::IDENTIFIER_END_TOKEN_ID);
        dev.extend(b" adds ".iter().map(|&byte| u32::from(byte)));
        dev.push(role_place_copy::LITERAL_START_TOKEN_ID);
        dev.push(u32::from(b'7'));
        dev.push(role_place_copy::LITERAL_END_TOKEN_ID);
        dev.extend(b".\n".iter().map(|&byte| u32::from(byte)));
        for token_id in dev {
            runtime
                .step(token_id, TemporalStepMode::ReadOnly)
                .expect("DEV prompt transition");
        }

        let mut surfaced = Vec::new();
        let mut active_copy = None;
        let mut entered_slots = Vec::new();
        for _ in 0..64 {
            let generation = runtime.generate().expect("teacher-free generation");
            if generation.path == TemporalGenerationPath::EnterCopy {
                let slot = generation.copy_slot;
                assert!(slot == CopySlot::Identifier as u8 || slot == CopySlot::Literal as u8);
                active_copy = Some((slot, generation.temporal_event_count));
                entered_slots.push(slot);
                assert_eq!(generation.copy_place_ordinal, 0);
            } else if generation.path == TemporalGenerationPath::Copy {
                let (slot, event_count) = active_copy.expect("COPY follows ENTER_COPY");
                assert_eq!(generation.copy_slot, slot);
                assert_eq!(generation.temporal_event_count, event_count);
            } else if generation.path == TemporalGenerationPath::ExitCopy {
                let (slot, event_count) = active_copy.take().expect("EXIT_COPY follows copy");
                assert_eq!(generation.copy_slot, slot);
                assert_eq!(generation.temporal_event_count, event_count + 1);
            }
            surfaced.push(generation.surface_token_id);
            if generation.surface_token_id == EOS {
                break;
            }
        }
        let mut expected = b"def boost(x):\n    return x + 7\n"
            .iter()
            .map(|&byte| u32::from(byte))
            .collect::<Vec<_>>();
        expected.push(EOS);
        assert_eq!(surfaced, expected);
        assert_eq!(
            entered_slots,
            vec![CopySlot::Identifier as u8, CopySlot::Literal as u8]
        );
        assert!(active_copy.is_none());
    }

    #[test]
    fn authoritative_physics_identity_preserves_literal_runtime_state_trace() {
        let mut runtime = python_oracle_runtime(255);
        let boundary = runtime.recognize(0).expect("boundary step");
        let active = runtime.recognize(1).expect("active replay step");

        assert_python_boundary_step(&boundary, 1);
        assert_python_active_step(&active, 2);
        assert_eq!(runtime.previous_context, active.context);
        assert_eq!(runtime.previous_winner_context, active.winner_context);
        assert_eq!(runtime.event_count, 2);
        assert_eq!(runtime.learning_step, 0);
        assert_eq!(runtime.scientific_stop, None);
    }

    #[test]
    fn read_only_success_and_multi_owner_race_make_zero_post_construction_allocations() {
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
                owner: 23,
                sources: sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 22,
                sources,
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
        ];
        let mut offsets = vec![0.0; physical_count(config)];
        offsets[22] = -0.024_791_163_874_911_56;
        offsets[23] = 0.033_505_381_138_863_34;
        let mut multi_owner = runtime_with_offsets(config, offsets, &segments);
        multi_owner.recognize(0).unwrap();
        let (raced, race_allocations) = count_allocations(|| multi_owner.recognize(1));
        assert_eq!(race_allocations, 0);
        assert_eq!(raced.unwrap().winner_spikes[0].cell_id, 23);
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
    fn multiple_active_and_matching_owners_enter_the_deterministic_physical_race() {
        for (ticks, expected_predicted_columns, class) in [
            (205, 0, CandidateClass::Matching { load: 1 }),
            (255, 1, CandidateClass::Active { load: 1 }),
        ] {
            let config = config(22, 2, TemporalContextPolicy::BoundWinners);
            let sources = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
            let segments = [
                TemporalSegmentInit {
                    owner: 43,
                    sources: sources.clone(),
                    permanence_ticks: vec![ticks; PAIRS_PER_SEGMENT],
                },
                TemporalSegmentInit {
                    owner: 42,
                    sources,
                    permanence_ticks: vec![ticks; PAIRS_PER_SEGMENT],
                },
            ];
            let mut offsets = vec![0.0; physical_count(config)];
            offsets[42] = -0.024_791_163_874_911_56;
            offsets[43] = 0.033_505_381_138_863_34;
            let mut first = runtime_with_offsets(config, offsets.clone(), &segments);
            let mut replay = runtime_with_offsets(config, offsets, &segments);
            first.recognize(0).expect("first context step");
            replay.recognize(0).expect("replay context step");

            let raced = first.recognize(1).expect("multi-owner race");
            let repeated = replay.recognize(1).expect("deterministic replay");
            assert_eq!(raced, repeated);
            assert_eq!(raced.winner_spikes[10].cell_id, 43);
            assert_eq!(raced.predicted_columns, expected_predicted_columns);
            assert_eq!(
                raced.winner_spikes[10].tick,
                singleton_latency_ticks(class.current_pa(), 0.033_505_381_138_863_34).unwrap()
            );
            assert_eq!(first.column_drive_scratch[0].cell_id, 42);
            assert_eq!(first.column_drive_scratch[1].cell_id, 43);
            assert_eq!(
                first.column_drive_scratch[0].current_pa.to_bits(),
                (class.current_pa() - TEMPORAL_PHYSICS_IDENTITY.novel_rank_decrement_pa).to_bits(),
            );
            assert_eq!(
                first.column_drive_scratch[1].current_pa.to_bits(),
                class.current_pa().to_bits(),
            );
        }
    }

    #[test]
    fn categorized_owner_count_above_total_race_width_remains_a_scientific_stop() {
        let mut config = config(22, 1, TemporalContextPolicy::BoundWinners);
        config.novel_race_cap = 1;
        config.max_pyramidal_cells += 1;
        config.max_segments_per_cell = 1;
        let sources = (0..PAIRS_PER_SEGMENT as u32).collect::<Vec<_>>();
        let segments = [
            TemporalSegmentInit {
                owner: 11,
                sources: sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 22,
                sources,
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
        ];
        let mut members = base_members(config);
        members[11].push(22);
        let mut runtime = TemporalCircuitRuntime::new(
            config,
            two_token_codebook(),
            members,
            vec![0.0; physical_count(config)],
            &segments,
        )
        .expect("valid committed reserve fixture");
        runtime.recognize(0).expect("boundary step");
        assert_eq!(
            runtime.recognize(1),
            Err(TemporalCircuitStop::CategorizedRaceCapacity {
                column: 11,
                categorized_count: 2,
                race_cap: 1,
            })
        );
        assert_eq!(
            runtime.scientific_stop,
            Some(TemporalPoisonCode::CategorizedRaceCapacity)
        );
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
        reserve_config.segment_capacity = 3;
        reserve_config.max_segments_per_cell = 1;
        let mut complete = base_members(reserve_config);
        complete[0].extend([22, 23]);
        let offsets = vec![0.0; physical_count(reserve_config)];
        let predecessor = TemporalSegmentInit {
            owner: 0,
            sources: Vec::new(),
            permanence_ticks: Vec::new(),
        };
        let cohort_witness = TemporalSegmentInit {
            owner: 22,
            sources: Vec::new(),
            permanence_ticks: Vec::new(),
        };
        let runtime = TemporalCircuitRuntime::new(
            reserve_config,
            two_token_codebook(),
            complete.clone(),
            offsets.clone(),
            &[predecessor.clone(), cohort_witness.clone()],
        )
        .unwrap();
        assert_eq!(runtime.cohort_columns, vec![0]);
        assert_eq!(runtime.next_reserve_cell, 24);

        assert!(matches!(
            TemporalCircuitRuntime::new(
                reserve_config,
                two_token_codebook(),
                complete.clone(),
                offsets.clone(),
                std::slice::from_ref(&predecessor),
            ),
            Err(TemporalRuntimeBuildError::Invalid(
                "committed reserve cohort has no owned segment"
            ))
        ));
        assert!(matches!(
            TemporalCircuitRuntime::new(
                reserve_config,
                two_token_codebook(),
                complete.clone(),
                offsets.clone(),
                &[cohort_witness],
            ),
            Err(TemporalRuntimeBuildError::Invalid(
                "reserve cohort was recruited before its column predecessors were exhausted"
            ))
        ));

        complete[0].pop();
        assert!(matches!(
            TemporalCircuitRuntime::new(
                reserve_config,
                two_token_codebook(),
                complete,
                offsets,
                &[predecessor],
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
    fn construction_rebuilds_global_cohort_order_and_replays_earlier_predecessors() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.max_pyramidal_cells = 28;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.segment_capacity = 6;
        let mut members = base_members(circuit_config);
        members[0].extend([24, 25]);
        members[5].extend([22, 23, 26, 27]);
        let empty_row = |owner| TemporalSegmentInit {
            owner,
            sources: Vec::new(),
            permanence_ticks: Vec::new(),
        };
        let valid = [
            empty_row(5),
            empty_row(22),
            empty_row(23),
            empty_row(0),
            empty_row(24),
            empty_row(26),
        ];
        let runtime = TemporalCircuitRuntime::new(
            circuit_config,
            two_token_codebook(),
            members.clone(),
            vec![0.0; physical_count(circuit_config)],
            &valid,
        )
        .unwrap();
        assert_eq!(runtime.cohort_columns, vec![5, 0, 5]);
        assert_eq!(runtime.next_reserve_cell, 28);

        let premature_later = [
            empty_row(5),
            empty_row(22),
            empty_row(0),
            empty_row(24),
            empty_row(26),
        ];
        assert!(matches!(
            TemporalCircuitRuntime::new(
                circuit_config,
                two_token_codebook(),
                members,
                vec![0.0; physical_count(circuit_config)],
                &premature_later,
            ),
            Err(TemporalRuntimeBuildError::Invalid(
                "reserve cohort was recruited before its column predecessors were exhausted"
            ))
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

        let mapped = [
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
                TemporalCircuitStop::PhiloxAddressExhaustion {
                    operation: "literal fixture",
                    segment_id: 7,
                },
                TemporalPoisonCode::PhiloxAddressExhaustion,
            ),
            (
                TemporalCircuitStop::ReserveCellCapacity { column: 11 },
                TemporalPoisonCode::ReserveCellCapacity,
            ),
            (
                TemporalCircuitStop::GlobalSegmentCapacity {
                    required: 12,
                    capacity: 11,
                },
                TemporalPoisonCode::GlobalSegmentCapacity,
            ),
            (
                TemporalCircuitStop::OwnerSegmentCapacity { owner: 22, cap: 7 },
                TemporalPoisonCode::OwnerSegmentCapacity,
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
        assert_eq!(
            TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::LearningCounter,
                location: 0,
            }
            .poison_code(),
            Some(TemporalPoisonCode::TickOrCounterOverflow),
        );
        assert_eq!(TemporalCircuitStop::UnknownToken(99).poison_code(), None,);
        assert_eq!(TemporalCircuitStop::Poisoned.poison_code(), None);

        let mut owner_cap_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        owner_cap_config.max_segments_per_cell = 7;
        let owner_cap_runtime = runtime_with_offsets(
            owner_cap_config,
            vec![0.0; physical_count(owner_cap_config)],
            &[],
        );
        assert_eq!(
            map_store_preflight_error(
                TemporalStorePreflightError::OwnerCapacity { owner: 11 },
                &owner_cap_runtime.store,
                1,
                owner_cap_config.max_segments_per_cell,
            ),
            TemporalCircuitStop::OwnerSegmentCapacity { owner: 11, cap: 7 },
        );
        assert_eq!(
            map_store_preflight_error(
                TemporalStorePreflightError::BackingInvariant,
                &owner_cap_runtime.store,
                0,
                owner_cap_config.max_segments_per_cell,
            ),
            TemporalCircuitStop::StoreOrPlanInvariant("temporal store mutation preflight failed"),
        );
        assert_eq!(
            map_store_preflight_error(
                TemporalStorePreflightError::BackingCapacityInvariant,
                &owner_cap_runtime.store,
                0,
                owner_cap_config.max_segments_per_cell,
            ),
            TemporalCircuitStop::PreallocatedCapacityInvariant(
                "temporal store mutation scratch or backing capacity is insufficient",
            ),
        );
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

    #[test]
    fn growth_off_learn_advances_only_transient_physics() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        circuit_config.plasticity_policy = TemporalPlasticityPolicy::GrowthOff;
        let mut runtime = runtime_with_offsets(
            circuit_config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &[],
        );
        assert!(runtime.active_segment_scratch.is_empty());
        assert_eq!(runtime.active_segment_scratch.capacity(), 0);
        let segment_count = runtime.store.runtime_segment_count();

        let learned = runtime
            .step(0, TemporalStepMode::Learn)
            .expect("growth-off LEARN must still run the physical step");

        assert_python_boundary_step(&learned, 1);
        assert_eq!(learned.allocated_segments, 0);
        assert_eq!(learned.punished_segments, 0);
        assert_eq!(runtime.store.runtime_segment_count(), segment_count);
        assert_eq!(runtime.learning_step, 0);
    }

    #[test]
    fn learn_domains_match_literal_python_init_and_regrowth_vectors() {
        let mut initial_scratch = [0u64; 8];
        assert_eq!(
            sample_without_replacement(8, 4, 17 ^ INITIAL_CONTACT_DOMAIN, 1, &mut initial_scratch,),
            Some(&[2, 3, 5, 7][..]),
        );
        let regrowth_key = philox2x64_10(44, 0, 17 ^ REGROWTH_CONTACT_DOMAIN).0;
        assert_eq!(regrowth_key, 0xC5AC_1D69_C2B8_FBC5);
        let mut regrowth_scratch = [0u64; ACTIVE_COLUMNS];
        assert_eq!(
            sample_without_replacement(11, 3, regrowth_key, 3, &mut regrowth_scratch),
            Some(&[2, 6, 7][..]),
        );
    }

    #[test]
    fn plastic_learn_increments_at_boundary_then_allocates_shared_literal_init_rows() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        let mut runtime = runtime_with_offsets(
            circuit_config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &[],
        );

        let boundary = runtime.step(0, TemporalStepMode::Learn).unwrap();
        assert_python_boundary_step(&boundary, 1);
        assert_eq!(runtime.learning_step, 1);
        assert_eq!(runtime.store.runtime_segment_count(), 0);

        let target = runtime.step(1, TemporalStepMode::Learn).unwrap();
        assert_eq!(
            target.winner_context[..ACTIVE_COLUMNS],
            (11..22).collect::<Vec<_>>()
        );
        assert_eq!(target.allocated_segments, 11);
        assert_eq!(target.punished_segments, 0);
        assert_eq!(runtime.learning_step, 2);
        assert_eq!(runtime.store.runtime_segment_count(), 11);
        let expected_sources = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        for segment_id in 0..11 {
            assert_eq!(
                runtime.store.runtime_segment_owner(segment_id),
                Some(11 + segment_id)
            );
            let row = runtime.store.runtime_segment_row(segment_id).unwrap();
            assert_eq!(row.normalized_sources, expected_sources);
            assert_eq!(row.permanence_ticks, [255; PAIRS_PER_SEGMENT]);
            assert_eq!(row.live_pair_count, PAIRS_PER_SEGMENT as u8);
        }
        runtime.reset_transient();
        assert_eq!(runtime.learning_step, 2);
        assert_eq!(runtime.store.runtime_segment_count(), 11);
    }

    #[test]
    fn partial_potential_match_cannot_own_winner_or_suppress_novel_allocation() {
        let circuit_config = config(22, 2, TemporalContextPolicy::BoundWinners);
        let active_sources: Vec<u32> = (0..20).step_by(2).collect();
        let matching_sources = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 43];
        let segments = [
            TemporalSegmentInit {
                owner: 22,
                sources: active_sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 22,
                sources: active_sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 0,
                sources: active_sources.clone(),
                permanence_ticks: vec![250; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 24,
                sources: matching_sources,
                permanence_ticks: vec![205, 205, 205, 205, 205, 205, 205, 205, 205, 50],
            },
            TemporalSegmentInit {
                owner: 0,
                sources: active_sources,
                permanence_ticks: vec![250; PAIRS_PER_SEGMENT],
            },
        ];
        let mut runtime = runtime_with_offsets(
            circuit_config,
            vec![0.0; physical_count(circuit_config)],
            &segments,
        );
        runtime.step(0, TemporalStepMode::Learn).unwrap();

        let target = runtime.step(1, TemporalStepMode::Learn).unwrap();

        assert_eq!(target.winner_spikes[0].cell_id, 22);
        assert_eq!(target.winner_spikes[1].cell_id, 25);
        assert_eq!(target.matching_segments, 5);
        assert_eq!(target.active_segments, 4);
        assert_eq!(target.predicted_columns, 1);
        assert_eq!(target.allocated_segments, 10);
        assert_eq!(target.punished_segments, 2);
        assert_eq!(runtime.learning_step, 2);
        assert_eq!(runtime.store.runtime_segment_count(), 15);
        assert_eq!(
            runtime
                .store
                .runtime_segment_row(0)
                .unwrap()
                .permanence_ticks,
            [305; PAIRS_PER_SEGMENT],
        );
        assert_eq!(
            runtime
                .store
                .runtime_segment_row(1)
                .unwrap()
                .permanence_ticks,
            [255; PAIRS_PER_SEGMENT],
            "an active non-best segment on the correct winner owner is exempt",
        );
        assert_eq!(
            runtime
                .store
                .runtime_segment_row(2)
                .unwrap()
                .permanence_ticks,
            [248; PAIRS_PER_SEGMENT],
            "an active segment whose owner did not win is punished",
        );
        let untouched_match = runtime.store.runtime_segment_row(3).unwrap();
        assert_eq!(
            untouched_match.normalized_sources,
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 43],
            "a potential-only segment remains evidence, not the new identity",
        );
        assert_eq!(
            untouched_match.permanence_ticks,
            [205, 205, 205, 205, 205, 205, 205, 205, 205, 50],
            "a potential-only segment is not rewritten toward the current context",
        );
        assert_eq!(
            runtime
                .store
                .runtime_segment_row(4)
                .unwrap()
                .permanence_ticks,
            [248; PAIRS_PER_SEGMENT],
            "a non-best active segment on a losing owner is still punished",
        );
    }

    #[test]
    fn eleven_exhausted_columns_recruit_full_cohorts_in_token_order_atomically() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.max_pyramidal_cells +=
            ACTIVE_COLUMNS as u32 * u32::from(circuit_config.novel_race_cap);
        let unrelated_sources = vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
        let segments: Vec<_> = (11..22)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: unrelated_sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut runtime = runtime_with_offsets(
            circuit_config,
            vec![0.0; physical_count(circuit_config)],
            &segments,
        );
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        let step = runtime.step(1, TemporalStepMode::Learn).unwrap();

        assert_eq!(runtime.cohort_columns, (11..22).collect::<Vec<_>>());
        assert_eq!(runtime.next_reserve_cell, 110);
        assert_eq!(
            runtime.column_offsets,
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 29, 38, 47, 56, 65, 74, 83, 92, 101, 110,
            ]
        );
        for (token_offset, column) in (11usize..22).enumerate() {
            let cell_start = runtime.column_offsets[column];
            let cell_end = runtime.column_offsets[column + 1];
            let cohort_start = 22 + token_offset as u32 * 8;
            let mut expected = vec![column as u32];
            expected.extend(cohort_start..cohort_start + 8);
            assert_eq!(&runtime.column_cells[cell_start..cell_end], expected);
            assert!((cohort_start..cohort_start + 8)
                .contains(&step.winner_spikes[token_offset].cell_id));
            assert_eq!(
                runtime
                    .store
                    .runtime_owner_load(step.winner_spikes[token_offset].cell_id),
                Some(1)
            );
        }
        assert_eq!(step.allocated_segments, ACTIVE_COLUMNS as u8);
        assert_eq!(runtime.store.runtime_segment_count(), 22);
        assert_eq!(runtime.learning_step, 2);
        assert_eq!(runtime.event_count, 2);
        assert_eq!(runtime.scientific_stop, None);
    }

    #[test]
    fn generic_afferent_policy_recruits_the_same_full_prospective_cohorts() {
        let mut circuit_config = config(22, 2, TemporalContextPolicy::GenericAfferents);
        circuit_config.novel_race_cap = 2;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.max_pyramidal_cells +=
            ACTIVE_COLUMNS as u32 * u32::from(circuit_config.novel_race_cap);
        let proximal_offset = 2 * circuit_config.max_pyramidal_cells + circuit_config.column_count;
        let unrelated_sources: Vec<_> = (proximal_offset + 11..proximal_offset + 21).collect();
        let segments: Vec<_> = (22..44)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: unrelated_sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut runtime = runtime_with_offsets(
            circuit_config,
            vec![0.0; physical_count(circuit_config)],
            &segments,
        );

        runtime.step(0, TemporalStepMode::Learn).unwrap();
        let step = runtime.step(1, TemporalStepMode::Learn).unwrap();

        assert_eq!(runtime.cohort_columns, (11..22).collect::<Vec<_>>());
        assert_eq!(runtime.next_reserve_cell, 66);
        for (token_offset, column) in (11usize..22).enumerate() {
            let start = runtime.column_offsets[column];
            let end = runtime.column_offsets[column + 1];
            let cohort_start = 44 + token_offset as u32 * 2;
            assert_eq!(
                &runtime.column_cells[start..end],
                &[
                    2 * column as u32,
                    2 * column as u32 + 1,
                    cohort_start,
                    cohort_start + 1,
                ]
            );
            assert!((cohort_start..cohort_start + 2)
                .contains(&step.winner_spikes[token_offset].cell_id));
        }
        assert_eq!(step.allocated_segments, ACTIVE_COLUMNS as u8);
        assert_eq!(runtime.store.runtime_segment_count(), 33);
    }

    #[test]
    fn recruiting_step_uses_prospective_scratch_without_allocation_or_capacity_change() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.max_pyramidal_cells +=
            ACTIVE_COLUMNS as u32 * u32::from(circuit_config.novel_race_cap);
        let segments: Vec<_> = (11..22)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: (11..21).collect(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut runtime = runtime_with_offsets(
            circuit_config,
            vec![0.0; physical_count(circuit_config)],
            &segments,
        );
        assert_eq!(runtime.column_drive_scratch.len(), 89);
        assert_eq!(runtime.novel_candidate_scratch.len(), 89);
        let pointers = (
            runtime.column_offsets.as_ptr(),
            runtime.column_cells.as_ptr(),
            runtime.alternate_column_offsets.as_ptr(),
            runtime.alternate_column_cells.as_ptr(),
            runtime.cohort_columns.as_ptr(),
            runtime.column_drive_scratch.as_ptr(),
            runtime.novel_candidate_scratch.as_ptr(),
        );
        let capacities = (
            runtime.column_offsets.capacity(),
            runtime.column_cells.capacity(),
            runtime.alternate_column_offsets.capacity(),
            runtime.alternate_column_cells.capacity(),
            runtime.cohort_columns.capacity(),
            runtime.column_drive_scratch.capacity(),
            runtime.novel_candidate_scratch.capacity(),
        );

        let (trace, allocations) = count_allocations(|| {
            [
                runtime.step(0, TemporalStepMode::Learn),
                runtime.step(1, TemporalStepMode::Learn),
            ]
        });
        assert!(trace.iter().all(Result::is_ok), "{trace:?}");
        assert_eq!(allocations, 0);
        assert_eq!(
            (
                runtime.column_offsets.as_ptr(),
                runtime.column_cells.as_ptr(),
                runtime.alternate_column_offsets.as_ptr(),
                runtime.alternate_column_cells.as_ptr(),
                runtime.cohort_columns.as_ptr(),
                runtime.column_drive_scratch.as_ptr(),
                runtime.novel_candidate_scratch.as_ptr(),
            ),
            (pointers.2, pointers.3, pointers.0, pointers.1, pointers.4, pointers.5, pointers.6,)
        );
        assert_eq!(
            (
                runtime.column_offsets.capacity(),
                runtime.column_cells.capacity(),
                runtime.alternate_column_offsets.capacity(),
                runtime.alternate_column_cells.capacity(),
                runtime.cohort_columns.capacity(),
                runtime.column_drive_scratch.capacity(),
                runtime.novel_candidate_scratch.capacity(),
            ),
            (
                capacities.2,
                capacities.3,
                capacities.0,
                capacities.1,
                capacities.4,
                capacities.5,
                capacities.6,
            )
        );
    }

    #[test]
    fn a_filled_recruited_cohort_crosses_the_next_full_cohort_boundary() {
        let mut circuit_config = config(44, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 2;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.max_pyramidal_cells += 4 * ACTIVE_COLUMNS as u32;
        let codebook: Vec<_> = (0..4u32)
            .map(|token_id| TokenCode {
                token_id,
                columns: std::array::from_fn(|index| {
                    token_id * ACTIVE_COLUMNS as u32 + index as u32
                }),
            })
            .collect();
        let initial_sources: Vec<_> = (33..43).collect();
        let segments: Vec<_> = (33..44)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: initial_sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut runtime = TemporalCircuitRuntime::new(
            circuit_config,
            codebook,
            base_members(circuit_config),
            vec![0.0; physical_count(circuit_config)],
            &segments,
        )
        .unwrap();

        runtime.step(0, TemporalStepMode::Learn).unwrap();
        let first = runtime.step(3, TemporalStepMode::Learn).unwrap();
        runtime.step(1, TemporalStepMode::Learn).unwrap();
        let fill = runtime.step(3, TemporalStepMode::Learn).unwrap();
        runtime.step(2, TemporalStepMode::Learn).unwrap();
        let second = runtime.step(3, TemporalStepMode::Learn).unwrap();

        assert_eq!(first.allocated_segments, ACTIVE_COLUMNS as u8);
        assert_eq!(fill.allocated_segments, ACTIVE_COLUMNS as u8);
        assert_eq!(second.allocated_segments, ACTIVE_COLUMNS as u8);
        assert_eq!(runtime.next_reserve_cell, 88);
        assert_eq!(
            runtime.cohort_columns,
            (33..44).chain(33..44).collect::<Vec<_>>()
        );
        for (column_offset, column) in (33usize..44).enumerate() {
            let start = runtime.column_offsets[column];
            let end = runtime.column_offsets[column + 1];
            assert_eq!(
                &runtime.column_cells[start..end],
                &[
                    column as u32,
                    44 + 2 * column_offset as u32,
                    45 + 2 * column_offset as u32,
                    66 + 2 * column_offset as u32,
                    67 + 2 * column_offset as u32,
                ]
            );
            assert!(runtime.column_cells[start + 1..start + 3]
                .iter()
                .all(|owner| runtime.store.runtime_owner_load(*owner) == Some(1)));
            assert!(runtime.column_cells[start + 3..end]
                .contains(&second.winner_spikes[column_offset].cell_id));
        }
    }

    #[test]
    fn literal_python_oracle_reserve_trace_is_bit_exact_through_fill_and_rerecruit() {
        // Generated with SparseTemporalCortex at Python oracle commit 9f55a87
        // (blob 2424d503...), zeroing its DendriticCortex threshold offsets,
        // importing the eleven literal base rows below, then executing
        // R(0), L(3), R(1), L(3), R(2), L(3). Every expectation below is a
        // copied Python result; Rust does not regenerate the fixture values.
        let mut circuit_config = config(14, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.max_pyramidal_cells = 58;
        circuit_config.novel_race_cap = 2;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.theta_match = 20;
        circuit_config.theta_active = 20;
        let codebook = vec![
            TokenCode {
                token_id: 0,
                columns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
            },
            TokenCode {
                token_id: 1,
                columns: [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13],
            },
            TokenCode {
                token_id: 2,
                columns: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
            },
            TokenCode {
                token_id: 3,
                columns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            },
        ];
        let segments: Vec<_> = (0..11)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut runtime = TemporalCircuitRuntime::new(
            circuit_config,
            codebook,
            base_members(circuit_config),
            vec![0.0; physical_count(circuit_config)],
            &segments,
        )
        .unwrap();

        let expected_trace = [
            (
                0,
                TemporalStepMode::ReadOnly,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
                [229, 229, 229, 229, 229, 229, 229, 229, 229, 222, 222],
                0,
                14,
            ),
            (
                3,
                TemporalStepMode::Learn,
                [14, 17, 18, 20, 23, 24, 26, 28, 30, 33, 35],
                [222; ACTIVE_COLUMNS],
                11,
                36,
            ),
            (
                1,
                TemporalStepMode::ReadOnly,
                [15, 16, 19, 21, 22, 25, 27, 29, 11, 12, 13],
                [222; ACTIVE_COLUMNS],
                0,
                36,
            ),
            (
                3,
                TemporalStepMode::Learn,
                [15, 16, 19, 21, 22, 25, 27, 29, 31, 32, 34],
                [222; ACTIVE_COLUMNS],
                11,
                36,
            ),
            (
                2,
                TemporalStepMode::ReadOnly,
                [16, 2, 20, 22, 24, 26, 7, 8, 33, 11, 12],
                [229, 229, 229, 229, 229, 229, 229, 229, 229, 222, 222],
                0,
                36,
            ),
            (
                3,
                TemporalStepMode::Learn,
                [36, 38, 41, 43, 44, 46, 48, 51, 52, 54, 56],
                [222; ACTIVE_COLUMNS],
                11,
                58,
            ),
        ];
        for (token, mode, winner_cells, winner_ticks, allocated, cursor) in expected_trace {
            let step = runtime.step(token, mode).unwrap();
            assert_eq!(step.winner_spikes, events(winner_cells, winner_ticks));
            assert_eq!(step.allocated_segments, allocated);
            assert_eq!(runtime.next_reserve_cell, cursor);
        }

        assert_eq!(runtime.learning_step, 3);
        assert_eq!(runtime.event_count, 6);
        assert_eq!(
            runtime.cohort_columns,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
        assert_eq!(
            runtime.column_offsets,
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 56, 57, 58]
        );
        assert_eq!(
            runtime.column_cells,
            [
                0, 14, 15, 36, 37, 1, 16, 17, 38, 39, 2, 18, 19, 40, 41, 3, 20, 21, 42, 43, 4, 22,
                23, 44, 45, 5, 24, 25, 46, 47, 6, 26, 27, 48, 49, 7, 28, 29, 50, 51, 8, 30, 31, 52,
                53, 9, 32, 33, 54, 55, 10, 34, 35, 56, 57, 11, 12, 13,
            ]
        );

        let expected_rows = [
            (
                [14, 17, 18, 20, 23, 24, 26, 28, 30, 33, 35],
                [0, 1, 3, 4, 5, 6, 7, 8, 11, 12],
            ),
            (
                [15, 16, 19, 21, 22, 25, 27, 29, 31, 32, 34],
                [11, 12, 13, 15, 19, 21, 22, 25, 27, 29],
            ),
            (
                [36, 38, 41, 43, 44, 46, 48, 51, 52, 54, 56],
                [2, 7, 8, 11, 12, 16, 20, 22, 26, 33],
            ),
        ];
        assert_eq!(runtime.store.runtime_segment_count(), 44);
        for (wave, (owners, sources)) in expected_rows.into_iter().enumerate() {
            for (offset, owner) in owners.into_iter().enumerate() {
                let segment_id = 11 + (wave * ACTIVE_COLUMNS + offset) as u32;
                assert_eq!(runtime.store.runtime_segment_owner(segment_id), Some(owner));
                let row = runtime.store.runtime_segment_row(segment_id).unwrap();
                assert_eq!(row.live_pair_count, PAIRS_PER_SEGMENT as u8);
                assert_eq!(row.normalized_sources, sources);
                assert_eq!(row.permanence_ticks, [255; PAIRS_PER_SEGMENT]);
            }
        }
    }

    #[test]
    fn read_only_and_growth_off_never_publish_a_required_reserve_cohort() {
        for (mode, plasticity) in [
            (
                TemporalStepMode::ReadOnly,
                TemporalPlasticityPolicy::Plastic,
            ),
            (TemporalStepMode::Learn, TemporalPlasticityPolicy::GrowthOff),
        ] {
            let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
            circuit_config.max_pyramidal_cells = 44;
            circuit_config.novel_race_cap = 2;
            circuit_config.max_segments_per_cell = 1;
            circuit_config.plasticity_policy = plasticity;
            let segments: Vec<_> = (11..22)
                .map(|owner| TemporalSegmentInit {
                    owner,
                    sources: vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                    permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
                })
                .collect();
            let mut runtime = runtime_with_offsets(
                circuit_config,
                vec![0.0; physical_count(circuit_config)],
                &segments,
            );
            let before_offsets = runtime.column_offsets.clone();
            let before_cells = runtime.column_cells.clone();

            runtime.step(0, mode).unwrap();
            let target = runtime.step(1, mode).unwrap();

            assert_eq!(target.allocated_segments, 0);
            assert_eq!(runtime.next_reserve_cell, 22);
            assert!(runtime.cohort_columns.is_empty());
            assert_eq!(runtime.column_offsets, before_offsets);
            assert_eq!(runtime.column_cells, before_cells);
            assert_eq!(runtime.store.runtime_segment_count(), 11);
            assert_eq!(runtime.learning_step, 0);
            assert_eq!(runtime.event_count, 2);
        }
    }

    #[test]
    fn insufficient_aggregate_reserve_room_latches_code12_without_logical_publish() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.max_pyramidal_cells +=
            (ACTIVE_COLUMNS as u32 - 1) * u32::from(circuit_config.novel_race_cap);
        let segments: Vec<_> = (11..22)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: (11..21).collect(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut runtime = runtime_with_offsets(
            circuit_config,
            vec![0.0; physical_count(circuit_config)],
            &segments,
        );
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        let before_offsets = runtime.column_offsets.clone();
        let before_cells = runtime.column_cells.clone();
        let before_context = runtime.previous_context;
        let before_winner_context = runtime.previous_winner_context;
        let before_event = runtime.event_count;
        let before_learning = runtime.learning_step;
        let before_segment_count = runtime.store.runtime_segment_count();

        assert_eq!(
            runtime.step(1, TemporalStepMode::Learn),
            Err(TemporalCircuitStop::ReserveCellCapacity { column: 11 })
        );
        assert_eq!(runtime.column_offsets, before_offsets);
        assert_eq!(runtime.column_cells, before_cells);
        assert!(runtime.cohort_columns.is_empty());
        assert_eq!(runtime.next_reserve_cell, 22);
        assert_eq!(runtime.previous_context, before_context);
        assert_eq!(runtime.previous_winner_context, before_winner_context);
        assert_eq!(runtime.event_count, before_event);
        assert_eq!(runtime.learning_step, before_learning);
        assert_eq!(runtime.store.runtime_segment_count(), before_segment_count);
        assert_eq!(
            runtime.scientific_stop,
            Some(TemporalPoisonCode::ReserveCellCapacity)
        );
    }

    #[test]
    fn event_overflow_after_reserve_staging_does_not_publish_cohorts() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.max_pyramidal_cells +=
            ACTIVE_COLUMNS as u32 * u32::from(circuit_config.novel_race_cap);
        let segments: Vec<_> = (11..22)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: (11..21).collect(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut runtime = runtime_with_offsets(
            circuit_config,
            vec![0.0; physical_count(circuit_config)],
            &segments,
        );
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        runtime.event_count = u64::MAX;
        let before_offsets = runtime.column_offsets.clone();
        let before_cells = runtime.column_cells.clone();
        let before_context = runtime.previous_context;
        let before_winner_context = runtime.previous_winner_context;
        let before_learning = runtime.learning_step;
        let before_segment_count = runtime.store.runtime_segment_count();

        assert_eq!(
            runtime.step(1, TemporalStepMode::Learn),
            Err(TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::EventCounter,
                location: 0,
            })
        );
        assert_eq!(runtime.column_offsets, before_offsets);
        assert_eq!(runtime.column_cells, before_cells);
        assert!(runtime.cohort_columns.is_empty());
        assert_eq!(runtime.next_reserve_cell, 22);
        assert_eq!(runtime.previous_context, before_context);
        assert_eq!(runtime.previous_winner_context, before_winner_context);
        assert_eq!(runtime.event_count, u64::MAX);
        assert_eq!(runtime.learning_step, before_learning);
        assert_eq!(runtime.store.runtime_segment_count(), before_segment_count);
    }

    #[test]
    fn column_physics_failure_after_reserve_staging_does_not_publish_cohorts() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.max_pyramidal_cells +=
            ACTIVE_COLUMNS as u32 * u32::from(circuit_config.novel_race_cap);
        let segments: Vec<_> = (11..22)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: (11..21).collect(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut offsets = vec![0.0; physical_count(circuit_config)];
        let first_target_basket = 2 * circuit_config.max_pyramidal_cells + 11;
        offsets[first_target_basket as usize] = 1.0e9;
        let mut runtime = runtime_with_offsets(circuit_config, offsets, &segments);
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        let before_offsets = runtime.column_offsets.clone();
        let before_cells = runtime.column_cells.clone();
        let before_context = runtime.previous_context;
        let before_event = runtime.event_count;
        let before_learning = runtime.learning_step;

        let stop = runtime.step(1, TemporalStepMode::Learn);
        assert!(
            matches!(
                stop,
                Err(TemporalCircuitStop::Physics {
                    stage: PhysicalStage::ColumnRace,
                    location: 11,
                    source: TemporalPhysicsStop::MultiplePyramidalSpikes { .. },
                })
            ),
            "{stop:?}"
        );
        assert_eq!(runtime.column_offsets, before_offsets);
        assert_eq!(runtime.column_cells, before_cells);
        assert!(runtime.cohort_columns.is_empty());
        assert_eq!(runtime.next_reserve_cell, 22);
        assert_eq!(runtime.previous_context, before_context);
        assert_eq!(runtime.event_count, before_event);
        assert_eq!(runtime.learning_step, before_learning);
        assert_eq!(runtime.store.runtime_segment_count(), ACTIVE_COLUMNS as u32);
        assert_eq!(
            runtime.scientific_stop,
            Some(TemporalPoisonCode::ColumnRacePhysics)
        );
    }

    #[test]
    fn global_segment_failure_after_reserve_staging_does_not_publish_cohorts() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.max_pyramidal_cells +=
            ACTIVE_COLUMNS as u32 * u32::from(circuit_config.novel_race_cap);
        circuit_config.segment_capacity = ACTIVE_COLUMNS as u32;
        let segments: Vec<_> = (11..22)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: (11..21).collect(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut runtime = runtime_with_offsets(
            circuit_config,
            vec![0.0; physical_count(circuit_config)],
            &segments,
        );
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        let before_offsets = runtime.column_offsets.clone();
        let before_cells = runtime.column_cells.clone();
        let before_context = runtime.previous_context;
        let before_winner_context = runtime.previous_winner_context;
        let before_event = runtime.event_count;
        let before_learning = runtime.learning_step;

        assert_eq!(
            runtime.step(1, TemporalStepMode::Learn),
            Err(TemporalCircuitStop::GlobalSegmentCapacity {
                required: 22,
                capacity: 11,
            })
        );
        assert_eq!(runtime.column_offsets, before_offsets);
        assert_eq!(runtime.column_cells, before_cells);
        assert!(runtime.cohort_columns.is_empty());
        assert_eq!(runtime.next_reserve_cell, 22);
        assert_eq!(runtime.previous_context, before_context);
        assert_eq!(runtime.previous_winner_context, before_winner_context);
        assert_eq!(runtime.event_count, before_event);
        assert_eq!(runtime.learning_step, before_learning);
        assert_eq!(runtime.store.runtime_segment_count(), ACTIVE_COLUMNS as u32);
        assert_eq!(
            runtime.scientific_stop,
            Some(TemporalPoisonCode::GlobalSegmentCapacity)
        );
    }

    #[test]
    fn proximal_physics_stop_precedes_reserve_required_planning() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        circuit_config.max_segments_per_cell = 1;
        circuit_config.max_pyramidal_cells += u32::from(circuit_config.novel_race_cap);
        let unrelated_sources = vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
        let segments: Vec<_> = (11..22)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: unrelated_sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let mut offsets = vec![0.0; physical_count(circuit_config)];
        let second_token_first_proximal = 2 * circuit_config.max_pyramidal_cells
            + circuit_config.column_count
            + ACTIVE_COLUMNS as u32;
        offsets[second_token_first_proximal as usize] = 1.0e9;
        let mut runtime = runtime_with_offsets(circuit_config, offsets, &segments);
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        runtime.learning_step = u64::MAX;
        let before_context = runtime.previous_context;
        let before_winner_context = runtime.previous_winner_context;
        let before_event = runtime.event_count;
        let before_learning = runtime.learning_step;

        assert_eq!(
            runtime.step(1, TemporalStepMode::Learn),
            Err(TemporalCircuitStop::Physics {
                stage: PhysicalStage::Proximal,
                location: second_token_first_proximal,
                source: TemporalPhysicsStop::NoPyramidalSpike,
            }),
        );
        assert_eq!(runtime.previous_context, before_context);
        assert_eq!(runtime.previous_winner_context, before_winner_context);
        assert_eq!(runtime.event_count, before_event);
        assert_eq!(runtime.learning_step, before_learning);
        assert_eq!(
            runtime.scientific_stop,
            Some(TemporalPoisonCode::ProximalPhysics)
        );
    }

    #[test]
    fn recognized_column_can_still_stop_on_the_physical_winners_exact_owner_cap() {
        let mut circuit_config = config(22, 2, TemporalContextPolicy::BoundWinners);
        circuit_config.max_segments_per_cell = 1;
        let segments = [
            TemporalSegmentInit {
                owner: 22,
                sources: (0..20).step_by(2).collect(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 23,
                sources: (1..20).step_by(2).collect(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
        ];
        let mut offsets = vec![0.0; physical_count(circuit_config)];
        offsets[22] = 1.0e9;
        let mut runtime = runtime_with_offsets(circuit_config, offsets, &segments);
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        runtime.learning_step = u64::MAX;
        let before_context = runtime.previous_context;
        let before_winner_context = runtime.previous_winner_context;
        let before_event = runtime.event_count;
        let before_learning = runtime.learning_step;
        let before_rows = [
            runtime.store.runtime_segment_row(0).unwrap(),
            runtime.store.runtime_segment_row(1).unwrap(),
        ];

        assert_eq!(
            runtime.step(1, TemporalStepMode::Learn),
            Err(TemporalCircuitStop::OwnerSegmentCapacity { owner: 23, cap: 1 }),
        );
        assert_eq!(runtime.previous_context, before_context);
        assert_eq!(runtime.previous_winner_context, before_winner_context);
        assert_eq!(runtime.event_count, before_event);
        assert_eq!(runtime.learning_step, before_learning);
        assert_eq!(runtime.store.runtime_segment_count(), 2);
        assert_eq!(
            runtime.store.runtime_segment_row(0).unwrap(),
            before_rows[0]
        );
        assert_eq!(
            runtime.store.runtime_segment_row(1).unwrap(),
            before_rows[1]
        );
        assert_eq!(
            runtime.scientific_stop,
            Some(TemporalPoisonCode::OwnerSegmentCapacity)
        );
    }

    #[test]
    fn construction_rejects_reserved_tokens_owner_overload_and_uncommitted_bound_sources() {
        let circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        let mut reserved_codebook = two_token_codebook();
        reserved_codebook[0].token_id = NONE;
        assert!(matches!(
            TemporalCircuitRuntime::new(
                circuit_config,
                reserved_codebook,
                base_members(circuit_config),
                vec![0.0; physical_count(circuit_config)],
                &[],
            ),
            Err(TemporalRuntimeBuildError::Invalid(
                "token IDs must be live uint32 values"
            ))
        ));

        let mut capped = circuit_config;
        capped.max_segments_per_cell = 1;
        let canonical_sources = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let overloaded = [
            TemporalSegmentInit {
                owner: 11,
                sources: canonical_sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 11,
                sources: canonical_sources,
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
        ];
        assert!(matches!(
            TemporalCircuitRuntime::new(
                capped,
                two_token_codebook(),
                base_members(capped),
                vec![0.0; physical_count(capped)],
                &overloaded,
            ),
            Err(TemporalRuntimeBuildError::Invalid(
                "initial owner load exceeds max_segments_per_cell"
            ))
        ));

        let mut has_uncommitted_reserve = circuit_config;
        has_uncommitted_reserve.max_pyramidal_cells += 1;
        let uncommitted_source = [TemporalSegmentInit {
            owner: 11,
            sources: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 22],
            permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
        }];
        assert!(matches!(
            TemporalCircuitRuntime::new(
                has_uncommitted_reserve,
                two_token_codebook(),
                base_members(has_uncommitted_reserve),
                vec![0.0; physical_count(has_uncommitted_reserve)],
                &uncommitted_source,
            ),
            Err(TemporalRuntimeBuildError::Invalid(
                "bound-winner segment source is not a committed column cell"
            ))
        ));
    }

    #[test]
    fn plastic_capacity_and_learning_overflow_stops_are_exact_and_atomic() {
        let mut capacity_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        capacity_config.novel_race_cap = 8;
        capacity_config.segment_capacity = 1;
        let mut capacity_runtime = runtime_with_offsets(
            capacity_config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &[],
        );
        capacity_runtime.step(0, TemporalStepMode::Learn).unwrap();
        capacity_runtime.learning_step = u64::MAX;
        let context = capacity_runtime.previous_context;
        let event = capacity_runtime.event_count;
        let learning = capacity_runtime.learning_step;
        assert_eq!(
            capacity_runtime.step(1, TemporalStepMode::Learn),
            Err(TemporalCircuitStop::GlobalSegmentCapacity {
                required: 11,
                capacity: 1,
            }),
        );
        assert_eq!(capacity_runtime.previous_context, context);
        assert_eq!(capacity_runtime.event_count, event);
        assert_eq!(capacity_runtime.learning_step, learning);
        assert_eq!(capacity_runtime.store.runtime_segment_count(), 0);
        assert_eq!(
            capacity_runtime.scientific_stop,
            Some(TemporalPoisonCode::GlobalSegmentCapacity),
        );

        let mut overflow_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        overflow_config.novel_race_cap = 8;
        let mut overflow_runtime = runtime_with_offsets(
            overflow_config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &[],
        );
        overflow_runtime.learning_step = u64::MAX;
        assert_eq!(
            overflow_runtime.step(0, TemporalStepMode::Learn),
            Err(TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::LearningCounter,
                location: 0,
            }),
        );
        assert_eq!(overflow_runtime.previous_context_len, 0);
        assert_eq!(overflow_runtime.previous_winner_context_len, 0);
        assert_eq!(overflow_runtime.event_count, 0);
        assert_eq!(overflow_runtime.learning_step, u64::MAX);
        assert_eq!(
            overflow_runtime.scientific_stop,
            Some(TemporalPoisonCode::TickOrCounterOverflow),
        );

        let mut ordered_runtime = runtime_with_offsets(
            overflow_config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &[],
        );
        ordered_runtime.event_count = u64::MAX;
        ordered_runtime.learning_step = u64::MAX;
        assert_eq!(
            ordered_runtime.step(0, TemporalStepMode::Learn),
            Err(TemporalCircuitStop::TickOrCounterOverflow {
                stage: TemporalOverflowStage::EventCounter,
                location: 0,
            }),
            "event overflow is preflighted after physics and before learning overflow",
        );
        assert_eq!(ordered_runtime.event_count, u64::MAX);
        assert_eq!(ordered_runtime.learning_step, u64::MAX);
    }

    #[test]
    fn reserved_or_unknown_token_is_retryable_before_any_learning_state_change() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        let mut runtime = runtime_with_offsets(
            circuit_config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &[],
        );

        assert_eq!(
            runtime.step(NONE, TemporalStepMode::Learn),
            Err(TemporalCircuitStop::UnknownToken(NONE)),
        );
        assert_eq!(runtime.scientific_stop, None);
        assert_eq!(runtime.event_count, 0);
        assert_eq!(runtime.learning_step, 0);
        assert_eq!(runtime.store.runtime_segment_count(), 0);
        assert_python_boundary_step(&runtime.step(0, TemporalStepMode::Learn).unwrap(), 1);
        assert_eq!(runtime.learning_step, 1);
    }

    #[test]
    fn read_only_and_growth_off_preserve_nonempty_learned_state_and_learning_address() {
        for (mode, plasticity) in [
            (
                TemporalStepMode::ReadOnly,
                TemporalPlasticityPolicy::Plastic,
            ),
            (TemporalStepMode::Learn, TemporalPlasticityPolicy::GrowthOff),
        ] {
            let mut runtime = python_oracle_runtime(255);
            runtime.config.plasticity_policy = plasticity;
            runtime.learning_step = 7;
            let before: Vec<_> = (0..runtime.store.runtime_segment_count())
                .map(|segment_id| runtime.store.runtime_segment_row(segment_id).unwrap())
                .collect();
            runtime.step(0, mode).unwrap();
            runtime.step(1, mode).unwrap();
            assert_eq!(runtime.learning_step, 7);
            assert_eq!(
                (0..runtime.store.runtime_segment_count())
                    .map(|segment_id| runtime.store.runtime_segment_row(segment_id).unwrap())
                    .collect::<Vec<_>>(),
                before,
            );
        }
    }

    #[test]
    fn plastic_success_and_capacity_stop_allocate_nothing_and_keep_runtime_buffers_stable() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        let mut runtime = runtime_with_offsets(
            circuit_config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &[],
        );
        let pointers = (
            runtime.token_codebook.as_ptr(),
            runtime.column_offsets.as_ptr(),
            runtime.column_cells.as_ptr(),
            runtime.threshold_offsets_mv.as_ptr(),
            runtime.recognition_scratch.as_ptr(),
            runtime.active_segment_scratch.as_ptr(),
            runtime.column_drive_scratch.as_ptr(),
            runtime.novel_candidate_scratch.as_ptr(),
        );
        let capacities = (
            runtime.token_codebook.capacity(),
            runtime.column_offsets.capacity(),
            runtime.column_cells.capacity(),
            runtime.threshold_offsets_mv.capacity(),
            runtime.recognition_scratch.capacity(),
            runtime.active_segment_scratch.capacity(),
            runtime.column_drive_scratch.capacity(),
            runtime.novel_candidate_scratch.capacity(),
        );
        let (trace, allocations) = count_allocations(|| {
            [
                runtime.step(0, TemporalStepMode::Learn),
                runtime.step(1, TemporalStepMode::Learn),
                runtime.step(0, TemporalStepMode::Learn),
                runtime.step(1, TemporalStepMode::Learn),
            ]
        });
        assert!(trace.iter().all(Result::is_ok), "{trace:?}");
        assert_eq!(allocations, 0);
        assert_eq!(
            (
                runtime.token_codebook.as_ptr(),
                runtime.column_offsets.as_ptr(),
                runtime.column_cells.as_ptr(),
                runtime.threshold_offsets_mv.as_ptr(),
                runtime.recognition_scratch.as_ptr(),
                runtime.active_segment_scratch.as_ptr(),
                runtime.column_drive_scratch.as_ptr(),
                runtime.novel_candidate_scratch.as_ptr(),
            ),
            pointers,
        );
        assert_eq!(
            (
                runtime.token_codebook.capacity(),
                runtime.column_offsets.capacity(),
                runtime.column_cells.capacity(),
                runtime.threshold_offsets_mv.capacity(),
                runtime.recognition_scratch.capacity(),
                runtime.active_segment_scratch.capacity(),
                runtime.column_drive_scratch.capacity(),
                runtime.novel_candidate_scratch.capacity(),
            ),
            capacities,
        );

        let mut stop_config = circuit_config;
        stop_config.segment_capacity = 1;
        let mut stopped = runtime_with_offsets(
            stop_config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &[],
        );
        stopped.step(0, TemporalStepMode::Learn).unwrap();
        let (stop, stop_allocations) =
            count_allocations(|| stopped.step(1, TemporalStepMode::Learn));
        assert!(matches!(
            stop,
            Err(TemporalCircuitStop::GlobalSegmentCapacity { .. })
        ));
        assert_eq!(stop_allocations, 0);
        let (poisoned, poisoned_allocations) =
            count_allocations(|| stopped.step(1, TemporalStepMode::Learn));
        assert_eq!(poisoned, Err(TemporalCircuitStop::Poisoned));
        assert_eq!(poisoned_allocations, 0);
    }

    #[test]
    fn plastic_multi_owner_race_reinforces_the_winner_and_punishes_the_loser() {
        let circuit_config = config(22, 2, TemporalContextPolicy::BoundWinners);
        let sources = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
        let segments = [
            TemporalSegmentInit {
                owner: 23,
                sources: sources.clone(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
            TemporalSegmentInit {
                owner: 22,
                sources,
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            },
        ];
        let mut offsets = vec![0.0; physical_count(circuit_config)];
        offsets[22] = -0.024_791_163_874_911_56;
        offsets[23] = 0.033_505_381_138_863_34;
        let mut runtime = runtime_with_offsets(circuit_config, offsets, &segments);
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        let learned = runtime.step(1, TemporalStepMode::Learn).unwrap();
        assert_eq!(learned.winner_spikes[0].cell_id, 23);
        assert_eq!(learned.punished_segments, 1);
        assert_eq!(
            runtime
                .store
                .runtime_segment_row(0)
                .unwrap()
                .permanence_ticks,
            [305; PAIRS_PER_SEGMENT],
        );
        assert_eq!(
            runtime
                .store
                .runtime_segment_row(1)
                .unwrap()
                .permanence_ticks,
            [253; PAIRS_PER_SEGMENT],
        );
        assert_eq!(runtime.scientific_stop, None);
    }

    #[test]
    fn malformed_paired_context_stops_before_read_only_physics_or_state_advance() {
        let mut runtime = python_oracle_runtime(255);
        runtime.recognize(0).unwrap();
        runtime.previous_context[ACTIVE_COLUMNS] += 1;
        let before_context = runtime.previous_context;
        let before_winner_context = runtime.previous_winner_context;
        let before_event = runtime.event_count;

        assert_eq!(
            runtime.step(1, TemporalStepMode::ReadOnly),
            Err(TemporalCircuitStop::MalformedContextOrPolicy(
                "context does not contain injective source/relay pairs"
            )),
        );
        assert_eq!(runtime.previous_context, before_context);
        assert_eq!(runtime.previous_winner_context, before_winner_context);
        assert_eq!(runtime.event_count, before_event);
        assert_eq!(runtime.learning_step, 0);
        assert_eq!(
            runtime.scientific_stop,
            Some(TemporalPoisonCode::MalformedContextOrPolicy),
        );
    }

    #[test]
    fn generic_afferent_learning_stores_normalized_sources_and_replays_active_rows() {
        let circuit_config = config(22, 2, TemporalContextPolicy::GenericAfferents);
        let mut runtime = runtime_with_offsets(
            circuit_config,
            vec![0.0; physical_count(circuit_config)],
            &[],
        );
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        runtime.step(1, TemporalStepMode::Learn).unwrap();
        let allocated = runtime.store.runtime_segment_row(0).unwrap();
        assert_eq!(
            allocated.normalized_sources,
            [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
        assert_eq!(allocated.permanence_ticks, [255; PAIRS_PER_SEGMENT]);

        runtime.step(0, TemporalStepMode::Learn).unwrap();
        let replay = runtime.step(1, TemporalStepMode::Learn).unwrap();
        assert_eq!(replay.predicted_columns, ACTIVE_COLUMNS as u8);
        assert_eq!(replay.allocated_segments, 0);
        assert_eq!(
            runtime
                .store
                .runtime_segment_row(0)
                .unwrap()
                .permanence_ticks,
            [305; PAIRS_PER_SEGMENT],
        );
        assert_eq!(runtime.learning_step, 4);
    }

    #[test]
    fn punishment_can_drop_every_pair_without_freeing_the_segment_or_owner_load() {
        let mut circuit_config = config(22, 1, TemporalContextPolicy::BoundWinners);
        circuit_config.novel_race_cap = 8;
        circuit_config.negative_decrement_ticks = 250;
        let segment = TemporalSegmentInit {
            owner: 0,
            sources: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            permanence_ticks: vec![250; PAIRS_PER_SEGMENT],
        };
        let mut runtime = runtime_with_offsets(
            circuit_config,
            PYTHON_OFFSET_BITS.map(f64::from_bits).to_vec(),
            &[segment],
        );
        runtime.step(0, TemporalStepMode::Learn).unwrap();
        let target = runtime.step(1, TemporalStepMode::Learn).unwrap();
        assert_eq!(target.punished_segments, 1);
        let empty = runtime.store.runtime_segment_row(0).unwrap();
        assert_eq!(empty.live_pair_count, 0);
        assert_eq!(empty.normalized_sources, [NONE; PAIRS_PER_SEGMENT]);
        assert_eq!(empty.permanence_ticks, [0; PAIRS_PER_SEGMENT]);
        assert_eq!(runtime.store.runtime_owner_load(0), Some(1));
        assert_eq!(runtime.store.runtime_segment_count(), 12);
    }
}
