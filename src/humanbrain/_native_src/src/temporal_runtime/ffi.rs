//! Unsafe C transport for the native temporal-circuit runtime.
//!
//! This is an internal, additive ABI beneath the future Python capability
//! layer. In particular, the two raw digest arrays carried here are transport
//! bytes only: they are not proof that a protocol is runnable, committed, or
//! live. The public Python layer is solely responsible for performing that
//! validation before it passes an authority to these functions.
//!
//! # Unsafe caller contract
//!
//! Every fixed object and counted region must be live for the duration of its
//! call and use the declared layout. All handle, request, definition input,
//! counted input, output, authority, and report regions must be pairwise
//! disjoint. A handle is destroyed exactly once, never concurrently with any
//! other operation. These exports validate nullness, alignment, integer
//! geometry, tags, reserved bytes, and output capacity, but cannot prove
//! foreign allocation provenance or lifetime. Mutex poisoning aborts the
//! process: unwinding may have interrupted a mutation, and release builds use
//! panic=abort, so continuing would make a scientific-state claim we cannot
//! support.

use std::{
    alloc::{alloc, dealloc, Layout},
    io::{self, Seek, SeekFrom, Write},
    mem::{align_of, size_of},
    ptr, slice,
    sync::{Mutex, MutexGuard},
};

#[cfg(test)]
use std::cell::Cell;

use super::{
    checkpoint::{
        import_canonical_empty_checkpoint_v1, import_checkpoint_v1, CheckpointV1ImportError,
        PrepareCheckpointV1ExportError, ValidatedCheckpointAuthority,
        ValidatedCheckpointV1RuntimeFactory, CHECKPOINT_V1_HEADER_BYTES,
    },
    checkpoint_publisher::{
        prepare_durable_directory_v1, prepare_immutable_bytes_v1, publish_prepared_checkpoint_v1,
        publish_prepared_immutable_bytes_v1, reconcile_existing_immutable_bytes_v1,
        validate_destination_v1, validate_durable_directory_v1, DestinationInputError,
        ImmutablePayloadInputError, PathDurabilityCompletion, PublicationCompletion,
        PublicationError, CLEANUP_CLEAN, CLEANUP_NOT_APPLICABLE, CLEANUP_ORPHANED,
        DURABILITY_OUTCOME_DURABLE, DURABILITY_OUTCOME_NONE, DURABILITY_OUTCOME_NOT_DURABLE,
        ERROR_DOMAIN_ENCODER, ERROR_DOMAIN_IO_WITHOUT_OS_CODE, ERROR_DOMAIN_NONE,
        ERROR_DOMAIN_POLICY, ERROR_DOMAIN_POSIX, ERROR_DOMAIN_WIN32, ERROR_STAGE_ENCODE,
        ERROR_STAGE_EXCLUSIVE_PUBLISH, ERROR_STAGE_FILESYSTEM_POLICY, ERROR_STAGE_FILE_CLOSE,
        ERROR_STAGE_FILE_SYNC, ERROR_STAGE_NONE, ERROR_STAGE_PARENT_IDENTITY,
        ERROR_STAGE_PARENT_OPEN, ERROR_STAGE_PARENT_SYNC, ERROR_STAGE_TEMP_CLEANUP,
        ERROR_STAGE_TEMP_CREATE, ERROR_STAGE_TEMP_IDENTITY, MAX_IMMUTABLE_PAYLOAD_BYTES,
        OUTCOME_COLLISION, OUTCOME_DURABLE_PUBLISHED, OUTCOME_NONE, OUTCOME_NOT_PUBLISHED,
        OUTCOME_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE,
    },
    typed_hole_state::TypedValue,
    typed_workspace::{TypedWorkspaceVocabulary, TYPED_CUE_TOKEN_COUNT, TYPED_SURFACE_TOKEN_COUNT},
    PhysicalStage, TemporalCircuitConfig, TemporalCircuitRuntime, TemporalCircuitStep,
    TemporalCircuitStop, TemporalContextPolicy, TemporalGeneration, TemporalOverflowStage,
    TemporalPlasticityPolicy, TemporalPoisonCode, TemporalStepMode, TokenCode, ACTIVE_COLUMNS,
    CONTEXT_CELLS, MAX_TYPED_GENERATIONS,
};
use crate::temporal_physics::SpikeEvent;

#[cfg(test)]
use super::checkpoint_publisher::{
    fail_next_destination_allocation, fail_next_immutable_payload_allocation,
};

pub const TEMPORAL_RUNTIME_OK: u32 = 0;
pub const TEMPORAL_RUNTIME_INVALID_ARGUMENT: u32 = 1;
pub const TEMPORAL_RUNTIME_INVALID_DEFINITION: u32 = 2;
pub const TEMPORAL_RUNTIME_CHECKPOINT_REJECTED: u32 = 3;
pub const TEMPORAL_RUNTIME_ALLOCATION_FAILED: u32 = 4;
pub const TEMPORAL_RUNTIME_OUTPUT_CAPACITY: u32 = 5;
pub const TEMPORAL_RUNTIME_UNKNOWN_TOKEN: u32 = 6;
pub const TEMPORAL_RUNTIME_POISONED: u32 = 7;
pub const TEMPORAL_RUNTIME_SCIENTIFIC_STOP: u32 = 8;
pub const TEMPORAL_RUNTIME_CAPACITY_STOP: u32 = 9;
pub const TEMPORAL_RUNTIME_INTERNAL_INVARIANT: u32 = 10;

pub const TEMPORAL_CHECKPOINT_PUBLICATION_NONE: u16 = OUTCOME_NONE;
pub const TEMPORAL_CHECKPOINT_DURABLE_PUBLISHED: u16 = OUTCOME_DURABLE_PUBLISHED;
pub const TEMPORAL_CHECKPOINT_COLLISION: u16 = OUTCOME_COLLISION;
pub const TEMPORAL_CHECKPOINT_NOT_PUBLISHED: u16 = OUTCOME_NOT_PUBLISHED;
pub const TEMPORAL_CHECKPOINT_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE: u16 =
    OUTCOME_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE;

pub const TEMPORAL_CHECKPOINT_CLEANUP_NOT_APPLICABLE: u8 = CLEANUP_NOT_APPLICABLE;
pub const TEMPORAL_CHECKPOINT_CLEANUP_CLEAN: u8 = CLEANUP_CLEAN;
pub const TEMPORAL_CHECKPOINT_CLEANUP_ORPHANED: u8 = CLEANUP_ORPHANED;

pub const TEMPORAL_CHECKPOINT_ERROR_DOMAIN_NONE: u16 = ERROR_DOMAIN_NONE;
pub const TEMPORAL_CHECKPOINT_ERROR_DOMAIN_POSIX: u16 = ERROR_DOMAIN_POSIX;
pub const TEMPORAL_CHECKPOINT_ERROR_DOMAIN_WIN32: u16 = ERROR_DOMAIN_WIN32;
pub const TEMPORAL_CHECKPOINT_ERROR_DOMAIN_POLICY: u16 = ERROR_DOMAIN_POLICY;
pub const TEMPORAL_CHECKPOINT_ERROR_DOMAIN_ENCODER: u16 = ERROR_DOMAIN_ENCODER;
pub const TEMPORAL_CHECKPOINT_ERROR_DOMAIN_IO: u16 = ERROR_DOMAIN_IO_WITHOUT_OS_CODE;

pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_NONE: u16 = ERROR_STAGE_NONE;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_PARENT_OPEN: u16 = ERROR_STAGE_PARENT_OPEN;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_PARENT_IDENTITY: u16 = ERROR_STAGE_PARENT_IDENTITY;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_FILESYSTEM_POLICY: u16 = ERROR_STAGE_FILESYSTEM_POLICY;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_TEMP_CREATE: u16 = ERROR_STAGE_TEMP_CREATE;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_ENCODE: u16 = ERROR_STAGE_ENCODE;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_FILE_SYNC: u16 = ERROR_STAGE_FILE_SYNC;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_FILE_CLOSE: u16 = ERROR_STAGE_FILE_CLOSE;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_TEMP_IDENTITY: u16 = ERROR_STAGE_TEMP_IDENTITY;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_EXCLUSIVE_PUBLISH: u16 = ERROR_STAGE_EXCLUSIVE_PUBLISH;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_TEMP_CLEANUP: u16 = ERROR_STAGE_TEMP_CLEANUP;
pub const TEMPORAL_CHECKPOINT_ERROR_STAGE_PARENT_SYNC: u16 = ERROR_STAGE_PARENT_SYNC;

pub const TEMPORAL_DURABILITY_NONE: u16 = DURABILITY_OUTCOME_NONE;
pub const TEMPORAL_DURABILITY_DURABLE: u16 = DURABILITY_OUTCOME_DURABLE;
pub const TEMPORAL_DURABILITY_NOT_DURABLE: u16 = DURABILITY_OUTCOME_NOT_DURABLE;

const STAGE_NONE: u16 = 0;
const STAGE_PROXIMAL: u16 = 1;
const STAGE_PROXIMAL_RELAY: u16 = 2;
const STAGE_COLUMN_RACE: u16 = 3;
const STAGE_WINNER_RELAY: u16 = 4;
const STAGE_PROXIMAL_RELAY_TICK_OVERFLOW: u16 = 5;
const STAGE_WINNER_RELAY_TICK_OVERFLOW: u16 = 6;
const STAGE_EVENT_COUNTER_OVERFLOW: u16 = 7;
const STAGE_LEARNING_COUNTER_OVERFLOW: u16 = 8;

const MODE_READ_ONLY: u8 = 0;
const MODE_LEARN: u8 = 1;
const CONTEXT_BOUND_WINNERS: u8 = 1;
const CONTEXT_GENERIC_AFFERENTS: u8 = 2;
const PLASTICITY_GROWTH_OFF: u8 = 0;
const PLASTICITY_PLASTIC: u8 = 1;
const TYPED_VALUE_INT: u8 = 0;
const TYPED_VALUE_LIST_INT: u8 = 1;
const TYPED_VALUE_INT_TO_INT: u8 = 2;
const TYPED_VALUE_BOOL: u8 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalValidatedCheckpointAuthorityV1 {
    pub protocol_identity_sha256: [u8; 32],
    pub source_bundle_identity_sha256: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalCheckpointIdentityV1 {
    pub header: [u8; CHECKPOINT_V1_HEADER_BYTES as usize],
}

impl TemporalCheckpointIdentityV1 {
    const ZERO: Self = Self {
        header: [0; CHECKPOINT_V1_HEADER_BYTES as usize],
    };
}

impl TemporalValidatedCheckpointAuthorityV1 {
    const fn into_validated(self) -> ValidatedCheckpointAuthority {
        ValidatedCheckpointAuthority::from_raw_sha256(
            self.protocol_identity_sha256,
            self.source_bundle_identity_sha256,
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalCircuitConfigV1 {
    pub column_count: u32,
    pub max_pyramidal_cells: u32,
    pub segment_capacity: u32,
    pub construction_seed: u32,
    pub connected_permanence_ticks: u16,
    pub initial_permanence_ticks: u16,
    pub permanence_increment_ticks: u16,
    pub negative_decrement_ticks: u16,
    pub base_cells_per_column: u8,
    pub novel_race_cap: u8,
    pub prediction_quorum: u8,
    pub max_segments_per_cell: u8,
    pub theta_match: u8,
    pub theta_active: u8,
    pub context_policy: u8,
    pub plasticity_policy: u8,
}

impl TemporalCircuitConfigV1 {
    fn into_runtime(self) -> Result<TemporalCircuitConfig, ()> {
        let context_policy = match self.context_policy {
            CONTEXT_BOUND_WINNERS => TemporalContextPolicy::BoundWinners,
            CONTEXT_GENERIC_AFFERENTS => TemporalContextPolicy::GenericAfferents,
            _ => return Err(()),
        };
        let plasticity_policy = match self.plasticity_policy {
            PLASTICITY_GROWTH_OFF => TemporalPlasticityPolicy::GrowthOff,
            PLASTICITY_PLASTIC => TemporalPlasticityPolicy::Plastic,
            _ => return Err(()),
        };
        Ok(TemporalCircuitConfig {
            column_count: self.column_count,
            base_cells_per_column: self.base_cells_per_column,
            max_pyramidal_cells: self.max_pyramidal_cells,
            segment_capacity: self.segment_capacity,
            novel_race_cap: self.novel_race_cap,
            prediction_quorum: self.prediction_quorum,
            construction_seed: self.construction_seed,
            connected_permanence_ticks: self.connected_permanence_ticks,
            initial_permanence_ticks: self.initial_permanence_ticks,
            permanence_increment_ticks: self.permanence_increment_ticks,
            negative_decrement_ticks: self.negative_decrement_ticks,
            max_segments_per_cell: self.max_segments_per_cell,
            theta_match: self.theta_match,
            theta_active: self.theta_active,
            context_policy,
            plasticity_policy,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalTokenCodeV1 {
    pub token_id: u32,
    pub columns: [u32; ACTIVE_COLUMNS],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalCircuitDefinitionV1 {
    pub config: TemporalCircuitConfigV1,
    pub token_codes: *const TemporalTokenCodeV1,
    pub token_code_count: u64,
    pub threshold_offsets_mv: *const f64,
    pub threshold_offset_count: u64,
    pub authority: TemporalValidatedCheckpointAuthorityV1,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalCircuitDefinitionIdentityV1Request {
    pub config: TemporalCircuitConfigV1,
    pub token_codes: *const TemporalTokenCodeV1,
    pub token_code_count: u64,
    pub threshold_offsets_mv: *const f64,
    pub threshold_offset_count: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalCircuitDefinitionIdentityV1 {
    pub circuit_identity_sha256: [u8; 32],
}

impl TemporalCircuitDefinitionIdentityV1 {
    const ZERO: Self = Self {
        circuit_identity_sha256: [0; 32],
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalCircuitImportCheckpointV1Request {
    pub definition: TemporalCircuitDefinitionV1,
    pub checkpoint_bytes: *const u8,
    pub checkpoint_len: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalCheckpointPathV1Request {
    pub parent_utf8: *const u8,
    pub parent_len: u64,
    pub final_basename_utf8: *const u8,
    pub final_basename_len: u64,
    pub reserved_zero: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalImmutableBytesPathV1Request {
    pub parent_utf8: *const u8,
    pub parent_len: u64,
    pub final_basename_utf8: *const u8,
    pub final_basename_len: u64,
    pub payload: *const u8,
    pub payload_len: u64,
    pub reserved_zero: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalDurableDirectoryPathV1Request {
    pub root_utf8: *const u8,
    pub root_len: u64,
    pub relative_directory_utf8: *const u8,
    pub relative_directory_len: u64,
    pub reserved_zero: [u8; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalCheckpointPublicationErrorV1 {
    pub stage: u16,
    pub domain: u16,
    pub code: u32,
}

impl TemporalCheckpointPublicationErrorV1 {
    const ZERO: Self = Self {
        stage: TEMPORAL_CHECKPOINT_ERROR_STAGE_NONE,
        domain: TEMPORAL_CHECKPOINT_ERROR_DOMAIN_NONE,
        code: 0,
    };
}

impl From<PublicationError> for TemporalCheckpointPublicationErrorV1 {
    fn from(error: PublicationError) -> Self {
        Self {
            stage: error.stage,
            domain: error.domain,
            code: error.code,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalCheckpointPublicationReportV1 {
    pub outcome: u16,
    pub temp_cleanup: u8,
    pub cleanup_error_count: u8,
    pub primary_error: TemporalCheckpointPublicationErrorV1,
    pub cleanup_errors: [TemporalCheckpointPublicationErrorV1; 4],
    pub owned_temp_pid: u32,
    pub owned_temp_nonce: u64,
    /// Exact payload length. Checkpoint calls provide the canonical length
    /// prepared under the runtime lock; immutable-byte calls provide the
    /// bounded owned-copy length.
    pub checkpoint_bytes: u64,
    /// Best-effort temporary-file length, capped at `checkpoint_bytes`.
    /// Zero means no trustworthy observation, not necessarily zero bytes.
    pub encoded_bytes: u64,
    pub reserved_zero: [u8; 8],
}

impl TemporalCheckpointPublicationReportV1 {
    const ZERO: Self = Self {
        outcome: TEMPORAL_CHECKPOINT_PUBLICATION_NONE,
        temp_cleanup: TEMPORAL_CHECKPOINT_CLEANUP_NOT_APPLICABLE,
        cleanup_error_count: 0,
        primary_error: TemporalCheckpointPublicationErrorV1::ZERO,
        cleanup_errors: [TemporalCheckpointPublicationErrorV1::ZERO; 4],
        owned_temp_pid: 0,
        owned_temp_nonce: 0,
        checkpoint_bytes: 0,
        encoded_bytes: 0,
        reserved_zero: [0; 8],
    };

    fn fill(&mut self, completion: PublicationCompletion) {
        self.outcome = completion.outcome;
        self.temp_cleanup = completion.temp_cleanup;
        self.cleanup_error_count = completion.cleanup_errors.len();
        self.primary_error = completion.primary_error.into();
        self.cleanup_errors = completion.cleanup_errors.errors().map(Into::into);
        self.owned_temp_pid = completion.owned_temp_pid;
        self.owned_temp_nonce = completion.owned_temp_nonce;
        // Preserve the committed checkpoint report ABI: generic payload and
        // write lengths occupy the historical length fields on the wire.
        self.checkpoint_bytes = completion.payload_bytes;
        self.encoded_bytes = completion.written_bytes;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalImmutableBytesReconciliationReportV1 {
    pub outcome: u16,
    pub cleanup_error_count: u8,
    pub reserved_zero_u8: u8,
    pub primary_error: TemporalCheckpointPublicationErrorV1,
    pub cleanup_errors: [TemporalCheckpointPublicationErrorV1; 2],
    pub reserved_zero_u32: u32,
    pub required_bytes: u64,
    pub verified_bytes: u64,
    pub reserved_zero: [u8; 16],
}

impl TemporalImmutableBytesReconciliationReportV1 {
    const ZERO: Self = Self {
        outcome: TEMPORAL_DURABILITY_NONE,
        cleanup_error_count: 0,
        reserved_zero_u8: 0,
        primary_error: TemporalCheckpointPublicationErrorV1::ZERO,
        cleanup_errors: [TemporalCheckpointPublicationErrorV1::ZERO; 2],
        reserved_zero_u32: 0,
        required_bytes: 0,
        verified_bytes: 0,
        reserved_zero: [0; 16],
    };

    fn fill(&mut self, completion: PathDurabilityCompletion) {
        self.outcome = completion.outcome;
        self.cleanup_error_count = completion.cleanup_errors.len();
        self.primary_error = completion.primary_error.into();
        self.cleanup_errors = completion.cleanup_errors.errors().map(Into::into);
        self.required_bytes = completion.expected_bytes;
        self.verified_bytes = completion.verified_bytes;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalDurableDirectoryPreparationReportV1 {
    pub outcome: u16,
    pub cleanup_error_count: u8,
    pub reserved_zero_u8: u8,
    pub primary_error: TemporalCheckpointPublicationErrorV1,
    pub cleanup_errors: [TemporalCheckpointPublicationErrorV1; 2],
    pub required_component_count: u32,
    pub confirmed_component_count: u32,
    pub created_component_count: u32,
    pub failed_component_index: u32,
    pub reserved_zero_u32: u32,
    pub reserved_zero: [u8; 16],
}

impl TemporalDurableDirectoryPreparationReportV1 {
    const ZERO: Self = Self {
        outcome: TEMPORAL_DURABILITY_NONE,
        cleanup_error_count: 0,
        reserved_zero_u8: 0,
        primary_error: TemporalCheckpointPublicationErrorV1::ZERO,
        cleanup_errors: [TemporalCheckpointPublicationErrorV1::ZERO; 2],
        required_component_count: 0,
        confirmed_component_count: 0,
        created_component_count: 0,
        failed_component_index: 0,
        reserved_zero_u32: 0,
        reserved_zero: [0; 16],
    };

    fn fill(&mut self, completion: PathDurabilityCompletion) {
        self.outcome = completion.outcome;
        self.cleanup_error_count = completion.cleanup_errors.len();
        self.primary_error = completion.primary_error.into();
        self.cleanup_errors = completion.cleanup_errors.errors().map(Into::into);
        self.required_component_count = completion.component_count;
        self.confirmed_component_count = completion.completed_component_count;
        self.created_component_count = completion.created_component_count;
        self.failed_component_index = completion.failed_component_index;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalSpikeV1 {
    pub cell_id: u32,
    pub tick: u16,
    pub reserved_zero: u16,
}

impl From<SpikeEvent> for TemporalSpikeV1 {
    fn from(spike: SpikeEvent) -> Self {
        Self {
            cell_id: spike.cell_id,
            tick: spike.tick,
            reserved_zero: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalCircuitStepV2 {
    pub token_id: u32,
    pub punished_segments: u32,
    pub event_count: u64,
    pub candidate_segments: u64,
    pub matching_segments: u64,
    pub active_segments: u64,
    pub postings_visited: u64,
    pub predicted: u8,
    pub predicted_columns: u8,
    pub allocated_segments: u8,
    pub reserved_zero: u8,
    pub inhibition_onset_ticks: [u16; ACTIVE_COLUMNS],
    pub column_simulated_ticks: [u16; ACTIVE_COLUMNS],
    pub winner_spikes: [TemporalSpikeV1; ACTIVE_COLUMNS],
    pub basket_spikes: [TemporalSpikeV1; ACTIVE_COLUMNS],
    pub proximal_spikes: [TemporalSpikeV1; ACTIVE_COLUMNS],
    pub proximal_relay_spikes: [TemporalSpikeV1; ACTIVE_COLUMNS],
    pub winner_relay_spikes: [TemporalSpikeV1; ACTIVE_COLUMNS],
    pub winner_context: [u32; CONTEXT_CELLS],
    pub proximal_context: [u32; CONTEXT_CELLS],
    pub context: [u32; CONTEXT_CELLS],
}

impl From<TemporalCircuitStep> for TemporalCircuitStepV2 {
    fn from(step: TemporalCircuitStep) -> Self {
        Self {
            token_id: step.token_id,
            punished_segments: step.punished_segments,
            event_count: step.event_count,
            candidate_segments: step.candidate_segments as u64,
            matching_segments: step.matching_segments as u64,
            active_segments: step.active_segments as u64,
            postings_visited: step.postings_visited,
            predicted: u8::from(step.predicted),
            predicted_columns: step.predicted_columns,
            allocated_segments: step.allocated_segments,
            reserved_zero: 0,
            inhibition_onset_ticks: step.inhibition_onset_ticks,
            column_simulated_ticks: step.column_simulated_ticks,
            winner_spikes: step.winner_spikes.map(Into::into),
            basket_spikes: step.basket_spikes.map(Into::into),
            proximal_spikes: step.proximal_spikes.map(Into::into),
            proximal_relay_spikes: step.proximal_relay_spikes.map(Into::into),
            winner_relay_spikes: step.winner_relay_spikes.map(Into::into),
            winner_context: step.winner_context,
            proximal_context: step.proximal_context,
            context: step.context,
        }
    }
}

/// One fully committed surface generation transaction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalGenerationV1 {
    pub surface_token_id: u32,
    pub surface_output_cell_id: u32,
    pub generation_index: u64,
    pub temporal_event_count: u64,
    pub activated_columns: u32,
    pub candidate_tokens: u32,
    pub control_token_id: u32,
    pub control_output_cell_id: u32,
    pub surface_winner_tick: u16,
    pub surface_basket_tick: u16,
    pub surface_inhibition_onset_tick: u16,
    pub total_simulated_ticks: u16,
    pub control_winner_tick: u16,
    pub control_basket_tick: u16,
    pub control_simulated_end_tick: u16,
    pub path: u8,
    pub winner_input_columns: u8,
    pub feedback_predicted_columns: u8,
    pub feedback_predicted: u8,
    pub copy_place_ordinal: u8,
    pub physical_election_count: u8,
    pub slot_ordinal: u8,
    pub reserved_zero: [u8; 3],
}

impl TemporalGenerationV1 {
    const ZERO: Self = Self {
        surface_token_id: 0,
        surface_output_cell_id: 0,
        generation_index: 0,
        temporal_event_count: 0,
        activated_columns: 0,
        candidate_tokens: 0,
        control_token_id: 0,
        control_output_cell_id: 0,
        surface_winner_tick: 0,
        surface_basket_tick: 0,
        surface_inhibition_onset_tick: 0,
        total_simulated_ticks: 0,
        control_winner_tick: 0,
        control_basket_tick: 0,
        control_simulated_end_tick: 0,
        path: 0,
        winner_input_columns: 0,
        feedback_predicted_columns: 0,
        feedback_predicted: 0,
        copy_place_ordinal: 0,
        physical_election_count: 0,
        slot_ordinal: 0,
        reserved_zero: [0; 3],
    };
}

impl From<TemporalGeneration> for TemporalGenerationV1 {
    fn from(generation: TemporalGeneration) -> Self {
        Self {
            surface_token_id: generation.surface_token_id,
            surface_output_cell_id: generation.surface_output_cell_id,
            generation_index: generation.generation_index,
            temporal_event_count: generation.temporal_event_count,
            activated_columns: generation.activated_columns,
            candidate_tokens: generation.candidate_tokens,
            control_token_id: generation.control_token_id,
            control_output_cell_id: generation.control_output_cell_id,
            surface_winner_tick: generation.surface_winner_tick,
            surface_basket_tick: generation.surface_basket_tick,
            surface_inhibition_onset_tick: generation.surface_inhibition_onset_tick,
            total_simulated_ticks: generation.total_simulated_ticks,
            control_winner_tick: generation.control_winner_tick,
            control_basket_tick: generation.control_basket_tick,
            control_simulated_end_tick: generation.control_simulated_end_tick,
            path: generation.path as u8,
            winner_input_columns: generation.winner_input_columns,
            feedback_predicted_columns: generation.feedback_predicted_columns,
            feedback_predicted: u8::from(generation.feedback_predicted),
            copy_place_ordinal: generation.copy_place_ordinal,
            physical_election_count: generation.physical_election_count,
            slot_ordinal: generation.slot_ordinal,
            reserved_zero: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalTypedWorkspaceVocabularyV1 {
    pub plan_commit_token_id: u32,
    pub address_token_base: u32,
    pub cue_token_ids: [u32; TYPED_CUE_TOKEN_COUNT],
    pub surface_token_ids: [u32; TYPED_SURFACE_TOKEN_COUNT],
}

impl From<TemporalTypedWorkspaceVocabularyV1> for TypedWorkspaceVocabulary {
    fn from(vocabulary: TemporalTypedWorkspaceVocabularyV1) -> Self {
        Self {
            plan_commit_token_id: vocabulary.plan_commit_token_id,
            address_token_base: vocabulary.address_token_base,
            cue_token_ids: vocabulary.cue_token_ids,
            surface_token_ids: vocabulary.surface_token_ids,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalTypedSessionV1Request {
    pub vocabulary: TemporalTypedWorkspaceVocabularyV1,
    pub parameter_type: u8,
    pub result_type: u8,
    pub mode: u8,
    pub reserved_zero: [u8; 5],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalCircuitAdvanceBlockV2Request {
    pub token_ids: *const u32,
    pub output: *mut TemporalCircuitStepV2,
    pub token_count: u64,
    pub output_capacity: u64,
    pub mode: u8,
    pub reset_before: u8,
    pub reserved_zero: [u8; 6],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TemporalCircuitReportV1 {
    pub status: u32,
    pub message_len: u32,
    pub completed_prefix: u64,
    pub required: u64,
    pub observed: u64,
    pub location: u64,
    pub poison_code: u16,
    pub stage_code: u16,
    pub reserved_zero: u32,
    pub message_utf8: [u8; 464],
}

impl TemporalCircuitReportV1 {
    const ZERO: Self = Self {
        status: TEMPORAL_RUNTIME_OK,
        message_len: 0,
        completed_prefix: 0,
        required: 0,
        observed: 0,
        location: 0,
        poison_code: 0,
        stage_code: STAGE_NONE,
        reserved_zero: 0,
        message_utf8: [0; 464],
    };

    fn finish(&mut self, status: u32, message: &'static str) -> u32 {
        self.status = status;
        let bytes = message.as_bytes();
        let count = bytes.len().min(self.message_utf8.len());
        self.message_utf8[..count].copy_from_slice(&bytes[..count]);
        self.message_len = count as u32;
        status
    }
}

pub struct TemporalCircuitRuntimeHandle {
    creation_authority: ValidatedCheckpointAuthority,
    runtime: Mutex<TemporalCircuitRuntime>,
}

fn fixed_pointer_is_valid<T>(pointer: *const T) -> bool {
    !pointer.is_null() && (pointer as usize).is_multiple_of(align_of::<T>())
}

fn counted_region<T>(pointer: *const T, count: u64) -> Option<usize> {
    let count = usize::try_from(count).ok()?;
    let byte_count = count.checked_mul(size_of::<T>())?;
    if byte_count > isize::MAX as usize
        || (!pointer.is_null() && !(pointer as usize).is_multiple_of(align_of::<T>()))
        || (count != 0 && pointer.is_null())
        || (count != 0 && (pointer as usize).checked_add(byte_count).is_none())
    {
        return None;
    }
    Some(count)
}

fn regions_overlap(left: *const u8, left_len: usize, right: *const u8, right_len: usize) -> bool {
    if left_len == 0 || right_len == 0 {
        return false;
    }
    let left_start = left as usize;
    let right_start = right as usize;
    let Some(left_end) = left_start.checked_add(left_len) else {
        return true;
    };
    let Some(right_end) = right_start.checked_add(right_len) else {
        return true;
    };
    left_start < right_end && right_start < left_end
}

unsafe fn begin_report(report_out: *mut TemporalCircuitReportV1) -> bool {
    if !fixed_pointer_is_valid(report_out.cast_const()) {
        return false;
    }
    report_out.write(TemporalCircuitReportV1::ZERO);
    true
}

unsafe fn begin_publication_report(
    publication_out: *mut TemporalCheckpointPublicationReportV1,
) -> bool {
    if !fixed_pointer_is_valid(publication_out.cast_const()) {
        return false;
    }
    publication_out.write(TemporalCheckpointPublicationReportV1::ZERO);
    true
}

unsafe fn begin_reconciliation_report(
    durability_out: *mut TemporalImmutableBytesReconciliationReportV1,
) -> bool {
    if !fixed_pointer_is_valid(durability_out.cast_const()) {
        return false;
    }
    durability_out.write(TemporalImmutableBytesReconciliationReportV1::ZERO);
    true
}

unsafe fn begin_directory_preparation_report(
    durability_out: *mut TemporalDurableDirectoryPreparationReportV1,
) -> bool {
    if !fixed_pointer_is_valid(durability_out.cast_const()) {
        return false;
    }
    durability_out.write(TemporalDurableDirectoryPreparationReportV1::ZERO);
    true
}

unsafe fn report<'a>(report_out: *mut TemporalCircuitReportV1) -> &'a mut TemporalCircuitReportV1 {
    &mut *report_out
}

unsafe fn publication_report<'a>(
    publication_out: *mut TemporalCheckpointPublicationReportV1,
) -> &'a mut TemporalCheckpointPublicationReportV1 {
    &mut *publication_out
}

#[derive(Clone, Copy)]
struct RawRegion {
    pointer: *const u8,
    byte_count: usize,
}

fn raw_regions_are_disjoint(regions: &[RawRegion]) -> bool {
    for left_index in 0..regions.len() {
        for right_index in left_index + 1..regions.len() {
            let left = regions[left_index];
            let right = regions[right_index];
            if regions_overlap(
                left.pointer,
                left.byte_count,
                right.pointer,
                right.byte_count,
            ) {
                return false;
            }
        }
    }
    true
}

unsafe fn foreign_bytes<'a>(pointer: *const u8, count: usize) -> &'a [u8] {
    if count == 0 {
        &[]
    } else {
        slice::from_raw_parts(pointer, count)
    }
}

unsafe fn allocate_owned<T>(value: T) -> *mut T {
    #[cfg(test)]
    if FAIL_NEXT_OWNED_ALLOCATION.with(|fail| fail.replace(false)) {
        drop(value);
        return ptr::null_mut();
    }

    let raw = alloc(Layout::new::<T>()) as *mut T;
    if raw.is_null() {
        drop(value);
        ptr::null_mut()
    } else {
        raw.write(value);
        raw
    }
}

#[cfg(test)]
thread_local! {
    static FAIL_NEXT_OWNED_ALLOCATION: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
fn fail_next_owned_allocation() {
    FAIL_NEXT_OWNED_ALLOCATION.with(|fail| fail.set(true));
}

unsafe fn destroy_owned<T>(raw: *mut T) {
    raw.drop_in_place();
    dealloc(raw.cast(), Layout::new::<T>());
}

fn lock_runtime(handle: &TemporalCircuitRuntimeHandle) -> MutexGuard<'_, TemporalCircuitRuntime> {
    match handle.runtime.lock() {
        Ok(runtime) => runtime,
        Err(_) => std::process::abort(),
    }
}

fn try_copy_token_codes(
    source: *const TemporalTokenCodeV1,
    count: usize,
) -> Result<Vec<TokenCode>, ()> {
    let mut copied = Vec::new();
    copied.try_reserve_exact(count).map_err(|_| ())?;
    for index in 0..count {
        // SAFETY: the ABI preflight validated the complete counted input
        // region and the caller contract keeps it live and disjoint.
        let code = unsafe { source.add(index).read() };
        copied.push(TokenCode {
            token_id: code.token_id,
            columns: code.columns,
        });
    }
    Ok(copied)
}

fn try_copy_thresholds(source: *const f64, count: usize) -> Result<Vec<f64>, ()> {
    let mut copied = Vec::new();
    copied.try_reserve_exact(count).map_err(|_| ())?;
    for index in 0..count {
        // SAFETY: the ABI preflight validated the complete counted input
        // region and the caller contract keeps it live and disjoint.
        copied.push(unsafe { source.add(index).read() });
    }
    Ok(copied)
}

#[derive(Clone, Copy)]
struct DefinitionRegionCounts {
    token_codes: usize,
    threshold_offsets: usize,
}

fn preflight_definition_regions(
    token_codes: *const TemporalTokenCodeV1,
    token_code_count: u64,
    threshold_offsets_mv: *const f64,
    threshold_offset_count: u64,
    report: &mut TemporalCircuitReportV1,
) -> Result<DefinitionRegionCounts, u32> {
    let Some(token_codes) = counted_region(token_codes, token_code_count) else {
        return Err(report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "token-code input geometry is invalid",
        ));
    };
    let Some(threshold_offsets) = counted_region(threshold_offsets_mv, threshold_offset_count)
    else {
        return Err(report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "threshold input geometry is invalid",
        ));
    };
    Ok(DefinitionRegionCounts {
        token_codes,
        threshold_offsets,
    })
}

unsafe fn borrow_token_codes<'a>(
    source: *const TemporalTokenCodeV1,
    count: usize,
) -> &'a [TokenCode] {
    debug_assert_eq!(size_of::<TemporalTokenCodeV1>(), size_of::<TokenCode>());
    debug_assert_eq!(align_of::<TemporalTokenCodeV1>(), align_of::<TokenCode>());
    if count == 0 {
        &[]
    } else {
        slice::from_raw_parts(source.cast::<TokenCode>(), count)
    }
}

unsafe fn borrow_threshold_offsets<'a>(source: *const f64, count: usize) -> &'a [f64] {
    if count == 0 {
        &[]
    } else {
        slice::from_raw_parts(source, count)
    }
}

fn build_validated_factory(
    definition: TemporalCircuitDefinitionV1,
    counts: DefinitionRegionCounts,
    report: &mut TemporalCircuitReportV1,
) -> Result<
    (
        ValidatedCheckpointAuthority,
        ValidatedCheckpointV1RuntimeFactory,
    ),
    u32,
> {
    let config = definition.config.into_runtime().map_err(|()| {
        report.finish(
            TEMPORAL_RUNTIME_INVALID_DEFINITION,
            "circuit definition contains an invalid policy tag",
        )
    })?;
    let token_codes =
        try_copy_token_codes(definition.token_codes, counts.token_codes).map_err(|()| {
            report.finish(
                TEMPORAL_RUNTIME_ALLOCATION_FAILED,
                "token-code copy allocation failed",
            )
        })?;
    let threshold_offsets_mv =
        try_copy_thresholds(definition.threshold_offsets_mv, counts.threshold_offsets).map_err(
            |()| {
                report.finish(
                    TEMPORAL_RUNTIME_ALLOCATION_FAILED,
                    "threshold copy allocation failed",
                )
            },
        )?;
    let authority = definition.authority.into_validated();
    let factory = ValidatedCheckpointV1RuntimeFactory::new(
        authority,
        config,
        token_codes,
        threshold_offsets_mv,
    )
    .map_err(|_| {
        report.finish(
            TEMPORAL_RUNTIME_INVALID_DEFINITION,
            "circuit definition is not canonical",
        )
    })?;
    Ok((authority, factory))
}

unsafe fn validate_handle_output(
    handle_out: *mut *mut TemporalCircuitRuntimeHandle,
    report: &mut TemporalCircuitReportV1,
) -> Result<(), u32> {
    if !fixed_pointer_is_valid(handle_out.cast_const()) {
        return Err(report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "handle output is null or misaligned",
        ));
    }
    if !handle_out.read().is_null() {
        return Err(report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "handle output must be initialized to null",
        ));
    }
    Ok(())
}

unsafe fn publish_runtime_handle(
    runtime: TemporalCircuitRuntime,
    authority: ValidatedCheckpointAuthority,
    handle_out: *mut *mut TemporalCircuitRuntimeHandle,
    report: &mut TemporalCircuitReportV1,
) -> u32 {
    let handle = allocate_owned(TemporalCircuitRuntimeHandle {
        creation_authority: authority,
        runtime: Mutex::new(runtime),
    });
    if handle.is_null() {
        return report.finish(
            TEMPORAL_RUNTIME_ALLOCATION_FAILED,
            "runtime handle allocation failed",
        );
    }
    handle_out.write(handle);
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

fn checkpoint_import_error(
    report: &mut TemporalCircuitReportV1,
    error: CheckpointV1ImportError,
) -> u32 {
    match error {
        CheckpointV1ImportError::RetryableAllocation(_) => report.finish(
            TEMPORAL_RUNTIME_ALLOCATION_FAILED,
            "checkpoint import allocation failed",
        ),
        CheckpointV1ImportError::Pass1(_)
        | CheckpointV1ImportError::State
        | CheckpointV1ImportError::SemanticIndexMismatch
        | CheckpointV1ImportError::CanonicalReexportMismatch { .. } => report.finish(
            TEMPORAL_RUNTIME_CHECKPOINT_REJECTED,
            "checkpoint-v1 was rejected",
        ),
        CheckpointV1ImportError::FactoryIdentityMismatch
        | CheckpointV1ImportError::AssemblyInvariant => report.finish(
            TEMPORAL_RUNTIME_INTERNAL_INVARIANT,
            "validated checkpoint assembly invariant failed",
        ),
    }
}

fn canonical_empty_import_error(
    report: &mut TemporalCircuitReportV1,
    error: CheckpointV1ImportError,
) -> u32 {
    match error {
        CheckpointV1ImportError::RetryableAllocation(_) => report.finish(
            TEMPORAL_RUNTIME_ALLOCATION_FAILED,
            "fresh runtime allocation failed",
        ),
        CheckpointV1ImportError::Pass1(_)
        | CheckpointV1ImportError::FactoryIdentityMismatch
        | CheckpointV1ImportError::State
        | CheckpointV1ImportError::SemanticIndexMismatch
        | CheckpointV1ImportError::CanonicalReexportMismatch { .. }
        | CheckpointV1ImportError::AssemblyInvariant => report.finish(
            TEMPORAL_RUNTIME_INTERNAL_INVARIANT,
            "canonical empty checkpoint construction failed",
        ),
    }
}

fn prepare_error(
    report: &mut TemporalCircuitReportV1,
    error: PrepareCheckpointV1ExportError,
) -> u32 {
    match error {
        PrepareCheckpointV1ExportError::RetryableAllocation(_) => report.finish(
            TEMPORAL_RUNTIME_ALLOCATION_FAILED,
            "checkpoint export scratch allocation failed",
        ),
        PrepareCheckpointV1ExportError::Identity(_)
        | PrepareCheckpointV1ExportError::Codec(_)
        | PrepareCheckpointV1ExportError::State
        | PrepareCheckpointV1ExportError::StoreIndex(_) => report.finish(
            TEMPORAL_RUNTIME_INTERNAL_INVARIANT,
            "live runtime is not canonically exportable",
        ),
    }
}

fn poison_status(poison: TemporalPoisonCode) -> u32 {
    match poison {
        TemporalPoisonCode::Unpoisoned => TEMPORAL_RUNTIME_INTERNAL_INVARIANT,
        TemporalPoisonCode::MultipleActiveOwners
        | TemporalPoisonCode::MultipleMatchingOwners
        | TemporalPoisonCode::CategorizedRaceCapacity
        | TemporalPoisonCode::MalformedContextOrPolicy
        | TemporalPoisonCode::ProximalPhysics
        | TemporalPoisonCode::ProximalRelayPhysics
        | TemporalPoisonCode::ColumnRacePhysics
        | TemporalPoisonCode::WinnerRelayPhysics
        | TemporalPoisonCode::TickOrCounterOverflow
        | TemporalPoisonCode::StoreOrPlanInvariant
        | TemporalPoisonCode::PhiloxAddressExhaustion => TEMPORAL_RUNTIME_SCIENTIFIC_STOP,
        TemporalPoisonCode::ReserveCellCapacity
        | TemporalPoisonCode::GlobalSegmentCapacity
        | TemporalPoisonCode::OwnerSegmentCapacity
        | TemporalPoisonCode::PreallocatedCapacityInvariant => TEMPORAL_RUNTIME_CAPACITY_STOP,
    }
}

fn stop_report(
    report: &mut TemporalCircuitReportV1,
    runtime: &TemporalCircuitRuntime,
    stop: TemporalCircuitStop,
) -> u32 {
    match stop {
        TemporalCircuitStop::UnknownToken(token_id) => {
            report.observed = u64::from(token_id);
            report.finish(TEMPORAL_RUNTIME_UNKNOWN_TOKEN, "unknown token ID")
        }
        TemporalCircuitStop::Poisoned => {
            report.poison_code = runtime.scientific_stop.map_or(0, |code| code as u16);
            report.finish(
                TEMPORAL_RUNTIME_POISONED,
                "temporal runtime is already poisoned",
            )
        }
        TemporalCircuitStop::CategorizedRaceCapacity {
            column,
            categorized_count,
            race_cap,
        } => {
            report.location = u64::from(column);
            report.required = u64::from(categorized_count);
            report.observed = u64::from(race_cap);
            terminal_stop(report, stop, "categorized race capacity exceeded")
        }
        TemporalCircuitStop::TickOrCounterOverflow { stage, location } => {
            report.location = u64::from(location);
            report.stage_code = match stage {
                TemporalOverflowStage::ProximalRelayTick => STAGE_PROXIMAL_RELAY_TICK_OVERFLOW,
                TemporalOverflowStage::WinnerRelayTick => STAGE_WINNER_RELAY_TICK_OVERFLOW,
                TemporalOverflowStage::EventCounter => STAGE_EVENT_COUNTER_OVERFLOW,
                TemporalOverflowStage::LearningCounter => STAGE_LEARNING_COUNTER_OVERFLOW,
            };
            terminal_stop(report, stop, "temporal tick or counter overflow")
        }
        TemporalCircuitStop::Physics {
            stage, location, ..
        } => {
            report.location = u64::from(location);
            report.stage_code = match stage {
                PhysicalStage::Proximal => STAGE_PROXIMAL,
                PhysicalStage::ProximalRelay => STAGE_PROXIMAL_RELAY,
                PhysicalStage::ColumnRace => STAGE_COLUMN_RACE,
                PhysicalStage::WinnerRelay => STAGE_WINNER_RELAY,
            };
            terminal_stop(report, stop, "temporal physical stage stopped")
        }
        TemporalCircuitStop::StoreRecognition(observed) => {
            report.observed = u64::from(observed);
            terminal_stop(report, stop, "temporal store recognition invariant failed")
        }
        TemporalCircuitStop::StoreOrPlanInvariant(message)
        | TemporalCircuitStop::MalformedContextOrPolicy(message)
        | TemporalCircuitStop::PreallocatedCapacityInvariant(message) => {
            terminal_stop(report, stop, message)
        }
        TemporalCircuitStop::PhiloxAddressExhaustion { segment_id, .. } => {
            report.location = u64::from(segment_id);
            terminal_stop(report, stop, "Philox address domain exhausted")
        }
        TemporalCircuitStop::ReserveCellCapacity { column } => {
            report.location = u64::from(column);
            terminal_stop(report, stop, "reserve cell capacity exhausted")
        }
        TemporalCircuitStop::GlobalSegmentCapacity { required, capacity } => {
            report.required = u64::from(required);
            report.observed = u64::from(capacity);
            terminal_stop(report, stop, "global segment capacity exhausted")
        }
        TemporalCircuitStop::OwnerSegmentCapacity { owner, cap } => {
            report.location = u64::from(owner);
            report.required = u64::from(cap).saturating_add(1);
            report.observed = u64::from(cap);
            terminal_stop(report, stop, "owner segment capacity exhausted")
        }
    }
}

fn terminal_stop(
    report: &mut TemporalCircuitReportV1,
    stop: TemporalCircuitStop,
    message: &'static str,
) -> u32 {
    let Some(poison) = stop.poison_code() else {
        return report.finish(
            TEMPORAL_RUNTIME_INTERNAL_INVARIANT,
            "terminal STOP did not define a poison code",
        );
    };
    report.poison_code = poison as u16;
    report.finish(poison_status(poison), message)
}

struct FixedCheckpointSink {
    output: *mut u8,
    capacity: usize,
    position: usize,
    logical_len: usize,
}

impl FixedCheckpointSink {
    const fn new(output: *mut u8, capacity: usize) -> Self {
        Self {
            output,
            capacity,
            position: 0,
            logical_len: 0,
        }
    }

    const fn logical_len(&self) -> usize {
        self.logical_len
    }

    fn seek_target(&self, position: SeekFrom) -> io::Result<usize> {
        let target = match position {
            SeekFrom::Start(offset) => i128::from(offset),
            SeekFrom::End(offset) => self.logical_len as i128 + i128::from(offset),
            SeekFrom::Current(offset) => self.position as i128 + i128::from(offset),
        };
        if target < 0 || target > self.logical_len as i128 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        usize::try_from(target).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))
    }
}

impl Write for FixedCheckpointSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let end = self
            .position
            .checked_add(bytes.len())
            .filter(|end| *end <= self.capacity)
            .ok_or_else(|| io::Error::from(io::ErrorKind::WriteZero))?;
        if !bytes.is_empty() {
            // SAFETY: export preflight validated the complete raw output
            // region. This sink never reads or borrows caller output, and the
            // caller contract makes its source and destination disjoint.
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    self.output.add(self.position),
                    bytes.len(),
                );
            }
        }
        self.position = end;
        self.logical_len = self.logical_len.max(end);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Seek for FixedCheckpointSink {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let target = self.seek_target(position)?;
        self.position = target;
        u64::try_from(target).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))
    }
}

/// Return the route-0 circuit identity for one canonical immutable definition.
///
/// This query borrows the caller's definition arrays only for the duration of
/// the call. It does not construct a runtime, consume checkpoint authority, or
/// allocate native heap storage.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. `output` may be
/// uninitialized and is zeroed before definition validation.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_definition_identity_v1(
    request: *const TemporalCircuitDefinitionIdentityV1Request,
    output: *mut TemporalCircuitDefinitionIdentityV1,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(output.cast_const()) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "circuit identity output is null or misaligned",
        );
    }
    if regions_overlap(
        output.cast(),
        size_of::<TemporalCircuitDefinitionIdentityV1>(),
        report_out.cast(),
        size_of::<TemporalCircuitReportV1>(),
    ) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "circuit identity output and report overlap",
        );
    }
    output.write(TemporalCircuitDefinitionIdentityV1::ZERO);
    if !fixed_pointer_is_valid(request) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "circuit identity request is null or misaligned",
        );
    }
    let request_value = request.read();
    let counts = match preflight_definition_regions(
        request_value.token_codes,
        request_value.token_code_count,
        request_value.threshold_offsets_mv,
        request_value.threshold_offset_count,
        report,
    ) {
        Ok(counts) => counts,
        Err(status) => return status,
    };
    let regions = [
        RawRegion {
            pointer: request.cast(),
            byte_count: size_of::<TemporalCircuitDefinitionIdentityV1Request>(),
        },
        RawRegion {
            pointer: request_value.token_codes.cast(),
            byte_count: counts.token_codes * size_of::<TemporalTokenCodeV1>(),
        },
        RawRegion {
            pointer: request_value.threshold_offsets_mv.cast(),
            byte_count: counts.threshold_offsets * size_of::<f64>(),
        },
        RawRegion {
            pointer: output.cast(),
            byte_count: size_of::<TemporalCircuitDefinitionIdentityV1>(),
        },
        RawRegion {
            pointer: report_out.cast(),
            byte_count: size_of::<TemporalCircuitReportV1>(),
        },
    ];
    if !raw_regions_are_disjoint(&regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "circuit identity input and output regions overlap",
        );
    }
    let config = match request_value.config.into_runtime() {
        Ok(config) => config,
        Err(()) => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_DEFINITION,
                "circuit definition contains an invalid policy tag",
            )
        }
    };
    let token_codes = borrow_token_codes(request_value.token_codes, counts.token_codes);
    let threshold_offsets =
        borrow_threshold_offsets(request_value.threshold_offsets_mv, counts.threshold_offsets);
    let circuit_identity =
        match ValidatedCheckpointV1RuntimeFactory::circuit_identity_for_definition(
            config,
            token_codes,
            threshold_offsets,
        ) {
            Ok(identity) => identity,
            Err(_) => {
                return report.finish(
                    TEMPORAL_RUNTIME_INVALID_DEFINITION,
                    "circuit definition is not canonical",
                )
            }
        };
    output.write(TemporalCircuitDefinitionIdentityV1 {
        circuit_identity_sha256: circuit_identity,
    });
    report.required = 32;
    report.observed = 32;
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

/// Create a fresh runtime by importing its canonical empty checkpoint.
///
/// The definition selects capacities and immutable circuit identity only. It
/// cannot inject transient or learned state; nonempty construction remains the
/// checkpoint-import interface.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. `*handle_out` must be
/// initialized to null; it remains null on every failure.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_create_v1(
    definition: *const TemporalCircuitDefinitionV1,
    handle_out: *mut *mut TemporalCircuitRuntimeHandle,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if let Err(status) = validate_handle_output(handle_out, report) {
        return status;
    }
    if !fixed_pointer_is_valid(definition) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "circuit definition is null or misaligned",
        );
    }
    let definition = definition.read();
    let counts = match preflight_definition_regions(
        definition.token_codes,
        definition.token_code_count,
        definition.threshold_offsets_mv,
        definition.threshold_offset_count,
        report,
    ) {
        Ok(counts) => counts,
        Err(status) => return status,
    };
    let (authority, factory) = match build_validated_factory(definition, counts, report) {
        Ok(built) => built,
        Err(status) => return status,
    };
    let runtime = match import_canonical_empty_checkpoint_v1(factory) {
        Ok(runtime) => runtime,
        Err(error) => return canonical_empty_import_error(report, error),
    };
    publish_runtime_handle(runtime, authority, handle_out, report)
}

/// Import one canonical checkpoint and publish a new opaque handle last.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. `*handle_out` must be
/// initialized to null; it remains null on every failure.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
    request: *const TemporalCircuitImportCheckpointV1Request,
    handle_out: *mut *mut TemporalCircuitRuntimeHandle,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if let Err(status) = validate_handle_output(handle_out, report) {
        return status;
    }
    if !fixed_pointer_is_valid(request) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "import request is null or misaligned",
        );
    }
    let request = request.read();
    let definition = request.definition;
    let counts = match preflight_definition_regions(
        definition.token_codes,
        definition.token_code_count,
        definition.threshold_offsets_mv,
        definition.threshold_offset_count,
        report,
    ) {
        Ok(counts) => counts,
        Err(status) => return status,
    };
    let Some(checkpoint_len) = counted_region(request.checkpoint_bytes, request.checkpoint_len)
    else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint input geometry is invalid",
        );
    };
    let (authority, factory) = match build_validated_factory(definition, counts, report) {
        Ok(built) => built,
        Err(status) => return status,
    };
    let checkpoint = if checkpoint_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(request.checkpoint_bytes, checkpoint_len)
    };
    let runtime = match import_checkpoint_v1(checkpoint, factory) {
        Ok(runtime) => runtime,
        Err(error) => return checkpoint_import_error(report, error),
    };
    publish_runtime_handle(runtime, authority, handle_out, report)
}

/// Advance a fully preflighted block, committing successful prefix tokens.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. Output slots may be
/// uninitialized; this function constructs a complete record locally and
/// publishes it with `ptr::write` only after its token commits.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_advance_block_v2(
    handle: *mut TemporalCircuitRuntimeHandle,
    request: *const TemporalCircuitAdvanceBlockV2Request,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(handle.cast_const()) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "runtime handle is null or misaligned",
        );
    }
    if !fixed_pointer_is_valid(request) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "block request is null or misaligned",
        );
    }
    let request = request.read();
    if request.reserved_zero != [0; 6] {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "block request reserved bytes must be zero",
        );
    }
    let mode = match request.mode {
        MODE_READ_ONLY => TemporalStepMode::ReadOnly,
        MODE_LEARN => TemporalStepMode::Learn,
        _ => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "block request mode is invalid",
            );
        }
    };
    let reset_before = match request.reset_before {
        0 => false,
        1 => true,
        _ => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "block reset flag is invalid",
            );
        }
    };
    let Some(token_count) = counted_region(request.token_ids, request.token_count) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "token input geometry is invalid",
        );
    };
    let Some(output_capacity) = counted_region(request.output, request.output_capacity) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "step output geometry is invalid",
        );
    };
    if reset_before && token_count == 0 {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reset-before requires at least one token",
        );
    }
    if output_capacity < token_count {
        report.required = request.token_count;
        report.observed = request.output_capacity;
        return report.finish(
            TEMPORAL_RUNTIME_OUTPUT_CAPACITY,
            "step output capacity is too small",
        );
    }
    let token_bytes = token_count * size_of::<u32>();
    let output_bytes = output_capacity * size_of::<TemporalCircuitStepV2>();
    if regions_overlap(
        request.token_ids.cast(),
        token_bytes,
        request.output.cast(),
        output_bytes,
    ) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "token input and step output overlap",
        );
    }

    let handle = &*handle;
    let mut runtime = lock_runtime(handle);
    if runtime.scientific_stop.is_some() {
        return stop_report(report, &runtime, TemporalCircuitStop::Poisoned);
    }
    for index in 0..token_count {
        let token_id = request.token_ids.add(index).read();
        let result = if reset_before && index == 0 {
            runtime.step_after_transient_reset(token_id, mode)
        } else {
            runtime.step(token_id, mode)
        };
        match result {
            Ok(step) => {
                let output = TemporalCircuitStepV2::from(step);
                request.output.add(index).write(output);
                report.completed_prefix = (index + 1) as u64;
            }
            Err(stop) => return stop_report(report, &runtime, stop),
        }
    }
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

/// Generate and commit one surface token through the native physical route.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. `output` and `report_out`
/// must be disjoint writable records.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_generate_v1(
    handle: *mut TemporalCircuitRuntimeHandle,
    output: *mut TemporalGenerationV1,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(output.cast_const()) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "generation result is null or misaligned",
        );
    }
    if regions_overlap(
        output.cast(),
        size_of::<TemporalGenerationV1>(),
        report_out.cast(),
        size_of::<TemporalCircuitReportV1>(),
    ) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "generation result and report overlap",
        );
    }
    output.write(TemporalGenerationV1::ZERO);
    if !fixed_pointer_is_valid(handle.cast_const()) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "runtime handle is null or misaligned",
        );
    }

    let handle = &*handle;
    let mut runtime = lock_runtime(handle);
    if runtime.scientific_stop.is_some() {
        return stop_report(report, &runtime, TemporalCircuitStop::Poisoned);
    }
    match runtime.generate() {
        Ok(generation) => {
            output.write(generation.into());
            report.finish(TEMPORAL_RUNTIME_OK, "")
        }
        Err(stop) => stop_report(report, &runtime, stop),
    }
}

fn typed_value(code: u8) -> Option<TypedValue> {
    match code {
        TYPED_VALUE_INT => Some(TypedValue::Int),
        TYPED_VALUE_LIST_INT => Some(TypedValue::ListInt),
        TYPED_VALUE_INT_TO_INT => Some(TypedValue::IntToInt),
        TYPED_VALUE_BOOL => Some(TypedValue::Bool),
        _ => None,
    }
}

/// Acquire the native typed workspace and initialize its structural session.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_begin_typed_v1(
    handle: *mut TemporalCircuitRuntimeHandle,
    request: *const TemporalTypedSessionV1Request,
    output: *mut TemporalCircuitStepV2,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(handle.cast_const())
        || !fixed_pointer_is_valid(request)
        || !fixed_pointer_is_valid(output.cast_const())
    {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "typed session pointer is null or misaligned",
        );
    }
    if !raw_regions_are_disjoint(&[
        RawRegion {
            pointer: request.cast(),
            byte_count: size_of::<TemporalTypedSessionV1Request>(),
        },
        RawRegion {
            pointer: output.cast(),
            byte_count: size_of::<TemporalCircuitStepV2>(),
        },
        RawRegion {
            pointer: report_out.cast(),
            byte_count: size_of::<TemporalCircuitReportV1>(),
        },
    ]) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "typed session input, output, and report overlap",
        );
    }
    let request = request.read();
    if request.reserved_zero != [0; 5] {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "typed session reserved bytes must be zero",
        );
    }
    let mode = match request.mode {
        MODE_READ_ONLY => TemporalStepMode::ReadOnly,
        MODE_LEARN => TemporalStepMode::Learn,
        _ => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "typed session mode is invalid",
            );
        }
    };
    let Some(parameter) = typed_value(request.parameter_type) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "typed parameter type is invalid",
        );
    };
    let Some(result) = typed_value(request.result_type) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "typed result type is invalid",
        );
    };
    let handle = &*handle;
    let mut runtime = lock_runtime(handle);
    if runtime.scientific_stop.is_some() {
        return stop_report(report, &runtime, TemporalCircuitStop::Poisoned);
    }
    match runtime.begin_typed_session(request.vocabulary.into(), parameter, result, mode) {
        Ok(step) => {
            output.write(step.into());
            report.completed_prefix = 1;
            report.finish(TEMPORAL_RUNTIME_OK, "")
        }
        Err(stop) => stop_report(report, &runtime, stop),
    }
}

/// Commit one teacher control/surface token through the current typed hole.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_teach_typed_v1(
    handle: *mut TemporalCircuitRuntimeHandle,
    token_id: u32,
    output: *mut TemporalCircuitStepV2,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(handle.cast_const()) || !fixed_pointer_is_valid(output.cast_const())
    {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "typed teacher pointer is null or misaligned",
        );
    }
    if regions_overlap(
        output.cast(),
        size_of::<TemporalCircuitStepV2>(),
        report_out.cast(),
        size_of::<TemporalCircuitReportV1>(),
    ) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "typed teacher output and report overlap",
        );
    }
    let handle = &*handle;
    let mut runtime = lock_runtime(handle);
    if runtime.scientific_stop.is_some() {
        return stop_report(report, &runtime, TemporalCircuitStop::Poisoned);
    }
    match runtime.teach_typed_token(token_id) {
        Ok(step) => {
            output.write(step.into());
            report.completed_prefix = 1;
            report.finish(TEMPORAL_RUNTIME_OK, "")
        }
        Err(stop) => stop_report(report, &runtime, stop),
    }
}

/// Atomically own typed generation through EOS and publish its committed
/// surface prefix into a caller-owned fixed buffer.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_generate_typed_v1(
    handle: *mut TemporalCircuitRuntimeHandle,
    output: *mut TemporalGenerationV1,
    output_capacity: u64,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    let Some(output_capacity_usize) = counted_region(output, output_capacity) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "typed generation output geometry is invalid",
        );
    };
    if output_capacity_usize < MAX_TYPED_GENERATIONS {
        report.required = MAX_TYPED_GENERATIONS as u64;
        report.observed = output_capacity;
        return report.finish(
            TEMPORAL_RUNTIME_OUTPUT_CAPACITY,
            "typed generation output capacity is too small",
        );
    }
    let output_bytes = match output_capacity_usize.checked_mul(size_of::<TemporalGenerationV1>()) {
        Some(value) => value,
        None => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "typed generation output geometry overflows",
            );
        }
    };
    if !fixed_pointer_is_valid(handle.cast_const())
        || regions_overlap(
            output.cast(),
            output_bytes,
            report_out.cast(),
            size_of::<TemporalCircuitReportV1>(),
        )
    {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "typed generation handle/output/report is invalid",
        );
    }
    for index in 0..MAX_TYPED_GENERATIONS {
        output.add(index).write(TemporalGenerationV1::ZERO);
    }
    let handle = &*handle;
    let mut runtime = lock_runtime(handle);
    if runtime.scientific_stop.is_some() {
        return stop_report(report, &runtime, TemporalCircuitStop::Poisoned);
    }
    for index in 0..MAX_TYPED_GENERATIONS {
        if runtime.typed_generation_complete().unwrap_or(false) {
            return report.finish(TEMPORAL_RUNTIME_OK, "");
        }
        match runtime.generate_typed_one() {
            Ok(Some(generation)) => {
                output.add(index).write(generation.into());
                report.completed_prefix = (index + 1) as u64;
            }
            Ok(None) => return report.finish(TEMPORAL_RUNTIME_OK, ""),
            Err(stop) => return stop_report(report, &runtime, stop),
        }
    }
    report.required = (MAX_TYPED_GENERATIONS + 1) as u64;
    report.observed = MAX_TYPED_GENERATIONS as u64;
    stop_report(
        report,
        &runtime,
        TemporalCircuitStop::MalformedContextOrPolicy(
            "typed generation exceeded its fixed surface budget",
        ),
    )
}

/// Validate exportability and report the exact checkpoint length in
/// `report.required`.
///
/// # Safety
///
/// The module-level unsafe caller contract applies.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_checkpoint_size_v1(
    handle: *const TemporalCircuitRuntimeHandle,
    fresh_authority: *const TemporalValidatedCheckpointAuthorityV1,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(handle) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "runtime handle is null or misaligned",
        );
    }
    if !fixed_pointer_is_valid(fresh_authority) {
        return report.finish(TEMPORAL_RUNTIME_INVALID_ARGUMENT, "fresh authority is null");
    }
    let authority = fresh_authority.read().into_validated();
    let handle = &*handle;
    let runtime = lock_runtime(handle);
    if authority != handle.creation_authority {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_DEFINITION,
            "fresh authority does not match handle creation authority",
        );
    }
    let prepared = match runtime.prepare_checkpoint_v1_export(authority) {
        Ok(prepared) => prepared,
        Err(error) => return prepare_error(report, error),
    };
    report.required = prepared.encoded_len();
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

/// Return the exact canonical checkpoint-v1 header for the live runtime
/// without materializing or publishing a full checkpoint.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. `output` may be
/// uninitialized and is written as all zeroes before any authority or runtime
/// validation; the canonical header replaces it only on success.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_checkpoint_identity_v1(
    handle: *const TemporalCircuitRuntimeHandle,
    fresh_authority: *const TemporalValidatedCheckpointAuthorityV1,
    output: *mut TemporalCheckpointIdentityV1,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(output.cast_const()) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint identity output is null or misaligned",
        );
    }
    if regions_overlap(
        output.cast(),
        size_of::<TemporalCheckpointIdentityV1>(),
        report_out.cast(),
        size_of::<TemporalCircuitReportV1>(),
    ) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint identity output and report overlap",
        );
    }
    output.write(TemporalCheckpointIdentityV1::ZERO);
    if !fixed_pointer_is_valid(handle) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "runtime handle is null or misaligned",
        );
    }
    if !fixed_pointer_is_valid(fresh_authority) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "fresh authority is null or misaligned",
        );
    }
    let fixed_regions = [
        RawRegion {
            pointer: handle.cast(),
            byte_count: size_of::<TemporalCircuitRuntimeHandle>(),
        },
        RawRegion {
            pointer: fresh_authority.cast(),
            byte_count: size_of::<TemporalValidatedCheckpointAuthorityV1>(),
        },
        RawRegion {
            pointer: output.cast(),
            byte_count: size_of::<TemporalCheckpointIdentityV1>(),
        },
        RawRegion {
            pointer: report_out.cast(),
            byte_count: size_of::<TemporalCircuitReportV1>(),
        },
    ];
    if !raw_regions_are_disjoint(&fixed_regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint identity fixed regions overlap",
        );
    }

    let authority = fresh_authority.read().into_validated();
    let handle = &*handle;
    let runtime = lock_runtime(handle);
    if authority != handle.creation_authority {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_DEFINITION,
            "fresh authority does not match handle creation authority",
        );
    }
    let prepared = match runtime.prepare_checkpoint_v1_export(authority) {
        Ok(prepared) => prepared,
        Err(error) => return prepare_error(report, error),
    };
    let header = match prepared.identity_header() {
        Ok(header) => header,
        Err(_) => {
            return report.finish(
                TEMPORAL_RUNTIME_INTERNAL_INVARIANT,
                "canonical checkpoint identity encoding failed",
            )
        }
    };
    report.required = CHECKPOINT_V1_HEADER_BYTES.into();
    report.observed = CHECKPOINT_V1_HEADER_BYTES.into();
    output.write(TemporalCheckpointIdentityV1 { header });
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

/// Encode a canonical checkpoint into caller-owned, potentially uninitialized
/// bytes. A short buffer is rejected before any output write.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. `output` is never read or
/// borrowed as a Rust slice.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
    handle: *const TemporalCircuitRuntimeHandle,
    fresh_authority: *const TemporalValidatedCheckpointAuthorityV1,
    output: *mut u8,
    output_capacity: u64,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(handle) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "runtime handle is null or misaligned",
        );
    }
    if !fixed_pointer_is_valid(fresh_authority) {
        return report.finish(TEMPORAL_RUNTIME_INVALID_ARGUMENT, "fresh authority is null");
    }
    let Some(output_capacity_usize) = counted_region(output.cast_const(), output_capacity) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint output geometry is invalid",
        );
    };
    let authority = fresh_authority.read().into_validated();
    let handle = &*handle;
    let runtime = lock_runtime(handle);
    if authority != handle.creation_authority {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_DEFINITION,
            "fresh authority does not match handle creation authority",
        );
    }
    let prepared = match runtime.prepare_checkpoint_v1_export(authority) {
        Ok(prepared) => prepared,
        Err(error) => return prepare_error(report, error),
    };
    let required = prepared.encoded_len();
    report.required = required;
    report.observed = output_capacity;
    if output_capacity < required {
        return report.finish(
            TEMPORAL_RUNTIME_OUTPUT_CAPACITY,
            "checkpoint output capacity is too small",
        );
    }
    let Ok(required_usize) = usize::try_from(required) else {
        return report.finish(
            TEMPORAL_RUNTIME_INTERNAL_INVARIANT,
            "checkpoint length exceeds the host address domain",
        );
    };
    if required_usize > output_capacity_usize {
        return report.finish(
            TEMPORAL_RUNTIME_INTERNAL_INVARIANT,
            "checkpoint length conversion changed capacity ordering",
        );
    }
    let mut sink = FixedCheckpointSink::new(output, required_usize);
    if prepared.encode_into(&mut sink).is_err() || sink.logical_len() != required_usize {
        return report.finish(
            TEMPORAL_RUNTIME_INTERNAL_INVARIANT,
            "canonical checkpoint encoding failed",
        );
    }
    report.observed = required;
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

/// Stream a canonical checkpoint to a same-parent temporary file and publish
/// its final basename with the platform's audited no-overwrite primitive.
///
/// Publication outcomes are completion-plane results: once authority and
/// canonical preparation succeed, filesystem and encoder failures return
/// `TEMPORAL_RUNTIME_OK` and are described only by `publication_out`. Outcome
/// zero is reserved for ABI, authority, and preparation failures.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. Both path inputs are
/// immutable counted UTF-8 bytes. `publication_out` may be uninitialized; it
/// is initialized to the all-zero/N/A report before any request read and is
/// never read by this function.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
    handle: *const TemporalCircuitRuntimeHandle,
    fresh_authority: *const TemporalValidatedCheckpointAuthorityV1,
    request: *const TemporalCheckpointPathV1Request,
    publication_out: *mut TemporalCheckpointPublicationReportV1,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(publication_out.cast_const()) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "publication output is null or misaligned",
        );
    }
    if regions_overlap(
        publication_out.cast(),
        size_of::<TemporalCheckpointPublicationReportV1>(),
        report_out.cast(),
        size_of::<TemporalCircuitReportV1>(),
    ) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "publication and runtime reports overlap",
        );
    }
    let initialized = begin_publication_report(publication_out);
    debug_assert!(initialized);

    if !fixed_pointer_is_valid(handle) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "runtime handle is null or misaligned",
        );
    }
    if !fixed_pointer_is_valid(fresh_authority) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "fresh authority is null or misaligned",
        );
    }
    if !fixed_pointer_is_valid(request) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint path request is null or misaligned",
        );
    }

    let fixed_regions = [
        RawRegion {
            pointer: handle.cast(),
            byte_count: size_of::<TemporalCircuitRuntimeHandle>(),
        },
        RawRegion {
            pointer: fresh_authority.cast(),
            byte_count: size_of::<TemporalValidatedCheckpointAuthorityV1>(),
        },
        RawRegion {
            pointer: request.cast(),
            byte_count: size_of::<TemporalCheckpointPathV1Request>(),
        },
        RawRegion {
            pointer: publication_out.cast(),
            byte_count: size_of::<TemporalCheckpointPublicationReportV1>(),
        },
        RawRegion {
            pointer: report_out.cast(),
            byte_count: size_of::<TemporalCircuitReportV1>(),
        },
    ];
    if !raw_regions_are_disjoint(&fixed_regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint path fixed regions overlap",
        );
    }

    let request_value = request.read();
    if request_value.reserved_zero != [0; 8] {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint path request reserved bytes must be zero",
        );
    }
    let Some(parent_len) = counted_region(request_value.parent_utf8, request_value.parent_len)
    else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint parent path geometry is invalid",
        );
    };
    let Some(final_basename_len) = counted_region(
        request_value.final_basename_utf8,
        request_value.final_basename_len,
    ) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint final basename geometry is invalid",
        );
    };
    let all_regions = [
        fixed_regions[0],
        fixed_regions[1],
        fixed_regions[2],
        fixed_regions[3],
        fixed_regions[4],
        RawRegion {
            pointer: request_value.parent_utf8,
            byte_count: parent_len,
        },
        RawRegion {
            pointer: request_value.final_basename_utf8,
            byte_count: final_basename_len,
        },
    ];
    if !raw_regions_are_disjoint(&all_regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "checkpoint path regions overlap",
        );
    }

    let parent_utf8 = foreign_bytes(request_value.parent_utf8, parent_len);
    let final_basename_utf8 = foreign_bytes(request_value.final_basename_utf8, final_basename_len);
    let destination = match validate_destination_v1(parent_utf8, final_basename_utf8) {
        Ok(destination) => destination,
        Err(DestinationInputError::Invalid) => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "checkpoint destination is invalid",
            )
        }
        Err(DestinationInputError::Allocation) => {
            return report.finish(
                TEMPORAL_RUNTIME_ALLOCATION_FAILED,
                "checkpoint destination copy allocation failed",
            )
        }
    };

    let authority = fresh_authority.read().into_validated();
    let handle = &*handle;
    let runtime = lock_runtime(handle);
    if authority != handle.creation_authority {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_DEFINITION,
            "fresh authority does not match handle creation authority",
        );
    }
    let prepared = match runtime.prepare_checkpoint_v1_export(authority) {
        Ok(prepared) => prepared,
        Err(error) => return prepare_error(report, error),
    };
    report.required = prepared.encoded_len();

    let completion = publish_prepared_checkpoint_v1(&prepared, &destination);
    report.observed = completion.written_bytes;
    publication_report(publication_out).fill(completion);
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

/// Publish one bounded opaque byte string with the same exclusive durable
/// filesystem contract as checkpoint publication.
///
/// This raw symbol validates transport only. The supported Python capability
/// performs the live protocol/source authority checks before reaching it.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. All three counted inputs
/// are immutable for the duration of this call and pairwise disjoint from the
/// request and output records.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_publish_immutable_bytes_path_v1(
    request: *const TemporalImmutableBytesPathV1Request,
    publication_out: *mut TemporalCheckpointPublicationReportV1,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(publication_out.cast_const()) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "publication output is null or misaligned",
        );
    }
    if regions_overlap(
        publication_out.cast(),
        size_of::<TemporalCheckpointPublicationReportV1>(),
        report_out.cast(),
        size_of::<TemporalCircuitReportV1>(),
    ) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "publication and runtime reports overlap",
        );
    }
    let initialized = begin_publication_report(publication_out);
    debug_assert!(initialized);
    if !fixed_pointer_is_valid(request) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "immutable-byte request is null or misaligned",
        );
    }

    let fixed_regions = [
        RawRegion {
            pointer: request.cast(),
            byte_count: size_of::<TemporalImmutableBytesPathV1Request>(),
        },
        RawRegion {
            pointer: publication_out.cast(),
            byte_count: size_of::<TemporalCheckpointPublicationReportV1>(),
        },
        RawRegion {
            pointer: report_out.cast(),
            byte_count: size_of::<TemporalCircuitReportV1>(),
        },
    ];
    if !raw_regions_are_disjoint(&fixed_regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "immutable-byte fixed regions overlap",
        );
    }

    let request_value = request.read();
    if request_value.reserved_zero != [0; 8] {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "immutable-byte request reserved bytes must be zero",
        );
    }
    let Some(parent_len) = counted_region(request_value.parent_utf8, request_value.parent_len)
    else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "immutable-byte parent path geometry is invalid",
        );
    };
    let Some(final_basename_len) = counted_region(
        request_value.final_basename_utf8,
        request_value.final_basename_len,
    ) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "immutable-byte final basename geometry is invalid",
        );
    };
    let Some(payload_len) = counted_region(request_value.payload, request_value.payload_len) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "immutable-byte payload geometry is invalid",
        );
    };
    if !(1..=MAX_IMMUTABLE_PAYLOAD_BYTES).contains(&payload_len) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "immutable-byte payload length is outside the bounded range",
        );
    }
    let all_regions = [
        fixed_regions[0],
        fixed_regions[1],
        fixed_regions[2],
        RawRegion {
            pointer: request_value.parent_utf8,
            byte_count: parent_len,
        },
        RawRegion {
            pointer: request_value.final_basename_utf8,
            byte_count: final_basename_len,
        },
        RawRegion {
            pointer: request_value.payload,
            byte_count: payload_len,
        },
    ];
    if !raw_regions_are_disjoint(&all_regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "immutable-byte request regions overlap",
        );
    }

    let parent_utf8 = foreign_bytes(request_value.parent_utf8, parent_len);
    let final_basename_utf8 = foreign_bytes(request_value.final_basename_utf8, final_basename_len);
    let destination = match validate_destination_v1(parent_utf8, final_basename_utf8) {
        Ok(destination) => destination,
        Err(DestinationInputError::Invalid) => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "immutable-byte destination is invalid",
            )
        }
        Err(DestinationInputError::Allocation) => {
            return report.finish(
                TEMPORAL_RUNTIME_ALLOCATION_FAILED,
                "immutable-byte destination copy allocation failed",
            )
        }
    };
    let payload = foreign_bytes(request_value.payload, payload_len);
    let prepared = match prepare_immutable_bytes_v1(payload) {
        Ok(prepared) => prepared,
        Err(ImmutablePayloadInputError::Invalid) => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "immutable-byte payload is invalid",
            )
        }
        Err(ImmutablePayloadInputError::Allocation) => {
            return report.finish(
                TEMPORAL_RUNTIME_ALLOCATION_FAILED,
                "immutable-byte payload copy allocation failed",
            )
        }
    };
    report.required = payload_len as u64;
    let completion = publish_prepared_immutable_bytes_v1(&prepared, &destination);
    report.observed = completion.written_bytes;
    publication_report(publication_out).fill(completion);
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

/// Re-establish durability for one already-existing immutable byte artifact.
///
/// This operation never creates, renames, replaces, truncates, writes, or
/// deletes a filesystem object. It verifies the exact bounded payload before
/// and after flushing the stable file and parent-directory identities.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. All three counted inputs
/// are immutable and pairwise disjoint from the request and output records.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_reconcile_existing_immutable_bytes_path_v1(
    request: *const TemporalImmutableBytesPathV1Request,
    durability_out: *mut TemporalImmutableBytesReconciliationReportV1,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(durability_out.cast_const()) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation output is null or misaligned",
        );
    }
    if regions_overlap(
        durability_out.cast(),
        size_of::<TemporalImmutableBytesReconciliationReportV1>(),
        report_out.cast(),
        size_of::<TemporalCircuitReportV1>(),
    ) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation and runtime reports overlap",
        );
    }
    let initialized = begin_reconciliation_report(durability_out);
    debug_assert!(initialized);
    if !fixed_pointer_is_valid(request) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation request is null or misaligned",
        );
    }
    let fixed_regions = [
        RawRegion {
            pointer: request.cast(),
            byte_count: size_of::<TemporalImmutableBytesPathV1Request>(),
        },
        RawRegion {
            pointer: durability_out.cast(),
            byte_count: size_of::<TemporalImmutableBytesReconciliationReportV1>(),
        },
        RawRegion {
            pointer: report_out.cast(),
            byte_count: size_of::<TemporalCircuitReportV1>(),
        },
    ];
    if !raw_regions_are_disjoint(&fixed_regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation fixed regions overlap",
        );
    }
    let request_value = request.read();
    if request_value.reserved_zero != [0; 8] {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation request reserved bytes must be zero",
        );
    }
    let Some(parent_len) = counted_region(request_value.parent_utf8, request_value.parent_len)
    else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation parent path geometry is invalid",
        );
    };
    let Some(final_basename_len) = counted_region(
        request_value.final_basename_utf8,
        request_value.final_basename_len,
    ) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation final basename geometry is invalid",
        );
    };
    let Some(payload_len) = counted_region(request_value.payload, request_value.payload_len) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation payload geometry is invalid",
        );
    };
    if !(1..=MAX_IMMUTABLE_PAYLOAD_BYTES).contains(&payload_len) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation payload length is outside the bounded range",
        );
    }
    let all_regions = [
        fixed_regions[0],
        fixed_regions[1],
        fixed_regions[2],
        RawRegion {
            pointer: request_value.parent_utf8,
            byte_count: parent_len,
        },
        RawRegion {
            pointer: request_value.final_basename_utf8,
            byte_count: final_basename_len,
        },
        RawRegion {
            pointer: request_value.payload,
            byte_count: payload_len,
        },
    ];
    if !raw_regions_are_disjoint(&all_regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "reconciliation request regions overlap",
        );
    }
    let parent_utf8 = foreign_bytes(request_value.parent_utf8, parent_len);
    let final_basename_utf8 = foreign_bytes(request_value.final_basename_utf8, final_basename_len);
    let destination = match validate_destination_v1(parent_utf8, final_basename_utf8) {
        Ok(destination) => destination,
        Err(DestinationInputError::Invalid) => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "reconciliation destination is invalid",
            )
        }
        Err(DestinationInputError::Allocation) => {
            return report.finish(
                TEMPORAL_RUNTIME_ALLOCATION_FAILED,
                "reconciliation destination copy allocation failed",
            )
        }
    };
    let expected = foreign_bytes(request_value.payload, payload_len);
    let prepared = match prepare_immutable_bytes_v1(expected) {
        Ok(prepared) => prepared,
        Err(ImmutablePayloadInputError::Invalid) => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "reconciliation payload is invalid",
            )
        }
        Err(ImmutablePayloadInputError::Allocation) => {
            return report.finish(
                TEMPORAL_RUNTIME_ALLOCATION_FAILED,
                "reconciliation payload copy allocation failed",
            )
        }
    };
    report.required = payload_len as u64;
    let completion = reconcile_existing_immutable_bytes_v1(&prepared, &destination);
    report.observed = completion.verified_bytes;
    (*durability_out).fill(completion);
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

/// Idempotently create or reconcile a relative directory closure beneath one
/// existing trusted root, flushing every child and parent directory.
///
/// # Safety
///
/// The module-level unsafe caller contract applies. Both counted path inputs
/// are immutable and pairwise disjoint from the request and output records.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_prepare_durable_directory_path_v1(
    request: *const TemporalDurableDirectoryPathV1Request,
    durability_out: *mut TemporalDurableDirectoryPreparationReportV1,
    report_out: *mut TemporalCircuitReportV1,
) -> u32 {
    if !begin_report(report_out) {
        return TEMPORAL_RUNTIME_INVALID_ARGUMENT;
    }
    let report = report(report_out);
    if !fixed_pointer_is_valid(durability_out.cast_const()) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "directory durability output is null or misaligned",
        );
    }
    if regions_overlap(
        durability_out.cast(),
        size_of::<TemporalDurableDirectoryPreparationReportV1>(),
        report_out.cast(),
        size_of::<TemporalCircuitReportV1>(),
    ) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "directory durability and runtime reports overlap",
        );
    }
    let initialized = begin_directory_preparation_report(durability_out);
    debug_assert!(initialized);
    if !fixed_pointer_is_valid(request) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "directory durability request is null or misaligned",
        );
    }
    let fixed_regions = [
        RawRegion {
            pointer: request.cast(),
            byte_count: size_of::<TemporalDurableDirectoryPathV1Request>(),
        },
        RawRegion {
            pointer: durability_out.cast(),
            byte_count: size_of::<TemporalDurableDirectoryPreparationReportV1>(),
        },
        RawRegion {
            pointer: report_out.cast(),
            byte_count: size_of::<TemporalCircuitReportV1>(),
        },
    ];
    if !raw_regions_are_disjoint(&fixed_regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "directory durability fixed regions overlap",
        );
    }
    let request_value = request.read();
    if request_value.reserved_zero != [0; 8] {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "directory durability request reserved bytes must be zero",
        );
    }
    let Some(root_len) = counted_region(request_value.root_utf8, request_value.root_len) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "directory durability root geometry is invalid",
        );
    };
    let Some(relative_directory_len) = counted_region(
        request_value.relative_directory_utf8,
        request_value.relative_directory_len,
    ) else {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "directory durability relative path geometry is invalid",
        );
    };
    let all_regions = [
        fixed_regions[0],
        fixed_regions[1],
        fixed_regions[2],
        RawRegion {
            pointer: request_value.root_utf8,
            byte_count: root_len,
        },
        RawRegion {
            pointer: request_value.relative_directory_utf8,
            byte_count: relative_directory_len,
        },
    ];
    if !raw_regions_are_disjoint(&all_regions) {
        return report.finish(
            TEMPORAL_RUNTIME_INVALID_ARGUMENT,
            "directory durability request regions overlap",
        );
    }
    let root_utf8 = foreign_bytes(request_value.root_utf8, root_len);
    let relative_directory_utf8 = foreign_bytes(
        request_value.relative_directory_utf8,
        relative_directory_len,
    );
    let directory = match validate_durable_directory_v1(root_utf8, relative_directory_utf8) {
        Ok(directory) => directory,
        Err(DestinationInputError::Invalid) => {
            return report.finish(
                TEMPORAL_RUNTIME_INVALID_ARGUMENT,
                "directory durability path is invalid",
            )
        }
        Err(DestinationInputError::Allocation) => {
            return report.finish(
                TEMPORAL_RUNTIME_ALLOCATION_FAILED,
                "directory durability path copy allocation failed",
            )
        }
    };
    report.required = u64::from(directory.component_count());
    let completion = prepare_durable_directory_v1(&directory);
    report.observed = u64::from(completion.completed_component_count);
    (*durability_out).fill(completion);
    report.finish(TEMPORAL_RUNTIME_OK, "")
}

/// Destroy an opaque handle. Null is a no-op.
///
/// # Safety
///
/// A non-null handle must have been returned by the create or import function,
/// must not have been destroyed before, and must not be used concurrently.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_circuit_runtime_destroy(
    handle: *mut TemporalCircuitRuntimeHandle,
) {
    if !handle.is_null() {
        destroy_owned(handle);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::Cursor,
        mem::{align_of, offset_of, size_of, MaybeUninit},
    };

    use sha2::{Digest, Sha256};

    use super::*;
    use crate::temporal_runtime::{
        checkpoint::tests::{
            cpython_empty_full_section_generic_checkpoint_v1,
            cpython_full_section_generic_checkpoint_v1,
        },
        tests::count_allocations,
        TemporalSegmentInit,
    };

    fn raw_bytes<T>(value: &T) -> &[u8] {
        // SAFETY: callers use fully initialized repr(C) records whose layout
        // assertions below prove contains no implicit padding.
        unsafe { slice::from_raw_parts((value as *const T).cast(), size_of::<T>()) }
    }

    fn report_message(report: &TemporalCircuitReportV1) -> &str {
        let message_len = usize::try_from(report.message_len).expect("message length fits usize");
        std::str::from_utf8(&report.message_utf8[..message_len])
            .expect("native report messages are UTF-8")
    }

    #[test]
    fn repr_c_layouts_match_the_frozen_abi() {
        assert_eq!(size_of::<TemporalValidatedCheckpointAuthorityV1>(), 64);
        assert_eq!(align_of::<TemporalValidatedCheckpointAuthorityV1>(), 1);
        assert_eq!(
            offset_of!(
                TemporalValidatedCheckpointAuthorityV1,
                source_bundle_identity_sha256
            ),
            32
        );

        assert_eq!(size_of::<TemporalCheckpointIdentityV1>(), 256);
        assert_eq!(align_of::<TemporalCheckpointIdentityV1>(), 1);
        assert_eq!(offset_of!(TemporalCheckpointIdentityV1, header), 0);

        assert_eq!(size_of::<TemporalCircuitConfigV1>(), 32);
        assert_eq!(align_of::<TemporalCircuitConfigV1>(), 4);
        assert_eq!(offset_of!(TemporalCircuitConfigV1, column_count), 0);
        assert_eq!(offset_of!(TemporalCircuitConfigV1, max_pyramidal_cells), 4);
        assert_eq!(offset_of!(TemporalCircuitConfigV1, segment_capacity), 8);
        assert_eq!(offset_of!(TemporalCircuitConfigV1, construction_seed), 12);
        assert_eq!(
            offset_of!(TemporalCircuitConfigV1, connected_permanence_ticks),
            16
        );
        assert_eq!(
            offset_of!(TemporalCircuitConfigV1, initial_permanence_ticks),
            18
        );
        assert_eq!(
            offset_of!(TemporalCircuitConfigV1, permanence_increment_ticks),
            20
        );
        assert_eq!(
            offset_of!(TemporalCircuitConfigV1, negative_decrement_ticks),
            22
        );
        assert_eq!(
            offset_of!(TemporalCircuitConfigV1, base_cells_per_column),
            24
        );
        assert_eq!(offset_of!(TemporalCircuitConfigV1, novel_race_cap), 25);
        assert_eq!(offset_of!(TemporalCircuitConfigV1, prediction_quorum), 26);
        assert_eq!(
            offset_of!(TemporalCircuitConfigV1, max_segments_per_cell),
            27
        );
        assert_eq!(offset_of!(TemporalCircuitConfigV1, theta_match), 28);
        assert_eq!(offset_of!(TemporalCircuitConfigV1, theta_active), 29);
        assert_eq!(offset_of!(TemporalCircuitConfigV1, context_policy), 30);
        assert_eq!(offset_of!(TemporalCircuitConfigV1, plasticity_policy), 31);

        assert_eq!(size_of::<TemporalTokenCodeV1>(), 48);
        assert_eq!(align_of::<TemporalTokenCodeV1>(), 4);
        assert_eq!(offset_of!(TemporalTokenCodeV1, token_id), 0);
        assert_eq!(offset_of!(TemporalTokenCodeV1, columns), 4);
        assert_eq!(size_of::<TemporalTokenCodeV1>(), size_of::<TokenCode>());
        assert_eq!(align_of::<TemporalTokenCodeV1>(), align_of::<TokenCode>());
        assert_eq!(offset_of!(TokenCode, token_id), 0);
        assert_eq!(offset_of!(TokenCode, columns), 4);

        assert_eq!(size_of::<TemporalCircuitDefinitionV1>(), 128);
        assert_eq!(align_of::<TemporalCircuitDefinitionV1>(), 8);
        assert_eq!(offset_of!(TemporalCircuitDefinitionV1, config), 0);
        assert_eq!(offset_of!(TemporalCircuitDefinitionV1, token_codes), 32);
        assert_eq!(
            offset_of!(TemporalCircuitDefinitionV1, token_code_count),
            40
        );
        assert_eq!(
            offset_of!(TemporalCircuitDefinitionV1, threshold_offsets_mv),
            48
        );
        assert_eq!(
            offset_of!(TemporalCircuitDefinitionV1, threshold_offset_count),
            56
        );
        assert_eq!(offset_of!(TemporalCircuitDefinitionV1, authority), 64);

        assert_eq!(size_of::<TemporalCircuitDefinitionIdentityV1Request>(), 64);
        assert_eq!(align_of::<TemporalCircuitDefinitionIdentityV1Request>(), 8);
        assert_eq!(
            offset_of!(TemporalCircuitDefinitionIdentityV1Request, config),
            0
        );
        assert_eq!(
            offset_of!(TemporalCircuitDefinitionIdentityV1Request, token_codes),
            32
        );
        assert_eq!(
            offset_of!(TemporalCircuitDefinitionIdentityV1Request, token_code_count),
            40
        );
        assert_eq!(
            offset_of!(
                TemporalCircuitDefinitionIdentityV1Request,
                threshold_offsets_mv
            ),
            48
        );
        assert_eq!(
            offset_of!(
                TemporalCircuitDefinitionIdentityV1Request,
                threshold_offset_count
            ),
            56
        );

        assert_eq!(size_of::<TemporalCircuitDefinitionIdentityV1>(), 32);
        assert_eq!(align_of::<TemporalCircuitDefinitionIdentityV1>(), 1);
        assert_eq!(
            offset_of!(TemporalCircuitDefinitionIdentityV1, circuit_identity_sha256),
            0
        );

        assert_eq!(size_of::<TemporalCircuitImportCheckpointV1Request>(), 144);
        assert_eq!(align_of::<TemporalCircuitImportCheckpointV1Request>(), 8);
        assert_eq!(
            offset_of!(TemporalCircuitImportCheckpointV1Request, definition),
            0
        );
        assert_eq!(
            offset_of!(TemporalCircuitImportCheckpointV1Request, checkpoint_bytes),
            128
        );
        assert_eq!(
            offset_of!(TemporalCircuitImportCheckpointV1Request, checkpoint_len),
            136
        );

        assert_eq!(size_of::<TemporalCheckpointPathV1Request>(), 40);
        assert_eq!(align_of::<TemporalCheckpointPathV1Request>(), 8);
        assert_eq!(offset_of!(TemporalCheckpointPathV1Request, parent_utf8), 0);
        assert_eq!(offset_of!(TemporalCheckpointPathV1Request, parent_len), 8);
        assert_eq!(
            offset_of!(TemporalCheckpointPathV1Request, final_basename_utf8),
            16
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPathV1Request, final_basename_len),
            24
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPathV1Request, reserved_zero),
            32
        );

        assert_eq!(size_of::<TemporalImmutableBytesPathV1Request>(), 56);
        assert_eq!(align_of::<TemporalImmutableBytesPathV1Request>(), 8);
        assert_eq!(
            offset_of!(TemporalImmutableBytesPathV1Request, parent_utf8),
            0
        );
        assert_eq!(
            offset_of!(TemporalImmutableBytesPathV1Request, parent_len),
            8
        );
        assert_eq!(
            offset_of!(TemporalImmutableBytesPathV1Request, final_basename_utf8),
            16
        );
        assert_eq!(
            offset_of!(TemporalImmutableBytesPathV1Request, final_basename_len),
            24
        );
        assert_eq!(offset_of!(TemporalImmutableBytesPathV1Request, payload), 32);
        assert_eq!(
            offset_of!(TemporalImmutableBytesPathV1Request, payload_len),
            40
        );
        assert_eq!(
            offset_of!(TemporalImmutableBytesPathV1Request, reserved_zero),
            48
        );

        assert_eq!(size_of::<TemporalCheckpointPublicationErrorV1>(), 8);
        assert_eq!(align_of::<TemporalCheckpointPublicationErrorV1>(), 4);
        assert_eq!(offset_of!(TemporalCheckpointPublicationErrorV1, stage), 0);
        assert_eq!(offset_of!(TemporalCheckpointPublicationErrorV1, domain), 2);
        assert_eq!(offset_of!(TemporalCheckpointPublicationErrorV1, code), 4);

        assert_eq!(size_of::<TemporalCheckpointPublicationReportV1>(), 80);
        assert_eq!(align_of::<TemporalCheckpointPublicationReportV1>(), 8);
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, outcome),
            0
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, temp_cleanup),
            2
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, cleanup_error_count),
            3
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, primary_error),
            4
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, cleanup_errors),
            12
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, owned_temp_pid),
            44
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, owned_temp_nonce),
            48
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, checkpoint_bytes),
            56
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, encoded_bytes),
            64
        );
        assert_eq!(
            offset_of!(TemporalCheckpointPublicationReportV1, reserved_zero),
            72
        );

        assert_eq!(size_of::<TemporalSpikeV1>(), 8);
        assert_eq!(align_of::<TemporalSpikeV1>(), 4);
        assert_eq!(offset_of!(TemporalSpikeV1, cell_id), 0);
        assert_eq!(offset_of!(TemporalSpikeV1, tick), 4);
        assert_eq!(offset_of!(TemporalSpikeV1, reserved_zero), 6);

        assert_eq!(size_of::<TemporalCircuitStepV2>(), 800);
        assert_eq!(align_of::<TemporalCircuitStepV2>(), 8);
        assert_eq!(offset_of!(TemporalCircuitStepV2, token_id), 0);
        assert_eq!(offset_of!(TemporalCircuitStepV2, punished_segments), 4);
        assert_eq!(offset_of!(TemporalCircuitStepV2, event_count), 8);
        assert_eq!(offset_of!(TemporalCircuitStepV2, candidate_segments), 16);
        assert_eq!(offset_of!(TemporalCircuitStepV2, matching_segments), 24);
        assert_eq!(offset_of!(TemporalCircuitStepV2, active_segments), 32);
        assert_eq!(offset_of!(TemporalCircuitStepV2, postings_visited), 40);
        assert_eq!(offset_of!(TemporalCircuitStepV2, predicted), 48);
        assert_eq!(offset_of!(TemporalCircuitStepV2, predicted_columns), 49);
        assert_eq!(offset_of!(TemporalCircuitStepV2, allocated_segments), 50);
        assert_eq!(offset_of!(TemporalCircuitStepV2, reserved_zero), 51);
        assert_eq!(
            offset_of!(TemporalCircuitStepV2, inhibition_onset_ticks),
            52
        );
        assert_eq!(
            offset_of!(TemporalCircuitStepV2, column_simulated_ticks),
            74
        );
        assert_eq!(offset_of!(TemporalCircuitStepV2, winner_spikes), 96);
        assert_eq!(offset_of!(TemporalCircuitStepV2, basket_spikes), 184);
        assert_eq!(offset_of!(TemporalCircuitStepV2, proximal_spikes), 272);
        assert_eq!(
            offset_of!(TemporalCircuitStepV2, proximal_relay_spikes),
            360
        );
        assert_eq!(offset_of!(TemporalCircuitStepV2, winner_relay_spikes), 448);
        assert_eq!(offset_of!(TemporalCircuitStepV2, winner_context), 536);
        assert_eq!(offset_of!(TemporalCircuitStepV2, proximal_context), 624);
        assert_eq!(offset_of!(TemporalCircuitStepV2, context), 712);

        assert_eq!(size_of::<TemporalGenerationV1>(), 64);
        assert_eq!(align_of::<TemporalGenerationV1>(), 8);
        assert_eq!(offset_of!(TemporalGenerationV1, surface_token_id), 0);
        assert_eq!(offset_of!(TemporalGenerationV1, surface_output_cell_id), 4);
        assert_eq!(offset_of!(TemporalGenerationV1, generation_index), 8);
        assert_eq!(offset_of!(TemporalGenerationV1, temporal_event_count), 16);
        assert_eq!(offset_of!(TemporalGenerationV1, activated_columns), 24);
        assert_eq!(offset_of!(TemporalGenerationV1, candidate_tokens), 28);
        assert_eq!(offset_of!(TemporalGenerationV1, control_token_id), 32);
        assert_eq!(offset_of!(TemporalGenerationV1, control_output_cell_id), 36);
        assert_eq!(offset_of!(TemporalGenerationV1, surface_winner_tick), 40);
        assert_eq!(offset_of!(TemporalGenerationV1, path), 54);
        assert_eq!(offset_of!(TemporalGenerationV1, copy_place_ordinal), 58);
        assert_eq!(offset_of!(TemporalGenerationV1, slot_ordinal), 60);
        assert_eq!(offset_of!(TemporalGenerationV1, reserved_zero), 61);

        assert_eq!(size_of::<TemporalTypedWorkspaceVocabularyV1>(), 196);
        assert_eq!(align_of::<TemporalTypedWorkspaceVocabularyV1>(), 4);
        assert_eq!(
            offset_of!(TemporalTypedWorkspaceVocabularyV1, plan_commit_token_id),
            0
        );
        assert_eq!(
            offset_of!(TemporalTypedWorkspaceVocabularyV1, address_token_base),
            4
        );
        assert_eq!(
            offset_of!(TemporalTypedWorkspaceVocabularyV1, cue_token_ids),
            8
        );
        assert_eq!(
            offset_of!(TemporalTypedWorkspaceVocabularyV1, surface_token_ids),
            108
        );

        assert_eq!(size_of::<TemporalTypedSessionV1Request>(), 204);
        assert_eq!(align_of::<TemporalTypedSessionV1Request>(), 4);
        assert_eq!(offset_of!(TemporalTypedSessionV1Request, vocabulary), 0);
        assert_eq!(
            offset_of!(TemporalTypedSessionV1Request, parameter_type),
            196
        );
        assert_eq!(offset_of!(TemporalTypedSessionV1Request, result_type), 197);
        assert_eq!(offset_of!(TemporalTypedSessionV1Request, mode), 198);
        assert_eq!(
            offset_of!(TemporalTypedSessionV1Request, reserved_zero),
            199
        );

        assert_eq!(size_of::<TemporalCircuitAdvanceBlockV2Request>(), 40);
        assert_eq!(align_of::<TemporalCircuitAdvanceBlockV2Request>(), 8);
        assert_eq!(
            offset_of!(TemporalCircuitAdvanceBlockV2Request, token_ids),
            0
        );
        assert_eq!(offset_of!(TemporalCircuitAdvanceBlockV2Request, output), 8);
        assert_eq!(
            offset_of!(TemporalCircuitAdvanceBlockV2Request, token_count),
            16
        );
        assert_eq!(
            offset_of!(TemporalCircuitAdvanceBlockV2Request, output_capacity),
            24
        );
        assert_eq!(offset_of!(TemporalCircuitAdvanceBlockV2Request, mode), 32);
        assert_eq!(
            offset_of!(TemporalCircuitAdvanceBlockV2Request, reset_before),
            33
        );
        assert_eq!(
            offset_of!(TemporalCircuitAdvanceBlockV2Request, reserved_zero),
            34
        );

        assert_eq!(size_of::<TemporalCircuitReportV1>(), 512);
        assert_eq!(align_of::<TemporalCircuitReportV1>(), 8);
        assert_eq!(offset_of!(TemporalCircuitReportV1, status), 0);
        assert_eq!(offset_of!(TemporalCircuitReportV1, message_len), 4);
        assert_eq!(offset_of!(TemporalCircuitReportV1, completed_prefix), 8);
        assert_eq!(offset_of!(TemporalCircuitReportV1, required), 16);
        assert_eq!(offset_of!(TemporalCircuitReportV1, observed), 24);
        assert_eq!(offset_of!(TemporalCircuitReportV1, location), 32);
        assert_eq!(offset_of!(TemporalCircuitReportV1, poison_code), 40);
        assert_eq!(offset_of!(TemporalCircuitReportV1, stage_code), 42);
        assert_eq!(offset_of!(TemporalCircuitReportV1, reserved_zero), 44);
        assert_eq!(offset_of!(TemporalCircuitReportV1, message_utf8), 48);
    }

    #[test]
    fn report_and_step_bytes_match_the_independent_python_oracle() {
        let message = "oracle/R1: snowman=☃; café; little-endian; stop@column-race";
        assert_eq!(message.len(), 62);
        let mut report = TemporalCircuitReportV1::ZERO;
        report.status = TEMPORAL_RUNTIME_SCIENTIFIC_STOP;
        report.message_len = message.len() as u32;
        report.completed_prefix = 0x0102_0304_0506_0708;
        report.required = 0x1112_1314_1516_1718;
        report.observed = 0x2122_2324_2526_2728;
        report.location = 0x3132_3334_3536_3738;
        report.poison_code = 7;
        report.stage_code = STAGE_COLUMN_RACE;
        report.message_utf8[..message.len()].copy_from_slice(message.as_bytes());
        let report_bytes = raw_bytes(&report);
        assert_eq!(
            &report_bytes[..48],
            &hex_bytes(
                "080000003e00000008070605040302011817161514131211282726252423222138373635343332310700030000000000"
            )[..]
        );
        assert_eq!(
            hex_digest(report_bytes),
            "5d7e11f6a153082ae9e39ef176f50e11e7a56067b023a6e6213efc4b0d97bda4"
        );

        let spikes: [[TemporalSpikeV1; ACTIVE_COLUMNS]; 5] = std::array::from_fn(|group| {
            std::array::from_fn(|index| {
                let n = group * ACTIVE_COLUMNS + index;
                TemporalSpikeV1 {
                    cell_id: 0x1020_3040 + (n as u32) * 0x0001_0101,
                    tick: 201 + n as u16,
                    reserved_zero: 0,
                }
            })
        });
        let contexts: [[u32; CONTEXT_CELLS]; 3] = std::array::from_fn(|group| {
            std::array::from_fn(|index| {
                let n = group * CONTEXT_CELLS + index;
                0x5060_7080 + (n as u32) * 0x0001_0003
            })
        });
        let step = TemporalCircuitStepV2 {
            token_id: 0x0A1B_2C3D,
            punished_segments: 0x1E2F_4051,
            event_count: 0x6273_8495_A6B7_C8D9,
            candidate_segments: 0x0102_0304_0506_0708,
            matching_segments: 0x1112_1314_1516_1718,
            active_segments: 0x2122_2324_2526_2728,
            postings_visited: 0x3132_3334_3536_3738,
            predicted: 1,
            predicted_columns: 7,
            allocated_segments: 5,
            reserved_zero: 0,
            inhibition_onset_ticks: std::array::from_fn(|index| 101 + 2 * index as u16),
            column_simulated_ticks: std::array::from_fn(|index| 102 + 2 * index as u16),
            winner_spikes: spikes[0],
            basket_spikes: spikes[1],
            proximal_spikes: spikes[2],
            proximal_relay_spikes: spikes[3],
            winner_relay_spikes: spikes[4],
            winner_context: contexts[0],
            proximal_context: contexts[1],
            context: contexts[2],
        };
        let step_bytes = raw_bytes(&step);
        assert_eq!(
            &step_bytes[..52],
            &hex_bytes(
                "3d2c1b0a51402f1ed9c8b7a695847362080706050403020118171615141312112827262524232221383736353433323101070500"
            )[..]
        );
        assert_eq!(
            &step_bytes[52..74],
            &hex_bytes("6500670069006b006d006f0071007300750077007900")[..]
        );
        assert_eq!(
            &step_bytes[74..96],
            &hex_bytes("660068006a006c006e00700072007400760078007a00")[..]
        );
        assert_eq!(
            hex_digest(step_bytes),
            "375b3a3e8996a37633df078068fbf8fb0e5d7863cb8a2276df1e552cbab33f92"
        );
    }

    #[test]
    fn publication_report_bytes_match_the_independent_cpython_oracle() {
        let report = TemporalCheckpointPublicationReportV1 {
            outcome: TEMPORAL_CHECKPOINT_PUBLISHED_BUT_NOT_CONFIRMED_DURABLE,
            temp_cleanup: TEMPORAL_CHECKPOINT_CLEANUP_ORPHANED,
            cleanup_error_count: 4,
            primary_error: TemporalCheckpointPublicationErrorV1 {
                stage: TEMPORAL_CHECKPOINT_ERROR_STAGE_EXCLUSIVE_PUBLISH,
                domain: TEMPORAL_CHECKPOINT_ERROR_DOMAIN_WIN32,
                code: 0x1122_3344,
            },
            cleanup_errors: [
                TemporalCheckpointPublicationErrorV1 {
                    stage: TEMPORAL_CHECKPOINT_ERROR_STAGE_TEMP_CLEANUP,
                    domain: TEMPORAL_CHECKPOINT_ERROR_DOMAIN_POLICY,
                    code: 0x0102_0304,
                },
                TemporalCheckpointPublicationErrorV1 {
                    stage: TEMPORAL_CHECKPOINT_ERROR_STAGE_PARENT_SYNC,
                    domain: TEMPORAL_CHECKPOINT_ERROR_DOMAIN_POSIX,
                    code: 0x5566_7788,
                },
                TemporalCheckpointPublicationErrorV1 {
                    stage: TEMPORAL_CHECKPOINT_ERROR_STAGE_FILE_CLOSE,
                    domain: TEMPORAL_CHECKPOINT_ERROR_DOMAIN_WIN32,
                    code: 0x99aa_bbcc,
                },
                TemporalCheckpointPublicationErrorV1 {
                    stage: TEMPORAL_CHECKPOINT_ERROR_STAGE_TEMP_IDENTITY,
                    domain: TEMPORAL_CHECKPOINT_ERROR_DOMAIN_ENCODER,
                    code: 0xddee_ff00,
                },
            ],
            owned_temp_pid: 0xa1b2_c3d4,
            owned_temp_nonce: 0x0102_0304_0506_0708,
            checkpoint_bytes: 0x1112_1314_1516_1718,
            encoded_bytes: 0x2122_2324_2526_2728,
            reserved_zero: [0; 8],
        };
        let bytes = raw_bytes(&report);
        assert_eq!(
            bytes,
            hex_bytes(
                "0400020409000200443322110a000300040302010b0001008877665507000200ccbbaa990800040000ffeeddd4c3b2a10807060504030201181716151413121128272625242322210000000000000000"
            )
        );
        assert_eq!(
            hex_digest(bytes),
            "bc3eb339a272206fc54074faa215cffb00b4a0469c5250efdea0eaecc0d63d5a"
        );
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let high = (pair[0] as char).to_digit(16).unwrap() as u8;
                let low = (pair[1] as char).to_digit(16).unwrap() as u8;
                high << 4 | low
            })
            .collect()
    }

    fn hex_digest(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    fn odd_byte_offset(base: *const u8) -> usize {
        if (base as usize + 1).is_multiple_of(2) {
            2
        } else {
            1
        }
    }

    struct Fixture {
        authority: TemporalValidatedCheckpointAuthorityV1,
        config: TemporalCircuitConfigV1,
        token_codes: [TemporalTokenCodeV1; 1],
        thresholds: Vec<f64>,
        checkpoint: Vec<u8>,
    }

    struct EmptyDefinitionFixture {
        authority: TemporalValidatedCheckpointAuthorityV1,
        config: TemporalCircuitConfigV1,
        token_codes: [TemporalTokenCodeV1; 1],
        thresholds: Vec<f64>,
    }

    impl EmptyDefinitionFixture {
        fn new(context_policy: u8, plasticity_policy: u8) -> Self {
            Self {
                authority: TemporalValidatedCheckpointAuthorityV1 {
                    protocol_identity_sha256: [0xa5; 32],
                    source_bundle_identity_sha256: std::array::from_fn(|index| index as u8),
                },
                config: TemporalCircuitConfigV1 {
                    column_count: 11,
                    max_pyramidal_cells: 26,
                    segment_capacity: 6,
                    construction_seed: 0x0102_0304,
                    connected_permanence_ticks: 250,
                    initial_permanence_ticks: 255,
                    permanence_increment_ticks: 50,
                    negative_decrement_ticks: 2,
                    base_cells_per_column: 2,
                    novel_race_cap: 2,
                    prediction_quorum: 1,
                    max_segments_per_cell: 1,
                    theta_match: 18,
                    theta_active: 20,
                    context_policy,
                    plasticity_policy,
                },
                token_codes: [TemporalTokenCodeV1 {
                    token_id: 7,
                    columns: std::array::from_fn(|index| index as u32),
                }],
                thresholds: vec![0.0; 85],
            }
        }

        fn definition(&self) -> TemporalCircuitDefinitionV1 {
            TemporalCircuitDefinitionV1 {
                config: self.config,
                token_codes: self.token_codes.as_ptr(),
                token_code_count: self.token_codes.len() as u64,
                threshold_offsets_mv: self.thresholds.as_ptr(),
                threshold_offset_count: self.thresholds.len() as u64,
                authority: self.authority,
            }
        }

        fn independently_constructed_checkpoint(&self) -> Vec<u8> {
            let config = self.config.into_runtime().unwrap();
            let runtime = TemporalCircuitRuntime::new(
                config,
                self.token_codes
                    .iter()
                    .map(|code| TokenCode {
                        token_id: code.token_id,
                        columns: code.columns,
                    })
                    .collect(),
                (0..config.column_count)
                    .map(|column| {
                        let base_width = u32::from(config.base_cells_per_column);
                        (0..base_width)
                            .map(|offset| column * base_width + offset)
                            .collect()
                    })
                    .collect(),
                self.thresholds.clone(),
                &[],
            )
            .unwrap();
            let prepared = runtime
                .prepare_checkpoint_v1_export(self.authority.into_validated())
                .unwrap();
            let mut sink = Cursor::new(Vec::new());
            prepared.encode_into(&mut sink).unwrap();
            sink.into_inner()
        }
    }

    unsafe fn create_from_definition(
        definition: &TemporalCircuitDefinitionV1,
    ) -> (*mut TemporalCircuitRuntimeHandle, TemporalCircuitReportV1) {
        let mut handle = ptr::null_mut();
        let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
        let status = humanbrain_temporal_circuit_runtime_create_v1(
            definition,
            &mut handle,
            report.as_mut_ptr(),
        );
        let report = report.assume_init();
        assert_eq!(status, report.status);
        (handle, report)
    }

    unsafe fn import_from_definition(
        definition: TemporalCircuitDefinitionV1,
        checkpoint: &[u8],
    ) -> (*mut TemporalCircuitRuntimeHandle, TemporalCircuitReportV1) {
        let request = TemporalCircuitImportCheckpointV1Request {
            definition,
            checkpoint_bytes: checkpoint.as_ptr(),
            checkpoint_len: checkpoint.len() as u64,
        };
        let mut handle = ptr::null_mut();
        let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
        let status = humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
            &request,
            &mut handle,
            report.as_mut_ptr(),
        );
        let report = report.assume_init();
        assert_eq!(status, report.status);
        (handle, report)
    }

    unsafe fn export_checkpoint(
        handle: *const TemporalCircuitRuntimeHandle,
        authority: &TemporalValidatedCheckpointAuthorityV1,
    ) -> Vec<u8> {
        let mut size_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
        assert_eq!(
            humanbrain_temporal_circuit_runtime_checkpoint_size_v1(
                handle,
                authority,
                size_report.as_mut_ptr(),
            ),
            TEMPORAL_RUNTIME_OK
        );
        let required = size_report.assume_init().required;
        let mut checkpoint = vec![0u8; usize::try_from(required).unwrap()];
        let mut export_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
        assert_eq!(
            humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                handle,
                authority,
                checkpoint.as_mut_ptr(),
                checkpoint.len() as u64,
                export_report.as_mut_ptr(),
            ),
            TEMPORAL_RUNTIME_OK
        );
        assert_eq!(export_report.assume_init().observed, required);
        checkpoint
    }

    fn fixture() -> Fixture {
        fixture_with_poison(None)
    }

    fn fixture_with_poison(poison: Option<TemporalPoisonCode>) -> Fixture {
        let authority = TemporalValidatedCheckpointAuthorityV1 {
            protocol_identity_sha256: [0x5a; 32],
            source_bundle_identity_sha256: [0xc3; 32],
        };
        let config = TemporalCircuitConfigV1 {
            column_count: ACTIVE_COLUMNS as u32,
            max_pyramidal_cells: ACTIVE_COLUMNS as u32,
            segment_capacity: 16,
            construction_seed: 0x1020_3040,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 8,
            negative_decrement_ticks: 1,
            base_cells_per_column: 1,
            novel_race_cap: 1,
            prediction_quorum: ACTIVE_COLUMNS as u8,
            max_segments_per_cell: 1,
            theta_match: 10,
            theta_active: 10,
            context_policy: CONTEXT_BOUND_WINNERS,
            plasticity_policy: PLASTICITY_GROWTH_OFF,
        };
        let token_codes = [TemporalTokenCodeV1 {
            token_id: 7,
            columns: std::array::from_fn(|index| index as u32),
        }];
        let thresholds = vec![0.0; 55];
        let mut runtime = TemporalCircuitRuntime::new(
            config.into_runtime().unwrap(),
            vec![TokenCode {
                token_id: token_codes[0].token_id,
                columns: token_codes[0].columns,
            }],
            (0..ACTIVE_COLUMNS)
                .map(|column| vec![column as u32])
                .collect(),
            thresholds.clone(),
            &[] as &[TemporalSegmentInit],
        )
        .unwrap();
        assert_eq!(runtime.recognize(7).unwrap().event_count, 1);
        runtime.scientific_stop = poison;
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority.into_validated())
            .unwrap();
        let mut sink = Cursor::new(Vec::new());
        prepared.encode_into(&mut sink).unwrap();
        let checkpoint = sink.into_inner();
        assert_eq!(checkpoint.len() as u64, prepared.encoded_len());
        Fixture {
            authority,
            config,
            token_codes,
            thresholds,
            checkpoint,
        }
    }

    unsafe fn import_fixture(
        fixture: &Fixture,
    ) -> (*mut TemporalCircuitRuntimeHandle, TemporalCircuitReportV1) {
        let mut unaligned_checkpoint = vec![0xa5; fixture.checkpoint.len() + 3];
        let checkpoint_offset = odd_byte_offset(unaligned_checkpoint.as_ptr());
        unaligned_checkpoint[checkpoint_offset..checkpoint_offset + fixture.checkpoint.len()]
            .copy_from_slice(&fixture.checkpoint);
        let checkpoint_bytes = unaligned_checkpoint.as_ptr().add(checkpoint_offset);
        assert!(!(checkpoint_bytes as usize).is_multiple_of(2));
        let request = TemporalCircuitImportCheckpointV1Request {
            definition: TemporalCircuitDefinitionV1 {
                config: fixture.config,
                token_codes: fixture.token_codes.as_ptr(),
                token_code_count: fixture.token_codes.len() as u64,
                threshold_offsets_mv: fixture.thresholds.as_ptr(),
                threshold_offset_count: fixture.thresholds.len() as u64,
                authority: fixture.authority,
            },
            checkpoint_bytes,
            checkpoint_len: fixture.checkpoint.len() as u64,
        };
        let mut handle = ptr::null_mut();
        let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
        assert_eq!(
            humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                &request,
                &mut handle,
                report.as_mut_ptr(),
            ),
            TEMPORAL_RUNTIME_OK
        );
        (handle, report.assume_init())
    }

    unsafe fn advance(
        handle: *mut TemporalCircuitRuntimeHandle,
        tokens: &[u32],
        output: *mut TemporalCircuitStepV2,
        output_capacity: u64,
        reset_before: u8,
    ) -> (u32, TemporalCircuitReportV1) {
        let request = TemporalCircuitAdvanceBlockV2Request {
            token_ids: if tokens.is_empty() {
                ptr::null()
            } else {
                tokens.as_ptr()
            },
            output,
            token_count: tokens.len() as u64,
            output_capacity,
            mode: MODE_READ_ONLY,
            reset_before,
            reserved_zero: [0; 6],
        };
        let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
        let status = humanbrain_temporal_circuit_runtime_advance_block_v2(
            handle,
            &request,
            report.as_mut_ptr(),
        );
        (status, report.assume_init())
    }

    fn sentinel_step(token_id: u32) -> TemporalCircuitStepV2 {
        let spike = TemporalSpikeV1 {
            cell_id: 0xa1b2_c3d4,
            tick: 0x5e6f,
            reserved_zero: 0,
        };
        TemporalCircuitStepV2 {
            token_id,
            punished_segments: 0x1122_3344,
            event_count: u64::MAX,
            candidate_segments: u64::MAX,
            matching_segments: u64::MAX,
            active_segments: u64::MAX,
            postings_visited: u64::MAX,
            predicted: 0,
            predicted_columns: 0,
            allocated_segments: 0,
            reserved_zero: 0,
            inhibition_onset_ticks: [0x1234; ACTIVE_COLUMNS],
            column_simulated_ticks: [0x5678; ACTIVE_COLUMNS],
            winner_spikes: [spike; ACTIVE_COLUMNS],
            basket_spikes: [spike; ACTIVE_COLUMNS],
            proximal_spikes: [spike; ACTIVE_COLUMNS],
            proximal_relay_spikes: [spike; ACTIVE_COLUMNS],
            winner_relay_spikes: [spike; ACTIVE_COLUMNS],
            winner_context: [0x1020_3040; CONTEXT_CELLS],
            proximal_context: [0x5060_7080; CONTEXT_CELLS],
            context: [0x90a0_b0c0; CONTEXT_CELLS],
        }
    }

    #[test]
    fn import_size_and_export_are_authority_bound_and_short_export_is_untouched() {
        let fixture = fixture();
        unsafe {
            let (handle, imported) = import_fixture(&fixture);
            assert!(!handle.is_null());
            assert_eq!(imported.status, TEMPORAL_RUNTIME_OK);

            let mut size_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_checkpoint_size_v1(
                    handle,
                    &fixture.authority,
                    size_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            let size_report = size_report.assume_init();
            assert_eq!(size_report.required, fixture.checkpoint.len() as u64);

            let mut short = vec![0xa5; fixture.checkpoint.len() - 1];
            let mut short_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                    handle,
                    &fixture.authority,
                    short.as_mut_ptr(),
                    short.len() as u64,
                    short_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OUTPUT_CAPACITY
            );
            assert!(short.iter().all(|byte| *byte == 0xa5));
            let short_report = short_report.assume_init();
            assert_eq!(short_report.required, fixture.checkpoint.len() as u64);
            assert_eq!(short_report.observed, short.len() as u64);

            let mut exported = vec![0u8; fixture.checkpoint.len()];
            let mut export_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                    handle,
                    &fixture.authority,
                    exported.as_mut_ptr(),
                    exported.len() as u64,
                    export_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            assert_eq!(exported, fixture.checkpoint);
            assert_eq!(export_report.assume_init().observed, exported.len() as u64);

            let mut oversized = vec![0x6d; fixture.checkpoint.len() + 37];
            let mut oversized_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                    handle,
                    &fixture.authority,
                    oversized.as_mut_ptr(),
                    oversized.len() as u64,
                    oversized_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            assert_eq!(
                &oversized[..fixture.checkpoint.len()],
                fixture.checkpoint.as_slice()
            );
            assert!(oversized[fixture.checkpoint.len()..]
                .iter()
                .all(|byte| *byte == 0x6d));

            let wrong_authority = TemporalValidatedCheckpointAuthorityV1 {
                protocol_identity_sha256: [0x5b; 32],
                ..fixture.authority
            };
            let mut authority_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_checkpoint_size_v1(
                    handle,
                    &wrong_authority,
                    authority_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_DEFINITION
            );
            assert_eq!(
                authority_report.assume_init().status,
                TEMPORAL_RUNTIME_INVALID_DEFINITION
            );
            let mut authority_output = vec![0x91; fixture.checkpoint.len()];
            let mut authority_export_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                    handle,
                    &wrong_authority,
                    authority_output.as_mut_ptr(),
                    authority_output.len() as u64,
                    authority_export_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_DEFINITION
            );
            assert!(authority_output.iter().all(|byte| *byte == 0x91));

            humanbrain_temporal_circuit_runtime_destroy(handle);
            humanbrain_temporal_circuit_runtime_destroy(ptr::null_mut());
        }
    }

    #[test]
    fn checkpoint_identity_matches_the_canonical_checkpoint_header() {
        let fixture = fixture();
        unsafe {
            let (handle, imported) = import_fixture(&fixture);
            assert_eq!(imported.status, TEMPORAL_RUNTIME_OK);

            let mut identity = MaybeUninit::<TemporalCheckpointIdentityV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            let status = humanbrain_temporal_circuit_runtime_checkpoint_identity_v1(
                handle,
                &fixture.authority,
                identity.as_mut_ptr(),
                report.as_mut_ptr(),
            );

            assert_eq!(status, TEMPORAL_RUNTIME_OK);
            assert_eq!(identity.assume_init().header, fixture.checkpoint[..256]);
            let report = report.assume_init();
            assert_eq!(report.status, TEMPORAL_RUNTIME_OK);
            assert_eq!(report.required, 256);
            assert_eq!(report.observed, 256);
            assert_eq!(report.location, 0);

            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn checkpoint_identity_rejects_invalid_inputs_and_wrong_authority_with_zero_output() {
        let fixture = fixture();
        unsafe {
            let (handle, imported) = import_fixture(&fixture);
            assert_eq!(imported.status, TEMPORAL_RUNTIME_OK);
            let wrong_authority = TemporalValidatedCheckpointAuthorityV1 {
                protocol_identity_sha256: [0x5b; 32],
                ..fixture.authority
            };
            let mut identity = TemporalCheckpointIdentityV1 {
                header: [0xa5; CHECKPOINT_V1_HEADER_BYTES as usize],
            };
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();

            let status = humanbrain_temporal_circuit_runtime_checkpoint_identity_v1(
                handle,
                &wrong_authority,
                &mut identity,
                report.as_mut_ptr(),
            );

            assert_eq!(status, TEMPORAL_RUNTIME_INVALID_DEFINITION);
            assert_eq!(identity, TemporalCheckpointIdentityV1::ZERO);
            let report = report.assume_init();
            assert_eq!(report.status, status);
            assert_eq!(report.required, 0);
            assert_eq!(report.observed, 0);
            assert_eq!(report.location, 0);

            identity.header.fill(0xb6);
            let mut null_handle_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            let null_handle_status = humanbrain_temporal_circuit_runtime_checkpoint_identity_v1(
                ptr::null(),
                &fixture.authority,
                &mut identity,
                null_handle_report.as_mut_ptr(),
            );
            assert_eq!(null_handle_status, TEMPORAL_RUNTIME_INVALID_ARGUMENT);
            assert_eq!(identity, TemporalCheckpointIdentityV1::ZERO);
            assert_eq!(null_handle_report.assume_init().status, null_handle_status);

            identity.header.fill(0xc7);
            let mut null_authority_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            let null_authority_status = humanbrain_temporal_circuit_runtime_checkpoint_identity_v1(
                handle,
                ptr::null(),
                &mut identity,
                null_authority_report.as_mut_ptr(),
            );
            assert_eq!(null_authority_status, TEMPORAL_RUNTIME_INVALID_ARGUMENT);
            assert_eq!(identity, TemporalCheckpointIdentityV1::ZERO);
            assert_eq!(
                null_authority_report.assume_init().status,
                null_authority_status
            );

            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn block_reset_is_first_token_transactional_and_prefix_failures_are_exact() {
        let fixture = fixture();
        unsafe {
            let (handle, _) = import_fixture(&fixture);

            let mut ordinary = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            let (status, report) = advance(handle, &[7], ordinary.as_mut_ptr(), 1, 0);
            assert_eq!(status, TEMPORAL_RUNTIME_OK);
            assert_eq!(report.completed_prefix, 1);
            assert_eq!(ordinary.assume_init().event_count, 2);

            let (status, report) = advance(handle, &[], ptr::null_mut(), 0, 1);
            assert_eq!(status, TEMPORAL_RUNTIME_INVALID_ARGUMENT);
            assert_eq!(report.completed_prefix, 0);

            let mut unknown_output = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            let (status, report) = advance(handle, &[99], unknown_output.as_mut_ptr(), 1, 1);
            assert_eq!(status, TEMPORAL_RUNTIME_UNKNOWN_TOKEN);
            assert_eq!(report.completed_prefix, 0);
            assert_eq!(report.observed, 99);

            let mut after_unknown = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            let (status, _) = advance(handle, &[7], after_unknown.as_mut_ptr(), 1, 0);
            assert_eq!(status, TEMPORAL_RUNTIME_OK);
            assert_eq!(after_unknown.assume_init().event_count, 3);

            let mut reset = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            let (status, _) = advance(handle, &[7], reset.as_mut_ptr(), 1, 1);
            assert_eq!(status, TEMPORAL_RUNTIME_OK);
            assert_eq!(reset.assume_init().event_count, 1);

            let (status, report) = advance(handle, &[7], ptr::null_mut(), 0, 0);
            assert_eq!(status, TEMPORAL_RUNTIME_OUTPUT_CAPACITY);
            assert_eq!(report.required, 1);
            assert_eq!(report.observed, 0);

            let sentinel = TemporalCircuitStepV2 {
                token_id: 0xfeed_beef,
                punished_segments: 0,
                event_count: u64::MAX,
                candidate_segments: 0,
                matching_segments: 0,
                active_segments: 0,
                postings_visited: 0,
                predicted: 0,
                predicted_columns: 0,
                allocated_segments: 0,
                reserved_zero: 0,
                inhibition_onset_ticks: [0; ACTIVE_COLUMNS],
                column_simulated_ticks: [0; ACTIVE_COLUMNS],
                winner_spikes: [TemporalSpikeV1 {
                    cell_id: 0,
                    tick: 0,
                    reserved_zero: 0,
                }; ACTIVE_COLUMNS],
                basket_spikes: [TemporalSpikeV1 {
                    cell_id: 0,
                    tick: 0,
                    reserved_zero: 0,
                }; ACTIVE_COLUMNS],
                proximal_spikes: [TemporalSpikeV1 {
                    cell_id: 0,
                    tick: 0,
                    reserved_zero: 0,
                }; ACTIVE_COLUMNS],
                proximal_relay_spikes: [TemporalSpikeV1 {
                    cell_id: 0,
                    tick: 0,
                    reserved_zero: 0,
                }; ACTIVE_COLUMNS],
                winner_relay_spikes: [TemporalSpikeV1 {
                    cell_id: 0,
                    tick: 0,
                    reserved_zero: 0,
                }; ACTIVE_COLUMNS],
                winner_context: [0; CONTEXT_CELLS],
                proximal_context: [0; CONTEXT_CELLS],
                context: [0; CONTEXT_CELLS],
            };
            let mut prefix = [sentinel; 3];
            let (status, report) = advance(handle, &[7, 99, 7], prefix.as_mut_ptr(), 3, 0);
            assert_eq!(status, TEMPORAL_RUNTIME_UNKNOWN_TOKEN);
            assert_eq!(report.completed_prefix, 1);
            assert_eq!(prefix[0].token_id, 7);
            assert_eq!(prefix[0].event_count, 2);
            assert_eq!(prefix[1].token_id, 0xfeed_beef);
            assert_eq!(prefix[2].token_id, 0xfeed_beef);

            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn reset_first_terminal_stop_retains_transient_state_and_only_latches_poison() {
        let fixture = fixture();
        unsafe {
            let (handle, _) = import_fixture(&fixture);
            let (before_context, before_winner_context, before_event) = {
                let mut runtime = lock_runtime(&*handle);
                let before = (
                    runtime.previous_context,
                    runtime.previous_winner_context,
                    runtime.event_count,
                );
                let first_proximal = runtime.proximal_offset as usize;
                runtime.threshold_offsets_mv[first_proximal] = f64::NAN;
                before
            };

            let mut output = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            let (status, report) = advance(handle, &[7], output.as_mut_ptr(), 1, 1);
            assert_eq!(status, TEMPORAL_RUNTIME_SCIENTIFIC_STOP);
            assert_eq!(report.completed_prefix, 0);
            assert_eq!(
                report.poison_code,
                TemporalPoisonCode::ProximalPhysics as u16
            );
            assert_eq!(report.stage_code, STAGE_PROXIMAL);
            {
                let runtime = lock_runtime(&*handle);
                assert_eq!(runtime.previous_context, before_context);
                assert_eq!(runtime.previous_winner_context, before_winner_context);
                assert_eq!(runtime.event_count, before_event);
                assert_eq!(
                    runtime.scientific_stop,
                    Some(TemporalPoisonCode::ProximalPhysics)
                );
            }

            let (status, report) = advance(handle, &[7], output.as_mut_ptr(), 1, 0);
            assert_eq!(status, TEMPORAL_RUNTIME_POISONED);
            assert_eq!(
                report.poison_code,
                TemporalPoisonCode::ProximalPhysics as u16
            );
            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn imported_runtime_owns_definition_and_checkpoint_after_foreign_inputs_change_and_drop() {
        let fixture = fixture();
        unsafe {
            let handle = {
                let mut token_codes = fixture.token_codes;
                let mut thresholds = fixture.thresholds.clone();
                let mut checkpoint_backing = vec![0x44; fixture.checkpoint.len() + 3];
                let checkpoint_offset = odd_byte_offset(checkpoint_backing.as_ptr());
                checkpoint_backing[checkpoint_offset..checkpoint_offset + fixture.checkpoint.len()]
                    .copy_from_slice(&fixture.checkpoint);
                let checkpoint_bytes = checkpoint_backing.as_ptr().add(checkpoint_offset);
                assert!(!(checkpoint_bytes as usize).is_multiple_of(2));
                let request = TemporalCircuitImportCheckpointV1Request {
                    definition: TemporalCircuitDefinitionV1 {
                        config: fixture.config,
                        token_codes: token_codes.as_ptr(),
                        token_code_count: token_codes.len() as u64,
                        threshold_offsets_mv: thresholds.as_ptr(),
                        threshold_offset_count: thresholds.len() as u64,
                        authority: fixture.authority,
                    },
                    checkpoint_bytes,
                    checkpoint_len: fixture.checkpoint.len() as u64,
                };
                let mut handle = ptr::null_mut();
                let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
                assert_eq!(
                    humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                        &request,
                        &mut handle,
                        report.as_mut_ptr(),
                    ),
                    TEMPORAL_RUNTIME_OK
                );
                assert_eq!(report.assume_init().status, TEMPORAL_RUNTIME_OK);

                token_codes[0].token_id = u32::MAX;
                assert_eq!(token_codes[0].token_id, u32::MAX);
                thresholds.fill(f64::NAN);
                checkpoint_backing.fill(0);
                handle
            };

            let mut exported = vec![0u8; fixture.checkpoint.len()];
            let mut export_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                    handle,
                    &fixture.authority,
                    exported.as_mut_ptr(),
                    exported.len() as u64,
                    export_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            assert_eq!(exported, fixture.checkpoint);

            let mut step = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            let (status, report) = advance(handle, &[7], step.as_mut_ptr(), 1, 0);
            assert_eq!(status, TEMPORAL_RUNTIME_OK);
            assert_eq!(report.completed_prefix, 1);
            assert_eq!(step.assume_init().event_count, 2);
            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn imported_poisoned_checkpoint_remains_exportable_and_rejects_zero_work_advance() {
        let fixture = fixture_with_poison(Some(TemporalPoisonCode::ProximalPhysics));
        unsafe {
            let (handle, _) = import_fixture(&fixture);
            let mut size_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_checkpoint_size_v1(
                    handle,
                    &fixture.authority,
                    size_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            assert_eq!(
                size_report.assume_init().required,
                fixture.checkpoint.len() as u64
            );

            let mut exported = vec![0u8; fixture.checkpoint.len()];
            let mut export_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                    handle,
                    &fixture.authority,
                    exported.as_mut_ptr(),
                    exported.len() as u64,
                    export_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            assert_eq!(exported, fixture.checkpoint);

            let (status, report) = advance(handle, &[], ptr::null_mut(), 0, 0);
            assert_eq!(status, TEMPORAL_RUNTIME_POISONED);
            assert_eq!(report.completed_prefix, 0);
            assert_eq!(
                report.poison_code,
                TemporalPoisonCode::ProximalPhysics as u16
            );
            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn invalid_block_geometry_tags_and_reserved_bytes_never_touch_output() {
        let fixture = fixture();
        unsafe {
            let (handle, _) = import_fixture(&fixture);
            let token = 7u32;
            for (mode, reset_before, reserved_zero) in [
                (2, 0, [0; 6]),
                (MODE_READ_ONLY, 2, [0; 6]),
                (MODE_READ_ONLY, 0, [1, 0, 0, 0, 0, 0]),
            ] {
                let mut output = MaybeUninit::new(sentinel_step(0xfeed_beef));
                let before = raw_bytes(output.assume_init_ref()).to_vec();
                let request = TemporalCircuitAdvanceBlockV2Request {
                    token_ids: &token,
                    output: output.as_mut_ptr(),
                    token_count: 1,
                    output_capacity: 1,
                    mode,
                    reset_before,
                    reserved_zero,
                };
                let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
                assert_eq!(
                    humanbrain_temporal_circuit_runtime_advance_block_v2(
                        handle,
                        &request,
                        report.as_mut_ptr(),
                    ),
                    TEMPORAL_RUNTIME_INVALID_ARGUMENT
                );
                assert_eq!(raw_bytes(output.assume_init_ref()), before.as_slice());
            }

            let mut token_backing = [0u8; 8];
            let token_offset = (0..4)
                .find(|offset| !(token_backing.as_ptr().add(*offset) as usize).is_multiple_of(4))
                .unwrap();
            token_backing[token_offset..token_offset + 4].copy_from_slice(&7u32.to_ne_bytes());
            let mut output = MaybeUninit::new(sentinel_step(0xfeed_beef));
            let before = raw_bytes(output.assume_init_ref()).to_vec();
            let request = TemporalCircuitAdvanceBlockV2Request {
                token_ids: token_backing.as_ptr().add(token_offset).cast(),
                output: output.as_mut_ptr(),
                token_count: 1,
                output_capacity: 1,
                mode: MODE_READ_ONLY,
                reset_before: 0,
                reserved_zero: [0; 6],
            };
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_advance_block_v2(
                    handle,
                    &request,
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(raw_bytes(output.assume_init_ref()), before.as_slice());

            let overflow_request = TemporalCircuitAdvanceBlockV2Request {
                token_ids: &token,
                output: output.as_mut_ptr(),
                token_count: u64::MAX,
                output_capacity: 1,
                mode: MODE_READ_ONLY,
                reset_before: 0,
                reserved_zero: [0; 6],
            };
            let mut overflow_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_advance_block_v2(
                    handle,
                    &overflow_request,
                    overflow_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(raw_bytes(output.assume_init_ref()), before.as_slice());

            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn import_policy_tag_and_nonnull_output_precedence_are_exact() {
        let fixture = fixture();
        let mut invalid_config = fixture.config;
        invalid_config.context_policy = 0;
        let request = TemporalCircuitImportCheckpointV1Request {
            definition: TemporalCircuitDefinitionV1 {
                config: invalid_config,
                token_codes: fixture.token_codes.as_ptr(),
                token_code_count: fixture.token_codes.len() as u64,
                threshold_offsets_mv: fixture.thresholds.as_ptr(),
                threshold_offset_count: fixture.thresholds.len() as u64,
                authority: fixture.authority,
            },
            checkpoint_bytes: fixture.checkpoint.as_ptr(),
            checkpoint_len: fixture.checkpoint.len() as u64,
        };
        unsafe {
            let mut handle = ptr::null_mut();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                    &request,
                    &mut handle,
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_DEFINITION
            );
            assert!(handle.is_null());

            let mut nonnull = ptr::NonNull::<TemporalCircuitRuntimeHandle>::dangling().as_ptr();
            let retained = nonnull;
            let mut precedence_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                    ptr::null(),
                    &mut nonnull,
                    precedence_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(nonnull, retained);
        }
    }

    #[test]
    fn import_rejects_bad_checkpoint_without_publishing_a_handle() {
        let fixture = fixture();
        let bad_checkpoint = [0u8; 32];
        let request = TemporalCircuitImportCheckpointV1Request {
            definition: TemporalCircuitDefinitionV1 {
                config: fixture.config,
                token_codes: fixture.token_codes.as_ptr(),
                token_code_count: fixture.token_codes.len() as u64,
                threshold_offsets_mv: fixture.thresholds.as_ptr(),
                threshold_offset_count: fixture.thresholds.len() as u64,
                authority: fixture.authority,
            },
            checkpoint_bytes: bad_checkpoint.as_ptr(),
            checkpoint_len: bad_checkpoint.len() as u64,
        };
        let mut handle = ptr::null_mut();
        let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
        unsafe {
            assert_eq!(
                humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                    &request,
                    &mut handle,
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_CHECKPOINT_REJECTED
            );
            assert!(handle.is_null());
            assert_eq!(
                report.assume_init().status,
                TEMPORAL_RUNTIME_CHECKPOINT_REJECTED
            );
        }
    }

    #[test]
    fn fresh_create_matches_exact_empty_import_across_all_policy_combinations() {
        let mut definition_digests = Vec::new();
        for context_policy in [CONTEXT_BOUND_WINNERS, CONTEXT_GENERIC_AFFERENTS] {
            for plasticity_policy in [PLASTICITY_GROWTH_OFF, PLASTICITY_PLASTIC] {
                let fixture = EmptyDefinitionFixture::new(context_policy, plasticity_policy);
                let definition = fixture.definition();
                let expected = fixture.independently_constructed_checkpoint();
                assert_eq!(expected.len(), 256);
                definition_digests.push(<[u8; 32]>::from(Sha256::digest(&expected)));

                unsafe {
                    let (created, create_report) = create_from_definition(&definition);
                    assert_eq!(create_report.status, TEMPORAL_RUNTIME_OK);
                    assert!(!created.is_null());
                    assert_eq!(export_checkpoint(created, &fixture.authority), expected);

                    let (imported, import_report) = import_from_definition(definition, &expected);
                    assert_eq!(import_report.status, TEMPORAL_RUNTIME_OK);
                    assert!(!imported.is_null());
                    assert_eq!(export_checkpoint(imported, &fixture.authority), expected);

                    let mut created_step = MaybeUninit::<TemporalCircuitStepV2>::uninit();
                    let mut imported_step = MaybeUninit::<TemporalCircuitStepV2>::uninit();
                    assert_eq!(
                        advance(created, &[7], created_step.as_mut_ptr(), 1, 0).0,
                        TEMPORAL_RUNTIME_OK
                    );
                    assert_eq!(
                        advance(imported, &[7], imported_step.as_mut_ptr(), 1, 0).0,
                        TEMPORAL_RUNTIME_OK
                    );
                    let created_step = created_step.assume_init();
                    let imported_step = imported_step.assume_init();
                    assert_eq!(created_step, imported_step);
                    assert_eq!(created_step.event_count, 1);

                    humanbrain_temporal_circuit_runtime_destroy(created);
                    humanbrain_temporal_circuit_runtime_destroy(imported);
                }
            }
        }
        for left in 0..definition_digests.len() {
            for right in left + 1..definition_digests.len() {
                assert_ne!(definition_digests[left], definition_digests[right]);
            }
        }
    }

    #[test]
    fn fresh_create_matches_the_independent_cpython_empty_checkpoint() {
        let fixture = EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
        let expected = cpython_empty_full_section_generic_checkpoint_v1();
        assert_eq!(expected.len(), 256);
        assert_eq!(
            hex_digest(&expected),
            "6aedd76db5a1bb53b33c26b5c7bd6aa117d9eb099fe2ab59b52b9a8ee01e6ee0"
        );

        unsafe {
            let (handle, report) = create_from_definition(&fixture.definition());
            assert_eq!(report.status, TEMPORAL_RUNTIME_OK);
            assert_eq!(export_checkpoint(handle, &fixture.authority), expected);
            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn fresh_create_owns_foreign_definition_and_resumes_from_its_checkpoint() {
        let authority = TemporalValidatedCheckpointAuthorityV1 {
            protocol_identity_sha256: [0xa5; 32],
            source_bundle_identity_sha256: std::array::from_fn(|index| index as u8),
        };
        unsafe {
            let handle = {
                let mut fixture =
                    EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
                let definition = fixture.definition();
                let (handle, report) = create_from_definition(&definition);
                assert_eq!(report.status, TEMPORAL_RUNTIME_OK);
                assert!(!handle.is_null());

                fixture.token_codes[0].token_id = u32::MAX;
                fixture.thresholds.fill(f64::NAN);
                assert_eq!(fixture.token_codes[0].token_id, u32::MAX);
                assert!(fixture.thresholds.iter().all(|value| value.is_nan()));
                handle
            };

            let mut first = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            let (status, report) = advance(handle, &[7], first.as_mut_ptr(), 1, 0);
            assert_eq!(status, TEMPORAL_RUNTIME_OK);
            assert_eq!(report.completed_prefix, 1);
            assert_eq!(first.assume_init().event_count, 1);

            let checkpoint = export_checkpoint(handle, &authority);
            let resume_fixture =
                EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
            let (resumed, report) =
                import_from_definition(resume_fixture.definition(), &checkpoint);
            assert_eq!(report.status, TEMPORAL_RUNTIME_OK);

            let mut uninterrupted = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            let mut resumed_step = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            assert_eq!(
                advance(handle, &[7], uninterrupted.as_mut_ptr(), 1, 0).0,
                TEMPORAL_RUNTIME_OK
            );
            assert_eq!(
                advance(resumed, &[7], resumed_step.as_mut_ptr(), 1, 0).0,
                TEMPORAL_RUNTIME_OK
            );
            let uninterrupted = uninterrupted.assume_init();
            let resumed_step = resumed_step.assume_init();
            assert_eq!(uninterrupted, resumed_step);
            assert_eq!(uninterrupted.event_count, 2);

            humanbrain_temporal_circuit_runtime_destroy(handle);
            humanbrain_temporal_circuit_runtime_destroy(resumed);
        }
    }

    #[test]
    fn fresh_create_error_precedence_and_nonpublication_are_exact() {
        let fixture = EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
        let definition = fixture.definition();
        unsafe {
            let mut handle = ptr::null_mut();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_create_v1(
                    &definition,
                    &mut handle,
                    ptr::null_mut(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert!(handle.is_null());

            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_create_v1(
                    &definition,
                    ptr::null_mut(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );

            let mut nonnull = ptr::NonNull::<TemporalCircuitRuntimeHandle>::dangling().as_ptr();
            let retained = nonnull;
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_create_v1(
                    ptr::null(),
                    &mut nonnull,
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(nonnull, retained);

            let mut handle = ptr::null_mut();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_create_v1(
                    ptr::null(),
                    &mut handle,
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert!(handle.is_null());

            let definition_backing = vec![0u8; size_of::<TemporalCircuitDefinitionV1>() + 8];
            let definition_offset = (0..8)
                .find(|offset| {
                    !(definition_backing.as_ptr().add(*offset) as usize)
                        .is_multiple_of(align_of::<TemporalCircuitDefinitionV1>())
                })
                .unwrap();
            let misaligned_definition = definition_backing
                .as_ptr()
                .add(definition_offset)
                .cast::<TemporalCircuitDefinitionV1>();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_create_v1(
                    misaligned_definition,
                    &mut handle,
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert!(handle.is_null());

            let mut bad_geometry = definition;
            bad_geometry.token_code_count = u64::MAX;
            let (bad_handle, bad_report) = create_from_definition(&bad_geometry);
            assert!(bad_handle.is_null());
            assert_eq!(bad_report.status, TEMPORAL_RUNTIME_INVALID_ARGUMENT);

            let mut bad_token_geometry_and_tag = bad_geometry;
            bad_token_geometry_and_tag.config.context_policy = 0;
            let (bad_handle, bad_report) = create_from_definition(&bad_token_geometry_and_tag);
            assert!(bad_handle.is_null());
            assert_eq!(bad_report.status, TEMPORAL_RUNTIME_INVALID_ARGUMENT);
            assert_eq!(
                report_message(&bad_report),
                "token-code input geometry is invalid"
            );

            let token_backing =
                vec![0u8; size_of::<TemporalTokenCodeV1>() + align_of::<TemporalTokenCodeV1>()];
            let token_offset = (0..align_of::<TemporalTokenCodeV1>())
                .find(|offset| {
                    !(token_backing.as_ptr().add(*offset) as usize)
                        .is_multiple_of(align_of::<TemporalTokenCodeV1>())
                })
                .unwrap();
            let mut bad_token_geometry = definition;
            bad_token_geometry.token_codes = token_backing
                .as_ptr()
                .add(token_offset)
                .cast::<TemporalTokenCodeV1>();
            bad_token_geometry.token_code_count = 1;
            let (bad_handle, bad_report) = create_from_definition(&bad_token_geometry);
            assert!(bad_handle.is_null());
            assert_eq!(bad_report.status, TEMPORAL_RUNTIME_INVALID_ARGUMENT);

            let mut overflowing_threshold_geometry = definition;
            overflowing_threshold_geometry.threshold_offset_count = u64::MAX;
            let (bad_handle, bad_report) = create_from_definition(&overflowing_threshold_geometry);
            assert!(bad_handle.is_null());
            assert_eq!(bad_report.status, TEMPORAL_RUNTIME_INVALID_ARGUMENT);

            let mut bad_threshold_geometry_and_tag = overflowing_threshold_geometry;
            bad_threshold_geometry_and_tag.config.context_policy = 0;
            let (bad_handle, bad_report) = create_from_definition(&bad_threshold_geometry_and_tag);
            assert!(bad_handle.is_null());
            assert_eq!(bad_report.status, TEMPORAL_RUNTIME_INVALID_ARGUMENT);
            assert_eq!(
                report_message(&bad_report),
                "threshold input geometry is invalid"
            );

            let threshold_backing = [0u8; 16];
            let threshold_offset = (0..8)
                .find(|offset| {
                    !(threshold_backing.as_ptr().add(*offset) as usize)
                        .is_multiple_of(align_of::<f64>())
                })
                .unwrap();
            let mut bad_threshold_geometry = definition;
            bad_threshold_geometry.threshold_offsets_mv =
                threshold_backing.as_ptr().add(threshold_offset).cast();
            bad_threshold_geometry.threshold_offset_count = 1;
            let (bad_handle, bad_report) = create_from_definition(&bad_threshold_geometry);
            assert!(bad_handle.is_null());
            assert_eq!(bad_report.status, TEMPORAL_RUNTIME_INVALID_ARGUMENT);

            let mut invalid_tag = definition;
            invalid_tag.config.context_policy = 0;
            let (bad_handle, bad_report) = create_from_definition(&invalid_tag);
            assert!(bad_handle.is_null());
            assert_eq!(bad_report.status, TEMPORAL_RUNTIME_INVALID_DEFINITION);

            let duplicate_codes = [fixture.token_codes[0], fixture.token_codes[0]];
            let mut noncanonical = definition;
            noncanonical.token_codes = duplicate_codes.as_ptr();
            noncanonical.token_code_count = duplicate_codes.len() as u64;
            let (bad_handle, bad_report) = create_from_definition(&noncanonical);
            assert!(bad_handle.is_null());
            assert_eq!(bad_report.status, TEMPORAL_RUNTIME_INVALID_DEFINITION);

            fail_next_owned_allocation();
            let (bad_handle, bad_report) = create_from_definition(&definition);
            assert!(bad_handle.is_null());
            assert_eq!(bad_report.status, TEMPORAL_RUNTIME_ALLOCATION_FAILED);
        }
    }

    #[test]
    fn fresh_create_allocation_count_matches_exact_empty_checkpoint_import() {
        let fixture = EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
        let definition = fixture.definition();
        let checkpoint = fixture.independently_constructed_checkpoint();
        let request = TemporalCircuitImportCheckpointV1Request {
            definition,
            checkpoint_bytes: checkpoint.as_ptr(),
            checkpoint_len: checkpoint.len() as u64,
        };

        unsafe {
            let mut created = ptr::null_mut();
            let mut create_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            let (create_status, create_allocations) = count_allocations(|| {
                humanbrain_temporal_circuit_runtime_create_v1(
                    &definition,
                    &mut created,
                    create_report.as_mut_ptr(),
                )
            });

            let mut imported = ptr::null_mut();
            let mut import_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            let (import_status, import_allocations) = count_allocations(|| {
                humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                    &request,
                    &mut imported,
                    import_report.as_mut_ptr(),
                )
            });

            assert_eq!(create_status, TEMPORAL_RUNTIME_OK);
            assert_eq!(import_status, TEMPORAL_RUNTIME_OK);
            assert_eq!(create_report.assume_init().status, TEMPORAL_RUNTIME_OK);
            assert_eq!(import_report.assume_init().status, TEMPORAL_RUNTIME_OK);
            assert_eq!(create_allocations, import_allocations);
            assert!(create_allocations > 0);
            assert_eq!(
                export_checkpoint(created, &fixture.authority),
                export_checkpoint(imported, &fixture.authority)
            );

            humanbrain_temporal_circuit_runtime_destroy(created);
            humanbrain_temporal_circuit_runtime_destroy(imported);
        }
    }

    #[test]
    fn import_checkpoint_geometry_still_precedes_definition_tag_validation() {
        let fixture = EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
        let mut definition = fixture.definition();
        definition.config.context_policy = 0;
        let request = TemporalCircuitImportCheckpointV1Request {
            definition,
            checkpoint_bytes: ptr::null(),
            checkpoint_len: 1,
        };

        unsafe {
            let mut handle = ptr::null_mut();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                    &request,
                    &mut handle,
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert!(handle.is_null());
            assert_eq!(
                report.assume_init().status,
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
        }
    }

    #[test]
    fn cpython_checkpoint_round_trips_from_provably_unaligned_foreign_storage() {
        let literal = cpython_full_section_generic_checkpoint_v1();
        assert_eq!(
            hex_digest(&literal),
            "d64cf550c30b6bcf7b499d39cbdc9c7bafeaef6a2e23b795ffe512e37117eb80"
        );
        let authority = TemporalValidatedCheckpointAuthorityV1 {
            protocol_identity_sha256: [0xa5; 32],
            source_bundle_identity_sha256: std::array::from_fn(|index| index as u8),
        };
        let config = TemporalCircuitConfigV1 {
            column_count: 11,
            max_pyramidal_cells: 26,
            segment_capacity: 6,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            base_cells_per_column: 2,
            novel_race_cap: 2,
            prediction_quorum: 1,
            max_segments_per_cell: 1,
            theta_match: 18,
            theta_active: 20,
            context_policy: CONTEXT_GENERIC_AFFERENTS,
            plasticity_policy: PLASTICITY_PLASTIC,
        };

        unsafe {
            let handle = {
                let mut token_codes = [TemporalTokenCodeV1 {
                    token_id: 7,
                    columns: std::array::from_fn(|index| index as u32),
                }];
                let mut thresholds = vec![0.0; 85];
                let mut checkpoint_backing = vec![0x6d; literal.len() + 3];
                let checkpoint_offset = odd_byte_offset(checkpoint_backing.as_ptr());
                checkpoint_backing[checkpoint_offset..checkpoint_offset + literal.len()]
                    .copy_from_slice(&literal);
                let before = checkpoint_backing.clone();
                let checkpoint_bytes = checkpoint_backing.as_ptr().add(checkpoint_offset);
                assert!(!(checkpoint_bytes as usize).is_multiple_of(align_of::<u16>()));
                assert!(!(checkpoint_bytes as usize).is_multiple_of(align_of::<u32>()));
                assert!(!(checkpoint_bytes as usize).is_multiple_of(align_of::<u64>()));

                let request = TemporalCircuitImportCheckpointV1Request {
                    definition: TemporalCircuitDefinitionV1 {
                        config,
                        token_codes: token_codes.as_ptr(),
                        token_code_count: token_codes.len() as u64,
                        threshold_offsets_mv: thresholds.as_ptr(),
                        threshold_offset_count: thresholds.len() as u64,
                        authority,
                    },
                    checkpoint_bytes,
                    checkpoint_len: literal.len() as u64,
                };
                let mut handle = ptr::null_mut();
                let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
                assert_eq!(
                    humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                        &request,
                        &mut handle,
                        report.as_mut_ptr(),
                    ),
                    TEMPORAL_RUNTIME_OK
                );
                assert_eq!(report.assume_init().status, TEMPORAL_RUNTIME_OK);
                assert!(!handle.is_null());
                assert_eq!(checkpoint_backing, before);

                token_codes[0].token_id = u32::MAX;
                thresholds.fill(f64::NAN);
                checkpoint_backing.fill(0);
                assert_eq!(token_codes[0].token_id, u32::MAX);
                assert!(thresholds.iter().all(|value| value.is_nan()));
                assert!(checkpoint_backing.iter().all(|byte| *byte == 0));
                handle
            };

            let mut exported = vec![0u8; literal.len()];
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_v1(
                    handle,
                    &authority,
                    exported.as_mut_ptr(),
                    exported.len() as u64,
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            let report = report.assume_init();
            assert_eq!(report.required, 824);
            assert_eq!(report.observed, 824);
            assert_eq!(exported, literal);

            let mut output = MaybeUninit::<TemporalCircuitStepV2>::uninit();
            let (status, report) = advance(handle, &[7], output.as_mut_ptr(), 1, 0);
            assert_eq!(status, TEMPORAL_RUNTIME_OK);
            assert_eq!(report.completed_prefix, 1);
            assert_eq!(output.assume_init().token_id, 7);
            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn handle_allocation_failure_reports_retryable_status_without_publication() {
        let fixture = fixture();
        let request = TemporalCircuitImportCheckpointV1Request {
            definition: TemporalCircuitDefinitionV1 {
                config: fixture.config,
                token_codes: fixture.token_codes.as_ptr(),
                token_code_count: fixture.token_codes.len() as u64,
                threshold_offsets_mv: fixture.thresholds.as_ptr(),
                threshold_offset_count: fixture.thresholds.len() as u64,
                authority: fixture.authority,
            },
            checkpoint_bytes: fixture.checkpoint.as_ptr(),
            checkpoint_len: fixture.checkpoint.len() as u64,
        };
        let mut handle = ptr::null_mut();
        let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
        fail_next_owned_allocation();
        unsafe {
            assert_eq!(
                humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                    &request,
                    &mut handle,
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_ALLOCATION_FAILED
            );
            assert!(handle.is_null());
            assert_eq!(
                report.assume_init().status,
                TEMPORAL_RUNTIME_ALLOCATION_FAILED
            );
        }
    }

    #[test]
    fn path_publication_precedence_keeps_outcome_zero_until_preparation_completes() {
        let fixture = EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
        let definition = fixture.definition();
        unsafe {
            let (handle, create_report) = create_from_definition(&definition);
            assert_eq!(create_report.status, TEMPORAL_RUNTIME_OK);
            let before = export_checkpoint(handle, &fixture.authority);
            let parent = b"relative-parent";
            let basename = b"state.chk";
            let request = TemporalCheckpointPathV1Request {
                parent_utf8: parent.as_ptr(),
                parent_len: parent.len() as u64,
                final_basename_utf8: basename.as_ptr(),
                final_basename_len: basename.len() as u64,
                reserved_zero: [0; 8],
            };

            let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                    handle,
                    &fixture.authority,
                    ptr::null(),
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );

            let wrong_authority = TemporalValidatedCheckpointAuthorityV1 {
                protocol_identity_sha256: [0x5b; 32],
                ..fixture.authority
            };
            let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                    handle,
                    &wrong_authority,
                    &request,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_DEFINITION
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );

            fail_next_destination_allocation();
            let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                    handle,
                    &fixture.authority,
                    &request,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_ALLOCATION_FAILED
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );

            let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                    handle,
                    &fixture.authority,
                    &request,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            let publication = publication.assume_init();
            assert_eq!(publication.outcome, TEMPORAL_CHECKPOINT_NOT_PUBLISHED);
            assert_eq!(
                publication.temp_cleanup,
                TEMPORAL_CHECKPOINT_CLEANUP_NOT_APPLICABLE
            );
            assert_eq!(publication.cleanup_error_count, 0);
            assert_eq!(
                publication.primary_error.domain,
                TEMPORAL_CHECKPOINT_ERROR_DOMAIN_POLICY
            );
            assert_eq!(
                publication.primary_error.stage,
                TEMPORAL_CHECKPOINT_ERROR_STAGE_PARENT_OPEN
            );
            assert_eq!(report.assume_init().status, TEMPORAL_RUNTIME_OK);
            assert_eq!(export_checkpoint(handle, &fixture.authority), before);

            let overlap = b"same-region";
            let overlap_request = TemporalCheckpointPathV1Request {
                parent_utf8: overlap.as_ptr(),
                parent_len: overlap.len() as u64,
                final_basename_utf8: overlap.as_ptr(),
                final_basename_len: overlap.len() as u64,
                reserved_zero: [0; 8],
            };
            let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                    handle,
                    &fixture.authority,
                    &overlap_request,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );

            let bad_geometry = TemporalCheckpointPathV1Request {
                parent_len: u64::MAX,
                ..request
            };
            let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                    handle,
                    &fixture.authority,
                    &bad_geometry,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );

            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
    }

    #[test]
    fn immutable_bytes_publication_rejects_invalid_transport_with_zero_outcome() {
        let parent = b"C:\\unused-parent";
        let basename = b"result.json";
        let payload = b"{}";
        let request = TemporalImmutableBytesPathV1Request {
            parent_utf8: parent.as_ptr(),
            parent_len: parent.len() as u64,
            final_basename_utf8: basename.as_ptr(),
            final_basename_len: basename.len() as u64,
            payload: payload.as_ptr(),
            payload_len: payload.len() as u64,
            reserved_zero: [1; 8],
        };
        let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
        let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();

        unsafe {
            assert_eq!(
                humanbrain_temporal_publish_immutable_bytes_path_v1(
                    &request,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );

            let zero_payload = TemporalImmutableBytesPathV1Request {
                payload: ptr::null(),
                payload_len: 0,
                reserved_zero: [0; 8],
                ..request
            };
            let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_publish_immutable_bytes_path_v1(
                    &zero_payload,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );

            let overlapping_payload = TemporalImmutableBytesPathV1Request {
                payload: parent.as_ptr(),
                payload_len: 1,
                reserved_zero: [0; 8],
                ..request
            };
            let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_publish_immutable_bytes_path_v1(
                    &overlapping_payload,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );

            let oversized_payload = TemporalImmutableBytesPathV1Request {
                payload_len: (MAX_IMMUTABLE_PAYLOAD_BYTES as u64) + 1,
                reserved_zero: [0; 8],
                ..request
            };
            let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_publish_immutable_bytes_path_v1(
                    &oversized_payload,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_INVALID_ARGUMENT
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );
        }
    }

    #[test]
    fn immutable_payload_copy_allocation_failure_precedes_filesystem_access() {
        let parent = b"C:\\path-does-not-need-to-exist";
        let basename = b"result.json";
        let payload = b"{}";
        let request = TemporalImmutableBytesPathV1Request {
            parent_utf8: parent.as_ptr(),
            parent_len: parent.len() as u64,
            final_basename_utf8: basename.as_ptr(),
            final_basename_len: basename.len() as u64,
            payload: payload.as_ptr(),
            payload_len: payload.len() as u64,
            reserved_zero: [0; 8],
        };
        let mut publication = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
        let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();

        fail_next_immutable_payload_allocation();
        unsafe {
            assert_eq!(
                humanbrain_temporal_publish_immutable_bytes_path_v1(
                    &request,
                    publication.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_ALLOCATION_FAILED
            );
            assert_eq!(
                publication.assume_init(),
                TemporalCheckpointPublicationReportV1::ZERO
            );
            assert_eq!(report.assume_init().required, 0);
        }
    }

    #[cfg(windows)]
    #[test]
    fn ntfs_immutable_bytes_publication_is_exclusive_and_bit_exact() {
        use std::{
            fs,
            time::{SystemTime, UNIX_EPOCH},
        };

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = std::env::temp_dir().join(format!(
            "humanbrain-bytes-publisher-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&parent).unwrap();
        let parent_utf8 = parent.to_str().unwrap().as_bytes();
        let basename = b"result.json";
        let final_path = parent.join("result.json");
        let payload = b"{\"completed\":true,\"opaque\":\"\\u0000\"}\0\xff";
        let request = TemporalImmutableBytesPathV1Request {
            parent_utf8: parent_utf8.as_ptr(),
            parent_len: parent_utf8.len() as u64,
            final_basename_utf8: basename.as_ptr(),
            final_basename_len: basename.len() as u64,
            payload: payload.as_ptr(),
            payload_len: payload.len() as u64,
            reserved_zero: [0; 8],
        };

        unsafe {
            let sentinel = b"preexisting-sentinel-must-not-be-overwritten";
            fs::write(&final_path, sentinel).unwrap();
            let mut collision = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_publish_immutable_bytes_path_v1(
                    &request,
                    collision.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            let collision = collision.assume_init();
            assert_eq!(collision.outcome, TEMPORAL_CHECKPOINT_COLLISION);
            assert_eq!(collision.checkpoint_bytes, payload.len() as u64);
            assert_eq!(collision.encoded_bytes, payload.len() as u64);
            assert_eq!(fs::read(&final_path).unwrap(), sentinel);
            fs::remove_file(&final_path).unwrap();

            let mut published = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_publish_immutable_bytes_path_v1(
                    &request,
                    published.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            let published = published.assume_init();
            assert_eq!(published.outcome, TEMPORAL_CHECKPOINT_DURABLE_PUBLISHED);
            assert_eq!(published.checkpoint_bytes, payload.len() as u64);
            assert_eq!(published.encoded_bytes, payload.len() as u64);
            assert_eq!(fs::read(&final_path).unwrap(), payload);
        }

        assert!(fs::read_dir(&parent).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".humanbrain-bytes-v1-")));
        fs::remove_file(final_path).unwrap();
        fs::remove_dir(parent).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn ntfs_path_publication_is_exclusive_and_import_bit_exact() {
        use std::{
            fs,
            time::{SystemTime, UNIX_EPOCH},
        };

        let fixture = EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
        let definition = fixture.definition();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = std::env::temp_dir().join(format!(
            "humanbrain-ckpt-publisher-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&parent).unwrap();
        let parent_utf8 = parent.to_str().unwrap().as_bytes();
        let basename = b"checkpoint-v1.bin";
        let final_path = parent.join("checkpoint-v1.bin");
        let request = TemporalCheckpointPathV1Request {
            parent_utf8: parent_utf8.as_ptr(),
            parent_len: parent_utf8.len() as u64,
            final_basename_utf8: basename.as_ptr(),
            final_basename_len: basename.len() as u64,
            reserved_zero: [0; 8],
        };

        unsafe {
            let (handle, create_report) = create_from_definition(&definition);
            assert_eq!(create_report.status, TEMPORAL_RUNTIME_OK);
            let expected = export_checkpoint(handle, &fixture.authority);

            let sentinel = b"preexisting-sentinel-must-not-be-overwritten";
            fs::write(&final_path, sentinel).unwrap();

            let mut collision = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                    handle,
                    &fixture.authority,
                    &request,
                    collision.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            let collision = collision.assume_init();
            assert_eq!(collision.outcome, TEMPORAL_CHECKPOINT_COLLISION);
            assert_eq!(
                collision.primary_error,
                TemporalCheckpointPublicationErrorV1::ZERO
            );
            assert_eq!(fs::read(&final_path).unwrap(), sentinel);
            assert!(fs::read_dir(&parent).unwrap().all(|entry| !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".humanbrain-ckpt-v1-")));
            fs::remove_file(&final_path).unwrap();

            let mut published = MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
            let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                    handle,
                    &fixture.authority,
                    &request,
                    published.as_mut_ptr(),
                    report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            let published = published.assume_init();
            assert_eq!(
                published.outcome, TEMPORAL_CHECKPOINT_DURABLE_PUBLISHED,
                "{published:?}"
            );
            assert_eq!(published.temp_cleanup, TEMPORAL_CHECKPOINT_CLEANUP_CLEAN);
            assert_eq!(published.checkpoint_bytes, expected.len() as u64);
            assert_eq!(published.encoded_bytes, expected.len() as u64);
            assert_eq!(fs::read(&final_path).unwrap(), expected);

            let checkpoint = fs::read(&final_path).unwrap();
            let import_request = TemporalCircuitImportCheckpointV1Request {
                definition,
                checkpoint_bytes: checkpoint.as_ptr(),
                checkpoint_len: checkpoint.len() as u64,
            };
            let mut imported = ptr::null_mut();
            let mut import_report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
            assert_eq!(
                humanbrain_temporal_circuit_runtime_import_checkpoint_v1(
                    &import_request,
                    &mut imported,
                    import_report.as_mut_ptr(),
                ),
                TEMPORAL_RUNTIME_OK
            );
            assert_eq!(export_checkpoint(imported, &fixture.authority), checkpoint);

            humanbrain_temporal_circuit_runtime_destroy(imported);
            humanbrain_temporal_circuit_runtime_destroy(handle);
        }
        fs::remove_file(&final_path).unwrap();
        fs::remove_dir(&parent).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn ntfs_many_writers_publish_exactly_one_complete_checkpoint() {
        use std::{
            fs,
            io::ErrorKind,
            sync::{
                atomic::{AtomicBool, Ordering},
                Arc, Barrier,
            },
            thread,
            time::{SystemTime, UNIX_EPOCH},
        };

        const WRITERS: usize = 8;
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = std::env::temp_dir().join(format!(
            "humanbrain-ckpt-publisher-race-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&parent).unwrap();
        let parent_utf8 = parent.to_str().unwrap().as_bytes().to_vec();
        let final_path = parent.join("checkpoint-v1.bin");

        let expected = unsafe {
            let fixture =
                EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
            let definition = fixture.definition();
            let (handle, report) = create_from_definition(&definition);
            assert_eq!(report.status, TEMPORAL_RUNTIME_OK);
            let expected = export_checkpoint(handle, &fixture.authority);
            humanbrain_temporal_circuit_runtime_destroy(handle);
            expected
        };
        let expected = Arc::new(expected);
        let stop_reader = Arc::new(AtomicBool::new(false));
        let reader_ready = Arc::new(AtomicBool::new(false));
        let reader = {
            let expected = Arc::clone(&expected);
            let final_path = final_path.clone();
            let stop_reader = Arc::clone(&stop_reader);
            let reader_ready = Arc::clone(&reader_ready);
            thread::spawn(move || {
                reader_ready.store(true, Ordering::Release);
                while !stop_reader.load(Ordering::Acquire) {
                    match fs::read(&final_path) {
                        Ok(observed) => assert_eq!(observed.as_slice(), expected.as_slice()),
                        Err(error) if error.kind() == ErrorKind::NotFound => {}
                        Err(error) => panic!("fresh reader failed: {error}"),
                    }
                    thread::yield_now();
                }
            })
        };
        while !reader_ready.load(Ordering::Acquire) {
            thread::yield_now();
        }

        let barrier = Arc::new(Barrier::new(WRITERS + 1));
        let writers = (0..WRITERS)
            .map(|_| {
                let parent_utf8 = parent_utf8.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || unsafe {
                    let fixture =
                        EmptyDefinitionFixture::new(CONTEXT_GENERIC_AFFERENTS, PLASTICITY_PLASTIC);
                    let definition = fixture.definition();
                    let (handle, create_report) = create_from_definition(&definition);
                    assert_eq!(create_report.status, TEMPORAL_RUNTIME_OK);
                    let basename = b"checkpoint-v1.bin";
                    let request = TemporalCheckpointPathV1Request {
                        parent_utf8: parent_utf8.as_ptr(),
                        parent_len: parent_utf8.len() as u64,
                        final_basename_utf8: basename.as_ptr(),
                        final_basename_len: basename.len() as u64,
                        reserved_zero: [0; 8],
                    };
                    let mut publication =
                        MaybeUninit::<TemporalCheckpointPublicationReportV1>::uninit();
                    let mut report = MaybeUninit::<TemporalCircuitReportV1>::uninit();
                    barrier.wait();
                    let status = humanbrain_temporal_circuit_runtime_export_checkpoint_path_v1(
                        handle,
                        &fixture.authority,
                        &request,
                        publication.as_mut_ptr(),
                        report.as_mut_ptr(),
                    );
                    let publication = publication.assume_init();
                    let report = report.assume_init();
                    humanbrain_temporal_circuit_runtime_destroy(handle);
                    assert_eq!(status, TEMPORAL_RUNTIME_OK);
                    assert_eq!(report.status, TEMPORAL_RUNTIME_OK);
                    publication
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let publications = writers
            .into_iter()
            .map(|writer| writer.join().unwrap())
            .collect::<Vec<_>>();
        stop_reader.store(true, Ordering::Release);
        reader.join().unwrap();

        assert_eq!(
            publications
                .iter()
                .filter(|report| report.outcome == TEMPORAL_CHECKPOINT_DURABLE_PUBLISHED)
                .count(),
            1
        );
        assert_eq!(
            publications
                .iter()
                .filter(|report| report.outcome == TEMPORAL_CHECKPOINT_COLLISION)
                .count(),
            WRITERS - 1
        );
        assert!(publications.iter().all(|report| {
            report.temp_cleanup == TEMPORAL_CHECKPOINT_CLEANUP_CLEAN
                && report.cleanup_error_count == 0
        }));
        assert_eq!(
            fs::read(&final_path).unwrap().as_slice(),
            expected.as_slice()
        );
        assert!(fs::read_dir(&parent).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".humanbrain-ckpt-v1-")));

        fs::remove_file(&final_path).unwrap();
        fs::remove_dir(&parent).unwrap();
    }
}
