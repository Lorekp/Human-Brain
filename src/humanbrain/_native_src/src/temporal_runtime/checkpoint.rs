//! Checkpoint-v1 identities and pure fixed-header codec for the native runtime.
//!
//! This module computes identities, checked byte layout, canonical scalar
//! bytes, the complete fixed header, allocation-free reachability proofs,
//! fallible export preparation over a borrowed live state, and allocation-free
//! streaming of a prepared checkpoint into a caller-owned sink. It also owns
//! allocation-free checkpoint import Pass 1, typed-scratch canonical-state
//! import Pass 2, and the direct one-shot runtime assembler; it does not publish
//! files or expose an ABI.

use std::{
    alloc::Layout,
    error::Error,
    fmt,
    io::{self, Seek, SeekFrom, Write},
    mem::size_of,
};

use sha2::{Digest, Sha256};

use super::{
    role_place_copy::RolePlaceCopyState, NovelCandidate, TemporalCircuitConfig,
    TemporalCircuitRuntime, TemporalContextPolicy, TemporalPlasticityPolicy, TemporalPoisonCode,
    TokenCode, ACTIVE_COLUMNS, CONTEXT_CELLS, CONTEXT_CELLS_U8, EMPTY_DIAGNOSTIC, EMPTY_DRIVE,
    EMPTY_NOVEL_CANDIDATE, NONE, PAIRS_PER_SEGMENT,
};
use crate::temporal::{
    temporal_source_meta_allocation_layout, TemporalCheckpointRebuildError,
    TemporalOwnerDiagnostic, TemporalSegmentRow, TemporalStoreAllocationSite,
    TemporalStoreCheckpointRebuild, TemporalStoreConfig, TemporalStorePreflightError,
};
use crate::temporal_physics::{
    CandidateDrive, CoupledRaceScratch, CoupledRaceScratchBuildError,
    CoupledRaceScratchReserveSite, TEMPORAL_PHYSICS_IDENTITY,
};

const LOGICAL_STATE_DOMAIN: &[u8] = b"humanbrain-r3-checkpoint-logical-state-v1\0";
const CIRCUIT_IDENTITY_DOMAIN: &[u8] = b"humanbrain-r3-temporal-circuit-v1";
const PHYSICAL_ROUTE_CERTIFICATE_DOMAIN: &[u8] = b"humanbrain-r3-physical-route-certificate-v1";
const PHYSICAL_CONTACTS_PER_PAIR: u8 = 2;
const PERMANENCE_MAX_TICKS: u16 = 500;
const CHECKPOINT_V1_STREAM_BUFFER_BYTES: usize = 8 * 1024;
pub(crate) const CHECKPOINT_V1_MAGIC: [u8; 8] = *b"R3NTCP1\0";
pub(crate) const CHECKPOINT_V1_FORMAT_VERSION: u16 = 1;
pub(crate) const CHECKPOINT_V1_HEADER_BYTES: u16 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CheckpointV1Digest([u8; 32]);

impl CheckpointV1Digest {
    pub(crate) const fn from_raw_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub(crate) const fn raw_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ValidatedCheckpointAuthority {
    protocol_identity: CheckpointV1Digest,
    source_bundle_identity: CheckpointV1Digest,
}

impl ValidatedCheckpointAuthority {
    /// Construct authority from exact SHA-256 output bytes.
    ///
    /// This API intentionally has no string or hexadecimal form. The caller
    /// must validate the protocol and source bundle before constructing it.
    pub(crate) const fn from_raw_sha256(
        protocol_identity: [u8; 32],
        source_bundle_identity: [u8; 32],
    ) -> Self {
        Self {
            protocol_identity: CheckpointV1Digest::from_raw_bytes(protocol_identity),
            source_bundle_identity: CheckpointV1Digest::from_raw_bytes(source_bundle_identity),
        }
    }

    pub(crate) const fn protocol_identity(&self) -> &CheckpointV1Digest {
        &self.protocol_identity
    }

    pub(crate) const fn source_bundle_identity(&self) -> &CheckpointV1Digest {
        &self.source_bundle_identity
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalRecognitionRoute {
    /// Exact recognition through a full scan or lossless intrusive postings.
    ExactGeneric = 0,
    /// Certified physical-source buckets; not executed by the current runtime.
    CertifiedPhysicalBuckets = 1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalCheckpointIdentityError {
    RuntimeInvariant(&'static str),
    RouteRequiresGenericAfferents,
    UnsafePhysicalRouteOverlap {
        logical_match_pairs: u8,
        max_distinct_code_overlap: u8,
    },
    UnsupportedExecutedRecognitionRoute,
    FramingLengthOverflow,
    FramingPayloadLength,
}

impl fmt::Display for TemporalCheckpointIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RuntimeInvariant(message) => write!(formatter, "runtime invariant: {message}"),
            Self::RouteRequiresGenericAfferents => {
                formatter.write_str("physical route requires generic afferents")
            }
            Self::UnsafePhysicalRouteOverlap {
                logical_match_pairs,
                max_distinct_code_overlap,
            } => write!(
                formatter,
                "physical route overlap {max_distinct_code_overlap} is not below logical match threshold {logical_match_pairs}"
            ),
            Self::UnsupportedExecutedRecognitionRoute => {
                formatter.write_str("the requested recognition route is not executable")
            }
            Self::FramingLengthOverflow => {
                formatter.write_str("circuit identity framing length overflowed")
            }
            Self::FramingPayloadLength => {
                formatter.write_str("circuit identity payload length did not match its frame")
            }
        }
    }
}

impl Error for TemporalCheckpointIdentityError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1CodecError {
    ExplicitUnpoisonedCode,
    LayoutScalarMismatch,
    LengthOverflow,
}

impl fmt::Display for CheckpointV1CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ExplicitUnpoisonedCode => {
                "an explicit unpoisoned enum value cannot be serialized"
            }
            Self::LayoutScalarMismatch => "checkpoint layout and scalar counts disagree",
            Self::LengthOverflow => "checkpoint length arithmetic overflowed",
        })
    }
}

impl Error for CheckpointV1CodecError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1PermanenceField {
    Connected,
    Initial,
    Increment,
    NegativeDecrement,
}

impl fmt::Display for CheckpointV1PermanenceField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Connected => "connected",
            Self::Initial => "initial",
            Self::Increment => "increment",
            Self::NegativeDecrement => "negative decrement",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1CircuitDefinitionError {
    FixedDimensions,
    ColumnCount {
        found: u32,
    },
    BaseCellsPerColumnZero,
    NovelRaceCap {
        found: u8,
    },
    BaseCellsExceedNovelRaceCap {
        base: u8,
        race_cap: u8,
    },
    SegmentCapacity {
        found: u32,
    },
    PredictionQuorum {
        found: u8,
    },
    MaxSegmentsPerCellZero,
    PermanenceTicks {
        field: CheckpointV1PermanenceField,
        found: u16,
    },
    ThresholdOrder {
        theta_match: u8,
        theta_active: u8,
    },
    GenericAfferentsBaseWidth {
        found: u8,
    },
    BasePopulationExceedsMaximum {
        base: u64,
        maximum: u32,
    },
    PhysicalOffsetsOverflow,
    ThresholdOffsetCount {
        expected: u32,
        actual: usize,
    },
    NonFiniteThresholdOffset {
        index: usize,
    },
    EmptyTokenCodebook,
    TokenCodebookLength {
        actual: usize,
    },
    SentinelTokenId {
        index: usize,
    },
    InvalidTokenColumns {
        index: usize,
    },
    TokenIdsNotStrict {
        left_index: usize,
    },
    DuplicateTokenColumns {
        left_index: usize,
        right_index: usize,
    },
    PhysicsInvariant,
}

impl CheckpointV1CircuitDefinitionError {
    fn runtime_invariant_message(self) -> &'static str {
        match self {
            Self::FixedDimensions => "checkpoint-v1 fixed dimensions changed",
            Self::BasePopulationExceedsMaximum { .. } => {
                "base population exceeds max_pyramidal_cells"
            }
            Self::PhysicalOffsetsOverflow => "physical offsets overflow uint32",
            Self::ThresholdOffsetCount { .. } | Self::NonFiniteThresholdOffset { .. } => {
                "threshold offsets are incomplete or non-finite"
            }
            Self::EmptyTokenCodebook
            | Self::TokenCodebookLength { .. }
            | Self::SentinelTokenId { .. }
            | Self::InvalidTokenColumns { .. }
            | Self::TokenIdsNotStrict { .. }
            | Self::DuplicateTokenColumns { .. } => "token codebook is not canonical",
            Self::PhysicsInvariant => "temporal physics identity is non-finite or impossible",
            Self::ColumnCount { .. }
            | Self::BaseCellsPerColumnZero
            | Self::NovelRaceCap { .. }
            | Self::BaseCellsExceedNovelRaceCap { .. }
            | Self::SegmentCapacity { .. }
            | Self::PredictionQuorum { .. }
            | Self::MaxSegmentsPerCellZero
            | Self::PermanenceTicks { .. }
            | Self::ThresholdOrder { .. }
            | Self::GenericAfferentsBaseWidth { .. } => {
                "runtime configuration is outside checkpoint-v1 domains"
            }
        }
    }
}

impl fmt::Display for CheckpointV1CircuitDefinitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FixedDimensions => formatter.write_str("checkpoint-v1 fixed dimensions changed"),
            Self::ColumnCount { found } => write!(
                formatter,
                "column_count {found} is below the required {ACTIVE_COLUMNS}"
            ),
            Self::BaseCellsPerColumnZero => {
                formatter.write_str("base_cells_per_column must be nonzero")
            }
            Self::NovelRaceCap { found } => {
                write!(formatter, "novel_race_cap {found} is outside 1..=8")
            }
            Self::BaseCellsExceedNovelRaceCap { base, race_cap } => write!(
                formatter,
                "base_cells_per_column {base} exceeds novel_race_cap {race_cap}"
            ),
            Self::SegmentCapacity { found } => write!(
                formatter,
                "segment_capacity {found} is outside the checkpoint-v1 domain"
            ),
            Self::PredictionQuorum { found } => write!(
                formatter,
                "prediction_quorum {found} is outside 1..={ACTIVE_COLUMNS}"
            ),
            Self::MaxSegmentsPerCellZero => {
                formatter.write_str("max_segments_per_cell must be nonzero")
            }
            Self::PermanenceTicks { field, found } => write!(
                formatter,
                "{field} permanence ticks {found} is outside 1..={PERMANENCE_MAX_TICKS}"
            ),
            Self::ThresholdOrder {
                theta_match,
                theta_active,
            } => write!(
                formatter,
                "thresholds must satisfy 1 <= theta_match ({theta_match}) <= theta_active ({theta_active}) <= {}",
                PHYSICAL_CONTACTS_PER_PAIR * PAIRS_PER_SEGMENT as u8
            ),
            Self::GenericAfferentsBaseWidth { found } => write!(
                formatter,
                "generic afferents require at least two base cells, found {found}"
            ),
            Self::BasePopulationExceedsMaximum { base, maximum } => write!(
                formatter,
                "base population {base} exceeds max_pyramidal_cells {maximum}"
            ),
            Self::PhysicalOffsetsOverflow => {
                formatter.write_str("physical cell offsets overflow uint32")
            }
            Self::ThresholdOffsetCount { expected, actual } => write!(
                formatter,
                "threshold offset count {actual} does not equal physical cell count {expected}"
            ),
            Self::NonFiniteThresholdOffset { index } => {
                write!(formatter, "threshold offset {index} is non-finite")
            }
            Self::EmptyTokenCodebook => formatter.write_str("token codebook is empty"),
            Self::TokenCodebookLength { actual } => write!(
                formatter,
                "token codebook length {actual} exceeds the uint32 domain"
            ),
            Self::SentinelTokenId { index } => {
                write!(formatter, "token codebook record {index} uses the sentinel token ID")
            }
            Self::InvalidTokenColumns { index } => write!(
                formatter,
                "token codebook record {index} has noncanonical columns"
            ),
            Self::TokenIdsNotStrict { left_index } => write!(
                formatter,
                "token IDs are not strictly increasing at records {left_index} and {}",
                left_index + 1
            ),
            Self::DuplicateTokenColumns {
                left_index,
                right_index,
            } => write!(
                formatter,
                "token records {left_index} and {right_index} share one column code"
            ),
            Self::PhysicsInvariant => {
                formatter.write_str("temporal physics identity is non-finite or impossible")
            }
        }
    }
}

impl Error for CheckpointV1CircuitDefinitionError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1FactoryError {
    Definition(CheckpointV1CircuitDefinitionError),
    Identity(TemporalCheckpointIdentityError),
}

impl fmt::Display for CheckpointV1FactoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Definition(source) => write!(formatter, "invalid circuit definition: {source}"),
            Self::Identity(source) => write!(formatter, "circuit identity failed: {source}"),
        }
    }
}

impl Error for CheckpointV1FactoryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Definition(source) => Some(source),
            Self::Identity(source) => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CheckpointV1Layout {
    pair_count: u64,
    cohort_columns_bytes: u64,
    prior_context_bytes: u64,
    winner_context_bytes: u64,
    segment_owner_bytes: u64,
    segment_source_bytes: u64,
    segment_tick_bytes: u64,
    body_bytes: u64,
    total_bytes: u64,
}

impl CheckpointV1Layout {
    pub(crate) fn checked(
        segment_count: u32,
        cohort_count: u32,
        prior_context_length: u8,
        winner_context_length: u8,
    ) -> Result<Self, CheckpointV1CodecError> {
        let segment_count = u64::from(segment_count);
        let cohort_count = u64::from(cohort_count);
        let prior_context_length = u64::from(prior_context_length);
        let winner_context_length = u64::from(winner_context_length);

        let pair_count = segment_count
            .checked_mul(PAIRS_PER_SEGMENT as u64)
            .ok_or(CheckpointV1CodecError::LengthOverflow)?;
        let cohort_columns_bytes = cohort_count
            .checked_mul(4)
            .ok_or(CheckpointV1CodecError::LengthOverflow)?;
        let prior_context_bytes = prior_context_length
            .checked_mul(4)
            .ok_or(CheckpointV1CodecError::LengthOverflow)?;
        let winner_context_bytes = winner_context_length
            .checked_mul(4)
            .ok_or(CheckpointV1CodecError::LengthOverflow)?;
        let segment_owner_bytes = segment_count
            .checked_mul(4)
            .ok_or(CheckpointV1CodecError::LengthOverflow)?;
        let segment_source_bytes = pair_count
            .checked_mul(4)
            .ok_or(CheckpointV1CodecError::LengthOverflow)?;
        let segment_tick_bytes = pair_count
            .checked_mul(2)
            .ok_or(CheckpointV1CodecError::LengthOverflow)?;
        let body_bytes = [
            cohort_columns_bytes,
            prior_context_bytes,
            winner_context_bytes,
            segment_owner_bytes,
            segment_source_bytes,
            segment_tick_bytes,
        ]
        .into_iter()
        .try_fold(0u64, |total, section| total.checked_add(section))
        .ok_or(CheckpointV1CodecError::LengthOverflow)?;
        let total_bytes = body_bytes
            .checked_add(u64::from(CHECKPOINT_V1_HEADER_BYTES))
            .ok_or(CheckpointV1CodecError::LengthOverflow)?;

        Ok(Self {
            pair_count,
            cohort_columns_bytes,
            prior_context_bytes,
            winner_context_bytes,
            segment_owner_bytes,
            segment_source_bytes,
            segment_tick_bytes,
            body_bytes,
            total_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CheckpointV1Scalars {
    segment_count: u32,
    cohort_count: u32,
    next_reserve_cell: u32,
    event_count: u64,
    learning_step: u64,
    prior_context_length: u8,
    winner_context_length: u8,
    poison_code: u16,
}

impl CheckpointV1Scalars {
    #[allow(clippy::too_many_arguments)] // The arguments are the exact frozen scalar fields.
    pub(crate) fn new(
        segment_count: u32,
        cohort_count: u32,
        next_reserve_cell: u32,
        event_count: u64,
        learning_step: u64,
        prior_context_length: u8,
        winner_context_length: u8,
        poison: Option<TemporalPoisonCode>,
    ) -> Result<Self, CheckpointV1CodecError> {
        Ok(Self {
            segment_count,
            cohort_count,
            next_reserve_cell,
            event_count,
            learning_step,
            prior_context_length,
            winner_context_length,
            poison_code: checkpoint_v1_poison_code(poison)?,
        })
    }

    pub(crate) fn to_le_bytes(self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&self.segment_count.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.cohort_count.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.next_reserve_cell.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.event_count.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.learning_step.to_le_bytes());
        bytes[28] = self.prior_context_length;
        bytes[29] = self.winner_context_length;
        bytes[30..32].copy_from_slice(&self.poison_code.to_le_bytes());
        bytes
    }
}

fn checkpoint_v1_poison_code(
    poison: Option<TemporalPoisonCode>,
) -> Result<u16, CheckpointV1CodecError> {
    Ok(match poison {
        None => 0,
        Some(TemporalPoisonCode::Unpoisoned) => {
            return Err(CheckpointV1CodecError::ExplicitUnpoisonedCode);
        }
        Some(TemporalPoisonCode::MultipleActiveOwners) => 1,
        Some(TemporalPoisonCode::MultipleMatchingOwners) => 2,
        Some(TemporalPoisonCode::CategorizedRaceCapacity) => 3,
        Some(TemporalPoisonCode::MalformedContextOrPolicy) => 4,
        Some(TemporalPoisonCode::ProximalPhysics) => 5,
        Some(TemporalPoisonCode::ProximalRelayPhysics) => 6,
        Some(TemporalPoisonCode::ColumnRacePhysics) => 7,
        Some(TemporalPoisonCode::WinnerRelayPhysics) => 8,
        Some(TemporalPoisonCode::TickOrCounterOverflow) => 9,
        Some(TemporalPoisonCode::StoreOrPlanInvariant) => 10,
        Some(TemporalPoisonCode::PhiloxAddressExhaustion) => 11,
        Some(TemporalPoisonCode::ReserveCellCapacity) => 12,
        Some(TemporalPoisonCode::GlobalSegmentCapacity) => 13,
        Some(TemporalPoisonCode::OwnerSegmentCapacity) => 14,
        Some(TemporalPoisonCode::PreallocatedCapacityInvariant) => 15,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TemporalCheckpointV1Identity {
    authority: ValidatedCheckpointAuthority,
    circuit_identity: CheckpointV1Digest,
    recognition_route: TemporalRecognitionRoute,
    recognition_certificate: Option<CheckpointV1Digest>,
    circuit_preimage_bytes: u64,
}

impl TemporalCheckpointV1Identity {
    pub(crate) const fn authority(&self) -> &ValidatedCheckpointAuthority {
        &self.authority
    }

    pub(crate) const fn circuit_identity(&self) -> &CheckpointV1Digest {
        &self.circuit_identity
    }

    pub(crate) const fn recognition_route(&self) -> TemporalRecognitionRoute {
        self.recognition_route
    }

    pub(crate) const fn recognition_certificate(&self) -> Option<&CheckpointV1Digest> {
        self.recognition_certificate.as_ref()
    }

    pub(crate) const fn circuit_preimage_bytes(&self) -> u64 {
        self.circuit_preimage_bytes
    }
}

pub(crate) fn checkpoint_v1_header_bytes(
    identity: &TemporalCheckpointV1Identity,
    layout: CheckpointV1Layout,
    scalars: CheckpointV1Scalars,
    body_digest: CheckpointV1Digest,
    semantic_index_digest: CheckpointV1Digest,
) -> Result<[u8; CHECKPOINT_V1_HEADER_BYTES as usize], CheckpointV1CodecError> {
    let scalar_layout = CheckpointV1Layout::checked(
        scalars.segment_count,
        scalars.cohort_count,
        scalars.prior_context_length,
        scalars.winner_context_length,
    )?;
    if layout != scalar_layout {
        return Err(CheckpointV1CodecError::LayoutScalarMismatch);
    }

    let scalar_bytes = scalars.to_le_bytes();
    let logical_state_digest =
        checkpoint_v1_logical_state_digest(body_digest.raw_bytes(), &scalar_bytes);
    let mut header = [0u8; CHECKPOINT_V1_HEADER_BYTES as usize];
    header[0..8].copy_from_slice(&CHECKPOINT_V1_MAGIC);
    header[8..10].copy_from_slice(&CHECKPOINT_V1_FORMAT_VERSION.to_le_bytes());
    header[10..12].copy_from_slice(&CHECKPOINT_V1_HEADER_BYTES.to_le_bytes());
    header[16..24].copy_from_slice(&layout.total_bytes.to_le_bytes());
    header[24..32].copy_from_slice(&layout.body_bytes.to_le_bytes());
    header[32..64].copy_from_slice(identity.authority().protocol_identity().raw_bytes());
    header[64..96].copy_from_slice(identity.authority().source_bundle_identity().raw_bytes());
    header[96..128].copy_from_slice(identity.circuit_identity().raw_bytes());
    header[128..160].copy_from_slice(body_digest.raw_bytes());
    header[160..192].copy_from_slice(semantic_index_digest.raw_bytes());
    header[192..224].copy_from_slice(&scalar_bytes);
    header[224..256].copy_from_slice(&logical_state_digest);
    Ok(header)
}

struct FramedSha256 {
    hasher: Sha256,
    bytes: u64,
}

impl FramedSha256 {
    fn new(namespace: &[u8]) -> Result<Self, TemporalCheckpointIdentityError> {
        let namespace_len = u64::try_from(namespace.len())
            .map_err(|_| TemporalCheckpointIdentityError::FramingLengthOverflow)?;
        let mut framed = Self {
            hasher: Sha256::new(),
            bytes: 0,
        };
        framed.raw(&namespace_len.to_be_bytes())?;
        framed.raw(namespace)?;
        Ok(framed)
    }

    fn raw(&mut self, bytes: &[u8]) -> Result<(), TemporalCheckpointIdentityError> {
        let byte_count = u64::try_from(bytes.len())
            .map_err(|_| TemporalCheckpointIdentityError::FramingLengthOverflow)?;
        self.bytes = self
            .bytes
            .checked_add(byte_count)
            .ok_or(TemporalCheckpointIdentityError::FramingLengthOverflow)?;
        self.hasher.update(bytes);
        Ok(())
    }

    fn entry(
        &mut self,
        name: &'static [u8],
        payload_len: u64,
        emit_payload: impl FnOnce(&mut Self) -> Result<(), TemporalCheckpointIdentityError>,
    ) -> Result<(), TemporalCheckpointIdentityError> {
        let name_len = u64::try_from(name.len())
            .map_err(|_| TemporalCheckpointIdentityError::FramingLengthOverflow)?;
        self.raw(&name_len.to_be_bytes())?;
        self.raw(name)?;
        self.raw(&payload_len.to_be_bytes())?;
        let payload_start = self.bytes;
        emit_payload(self)?;
        if self.bytes.checked_sub(payload_start) != Some(payload_len) {
            return Err(TemporalCheckpointIdentityError::FramingPayloadLength);
        }
        Ok(())
    }

    fn entry_bytes(
        &mut self,
        name: &'static [u8],
        payload: &[u8],
    ) -> Result<(), TemporalCheckpointIdentityError> {
        let payload_len = u64::try_from(payload.len())
            .map_err(|_| TemporalCheckpointIdentityError::FramingLengthOverflow)?;
        self.entry(name, payload_len, |framed| framed.raw(payload))
    }

    fn finish(self) -> (CheckpointV1Digest, u64) {
        (
            CheckpointV1Digest(self.hasher.finalize().into()),
            self.bytes,
        )
    }
}

#[derive(Clone, Copy)]
struct DerivedCircuitIdentity {
    base_population: u32,
    relay_offset: u32,
    basket_offset: u32,
    proximal_offset: u32,
    proximal_relay_offset: u32,
    physical_cell_count: u32,
    source_gid_base: u32,
    source_count: u32,
}

#[derive(Clone, Copy)]
struct CheckpointV1CircuitDefinition<'a> {
    config: TemporalCircuitConfig,
    token_codebook: &'a [TokenCode],
    threshold_offsets_mv: &'a [f64],
}

impl<'a> CheckpointV1CircuitDefinition<'a> {
    fn from_runtime(runtime: &'a TemporalCircuitRuntime) -> Self {
        Self {
            config: runtime.config,
            token_codebook: &runtime.token_codebook,
            threshold_offsets_mv: &runtime.threshold_offsets_mv,
        }
    }
}

pub(crate) struct ValidatedCheckpointV1RuntimeFactory {
    config: TemporalCircuitConfig,
    token_codebook: Vec<TokenCode>,
    threshold_offsets_mv: Vec<f64>,
    identity: TemporalCheckpointV1Identity,
    derived: DerivedCircuitIdentity,
}

impl ValidatedCheckpointV1RuntimeFactory {
    pub(crate) fn circuit_identity_for_definition(
        config: TemporalCircuitConfig,
        token_codebook: &[TokenCode],
        threshold_offsets_mv: &[f64],
    ) -> Result<[u8; 32], CheckpointV1FactoryError> {
        let definition = CheckpointV1CircuitDefinition {
            config,
            token_codebook,
            threshold_offsets_mv,
        };
        let derived = validate_checkpoint_v1_circuit_definition(definition)
            .map_err(CheckpointV1FactoryError::Definition)?;
        let (identity, _) = circuit_identity_route_zero(definition, derived)
            .map_err(CheckpointV1FactoryError::Identity)?;
        Ok(*identity.raw_bytes())
    }

    pub(crate) fn new(
        authority: ValidatedCheckpointAuthority,
        config: TemporalCircuitConfig,
        token_codebook: Vec<TokenCode>,
        threshold_offsets_mv: Vec<f64>,
    ) -> Result<Self, CheckpointV1FactoryError> {
        let definition = CheckpointV1CircuitDefinition {
            config,
            token_codebook: &token_codebook,
            threshold_offsets_mv: &threshold_offsets_mv,
        };
        let derived = validate_checkpoint_v1_circuit_definition(definition)
            .map_err(CheckpointV1FactoryError::Definition)?;
        let identity = checkpoint_v1_identity_from_definition(
            authority,
            definition,
            derived,
            TemporalRecognitionRoute::ExactGeneric,
        )
        .map_err(CheckpointV1FactoryError::Identity)?;

        Ok(Self {
            config,
            token_codebook,
            threshold_offsets_mv,
            identity,
            derived,
        })
    }

    pub(crate) const fn identity(&self) -> &TemporalCheckpointV1Identity {
        &self.identity
    }

    fn definition(&self) -> CheckpointV1CircuitDefinition<'_> {
        CheckpointV1CircuitDefinition {
            config: self.config,
            token_codebook: &self.token_codebook,
            threshold_offsets_mv: &self.threshold_offsets_mv,
        }
    }
}

impl TemporalCircuitRuntime {
    pub(crate) fn checkpoint_v1_identity(
        &self,
        authority: ValidatedCheckpointAuthority,
        recognition_route: TemporalRecognitionRoute,
    ) -> Result<TemporalCheckpointV1Identity, TemporalCheckpointIdentityError> {
        let definition = CheckpointV1CircuitDefinition::from_runtime(self);
        let derived = validate_runtime_identity_inputs(self, definition)?;
        checkpoint_v1_identity_from_definition(authority, definition, derived, recognition_route)
    }
}

fn checkpoint_v1_identity_from_definition(
    authority: ValidatedCheckpointAuthority,
    definition: CheckpointV1CircuitDefinition<'_>,
    derived: DerivedCircuitIdentity,
    recognition_route: TemporalRecognitionRoute,
) -> Result<TemporalCheckpointV1Identity, TemporalCheckpointIdentityError> {
    if recognition_route == TemporalRecognitionRoute::CertifiedPhysicalBuckets {
        return Err(TemporalCheckpointIdentityError::UnsupportedExecutedRecognitionRoute);
    }

    let (circuit_identity, circuit_preimage_bytes) =
        circuit_identity_route_zero(definition, derived)?;
    Ok(TemporalCheckpointV1Identity {
        authority,
        circuit_identity,
        recognition_route,
        recognition_certificate: None,
        circuit_preimage_bytes,
    })
}

/// Recompute only the route-1 codebook eligibility certificate.
///
/// A successful result does not claim that `runtime` executes physical-source
/// buckets. The current runtime executes exact intrusive postings and therefore
/// still identifies only as route 0.
pub(crate) fn recompute_checkpoint_v1_physical_route_certificate(
    runtime: &TemporalCircuitRuntime,
) -> Result<CheckpointV1Digest, TemporalCheckpointIdentityError> {
    let definition = CheckpointV1CircuitDefinition::from_runtime(runtime);
    validate_runtime_identity_inputs(runtime, definition)?;
    if definition.config.context_policy != TemporalContextPolicy::GenericAfferents {
        return Err(TemporalCheckpointIdentityError::RouteRequiresGenericAfferents);
    }
    let logical_match_pairs = definition
        .config
        .theta_match
        .checked_add(PHYSICAL_CONTACTS_PER_PAIR - 1)
        .ok_or(TemporalCheckpointIdentityError::RuntimeInvariant(
            "theta_match cannot be rounded to logical pairs",
        ))?
        / PHYSICAL_CONTACTS_PER_PAIR;
    if logical_match_pairs == 0 || logical_match_pairs > PAIRS_PER_SEGMENT as u8 {
        return Err(TemporalCheckpointIdentityError::RuntimeInvariant(
            "theta_match is outside the physical contact domain",
        ));
    }
    let max_distinct_code_overlap = max_distinct_code_overlap(definition.token_codebook);
    if max_distinct_code_overlap >= logical_match_pairs {
        return Err(
            TemporalCheckpointIdentityError::UnsafePhysicalRouteOverlap {
                logical_match_pairs,
                max_distinct_code_overlap,
            },
        );
    }

    let mut framed = FramedSha256::new(PHYSICAL_ROUTE_CERTIFICATE_DOMAIN)?;
    framed.entry_bytes(b"logical_match_pairs", &[logical_match_pairs])?;
    framed.entry_bytes(b"max_distinct_code_overlap", &[max_distinct_code_overlap])?;
    emit_token_codebook_entry(&mut framed, definition.token_codebook)?;
    Ok(framed.finish().0)
}

fn circuit_identity_route_zero(
    definition: CheckpointV1CircuitDefinition<'_>,
    derived: DerivedCircuitIdentity,
) -> Result<(CheckpointV1Digest, u64), TemporalCheckpointIdentityError> {
    let config = definition.config;
    let context_policy = match config.context_policy {
        TemporalContextPolicy::BoundWinners => 1,
        TemporalContextPolicy::GenericAfferents => 2,
    };
    let plasticity_policy = match config.plasticity_policy {
        TemporalPlasticityPolicy::GrowthOff => 0,
        TemporalPlasticityPolicy::Plastic => 1,
    };
    let mut framed = FramedSha256::new(CIRCUIT_IDENTITY_DOMAIN)?;
    framed.entry_bytes(b"active_columns", &[ACTIVE_COLUMNS as u8])?;
    framed.entry_bytes(b"context_cells", &[CONTEXT_CELLS as u8])?;
    framed.entry_bytes(b"pairs_per_segment", &[PAIRS_PER_SEGMENT as u8])?;
    framed.entry_bytes(b"physical_contacts_per_pair", &[PHYSICAL_CONTACTS_PER_PAIR])?;
    framed.entry_bytes(b"context_policy", &[context_policy])?;
    framed.entry_bytes(b"plasticity_policy", &[plasticity_policy])?;
    framed.entry_bytes(
        b"recognition_route",
        &[TemporalRecognitionRoute::ExactGeneric as u8],
    )?;
    framed.entry_bytes(b"column_count", &config.column_count.to_le_bytes())?;
    framed.entry_bytes(b"base_cells_per_column", &[config.base_cells_per_column])?;
    framed.entry_bytes(
        b"max_pyramidal_cells",
        &config.max_pyramidal_cells.to_le_bytes(),
    )?;
    framed.entry_bytes(b"segment_capacity", &config.segment_capacity.to_le_bytes())?;
    framed.entry_bytes(b"novel_race_cap", &[config.novel_race_cap])?;
    framed.entry_bytes(b"max_segments_per_cell", &[config.max_segments_per_cell])?;
    framed.entry_bytes(b"prediction_quorum", &[config.prediction_quorum])?;
    framed.entry_bytes(
        b"construction_seed",
        &config.construction_seed.to_le_bytes(),
    )?;
    framed.entry_bytes(
        b"connected_permanence_ticks",
        &config.connected_permanence_ticks.to_le_bytes(),
    )?;
    framed.entry_bytes(b"theta_match", &[config.theta_match])?;
    framed.entry_bytes(b"theta_active", &[config.theta_active])?;
    framed.entry_bytes(b"permanence_max_ticks", &PERMANENCE_MAX_TICKS.to_le_bytes())?;
    framed.entry_bytes(
        b"initial_permanence_ticks",
        &config.initial_permanence_ticks.to_le_bytes(),
    )?;
    framed.entry_bytes(
        b"permanence_increment_ticks",
        &config.permanence_increment_ticks.to_le_bytes(),
    )?;
    framed.entry_bytes(
        b"negative_decrement_ticks",
        &config.negative_decrement_ticks.to_le_bytes(),
    )?;
    framed.entry_bytes(b"relay_offset", &derived.relay_offset.to_le_bytes())?;
    framed.entry_bytes(b"basket_offset", &derived.basket_offset.to_le_bytes())?;
    framed.entry_bytes(b"proximal_offset", &derived.proximal_offset.to_le_bytes())?;
    framed.entry_bytes(
        b"proximal_relay_offset",
        &derived.proximal_relay_offset.to_le_bytes(),
    )?;
    framed.entry_bytes(
        b"physical_cell_count",
        &derived.physical_cell_count.to_le_bytes(),
    )?;
    framed.entry_bytes(b"source_gid_base", &derived.source_gid_base.to_le_bytes())?;
    framed.entry_bytes(b"source_count", &derived.source_count.to_le_bytes())?;
    framed.entry_bytes(
        b"integration_step_ms",
        &TEMPORAL_PHYSICS_IDENTITY
            .integration_step_ms
            .to_bits()
            .to_le_bytes(),
    )?;
    framed.entry_bytes(
        b"race_ticks",
        &TEMPORAL_PHYSICS_IDENTITY.race_ticks.to_le_bytes(),
    )?;
    framed.entry(b"physics_f64", 14 * 8, |framed| {
        for word in TEMPORAL_PHYSICS_IDENTITY.physics_f64_bits() {
            framed.raw(&word.to_le_bytes())?;
        }
        Ok(())
    })?;
    emit_token_codebook_entry(&mut framed, definition.token_codebook)?;
    emit_threshold_offsets_entry(&mut framed, definition.threshold_offsets_mv)?;
    framed.entry_bytes(b"recognition_certificate", &[])?;
    Ok(framed.finish())
}

fn emit_token_codebook_entry(
    framed: &mut FramedSha256,
    codebook: &[super::TokenCode],
) -> Result<(), TemporalCheckpointIdentityError> {
    let code_count = u32::try_from(codebook.len())
        .map_err(|_| TemporalCheckpointIdentityError::FramingLengthOverflow)?;
    let record_bytes = 4u64
        .checked_add(
            4u64.checked_mul(ACTIVE_COLUMNS as u64)
                .ok_or(TemporalCheckpointIdentityError::FramingLengthOverflow)?,
        )
        .ok_or(TemporalCheckpointIdentityError::FramingLengthOverflow)?;
    let payload_len = 4u64
        .checked_add(
            u64::from(code_count)
                .checked_mul(record_bytes)
                .ok_or(TemporalCheckpointIdentityError::FramingLengthOverflow)?,
        )
        .ok_or(TemporalCheckpointIdentityError::FramingLengthOverflow)?;
    framed.entry(b"token_codebook", payload_len, |framed| {
        framed.raw(&code_count.to_le_bytes())?;
        for code in codebook {
            framed.raw(&code.token_id.to_le_bytes())?;
            for column in code.columns {
                framed.raw(&column.to_le_bytes())?;
            }
        }
        Ok(())
    })
}

fn emit_threshold_offsets_entry(
    framed: &mut FramedSha256,
    threshold_offsets_mv: &[f64],
) -> Result<(), TemporalCheckpointIdentityError> {
    let physical_cell_count = u32::try_from(threshold_offsets_mv.len())
        .map_err(|_| TemporalCheckpointIdentityError::FramingLengthOverflow)?;
    let payload_len = 4u64
        .checked_add(
            u64::from(physical_cell_count)
                .checked_mul(8)
                .ok_or(TemporalCheckpointIdentityError::FramingLengthOverflow)?,
        )
        .ok_or(TemporalCheckpointIdentityError::FramingLengthOverflow)?;
    framed.entry(b"threshold_offsets_mv", payload_len, |framed| {
        framed.raw(&physical_cell_count.to_le_bytes())?;
        for offset in threshold_offsets_mv {
            framed.raw(&offset.to_bits().to_le_bytes())?;
        }
        Ok(())
    })
}

fn validate_runtime_identity_inputs(
    runtime: &TemporalCircuitRuntime,
    definition: CheckpointV1CircuitDefinition<'_>,
) -> Result<DerivedCircuitIdentity, TemporalCheckpointIdentityError> {
    let derived = validate_checkpoint_v1_circuit_definition(definition).map_err(|error| {
        TemporalCheckpointIdentityError::RuntimeInvariant(error.runtime_invariant_message())
    })?;
    if runtime.relay_offset != derived.relay_offset
        || runtime.basket_offset != derived.basket_offset
        || runtime.proximal_offset != derived.proximal_offset
        || runtime.proximal_relay_offset != derived.proximal_relay_offset
    {
        return Err(TemporalCheckpointIdentityError::RuntimeInvariant(
            "stored runtime offsets disagree with executed configuration",
        ));
    }
    Ok(derived)
}

fn validate_checkpoint_v1_circuit_definition(
    definition: CheckpointV1CircuitDefinition<'_>,
) -> Result<DerivedCircuitIdentity, CheckpointV1CircuitDefinitionError> {
    if ACTIVE_COLUMNS != 11 || CONTEXT_CELLS != 22 || PAIRS_PER_SEGMENT != 10 {
        return Err(CheckpointV1CircuitDefinitionError::FixedDimensions);
    }
    let config = definition.config;
    if config.column_count < ACTIVE_COLUMNS as u32 {
        return Err(CheckpointV1CircuitDefinitionError::ColumnCount {
            found: config.column_count,
        });
    }
    if config.base_cells_per_column == 0 {
        return Err(CheckpointV1CircuitDefinitionError::BaseCellsPerColumnZero);
    }
    if !(1..=8).contains(&config.novel_race_cap) {
        return Err(CheckpointV1CircuitDefinitionError::NovelRaceCap {
            found: config.novel_race_cap,
        });
    }
    if config.base_cells_per_column > config.novel_race_cap {
        return Err(
            CheckpointV1CircuitDefinitionError::BaseCellsExceedNovelRaceCap {
                base: config.base_cells_per_column,
                race_cap: config.novel_race_cap,
            },
        );
    }
    if config.segment_capacity == 0 || config.segment_capacity > u32::MAX / PAIRS_PER_SEGMENT as u32
    {
        return Err(CheckpointV1CircuitDefinitionError::SegmentCapacity {
            found: config.segment_capacity,
        });
    }
    if !(1..=ACTIVE_COLUMNS as u8).contains(&config.prediction_quorum) {
        return Err(CheckpointV1CircuitDefinitionError::PredictionQuorum {
            found: config.prediction_quorum,
        });
    }
    if config.max_segments_per_cell == 0 {
        return Err(CheckpointV1CircuitDefinitionError::MaxSegmentsPerCellZero);
    }
    for (field, found) in [
        (
            CheckpointV1PermanenceField::Connected,
            config.connected_permanence_ticks,
        ),
        (
            CheckpointV1PermanenceField::Initial,
            config.initial_permanence_ticks,
        ),
        (
            CheckpointV1PermanenceField::Increment,
            config.permanence_increment_ticks,
        ),
        (
            CheckpointV1PermanenceField::NegativeDecrement,
            config.negative_decrement_ticks,
        ),
    ] {
        if !(1..=PERMANENCE_MAX_TICKS).contains(&found) {
            return Err(CheckpointV1CircuitDefinitionError::PermanenceTicks { field, found });
        }
    }
    if !(1 <= config.theta_match
        && config.theta_match <= config.theta_active
        && config.theta_active <= PHYSICAL_CONTACTS_PER_PAIR * PAIRS_PER_SEGMENT as u8)
    {
        return Err(CheckpointV1CircuitDefinitionError::ThresholdOrder {
            theta_match: config.theta_match,
            theta_active: config.theta_active,
        });
    }
    if config.context_policy == TemporalContextPolicy::GenericAfferents
        && config.base_cells_per_column < 2
    {
        return Err(
            CheckpointV1CircuitDefinitionError::GenericAfferentsBaseWidth {
                found: config.base_cells_per_column,
            },
        );
    }
    let base_population = u64::from(config.column_count) * u64::from(config.base_cells_per_column);
    if base_population > u64::from(config.max_pyramidal_cells) {
        return Err(
            CheckpointV1CircuitDefinitionError::BasePopulationExceedsMaximum {
                base: base_population,
                maximum: config.max_pyramidal_cells,
            },
        );
    }
    let base_population_u32 = config
        .column_count
        .checked_mul(u32::from(config.base_cells_per_column))
        .ok_or(
            CheckpointV1CircuitDefinitionError::BasePopulationExceedsMaximum {
                base: base_population,
                maximum: config.max_pyramidal_cells,
            },
        )?;

    let relay_offset = config.max_pyramidal_cells;
    let basket_offset = relay_offset
        .checked_mul(2)
        .ok_or(CheckpointV1CircuitDefinitionError::PhysicalOffsetsOverflow)?;
    let proximal_offset = basket_offset
        .checked_add(config.column_count)
        .ok_or(CheckpointV1CircuitDefinitionError::PhysicalOffsetsOverflow)?;
    let proximal_relay_offset = proximal_offset
        .checked_add(config.column_count)
        .ok_or(CheckpointV1CircuitDefinitionError::PhysicalOffsetsOverflow)?;
    let physical_cell_count = proximal_relay_offset
        .checked_add(config.column_count)
        .ok_or(CheckpointV1CircuitDefinitionError::PhysicalOffsetsOverflow)?;
    if definition.threshold_offsets_mv.len() != physical_cell_count as usize {
        return Err(CheckpointV1CircuitDefinitionError::ThresholdOffsetCount {
            expected: physical_cell_count,
            actual: definition.threshold_offsets_mv.len(),
        });
    }
    if let Some(index) = definition
        .threshold_offsets_mv
        .iter()
        .position(|offset| !offset.is_finite())
    {
        return Err(CheckpointV1CircuitDefinitionError::NonFiniteThresholdOffset { index });
    }
    if definition.token_codebook.is_empty() {
        return Err(CheckpointV1CircuitDefinitionError::EmptyTokenCodebook);
    }
    if u32::try_from(definition.token_codebook.len()).is_err() {
        return Err(CheckpointV1CircuitDefinitionError::TokenCodebookLength {
            actual: definition.token_codebook.len(),
        });
    }
    for (index, code) in definition.token_codebook.iter().enumerate() {
        if code.token_id == u32::MAX {
            return Err(CheckpointV1CircuitDefinitionError::SentinelTokenId { index });
        }
        if code
            .columns
            .iter()
            .any(|&column| column >= config.column_count)
            || code.columns.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(CheckpointV1CircuitDefinitionError::InvalidTokenColumns { index });
        }
    }
    if let Some(left_index) = definition
        .token_codebook
        .windows(2)
        .position(|pair| pair[0].token_id >= pair[1].token_id)
    {
        return Err(CheckpointV1CircuitDefinitionError::TokenIdsNotStrict { left_index });
    }
    for left_index in 0..definition.token_codebook.len() {
        if let Some(relative_right) = definition.token_codebook[left_index + 1..]
            .iter()
            .position(|right| right.columns == definition.token_codebook[left_index].columns)
        {
            return Err(CheckpointV1CircuitDefinitionError::DuplicateTokenColumns {
                left_index,
                right_index: left_index + 1 + relative_right,
            });
        }
    }
    let physics = TEMPORAL_PHYSICS_IDENTITY;
    if !physics.integration_step_ms.is_finite()
        || physics.integration_step_ms <= 0.0
        || physics.race_ticks == 0
        || physics
            .physics_f64_bits()
            .iter()
            .any(|bits| !f64::from_bits(*bits).is_finite())
    {
        return Err(CheckpointV1CircuitDefinitionError::PhysicsInvariant);
    }

    let (source_gid_base, source_count) = match config.context_policy {
        TemporalContextPolicy::BoundWinners => (0, config.max_pyramidal_cells),
        TemporalContextPolicy::GenericAfferents => (proximal_offset, config.column_count),
    };
    Ok(DerivedCircuitIdentity {
        base_population: base_population_u32,
        relay_offset,
        basket_offset,
        proximal_offset,
        proximal_relay_offset,
        physical_cell_count,
        source_gid_base,
        source_count,
    })
}

fn max_distinct_code_overlap(codebook: &[super::TokenCode]) -> u8 {
    let mut maximum = 0u8;
    for left in 0..codebook.len() {
        for right in &codebook[left + 1..] {
            let mut left_index = 0usize;
            let mut right_index = 0usize;
            let mut overlap = 0u8;
            while left_index < ACTIVE_COLUMNS && right_index < ACTIVE_COLUMNS {
                match codebook[left].columns[left_index].cmp(&right.columns[right_index]) {
                    std::cmp::Ordering::Less => left_index += 1,
                    std::cmp::Ordering::Greater => right_index += 1,
                    std::cmp::Ordering::Equal => {
                        overlap += 1;
                        left_index += 1;
                        right_index += 1;
                    }
                }
            }
            maximum = maximum.max(overlap);
        }
    }
    maximum
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1AllocationSite {
    ColumnReplay,
    SemanticPostings,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrepareCheckpointV1ExportError {
    Identity(TemporalCheckpointIdentityError),
    Codec(CheckpointV1CodecError),
    State,
    StoreIndex(TemporalStorePreflightError),
    RetryableAllocation(CheckpointV1AllocationSite),
}

pub(crate) struct PreparedCheckpointV1Export<'a> {
    runtime: &'a TemporalCircuitRuntime,
    identity: TemporalCheckpointV1Identity,
    layout: CheckpointV1Layout,
    scalars: CheckpointV1Scalars,
    semantic_index_digest: CheckpointV1Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckpointV1CanonicalMismatch {
    ByteMismatch {
        offset: u64,
        expected: u8,
        actual: u8,
    },
    ExpectedEnded {
        offset: u64,
        actual: u8,
    },
    ExpectedTrailing {
        offset: u64,
        expected: u8,
    },
}

impl CheckpointV1CanonicalMismatch {
    const fn offset(self) -> u64 {
        match self {
            Self::ByteMismatch { offset, .. }
            | Self::ExpectedEnded { offset, .. }
            | Self::ExpectedTrailing { offset, .. } => offset,
        }
    }
}

#[derive(Debug)]
enum CompareCheckpointV1CanonicalBytesError {
    Mismatch(CheckpointV1CanonicalMismatch),
    Encode(CheckpointV1EncodeError),
}

struct CheckpointV1ExportPreparation<'a> {
    runtime: &'a TemporalCircuitRuntime,
    identity: TemporalCheckpointV1Identity,
    layout: CheckpointV1Layout,
    scalars: CheckpointV1Scalars,
    column_scratch_len: usize,
}

struct CheckpointV1ValidatedExportState<'a> {
    preparation: CheckpointV1ExportPreparation<'a>,
    semantic_scratch_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1BodySection {
    CohortColumns,
    PriorContext,
    WinnerContext,
    SegmentOwners,
    SegmentSources,
    SegmentTicks,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1SeekStage {
    InspectLogicalEnd,
    HeaderBackfill,
    VerifyLogicalEnd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1WriteStage {
    ReserveHeader,
    Body(CheckpointV1BodySection),
    BackfillHeader,
}

#[derive(Debug)]
pub(crate) enum CheckpointV1EncodeError {
    PreparedCodec(CheckpointV1CodecError),
    PreparedScalarsMismatch,
    PreparedPairCountMismatch {
        declared: u64,
        observed: u64,
    },
    BodySectionCount {
        section: CheckpointV1BodySection,
        declared: u64,
        observed: u64,
    },
    BodyByteCount {
        declared: u64,
        observed: u64,
    },
    TotalByteCount {
        declared: u64,
        observed: u64,
    },
    BodyBufferInvariant(CheckpointV1BodySection),
    SinkNotEmpty {
        logical_end: u64,
    },
    Seek {
        stage: CheckpointV1SeekStage,
        source: io::Error,
    },
    SeekPosition {
        stage: CheckpointV1SeekStage,
        expected: u64,
        actual: u64,
    },
    Write {
        stage: CheckpointV1WriteStage,
        source: io::Error,
    },
    WriteCount {
        stage: CheckpointV1WriteStage,
        requested: usize,
        reported: usize,
    },
    HeaderCodec(CheckpointV1CodecError),
    Flush {
        source: io::Error,
    },
}

impl std::fmt::Display for CheckpointV1EncodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreparedCodec(source) => {
                write!(formatter, "prepared checkpoint codec contradiction: {source:?}")
            }
            Self::PreparedScalarsMismatch => {
                formatter.write_str("prepared checkpoint scalars no longer match the runtime")
            }
            Self::PreparedPairCountMismatch { declared, observed } => write!(
                formatter,
                "prepared checkpoint pair count mismatch: declared {declared}, observed {observed}",
            ),
            Self::BodySectionCount {
                section,
                declared,
                observed,
            } => write!(
                formatter,
                "checkpoint body section {section:?} byte count mismatch: declared {declared}, observed {observed}",
            ),
            Self::BodyByteCount { declared, observed } => write!(
                formatter,
                "checkpoint body byte count mismatch: declared {declared}, observed {observed}",
            ),
            Self::TotalByteCount { declared, observed } => write!(
                formatter,
                "checkpoint total byte count mismatch: declared {declared}, observed {observed}",
            ),
            Self::BodyBufferInvariant(section) => write!(
                formatter,
                "checkpoint body buffer invariant failed in section {section:?}",
            ),
            Self::SinkNotEmpty { logical_end } => write!(
                formatter,
                "checkpoint sink must be empty, but its logical end is {logical_end}",
            ),
            Self::Seek { stage, source } => {
                write!(formatter, "checkpoint seek failed at {stage:?}: {source}")
            }
            Self::SeekPosition {
                stage,
                expected,
                actual,
            } => write!(
                formatter,
                "checkpoint seek at {stage:?} reported {actual}, expected {expected}",
            ),
            Self::Write { stage, source } => {
                write!(formatter, "checkpoint write failed at {stage:?}: {source}")
            }
            Self::WriteCount {
                stage,
                requested,
                reported,
            } => write!(
                formatter,
                "checkpoint write at {stage:?} reported {reported} bytes for a {requested}-byte request",
            ),
            Self::HeaderCodec(source) => {
                write!(formatter, "checkpoint header codec contradiction: {source:?}")
            }
            Self::Flush { source } => write!(formatter, "checkpoint flush failed: {source}"),
        }
    }
}

impl std::error::Error for CheckpointV1EncodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Seek { source, .. } | Self::Write { source, .. } | Self::Flush { source } => {
                Some(source)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::enum_variant_names)] // Every private category names the invariant family it proves.
enum CheckpointV1StateError {
    ScalarInvariant,
    LayoutInvariant,
    ColumnCsrInvariant,
    LearnedStateInvariant,
    CohortChronologyInvariant,
    ContextInvariant,
}

impl fmt::Display for CheckpointV1StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ScalarInvariant => "checkpoint scalar relationships are not canonical",
            Self::LayoutInvariant => "checkpoint dynamic layout is not canonical",
            Self::ColumnCsrInvariant => "live runtime column CSR is not canonical",
            Self::LearnedStateInvariant => "checkpoint learned rows are not canonical",
            Self::CohortChronologyInvariant => "checkpoint cohort chronology is unreachable",
            Self::ContextInvariant => "checkpoint context is not runtime-reachable",
        })
    }
}

impl Error for CheckpointV1StateError {}

#[derive(Clone, Copy)]
enum CheckpointV1U32View<'a> {
    Native(&'a [u32]),
    LittleEndian(CheckpointV1LeU32Slice<'a>),
}

impl CheckpointV1U32View<'_> {
    fn len(self) -> usize {
        match self {
            Self::Native(values) => values.len(),
            Self::LittleEndian(values) => values.len(),
        }
    }

    fn get(self, index: usize) -> Option<u32> {
        match self {
            Self::Native(values) => values.get(index).copied(),
            Self::LittleEndian(values) => values.get(index),
        }
    }
}

#[derive(Clone, Copy)]
struct CheckpointV1LogicalStateView<'a> {
    config: TemporalCircuitConfig,
    token_codebook: &'a [TokenCode],
    cohort_columns: CheckpointV1U32View<'a>,
    next_reserve_cell: u32,
    segment_count: u32,
    previous_context: CheckpointV1U32View<'a>,
    previous_context_len: u8,
    previous_winner_context: CheckpointV1U32View<'a>,
    previous_winner_context_len: u8,
    event_count: u64,
    learning_step: u64,
    scientific_stop: Option<TemporalPoisonCode>,
    relay_offset: u32,
    proximal_offset: u32,
    proximal_relay_offset: u32,
}

#[derive(Clone, Copy)]
struct CheckedCheckpointV1LogicalShape {
    contexts_empty: bool,
    column_count: usize,
    base_width: u32,
    race_cap: u32,
    base_total: u32,
    committed_owner_count: u32,
}

#[derive(Clone, Copy)]
struct CheckpointV1StateView<'a> {
    config: TemporalCircuitConfig,
    token_codebook: &'a [TokenCode],
    column_offsets: &'a [usize],
    column_cells: &'a [u32],
    cohort_columns: &'a [u32],
    next_reserve_cell: u32,
    segment_count: u32,
    previous_context: &'a [u32; CONTEXT_CELLS],
    previous_context_len: u8,
    previous_winner_context: &'a [u32; CONTEXT_CELLS],
    previous_winner_context_len: u8,
    event_count: u64,
    learning_step: u64,
    scientific_stop: Option<TemporalPoisonCode>,
    relay_offset: u32,
    proximal_offset: u32,
    proximal_relay_offset: u32,
}

#[derive(Clone, Copy)]
struct CheckpointV1OwnedRow {
    owner: u32,
    row: TemporalSegmentRow,
}

trait CheckpointV1RowView {
    fn owner(&self) -> Option<u32>;

    fn declared_live_pair_count(&self) -> Option<usize>;

    fn source(&self, slot: usize) -> Option<u32>;

    fn permanence_ticks(&self, slot: usize) -> Option<u16>;
}

impl CheckpointV1RowView for CheckpointV1OwnedRow {
    fn owner(&self) -> Option<u32> {
        Some(self.owner)
    }

    fn declared_live_pair_count(&self) -> Option<usize> {
        Some(usize::from(self.row.live_pair_count))
    }

    fn source(&self, slot: usize) -> Option<u32> {
        self.row.normalized_sources.get(slot).copied()
    }

    fn permanence_ticks(&self, slot: usize) -> Option<u16> {
        self.row.permanence_ticks.get(slot).copied()
    }
}

#[derive(Clone, Copy)]
struct CheckpointV1SoaRow<'a> {
    segment_index: usize,
    owners: CheckpointV1LeU32Slice<'a>,
    sources: CheckpointV1LeU32Slice<'a>,
    ticks: CheckpointV1LeU16Slice<'a>,
}

impl CheckpointV1RowView for CheckpointV1SoaRow<'_> {
    fn owner(&self) -> Option<u32> {
        self.owners.get(self.segment_index)
    }

    fn declared_live_pair_count(&self) -> Option<usize> {
        None
    }

    fn source(&self, slot: usize) -> Option<u32> {
        let index = self
            .segment_index
            .checked_mul(PAIRS_PER_SEGMENT)?
            .checked_add(slot)?;
        self.sources.get(index)
    }

    fn permanence_ticks(&self, slot: usize) -> Option<u16> {
        let index = self
            .segment_index
            .checked_mul(PAIRS_PER_SEGMENT)?
            .checked_add(slot)?;
        self.ticks.get(index)
    }
}

trait CheckpointV1OwnerLoads {
    fn observe_owner(&mut self, owner: u32) -> Option<u16>;

    fn load(&mut self, owner: u32) -> Option<u16>;
}

struct CheckpointV1AuthoritativeOwnerLoads<F> {
    load: F,
}

impl<F> CheckpointV1OwnerLoads for CheckpointV1AuthoritativeOwnerLoads<F>
where
    F: FnMut(u32) -> Option<u8>,
{
    fn observe_owner(&mut self, owner: u32) -> Option<u16> {
        (self.load)(owner).map(u16::from)
    }

    fn load(&mut self, owner: u32) -> Option<u16> {
        (self.load)(owner).map(u16::from)
    }
}

struct CheckpointV1ReplayedOwnerLoads<'a> {
    loads: &'a mut [u16],
}

impl CheckpointV1OwnerLoads for CheckpointV1ReplayedOwnerLoads<'_> {
    fn observe_owner(&mut self, owner: u32) -> Option<u16> {
        let load = self.loads.get_mut(usize::try_from(owner).ok()?)?;
        *load = load.checked_add(1)?;
        Some(*load)
    }

    fn load(&mut self, owner: u32) -> Option<u16> {
        self.loads.get(usize::try_from(owner).ok()?).copied()
    }
}

fn checkpoint_v1_state_view(runtime: &TemporalCircuitRuntime) -> CheckpointV1StateView<'_> {
    CheckpointV1StateView {
        config: runtime.config,
        token_codebook: &runtime.token_codebook,
        column_offsets: &runtime.column_offsets,
        column_cells: &runtime.column_cells,
        cohort_columns: &runtime.cohort_columns,
        next_reserve_cell: runtime.next_reserve_cell,
        segment_count: runtime.store.runtime_segment_count(),
        previous_context: &runtime.previous_context,
        previous_context_len: runtime.previous_context_len,
        previous_winner_context: &runtime.previous_winner_context,
        previous_winner_context_len: runtime.previous_winner_context_len,
        event_count: runtime.event_count,
        learning_step: runtime.learning_step,
        scientific_stop: runtime.scientific_stop,
        relay_offset: runtime.relay_offset,
        proximal_offset: runtime.proximal_offset,
        proximal_relay_offset: runtime.proximal_relay_offset,
    }
}

fn checkpoint_v1_logical_state_view(
    state: CheckpointV1StateView<'_>,
) -> CheckpointV1LogicalStateView<'_> {
    CheckpointV1LogicalStateView {
        config: state.config,
        token_codebook: state.token_codebook,
        cohort_columns: CheckpointV1U32View::Native(state.cohort_columns),
        next_reserve_cell: state.next_reserve_cell,
        segment_count: state.segment_count,
        previous_context: CheckpointV1U32View::Native(state.previous_context),
        previous_context_len: state.previous_context_len,
        previous_winner_context: CheckpointV1U32View::Native(state.previous_winner_context),
        previous_winner_context_len: state.previous_winner_context_len,
        event_count: state.event_count,
        learning_step: state.learning_step,
        scientific_stop: state.scientific_stop,
        relay_offset: state.relay_offset,
        proximal_offset: state.proximal_offset,
        proximal_relay_offset: state.proximal_relay_offset,
    }
}

fn validate_checkpoint_v1_runtime_state(
    runtime: &TemporalCircuitRuntime,
    column_replay: &mut [u32],
) -> Result<(), CheckpointV1StateError> {
    validate_checkpoint_v1_state(
        checkpoint_v1_state_view(runtime),
        column_replay,
        |segment_id| {
            Some(CheckpointV1OwnedRow {
                owner: runtime.store.runtime_segment_owner(segment_id)?,
                row: runtime.store.runtime_segment_row(segment_id).ok()?,
            })
        },
        |owner| runtime.store.runtime_owner_load(owner),
    )
}

/// Validate one borrowed logical state without allocating.
///
/// Immutable runtime identity must already have succeeded. `row_at` is the
/// exact segment-ID logical-row view and `owner_load` is the store-owned cache
/// maintained by live construction and mutation. Checkpoint import uses the
/// same logical policy through direct checked-SoA rows and replayed typed loads.
fn validate_checkpoint_v1_state(
    state: CheckpointV1StateView<'_>,
    column_replay: &mut [u32],
    mut row_at: impl FnMut(u32) -> Option<CheckpointV1OwnedRow>,
    owner_load: impl FnMut(u32) -> Option<u8>,
) -> Result<(), CheckpointV1StateError> {
    let logical = checkpoint_v1_logical_state_view(state);
    let shape = validate_checkpoint_v1_logical_shape(&logical)?;
    replay_checkpoint_v1_final_occupancies(&logical, shape, column_replay)?;
    validate_checkpoint_v1_live_csr(state, &logical, shape, column_replay)?;
    let mut owner_loads = CheckpointV1AuthoritativeOwnerLoads { load: owner_load };
    validate_checkpoint_v1_logical_content(
        &logical,
        shape,
        column_replay,
        &mut row_at,
        &mut owner_loads,
    )?;
    Ok(())
}

/// Prove scalar relationships, dynamic shape, cursor, and cohort domains
/// without allocating or reading segment-row content.
fn validate_checkpoint_v1_logical_shape(
    state: &CheckpointV1LogicalStateView<'_>,
) -> Result<CheckedCheckpointV1LogicalShape, CheckpointV1StateError> {
    let contexts_empty = match (
        state.previous_context_len,
        state.previous_winner_context_len,
    ) {
        (0, 0) => true,
        (CONTEXT_CELLS_U8, CONTEXT_CELLS_U8) => false,
        _ => return Err(CheckpointV1StateError::ScalarInvariant),
    };
    if (state.event_count == 0) != contexts_empty
        || (state.config.plasticity_policy == TemporalPlasticityPolicy::GrowthOff
            && state.learning_step != 0)
        || checkpoint_v1_poison_code(state.scientific_stop).is_err()
    {
        return Err(CheckpointV1StateError::ScalarInvariant);
    }

    let column_count = usize::try_from(state.config.column_count)
        .map_err(|_| CheckpointV1StateError::LayoutInvariant)?;
    let base_width = u32::from(state.config.base_cells_per_column);
    let race_cap = u32::from(state.config.novel_race_cap);
    if column_count < ACTIVE_COLUMNS
        || base_width == 0
        || race_cap == 0
        || race_cap > 8
        || base_width > race_cap
        || (state.config.context_policy == TemporalContextPolicy::GenericAfferents
            && base_width < 2)
        || state.config.segment_capacity == 0
        || state.config.segment_capacity > u32::MAX / PAIRS_PER_SEGMENT as u32
        || !(1..=ACTIVE_COLUMNS as u8).contains(&state.config.prediction_quorum)
        || state.config.max_segments_per_cell == 0
        || !(1..=PERMANENCE_MAX_TICKS).contains(&state.config.connected_permanence_ticks)
        || !(1..=PERMANENCE_MAX_TICKS).contains(&state.config.initial_permanence_ticks)
        || !(1..=PERMANENCE_MAX_TICKS).contains(&state.config.permanence_increment_ticks)
        || !(1..=PERMANENCE_MAX_TICKS).contains(&state.config.negative_decrement_ticks)
        || !(1 <= state.config.theta_match
            && state.config.theta_match <= state.config.theta_active
            && state.config.theta_active <= PHYSICAL_CONTACTS_PER_PAIR * PAIRS_PER_SEGMENT as u8)
        || state.segment_count > state.config.segment_capacity
    {
        return Err(CheckpointV1StateError::LayoutInvariant);
    }
    let base_total = u64::from(state.config.column_count)
        .checked_mul(u64::from(base_width))
        .filter(|total| *total <= u64::from(state.config.max_pyramidal_cells))
        .and_then(|total| u32::try_from(total).ok())
        .ok_or(CheckpointV1StateError::LayoutInvariant)?;
    let relay_offset = state.config.max_pyramidal_cells;
    let basket_offset = relay_offset
        .checked_mul(2)
        .ok_or(CheckpointV1StateError::LayoutInvariant)?;
    let proximal_offset = basket_offset
        .checked_add(state.config.column_count)
        .ok_or(CheckpointV1StateError::LayoutInvariant)?;
    let proximal_relay_offset = proximal_offset
        .checked_add(state.config.column_count)
        .ok_or(CheckpointV1StateError::LayoutInvariant)?;
    proximal_relay_offset
        .checked_add(state.config.column_count)
        .ok_or(CheckpointV1StateError::LayoutInvariant)?;
    if state.relay_offset != relay_offset
        || state.proximal_offset != proximal_offset
        || state.proximal_relay_offset != proximal_relay_offset
    {
        return Err(CheckpointV1StateError::LayoutInvariant);
    }
    let cohort_count = u32::try_from(state.cohort_columns.len())
        .map_err(|_| CheckpointV1StateError::LayoutInvariant)?;
    let next_reserve_cell = u64::from(cohort_count)
        .checked_mul(u64::from(race_cap))
        .and_then(|reserve_cells| reserve_cells.checked_add(u64::from(base_total)))
        .filter(|next| *next <= u64::from(state.config.max_pyramidal_cells))
        .and_then(|next| u32::try_from(next).ok())
        .ok_or(CheckpointV1StateError::LayoutInvariant)?;
    if state.next_reserve_cell != next_reserve_cell {
        return Err(CheckpointV1StateError::LayoutInvariant);
    }

    for index in 0..state.cohort_columns.len() {
        let column = state
            .cohort_columns
            .get(index)
            .ok_or(CheckpointV1StateError::LayoutInvariant)?;
        if column >= state.config.column_count {
            return Err(CheckpointV1StateError::LayoutInvariant);
        }
    }

    Ok(CheckedCheckpointV1LogicalShape {
        contexts_empty,
        column_count,
        base_width,
        race_cap,
        base_total,
        committed_owner_count: next_reserve_cell,
    })
}

fn replay_checkpoint_v1_final_occupancies(
    state: &CheckpointV1LogicalStateView<'_>,
    shape: CheckedCheckpointV1LogicalShape,
    column_replay: &mut [u32],
) -> Result<usize, CheckpointV1StateError> {
    if column_replay.len() != shape.column_count {
        return Err(CheckpointV1StateError::LayoutInvariant);
    }
    column_replay.fill(shape.base_width);
    for index in 0..state.cohort_columns.len() {
        let column = state
            .cohort_columns
            .get(index)
            .ok_or(CheckpointV1StateError::LayoutInvariant)?;
        let column_index =
            usize::try_from(column).map_err(|_| CheckpointV1StateError::LayoutInvariant)?;
        let replay = column_replay
            .get_mut(column_index)
            .ok_or(CheckpointV1StateError::LayoutInvariant)?;
        *replay = replay
            .checked_add(shape.race_cap)
            .ok_or(CheckpointV1StateError::LayoutInvariant)?;
    }

    let max = column_replay.iter().copied().max().unwrap_or(0);
    usize::try_from(max).map_err(|_| CheckpointV1StateError::LayoutInvariant)
}

fn validate_checkpoint_v1_live_csr(
    state: CheckpointV1StateView<'_>,
    logical: &CheckpointV1LogicalStateView<'_>,
    shape: CheckedCheckpointV1LogicalShape,
    column_replay: &[u32],
) -> Result<(), CheckpointV1StateError> {
    let expected_offset_count = shape
        .column_count
        .checked_add(1)
        .ok_or(CheckpointV1StateError::ColumnCsrInvariant)?;
    let committed_count = usize::try_from(shape.committed_owner_count)
        .map_err(|_| CheckpointV1StateError::ColumnCsrInvariant)?;
    if state.column_offsets.len() != expected_offset_count
        || state.column_offsets.first().copied() != Some(0)
        || state.column_cells.len() != committed_count
    {
        return Err(CheckpointV1StateError::ColumnCsrInvariant);
    }
    let base_width_usize = usize::try_from(shape.base_width)
        .map_err(|_| CheckpointV1StateError::ColumnCsrInvariant)?;
    let mut expected_start = 0usize;
    for (column, &replayed_occupancy) in column_replay.iter().enumerate() {
        let occupancy = usize::try_from(replayed_occupancy)
            .map_err(|_| CheckpointV1StateError::ColumnCsrInvariant)?;
        let expected_end = expected_start
            .checked_add(occupancy)
            .ok_or(CheckpointV1StateError::ColumnCsrInvariant)?;
        if state.column_offsets[column] != expected_start
            || state.column_offsets[column + 1] != expected_end
            || expected_end > state.column_cells.len()
        {
            return Err(CheckpointV1StateError::ColumnCsrInvariant);
        }
        let cells = &state.column_cells[expected_start..expected_end];
        if cells.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(CheckpointV1StateError::ColumnCsrInvariant);
        }
        let base_start = u64::try_from(column)
            .ok()
            .and_then(|column| column.checked_mul(u64::from(shape.base_width)))
            .and_then(|start| u32::try_from(start).ok())
            .ok_or(CheckpointV1StateError::ColumnCsrInvariant)?;
        for (offset, &cell) in cells[..base_width_usize].iter().enumerate() {
            let expected = base_start
                .checked_add(offset as u32)
                .ok_or(CheckpointV1StateError::ColumnCsrInvariant)?;
            if cell != expected {
                return Err(CheckpointV1StateError::ColumnCsrInvariant);
            }
        }
        for &cell in &cells[base_width_usize..] {
            if !(shape.base_total..shape.committed_owner_count).contains(&cell) {
                return Err(CheckpointV1StateError::ColumnCsrInvariant);
            }
            let cohort_index = usize::try_from((cell - shape.base_total) / shape.race_cap)
                .map_err(|_| CheckpointV1StateError::ColumnCsrInvariant)?;
            if logical.cohort_columns.get(cohort_index) != Some(column as u32) {
                return Err(CheckpointV1StateError::ColumnCsrInvariant);
            }
        }
        expected_start = expected_end;
    }
    if expected_start != state.column_cells.len() {
        return Err(CheckpointV1StateError::ColumnCsrInvariant);
    }

    Ok(())
}

fn validate_checkpoint_v1_logical_content<R>(
    state: &CheckpointV1LogicalStateView<'_>,
    shape: CheckedCheckpointV1LogicalShape,
    column_replay: &mut [u32],
    mut row_at: impl FnMut(u32) -> Option<R>,
    owner_loads: &mut impl CheckpointV1OwnerLoads,
) -> Result<usize, CheckpointV1StateError>
where
    R: CheckpointV1RowView,
{
    let owner_cap = u16::from(state.config.max_segments_per_cell);
    for segment_id in 0..state.segment_count {
        let row = row_at(segment_id).ok_or(CheckpointV1StateError::LearnedStateInvariant)?;
        let owner = row
            .owner()
            .ok_or(CheckpointV1StateError::LearnedStateInvariant)?;
        if checkpoint_v1_owner_column(state, shape, owner).is_none()
            || owner_loads
                .observe_owner(owner)
                .filter(|load| *load != 0 && *load <= owner_cap && *load <= u16::from(u8::MAX))
                .is_none()
            || !checkpoint_v1_row_is_reachable(state, shape, &row)
        {
            return Err(CheckpointV1StateError::LearnedStateInvariant);
        }
    }

    for owner in 0..shape.committed_owner_count {
        let load = owner_loads
            .load(owner)
            .ok_or(CheckpointV1StateError::LearnedStateInvariant)?;
        if load > owner_cap || load > u16::from(u8::MAX) {
            return Err(CheckpointV1StateError::LearnedStateInvariant);
        }
    }

    column_replay.fill(0);
    for cohort_index in 0..state.cohort_columns.len() {
        let column = state
            .cohort_columns
            .get(cohort_index)
            .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?;
        let column_index = usize::try_from(column)
            .map_err(|_| CheckpointV1StateError::CohortChronologyInvariant)?;
        let previous_cohort_plus_one = *column_replay
            .get(column_index)
            .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?;
        let predecessor_start = if previous_cohort_plus_one == 0 {
            column
                .checked_mul(shape.base_width)
                .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?
        } else {
            shape
                .base_total
                .checked_add(
                    previous_cohort_plus_one
                        .checked_sub(1)
                        .and_then(|previous| previous.checked_mul(shape.race_cap))
                        .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?,
                )
                .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?
        };
        let predecessor_count = if previous_cohort_plus_one == 0 {
            shape.base_width
        } else {
            shape.race_cap
        };
        let predecessor_end = predecessor_start
            .checked_add(predecessor_count)
            .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?;
        if (predecessor_start..predecessor_end)
            .any(|owner| owner_loads.load(owner) != Some(owner_cap))
        {
            return Err(CheckpointV1StateError::CohortChronologyInvariant);
        }

        let cohort_index = u32::try_from(cohort_index)
            .map_err(|_| CheckpointV1StateError::CohortChronologyInvariant)?;
        let cohort_start = shape
            .base_total
            .checked_add(
                cohort_index
                    .checked_mul(shape.race_cap)
                    .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?,
            )
            .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?;
        let cohort_end = cohort_start
            .checked_add(shape.race_cap)
            .filter(|end| *end <= shape.committed_owner_count)
            .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?;
        if !(cohort_start..cohort_end)
            .any(|owner| owner_loads.load(owner).is_some_and(|load| load != 0))
        {
            return Err(CheckpointV1StateError::CohortChronologyInvariant);
        }
        column_replay[column_index] = cohort_index
            .checked_add(1)
            .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?;
    }

    validate_checkpoint_v1_context(state, shape)?;
    replay_checkpoint_v1_final_occupancies(state, shape, column_replay)
}

fn validate_checkpoint_v1_context(
    state: &CheckpointV1LogicalStateView<'_>,
    shape: CheckedCheckpointV1LogicalShape,
) -> Result<(), CheckpointV1StateError> {
    if shape.contexts_empty {
        return Ok(());
    }
    let mut context_columns = [0u32; ACTIVE_COLUMNS];
    match state.config.context_policy {
        TemporalContextPolicy::BoundWinners => {
            for index in 0..CONTEXT_CELLS {
                if state.previous_context.get(index) != state.previous_winner_context.get(index) {
                    return Err(CheckpointV1StateError::ContextInvariant);
                }
            }
            for (index, column) in context_columns.iter_mut().enumerate() {
                let winner = state
                    .previous_context
                    .get(index)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                *column = checkpoint_v1_owner_column(state, shape, winner)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                let expected_relay = state
                    .relay_offset
                    .checked_add(winner)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                if state.previous_context.get(ACTIVE_COLUMNS + index) != Some(expected_relay) {
                    return Err(CheckpointV1StateError::ContextInvariant);
                }
            }
        }
        TemporalContextPolicy::GenericAfferents => {
            for (index, column) in context_columns.iter_mut().enumerate() {
                let source = state
                    .previous_context
                    .get(index)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                *column = source
                    .checked_sub(state.proximal_offset)
                    .filter(|column| *column < state.config.column_count)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                let expected_source_relay = state
                    .proximal_relay_offset
                    .checked_add(*column)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                if state.previous_context.get(ACTIVE_COLUMNS + index) != Some(expected_source_relay)
                {
                    return Err(CheckpointV1StateError::ContextInvariant);
                }

                let winner = state
                    .previous_winner_context
                    .get(index)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                if checkpoint_v1_owner_column(state, shape, winner) != Some(*column) {
                    return Err(CheckpointV1StateError::ContextInvariant);
                }
                let expected_winner_relay = state
                    .relay_offset
                    .checked_add(winner)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                if state.previous_winner_context.get(ACTIVE_COLUMNS + index)
                    != Some(expected_winner_relay)
                {
                    return Err(CheckpointV1StateError::ContextInvariant);
                }
            }
        }
    }
    if context_columns.windows(2).any(|pair| pair[0] >= pair[1])
        || state
            .token_codebook
            .iter()
            .filter(|code| code.columns == context_columns)
            .count()
            != 1
    {
        return Err(CheckpointV1StateError::ContextInvariant);
    }
    Ok(())
}

fn checkpoint_v1_owner_column(
    state: &CheckpointV1LogicalStateView<'_>,
    shape: CheckedCheckpointV1LogicalShape,
    owner: u32,
) -> Option<u32> {
    if owner < shape.base_total {
        return Some(owner / shape.base_width);
    }
    if owner >= shape.committed_owner_count {
        return None;
    }
    let cohort_index = usize::try_from((owner - shape.base_total) / shape.race_cap).ok()?;
    state
        .cohort_columns
        .get(cohort_index)
        .filter(|column| *column < state.config.column_count)
}

fn checkpoint_v1_row_is_reachable(
    state: &CheckpointV1LogicalStateView<'_>,
    shape: CheckedCheckpointV1LogicalShape,
    row: &impl CheckpointV1RowView,
) -> bool {
    let mut saw_sentinel = false;
    let mut live_pair_count = 0usize;
    let mut previous_source = None;
    for slot in 0..PAIRS_PER_SEGMENT {
        let Some(source) = row.source(slot) else {
            return false;
        };
        let Some(ticks) = row.permanence_ticks(slot) else {
            return false;
        };
        if source == u32::MAX {
            saw_sentinel = true;
            if ticks != 0 {
                return false;
            }
            continue;
        }
        if saw_sentinel
            || !(1..=PERMANENCE_MAX_TICKS).contains(&ticks)
            || previous_source.is_some_and(|previous| previous >= source)
            || match state.config.context_policy {
                TemporalContextPolicy::BoundWinners => {
                    checkpoint_v1_owner_column(state, shape, source).is_none()
                }
                TemporalContextPolicy::GenericAfferents => source >= state.config.column_count,
            }
        {
            return false;
        }
        previous_source = Some(source);
        live_pair_count += 1;
    }
    row.declared_live_pair_count()
        .is_none_or(|declared| declared == live_pair_count)
}

impl TemporalCircuitRuntime {
    /// Freeze the currently executable exact-generic recognition route.
    ///
    /// Route 1 remains a certificate-only design and cannot identify a live
    /// runtime until its physical bucket execution path exists.
    pub(crate) fn prepare_checkpoint_v1_export(
        &self,
        authority: ValidatedCheckpointAuthority,
    ) -> Result<PreparedCheckpointV1Export<'_>, PrepareCheckpointV1ExportError> {
        self.prepare_checkpoint_v1_export_inner(authority, |_| false)
    }

    #[cfg(test)]
    fn prepare_checkpoint_v1_export_with_failpoint(
        &self,
        authority: ValidatedCheckpointAuthority,
        failpoint: CheckpointV1AllocationSite,
    ) -> Result<PreparedCheckpointV1Export<'_>, PrepareCheckpointV1ExportError> {
        self.prepare_checkpoint_v1_export_inner(authority, |site| site == failpoint)
    }

    fn prepare_checkpoint_v1_export_inner(
        &self,
        authority: ValidatedCheckpointAuthority,
        mut fail_before_reserve: impl FnMut(CheckpointV1AllocationSite) -> bool,
    ) -> Result<PreparedCheckpointV1Export<'_>, PrepareCheckpointV1ExportError> {
        let preparation = self.begin_checkpoint_v1_export(authority)?;
        let mut column_replay = checkpoint_v1_zeroed_u32_scratch(
            preparation.column_scratch_len(),
            CheckpointV1AllocationSite::ColumnReplay,
            &mut fail_before_reserve,
        )?;
        let validated = preparation.validate_state(&mut column_replay)?;
        let mut semantic_scratch = checkpoint_v1_zeroed_u32_scratch(
            validated.semantic_scratch_len(),
            CheckpointV1AllocationSite::SemanticPostings,
            &mut fail_before_reserve,
        )?;
        validated.finish(&mut semantic_scratch)
    }

    fn begin_checkpoint_v1_export(
        &self,
        authority: ValidatedCheckpointAuthority,
    ) -> Result<CheckpointV1ExportPreparation<'_>, PrepareCheckpointV1ExportError> {
        let identity = self
            .checkpoint_v1_identity(authority, TemporalRecognitionRoute::ExactGeneric)
            .map_err(PrepareCheckpointV1ExportError::Identity)?;

        let segment_count = self.store.runtime_segment_count();
        let cohort_count = u32::try_from(self.cohort_columns.len()).map_err(|_| {
            PrepareCheckpointV1ExportError::Codec(CheckpointV1CodecError::LengthOverflow)
        })?;
        let scalars = CheckpointV1Scalars::new(
            segment_count,
            cohort_count,
            self.next_reserve_cell,
            self.event_count,
            self.learning_step,
            self.previous_context_len,
            self.previous_winner_context_len,
            self.scientific_stop,
        )
        .map_err(PrepareCheckpointV1ExportError::Codec)?;
        let layout = CheckpointV1Layout::checked(
            segment_count,
            cohort_count,
            self.previous_context_len,
            self.previous_winner_context_len,
        )
        .map_err(PrepareCheckpointV1ExportError::Codec)?;

        let column_count = usize::try_from(self.config.column_count).map_err(|_| {
            PrepareCheckpointV1ExportError::Codec(CheckpointV1CodecError::LengthOverflow)
        })?;

        Ok(CheckpointV1ExportPreparation {
            runtime: self,
            identity,
            layout,
            scalars,
            column_scratch_len: column_count,
        })
    }
}

impl<'a> CheckpointV1ExportPreparation<'a> {
    const fn column_scratch_len(&self) -> usize {
        self.column_scratch_len
    }

    fn validate_state(
        self,
        column_replay: &mut [u32],
    ) -> Result<CheckpointV1ValidatedExportState<'a>, PrepareCheckpointV1ExportError> {
        let provided = column_replay.len().min(self.column_scratch_len);
        validate_checkpoint_v1_runtime_state(self.runtime, &mut column_replay[..provided])
            .map_err(|_| PrepareCheckpointV1ExportError::State)?;

        let semantic_scratch_required = self
            .runtime
            .store
            .runtime_semantic_posting_scratch_required()
            .map_err(PrepareCheckpointV1ExportError::StoreIndex)?;
        let semantic_scratch_len = usize::try_from(semantic_scratch_required).map_err(|_| {
            PrepareCheckpointV1ExportError::Codec(CheckpointV1CodecError::LengthOverflow)
        })?;

        Ok(CheckpointV1ValidatedExportState {
            preparation: self,
            semantic_scratch_len,
        })
    }
}

impl<'a> CheckpointV1ValidatedExportState<'a> {
    const fn semantic_scratch_len(&self) -> usize {
        self.semantic_scratch_len
    }

    fn finish(
        self,
        semantic_scratch: &mut [u32],
    ) -> Result<PreparedCheckpointV1Export<'a>, PrepareCheckpointV1ExportError> {
        let provided = semantic_scratch.len().min(self.semantic_scratch_len);
        let mut semantic_hasher = Sha256::new();
        self.preparation
            .runtime
            .store
            .visit_runtime_semantic_postings(&mut semantic_scratch[..provided], |posting| {
                semantic_hasher.update(posting.normalized_source.to_le_bytes());
                semantic_hasher.update(posting.segment_id.to_le_bytes());
                semantic_hasher.update(posting.slot.to_le_bytes());
            })
            .map_err(PrepareCheckpointV1ExportError::StoreIndex)?;
        let semantic_index_digest = CheckpointV1Digest(semantic_hasher.finalize().into());

        Ok(PreparedCheckpointV1Export {
            runtime: self.preparation.runtime,
            identity: self.preparation.identity,
            layout: self.preparation.layout,
            scalars: self.preparation.scalars,
            semantic_index_digest,
        })
    }
}

fn checkpoint_v1_zeroed_u32_scratch(
    element_count: usize,
    site: CheckpointV1AllocationSite,
    fail_before_reserve: &mut impl FnMut(CheckpointV1AllocationSite) -> bool,
) -> Result<Vec<u32>, PrepareCheckpointV1ExportError> {
    if element_count == 0 {
        return Ok(Vec::new());
    }
    if fail_before_reserve(site) {
        return Err(PrepareCheckpointV1ExportError::RetryableAllocation(site));
    }
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(element_count)
        .map_err(|_| PrepareCheckpointV1ExportError::RetryableAllocation(site))?;
    scratch.resize(element_count, 0);
    Ok(scratch)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckpointV1ComparePhase {
    ReserveHeader,
    Body,
    BackfillHeader,
    Backfilled,
    VerifyEnd,
    Flushed,
}

struct CheckpointV1CompareSink<'a> {
    expected: &'a [u8],
    position: u64,
    logical_end: u64,
    phase: CheckpointV1ComparePhase,
    first_mismatch: Option<CheckpointV1CanonicalMismatch>,
}

impl<'a> CheckpointV1CompareSink<'a> {
    const fn new(expected: &'a [u8]) -> Self {
        Self {
            expected,
            position: 0,
            logical_end: 0,
            phase: CheckpointV1ComparePhase::ReserveHeader,
            first_mismatch: None,
        }
    }

    fn record_mismatch(&mut self, mismatch: CheckpointV1CanonicalMismatch) {
        if self
            .first_mismatch
            .is_none_or(|current| mismatch.offset() < current.offset())
        {
            self.first_mismatch = Some(mismatch);
        }
    }

    fn compare_at_position(&mut self, bytes: &[u8]) -> io::Result<()> {
        for (index, &actual) in bytes.iter().enumerate() {
            let index =
                u64::try_from(index).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
            let offset = self
                .position
                .checked_add(index)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            let expected = usize::try_from(offset)
                .ok()
                .and_then(|offset| self.expected.get(offset))
                .copied();
            let mismatch = match expected {
                Some(expected) if expected != actual => {
                    Some(CheckpointV1CanonicalMismatch::ByteMismatch {
                        offset,
                        expected,
                        actual,
                    })
                }
                None => Some(CheckpointV1CanonicalMismatch::ExpectedEnded { offset, actual }),
                _ => None,
            };
            if let Some(mismatch) = mismatch {
                self.record_mismatch(mismatch);
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<(), CompareCheckpointV1CanonicalBytesError> {
        if let Some(mismatch) = self.first_mismatch {
            return Err(CompareCheckpointV1CanonicalBytesError::Mismatch(mismatch));
        }
        if self.phase != CheckpointV1ComparePhase::Flushed || self.position != self.logical_end {
            return Err(CompareCheckpointV1CanonicalBytesError::Encode(
                CheckpointV1EncodeError::Flush {
                    source: io::Error::from(io::ErrorKind::InvalidData),
                },
            ));
        }
        let expected_len = u64::try_from(self.expected.len()).map_err(|_| {
            CompareCheckpointV1CanonicalBytesError::Encode(CheckpointV1EncodeError::Flush {
                source: io::Error::from(io::ErrorKind::InvalidInput),
            })
        })?;
        if expected_len > self.logical_end {
            let offset = self.logical_end;
            let expected = usize::try_from(offset)
                .ok()
                .and_then(|offset| self.expected.get(offset))
                .copied()
                .ok_or_else(|| {
                    CompareCheckpointV1CanonicalBytesError::Encode(CheckpointV1EncodeError::Flush {
                        source: io::Error::from(io::ErrorKind::InvalidData),
                    })
                })?;
            return Err(CompareCheckpointV1CanonicalBytesError::Mismatch(
                CheckpointV1CanonicalMismatch::ExpectedTrailing { offset, expected },
            ));
        }
        if expected_len != self.logical_end {
            return Err(CompareCheckpointV1CanonicalBytesError::Encode(
                CheckpointV1EncodeError::Flush {
                    source: io::Error::from(io::ErrorKind::InvalidData),
                },
            ));
        }
        Ok(())
    }
}

impl Write for CheckpointV1CompareSink<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let byte_count =
            u64::try_from(bytes.len()).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        match self.phase {
            CheckpointV1ComparePhase::ReserveHeader => {
                if self.position != 0
                    || self.logical_end != 0
                    || bytes.len() != usize::from(CHECKPOINT_V1_HEADER_BYTES)
                {
                    return Err(io::Error::from(io::ErrorKind::InvalidInput));
                }
                if bytes.iter().any(|&byte| byte != 0) {
                    return Err(io::Error::from(io::ErrorKind::InvalidData));
                }
                self.position = byte_count;
                self.logical_end = byte_count;
                self.phase = CheckpointV1ComparePhase::Body;
                Ok(bytes.len())
            }
            CheckpointV1ComparePhase::Body => {
                if bytes.is_empty() || self.position != self.logical_end {
                    return Err(io::Error::from(io::ErrorKind::InvalidInput));
                }
                self.compare_at_position(bytes)?;
                let end = self
                    .position
                    .checked_add(byte_count)
                    .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
                self.position = end;
                self.logical_end = end;
                Ok(bytes.len())
            }
            CheckpointV1ComparePhase::BackfillHeader => {
                if self.position != 0
                    || self.logical_end < u64::from(CHECKPOINT_V1_HEADER_BYTES)
                    || bytes.len() != usize::from(CHECKPOINT_V1_HEADER_BYTES)
                {
                    return Err(io::Error::from(io::ErrorKind::InvalidInput));
                }
                self.compare_at_position(bytes)?;
                self.position = byte_count;
                self.phase = CheckpointV1ComparePhase::Backfilled;
                Ok(bytes.len())
            }
            CheckpointV1ComparePhase::Backfilled
            | CheckpointV1ComparePhase::VerifyEnd
            | CheckpointV1ComparePhase::Flushed => {
                Err(io::Error::from(io::ErrorKind::InvalidInput))
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.phase != CheckpointV1ComparePhase::VerifyEnd || self.position != self.logical_end {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        self.phase = CheckpointV1ComparePhase::Flushed;
        Ok(())
    }
}

impl Seek for CheckpointV1CompareSink<'_> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        match (self.phase, position) {
            (CheckpointV1ComparePhase::ReserveHeader, SeekFrom::End(0))
                if self.position == 0 && self.logical_end == 0 =>
            {
                Ok(0)
            }
            (CheckpointV1ComparePhase::Body, SeekFrom::Start(0))
                if self.position == self.logical_end
                    && self.logical_end >= u64::from(CHECKPOINT_V1_HEADER_BYTES) =>
            {
                self.position = 0;
                self.phase = CheckpointV1ComparePhase::BackfillHeader;
                Ok(0)
            }
            (CheckpointV1ComparePhase::Backfilled, SeekFrom::End(0))
                if self.position == u64::from(CHECKPOINT_V1_HEADER_BYTES) =>
            {
                self.position = self.logical_end;
                self.phase = CheckpointV1ComparePhase::VerifyEnd;
                Ok(self.logical_end)
            }
            _ => Err(io::Error::from(io::ErrorKind::InvalidInput)),
        }
    }
}

impl PreparedCheckpointV1Export<'_> {
    pub(crate) const fn encoded_len(&self) -> u64 {
        self.layout.total_bytes
    }

    pub(crate) fn identity_header(
        &self,
    ) -> Result<[u8; CHECKPOINT_V1_HEADER_BYTES as usize], CheckpointV1EncodeError> {
        self.validate_encoding_proof()?;
        let body_digest = self.encode_body_into(&mut io::sink())?;
        checkpoint_v1_header_bytes(
            &self.identity,
            self.layout,
            self.scalars,
            body_digest,
            self.semantic_index_digest,
        )
        .map_err(CheckpointV1EncodeError::HeaderCodec)
    }

    fn compare_canonical_bytes(
        &self,
        expected: &[u8],
    ) -> Result<(), CompareCheckpointV1CanonicalBytesError> {
        let mut sink = CheckpointV1CompareSink::new(expected);
        self.encode_into(&mut sink)
            .map_err(CompareCheckpointV1CanonicalBytesError::Encode)?;
        sink.finish()
    }

    pub(crate) fn encode_into<W: Write + Seek>(
        &self,
        sink: &mut W,
    ) -> Result<(), CheckpointV1EncodeError> {
        self.validate_encoding_proof()?;

        let logical_end = checkpoint_v1_seek(
            sink,
            SeekFrom::End(0),
            CheckpointV1SeekStage::InspectLogicalEnd,
        )?;
        if logical_end != 0 {
            return Err(CheckpointV1EncodeError::SinkNotEmpty { logical_end });
        }

        let zero_header = [0u8; CHECKPOINT_V1_HEADER_BYTES as usize];
        checkpoint_v1_write_all(sink, &zero_header, CheckpointV1WriteStage::ReserveHeader)?;
        let body_digest = self.encode_body_into(sink)?;
        let header = checkpoint_v1_header_bytes(
            &self.identity,
            self.layout,
            self.scalars,
            body_digest,
            self.semantic_index_digest,
        )
        .map_err(CheckpointV1EncodeError::HeaderCodec)?;

        let header_position = checkpoint_v1_seek(
            sink,
            SeekFrom::Start(0),
            CheckpointV1SeekStage::HeaderBackfill,
        )?;
        if header_position != 0 {
            return Err(CheckpointV1EncodeError::SeekPosition {
                stage: CheckpointV1SeekStage::HeaderBackfill,
                expected: 0,
                actual: header_position,
            });
        }
        checkpoint_v1_write_all(sink, &header, CheckpointV1WriteStage::BackfillHeader)?;

        let final_position = checkpoint_v1_seek(
            sink,
            SeekFrom::End(0),
            CheckpointV1SeekStage::VerifyLogicalEnd,
        )?;
        if final_position != self.layout.total_bytes {
            return Err(CheckpointV1EncodeError::SeekPosition {
                stage: CheckpointV1SeekStage::VerifyLogicalEnd,
                expected: self.layout.total_bytes,
                actual: final_position,
            });
        }
        checkpoint_v1_flush(sink)
    }

    fn validate_encoding_proof(&self) -> Result<(), CheckpointV1EncodeError> {
        let segment_count = self.runtime.store.runtime_segment_count();
        let cohort_count = u32::try_from(self.runtime.cohort_columns.len()).map_err(|_| {
            CheckpointV1EncodeError::PreparedCodec(CheckpointV1CodecError::LengthOverflow)
        })?;
        let observed_scalars = CheckpointV1Scalars::new(
            segment_count,
            cohort_count,
            self.runtime.next_reserve_cell,
            self.runtime.event_count,
            self.runtime.learning_step,
            self.runtime.previous_context_len,
            self.runtime.previous_winner_context_len,
            self.runtime.scientific_stop,
        )
        .map_err(CheckpointV1EncodeError::PreparedCodec)?;
        if observed_scalars != self.scalars {
            return Err(CheckpointV1EncodeError::PreparedScalarsMismatch);
        }
        let observed_layout = CheckpointV1Layout::checked(
            segment_count,
            cohort_count,
            self.runtime.previous_context_len,
            self.runtime.previous_winner_context_len,
        )
        .map_err(CheckpointV1EncodeError::PreparedCodec)?;
        if self.layout.pair_count != observed_layout.pair_count {
            return Err(CheckpointV1EncodeError::PreparedPairCountMismatch {
                declared: self.layout.pair_count,
                observed: observed_layout.pair_count,
            });
        }
        for (section, declared, observed) in [
            (
                CheckpointV1BodySection::CohortColumns,
                self.layout.cohort_columns_bytes,
                observed_layout.cohort_columns_bytes,
            ),
            (
                CheckpointV1BodySection::PriorContext,
                self.layout.prior_context_bytes,
                observed_layout.prior_context_bytes,
            ),
            (
                CheckpointV1BodySection::WinnerContext,
                self.layout.winner_context_bytes,
                observed_layout.winner_context_bytes,
            ),
            (
                CheckpointV1BodySection::SegmentOwners,
                self.layout.segment_owner_bytes,
                observed_layout.segment_owner_bytes,
            ),
            (
                CheckpointV1BodySection::SegmentSources,
                self.layout.segment_source_bytes,
                observed_layout.segment_source_bytes,
            ),
            (
                CheckpointV1BodySection::SegmentTicks,
                self.layout.segment_tick_bytes,
                observed_layout.segment_tick_bytes,
            ),
        ] {
            if declared != observed {
                return Err(CheckpointV1EncodeError::BodySectionCount {
                    section,
                    declared,
                    observed,
                });
            }
        }
        if self.layout.body_bytes != observed_layout.body_bytes {
            return Err(CheckpointV1EncodeError::BodyByteCount {
                declared: self.layout.body_bytes,
                observed: observed_layout.body_bytes,
            });
        }
        if self.layout.total_bytes != observed_layout.total_bytes {
            return Err(CheckpointV1EncodeError::TotalByteCount {
                declared: self.layout.total_bytes,
                observed: observed_layout.total_bytes,
            });
        }
        Ok(())
    }

    fn encode_body_into<W: Write>(
        &self,
        sink: &mut W,
    ) -> Result<CheckpointV1Digest, CheckpointV1EncodeError> {
        let mut body = CheckpointV1BodyWriter::new(sink);

        for column in self.runtime.cohort_columns.iter().copied() {
            body.push_u32(CheckpointV1BodySection::CohortColumns, column)?;
        }
        body.finish_section(
            CheckpointV1BodySection::CohortColumns,
            self.layout.cohort_columns_bytes,
        )?;

        for source in self
            .runtime
            .previous_context
            .iter()
            .take(usize::from(self.scalars.prior_context_length))
            .copied()
        {
            body.push_u32(CheckpointV1BodySection::PriorContext, source)?;
        }
        body.finish_section(
            CheckpointV1BodySection::PriorContext,
            self.layout.prior_context_bytes,
        )?;

        for source in self
            .runtime
            .previous_winner_context
            .iter()
            .take(usize::from(self.scalars.winner_context_length))
            .copied()
        {
            body.push_u32(CheckpointV1BodySection::WinnerContext, source)?;
        }
        body.finish_section(
            CheckpointV1BodySection::WinnerContext,
            self.layout.winner_context_bytes,
        )?;

        for segment_id in 0..self.scalars.segment_count {
            let owner = self
                .runtime
                .store
                .commit_prevalidated_runtime_segment_owner(segment_id);
            body.push_u32(CheckpointV1BodySection::SegmentOwners, owner)?;
        }
        body.finish_section(
            CheckpointV1BodySection::SegmentOwners,
            self.layout.segment_owner_bytes,
        )?;

        for segment_id in 0..self.scalars.segment_count {
            let row = self
                .runtime
                .store
                .commit_prevalidated_runtime_segment_row(segment_id);
            for source in row.normalized_sources {
                body.push_u32(CheckpointV1BodySection::SegmentSources, source)?;
            }
        }
        body.finish_section(
            CheckpointV1BodySection::SegmentSources,
            self.layout.segment_source_bytes,
        )?;

        for segment_id in 0..self.scalars.segment_count {
            let row = self
                .runtime
                .store
                .commit_prevalidated_runtime_segment_row(segment_id);
            for ticks in row.permanence_ticks {
                body.push_u16(CheckpointV1BodySection::SegmentTicks, ticks)?;
            }
        }
        body.finish_section(
            CheckpointV1BodySection::SegmentTicks,
            self.layout.segment_tick_bytes,
        )?;

        body.finish(self.layout.body_bytes)
    }
}

struct CheckpointV1BodyWriter<'a, W> {
    sink: &'a mut W,
    buffer: [u8; CHECKPOINT_V1_STREAM_BUFFER_BYTES],
    buffered: usize,
    successful_bytes: u64,
    section_start: u64,
    hasher: Sha256,
}

impl<'a, W: Write> CheckpointV1BodyWriter<'a, W> {
    fn new(sink: &'a mut W) -> Self {
        Self {
            sink,
            buffer: [0; CHECKPOINT_V1_STREAM_BUFFER_BYTES],
            buffered: 0,
            successful_bytes: 0,
            section_start: 0,
            hasher: Sha256::new(),
        }
    }

    fn push_u32(
        &mut self,
        section: CheckpointV1BodySection,
        value: u32,
    ) -> Result<(), CheckpointV1EncodeError> {
        self.push_bytes(section, &value.to_le_bytes())
    }

    fn push_u16(
        &mut self,
        section: CheckpointV1BodySection,
        value: u16,
    ) -> Result<(), CheckpointV1EncodeError> {
        self.push_bytes(section, &value.to_le_bytes())
    }

    fn push_bytes(
        &mut self,
        section: CheckpointV1BodySection,
        mut bytes: &[u8],
    ) -> Result<(), CheckpointV1EncodeError> {
        while !bytes.is_empty() {
            if self.buffered == self.buffer.len() {
                self.drain(section)?;
            }
            let available = self
                .buffer
                .len()
                .checked_sub(self.buffered)
                .ok_or(CheckpointV1EncodeError::BodyBufferInvariant(section))?;
            let copied = available.min(bytes.len());
            let end = self
                .buffered
                .checked_add(copied)
                .ok_or(CheckpointV1EncodeError::BodyBufferInvariant(section))?;
            let destination = self
                .buffer
                .get_mut(self.buffered..end)
                .ok_or(CheckpointV1EncodeError::BodyBufferInvariant(section))?;
            destination.copy_from_slice(&bytes[..copied]);
            self.buffered = end;
            bytes = &bytes[copied..];
        }
        Ok(())
    }

    fn finish_section(
        &mut self,
        section: CheckpointV1BodySection,
        declared: u64,
    ) -> Result<(), CheckpointV1EncodeError> {
        self.drain(section)?;
        let observed = self
            .successful_bytes
            .checked_sub(self.section_start)
            .ok_or(CheckpointV1EncodeError::BodyBufferInvariant(section))?;
        if observed != declared {
            return Err(CheckpointV1EncodeError::BodySectionCount {
                section,
                declared,
                observed,
            });
        }
        self.section_start = self.successful_bytes;
        Ok(())
    }

    fn drain(&mut self, section: CheckpointV1BodySection) -> Result<(), CheckpointV1EncodeError> {
        let mut offset = 0usize;
        while offset < self.buffered {
            let bytes = self
                .buffer
                .get(offset..self.buffered)
                .ok_or(CheckpointV1EncodeError::BodyBufferInvariant(section))?;
            let requested = bytes.len();
            match self.sink.write(bytes) {
                Ok(0) => {
                    return Err(CheckpointV1EncodeError::Write {
                        stage: CheckpointV1WriteStage::Body(section),
                        source: io::Error::from(io::ErrorKind::WriteZero),
                    });
                }
                Ok(reported) if reported > requested => {
                    return Err(CheckpointV1EncodeError::WriteCount {
                        stage: CheckpointV1WriteStage::Body(section),
                        requested,
                        reported,
                    });
                }
                Ok(reported) => {
                    let end = offset
                        .checked_add(reported)
                        .ok_or(CheckpointV1EncodeError::BodyBufferInvariant(section))?;
                    let successful = self
                        .buffer
                        .get(offset..end)
                        .ok_or(CheckpointV1EncodeError::BodyBufferInvariant(section))?;
                    let successful_bytes = self
                        .successful_bytes
                        .checked_add(reported as u64)
                        .ok_or(CheckpointV1EncodeError::BodyBufferInvariant(section))?;
                    self.hasher.update(successful);
                    self.successful_bytes = successful_bytes;
                    offset = end;
                }
                Err(source) if source.kind() == io::ErrorKind::Interrupted => {}
                Err(source) => {
                    return Err(CheckpointV1EncodeError::Write {
                        stage: CheckpointV1WriteStage::Body(section),
                        source,
                    });
                }
            }
        }
        self.buffered = 0;
        Ok(())
    }

    fn finish(self, declared: u64) -> Result<CheckpointV1Digest, CheckpointV1EncodeError> {
        if self.buffered != 0 {
            return Err(CheckpointV1EncodeError::BodyBufferInvariant(
                CheckpointV1BodySection::SegmentTicks,
            ));
        }
        if self.successful_bytes != declared {
            return Err(CheckpointV1EncodeError::BodyByteCount {
                declared,
                observed: self.successful_bytes,
            });
        }
        Ok(CheckpointV1Digest(self.hasher.finalize().into()))
    }
}

fn checkpoint_v1_seek<W: Seek>(
    sink: &mut W,
    position: SeekFrom,
    stage: CheckpointV1SeekStage,
) -> Result<u64, CheckpointV1EncodeError> {
    loop {
        match sink.seek(position) {
            Ok(actual) => return Ok(actual),
            Err(source) if source.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => return Err(CheckpointV1EncodeError::Seek { stage, source }),
        }
    }
}

fn checkpoint_v1_write_all<W: Write>(
    sink: &mut W,
    mut bytes: &[u8],
    stage: CheckpointV1WriteStage,
) -> Result<(), CheckpointV1EncodeError> {
    while !bytes.is_empty() {
        let requested = bytes.len();
        match sink.write(bytes) {
            Ok(0) => {
                return Err(CheckpointV1EncodeError::Write {
                    stage,
                    source: io::Error::from(io::ErrorKind::WriteZero),
                });
            }
            Ok(reported) if reported > requested => {
                return Err(CheckpointV1EncodeError::WriteCount {
                    stage,
                    requested,
                    reported,
                });
            }
            Ok(reported) => bytes = &bytes[reported..],
            Err(source) if source.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => return Err(CheckpointV1EncodeError::Write { stage, source }),
        }
    }
    Ok(())
}

fn checkpoint_v1_flush<W: Write>(sink: &mut W) -> Result<(), CheckpointV1EncodeError> {
    loop {
        match sink.flush() {
            Ok(()) => return Ok(()),
            Err(source) if source.kind() == io::ErrorKind::Interrupted => {}
            Err(source) => return Err(CheckpointV1EncodeError::Flush { source }),
        }
    }
}

pub(crate) fn checkpoint_v1_logical_state_digest(
    stored_body_digest: &[u8; 32],
    scalar_block: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(LOGICAL_STATE_DOMAIN);
    hasher.update(stored_body_digest);
    hasher.update(scalar_block);
    hasher.finalize().into()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1ContextLengthField {
    Prior,
    Winner,
}

impl fmt::Display for CheckpointV1ContextLengthField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Prior => "prior context",
            Self::Winner => "winner context",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1IdentityField {
    Protocol,
    SourceBundle,
    Circuit,
}

impl fmt::Display for CheckpointV1IdentityField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Protocol => "protocol",
            Self::SourceBundle => "source-bundle",
            Self::Circuit => "circuit",
        })
    }
}

impl fmt::Display for CheckpointV1BodySection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::CohortColumns => "cohort columns",
            Self::PriorContext => "prior context",
            Self::WinnerContext => "winner context",
            Self::SegmentOwners => "segment owners",
            Self::SegmentSources => "segment sources",
            Self::SegmentTicks => "segment ticks",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1ImportAllocationSite {
    Pass2OwnerLoads,
    Pass2ColumnReplay,
    RuntimeColumnOffsets,
    RuntimeColumnCells,
    RuntimeAlternateColumnOffsets,
    RuntimeAlternateColumnCells,
    RuntimeCohortColumns,
    RuntimeRecognitionDiagnostics,
    RuntimeActiveSegments,
    RuntimeColumnDrive,
    RuntimeNovelCandidates,
    RuntimeRaceVoltage,
    RuntimeRaceThresholds,
    RuntimeRaceExcitatory,
    RuntimeRaceEmitted,
    RuntimeRaceSpikingIndices,
    StoreOwners,
    StoreSources,
    StorePermanences,
    StoreOwnerSegmentCounts,
    StoreSourceMeta,
    StoreNextPair,
    StorePreviousPair,
    StoreSourceStamps,
    StoreSegmentStamps,
    StorePotentialContacts,
    StoreConnectedContacts,
    StoreTouchedSegments,
    StoreOwnerStamps,
    StoreBestMatching,
    StoreBestActive,
    StoreTouchedOwners,
    VerificationSemanticPostings,
}

impl fmt::Display for CheckpointV1ImportAllocationSite {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Pass2OwnerLoads => "Pass 2 owner loads",
            Self::Pass2ColumnReplay => "Pass 2 column replay",
            Self::RuntimeColumnOffsets => "runtime column offsets",
            Self::RuntimeColumnCells => "runtime column cells",
            Self::RuntimeAlternateColumnOffsets => "runtime alternate column offsets",
            Self::RuntimeAlternateColumnCells => "runtime alternate column cells",
            Self::RuntimeCohortColumns => "runtime cohort columns",
            Self::RuntimeRecognitionDiagnostics => "runtime recognition diagnostics",
            Self::RuntimeActiveSegments => "runtime active segments",
            Self::RuntimeColumnDrive => "runtime column drive",
            Self::RuntimeNovelCandidates => "runtime novel candidates",
            Self::RuntimeRaceVoltage => "runtime race voltage",
            Self::RuntimeRaceThresholds => "runtime race thresholds",
            Self::RuntimeRaceExcitatory => "runtime race excitatory current",
            Self::RuntimeRaceEmitted => "runtime race emitted flags",
            Self::RuntimeRaceSpikingIndices => "runtime race spiking indices",
            Self::StoreOwners => "store owners",
            Self::StoreSources => "store sources",
            Self::StorePermanences => "store permanences",
            Self::StoreOwnerSegmentCounts => "store owner segment counts",
            Self::StoreSourceMeta => "store source metadata",
            Self::StoreNextPair => "store next-pair postings",
            Self::StorePreviousPair => "store previous-pair postings",
            Self::StoreSourceStamps => "store source stamps",
            Self::StoreSegmentStamps => "store segment stamps",
            Self::StorePotentialContacts => "store potential contacts",
            Self::StoreConnectedContacts => "store connected contacts",
            Self::StoreTouchedSegments => "store touched segments",
            Self::StoreOwnerStamps => "store owner stamps",
            Self::StoreBestMatching => "store best matching segments",
            Self::StoreBestActive => "store best active segments",
            Self::StoreTouchedOwners => "store touched owners",
            Self::VerificationSemanticPostings => "verification semantic postings",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1Pass1Error {
    HeaderTooShort {
        actual: usize,
        required: usize,
    },
    MagicMismatch,
    UnsupportedFormatVersion {
        found: u16,
    },
    HeaderSizeMismatch {
        found: u16,
    },
    ReservedHeaderByte {
        offset: usize,
        found: u8,
    },
    ContextLength {
        field: CheckpointV1ContextLengthField,
        found: u8,
    },
    PoisonCode {
        found: u16,
    },
    SegmentCountExceedsCapacity {
        declared: u32,
        capacity: u32,
    },
    CohortCountExceedsCapacity {
        declared: u32,
        capacity: u32,
    },
    ReserveCursorOutsideCapacity {
        declared: u32,
        minimum: u32,
        maximum: u32,
    },
    Layout(CheckpointV1CodecError),
    BodyLengthMismatch {
        declared: u64,
        expected: u64,
    },
    TotalLengthMismatch {
        declared: u64,
        expected: u64,
    },
    InputLengthUnrepresentable {
        actual: usize,
    },
    InputLengthMismatch {
        declared: u64,
        actual: u64,
    },
    SectionRangeOverflow {
        section: CheckpointV1BodySection,
    },
    SectionNotAddressable {
        section: CheckpointV1BodySection,
        end: u64,
        platform_limit: u64,
    },
    AllocationCountArithmetic {
        site: CheckpointV1ImportAllocationSite,
    },
    AllocationLayout {
        site: CheckpointV1ImportAllocationSite,
        element_count: usize,
    },
    IdentityMismatch {
        field: CheckpointV1IdentityField,
    },
    BodyDigestMismatch,
    LogicalStateDigestMismatch,
}

impl fmt::Display for CheckpointV1Pass1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HeaderTooShort { actual, required } => write!(
                formatter,
                "checkpoint header is truncated: {actual} bytes, requires {required}"
            ),
            Self::MagicMismatch => formatter.write_str("checkpoint magic does not match v1"),
            Self::UnsupportedFormatVersion { found } => {
                write!(formatter, "unsupported checkpoint format version {found}")
            }
            Self::HeaderSizeMismatch { found } => write!(
                formatter,
                "checkpoint header size {found} does not equal {CHECKPOINT_V1_HEADER_BYTES}"
            ),
            Self::ReservedHeaderByte { offset, found } => write!(
                formatter,
                "reserved checkpoint header byte {offset} is nonzero ({found})"
            ),
            Self::ContextLength { field, found } => {
                write!(
                    formatter,
                    "{field} length {found} is not 0 or {CONTEXT_CELLS}"
                )
            }
            Self::PoisonCode { found } => {
                write!(
                    formatter,
                    "checkpoint poison code {found} is outside 0..=15"
                )
            }
            Self::SegmentCountExceedsCapacity { declared, capacity } => write!(
                formatter,
                "checkpoint segment count {declared} exceeds capacity {capacity}"
            ),
            Self::CohortCountExceedsCapacity { declared, capacity } => write!(
                formatter,
                "checkpoint cohort count {declared} exceeds capacity {capacity}"
            ),
            Self::ReserveCursorOutsideCapacity {
                declared,
                minimum,
                maximum,
            } => write!(
                formatter,
                "checkpoint reserve cursor {declared} is outside {minimum}..={maximum}"
            ),
            Self::Layout(source) => write!(formatter, "checkpoint layout is invalid: {source}"),
            Self::BodyLengthMismatch { declared, expected } => write!(
                formatter,
                "checkpoint body length {declared} does not equal canonical length {expected}"
            ),
            Self::TotalLengthMismatch { declared, expected } => write!(
                formatter,
                "checkpoint total length {declared} does not equal canonical length {expected}"
            ),
            Self::InputLengthUnrepresentable { actual } => write!(
                formatter,
                "checkpoint input length {actual} is outside the u64 format domain"
            ),
            Self::InputLengthMismatch { declared, actual } => write!(
                formatter,
                "checkpoint declares {declared} total bytes but input contains {actual}"
            ),
            Self::SectionRangeOverflow { section } => {
                write!(formatter, "checkpoint {section} byte range overflowed")
            }
            Self::SectionNotAddressable {
                section,
                end,
                platform_limit,
            } => write!(
                formatter,
                "checkpoint {section} end {end} exceeds platform address limit {platform_limit}"
            ),
            Self::AllocationCountArithmetic { site } => {
                write!(
                    formatter,
                    "checkpoint {site} element-count arithmetic overflowed"
                )
            }
            Self::AllocationLayout {
                site,
                element_count,
            } => write!(
                formatter,
                "checkpoint {site} layout cannot address {element_count} elements on this platform"
            ),
            Self::IdentityMismatch { field } => {
                write!(formatter, "checkpoint {field} identity does not match")
            }
            Self::BodyDigestMismatch => {
                formatter.write_str("checkpoint body digest does not match the body")
            }
            Self::LogicalStateDigestMismatch => {
                formatter.write_str("checkpoint logical-state digest does not match")
            }
        }
    }
}

impl Error for CheckpointV1Pass1Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Layout(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct CheckpointV1BodyView<'a> {
    cohort_columns: &'a [u8],
    prior_context: &'a [u8],
    winner_context: &'a [u8],
    segment_owners: &'a [u8],
    segment_sources: &'a [u8],
    segment_ticks: &'a [u8],
}

impl<'a> CheckpointV1BodyView<'a> {
    fn sections(&self) -> [&'a [u8]; 6] {
        [
            self.cohort_columns,
            self.prior_context,
            self.winner_context,
            self.segment_owners,
            self.segment_sources,
            self.segment_ticks,
        ]
    }

    fn cohort_columns(&self) -> CheckpointV1LeU32Slice<'a> {
        CheckpointV1LeU32Slice::from_checked_bytes(self.cohort_columns)
    }

    fn prior_context(&self) -> CheckpointV1LeU32Slice<'a> {
        CheckpointV1LeU32Slice::from_checked_bytes(self.prior_context)
    }

    fn winner_context(&self) -> CheckpointV1LeU32Slice<'a> {
        CheckpointV1LeU32Slice::from_checked_bytes(self.winner_context)
    }

    fn segment_owners(&self) -> CheckpointV1LeU32Slice<'a> {
        CheckpointV1LeU32Slice::from_checked_bytes(self.segment_owners)
    }

    fn segment_sources(&self) -> CheckpointV1LeU32Slice<'a> {
        CheckpointV1LeU32Slice::from_checked_bytes(self.segment_sources)
    }

    fn segment_ticks(&self) -> CheckpointV1LeU16Slice<'a> {
        CheckpointV1LeU16Slice::from_checked_bytes(self.segment_ticks)
    }
}

#[derive(Clone, Copy)]
struct CheckpointV1LeU32Slice<'a> {
    bytes: &'a [u8],
}

impl<'a> CheckpointV1LeU32Slice<'a> {
    /// Wrap a section whose byte length was proved to be an exact multiple of
    /// four by the checked checkpoint-v1 layout.
    fn from_checked_bytes(bytes: &'a [u8]) -> Self {
        debug_assert_eq!(bytes.len() % size_of::<u32>(), 0);
        Self { bytes }
    }

    fn len(self) -> usize {
        self.bytes.len() / size_of::<u32>()
    }

    fn get(self, index: usize) -> Option<u32> {
        let start = index.checked_mul(size_of::<u32>())?;
        let bytes: [u8; 4] = self
            .bytes
            .get(start..start.checked_add(4)?)?
            .try_into()
            .ok()?;
        Some(u32::from_le_bytes(bytes))
    }
}

#[derive(Clone, Copy)]
struct CheckpointV1LeU16Slice<'a> {
    bytes: &'a [u8],
}

impl<'a> CheckpointV1LeU16Slice<'a> {
    /// Wrap a section whose byte length was proved to be an exact multiple of
    /// two by the checked checkpoint-v1 layout.
    fn from_checked_bytes(bytes: &'a [u8]) -> Self {
        debug_assert_eq!(bytes.len() % size_of::<u16>(), 0);
        Self { bytes }
    }

    fn len(self) -> usize {
        self.bytes.len() / size_of::<u16>()
    }

    fn get(self, index: usize) -> Option<u16> {
        let start = index.checked_mul(size_of::<u16>())?;
        let bytes: [u8; 2] = self
            .bytes
            .get(start..start.checked_add(2)?)?
            .try_into()
            .ok()?;
        Some(u16::from_le_bytes(bytes))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CheckpointV1ImportAllocationPlan {
    committed_owner_count: usize,
    column_count: usize,
    owner_capacity: usize,
    segment_capacity: usize,
    posting_capacity: usize,
    source_count: usize,
    cohort_capacity: usize,
    prospective_column_occupancy_limit: usize,
    active_segment_capacity: usize,
    semantic_posting_scratch: usize,
}

struct ValidatedCheckpointV1Pass1<'a> {
    checkpoint: &'a [u8],
    factory_identity: TemporalCheckpointV1Identity,
    body: CheckpointV1BodyView<'a>,
    layout: CheckpointV1Layout,
    scalars: CheckpointV1Scalars,
    allocation_plan: CheckpointV1ImportAllocationPlan,
    semantic_index_digest: CheckpointV1Digest,
    poison: Option<TemporalPoisonCode>,
}

fn validate_checkpoint_v1_pass1<'a>(
    checkpoint: &'a [u8],
    factory: &ValidatedCheckpointV1RuntimeFactory,
) -> Result<ValidatedCheckpointV1Pass1<'a>, CheckpointV1Pass1Error> {
    let definition = factory.definition();
    let header_bytes = usize::from(CHECKPOINT_V1_HEADER_BYTES);
    if checkpoint.len() < header_bytes {
        return Err(CheckpointV1Pass1Error::HeaderTooShort {
            actual: checkpoint.len(),
            required: header_bytes,
        });
    }
    let header = &checkpoint[..header_bytes];
    if header[..8] != CHECKPOINT_V1_MAGIC {
        return Err(CheckpointV1Pass1Error::MagicMismatch);
    }
    let format_version = read_checkpoint_v1_u16(header, 8);
    if format_version != CHECKPOINT_V1_FORMAT_VERSION {
        return Err(CheckpointV1Pass1Error::UnsupportedFormatVersion {
            found: format_version,
        });
    }
    let declared_header_bytes = read_checkpoint_v1_u16(header, 10);
    if declared_header_bytes != CHECKPOINT_V1_HEADER_BYTES {
        return Err(CheckpointV1Pass1Error::HeaderSizeMismatch {
            found: declared_header_bytes,
        });
    }
    if let Some((relative, &found)) = header[12..16]
        .iter()
        .enumerate()
        .find(|(_, byte)| **byte != 0)
    {
        return Err(CheckpointV1Pass1Error::ReservedHeaderByte {
            offset: 12 + relative,
            found,
        });
    }

    let declared_total_bytes = read_checkpoint_v1_u64(header, 16);
    let declared_body_bytes = read_checkpoint_v1_u64(header, 24);
    let segment_count = read_checkpoint_v1_u32(header, 192);
    let cohort_count = read_checkpoint_v1_u32(header, 196);
    let next_reserve_cell = read_checkpoint_v1_u32(header, 200);
    let event_count = read_checkpoint_v1_u64(header, 204);
    let learning_step = read_checkpoint_v1_u64(header, 212);
    let prior_context_length = header[220];
    let winner_context_length = header[221];
    let poison_code = read_checkpoint_v1_u16(header, 222);

    for (field, found) in [
        (CheckpointV1ContextLengthField::Prior, prior_context_length),
        (
            CheckpointV1ContextLengthField::Winner,
            winner_context_length,
        ),
    ] {
        if found != 0 && found != CONTEXT_CELLS_U8 {
            return Err(CheckpointV1Pass1Error::ContextLength { field, found });
        }
    }
    let poison = decode_checkpoint_v1_poison_code(poison_code)?;
    if segment_count > definition.config.segment_capacity {
        return Err(CheckpointV1Pass1Error::SegmentCountExceedsCapacity {
            declared: segment_count,
            capacity: definition.config.segment_capacity,
        });
    }
    let base_population = factory.derived.base_population;
    let cohort_capacity = (definition.config.max_pyramidal_cells - base_population)
        / u32::from(definition.config.novel_race_cap);
    if cohort_count > cohort_capacity {
        return Err(CheckpointV1Pass1Error::CohortCountExceedsCapacity {
            declared: cohort_count,
            capacity: cohort_capacity,
        });
    }
    if !(base_population..=definition.config.max_pyramidal_cells).contains(&next_reserve_cell) {
        return Err(CheckpointV1Pass1Error::ReserveCursorOutsideCapacity {
            declared: next_reserve_cell,
            minimum: base_population,
            maximum: definition.config.max_pyramidal_cells,
        });
    }

    let layout = CheckpointV1Layout::checked(
        segment_count,
        cohort_count,
        prior_context_length,
        winner_context_length,
    )
    .map_err(CheckpointV1Pass1Error::Layout)?;
    if declared_body_bytes != layout.body_bytes {
        return Err(CheckpointV1Pass1Error::BodyLengthMismatch {
            declared: declared_body_bytes,
            expected: layout.body_bytes,
        });
    }
    if declared_total_bytes != layout.total_bytes {
        return Err(CheckpointV1Pass1Error::TotalLengthMismatch {
            declared: declared_total_bytes,
            expected: layout.total_bytes,
        });
    }
    let actual_total_bytes = u64::try_from(checkpoint.len()).map_err(|_| {
        CheckpointV1Pass1Error::InputLengthUnrepresentable {
            actual: checkpoint.len(),
        }
    })?;
    if declared_total_bytes != actual_total_bytes {
        return Err(CheckpointV1Pass1Error::InputLengthMismatch {
            declared: declared_total_bytes,
            actual: actual_total_bytes,
        });
    }

    let platform_limit = usize::MAX as u64;
    let mut section_start = u64::from(CHECKPOINT_V1_HEADER_BYTES);
    let cohort_columns = checked_checkpoint_v1_section_range(
        CheckpointV1BodySection::CohortColumns,
        section_start,
        layout.cohort_columns_bytes,
        platform_limit,
    )?;
    section_start = cohort_columns.1;
    let prior_context = checked_checkpoint_v1_section_range(
        CheckpointV1BodySection::PriorContext,
        section_start,
        layout.prior_context_bytes,
        platform_limit,
    )?;
    section_start = prior_context.1;
    let winner_context = checked_checkpoint_v1_section_range(
        CheckpointV1BodySection::WinnerContext,
        section_start,
        layout.winner_context_bytes,
        platform_limit,
    )?;
    section_start = winner_context.1;
    let segment_owners = checked_checkpoint_v1_section_range(
        CheckpointV1BodySection::SegmentOwners,
        section_start,
        layout.segment_owner_bytes,
        platform_limit,
    )?;
    section_start = segment_owners.1;
    let segment_sources = checked_checkpoint_v1_section_range(
        CheckpointV1BodySection::SegmentSources,
        section_start,
        layout.segment_source_bytes,
        platform_limit,
    )?;
    section_start = segment_sources.1;
    let segment_ticks = checked_checkpoint_v1_section_range(
        CheckpointV1BodySection::SegmentTicks,
        section_start,
        layout.segment_tick_bytes,
        platform_limit,
    )?;
    if segment_ticks.1 != layout.total_bytes {
        return Err(CheckpointV1Pass1Error::TotalLengthMismatch {
            declared: segment_ticks.1,
            expected: layout.total_bytes,
        });
    }
    let allocation_plan =
        checked_checkpoint_v1_import_allocation_plan(factory, segment_count, cohort_count)?;

    validate_checkpoint_v1_identity_field(
        header,
        32,
        factory.identity.authority.protocol_identity.raw_bytes(),
        CheckpointV1IdentityField::Protocol,
    )?;
    validate_checkpoint_v1_identity_field(
        header,
        64,
        factory
            .identity
            .authority
            .source_bundle_identity
            .raw_bytes(),
        CheckpointV1IdentityField::SourceBundle,
    )?;
    validate_checkpoint_v1_identity_field(
        header,
        96,
        factory.identity.circuit_identity.raw_bytes(),
        CheckpointV1IdentityField::Circuit,
    )?;

    let body = &checkpoint[header_bytes..];
    let computed_body_digest: [u8; 32] = Sha256::digest(body).into();
    if header[128..160] != computed_body_digest {
        return Err(CheckpointV1Pass1Error::BodyDigestMismatch);
    }
    let mut stored_body_digest = [0u8; 32];
    stored_body_digest.copy_from_slice(&header[128..160]);
    let mut scalar_block = [0u8; 32];
    scalar_block.copy_from_slice(&header[192..224]);
    let logical_state_digest =
        checkpoint_v1_logical_state_digest(&stored_body_digest, &scalar_block);
    if header[224..256] != logical_state_digest {
        return Err(CheckpointV1Pass1Error::LogicalStateDigestMismatch);
    }

    let body_view = CheckpointV1BodyView {
        cohort_columns: &checkpoint[cohort_columns.0 as usize..cohort_columns.1 as usize],
        prior_context: &checkpoint[prior_context.0 as usize..prior_context.1 as usize],
        winner_context: &checkpoint[winner_context.0 as usize..winner_context.1 as usize],
        segment_owners: &checkpoint[segment_owners.0 as usize..segment_owners.1 as usize],
        segment_sources: &checkpoint[segment_sources.0 as usize..segment_sources.1 as usize],
        segment_ticks: &checkpoint[segment_ticks.0 as usize..segment_ticks.1 as usize],
    };
    let mut semantic_index_digest = [0u8; 32];
    semantic_index_digest.copy_from_slice(&header[160..192]);
    Ok(ValidatedCheckpointV1Pass1 {
        checkpoint,
        factory_identity: factory.identity,
        body: body_view,
        layout,
        scalars: CheckpointV1Scalars {
            segment_count,
            cohort_count,
            next_reserve_cell,
            event_count,
            learning_step,
            prior_context_length,
            winner_context_length,
            poison_code,
        },
        allocation_plan,
        semantic_index_digest: CheckpointV1Digest::from_raw_bytes(semantic_index_digest),
        poison,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckpointV1Pass2Error {
    FactoryIdentityMismatch,
    State(CheckpointV1StateError),
    RetryableAllocation(CheckpointV1ImportAllocationSite),
}

impl fmt::Display for CheckpointV1Pass2Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FactoryIdentityMismatch => {
                formatter.write_str("checkpoint Pass 1 proof belongs to another runtime factory")
            }
            Self::State(source) => write!(formatter, "checkpoint Pass 2 rejected state: {source}"),
            Self::RetryableAllocation(site) => {
                write!(formatter, "checkpoint allocation failed at {site}")
            }
        }
    }
}

impl Error for CheckpointV1Pass2Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::State(source) => Some(source),
            _ => None,
        }
    }
}

struct ValidatedCheckpointV1Pass2<'a> {
    pass1: ValidatedCheckpointV1Pass1<'a>,
    column_occupancies: Vec<u32>,
    max_column_occupancy: usize,
}

fn validate_checkpoint_v1_pass2<'a>(
    pass1: ValidatedCheckpointV1Pass1<'a>,
    factory: &ValidatedCheckpointV1RuntimeFactory,
) -> Result<ValidatedCheckpointV1Pass2<'a>, CheckpointV1Pass2Error> {
    validate_checkpoint_v1_pass2_inner(pass1, factory, |_, _| false)
}

fn validate_checkpoint_v1_pass2_inner<'a>(
    pass1: ValidatedCheckpointV1Pass1<'a>,
    factory: &ValidatedCheckpointV1RuntimeFactory,
    mut fail_before_reserve: impl FnMut(CheckpointV1ImportAllocationSite, usize) -> bool,
) -> Result<ValidatedCheckpointV1Pass2<'a>, CheckpointV1Pass2Error> {
    if pass1.factory_identity != *factory.identity() {
        return Err(CheckpointV1Pass2Error::FactoryIdentityMismatch);
    }

    let definition = factory.definition();
    let logical = CheckpointV1LogicalStateView {
        config: definition.config,
        token_codebook: definition.token_codebook,
        cohort_columns: CheckpointV1U32View::LittleEndian(pass1.body.cohort_columns()),
        next_reserve_cell: pass1.scalars.next_reserve_cell,
        segment_count: pass1.scalars.segment_count,
        previous_context: CheckpointV1U32View::LittleEndian(pass1.body.prior_context()),
        previous_context_len: pass1.scalars.prior_context_length,
        previous_winner_context: CheckpointV1U32View::LittleEndian(pass1.body.winner_context()),
        previous_winner_context_len: pass1.scalars.winner_context_length,
        event_count: pass1.scalars.event_count,
        learning_step: pass1.scalars.learning_step,
        scientific_stop: pass1.poison,
        relay_offset: factory.derived.relay_offset,
        proximal_offset: factory.derived.proximal_offset,
        proximal_relay_offset: factory.derived.proximal_relay_offset,
    };
    let shape =
        validate_checkpoint_v1_logical_shape(&logical).map_err(CheckpointV1Pass2Error::State)?;
    if usize::try_from(shape.committed_owner_count).ok()
        != Some(pass1.allocation_plan.committed_owner_count)
        || shape.column_count != pass1.allocation_plan.column_count
    {
        return Err(CheckpointV1Pass2Error::State(
            CheckpointV1StateError::LayoutInvariant,
        ));
    }

    let mut owner_loads = checkpoint_v1_pass2_zeroed_scratch(
        pass1.allocation_plan.committed_owner_count,
        0u16,
        CheckpointV1ImportAllocationSite::Pass2OwnerLoads,
        &mut fail_before_reserve,
    )?;
    let mut column_occupancies = checkpoint_v1_pass2_zeroed_scratch(
        pass1.allocation_plan.column_count,
        0u32,
        CheckpointV1ImportAllocationSite::Pass2ColumnReplay,
        &mut fail_before_reserve,
    )?;

    let owners = pass1.body.segment_owners();
    let sources = pass1.body.segment_sources();
    let ticks = pass1.body.segment_ticks();
    let max_column_occupancy = {
        let mut replayed_owner_loads = CheckpointV1ReplayedOwnerLoads {
            loads: &mut owner_loads,
        };
        validate_checkpoint_v1_logical_content(
            &logical,
            shape,
            &mut column_occupancies,
            |segment_id| {
                Some(CheckpointV1SoaRow {
                    segment_index: usize::try_from(segment_id).ok()?,
                    owners,
                    sources,
                    ticks,
                })
            },
            &mut replayed_owner_loads,
        )
        .map_err(CheckpointV1Pass2Error::State)?
    };
    drop(owner_loads);

    Ok(ValidatedCheckpointV1Pass2 {
        pass1,
        column_occupancies,
        max_column_occupancy,
    })
}

fn checkpoint_v1_pass2_zeroed_scratch<T: Copy>(
    element_count: usize,
    zero: T,
    site: CheckpointV1ImportAllocationSite,
    fail_before_reserve: &mut impl FnMut(CheckpointV1ImportAllocationSite, usize) -> bool,
) -> Result<Vec<T>, CheckpointV1Pass2Error> {
    if fail_before_reserve(site, element_count) {
        return Err(CheckpointV1Pass2Error::RetryableAllocation(site));
    }
    let mut scratch = Vec::new();
    scratch
        .try_reserve_exact(element_count)
        .map_err(|_| CheckpointV1Pass2Error::RetryableAllocation(site))?;
    scratch.resize(element_count, zero);
    Ok(scratch)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1ImportError {
    Pass1(CheckpointV1Pass1Error),
    FactoryIdentityMismatch,
    State,
    RetryableAllocation(CheckpointV1ImportAllocationSite),
    SemanticIndexMismatch,
    CanonicalReexportMismatch { offset: u64 },
    AssemblyInvariant,
}

fn canonical_empty_checkpoint_v1(
    factory: &ValidatedCheckpointV1RuntimeFactory,
) -> Result<[u8; CHECKPOINT_V1_HEADER_BYTES as usize], CheckpointV1CodecError> {
    let layout = CheckpointV1Layout::checked(0, 0, 0, 0)?;
    let scalars =
        CheckpointV1Scalars::new(0, 0, factory.derived.base_population, 0, 0, 0, 0, None)?;
    let empty_digest = CheckpointV1Digest(Sha256::digest([]).into());
    checkpoint_v1_header_bytes(
        factory.identity(),
        layout,
        scalars,
        empty_digest,
        empty_digest,
    )
}

pub(crate) fn import_canonical_empty_checkpoint_v1(
    factory: ValidatedCheckpointV1RuntimeFactory,
) -> Result<TemporalCircuitRuntime, CheckpointV1ImportError> {
    let checkpoint = canonical_empty_checkpoint_v1(&factory)
        .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
    import_checkpoint_v1(&checkpoint, factory)
}

impl fmt::Display for CheckpointV1ImportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass1(source) => write!(formatter, "checkpoint Pass 1 failed: {source}"),
            Self::FactoryIdentityMismatch => {
                formatter.write_str("checkpoint proof belongs to another runtime factory")
            }
            Self::State => formatter.write_str("checkpoint state is not runtime-reachable"),
            Self::RetryableAllocation(site) => {
                write!(formatter, "checkpoint allocation failed at {site}")
            }
            Self::SemanticIndexMismatch => {
                formatter.write_str("rebuilt semantic index digest does not match checkpoint")
            }
            Self::CanonicalReexportMismatch { offset } => write!(
                formatter,
                "rebuilt checkpoint differs from canonical input at byte {offset}"
            ),
            Self::AssemblyInvariant => {
                formatter.write_str("validated checkpoint could not be assembled")
            }
        }
    }
}

impl Error for CheckpointV1ImportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Pass1(source) => Some(source),
            _ => None,
        }
    }
}

pub(crate) fn import_checkpoint_v1(
    checkpoint: &[u8],
    factory: ValidatedCheckpointV1RuntimeFactory,
) -> Result<TemporalCircuitRuntime, CheckpointV1ImportError> {
    let pass1 = validate_checkpoint_v1_pass1(checkpoint, &factory)
        .map_err(CheckpointV1ImportError::Pass1)?;
    let pass2 = validate_checkpoint_v1_pass2(pass1, &factory).map_err(|error| match error {
        CheckpointV1Pass2Error::FactoryIdentityMismatch => {
            CheckpointV1ImportError::FactoryIdentityMismatch
        }
        CheckpointV1Pass2Error::State(_) => CheckpointV1ImportError::State,
        CheckpointV1Pass2Error::RetryableAllocation(site) => {
            CheckpointV1ImportError::RetryableAllocation(site)
        }
    })?;
    assemble_validated_checkpoint_v1_inner(pass2, factory, &mut |_, _| false)
}

fn assemble_validated_checkpoint_v1_inner<'a>(
    pass2: ValidatedCheckpointV1Pass2<'a>,
    factory: ValidatedCheckpointV1RuntimeFactory,
    fail_before_reserve: &mut impl FnMut(CheckpointV1ImportAllocationSite, usize) -> bool,
) -> Result<TemporalCircuitRuntime, CheckpointV1ImportError> {
    if pass2.pass1.factory_identity != factory.identity {
        return Err(CheckpointV1ImportError::FactoryIdentityMismatch);
    }

    let plan = pass2.pass1.allocation_plan;
    let scalars = pass2.pass1.scalars;
    let body = pass2.pass1.body;
    let checkpoint = pass2.pass1.checkpoint;
    let layout = pass2.pass1.layout;
    let retained_semantic_digest = pass2.pass1.semantic_index_digest;
    let poison = pass2.pass1.poison;
    let config = factory.config;
    let derived = factory.derived;

    let column_offset_count = plan
        .column_count
        .checked_add(1)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    let race_cap = usize::from(config.novel_race_cap);
    if race_cap == 0 {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }
    let base_width = usize::from(config.base_cells_per_column);
    let base_population = usize::try_from(derived.base_population)
        .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
    let cohort_count = usize::try_from(scalars.cohort_count)
        .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
    let segment_count = usize::try_from(scalars.segment_count)
        .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
    let committed_owner_count = usize::try_from(scalars.next_reserve_cell)
        .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
    let expected_base_population = plan
        .column_count
        .checked_mul(base_width)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    let committed_reserve_count = cohort_count
        .checked_mul(race_cap)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    let expected_committed_count = base_population
        .checked_add(committed_reserve_count)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    let expected_posting_capacity = plan
        .segment_capacity
        .checked_mul(PAIRS_PER_SEGMENT)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    let live_pair_slots = segment_count
        .checked_mul(PAIRS_PER_SEGMENT)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    if base_population != expected_base_population
        || committed_owner_count != plan.committed_owner_count
        || expected_committed_count != committed_owner_count
        || committed_owner_count > plan.owner_capacity
        || cohort_count > plan.cohort_capacity
        || segment_count > plan.segment_capacity
        || expected_posting_capacity != plan.posting_capacity
        || plan.semantic_posting_scratch != segment_count
        || usize::try_from(derived.source_count).ok() != Some(plan.source_count)
        || body.cohort_columns().len() != cohort_count
        || body.segment_owners().len() != segment_count
        || body.segment_sources().len() != live_pair_slots
        || body.segment_ticks().len() != live_pair_slots
    {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }
    let expected_active_capacity = match config.plasticity_policy {
        TemporalPlasticityPolicy::GrowthOff => 0,
        TemporalPlasticityPolicy::Plastic => plan.segment_capacity,
    };
    if plan.active_segment_capacity != expected_active_capacity {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }

    if pass2.column_occupancies.len() != plan.column_count {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }
    let mut replayed_committed = 0usize;
    let mut replayed_maximum = 0usize;
    for occupancy in pass2.column_occupancies.iter().copied() {
        let occupancy =
            usize::try_from(occupancy).map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
        let recruited = occupancy
            .checked_sub(base_width)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        if recruited % race_cap != 0 {
            return Err(CheckpointV1ImportError::AssemblyInvariant);
        }
        replayed_committed = replayed_committed
            .checked_add(occupancy)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        replayed_maximum = replayed_maximum.max(occupancy);
    }
    if replayed_committed != committed_owner_count || replayed_maximum != pass2.max_column_occupancy
    {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }
    for index in 0..cohort_count {
        let column = body
            .cohort_columns()
            .get(index)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        if usize::try_from(column)
            .ok()
            .is_none_or(|column| column >= plan.column_count)
        {
            return Err(CheckpointV1ImportError::AssemblyInvariant);
        }
    }

    let remaining_owner_count = plan
        .owner_capacity
        .checked_sub(committed_owner_count)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    let remaining_full_cohort_cells = (remaining_owner_count / race_cap)
        .checked_mul(race_cap)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    let race_capacity = pass2
        .max_column_occupancy
        .checked_add(remaining_full_cohort_cells)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    if race_capacity == 0 || race_capacity > plan.prospective_column_occupancy_limit {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }

    let previous_context =
        checkpoint_v1_import_context(body.prior_context(), scalars.prior_context_length)?;
    let previous_winner_context =
        checkpoint_v1_import_context(body.winner_context(), scalars.winner_context_length)?;

    let mut column_offsets = checkpoint_v1_import_filled_vec(
        column_offset_count,
        column_offset_count,
        0usize,
        CheckpointV1ImportAllocationSite::RuntimeColumnOffsets,
        fail_before_reserve,
    )?;
    let mut column_cells = checkpoint_v1_import_filled_vec(
        plan.owner_capacity,
        committed_owner_count,
        NONE,
        CheckpointV1ImportAllocationSite::RuntimeColumnCells,
        fail_before_reserve,
    )?;
    let alternate_column_offsets = checkpoint_v1_import_filled_vec(
        column_offset_count,
        0,
        0usize,
        CheckpointV1ImportAllocationSite::RuntimeAlternateColumnOffsets,
        fail_before_reserve,
    )?;
    let alternate_column_cells = checkpoint_v1_import_filled_vec(
        plan.owner_capacity,
        0,
        0u32,
        CheckpointV1ImportAllocationSite::RuntimeAlternateColumnCells,
        fail_before_reserve,
    )?;
    let mut cohort_columns = checkpoint_v1_import_filled_vec(
        plan.cohort_capacity,
        cohort_count,
        0u32,
        CheckpointV1ImportAllocationSite::RuntimeCohortColumns,
        fail_before_reserve,
    )?;
    let recognition_scratch = checkpoint_v1_import_filled_vec(
        plan.owner_capacity,
        plan.owner_capacity,
        EMPTY_DIAGNOSTIC,
        CheckpointV1ImportAllocationSite::RuntimeRecognitionDiagnostics,
        fail_before_reserve,
    )?;
    let active_segment_scratch = checkpoint_v1_import_filled_vec(
        plan.active_segment_capacity,
        plan.active_segment_capacity,
        NONE,
        CheckpointV1ImportAllocationSite::RuntimeActiveSegments,
        fail_before_reserve,
    )?;
    let column_drive_scratch = checkpoint_v1_import_filled_vec(
        race_capacity,
        race_capacity,
        EMPTY_DRIVE,
        CheckpointV1ImportAllocationSite::RuntimeColumnDrive,
        fail_before_reserve,
    )?;
    let novel_candidate_scratch: Vec<NovelCandidate> = checkpoint_v1_import_filled_vec(
        race_capacity,
        race_capacity,
        EMPTY_NOVEL_CANDIDATE,
        CheckpointV1ImportAllocationSite::RuntimeNovelCandidates,
        fail_before_reserve,
    )?;

    let mut column_occupancies = pass2.column_occupancies;
    checkpoint_v1_import_populate_csr(
        &mut column_offsets,
        &mut column_cells,
        &mut cohort_columns,
        &mut column_occupancies,
        body.cohort_columns(),
        base_population,
        base_width,
        race_cap,
    )?;

    let race_scratch =
        CoupledRaceScratch::try_new_with_reserve(race_capacity, |site, element_count| {
            fail_before_reserve(checkpoint_v1_import_race_site(site), element_count)
        })
        .map_err(checkpoint_v1_import_race_error)?;

    let store_config = TemporalStoreConfig {
        source_count: plan.source_count,
        owner_count: plan.owner_capacity,
        segment_capacity: plan.segment_capacity,
        source_gid_base: derived.source_gid_base,
        connected_permanence_ticks: config.connected_permanence_ticks,
        theta_match: config.theta_match,
        theta_active: config.theta_active,
        padding_zero: 0,
    };
    let mut rebuild = TemporalStoreCheckpointRebuild::new_with_reserve(
        store_config,
        scalars.segment_count,
        |site, element_count| {
            fail_before_reserve(checkpoint_v1_import_store_site(site), element_count)
        },
    )
    .map_err(checkpoint_v1_import_store_error)?;
    let owners = body.segment_owners();
    let sources = body.segment_sources();
    let ticks = body.segment_ticks();
    for segment_index in 0..segment_count {
        let owner = owners
            .get(segment_index)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let first_pair = segment_index
            .checked_mul(PAIRS_PER_SEGMENT)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let mut row = TemporalSegmentRow {
            normalized_sources: [NONE; PAIRS_PER_SEGMENT],
            permanence_ticks: [0; PAIRS_PER_SEGMENT],
            live_pair_count: 0,
        };
        for slot in 0..PAIRS_PER_SEGMENT {
            let pair_index = first_pair
                .checked_add(slot)
                .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
            *row.normalized_sources
                .get_mut(slot)
                .ok_or(CheckpointV1ImportError::AssemblyInvariant)? = sources
                .get(pair_index)
                .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
            *row.permanence_ticks
                .get_mut(slot)
                .ok_or(CheckpointV1ImportError::AssemblyInvariant)? = ticks
                .get(pair_index)
                .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        }
        let live_pair_count = row
            .normalized_sources
            .iter()
            .position(|&source| source == NONE)
            .unwrap_or(PAIRS_PER_SEGMENT);
        row.live_pair_count = u8::try_from(live_pair_count)
            .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
        let appended = rebuild
            .append_canonical_row(owner, &row)
            .map_err(checkpoint_v1_import_store_error)?;
        if usize::try_from(appended).ok() != Some(segment_index) {
            return Err(CheckpointV1ImportError::AssemblyInvariant);
        }
    }
    let store = rebuild.finish().map_err(checkpoint_v1_import_store_error)?;

    let ValidatedCheckpointV1RuntimeFactory {
        config,
        token_codebook,
        threshold_offsets_mv,
        identity,
        derived,
    } = factory;
    let role_place_copy = RolePlaceCopyState::new(
        config.construction_seed,
        config.column_count,
        derived.physical_cell_count,
        &token_codebook,
    )
    .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
    let runtime = TemporalCircuitRuntime {
        config,
        token_codebook,
        column_offsets,
        column_cells,
        alternate_column_offsets,
        alternate_column_cells,
        cohort_columns,
        next_reserve_cell: scalars.next_reserve_cell,
        threshold_offsets_mv,
        store,
        recognition_scratch,
        active_segment_scratch,
        column_drive_scratch,
        novel_candidate_scratch,
        race_scratch,
        previous_context,
        previous_context_len: scalars.prior_context_length,
        previous_winner_context,
        previous_winner_context_len: scalars.winner_context_length,
        event_count: scalars.event_count,
        learning_step: scalars.learning_step,
        scientific_stop: poison,
        role_place_copy,
        generation_index: 0,
        relay_offset: derived.relay_offset,
        basket_offset: derived.basket_offset,
        proximal_offset: derived.proximal_offset,
        proximal_relay_offset: derived.proximal_relay_offset,
    };

    let preparation = CheckpointV1ExportPreparation {
        runtime: &runtime,
        identity,
        layout,
        scalars,
        column_scratch_len: plan.column_count,
    };
    let validated = preparation
        .validate_state(&mut column_occupancies)
        .map_err(checkpoint_v1_import_preparation_error)?;
    if validated.semantic_scratch_len() > plan.semantic_posting_scratch {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }
    let mut semantic_scratch = checkpoint_v1_import_filled_vec(
        plan.semantic_posting_scratch,
        plan.semantic_posting_scratch,
        0u32,
        CheckpointV1ImportAllocationSite::VerificationSemanticPostings,
        fail_before_reserve,
    )?;
    let prepared = validated
        .finish(&mut semantic_scratch)
        .map_err(checkpoint_v1_import_preparation_error)?;
    if prepared.semantic_index_digest != retained_semantic_digest {
        return Err(CheckpointV1ImportError::SemanticIndexMismatch);
    }
    prepared
        .compare_canonical_bytes(checkpoint)
        .map_err(|error| match error {
            CompareCheckpointV1CanonicalBytesError::Mismatch(mismatch) => {
                CheckpointV1ImportError::CanonicalReexportMismatch {
                    offset: mismatch.offset(),
                }
            }
            CompareCheckpointV1CanonicalBytesError::Encode(_) => {
                CheckpointV1ImportError::AssemblyInvariant
            }
        })?;

    Ok(runtime)
}

fn checkpoint_v1_import_context(
    encoded: CheckpointV1LeU32Slice<'_>,
    encoded_length: u8,
) -> Result<[u32; CONTEXT_CELLS], CheckpointV1ImportError> {
    let expected_length = match encoded_length {
        0 => 0,
        CONTEXT_CELLS_U8 => CONTEXT_CELLS,
        _ => return Err(CheckpointV1ImportError::AssemblyInvariant),
    };
    if encoded.len() != expected_length {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }
    let mut context = [0; CONTEXT_CELLS];
    for index in 0..expected_length {
        *context
            .get_mut(index)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)? = encoded
            .get(index)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    }
    Ok(context)
}

fn checkpoint_v1_import_filled_vec<T: Copy>(
    capacity: usize,
    length: usize,
    value: T,
    site: CheckpointV1ImportAllocationSite,
    fail_before_reserve: &mut impl FnMut(CheckpointV1ImportAllocationSite, usize) -> bool,
) -> Result<Vec<T>, CheckpointV1ImportError> {
    if length > capacity {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }
    if capacity == 0 {
        return Ok(Vec::new());
    }
    if fail_before_reserve(site, capacity) {
        return Err(CheckpointV1ImportError::RetryableAllocation(site));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| CheckpointV1ImportError::RetryableAllocation(site))?;
    values.resize(length, value);
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn checkpoint_v1_import_populate_csr(
    column_offsets: &mut [usize],
    column_cells: &mut [u32],
    cohort_columns: &mut [u32],
    column_cursors: &mut [u32],
    encoded_cohort_columns: CheckpointV1LeU32Slice<'_>,
    base_population: usize,
    base_width: usize,
    race_cap: usize,
) -> Result<(), CheckpointV1ImportError> {
    let expected_offset_count = column_cursors
        .len()
        .checked_add(1)
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
    if column_offsets.len() != expected_offset_count
        || cohort_columns.len() != encoded_cohort_columns.len()
    {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }
    let mut next_offset = 0usize;
    *column_offsets
        .first_mut()
        .ok_or(CheckpointV1ImportError::AssemblyInvariant)? = 0;
    for (column, occupancy) in column_cursors.iter().copied().enumerate() {
        let occupancy =
            usize::try_from(occupancy).map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
        next_offset = next_offset
            .checked_add(occupancy)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        *column_offsets
            .get_mut(column + 1)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)? = next_offset;
    }
    if next_offset != column_cells.len() {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }

    for column in 0..column_cursors.len() {
        let column_start = *column_offsets
            .get(column)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let base_start = column
            .checked_mul(base_width)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let base_end = base_start
            .checked_add(base_width)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let destination_end = column_start
            .checked_add(base_width)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        if base_end > base_population {
            return Err(CheckpointV1ImportError::AssemblyInvariant);
        }
        let destination = column_cells
            .get_mut(column_start..destination_end)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        for (offset, cell) in destination.iter_mut().enumerate() {
            *cell = u32::try_from(
                base_start
                    .checked_add(offset)
                    .ok_or(CheckpointV1ImportError::AssemblyInvariant)?,
            )
            .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
        }
    }
    let base_cursor =
        u32::try_from(base_width).map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
    column_cursors.fill(base_cursor);

    for cohort_index in 0..cohort_columns.len() {
        let column = encoded_cohort_columns
            .get(cohort_index)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        *cohort_columns
            .get_mut(cohort_index)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)? = column;
        let column =
            usize::try_from(column).map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
        let local_start = usize::try_from(
            *column_cursors
                .get(column)
                .ok_or(CheckpointV1ImportError::AssemblyInvariant)?,
        )
        .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
        let local_end = local_start
            .checked_add(race_cap)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let column_start = *column_offsets
            .get(column)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let destination_start = column_start
            .checked_add(local_start)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let destination_end = column_start
            .checked_add(local_end)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let first_gid = base_population
            .checked_add(
                cohort_index
                    .checked_mul(race_cap)
                    .ok_or(CheckpointV1ImportError::AssemblyInvariant)?,
            )
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let destination = column_cells
            .get_mut(destination_start..destination_end)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        for (offset, cell) in destination.iter_mut().enumerate() {
            *cell = u32::try_from(
                first_gid
                    .checked_add(offset)
                    .ok_or(CheckpointV1ImportError::AssemblyInvariant)?,
            )
            .map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
        }
        *column_cursors
            .get_mut(column)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)? =
            u32::try_from(local_end).map_err(|_| CheckpointV1ImportError::AssemblyInvariant)?;
    }

    for column in 0..column_cursors.len() {
        let start = *column_offsets
            .get(column)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let end = *column_offsets
            .get(column + 1)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        let expected = end
            .checked_sub(start)
            .ok_or(CheckpointV1ImportError::AssemblyInvariant)?;
        if usize::try_from(
            *column_cursors
                .get(column)
                .ok_or(CheckpointV1ImportError::AssemblyInvariant)?,
        )
        .ok()
            != Some(expected)
        {
            return Err(CheckpointV1ImportError::AssemblyInvariant);
        }
    }
    if column_cells.contains(&NONE) {
        return Err(CheckpointV1ImportError::AssemblyInvariant);
    }
    Ok(())
}

const fn checkpoint_v1_import_race_site(
    site: CoupledRaceScratchReserveSite,
) -> CheckpointV1ImportAllocationSite {
    match site {
        CoupledRaceScratchReserveSite::Voltage => {
            CheckpointV1ImportAllocationSite::RuntimeRaceVoltage
        }
        CoupledRaceScratchReserveSite::Thresholds => {
            CheckpointV1ImportAllocationSite::RuntimeRaceThresholds
        }
        CoupledRaceScratchReserveSite::Excitatory => {
            CheckpointV1ImportAllocationSite::RuntimeRaceExcitatory
        }
        CoupledRaceScratchReserveSite::Emitted => {
            CheckpointV1ImportAllocationSite::RuntimeRaceEmitted
        }
        CoupledRaceScratchReserveSite::SpikingIndices => {
            CheckpointV1ImportAllocationSite::RuntimeRaceSpikingIndices
        }
    }
}

fn checkpoint_v1_import_race_error(error: CoupledRaceScratchBuildError) -> CheckpointV1ImportError {
    match error {
        CoupledRaceScratchBuildError::RetryableAllocation(site) => {
            CheckpointV1ImportError::RetryableAllocation(checkpoint_v1_import_race_site(site))
        }
        CoupledRaceScratchBuildError::ZeroCapacity
        | CoupledRaceScratchBuildError::CapacityOverflow => {
            CheckpointV1ImportError::AssemblyInvariant
        }
    }
}

const fn checkpoint_v1_import_store_site(
    site: TemporalStoreAllocationSite,
) -> CheckpointV1ImportAllocationSite {
    match site {
        TemporalStoreAllocationSite::Owners => CheckpointV1ImportAllocationSite::StoreOwners,
        TemporalStoreAllocationSite::Sources => CheckpointV1ImportAllocationSite::StoreSources,
        TemporalStoreAllocationSite::Permanences => {
            CheckpointV1ImportAllocationSite::StorePermanences
        }
        TemporalStoreAllocationSite::OwnerSegmentCounts => {
            CheckpointV1ImportAllocationSite::StoreOwnerSegmentCounts
        }
        TemporalStoreAllocationSite::SourceMeta => {
            CheckpointV1ImportAllocationSite::StoreSourceMeta
        }
        TemporalStoreAllocationSite::NextPair => CheckpointV1ImportAllocationSite::StoreNextPair,
        TemporalStoreAllocationSite::PreviousPair => {
            CheckpointV1ImportAllocationSite::StorePreviousPair
        }
        TemporalStoreAllocationSite::SourceStamps => {
            CheckpointV1ImportAllocationSite::StoreSourceStamps
        }
        TemporalStoreAllocationSite::SegmentStamps => {
            CheckpointV1ImportAllocationSite::StoreSegmentStamps
        }
        TemporalStoreAllocationSite::PotentialContacts => {
            CheckpointV1ImportAllocationSite::StorePotentialContacts
        }
        TemporalStoreAllocationSite::ConnectedContacts => {
            CheckpointV1ImportAllocationSite::StoreConnectedContacts
        }
        TemporalStoreAllocationSite::TouchedSegments => {
            CheckpointV1ImportAllocationSite::StoreTouchedSegments
        }
        TemporalStoreAllocationSite::OwnerStamps => {
            CheckpointV1ImportAllocationSite::StoreOwnerStamps
        }
        TemporalStoreAllocationSite::BestMatching => {
            CheckpointV1ImportAllocationSite::StoreBestMatching
        }
        TemporalStoreAllocationSite::BestActive => {
            CheckpointV1ImportAllocationSite::StoreBestActive
        }
        TemporalStoreAllocationSite::TouchedOwners => {
            CheckpointV1ImportAllocationSite::StoreTouchedOwners
        }
    }
}

fn checkpoint_v1_import_store_error(
    error: TemporalCheckpointRebuildError,
) -> CheckpointV1ImportError {
    match error {
        TemporalCheckpointRebuildError::RetryableAllocation(site) => {
            CheckpointV1ImportError::RetryableAllocation(checkpoint_v1_import_store_site(site))
        }
        TemporalCheckpointRebuildError::StoreConstruction(_)
        | TemporalCheckpointRebuildError::DeclaredSegmentCountExceedsCapacity { .. }
        | TemporalCheckpointRebuildError::DeclaredSegmentCountExceeded { .. }
        | TemporalCheckpointRebuildError::SegmentCountMismatch { .. }
        | TemporalCheckpointRebuildError::Store(_) => CheckpointV1ImportError::AssemblyInvariant,
    }
}

fn checkpoint_v1_import_preparation_error(
    error: PrepareCheckpointV1ExportError,
) -> CheckpointV1ImportError {
    match error {
        PrepareCheckpointV1ExportError::State => CheckpointV1ImportError::State,
        PrepareCheckpointV1ExportError::Identity(_)
        | PrepareCheckpointV1ExportError::Codec(_)
        | PrepareCheckpointV1ExportError::StoreIndex(_)
        | PrepareCheckpointV1ExportError::RetryableAllocation(_) => {
            CheckpointV1ImportError::AssemblyInvariant
        }
    }
}

fn checked_checkpoint_v1_import_allocation_plan(
    factory: &ValidatedCheckpointV1RuntimeFactory,
    segment_count: u32,
    cohort_count: u32,
) -> Result<CheckpointV1ImportAllocationPlan, CheckpointV1Pass1Error> {
    let definition = factory.definition();
    let column_count = checkpoint_v1_allocation_usize(
        CheckpointV1ImportAllocationSite::RuntimeColumnOffsets,
        definition.config.column_count,
    )?;
    let owner_capacity = checkpoint_v1_allocation_usize(
        CheckpointV1ImportAllocationSite::RuntimeColumnCells,
        definition.config.max_pyramidal_cells,
    )?;
    let segment_capacity = checkpoint_v1_allocation_usize(
        CheckpointV1ImportAllocationSite::StoreOwners,
        definition.config.segment_capacity,
    )?;
    let source_count = checkpoint_v1_allocation_usize(
        CheckpointV1ImportAllocationSite::StoreSourceMeta,
        factory.derived.source_count,
    )?;
    let base_population = checkpoint_v1_allocation_usize(
        CheckpointV1ImportAllocationSite::Pass2OwnerLoads,
        factory.derived.base_population,
    )?;
    let cohort_count = checkpoint_v1_allocation_usize(
        CheckpointV1ImportAllocationSite::Pass2OwnerLoads,
        cohort_count,
    )?;
    let segment_count = checkpoint_v1_allocation_usize(
        CheckpointV1ImportAllocationSite::VerificationSemanticPostings,
        segment_count,
    )?;
    let race_cap = usize::from(definition.config.novel_race_cap);
    let base_width = usize::from(definition.config.base_cells_per_column);
    let cohort_capacity = owner_capacity.checked_sub(base_population).ok_or(
        CheckpointV1Pass1Error::AllocationCountArithmetic {
            site: CheckpointV1ImportAllocationSite::RuntimeCohortColumns,
        },
    )? / race_cap;
    let recruited_count = cohort_count.checked_mul(race_cap).ok_or(
        CheckpointV1Pass1Error::AllocationCountArithmetic {
            site: CheckpointV1ImportAllocationSite::Pass2OwnerLoads,
        },
    )?;
    let committed_owner_count = base_population.checked_add(recruited_count).ok_or(
        CheckpointV1Pass1Error::AllocationCountArithmetic {
            site: CheckpointV1ImportAllocationSite::Pass2OwnerLoads,
        },
    )?;
    if committed_owner_count > owner_capacity {
        return Err(CheckpointV1Pass1Error::AllocationCountArithmetic {
            site: CheckpointV1ImportAllocationSite::Pass2OwnerLoads,
        });
    }
    let posting_capacity = segment_capacity.checked_mul(PAIRS_PER_SEGMENT).ok_or(
        CheckpointV1Pass1Error::AllocationCountArithmetic {
            site: CheckpointV1ImportAllocationSite::StoreSources,
        },
    )?;
    let prospective_column_occupancy_limit =
        checked_checkpoint_v1_prospective_occupancy_limit(base_width, race_cap, cohort_capacity)
            .ok_or(CheckpointV1Pass1Error::AllocationCountArithmetic {
                site: CheckpointV1ImportAllocationSite::RuntimeColumnDrive,
            })?;
    let column_offset_count =
        column_count
            .checked_add(1)
            .ok_or(CheckpointV1Pass1Error::AllocationCountArithmetic {
                site: CheckpointV1ImportAllocationSite::RuntimeColumnOffsets,
            })?;
    let active_segment_capacity = match definition.config.plasticity_policy {
        TemporalPlasticityPolicy::GrowthOff => 0,
        TemporalPlasticityPolicy::Plastic => segment_capacity,
    };

    checkpoint_v1_preflight_layout::<u16>(
        CheckpointV1ImportAllocationSite::Pass2OwnerLoads,
        committed_owner_count,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::Pass2ColumnReplay,
        column_count,
    )?;
    checkpoint_v1_preflight_layout::<usize>(
        CheckpointV1ImportAllocationSite::RuntimeColumnOffsets,
        column_offset_count,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::RuntimeColumnCells,
        owner_capacity,
    )?;
    checkpoint_v1_preflight_layout::<usize>(
        CheckpointV1ImportAllocationSite::RuntimeAlternateColumnOffsets,
        column_offset_count,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::RuntimeAlternateColumnCells,
        owner_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::RuntimeCohortColumns,
        cohort_capacity,
    )?;
    checkpoint_v1_preflight_layout::<TemporalOwnerDiagnostic>(
        CheckpointV1ImportAllocationSite::RuntimeRecognitionDiagnostics,
        owner_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::RuntimeActiveSegments,
        active_segment_capacity,
    )?;
    checkpoint_v1_preflight_layout::<CandidateDrive>(
        CheckpointV1ImportAllocationSite::RuntimeColumnDrive,
        prospective_column_occupancy_limit,
    )?;
    checkpoint_v1_preflight_layout::<super::NovelCandidate>(
        CheckpointV1ImportAllocationSite::RuntimeNovelCandidates,
        prospective_column_occupancy_limit,
    )?;
    for site in [
        CheckpointV1ImportAllocationSite::RuntimeRaceVoltage,
        CheckpointV1ImportAllocationSite::RuntimeRaceThresholds,
        CheckpointV1ImportAllocationSite::RuntimeRaceExcitatory,
    ] {
        checkpoint_v1_preflight_layout::<f64>(site, prospective_column_occupancy_limit)?;
    }
    checkpoint_v1_preflight_layout::<u8>(
        CheckpointV1ImportAllocationSite::RuntimeRaceEmitted,
        prospective_column_occupancy_limit,
    )?;
    checkpoint_v1_preflight_layout::<usize>(
        CheckpointV1ImportAllocationSite::RuntimeRaceSpikingIndices,
        prospective_column_occupancy_limit,
    )?;

    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::StoreOwners,
        segment_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::StoreSources,
        posting_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u16>(
        CheckpointV1ImportAllocationSite::StorePermanences,
        posting_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u8>(
        CheckpointV1ImportAllocationSite::StoreOwnerSegmentCounts,
        owner_capacity,
    )?;
    temporal_source_meta_allocation_layout(source_count).map_err(|_| {
        CheckpointV1Pass1Error::AllocationLayout {
            site: CheckpointV1ImportAllocationSite::StoreSourceMeta,
            element_count: source_count,
        }
    })?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::StoreNextPair,
        posting_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::StorePreviousPair,
        posting_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::StoreSourceStamps,
        source_count,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::StoreSegmentStamps,
        segment_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u8>(
        CheckpointV1ImportAllocationSite::StorePotentialContacts,
        segment_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u8>(
        CheckpointV1ImportAllocationSite::StoreConnectedContacts,
        segment_capacity,
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::StoreTouchedSegments,
        segment_capacity,
    )?;
    for site in [
        CheckpointV1ImportAllocationSite::StoreOwnerStamps,
        CheckpointV1ImportAllocationSite::StoreBestMatching,
        CheckpointV1ImportAllocationSite::StoreBestActive,
    ] {
        checkpoint_v1_preflight_layout::<u32>(site, owner_capacity)?;
    }
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::StoreTouchedOwners,
        owner_capacity.min(segment_capacity),
    )?;
    checkpoint_v1_preflight_layout::<u32>(
        CheckpointV1ImportAllocationSite::VerificationSemanticPostings,
        segment_count,
    )?;

    Ok(CheckpointV1ImportAllocationPlan {
        committed_owner_count,
        column_count,
        owner_capacity,
        segment_capacity,
        posting_capacity,
        source_count,
        cohort_capacity,
        prospective_column_occupancy_limit,
        active_segment_capacity,
        semantic_posting_scratch: segment_count,
    })
}

/// Envelope-only upper bound for every current and future column occupancy.
///
/// Pass 1 deliberately does not inspect cohort columns. If all `Kcap` cohorts
/// target one column, that column has `B + R*Kcap` cells; after any prefix of
/// `k` cohorts it has at most `B + R*k`, with exactly `R*(Kcap-k)` recruitable
/// cells remaining, so the same bound covers the runtime's full lifetime.
fn checked_checkpoint_v1_prospective_occupancy_limit(
    base_width: usize,
    race_cap: usize,
    cohort_capacity: usize,
) -> Option<usize> {
    race_cap
        .checked_mul(cohort_capacity)
        .and_then(|reserve| base_width.checked_add(reserve))
}

fn checkpoint_v1_allocation_usize(
    site: CheckpointV1ImportAllocationSite,
    count: u32,
) -> Result<usize, CheckpointV1Pass1Error> {
    usize::try_from(count).map_err(|_| CheckpointV1Pass1Error::AllocationCountArithmetic { site })
}

fn checkpoint_v1_preflight_layout<T>(
    site: CheckpointV1ImportAllocationSite,
    element_count: usize,
) -> Result<(), CheckpointV1Pass1Error> {
    Layout::array::<T>(element_count).map(|_| ()).map_err(|_| {
        CheckpointV1Pass1Error::AllocationLayout {
            site,
            element_count,
        }
    })
}

fn read_checkpoint_v1_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_checkpoint_v1_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_checkpoint_v1_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn decode_checkpoint_v1_poison_code(
    code: u16,
) -> Result<Option<TemporalPoisonCode>, CheckpointV1Pass1Error> {
    Ok(match code {
        0 => None,
        1 => Some(TemporalPoisonCode::MultipleActiveOwners),
        2 => Some(TemporalPoisonCode::MultipleMatchingOwners),
        3 => Some(TemporalPoisonCode::CategorizedRaceCapacity),
        4 => Some(TemporalPoisonCode::MalformedContextOrPolicy),
        5 => Some(TemporalPoisonCode::ProximalPhysics),
        6 => Some(TemporalPoisonCode::ProximalRelayPhysics),
        7 => Some(TemporalPoisonCode::ColumnRacePhysics),
        8 => Some(TemporalPoisonCode::WinnerRelayPhysics),
        9 => Some(TemporalPoisonCode::TickOrCounterOverflow),
        10 => Some(TemporalPoisonCode::StoreOrPlanInvariant),
        11 => Some(TemporalPoisonCode::PhiloxAddressExhaustion),
        12 => Some(TemporalPoisonCode::ReserveCellCapacity),
        13 => Some(TemporalPoisonCode::GlobalSegmentCapacity),
        14 => Some(TemporalPoisonCode::OwnerSegmentCapacity),
        15 => Some(TemporalPoisonCode::PreallocatedCapacityInvariant),
        found => return Err(CheckpointV1Pass1Error::PoisonCode { found }),
    })
}

fn checked_checkpoint_v1_section_range(
    section: CheckpointV1BodySection,
    start: u64,
    byte_count: u64,
    platform_limit: u64,
) -> Result<(u64, u64), CheckpointV1Pass1Error> {
    let end = start
        .checked_add(byte_count)
        .ok_or(CheckpointV1Pass1Error::SectionRangeOverflow { section })?;
    if start > platform_limit || end > platform_limit {
        return Err(CheckpointV1Pass1Error::SectionNotAddressable {
            section,
            end,
            platform_limit,
        });
    }
    Ok((start, end))
}

fn validate_checkpoint_v1_identity_field(
    header: &[u8],
    offset: usize,
    expected: &[u8; 32],
    field: CheckpointV1IdentityField,
) -> Result<(), CheckpointV1Pass1Error> {
    if header[offset..offset + 32] != expected[..] {
        return Err(CheckpointV1Pass1Error::IdentityMismatch { field });
    }
    Ok(())
}

#[cfg(test)]
pub(in crate::temporal_runtime) mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::temporal_runtime::tests::count_allocations;
    use crate::temporal_runtime::{
        TemporalCircuitConfig, TemporalCircuitRuntime, TemporalCircuitStop, TemporalContextPolicy,
        TemporalPlasticityPolicy, TemporalRuntimeBuildError, TemporalSegmentInit, TemporalStepMode,
        TokenCode,
    };

    const EMPTY_SEMANTIC_DIGEST: [u8; 32] = [
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9,
        0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52,
        0xb8, 0x55,
    ];
    // Independent CPython `struct.pack("<III", ...)` + SHA-256 over
    // [(0,0,0), (0,1,0), (1,1,1), (2,0,1), (2,1,2)].
    const SEMANTIC_FIXTURE_DIGEST: [u8; 32] = [
        0x59, 0x56, 0x91, 0x25, 0xdf, 0x76, 0x64, 0x19, 0xd5, 0x69, 0x68, 0x78, 0x5b, 0x84, 0xb7,
        0x59, 0xb4, 0x57, 0x63, 0x1e, 0xb4, 0x6f, 0x26, 0x08, 0x94, 0x3c, 0xb8, 0x10, 0x59, 0xb6,
        0xc1, 0xef,
    ];
    const SEMANTIC_FIXTURE_BODY_HEX: &str = "00000000020000000000000002000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff000000000100000002000000fffffffffffffffffffffffffffffffffffffffffffffffffffffffffa00fb00000000000000000000000000000000002c012d012e010000000000000000000000000000";
    const SEMANTIC_FIXTURE_BOUND_HEADER_HEX: &str = "52334e5443503100010000010000000080010000000000008000000000000000a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f4fa5c0a71a3de926a135486658bb1388b6bcf3f149d98b8f38489c29ad227c5371b9e70f4a7cf28df3e381695159e7f84b3fe1990bd9dadf75bfcc5de9c3427959569125df766419d56968785b84b759b457631eb46f2608943cb81059b6c1ef0200000000000000160000000000000000000000000000000000000000000000ef9373a4c6e5435b1d26aac601496812015351e33aefd3bbe13e1e45aedb3d9b";
    const SEMANTIC_FIXTURE_GENERIC_HEADER_HEX: &str = "52334e5443503100010000010000000080010000000000008000000000000000a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f006c48cd29d0cb25494383ee70390eeffda0adde2014c28041d99b3027ac531171b9e70f4a7cf28df3e381695159e7f84b3fe1990bd9dadf75bfcc5de9c3427959569125df766419d56968785b84b759b457631eb46f2608943cb81059b6c1ef0200000000000000160000000000000000000000000000000000000000000000ef9373a4c6e5435b1d26aac601496812015351e33aefd3bbe13e1e45aedb3d9b";
    // Independent CPython stdlib oracle over a GenericAfferents runtime with
    // two recruited cohorts and distinct source/winner context sections.
    const FULL_SECTION_GENERIC_BODY_HEX: &str = "00000000010000003f000000400000004100000042000000430000004400000045000000460000004700000048000000490000004a0000004b0000004c0000004d0000004e0000004f000000500000005100000052000000530000005400000000000000020000000400000006000000080000000a0000000c0000000e0000001000000012000000140000001a0000001c0000001e00000020000000220000002400000026000000280000002a0000002c0000002e0000000000000001000000160000000200000003000000180000000000000002000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff01000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff030000000500000007000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff000000000a000000fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffa00fb000000000000000000000000000000000000000000000000000000000000000000000000002c01000000000000000000000000000000000000ff007701f40100000000000000000000000000000000000000000000000000000000000000000000f900900100000000000000000000000000000000";
    const FULL_SECTION_GENERIC_HEADER_HEX: &str = "52334e5443503100010000010000000038030000000000003802000000000000a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1fca61287978e0a7c3b76ce050c9cf4961d00c441a9036eaecbe645ad628088c4fdb5a0520fa90d4bc79531d0b1a77ecba759c42e6d0be9567004753e6e260b669dc6222193dfca80b9e29b83e57ec1581d6ef6d94a339046249b8b9a7c8c0a1a506000000020000001a0000000100000000000000000000000000000016160000ba1e27b132140f8dd0f8eabc3eb75a591ed99feb2a2e8cba5d02fc3ac90c2f38";
    const EMPTY_FULL_SECTION_GENERIC_HEADER_HEX: &str = "52334e5443503100010000010000000000010000000000000000000000000000a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1fca61287978e0a7c3b76ce050c9cf4961d00c441a9036eaecbe645ad628088c4fe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8550000000000000000160000000000000000000000000000000000000000000000bbab675473ff34aa5e25ab8ab46d6d87bc54a54480154a9da152c4a2a1894125";
    const FULL_IMPORT_EXPECTED_ALLOCATIONS: [(CheckpointV1ImportAllocationSite, usize); 31] = [
        (CheckpointV1ImportAllocationSite::RuntimeColumnOffsets, 12),
        (CheckpointV1ImportAllocationSite::RuntimeColumnCells, 26),
        (
            CheckpointV1ImportAllocationSite::RuntimeAlternateColumnOffsets,
            12,
        ),
        (
            CheckpointV1ImportAllocationSite::RuntimeAlternateColumnCells,
            26,
        ),
        (CheckpointV1ImportAllocationSite::RuntimeCohortColumns, 2),
        (
            CheckpointV1ImportAllocationSite::RuntimeRecognitionDiagnostics,
            26,
        ),
        (CheckpointV1ImportAllocationSite::RuntimeActiveSegments, 6),
        (CheckpointV1ImportAllocationSite::RuntimeColumnDrive, 4),
        (CheckpointV1ImportAllocationSite::RuntimeNovelCandidates, 4),
        (CheckpointV1ImportAllocationSite::RuntimeRaceVoltage, 4),
        (CheckpointV1ImportAllocationSite::RuntimeRaceThresholds, 4),
        (CheckpointV1ImportAllocationSite::RuntimeRaceExcitatory, 4),
        (CheckpointV1ImportAllocationSite::RuntimeRaceEmitted, 4),
        (
            CheckpointV1ImportAllocationSite::RuntimeRaceSpikingIndices,
            4,
        ),
        (CheckpointV1ImportAllocationSite::StoreOwners, 6),
        (CheckpointV1ImportAllocationSite::StoreSources, 60),
        (CheckpointV1ImportAllocationSite::StorePermanences, 60),
        (
            CheckpointV1ImportAllocationSite::StoreOwnerSegmentCounts,
            26,
        ),
        (CheckpointV1ImportAllocationSite::StoreSourceMeta, 11),
        (CheckpointV1ImportAllocationSite::StoreNextPair, 60),
        (CheckpointV1ImportAllocationSite::StorePreviousPair, 60),
        (CheckpointV1ImportAllocationSite::StoreSourceStamps, 11),
        (CheckpointV1ImportAllocationSite::StoreSegmentStamps, 6),
        (CheckpointV1ImportAllocationSite::StorePotentialContacts, 6),
        (CheckpointV1ImportAllocationSite::StoreConnectedContacts, 6),
        (CheckpointV1ImportAllocationSite::StoreTouchedSegments, 6),
        (CheckpointV1ImportAllocationSite::StoreOwnerStamps, 26),
        (CheckpointV1ImportAllocationSite::StoreBestMatching, 26),
        (CheckpointV1ImportAllocationSite::StoreBestActive, 26),
        (CheckpointV1ImportAllocationSite::StoreTouchedOwners, 6),
        (
            CheckpointV1ImportAllocationSite::VerificationSemanticPostings,
            6,
        ),
    ];

    fn expected_bound_import_allocations(
        owner_capacity: usize,
        segment_capacity: usize,
        cohort_capacity: usize,
        active_segment_capacity: usize,
        race_capacity: usize,
        semantic_posting_count: usize,
    ) -> Vec<(CheckpointV1ImportAllocationSite, usize)> {
        let column_offset_count = 12;
        let posting_capacity = PAIRS_PER_SEGMENT * segment_capacity;
        let source_count = owner_capacity;
        let touched_owner_capacity = owner_capacity.min(segment_capacity);
        let mut expected = vec![
            (
                CheckpointV1ImportAllocationSite::RuntimeColumnOffsets,
                column_offset_count,
            ),
            (
                CheckpointV1ImportAllocationSite::RuntimeColumnCells,
                owner_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::RuntimeAlternateColumnOffsets,
                column_offset_count,
            ),
            (
                CheckpointV1ImportAllocationSite::RuntimeAlternateColumnCells,
                owner_capacity,
            ),
        ];
        if cohort_capacity != 0 {
            expected.push((
                CheckpointV1ImportAllocationSite::RuntimeCohortColumns,
                cohort_capacity,
            ));
        }
        expected.push((
            CheckpointV1ImportAllocationSite::RuntimeRecognitionDiagnostics,
            owner_capacity,
        ));
        if active_segment_capacity != 0 {
            expected.push((
                CheckpointV1ImportAllocationSite::RuntimeActiveSegments,
                active_segment_capacity,
            ));
        }
        expected.extend([
            (
                CheckpointV1ImportAllocationSite::RuntimeColumnDrive,
                race_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::RuntimeNovelCandidates,
                race_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::RuntimeRaceVoltage,
                race_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::RuntimeRaceThresholds,
                race_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::RuntimeRaceExcitatory,
                race_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::RuntimeRaceEmitted,
                race_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::RuntimeRaceSpikingIndices,
                race_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreOwners,
                segment_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreSources,
                posting_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StorePermanences,
                posting_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreOwnerSegmentCounts,
                owner_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreSourceMeta,
                source_count,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreNextPair,
                posting_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StorePreviousPair,
                posting_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreSourceStamps,
                source_count,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreSegmentStamps,
                segment_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StorePotentialContacts,
                segment_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreConnectedContacts,
                segment_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreTouchedSegments,
                segment_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreOwnerStamps,
                owner_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreBestMatching,
                owner_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreBestActive,
                owner_capacity,
            ),
            (
                CheckpointV1ImportAllocationSite::StoreTouchedOwners,
                touched_owner_capacity,
            ),
        ]);
        if semantic_posting_count != 0 {
            expected.push((
                CheckpointV1ImportAllocationSite::VerificationSemanticPostings,
                semantic_posting_count,
            ));
        }
        expected
    }

    #[derive(Debug, Eq, PartialEq)]
    struct CheckpointV1LogicalStateSnapshot {
        column_offsets: Vec<usize>,
        column_cells: Vec<u32>,
        alternate_column_offsets: Vec<usize>,
        alternate_column_cells: Vec<u32>,
        cohort_columns: Vec<u32>,
        next_reserve_cell: u32,
        previous_context: [u32; CONTEXT_CELLS],
        previous_context_len: u8,
        previous_winner_context: [u32; CONTEXT_CELLS],
        previous_winner_context_len: u8,
        event_count: u64,
        learning_step: u64,
        scientific_stop: Option<TemporalPoisonCode>,
        rows: Vec<(u32, TemporalSegmentRow)>,
        owner_loads: Vec<Option<u8>>,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum TestSinkCall {
        SeekStart(u64),
        SeekEnd(i64),
        SeekCurrent(i64),
        Write(usize),
        Flush,
    }

    #[derive(Clone, Copy, Debug)]
    enum TestWriteFault {
        Error(io::ErrorKind),
        Zero,
        Overreport,
    }

    struct TestSink {
        inner: Cursor<Vec<u8>>,
        calls: Vec<TestSinkCall>,
        seek_calls: usize,
        write_calls: usize,
        flush_calls: usize,
        seek_fault: Option<(usize, io::ErrorKind)>,
        seek_position: Option<(usize, u64)>,
        write_fault: Option<(usize, TestWriteFault)>,
        flush_fault: Option<(usize, io::ErrorKind)>,
        max_write: Option<usize>,
    }

    impl TestSink {
        fn new() -> Self {
            Self::with_bytes(Vec::new())
        }

        fn with_bytes(bytes: Vec<u8>) -> Self {
            Self {
                inner: Cursor::new(bytes),
                calls: Vec::new(),
                seek_calls: 0,
                write_calls: 0,
                flush_calls: 0,
                seek_fault: None,
                seek_position: None,
                write_fault: None,
                flush_fault: None,
                max_write: None,
            }
        }

        fn bytes(&self) -> &[u8] {
            self.inner.get_ref()
        }

        fn position(&self) -> u64 {
            self.inner.position()
        }
    }

    impl Write for TestSink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.write_calls += 1;
            self.calls.push(TestSinkCall::Write(bytes.len()));
            if self
                .write_fault
                .as_ref()
                .is_some_and(|(call, _)| *call == self.write_calls)
            {
                let (_, fault) = self.write_fault.take().expect("matching fault must exist");
                return match fault {
                    TestWriteFault::Error(kind) => Err(io::Error::new(kind, "injected write")),
                    TestWriteFault::Zero => Ok(0),
                    TestWriteFault::Overreport => Ok(bytes.len().saturating_add(1)),
                };
            }
            let accepted = self.max_write.unwrap_or(bytes.len()).min(bytes.len());
            self.inner.write(&bytes[..accepted])
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flush_calls += 1;
            self.calls.push(TestSinkCall::Flush);
            if self
                .flush_fault
                .as_ref()
                .is_some_and(|(call, _)| *call == self.flush_calls)
            {
                let (_, kind) = self.flush_fault.take().expect("matching fault must exist");
                return Err(io::Error::new(kind, "injected flush"));
            }
            self.inner.flush()
        }
    }

    impl Seek for TestSink {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.seek_calls += 1;
            self.calls.push(match position {
                SeekFrom::Start(offset) => TestSinkCall::SeekStart(offset),
                SeekFrom::End(offset) => TestSinkCall::SeekEnd(offset),
                SeekFrom::Current(offset) => TestSinkCall::SeekCurrent(offset),
            });
            if self
                .seek_fault
                .as_ref()
                .is_some_and(|(call, _)| *call == self.seek_calls)
            {
                let (_, kind) = self.seek_fault.take().expect("matching fault must exist");
                return Err(io::Error::new(kind, "injected seek"));
            }
            let actual = self.inner.seek(position)?;
            if self
                .seek_position
                .as_ref()
                .is_some_and(|(call, _)| *call == self.seek_calls)
            {
                return Ok(self
                    .seek_position
                    .take()
                    .expect("matching position must exist")
                    .1);
            }
            Ok(actual)
        }
    }

    fn hex_bytes(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let pair = std::str::from_utf8(pair).expect("literal hex must be ASCII");
                u8::from_str_radix(pair, 16).expect("literal hex must be valid")
            })
            .collect()
    }

    fn identity_runtime(
        column_count: u32,
        context_policy: TemporalContextPolicy,
        codebook: Vec<TokenCode>,
        mut threshold_offsets_mv: Vec<f64>,
    ) -> Result<TemporalCircuitRuntime, TemporalRuntimeBuildError> {
        let config = TemporalCircuitConfig {
            column_count,
            base_cells_per_column: 2,
            max_pyramidal_cells: 2 * column_count,
            segment_capacity: 1,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 1,
            theta_match: 18,
            theta_active: 20,
            context_policy,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let members = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        if threshold_offsets_mv.is_empty() {
            threshold_offsets_mv.resize((7 * column_count) as usize, 0.0);
        }

        TemporalCircuitRuntime::new(config, codebook, members, threshold_offsets_mv, &[])
    }

    fn frozen_route_zero_runtime() -> TemporalCircuitRuntime {
        let codebook = vec![TokenCode {
            token_id: 7,
            columns: std::array::from_fn(|index| index as u32),
        }];

        identity_runtime(
            11,
            TemporalContextPolicy::BoundWinners,
            codebook,
            Vec::new(),
        )
        .expect("literal route-0 runtime must build")
    }

    fn nonuniform_generic_route_zero_runtime() -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 37,
            base_cells_per_column: 3,
            max_pyramidal_cells: 131,
            segment_capacity: 19,
            novel_race_cap: 7,
            prediction_quorum: 5,
            construction_seed: 0xa1b2_c3d4,
            connected_permanence_ticks: 173,
            initial_permanence_ticks: 311,
            permanence_increment_ticks: 43,
            negative_decrement_ticks: 7,
            max_segments_per_cell: 6,
            theta_match: 9,
            theta_active: 17,
            context_policy: TemporalContextPolicy::GenericAfferents,
            plasticity_policy: TemporalPlasticityPolicy::GrowthOff,
        };
        let codebook = vec![
            TokenCode {
                token_id: 0x1122_3344,
                columns: [0, 2, 5, 9, 12, 16, 20, 24, 29, 33, 36],
            },
            TokenCode {
                token_id: 0x5566_7788,
                columns: [1, 3, 6, 10, 13, 17, 21, 25, 28, 32, 35],
            },
        ];
        let members = (0..config.column_count)
            .map(|column| vec![3 * column, 3 * column + 1, 3 * column + 2])
            .collect();
        let mut threshold_offsets_mv = vec![0.0; 373];
        threshold_offsets_mv[0] = f64::from_bits(0x3ff4_0000_0000_0000);
        threshold_offsets_mv[1] = f64::from_bits(0x8000_0000_0000_0000);
        threshold_offsets_mv[372] = f64::from_bits(0xc004_0000_0000_0000);

        TemporalCircuitRuntime::new(config, codebook, members, threshold_offsets_mv, &[])
            .expect("nonuniform generic route-0 runtime must build")
    }

    fn authority() -> ValidatedCheckpointAuthority {
        ValidatedCheckpointAuthority::from_raw_sha256(
            [0xa5; 32],
            std::array::from_fn(|index| index as u8),
        )
    }

    fn factory_for_runtime(
        runtime: &TemporalCircuitRuntime,
    ) -> ValidatedCheckpointV1RuntimeFactory {
        ValidatedCheckpointV1RuntimeFactory::new(
            authority(),
            runtime.config,
            runtime.token_codebook.clone(),
            runtime.threshold_offsets_mv.clone(),
        )
        .expect("runtime fixture must define one validated checkpoint factory")
    }

    fn factory_for_runtime_with_authority(
        runtime: &TemporalCircuitRuntime,
        checkpoint_authority: ValidatedCheckpointAuthority,
    ) -> ValidatedCheckpointV1RuntimeFactory {
        ValidatedCheckpointV1RuntimeFactory::new(
            checkpoint_authority,
            runtime.config,
            runtime.token_codebook.clone(),
            runtime.threshold_offsets_mv.clone(),
        )
        .expect("runtime fixture must define one validated checkpoint factory")
    }

    fn encoded_checkpoint(runtime: &TemporalCircuitRuntime) -> Vec<u8> {
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("fixture must prepare for checkpoint export");
        let mut sink = Cursor::new(Vec::new());
        prepared
            .encode_into(&mut sink)
            .expect("fixture checkpoint must encode");
        sink.into_inner()
    }

    fn literal_checkpoint(header_hex: &str, body_hex: &str) -> Vec<u8> {
        let mut checkpoint = hex_bytes(header_hex);
        checkpoint.extend_from_slice(&hex_bytes(body_hex));
        checkpoint
    }

    pub(in crate::temporal_runtime) fn cpython_full_section_generic_checkpoint_v1() -> Vec<u8> {
        let checkpoint = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        assert_eq!(checkpoint.len(), 824);
        checkpoint
    }

    pub(in crate::temporal_runtime) fn cpython_empty_full_section_generic_checkpoint_v1() -> Vec<u8>
    {
        let checkpoint = hex_bytes(EMPTY_FULL_SECTION_GENERIC_HEADER_HEX);
        assert_eq!(checkpoint.len(), usize::from(CHECKPOINT_V1_HEADER_BYTES));
        checkpoint
    }

    fn refresh_logical_state_digest(checkpoint: &mut [u8]) {
        let mut stored_body_digest = [0u8; 32];
        stored_body_digest.copy_from_slice(&checkpoint[128..160]);
        let mut scalar_block = [0u8; 32];
        scalar_block.copy_from_slice(&checkpoint[192..224]);
        let digest = checkpoint_v1_logical_state_digest(&stored_body_digest, &scalar_block);
        checkpoint[224..256].copy_from_slice(&digest);
    }

    fn refresh_body_and_logical_state_digests(checkpoint: &mut [u8]) {
        let body_digest: [u8; 32] = Sha256::digest(&checkpoint[256..]).into();
        checkpoint[128..160].copy_from_slice(&body_digest);
        refresh_logical_state_digest(checkpoint);
    }

    #[derive(Clone, Copy)]
    struct TestCheckpointV1SectionStarts {
        cohort_columns: usize,
        prior_context: usize,
        winner_context: usize,
        segment_owners: usize,
        segment_sources: usize,
        segment_ticks: usize,
    }

    fn checkpoint_v1_test_section_starts(checkpoint: &[u8]) -> TestCheckpointV1SectionStarts {
        let layout = CheckpointV1Layout::checked(
            read_checkpoint_v1_u32(checkpoint, 192),
            read_checkpoint_v1_u32(checkpoint, 196),
            checkpoint[220],
            checkpoint[221],
        )
        .expect("test checkpoint must have a representable layout");
        let cohort_columns = usize::from(CHECKPOINT_V1_HEADER_BYTES);
        let prior_context = cohort_columns + layout.cohort_columns_bytes as usize;
        let winner_context = prior_context + layout.prior_context_bytes as usize;
        let segment_owners = winner_context + layout.winner_context_bytes as usize;
        let segment_sources = segment_owners + layout.segment_owner_bytes as usize;
        let segment_ticks = segment_sources + layout.segment_source_bytes as usize;
        TestCheckpointV1SectionStarts {
            cohort_columns,
            prior_context,
            winner_context,
            segment_owners,
            segment_sources,
            segment_ticks,
        }
    }

    fn write_checkpoint_v1_u32(checkpoint: &mut [u8], start: usize, index: usize, value: u32) {
        let offset = start + index * size_of::<u32>();
        checkpoint[offset..offset + size_of::<u32>()].copy_from_slice(&value.to_le_bytes());
    }

    fn write_checkpoint_v1_u16(checkpoint: &mut [u8], start: usize, index: usize, value: u16) {
        let offset = start + index * size_of::<u16>();
        checkpoint[offset..offset + size_of::<u16>()].copy_from_slice(&value.to_le_bytes());
    }

    fn pass2_error(
        checkpoint: &[u8],
        factory: &ValidatedCheckpointV1RuntimeFactory,
    ) -> CheckpointV1Pass2Error {
        let pass1 = validate_checkpoint_v1_pass1(checkpoint, factory)
            .expect("digest-refreshed corruption must reach Pass 2");
        match validate_checkpoint_v1_pass2(pass1, factory) {
            Ok(_) => panic!("checkpoint unexpectedly passed canonical-state validation"),
            Err(error) => error,
        }
    }

    fn remove_checkpoint_v1_context_section(
        checkpoint: &mut Vec<u8>,
        field: CheckpointV1ContextLengthField,
    ) {
        let starts = checkpoint_v1_test_section_starts(checkpoint);
        let (range, length_offset) = match field {
            CheckpointV1ContextLengthField::Prior => {
                (starts.prior_context..starts.winner_context, 220)
            }
            CheckpointV1ContextLengthField::Winner => {
                (starts.winner_context..starts.segment_owners, 221)
            }
        };
        let removed = range.len() as u64;
        checkpoint.drain(range);
        checkpoint[length_offset] = 0;
        let total = read_checkpoint_v1_u64(checkpoint, 16) - removed;
        let body = read_checkpoint_v1_u64(checkpoint, 24) - removed;
        checkpoint[16..24].copy_from_slice(&total.to_le_bytes());
        checkpoint[24..32].copy_from_slice(&body.to_le_bytes());
        refresh_body_and_logical_state_digests(checkpoint);
    }

    fn pass1_error(
        checkpoint: &[u8],
        factory: &ValidatedCheckpointV1RuntimeFactory,
    ) -> CheckpointV1Pass1Error {
        match validate_checkpoint_v1_pass1(checkpoint, factory) {
            Ok(_) => panic!("checkpoint unexpectedly passed envelope validation"),
            Err(error) => error,
        }
    }

    fn empty_runtime(context_policy: TemporalContextPolicy) -> TemporalCircuitRuntime {
        identity_runtime(
            11,
            context_policy,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            Vec::new(),
        )
        .expect("empty state fixture must build")
    }

    fn empty_runtime_with_plasticity(
        context_policy: TemporalContextPolicy,
        plasticity_policy: TemporalPlasticityPolicy,
    ) -> TemporalCircuitRuntime {
        let baseline = empty_runtime(context_policy);
        let config = TemporalCircuitConfig {
            plasticity_policy,
            ..baseline.config
        };
        let members = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        TemporalCircuitRuntime::new(
            config,
            baseline.token_codebook.clone(),
            members,
            baseline.threshold_offsets_mv.clone(),
            &[],
        )
        .expect("empty policy fixture must build")
    }

    fn recruited_runtime(context_policy: TemporalContextPolicy) -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 11,
            base_cells_per_column: 2,
            max_pyramidal_cells: 24,
            segment_capacity: 8,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 2,
            theta_match: 18,
            theta_active: 20,
            context_policy,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let mut members: Vec<Vec<u32>> = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        members[0].extend([22, 23]);
        let segments = [0, 0, 1, 1, 22].map(|owner| TemporalSegmentInit {
            owner,
            sources: Vec::new(),
            permanence_ticks: Vec::new(),
        });

        TemporalCircuitRuntime::new(
            config,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            members,
            vec![0.0; 81],
            &segments,
        )
        .expect("chronologically recruited fixture must build")
    }

    fn full_section_generic_runtime() -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 11,
            base_cells_per_column: 2,
            max_pyramidal_cells: 26,
            segment_capacity: 6,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 1,
            theta_match: 18,
            theta_active: 20,
            context_policy: TemporalContextPolicy::GenericAfferents,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let mut members: Vec<Vec<u32>> = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        members[0].extend([22, 23]);
        members[1].extend([24, 25]);
        let segments = [
            TemporalSegmentInit {
                owner: 0,
                sources: vec![63, 65],
                permanence_ticks: vec![250, 251],
            },
            TemporalSegmentInit {
                owner: 1,
                sources: Vec::new(),
                permanence_ticks: Vec::new(),
            },
            TemporalSegmentInit {
                owner: 22,
                sources: vec![64],
                permanence_ticks: vec![300],
            },
            TemporalSegmentInit {
                owner: 2,
                sources: vec![66, 68, 70],
                permanence_ticks: vec![255, 375, 500],
            },
            TemporalSegmentInit {
                owner: 3,
                sources: Vec::new(),
                permanence_ticks: Vec::new(),
            },
            TemporalSegmentInit {
                owner: 24,
                sources: vec![63, 73],
                permanence_ticks: vec![249, 400],
            },
        ];
        let mut runtime = TemporalCircuitRuntime::new(
            config,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            members,
            vec![0.0; 85],
            &segments,
        )
        .expect("full-section generic fixture must build");
        set_reachable_context(&mut runtime);
        runtime
    }

    fn full_policy_runtime(
        context_policy: TemporalContextPolicy,
        plasticity_policy: TemporalPlasticityPolicy,
    ) -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 11,
            base_cells_per_column: 2,
            max_pyramidal_cells: 26,
            segment_capacity: 6,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 1,
            theta_match: 18,
            theta_active: 20,
            context_policy,
            plasticity_policy,
        };
        let mut members: Vec<Vec<u32>> = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        members[0].extend([22, 23]);
        members[1].extend([24, 25]);
        let segments = [0, 1, 22, 2, 3, 24].map(|owner| TemporalSegmentInit {
            owner,
            sources: Vec::new(),
            permanence_ticks: Vec::new(),
        });
        let mut runtime = TemporalCircuitRuntime::new(
            config,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            members,
            vec![0.0; 85],
            &segments,
        )
        .expect("full policy fixture must build");
        set_reachable_context(&mut runtime);
        runtime
    }

    fn exact_source_reservation_runtime() -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 6 * ACTIVE_COLUMNS as u32,
            base_cells_per_column: 2,
            max_pyramidal_cells: 12 * ACTIVE_COLUMNS as u32,
            segment_capacity: 64,
            novel_race_cap: 2,
            prediction_quorum: ACTIVE_COLUMNS as u8,
            construction_seed: 17,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 4,
            theta_match: 16,
            theta_active: 18,
            context_policy: TemporalContextPolicy::BoundWinners,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let codebook = (0..6u32)
            .map(|token_id| TokenCode {
                token_id,
                columns: std::array::from_fn(|index| {
                    token_id * ACTIVE_COLUMNS as u32 + index as u32
                }),
            })
            .collect();
        let members = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        let physical_count = (2 * config.max_pyramidal_cells + 3 * config.column_count) as usize;
        TemporalCircuitRuntime::new(config, codebook, members, vec![0.0; physical_count], &[])
            .expect("exact-source reservation fixture must build")
    }

    fn recover_exact_source_owner(
        runtime: &mut TemporalCircuitRuntime,
        expected_source_winners: [u32; CONTEXT_CELLS],
        expected_target_winners: [u32; CONTEXT_CELLS],
    ) {
        runtime.reset_transient();
        let source = runtime.step(0, TemporalStepMode::Learn).unwrap();
        assert_eq!(source.winner_context, expected_source_winners);
        let recovered = runtime.step(1, TemporalStepMode::Learn).unwrap();
        assert_eq!(recovered.predicted_columns, 0);
        assert_eq!(recovered.allocated_segments, 0);
        assert_eq!(recovered.winner_context, expected_target_winners);
        for segment_id in 0..ACTIVE_COLUMNS as u32 {
            assert_eq!(
                runtime
                    .store
                    .runtime_segment_row(segment_id)
                    .unwrap()
                    .permanence_ticks,
                [299; PAIRS_PER_SEGMENT],
            );
        }

        runtime.reset_transient();
        runtime.step(0, TemporalStepMode::ReadOnly).unwrap();
        let active = runtime.step(1, TemporalStepMode::ReadOnly).unwrap();
        assert_eq!(active.predicted_columns, ACTIVE_COLUMNS as u8);
        assert_eq!(active.allocated_segments, 0);
        assert_eq!(active.winner_context, expected_target_winners);

        runtime.reset_transient();
        let novel_source = runtime.step(5, TemporalStepMode::Learn).unwrap();
        let pair_overlap = novel_source.winner_context[..ACTIVE_COLUMNS]
            .iter()
            .filter(|winner| expected_source_winners[..ACTIVE_COLUMNS].contains(winner))
            .count();
        assert!(pair_overlap <= 7);
        let novel = runtime.step(1, TemporalStepMode::Learn).unwrap();
        assert_eq!(novel.predicted_columns, 0);
        assert_eq!(novel.allocated_segments, ACTIVE_COLUMNS as u8);
    }

    fn capacity_only_policy_runtime(
        context_policy: TemporalContextPolicy,
        plasticity_policy: TemporalPlasticityPolicy,
    ) -> TemporalCircuitRuntime {
        let full = full_policy_runtime(context_policy, plasticity_policy);
        let members = (0..full.config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        TemporalCircuitRuntime::new(
            full.config,
            full.token_codebook.clone(),
            members,
            full.threshold_offsets_mv.clone(),
            &[],
        )
        .expect("capacity-only import fixture must build")
    }

    fn bound_winner_source_runtime() -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 11,
            base_cells_per_column: 2,
            max_pyramidal_cells: 26,
            segment_capacity: 6,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 1,
            theta_match: 18,
            theta_active: 20,
            context_policy: TemporalContextPolicy::BoundWinners,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let mut members: Vec<Vec<u32>> = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        members[0].extend([22, 23]);
        members[1].extend([24, 25]);
        let segments = [
            TemporalSegmentInit {
                owner: 0,
                sources: vec![23],
                permanence_ticks: vec![300],
            },
            TemporalSegmentInit {
                owner: 1,
                sources: Vec::new(),
                permanence_ticks: Vec::new(),
            },
            TemporalSegmentInit {
                owner: 22,
                sources: Vec::new(),
                permanence_ticks: Vec::new(),
            },
            TemporalSegmentInit {
                owner: 2,
                sources: Vec::new(),
                permanence_ticks: Vec::new(),
            },
            TemporalSegmentInit {
                owner: 3,
                sources: Vec::new(),
                permanence_ticks: Vec::new(),
            },
            TemporalSegmentInit {
                owner: 24,
                sources: Vec::new(),
                permanence_ticks: Vec::new(),
            },
        ];
        let mut runtime = TemporalCircuitRuntime::new(
            config,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            members,
            vec![0.0; 85],
            &segments,
        )
        .expect("nonempty bound-winner fixture must build");
        set_reachable_context(&mut runtime);
        runtime
    }

    fn growth_ready_bound_runtime() -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 22,
            base_cells_per_column: 1,
            max_pyramidal_cells: 110,
            segment_capacity: 512,
            novel_race_cap: 8,
            prediction_quorum: ACTIVE_COLUMNS as u8,
            construction_seed: 17,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 1,
            theta_match: 18,
            theta_active: 20,
            context_policy: TemporalContextPolicy::BoundWinners,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let members = (0..config.column_count)
            .map(|column| vec![column])
            .collect();
        let segments: Vec<_> = (11..22)
            .map(|owner| TemporalSegmentInit {
                owner,
                sources: (11..21).collect(),
                permanence_ticks: vec![255; PAIRS_PER_SEGMENT],
            })
            .collect();
        let codebook = vec![
            TokenCode {
                token_id: 0,
                columns: std::array::from_fn(|index| index as u32),
            },
            TokenCode {
                token_id: 1,
                columns: std::array::from_fn(|index| (ACTIVE_COLUMNS + index) as u32),
            },
        ];
        let physical_count = (2 * config.max_pyramidal_cells + 3 * config.column_count) as usize;

        TemporalCircuitRuntime::new(
            config,
            codebook,
            members,
            vec![0.0; physical_count],
            &segments,
        )
        .expect("growth-ready bound-winner fixture must build")
    }

    fn interleaved_cohort_runtime() -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 11,
            base_cells_per_column: 2,
            max_pyramidal_cells: 28,
            segment_capacity: 8,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 1,
            theta_match: 18,
            theta_active: 20,
            context_policy: TemporalContextPolicy::BoundWinners,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let mut members: Vec<Vec<u32>> = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        members[0].extend([22, 23, 26, 27]);
        members[1].extend([24, 25]);
        let segments = [0, 1, 22, 23, 26, 2, 3, 24].map(|owner| TemporalSegmentInit {
            owner,
            sources: Vec::new(),
            permanence_ticks: Vec::new(),
        });
        TemporalCircuitRuntime::new(
            config,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            members,
            vec![0.0; 89],
            &segments,
        )
        .expect("interleaved cohort fixture must build")
    }

    fn every_row_width_generic_runtime() -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 11,
            base_cells_per_column: 2,
            max_pyramidal_cells: 22,
            segment_capacity: 11,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 1,
            theta_match: 18,
            theta_active: 20,
            context_policy: TemporalContextPolicy::GenericAfferents,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let proximal_offset = 55;
        let segments: Vec<_> = (0u32..=10)
            .map(|width| TemporalSegmentInit {
                owner: width,
                sources: (0..width).map(|source| proximal_offset + source).collect(),
                permanence_ticks: (0..width)
                    .map(|slot| 250 + u16::try_from(slot).unwrap())
                    .collect(),
            })
            .collect();
        let members = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        TemporalCircuitRuntime::new(
            config,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            members,
            vec![0.0; 77],
            &segments,
        )
        .expect("every-width generic row fixture must build")
    }

    fn two_cohorts_same_column_runtime() -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 11,
            base_cells_per_column: 2,
            max_pyramidal_cells: 26,
            segment_capacity: 9,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 2,
            theta_match: 18,
            theta_active: 20,
            context_policy: TemporalContextPolicy::BoundWinners,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let mut members: Vec<Vec<u32>> = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        members[0].extend([22, 23, 24, 25]);
        let segments = [0, 0, 1, 1, 22, 22, 23, 23, 24].map(|owner| TemporalSegmentInit {
            owner,
            sources: Vec::new(),
            permanence_ticks: Vec::new(),
        });
        TemporalCircuitRuntime::new(
            config,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            members,
            vec![0.0; 85],
            &segments,
        )
        .expect("same-column chronology fixture must build")
    }

    fn semantic_runtime(context_policy: TemporalContextPolicy) -> TemporalCircuitRuntime {
        let config = TemporalCircuitConfig {
            column_count: 11,
            base_cells_per_column: 2,
            max_pyramidal_cells: 22,
            segment_capacity: 4,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: 2,
            theta_match: 18,
            theta_active: 20,
            context_policy,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let proximal_offset = 55;
        let source = |normalized| match context_policy {
            TemporalContextPolicy::BoundWinners => normalized,
            TemporalContextPolicy::GenericAfferents => proximal_offset + normalized,
        };
        let segments = [
            TemporalSegmentInit {
                owner: 0,
                sources: vec![source(0), source(2)],
                permanence_ticks: vec![250, 251],
            },
            TemporalSegmentInit {
                owner: 2,
                sources: vec![source(0), source(1), source(2)],
                permanence_ticks: vec![300, 301, 302],
            },
        ];
        let members = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();

        TemporalCircuitRuntime::new(
            config,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            members,
            vec![0.0; 77],
            &segments,
        )
        .expect("semantic digest fixture must build")
    }

    fn multi_code_context_runtime(context_policy: TemporalContextPolicy) -> TemporalCircuitRuntime {
        let first_columns: [u32; ACTIVE_COLUMNS] = std::array::from_fn(|index| index as u32);
        let second_columns: [u32; ACTIVE_COLUMNS] = std::array::from_fn(|index| index as u32 + 2);
        let mut runtime = identity_runtime(
            13,
            context_policy,
            vec![
                TokenCode {
                    token_id: 7,
                    columns: first_columns,
                },
                TokenCode {
                    token_id: 8,
                    columns: second_columns,
                },
            ],
            Vec::new(),
        )
        .expect("multi-code context fixture must build");
        let winners = second_columns.map(|column| 2 * column);
        runtime.previous_winner_context = std::array::from_fn(|index| {
            if index < ACTIVE_COLUMNS {
                winners[index]
            } else {
                runtime.relay_offset + winners[index - ACTIVE_COLUMNS]
            }
        });
        runtime.previous_context = match context_policy {
            TemporalContextPolicy::BoundWinners => runtime.previous_winner_context,
            TemporalContextPolicy::GenericAfferents => std::array::from_fn(|index| {
                let column = second_columns[index % ACTIVE_COLUMNS];
                if index < ACTIVE_COLUMNS {
                    runtime.proximal_offset + column
                } else {
                    runtime.proximal_relay_offset + column
                }
            }),
        };
        runtime.previous_context_len = CONTEXT_CELLS_U8;
        runtime.previous_winner_context_len = CONTEXT_CELLS_U8;
        runtime.event_count = 1;
        runtime
    }

    fn large_streaming_runtime() -> TemporalCircuitRuntime {
        const SEGMENTS: u32 = 257;
        let config = TemporalCircuitConfig {
            column_count: 11,
            base_cells_per_column: 2,
            max_pyramidal_cells: 22,
            segment_capacity: SEGMENTS,
            novel_race_cap: 2,
            prediction_quorum: 1,
            construction_seed: 0x0102_0304,
            connected_permanence_ticks: 250,
            initial_permanence_ticks: 255,
            permanence_increment_ticks: 50,
            negative_decrement_ticks: 2,
            max_segments_per_cell: u8::MAX,
            theta_match: 18,
            theta_active: 20,
            context_policy: TemporalContextPolicy::BoundWinners,
            plasticity_policy: TemporalPlasticityPolicy::Plastic,
        };
        let members = (0..config.column_count)
            .map(|column| vec![2 * column, 2 * column + 1])
            .collect();
        let segments: Vec<_> = (0..SEGMENTS)
            .map(|segment_id| TemporalSegmentInit {
                owner: segment_id % 2,
                sources: Vec::new(),
                permanence_ticks: Vec::new(),
            })
            .collect();

        TemporalCircuitRuntime::new(
            config,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            members,
            vec![0.0; 77],
            &segments,
        )
        .expect("large streaming fixture must build")
    }

    fn checkpoint_v1_logical_state_snapshot(
        runtime: &TemporalCircuitRuntime,
    ) -> CheckpointV1LogicalStateSnapshot {
        CheckpointV1LogicalStateSnapshot {
            column_offsets: runtime.column_offsets.clone(),
            column_cells: runtime.column_cells.clone(),
            alternate_column_offsets: runtime.alternate_column_offsets.clone(),
            alternate_column_cells: runtime.alternate_column_cells.clone(),
            cohort_columns: runtime.cohort_columns.clone(),
            next_reserve_cell: runtime.next_reserve_cell,
            previous_context: runtime.previous_context,
            previous_context_len: runtime.previous_context_len,
            previous_winner_context: runtime.previous_winner_context,
            previous_winner_context_len: runtime.previous_winner_context_len,
            event_count: runtime.event_count,
            learning_step: runtime.learning_step,
            scientific_stop: runtime.scientific_stop,
            rows: (0..runtime.store.runtime_segment_count())
                .map(|segment_id| {
                    (
                        runtime
                            .store
                            .runtime_segment_owner(segment_id)
                            .expect("snapshot owner must exist"),
                        runtime
                            .store
                            .runtime_segment_row(segment_id)
                            .expect("snapshot row must be canonical"),
                    )
                })
                .collect(),
            owner_loads: (0..runtime.config.max_pyramidal_cells)
                .map(|owner| runtime.store.runtime_owner_load(owner))
                .collect(),
        }
    }

    fn assert_same_prepared_proof(
        left: &PreparedCheckpointV1Export<'_>,
        right: &PreparedCheckpointV1Export<'_>,
    ) {
        assert!(std::ptr::eq(left.runtime, right.runtime));
        assert_eq!(left.identity, right.identity);
        assert_eq!(left.layout, right.layout);
        assert_eq!(left.scalars, right.scalars);
        assert_eq!(left.semantic_index_digest, right.semantic_index_digest);
    }

    fn runtime_owned_row(
        runtime: &TemporalCircuitRuntime,
        segment_id: u32,
    ) -> Option<CheckpointV1OwnedRow> {
        Some(CheckpointV1OwnedRow {
            owner: runtime.store.runtime_segment_owner(segment_id)?,
            row: runtime.store.runtime_segment_row(segment_id).ok()?,
        })
    }

    fn validate_view(
        runtime: &TemporalCircuitRuntime,
        state: CheckpointV1StateView<'_>,
        column_replay: &mut [u32],
    ) -> Result<(), CheckpointV1StateError> {
        validate_checkpoint_v1_state(
            state,
            column_replay,
            |segment_id| runtime_owned_row(runtime, segment_id),
            |owner| runtime.store.runtime_owner_load(owner),
        )
    }

    fn set_reachable_context(runtime: &mut TemporalCircuitRuntime) {
        let winners: [u32; ACTIVE_COLUMNS] = std::array::from_fn(|index| 2 * index as u32);
        let winner_context = std::array::from_fn(|index| {
            if index < ACTIVE_COLUMNS {
                winners[index]
            } else {
                runtime.relay_offset + winners[index - ACTIVE_COLUMNS]
            }
        });
        runtime.previous_winner_context = winner_context;
        runtime.previous_context = match runtime.config.context_policy {
            TemporalContextPolicy::BoundWinners => winner_context,
            TemporalContextPolicy::GenericAfferents => std::array::from_fn(|index| {
                if index < ACTIVE_COLUMNS {
                    runtime.proximal_offset + index as u32
                } else {
                    runtime.proximal_relay_offset + (index - ACTIVE_COLUMNS) as u32
                }
            }),
        };
        runtime.previous_context_len = CONTEXT_CELLS_U8;
        runtime.previous_winner_context_len = CONTEXT_CELLS_U8;
        runtime.event_count = 1;
    }

    #[test]
    fn checkpoint_state_validator_accepts_empty_recruited_and_context_states_for_both_policies() {
        for policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            let empty = empty_runtime(policy);
            let mut column_replay = [0u32; ACTIVE_COLUMNS];
            assert_eq!(
                validate_checkpoint_v1_runtime_state(&empty, &mut column_replay),
                Ok(())
            );

            let recruited = recruited_runtime(policy);
            assert_eq!(
                validate_checkpoint_v1_runtime_state(&recruited, &mut column_replay),
                Ok(())
            );

            let mut with_context = empty_runtime(policy);
            set_reachable_context(&mut with_context);
            assert_eq!(
                validate_checkpoint_v1_runtime_state(&with_context, &mut column_replay),
                Ok(())
            );
        }
    }

    #[test]
    fn checkpoint_state_validator_rejects_scalar_domain_corruption() {
        let runtime = empty_runtime(TemporalContextPolicy::BoundWinners);
        let baseline = checkpoint_v1_state_view(&runtime);
        let mut column_replay = [0u32; ACTIVE_COLUMNS];

        let mut mismatched_lengths = baseline;
        mismatched_lengths.previous_context_len = CONTEXT_CELLS_U8;
        assert_eq!(
            validate_view(&runtime, mismatched_lengths, &mut column_replay),
            Err(CheckpointV1StateError::ScalarInvariant)
        );

        let mut impossible_event = baseline;
        impossible_event.event_count = 1;
        assert_eq!(
            validate_view(&runtime, impossible_event, &mut column_replay),
            Err(CheckpointV1StateError::ScalarInvariant)
        );

        let mut growth_off_advanced = baseline;
        growth_off_advanced.config.plasticity_policy = TemporalPlasticityPolicy::GrowthOff;
        growth_off_advanced.learning_step = 1;
        assert_eq!(
            validate_view(&runtime, growth_off_advanced, &mut column_replay),
            Err(CheckpointV1StateError::ScalarInvariant)
        );

        let mut explicit_unpoisoned = baseline;
        explicit_unpoisoned.scientific_stop = Some(TemporalPoisonCode::Unpoisoned);
        assert_eq!(
            validate_view(&runtime, explicit_unpoisoned, &mut column_replay),
            Err(CheckpointV1StateError::ScalarInvariant)
        );
    }

    #[test]
    fn checkpoint_state_validator_rejects_cohort_cursor_and_csr_corruption() {
        let mut cursor = recruited_runtime(TemporalContextPolicy::BoundWinners);
        cursor.next_reserve_cell += 1;
        let mut column_replay = [0u32; ACTIVE_COLUMNS];
        assert_eq!(
            validate_checkpoint_v1_runtime_state(&cursor, &mut column_replay),
            Err(CheckpointV1StateError::LayoutInvariant)
        );

        let mut cohort = recruited_runtime(TemporalContextPolicy::BoundWinners);
        cohort.cohort_columns[0] = cohort.config.column_count;
        assert_eq!(
            validate_checkpoint_v1_runtime_state(&cohort, &mut column_replay),
            Err(CheckpointV1StateError::LayoutInvariant)
        );

        let mut csr = recruited_runtime(TemporalContextPolicy::BoundWinners);
        csr.column_offsets[1] -= 1;
        assert_eq!(
            validate_checkpoint_v1_runtime_state(&csr, &mut column_replay),
            Err(CheckpointV1StateError::ColumnCsrInvariant)
        );

        assert_eq!(
            validate_checkpoint_v1_runtime_state(&csr, &mut column_replay[..ACTIVE_COLUMNS - 1]),
            Err(CheckpointV1StateError::LayoutInvariant)
        );
    }

    #[test]
    fn checkpoint_state_validator_rederives_offsets_and_fixed_config_domains() {
        let runtime = empty_runtime(TemporalContextPolicy::BoundWinners);
        let baseline = checkpoint_v1_state_view(&runtime);
        let mut column_replay = [0u32; ACTIVE_COLUMNS];

        let mut corrupt_offset = baseline;
        corrupt_offset.proximal_offset += 1;
        assert_eq!(
            validate_view(&runtime, corrupt_offset, &mut column_replay),
            Err(CheckpointV1StateError::LayoutInvariant)
        );

        let mut unsupported_race_cap = baseline;
        unsupported_race_cap.config.novel_race_cap = 9;
        assert_eq!(
            validate_view(&runtime, unsupported_race_cap, &mut column_replay),
            Err(CheckpointV1StateError::LayoutInvariant)
        );

        let mut oversized_segment_capacity = baseline;
        oversized_segment_capacity.config.segment_capacity =
            u32::MAX / PAIRS_PER_SEGMENT as u32 + 1;
        assert_eq!(
            validate_view(&runtime, oversized_segment_capacity, &mut column_replay,),
            Err(CheckpointV1StateError::LayoutInvariant)
        );
    }

    #[test]
    fn checkpoint_state_validator_rejects_owner_row_source_and_load_corruption() {
        let bound = recruited_runtime(TemporalContextPolicy::BoundWinners);
        let state = checkpoint_v1_state_view(&bound);
        let mut column_replay = [0u32; ACTIVE_COLUMNS];

        assert_eq!(
            validate_checkpoint_v1_state(
                state,
                &mut column_replay,
                |segment_id| {
                    let mut owned = runtime_owned_row(&bound, segment_id)?;
                    if segment_id == 0 {
                        owned.owner = state.next_reserve_cell;
                    }
                    Some(owned)
                },
                |owner| bound.store.runtime_owner_load(owner),
            ),
            Err(CheckpointV1StateError::LearnedStateInvariant)
        );

        assert_eq!(
            validate_checkpoint_v1_state(
                state,
                &mut column_replay,
                |segment_id| {
                    let mut owned = runtime_owned_row(&bound, segment_id)?;
                    if segment_id == 0 {
                        owned.row.live_pair_count = 1;
                        owned.row.normalized_sources[0] = 0;
                        owned.row.permanence_ticks[0] = 1;
                        owned.row.normalized_sources[1] = 0;
                    }
                    Some(owned)
                },
                |owner| bound.store.runtime_owner_load(owner),
            ),
            Err(CheckpointV1StateError::LearnedStateInvariant)
        );

        assert_eq!(
            validate_checkpoint_v1_state(
                state,
                &mut column_replay,
                |segment_id| {
                    let mut owned = runtime_owned_row(&bound, segment_id)?;
                    if segment_id == 0 {
                        owned.row.live_pair_count = 1;
                        owned.row.normalized_sources[0] = state.next_reserve_cell;
                        owned.row.permanence_ticks[0] = 1;
                    }
                    Some(owned)
                },
                |owner| bound.store.runtime_owner_load(owner),
            ),
            Err(CheckpointV1StateError::LearnedStateInvariant)
        );

        assert_eq!(
            validate_checkpoint_v1_state(
                state,
                &mut column_replay,
                |segment_id| runtime_owned_row(&bound, segment_id),
                |owner| {
                    if owner == 0 {
                        Some(state.config.max_segments_per_cell + 1)
                    } else {
                        bound.store.runtime_owner_load(owner)
                    }
                },
            ),
            Err(CheckpointV1StateError::LearnedStateInvariant)
        );

        let generic = recruited_runtime(TemporalContextPolicy::GenericAfferents);
        let generic_state = checkpoint_v1_state_view(&generic);
        assert_eq!(
            validate_checkpoint_v1_state(
                generic_state,
                &mut column_replay,
                |segment_id| {
                    let mut owned = runtime_owned_row(&generic, segment_id)?;
                    if segment_id == 0 {
                        owned.row.live_pair_count = 1;
                        owned.row.normalized_sources[0] = generic_state.config.column_count;
                        owned.row.permanence_ticks[0] = 1;
                    }
                    Some(owned)
                },
                |owner| generic.store.runtime_owner_load(owner),
            ),
            Err(CheckpointV1StateError::LearnedStateInvariant)
        );
    }

    #[test]
    fn checkpoint_state_validator_rejects_chronology_without_recount_scratch() {
        let runtime = recruited_runtime(TemporalContextPolicy::BoundWinners);
        let state = checkpoint_v1_state_view(&runtime);
        let mut column_replay = [0u32; ACTIVE_COLUMNS];

        assert_eq!(
            validate_checkpoint_v1_state(
                state,
                &mut column_replay,
                |segment_id| runtime_owned_row(&runtime, segment_id),
                |owner| {
                    if owner == 0 {
                        Some(1)
                    } else {
                        runtime.store.runtime_owner_load(owner)
                    }
                },
            ),
            Err(CheckpointV1StateError::CohortChronologyInvariant)
        );

        assert_eq!(
            validate_checkpoint_v1_state(
                state,
                &mut column_replay,
                |segment_id| {
                    let mut owned = runtime_owned_row(&runtime, segment_id)?;
                    if owned.owner == 22 {
                        owned.owner = 0;
                    }
                    Some(owned)
                },
                |owner| {
                    if owner == 22 {
                        Some(0)
                    } else {
                        runtime.store.runtime_owner_load(owner)
                    }
                },
            ),
            Err(CheckpointV1StateError::CohortChronologyInvariant)
        );
    }

    #[test]
    fn checkpoint_state_validator_rejects_context_order_relay_and_token_column_corruption() {
        for policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            let mut runtime = empty_runtime(policy);
            set_reachable_context(&mut runtime);
            let mut column_replay = [0u32; ACTIVE_COLUMNS];

            let source_context = if policy == TemporalContextPolicy::BoundWinners {
                &mut runtime.previous_winner_context
            } else {
                &mut runtime.previous_context
            };
            source_context.swap(0, 1);
            source_context.swap(ACTIVE_COLUMNS, ACTIVE_COLUMNS + 1);
            if policy == TemporalContextPolicy::BoundWinners {
                runtime.previous_context = runtime.previous_winner_context;
            }
            assert_eq!(
                validate_checkpoint_v1_runtime_state(&runtime, &mut column_replay),
                Err(CheckpointV1StateError::ContextInvariant)
            );

            set_reachable_context(&mut runtime);
            runtime.previous_context[ACTIVE_COLUMNS + 3] += 1;
            if policy == TemporalContextPolicy::BoundWinners {
                runtime.previous_winner_context = runtime.previous_context;
            }
            assert_eq!(
                validate_checkpoint_v1_runtime_state(&runtime, &mut column_replay),
                Err(CheckpointV1StateError::ContextInvariant)
            );

            set_reachable_context(&mut runtime);
            runtime.previous_winner_context[4] = 2 * 5;
            runtime.previous_winner_context[ACTIVE_COLUMNS + 4] = runtime.relay_offset + 2 * 5;
            if policy == TemporalContextPolicy::BoundWinners {
                runtime.previous_context = runtime.previous_winner_context;
            }
            assert_eq!(
                validate_checkpoint_v1_runtime_state(&runtime, &mut column_replay),
                Err(CheckpointV1StateError::ContextInvariant)
            );
        }
    }

    #[test]
    fn prepared_checkpoint_export_borrows_an_empty_runtime_and_hashes_no_postings() {
        for policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            let runtime = empty_runtime(policy);

            let prepared = runtime
                .prepare_checkpoint_v1_export(authority())
                .expect("valid empty runtime must prepare");

            assert!(std::ptr::eq(prepared.runtime, &runtime));
            assert_eq!(
                prepared.identity.recognition_route(),
                TemporalRecognitionRoute::ExactGeneric
            );
            assert_eq!(prepared.layout.body_bytes, 0);
            assert_eq!(prepared.layout.total_bytes, 256);
            assert_eq!(
                prepared.scalars.to_le_bytes(),
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ]
            );
            assert_eq!(
                prepared.semantic_index_digest.raw_bytes(),
                &EMPTY_SEMANTIC_DIGEST
            );
        }
    }

    #[test]
    fn caller_owned_export_scratch_matches_allocating_proof_without_allocation_or_mutation() {
        const SENTINEL: u32 = 0xa5a5_5a5a;
        let runtime = full_section_generic_runtime();
        let before = checkpoint_v1_logical_state_snapshot(&runtime);
        let baseline = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("allocating baseline must prepare");
        let mut column_scratch = vec![SENTINEL; 13];
        let mut semantic_scratch = vec![SENTINEL; 6];
        let column_pointer = column_scratch.as_ptr();
        let column_capacity = column_scratch.capacity();
        let semantic_pointer = semantic_scratch.as_ptr();
        let semantic_capacity = semantic_scratch.capacity();

        let (prepared, allocations) = count_allocations(|| {
            let preparation = runtime
                .begin_checkpoint_v1_export(authority())
                .expect("begin must validate identity and codec without allocation");
            assert_eq!(preparation.column_scratch_len(), 11);
            let validated = preparation
                .validate_state(&mut column_scratch)
                .expect("caller column scratch must validate the full state");
            assert_eq!(validated.semantic_scratch_len(), 2);
            validated
                .finish(&mut semantic_scratch)
                .expect("caller semantic scratch must finish the proof")
        });

        assert_eq!(allocations, 0);
        assert_same_prepared_proof(&baseline, &prepared);
        assert_eq!(&column_scratch[..11], &[4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
        assert_eq!(&column_scratch[11..], &[SENTINEL; 2]);
        assert_eq!(&semantic_scratch[2..], &[SENTINEL; 4]);
        assert_eq!(column_scratch.as_ptr(), column_pointer);
        assert_eq!(column_scratch.capacity(), column_capacity);
        assert_eq!(semantic_scratch.as_ptr(), semantic_pointer);
        assert_eq!(semantic_scratch.capacity(), semantic_capacity);
        assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
    }

    #[test]
    fn caller_owned_export_rejects_short_column_scratch_unchanged_and_retries_exactly() {
        const SENTINEL: u32 = 0x55aa_33cc;
        let runtime = full_section_generic_runtime();
        let baseline = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("allocating baseline must prepare");
        let mut short = [SENTINEL; 10];

        let (result, allocations) = count_allocations(|| {
            runtime
                .begin_checkpoint_v1_export(authority())
                .expect("begin must remain allocation-free")
                .validate_state(&mut short)
        });

        assert_eq!(allocations, 0);
        assert!(matches!(result, Err(PrepareCheckpointV1ExportError::State)));
        assert_eq!(short, [SENTINEL; 10]);

        let mut exact_columns = [SENTINEL; 11];
        let mut exact_semantic = [SENTINEL; 2];
        let (retried, retry_allocations) = count_allocations(|| {
            runtime
                .begin_checkpoint_v1_export(authority())
                .expect("fresh retry must begin")
                .validate_state(&mut exact_columns)
                .expect("exact column scratch must validate")
                .finish(&mut exact_semantic)
                .expect("exact semantic scratch must finish")
        });
        assert_eq!(retry_allocations, 0);
        assert_same_prepared_proof(&baseline, &retried);
    }

    #[test]
    fn caller_owned_export_types_short_semantic_scratch_and_retries_oversized() {
        const SENTINEL: u32 = 0x5aa5_c33c;
        let runtime = full_section_generic_runtime();
        let baseline = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("allocating baseline must prepare");
        let mut columns = [SENTINEL; 11];
        let validated = runtime
            .begin_checkpoint_v1_export(authority())
            .expect("begin must succeed")
            .validate_state(&mut columns)
            .expect("column scratch must validate");
        let mut short = [SENTINEL; 1];

        let (result, allocations) = count_allocations(|| validated.finish(&mut short));

        assert_eq!(allocations, 0);
        assert!(matches!(
            result,
            Err(PrepareCheckpointV1ExportError::StoreIndex(
                TemporalStorePreflightError::OutputCapacity {
                    required: 2,
                    provided: 1,
                }
            ))
        ));
        assert_eq!(short, [SENTINEL; 1]);

        let mut retry_columns = [SENTINEL; 11];
        let mut oversized = vec![SENTINEL; 6];
        let pointer = oversized.as_ptr();
        let capacity = oversized.capacity();
        let (retried, retry_allocations) = count_allocations(|| {
            runtime
                .begin_checkpoint_v1_export(authority())
                .expect("fresh retry must begin")
                .validate_state(&mut retry_columns)
                .expect("fresh retry must validate")
                .finish(&mut oversized)
                .expect("oversized semantic scratch must finish")
        });
        assert_eq!(retry_allocations, 0);
        assert_same_prepared_proof(&baseline, &retried);
        assert_eq!(&oversized[2..], &[SENTINEL; 4]);
        assert_eq!(oversized.as_ptr(), pointer);
        assert_eq!(oversized.capacity(), capacity);
    }

    #[test]
    fn prepared_checkpoint_export_column_scratch_failure_is_typed_and_retryable() {
        let runtime = empty_runtime(TemporalContextPolicy::BoundWinners);
        let baseline = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("baseline must prepare");

        assert!(matches!(
            runtime.prepare_checkpoint_v1_export_with_failpoint(
                authority(),
                CheckpointV1AllocationSite::ColumnReplay,
            ),
            Err(PrepareCheckpointV1ExportError::RetryableAllocation(
                CheckpointV1AllocationSite::ColumnReplay,
            ))
        ));

        let retried = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("the identical retry must succeed");
        assert_same_prepared_proof(&baseline, &retried);
    }

    #[test]
    fn prepared_checkpoint_export_hashes_literal_postings_for_both_context_policies() {
        for policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            let runtime = semantic_runtime(policy);

            let prepared = runtime
                .prepare_checkpoint_v1_export(authority())
                .expect("valid nonempty runtime must prepare");

            assert_eq!(
                prepared.semantic_index_digest.raw_bytes(),
                &SEMANTIC_FIXTURE_DIGEST
            );
        }
    }

    #[test]
    fn prepared_checkpoint_export_semantic_scratch_failure_is_typed_and_retryable() {
        let runtime = semantic_runtime(TemporalContextPolicy::BoundWinners);
        let baseline = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("baseline must prepare");

        assert!(matches!(
            runtime.prepare_checkpoint_v1_export_with_failpoint(
                authority(),
                CheckpointV1AllocationSite::SemanticPostings,
            ),
            Err(PrepareCheckpointV1ExportError::RetryableAllocation(
                CheckpointV1AllocationSite::SemanticPostings,
            ))
        ));

        let retried = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("the identical retry must succeed");
        assert_same_prepared_proof(&baseline, &retried);
    }

    #[test]
    fn prepared_checkpoint_export_semantic_digest_is_history_independent() {
        let mut runtime = semantic_runtime(TemporalContextPolicy::BoundWinners);
        let original = runtime
            .store
            .runtime_segment_row(0)
            .expect("fixture row must exist");
        let empty = TemporalSegmentRow {
            normalized_sources: [u32::MAX; PAIRS_PER_SEGMENT],
            permanence_ticks: [0; PAIRS_PER_SEGMENT],
            live_pair_count: 0,
        };
        let before = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("baseline must prepare")
            .semantic_index_digest;

        runtime
            .store
            .preflight_runtime_canonical_rewrite(0, &empty)
            .expect("empty rewrite must preflight");
        runtime
            .store
            .commit_prevalidated_runtime_canonical_rewrite(0, &empty);
        runtime
            .store
            .preflight_runtime_canonical_rewrite(0, &original)
            .expect("restoring rewrite must preflight");
        runtime
            .store
            .commit_prevalidated_runtime_canonical_rewrite(0, &original);

        let after = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("history-only rewrite must still prepare")
            .semantic_index_digest;
        assert_eq!(after, before);
        assert_eq!(after.raw_bytes(), &SEMANTIC_FIXTURE_DIGEST);
    }

    #[test]
    fn prepared_checkpoint_export_does_not_consult_failpoint_for_zero_semantic_scratch() {
        let runtime = empty_runtime(TemporalContextPolicy::BoundWinners);

        let prepared = runtime
            .prepare_checkpoint_v1_export_with_failpoint(
                authority(),
                CheckpointV1AllocationSite::SemanticPostings,
            )
            .expect("zero semantic scratch must not reserve");

        assert_eq!(
            prepared.semantic_index_digest.raw_bytes(),
            &EMPTY_SEMANTIC_DIGEST
        );
    }

    #[test]
    fn prepared_checkpoint_export_validates_each_layer_before_the_next_allocation() {
        let mut invalid_identity = empty_runtime(TemporalContextPolicy::BoundWinners);
        invalid_identity.proximal_offset += 1;
        invalid_identity.scientific_stop = Some(TemporalPoisonCode::Unpoisoned);
        assert!(matches!(
            invalid_identity.prepare_checkpoint_v1_export_with_failpoint(
                authority(),
                CheckpointV1AllocationSite::ColumnReplay,
            ),
            Err(PrepareCheckpointV1ExportError::Identity(_))
        ));

        let mut invalid_scalar = empty_runtime(TemporalContextPolicy::BoundWinners);
        invalid_scalar.scientific_stop = Some(TemporalPoisonCode::Unpoisoned);
        assert!(matches!(
            invalid_scalar.prepare_checkpoint_v1_export_with_failpoint(
                authority(),
                CheckpointV1AllocationSite::ColumnReplay,
            ),
            Err(PrepareCheckpointV1ExportError::Codec(
                CheckpointV1CodecError::ExplicitUnpoisonedCode,
            ))
        ));

        let mut invalid_state = empty_runtime(TemporalContextPolicy::BoundWinners);
        invalid_state.column_offsets[1] -= 1;
        assert!(matches!(
            invalid_state.prepare_checkpoint_v1_export_with_failpoint(
                authority(),
                CheckpointV1AllocationSite::ColumnReplay,
            ),
            Err(PrepareCheckpointV1ExportError::RetryableAllocation(
                CheckpointV1AllocationSite::ColumnReplay,
            ))
        ));
        assert!(matches!(
            invalid_state.prepare_checkpoint_v1_export_with_failpoint(
                authority(),
                CheckpointV1AllocationSite::SemanticPostings,
            ),
            Err(PrepareCheckpointV1ExportError::State)
        ));
        assert!(matches!(
            invalid_state.prepare_checkpoint_v1_export(authority()),
            Err(PrepareCheckpointV1ExportError::State)
        ));
    }

    #[test]
    fn prepared_checkpoint_export_borrow_preserves_all_checkpoint_visible_state() {
        let runtime = semantic_runtime(TemporalContextPolicy::GenericAfferents);
        let before = checkpoint_v1_logical_state_snapshot(&runtime);

        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("immutable preparation must succeed");

        assert!(std::ptr::eq(prepared.runtime, &runtime));
        assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
    }

    #[test]
    fn checkpoint_encoder_writes_exact_empty_checkpoint_for_both_policies() {
        for policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            let runtime = empty_runtime(policy);
            let prepared = runtime
                .prepare_checkpoint_v1_export(authority())
                .expect("empty runtime must prepare");
            let mut sink = Cursor::new(Vec::new());

            prepared
                .encode_into(&mut sink)
                .expect("empty checkpoint must encode");

            assert_eq!(sink.position(), u64::from(CHECKPOINT_V1_HEADER_BYTES));
            assert_eq!(
                sink.get_ref().len(),
                usize::from(CHECKPOINT_V1_HEADER_BYTES)
            );
            assert_eq!(&sink.get_ref()[0..8], &CHECKPOINT_V1_MAGIC);
            assert_eq!(&sink.get_ref()[128..160], &EMPTY_SEMANTIC_DIGEST);
            assert_eq!(&sink.get_ref()[160..192], &EMPTY_SEMANTIC_DIGEST);
        }
    }

    #[test]
    fn canonical_empty_checkpoint_is_allocation_free_importer_input_for_every_policy() {
        for context_policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            for plasticity_policy in [
                TemporalPlasticityPolicy::GrowthOff,
                TemporalPlasticityPolicy::Plastic,
            ] {
                let runtime = empty_runtime_with_plasticity(context_policy, plasticity_policy);
                let expected = encoded_checkpoint(&runtime);
                let factory = factory_for_runtime(&runtime);
                let (checkpoint, allocations) =
                    count_allocations(|| canonical_empty_checkpoint_v1(&factory));
                let checkpoint = checkpoint.expect("empty checkpoint encoding must succeed");

                assert_eq!(allocations, 0);
                assert_eq!(checkpoint.len(), 256);
                assert_eq!(checkpoint.as_slice(), expected);

                let imported = import_canonical_empty_checkpoint_v1(factory)
                    .expect("canonical empty checkpoint must import");
                assert_eq!(encoded_checkpoint(&imported), expected);
                assert_eq!(imported.event_count, 0);
                assert_eq!(imported.learning_step, 0);
                assert_eq!(imported.scientific_stop, None);
            }
        }
    }

    #[test]
    fn canonical_empty_checkpoint_matches_the_independent_cpython_oracle() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let checkpoint = canonical_empty_checkpoint_v1(&factory)
            .expect("independent empty checkpoint must encode");
        let expected = cpython_empty_full_section_generic_checkpoint_v1();

        assert_eq!(checkpoint.as_slice(), expected);
        assert_eq!(
            Sha256::digest(checkpoint).as_slice(),
            hex_bytes("6aedd76db5a1bb53b33c26b5c7bd6aa117d9eb099fe2ab59b52b9a8ee01e6ee0")
        );
    }

    #[test]
    fn checkpoint_encoder_matches_independent_header_body_and_digest_vectors() {
        for (policy, header_hex, full_digest_hex) in [
            (
                TemporalContextPolicy::BoundWinners,
                SEMANTIC_FIXTURE_BOUND_HEADER_HEX,
                "64a4f703fc32423f2a26635eee1ae8afd284a9c3bac1e83698b1cb27ccab44fb",
            ),
            (
                TemporalContextPolicy::GenericAfferents,
                SEMANTIC_FIXTURE_GENERIC_HEADER_HEX,
                "6fae68243c443547eeb7f5c06119961adaa265ad7617c411eb0f735ec0264584",
            ),
        ] {
            let runtime = semantic_runtime(policy);
            let prepared = runtime
                .prepare_checkpoint_v1_export(authority())
                .expect("semantic fixture must prepare");
            let mut sink = Cursor::new(Vec::new());

            prepared
                .encode_into(&mut sink)
                .expect("semantic fixture must encode");

            let bytes = sink.get_ref();
            assert_eq!(sink.position(), 384);
            assert_eq!(bytes.len(), 384);
            assert_eq!(&bytes[..256], hex_bytes(header_hex));
            assert_eq!(&bytes[256..], hex_bytes(SEMANTIC_FIXTURE_BODY_HEX));
            assert_eq!(
                &bytes[128..160],
                hex_bytes("71b9e70f4a7cf28df3e381695159e7f84b3fe1990bd9dadf75bfcc5de9c34279")
            );
            assert_eq!(&bytes[160..192], &SEMANTIC_FIXTURE_DIGEST);
            assert_eq!(
                &bytes[192..224],
                hex_bytes("0200000000000000160000000000000000000000000000000000000000000000")
            );
            assert_eq!(
                &bytes[224..256],
                hex_bytes("ef9373a4c6e5435b1d26aac601496812015351e33aefd3bbe13e1e45aedb3d9b")
            );
            assert_eq!(Sha256::digest(bytes).as_slice(), hex_bytes(full_digest_hex));
        }
    }

    #[test]
    fn checkpoint_encoder_matches_independent_full_section_generic_vector() {
        let runtime = full_section_generic_runtime();
        let before = checkpoint_v1_logical_state_snapshot(&runtime);
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("full-section generic fixture must prepare");
        let expected_header = hex_bytes(FULL_SECTION_GENERIC_HEADER_HEX);
        let expected_body = hex_bytes(FULL_SECTION_GENERIC_BODY_HEX);
        let mut expected_file = expected_header.clone();
        expected_file.extend_from_slice(&expected_body);
        let mut sink = Cursor::new(Vec::new());

        prepared
            .encode_into(&mut sink)
            .expect("full-section generic fixture must encode");

        let bytes = sink.get_ref();
        assert_eq!(prepared.identity.circuit_preimage_bytes(), 2_128);
        assert_eq!(prepared.layout.body_bytes, 568);
        assert_eq!(prepared.layout.total_bytes, 824);
        assert_eq!(sink.position(), 824);
        assert_eq!(expected_header.len(), 256);
        assert_eq!(expected_body.len(), 568);
        assert_eq!(bytes, &expected_file);
        assert_eq!(&bytes[..256], expected_header);
        assert_eq!(&bytes[256..], expected_body);
        assert_eq!(
            &bytes[96..128],
            hex_bytes("ca61287978e0a7c3b76ce050c9cf4961d00c441a9036eaecbe645ad628088c4f")
        );
        assert_eq!(
            Sha256::digest(&bytes[256..]).as_slice(),
            hex_bytes("db5a0520fa90d4bc79531d0b1a77ecba759c42e6d0be9567004753e6e260b669")
        );
        assert_eq!(
            &bytes[128..160],
            hex_bytes("db5a0520fa90d4bc79531d0b1a77ecba759c42e6d0be9567004753e6e260b669")
        );
        assert_eq!(
            &bytes[160..192],
            hex_bytes("dc6222193dfca80b9e29b83e57ec1581d6ef6d94a339046249b8b9a7c8c0a1a5")
        );
        assert_eq!(
            &bytes[192..224],
            hex_bytes("06000000020000001a0000000100000000000000000000000000000016160000")
        );
        assert_eq!(
            &bytes[224..256],
            hex_bytes("ba1e27b132140f8dd0f8eabc3eb75a591ed99feb2a2e8cba5d02fc3ac90c2f38")
        );
        assert_eq!(
            Sha256::digest(bytes).as_slice(),
            hex_bytes("d64cf550c30b6bcf7b499d39cbdc9c7bafeaef6a2e23b795ffe512e37117eb80")
        );
        assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
    }

    #[test]
    fn prepared_checkpoint_compares_independent_canonical_bytes_without_allocation_or_mutation() {
        let runtime = full_section_generic_runtime();
        let before_runtime = checkpoint_v1_logical_state_snapshot(&runtime);
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("full-section fixture must prepare");
        let expected = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        let before_expected = expected.clone();

        let (result, allocations) =
            count_allocations(|| prepared.compare_canonical_bytes(&expected));

        assert_eq!(allocations, 0);
        assert!(result.is_ok());
        assert_eq!(expected, before_expected);
        assert_eq!(
            checkpoint_v1_logical_state_snapshot(&runtime),
            before_runtime
        );
    }

    #[test]
    fn prepared_checkpoint_comparison_types_header_body_and_length_mismatches_exactly() {
        let runtime = full_section_generic_runtime();
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("full-section fixture must prepare");
        let canonical = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        assert_eq!(canonical.len(), 824);

        let mut final_header_mismatch = canonical.clone();
        final_header_mismatch[0] = 0;
        let (result, allocations) =
            count_allocations(|| prepared.compare_canonical_bytes(&final_header_mismatch));
        assert_eq!(allocations, 0);
        assert!(matches!(
            result,
            Err(CompareCheckpointV1CanonicalBytesError::Mismatch(
                CheckpointV1CanonicalMismatch::ByteMismatch {
                    offset: 0,
                    expected: 0,
                    actual: 0x52,
                }
            ))
        ));

        let mut first_body_mismatch = canonical.clone();
        first_body_mismatch[256] = 0xa5;
        let (result, allocations) =
            count_allocations(|| prepared.compare_canonical_bytes(&first_body_mismatch));
        assert_eq!(allocations, 0);
        assert!(matches!(
            result,
            Err(CompareCheckpointV1CanonicalBytesError::Mismatch(
                CheckpointV1CanonicalMismatch::ByteMismatch {
                    offset: 256,
                    expected: 0xa5,
                    actual: 0,
                }
            ))
        ));

        let mut simultaneous_header_and_body_mismatch = canonical.clone();
        simultaneous_header_and_body_mismatch[0] = 0;
        simultaneous_header_and_body_mismatch[256] = 0xa5;
        let (result, allocations) = count_allocations(|| {
            prepared.compare_canonical_bytes(&simultaneous_header_and_body_mismatch)
        });
        assert_eq!(allocations, 0);
        assert!(matches!(
            result,
            Err(CompareCheckpointV1CanonicalBytesError::Mismatch(
                CheckpointV1CanonicalMismatch::ByteMismatch {
                    offset: 0,
                    expected: 0,
                    actual: 0x52,
                }
            ))
        ));

        let (result, allocations) =
            count_allocations(|| prepared.compare_canonical_bytes(&canonical[..823]));
        assert_eq!(allocations, 0);
        assert!(matches!(
            result,
            Err(CompareCheckpointV1CanonicalBytesError::Mismatch(
                CheckpointV1CanonicalMismatch::ExpectedEnded {
                    offset: 823,
                    actual: 0,
                }
            ))
        ));

        let (result, allocations) =
            count_allocations(|| prepared.compare_canonical_bytes(&canonical[..32]));
        assert_eq!(allocations, 0);
        assert!(matches!(
            result,
            Err(CompareCheckpointV1CanonicalBytesError::Mismatch(
                CheckpointV1CanonicalMismatch::ExpectedEnded {
                    offset: 32,
                    actual: 0xa5,
                }
            ))
        ));

        let mut trailing = canonical;
        trailing.push(0xa5);
        let (result, allocations) =
            count_allocations(|| prepared.compare_canonical_bytes(&trailing));
        assert_eq!(allocations, 0);
        assert!(matches!(
            result,
            Err(CompareCheckpointV1CanonicalBytesError::Mismatch(
                CheckpointV1CanonicalMismatch::ExpectedTrailing {
                    offset: 824,
                    expected: 0xa5,
                }
            ))
        ));
    }

    #[test]
    fn prepared_checkpoint_comparison_accepts_actually_unaligned_expected_bytes() {
        let runtime = full_section_generic_runtime();
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("full-section fixture must prepare");
        let canonical = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        let alignment = std::mem::align_of::<u32>();
        let mut backing = vec![0xcc; canonical.len() + alignment];
        let start = (0..alignment)
            .find(|&start| (backing.as_ptr().wrapping_add(start + 256) as usize) % alignment != 0)
            .expect("one short prefix must make the body unaligned");
        let end = start + canonical.len();
        backing[start..end].copy_from_slice(&canonical);
        let before = backing.clone();
        let expected = &backing[start..end];
        assert_ne!(expected.as_ptr().wrapping_add(256) as usize % alignment, 0);

        let (result, allocations) =
            count_allocations(|| prepared.compare_canonical_bytes(expected));

        assert_eq!(allocations, 0);
        assert!(result.is_ok());
        assert_eq!(backing, before);
    }

    #[test]
    fn prepared_checkpoint_comparison_keeps_proof_errors_distinct_from_byte_mismatches() {
        let runtime = full_section_generic_runtime();
        let mut prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("full-section fixture must prepare");
        let mut expected = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        expected[256] = 0xa5;
        let before = expected.clone();
        prepared.scalars.event_count += 1;

        let (result, allocations) =
            count_allocations(|| prepared.compare_canonical_bytes(&expected));

        assert_eq!(allocations, 0);
        assert!(matches!(
            result,
            Err(CompareCheckpointV1CanonicalBytesError::Encode(
                CheckpointV1EncodeError::PreparedScalarsMismatch
            ))
        ));
        assert_eq!(expected, before);
    }

    #[test]
    fn pass1_borrows_the_exact_six_full_fixture_sections() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let mut checkpoint = hex_bytes(FULL_SECTION_GENERIC_HEADER_HEX);
        checkpoint.extend_from_slice(&hex_bytes(FULL_SECTION_GENERIC_BODY_HEX));

        let validated = validate_checkpoint_v1_pass1(&checkpoint, &factory)
            .expect("the literal full-section checkpoint must pass envelope validation");
        let base = checkpoint.as_ptr() as usize;
        let sections = validated.body.sections();
        let expected_boundaries = [256usize, 264, 352, 440, 464, 704, 824];

        assert_eq!(validated.checkpoint.as_ptr(), checkpoint.as_ptr());
        assert_eq!(validated.checkpoint.len(), checkpoint.len());
        assert_eq!(validated.layout.body_bytes, 568);
        assert_eq!(validated.layout.total_bytes, 824);
        assert_eq!(validated.scalars.segment_count, 6);
        assert_eq!(validated.scalars.cohort_count, 2);
        assert_eq!(
            validated.allocation_plan,
            CheckpointV1ImportAllocationPlan {
                committed_owner_count: 26,
                column_count: 11,
                owner_capacity: 26,
                segment_capacity: 6,
                posting_capacity: 60,
                source_count: 11,
                cohort_capacity: 2,
                prospective_column_occupancy_limit: 6,
                active_segment_capacity: 6,
                semantic_posting_scratch: 6,
            }
        );
        assert_eq!(
            validated.semantic_index_digest.raw_bytes().as_slice(),
            hex_bytes("dc6222193dfca80b9e29b83e57ec1581d6ef6d94a339046249b8b9a7c8c0a1a5")
                .as_slice()
        );
        assert_eq!(validated.poison, None);
        for (section, expected) in sections.iter().zip(expected_boundaries.windows(2)) {
            assert_eq!(section.as_ptr() as usize - base, expected[0]);
            assert_eq!(section.len(), expected[1] - expected[0]);
        }
        let cohorts = validated.body.cohort_columns();
        let prior = validated.body.prior_context();
        let winner = validated.body.winner_context();
        let owners = validated.body.segment_owners();
        let sources = validated.body.segment_sources();
        let ticks = validated.body.segment_ticks();
        assert_eq!(
            (cohorts.len(), cohorts.get(0), cohorts.get(1)),
            (2, Some(0), Some(1))
        );
        assert_eq!(
            (prior.len(), prior.get(0), prior.get(21)),
            (22, Some(63), Some(84))
        );
        assert_eq!(
            (winner.len(), winner.get(0), winner.get(10), winner.get(11)),
            (22, Some(0), Some(20), Some(26))
        );
        assert_eq!(
            std::array::from_fn::<Option<u32>, 6, _>(|index| owners.get(index)),
            [Some(0), Some(1), Some(22), Some(2), Some(3), Some(24)]
        );
        assert_eq!(
            (sources.len(), sources.get(0), sources.get(1)),
            (60, Some(0), Some(2))
        );
        assert_eq!(
            (ticks.len(), ticks.get(0), ticks.get(1)),
            (60, Some(250), Some(251))
        );
        assert_eq!(sources.get(sources.len()), None);
        assert_eq!(ticks.get(ticks.len()), None);
    }

    #[test]
    fn pass2_validates_the_literal_full_fixture_and_retains_exact_assembly_occupancies() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let checkpoint = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        let checkpoint_pointer = checkpoint.as_ptr();
        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory)
            .expect("the literal full fixture must pass envelope validation");

        let validated = validate_checkpoint_v1_pass2(pass1, &factory)
            .expect("the canonical literal full fixture must pass reachability validation");
        let ValidatedCheckpointV1Pass2 {
            pass1,
            column_occupancies,
            max_column_occupancy,
        } = validated;

        assert_eq!(pass1.checkpoint.as_ptr(), checkpoint_pointer);
        assert_eq!(column_occupancies, [4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
        assert_eq!(max_column_occupancy, 4);
        assert_eq!(
            pass1.semantic_index_digest.raw_bytes().as_slice(),
            hex_bytes("dc6222193dfca80b9e29b83e57ec1581d6ef6d94a339046249b8b9a7c8c0a1a5")
                .as_slice()
        );
        assert_eq!(*factory.identity(), pass1.factory_identity);
    }

    #[test]
    fn pass2_rejects_factory_and_scalar_mismatches_before_any_scratch_reserve() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let foreign_factory = factory_for_runtime_with_authority(
            &runtime,
            ValidatedCheckpointAuthority::from_raw_sha256([0x5a; 32], [0x6b; 32]),
        );
        let checkpoint = encoded_checkpoint(&runtime);
        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory)
            .expect("the generated fixture must pass envelope validation");
        let mut allocation_calls = 0usize;

        let error = match validate_checkpoint_v1_pass2_inner(pass1, &foreign_factory, |_, _| {
            allocation_calls += 1;
            false
        }) {
            Ok(_) => panic!("a proof from another factory must not pass"),
            Err(error) => error,
        };
        assert_eq!(error, CheckpointV1Pass2Error::FactoryIdentityMismatch);
        assert_eq!(allocation_calls, 0);

        let mut bad_cursor = checkpoint;
        bad_cursor[200..204].copy_from_slice(&25u32.to_le_bytes());
        refresh_logical_state_digest(&mut bad_cursor);
        let pass1 = validate_checkpoint_v1_pass1(&bad_cursor, &factory)
            .expect("Pass 1 intentionally defers exact cursor reconstruction");
        let mut allocation_calls = 0usize;
        let error = match validate_checkpoint_v1_pass2_inner(pass1, &factory, |_, _| {
            allocation_calls += 1;
            false
        }) {
            Ok(_) => panic!("a noncanonical reserve cursor must not pass"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            CheckpointV1Pass2Error::State(CheckpointV1StateError::LayoutInvariant)
        );
        assert_eq!(allocation_calls, 0);
    }

    #[test]
    fn pass2_scratch_reserves_are_typed_ordered_and_retryable() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let checkpoint = encoded_checkpoint(&runtime);

        for (failed_site, expected_calls) in [
            (
                CheckpointV1ImportAllocationSite::Pass2OwnerLoads,
                [
                    Some((CheckpointV1ImportAllocationSite::Pass2OwnerLoads, 26)),
                    None,
                ],
            ),
            (
                CheckpointV1ImportAllocationSite::Pass2ColumnReplay,
                [
                    Some((CheckpointV1ImportAllocationSite::Pass2OwnerLoads, 26)),
                    Some((CheckpointV1ImportAllocationSite::Pass2ColumnReplay, 11)),
                ],
            ),
        ] {
            let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory)
                .expect("the generated fixture must pass envelope validation");
            let mut calls = [None; 2];
            let mut call_count = 0usize;
            let error =
                match validate_checkpoint_v1_pass2_inner(pass1, &factory, |site, element_count| {
                    calls[call_count] = Some((site, element_count));
                    call_count += 1;
                    site == failed_site
                }) {
                    Ok(_) => panic!("the selected reserve must fail"),
                    Err(error) => error,
                };

            assert_eq!(
                error,
                CheckpointV1Pass2Error::RetryableAllocation(failed_site)
            );
            assert_eq!(calls, expected_calls);

            let retry = validate_checkpoint_v1_pass2(
                validate_checkpoint_v1_pass1(&checkpoint, &factory)
                    .expect("the identical retry must pass envelope validation"),
                &factory,
            )
            .expect("the identical retry must pass canonical validation");
            assert_eq!(retry.max_column_occupancy, 4);
        }

        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory)
            .expect("the generated fixture must pass envelope validation");
        let mut calls = [None; 2];
        let mut call_count = 0usize;
        let validated =
            validate_checkpoint_v1_pass2_inner(pass1, &factory, |site, element_count| {
                calls[call_count] = Some((site, element_count));
                call_count += 1;
                false
            })
            .expect("successful validation must reserve both planned scratch arrays");
        assert_eq!(
            calls,
            [
                Some((CheckpointV1ImportAllocationSite::Pass2OwnerLoads, 26)),
                Some((CheckpointV1ImportAllocationSite::Pass2ColumnReplay, 11)),
            ]
        );
        assert_eq!(validated.max_column_occupancy, 4);

        let large = large_streaming_runtime();
        let large_factory = factory_for_runtime(&large);
        let large_checkpoint = encoded_checkpoint(&large);
        let pass1 = validate_checkpoint_v1_pass1(&large_checkpoint, &large_factory)
            .expect("the 257-row fixture must pass envelope validation");
        let mut calls = [None; 2];
        let mut call_count = 0usize;
        validate_checkpoint_v1_pass2_inner(pass1, &large_factory, |site, element_count| {
            calls[call_count] = Some((site, element_count));
            call_count += 1;
            false
        })
        .expect("the 257-row fixture must not require segment-sized Pass 2 scratch");
        assert_eq!(
            calls,
            [
                Some((CheckpointV1ImportAllocationSite::Pass2OwnerLoads, 22)),
                Some((CheckpointV1ImportAllocationSite::Pass2ColumnReplay, 11)),
            ]
        );

        let mut corrupt_before_traversal = checkpoint.clone();
        let starts = checkpoint_v1_test_section_starts(&corrupt_before_traversal);
        write_checkpoint_v1_u32(
            &mut corrupt_before_traversal,
            starts.segment_owners,
            0,
            u32::MAX,
        );
        refresh_body_and_logical_state_digests(&mut corrupt_before_traversal);
        let pass1 = validate_checkpoint_v1_pass1(&corrupt_before_traversal, &factory)
            .expect("row corruption belongs to Pass 2");
        assert!(matches!(
            validate_checkpoint_v1_pass2_inner(pass1, &factory, |site, _| {
                site == CheckpointV1ImportAllocationSite::Pass2ColumnReplay
            }),
            Err(CheckpointV1Pass2Error::RetryableAllocation(
                CheckpointV1ImportAllocationSite::Pass2ColumnReplay
            ))
        ));
    }

    #[test]
    fn pass2_accepts_all_context_and_plasticity_policies_empty_full_unaligned_and_terminal() {
        for context_policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            for plasticity_policy in [
                TemporalPlasticityPolicy::Plastic,
                TemporalPlasticityPolicy::GrowthOff,
            ] {
                let runtime = full_policy_runtime(context_policy, plasticity_policy);
                let factory = factory_for_runtime(&runtime);
                let checkpoint = encoded_checkpoint(&runtime);
                let validated = validate_checkpoint_v1_pass2(
                    validate_checkpoint_v1_pass1(&checkpoint, &factory)
                        .expect("full policy fixture must pass envelope validation"),
                    &factory,
                )
                .expect("full policy fixture must pass canonical validation");
                assert_eq!(
                    validated.column_occupancies,
                    [4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                );
                assert_eq!(validated.max_column_occupancy, 4);

                let empty = empty_runtime_with_plasticity(context_policy, plasticity_policy);
                let empty_factory = factory_for_runtime(&empty);
                let empty_checkpoint = encoded_checkpoint(&empty);
                let validated = validate_checkpoint_v1_pass2(
                    validate_checkpoint_v1_pass1(&empty_checkpoint, &empty_factory)
                        .expect("empty policy fixture must pass envelope validation"),
                    &empty_factory,
                )
                .expect("empty policy fixture must pass canonical validation");
                assert_eq!(validated.column_occupancies, [2; ACTIVE_COLUMNS]);
                assert_eq!(validated.max_column_occupancy, 2);
            }
        }

        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let checkpoint = encoded_checkpoint(&runtime);
        let alignment = std::mem::align_of::<u32>();
        let mut unaligned = Vec::with_capacity(checkpoint.len() + alignment);
        let padding = (1..=alignment)
            .find(|padding| (unaligned.as_ptr() as usize + padding) % alignment != 0)
            .expect("at least one padding must leave a u32-misaligned address");
        unaligned.resize(padding, 0xff);
        unaligned.extend_from_slice(&checkpoint);
        let unaligned_checkpoint = &unaligned[padding..];
        assert_ne!(unaligned_checkpoint.as_ptr() as usize % alignment, 0);
        let validated = validate_checkpoint_v1_pass2(
            validate_checkpoint_v1_pass1(unaligned_checkpoint, &factory)
                .expect("unaligned backing must pass envelope validation"),
            &factory,
        )
        .expect("unaligned backing must pass direct SoA validation");
        assert_eq!(
            validated.pass1.checkpoint.as_ptr(),
            unaligned_checkpoint.as_ptr()
        );

        let mut plastic = empty_runtime(TemporalContextPolicy::BoundWinners);
        plastic.learning_step = u64::MAX;
        plastic.scientific_stop = Some(TemporalPoisonCode::PreallocatedCapacityInvariant);
        let plastic_factory = factory_for_runtime(&plastic);
        let plastic_checkpoint = encoded_checkpoint(&plastic);
        let validated = validate_checkpoint_v1_pass2(
            validate_checkpoint_v1_pass1(&plastic_checkpoint, &plastic_factory)
                .expect("terminal Plastic state must pass envelope validation"),
            &plastic_factory,
        )
        .expect("Plastic permits a saturated learning counter and stable poison");
        assert_eq!(
            validated.pass1.poison,
            Some(TemporalPoisonCode::PreallocatedCapacityInvariant)
        );
    }

    #[test]
    fn pass2_rejects_all_deferred_shape_rules_before_scratch_allocation() {
        fn assert_preallocation_rejection(
            checkpoint: &[u8],
            factory: &ValidatedCheckpointV1RuntimeFactory,
            expected: CheckpointV1StateError,
        ) {
            let pass1 = validate_checkpoint_v1_pass1(checkpoint, factory)
                .expect("the cross-field corruption must pass envelope validation");
            let mut allocation_calls = 0usize;
            let error = match validate_checkpoint_v1_pass2_inner(pass1, factory, |_, _| {
                allocation_calls += 1;
                false
            }) {
                Ok(_) => panic!("the deferred shape corruption must not pass"),
                Err(error) => error,
            };
            assert_eq!(error, CheckpointV1Pass2Error::State(expected));
            assert_eq!(allocation_calls, 0);
        }

        let full = full_section_generic_runtime();
        let full_factory = factory_for_runtime(&full);
        let baseline = encoded_checkpoint(&full);

        let mut zero_event_with_context = baseline.clone();
        zero_event_with_context[204..212].copy_from_slice(&0u64.to_le_bytes());
        refresh_logical_state_digest(&mut zero_event_with_context);
        assert_preallocation_rejection(
            &zero_event_with_context,
            &full_factory,
            CheckpointV1StateError::ScalarInvariant,
        );

        for field in [
            CheckpointV1ContextLengthField::Prior,
            CheckpointV1ContextLengthField::Winner,
        ] {
            let mut mismatched_context = baseline.clone();
            remove_checkpoint_v1_context_section(&mut mismatched_context, field);
            assert_preallocation_rejection(
                &mismatched_context,
                &full_factory,
                CheckpointV1StateError::ScalarInvariant,
            );
        }

        let mut bad_cohort = baseline.clone();
        let starts = checkpoint_v1_test_section_starts(&bad_cohort);
        write_checkpoint_v1_u32(
            &mut bad_cohort,
            starts.cohort_columns,
            0,
            full.config.column_count,
        );
        refresh_body_and_logical_state_digests(&mut bad_cohort);
        assert_preallocation_rejection(
            &bad_cohort,
            &full_factory,
            CheckpointV1StateError::LayoutInvariant,
        );

        let empty = empty_runtime(TemporalContextPolicy::BoundWinners);
        let empty_factory = factory_for_runtime(&empty);
        let mut event_without_context = encoded_checkpoint(&empty);
        event_without_context[204..212].copy_from_slice(&1u64.to_le_bytes());
        refresh_logical_state_digest(&mut event_without_context);
        assert_preallocation_rejection(
            &event_without_context,
            &empty_factory,
            CheckpointV1StateError::ScalarInvariant,
        );

        let growth_off = nonuniform_generic_route_zero_runtime();
        let growth_off_factory = factory_for_runtime(&growth_off);
        let mut advanced_growth_off = encoded_checkpoint(&growth_off);
        advanced_growth_off[212..220].copy_from_slice(&1u64.to_le_bytes());
        refresh_logical_state_digest(&mut advanced_growth_off);
        assert_preallocation_rejection(
            &advanced_growth_off,
            &growth_off_factory,
            CheckpointV1StateError::ScalarInvariant,
        );
    }

    #[test]
    fn pass2_direct_soa_rows_reject_every_canonical_grammar_family() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = encoded_checkpoint(&runtime);
        let starts = checkpoint_v1_test_section_starts(&baseline);
        let learned = CheckpointV1Pass2Error::State(CheckpointV1StateError::LearnedStateInvariant);

        for owner in [runtime.next_reserve_cell, u32::MAX] {
            let mut checkpoint = baseline.clone();
            write_checkpoint_v1_u32(&mut checkpoint, starts.segment_owners, 0, owner);
            refresh_body_and_logical_state_digests(&mut checkpoint);
            assert_eq!(pass2_error(&checkpoint, &factory), learned);
        }

        let mut overloaded = baseline.clone();
        write_checkpoint_v1_u32(&mut overloaded, starts.segment_owners, 1, 0);
        refresh_body_and_logical_state_digests(&mut overloaded);
        assert_eq!(pass2_error(&overloaded, &factory), learned);

        let mut sentinel_gap = baseline.clone();
        write_checkpoint_v1_u32(&mut sentinel_gap, starts.segment_sources, 0, u32::MAX);
        write_checkpoint_v1_u16(&mut sentinel_gap, starts.segment_ticks, 0, 0);
        refresh_body_and_logical_state_digests(&mut sentinel_gap);
        assert_eq!(pass2_error(&sentinel_gap, &factory), learned);

        let mut nonzero_sentinel_tick = baseline.clone();
        write_checkpoint_v1_u16(&mut nonzero_sentinel_tick, starts.segment_ticks, 2, 1);
        refresh_body_and_logical_state_digests(&mut nonzero_sentinel_tick);
        assert_eq!(pass2_error(&nonzero_sentinel_tick, &factory), learned);

        for ticks in [0u16, PERMANENCE_MAX_TICKS + 1] {
            let mut checkpoint = baseline.clone();
            write_checkpoint_v1_u16(&mut checkpoint, starts.segment_ticks, 0, ticks);
            refresh_body_and_logical_state_digests(&mut checkpoint);
            assert_eq!(pass2_error(&checkpoint, &factory), learned);
        }

        for (first, second) in [(0u32, 0u32), (3, 2)] {
            let mut checkpoint = baseline.clone();
            write_checkpoint_v1_u32(&mut checkpoint, starts.segment_sources, 0, first);
            write_checkpoint_v1_u32(&mut checkpoint, starts.segment_sources, 1, second);
            refresh_body_and_logical_state_digests(&mut checkpoint);
            assert_eq!(pass2_error(&checkpoint, &factory), learned);
        }

        let mut generic_out_of_domain = baseline.clone();
        write_checkpoint_v1_u32(
            &mut generic_out_of_domain,
            starts.segment_sources,
            1,
            runtime.config.column_count,
        );
        refresh_body_and_logical_state_digests(&mut generic_out_of_domain);
        assert_eq!(pass2_error(&generic_out_of_domain, &factory), learned);

        let mut full_row = baseline.clone();
        for slot in 0..PAIRS_PER_SEGMENT {
            write_checkpoint_v1_u32(&mut full_row, starts.segment_sources, slot, slot as u32);
            write_checkpoint_v1_u16(&mut full_row, starts.segment_ticks, slot, slot as u16 + 1);
        }
        refresh_body_and_logical_state_digests(&mut full_row);
        full_row[160] ^= 0x80;
        let mut mutated_semantic_index_digest = [0u8; 32];
        mutated_semantic_index_digest.copy_from_slice(&full_row[160..192]);
        let validated = validate_checkpoint_v1_pass2(
            validate_checkpoint_v1_pass1(&full_row, &factory)
                .expect("semantic digest remains opaque through Pass 2"),
            &factory,
        )
        .expect("a full ten-pair canonical row must pass");
        assert_eq!(validated.max_column_occupancy, 4);
        assert_eq!(
            validated.pass1.semantic_index_digest.raw_bytes(),
            &mutated_semantic_index_digest
        );
    }

    #[test]
    fn pass2_owner_load_replay_accepts_255_and_rejects_the_256th_row() {
        let runtime = large_streaming_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = encoded_checkpoint(&runtime);
        let starts = checkpoint_v1_test_section_starts(&baseline);

        let mut exact_limit = baseline.clone();
        for segment_index in 0..255 {
            write_checkpoint_v1_u32(&mut exact_limit, starts.segment_owners, segment_index, 0);
        }
        for segment_index in 255..257 {
            write_checkpoint_v1_u32(&mut exact_limit, starts.segment_owners, segment_index, 1);
        }
        refresh_body_and_logical_state_digests(&mut exact_limit);
        validate_checkpoint_v1_pass2(
            validate_checkpoint_v1_pass1(&exact_limit, &factory)
                .expect("the exact u8 load limit must pass envelope validation"),
            &factory,
        )
        .expect("255 allocated rows for one owner are representable");

        let mut overflow = baseline;
        for segment_index in 0..256 {
            write_checkpoint_v1_u32(&mut overflow, starts.segment_owners, segment_index, 0);
        }
        write_checkpoint_v1_u32(&mut overflow, starts.segment_owners, 256, 1);
        refresh_body_and_logical_state_digests(&mut overflow);
        assert_eq!(
            pass2_error(&overflow, &factory),
            CheckpointV1Pass2Error::State(CheckpointV1StateError::LearnedStateInvariant)
        );
    }

    #[test]
    fn pass2_source_domains_distinguish_bound_winners_from_generic_afferents() {
        let bound = recruited_runtime(TemporalContextPolicy::BoundWinners);
        let bound_factory = factory_for_runtime(&bound);
        let mut bound_checkpoint = encoded_checkpoint(&bound);
        let starts = checkpoint_v1_test_section_starts(&bound_checkpoint);
        write_checkpoint_v1_u32(&mut bound_checkpoint, starts.segment_sources, 0, 11);
        write_checkpoint_v1_u16(&mut bound_checkpoint, starts.segment_ticks, 0, 1);
        refresh_body_and_logical_state_digests(&mut bound_checkpoint);
        validate_checkpoint_v1_pass2(
            validate_checkpoint_v1_pass1(&bound_checkpoint, &bound_factory)
                .expect("committed Bound source must pass envelope validation"),
            &bound_factory,
        )
        .expect("committed owner 11 is a valid Bound source");

        assert_eq!(bound.next_reserve_cell, 24);
        let committed_upper_bound = bound
            .next_reserve_cell
            .checked_sub(1)
            .expect("recruited fixture must have a committed source");
        assert_eq!(committed_upper_bound, 23);
        write_checkpoint_v1_u32(
            &mut bound_checkpoint,
            starts.segment_sources,
            0,
            committed_upper_bound,
        );
        refresh_body_and_logical_state_digests(&mut bound_checkpoint);
        validate_checkpoint_v1_pass2(
            validate_checkpoint_v1_pass1(&bound_checkpoint, &bound_factory)
                .expect("committed Bound upper boundary must pass envelope validation"),
            &bound_factory,
        )
        .expect("next_reserve_cell - 1 is a valid recruited Bound source");

        let generic = recruited_runtime(TemporalContextPolicy::GenericAfferents);
        let generic_factory = factory_for_runtime(&generic);
        let mut generic_checkpoint = encoded_checkpoint(&generic);
        let generic_starts = checkpoint_v1_test_section_starts(&generic_checkpoint);
        write_checkpoint_v1_u32(
            &mut generic_checkpoint,
            generic_starts.segment_sources,
            0,
            11,
        );
        write_checkpoint_v1_u16(&mut generic_checkpoint, generic_starts.segment_ticks, 0, 1);
        refresh_body_and_logical_state_digests(&mut generic_checkpoint);
        assert_eq!(
            pass2_error(&generic_checkpoint, &generic_factory),
            CheckpointV1Pass2Error::State(CheckpointV1StateError::LearnedStateInvariant)
        );

        write_checkpoint_v1_u32(
            &mut bound_checkpoint,
            starts.segment_sources,
            0,
            bound.next_reserve_cell,
        );
        refresh_body_and_logical_state_digests(&mut bound_checkpoint);
        assert_eq!(
            pass2_error(&bound_checkpoint, &bound_factory),
            CheckpointV1Pass2Error::State(CheckpointV1StateError::LearnedStateInvariant)
        );
    }

    #[test]
    fn pass2_context_requires_one_exact_codebook_entry_for_both_policies() {
        for context_policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            let runtime = multi_code_context_runtime(context_policy);
            let second_columns: [u32; ACTIVE_COLUMNS] =
                std::array::from_fn(|index| index as u32 + 2);
            assert_ne!(runtime.token_codebook[0].columns, second_columns);
            assert_eq!(runtime.token_codebook[1].columns, second_columns);
            assert_eq!(runtime.previous_context_len, CONTEXT_CELLS_U8);
            assert_eq!(runtime.previous_winner_context_len, CONTEXT_CELLS_U8);

            let factory = factory_for_runtime(&runtime);
            let baseline = encoded_checkpoint(&runtime);
            validate_checkpoint_v1_pass2(
                validate_checkpoint_v1_pass1(&baseline, &factory)
                    .expect("the populated second-code context must pass envelope validation"),
                &factory,
            )
            .expect("the populated context must match the second codebook entry exactly");

            let hybrid_columns: [u32; ACTIVE_COLUMNS] =
                std::array::from_fn(|index| if index == 0 { 1 } else { index as u32 + 2 });
            assert!(hybrid_columns.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(runtime
                .token_codebook
                .iter()
                .all(|code| code.columns != hybrid_columns));

            let mut hybrid = baseline;
            let starts = checkpoint_v1_test_section_starts(&hybrid);
            for (index, column) in hybrid_columns.into_iter().enumerate() {
                let winner = 2 * column;
                write_checkpoint_v1_u32(&mut hybrid, starts.winner_context, index, winner);
                write_checkpoint_v1_u32(
                    &mut hybrid,
                    starts.winner_context,
                    ACTIVE_COLUMNS + index,
                    runtime.relay_offset + winner,
                );
                match context_policy {
                    TemporalContextPolicy::BoundWinners => {
                        write_checkpoint_v1_u32(&mut hybrid, starts.prior_context, index, winner);
                        write_checkpoint_v1_u32(
                            &mut hybrid,
                            starts.prior_context,
                            ACTIVE_COLUMNS + index,
                            runtime.relay_offset + winner,
                        );
                    }
                    TemporalContextPolicy::GenericAfferents => {
                        write_checkpoint_v1_u32(
                            &mut hybrid,
                            starts.prior_context,
                            index,
                            runtime.proximal_offset + column,
                        );
                        write_checkpoint_v1_u32(
                            &mut hybrid,
                            starts.prior_context,
                            ACTIVE_COLUMNS + index,
                            runtime.proximal_relay_offset + column,
                        );
                    }
                }
            }
            refresh_body_and_logical_state_digests(&mut hybrid);
            assert_eq!(
                pass2_error(&hybrid, &factory),
                CheckpointV1Pass2Error::State(CheckpointV1StateError::ContextInvariant)
            );
        }
    }

    #[test]
    fn pass2_replays_global_cohort_chronology_without_live_csr() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = encoded_checkpoint(&runtime);
        let starts = checkpoint_v1_test_section_starts(&baseline);
        for segment_index in [1usize, 2, 4, 5] {
            let mut checkpoint = baseline.clone();
            write_checkpoint_v1_u32(&mut checkpoint, starts.segment_owners, segment_index, 4);
            refresh_body_and_logical_state_digests(&mut checkpoint);
            assert_eq!(
                pass2_error(&checkpoint, &factory),
                CheckpointV1Pass2Error::State(CheckpointV1StateError::CohortChronologyInvariant)
            );
        }

        let same_column = two_cohorts_same_column_runtime();
        let same_factory = factory_for_runtime(&same_column);
        let same_checkpoint = encoded_checkpoint(&same_column);
        let validated = validate_checkpoint_v1_pass2(
            validate_checkpoint_v1_pass1(&same_checkpoint, &same_factory)
                .expect("same-column cohorts must pass envelope validation"),
            &same_factory,
        )
        .expect("same-column cohorts must pass inductive chronology replay");
        assert_eq!(
            validated.column_occupancies,
            [6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        );
        assert_eq!(validated.max_column_occupancy, 6);

        let mut previous_cohort_not_full = same_checkpoint;
        let starts = checkpoint_v1_test_section_starts(&previous_cohort_not_full);
        write_checkpoint_v1_u32(&mut previous_cohort_not_full, starts.segment_owners, 7, 25);
        refresh_body_and_logical_state_digests(&mut previous_cohort_not_full);
        assert_eq!(
            pass2_error(&previous_cohort_not_full, &same_factory),
            CheckpointV1Pass2Error::State(CheckpointV1StateError::CohortChronologyInvariant)
        );
    }

    #[test]
    fn pass2_enforces_exact_context_grammar_for_both_policies_and_plasticities() {
        for context_policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            for plasticity_policy in [
                TemporalPlasticityPolicy::Plastic,
                TemporalPlasticityPolicy::GrowthOff,
            ] {
                let runtime = full_policy_runtime(context_policy, plasticity_policy);
                let factory = factory_for_runtime(&runtime);
                let baseline = encoded_checkpoint(&runtime);
                let starts = checkpoint_v1_test_section_starts(&baseline);
                let context_error =
                    CheckpointV1Pass2Error::State(CheckpointV1StateError::ContextInvariant);

                let mut wrong_relay = baseline.clone();
                let relay = read_checkpoint_v1_u32(
                    &wrong_relay,
                    starts.prior_context + ACTIVE_COLUMNS * size_of::<u32>(),
                );
                write_checkpoint_v1_u32(
                    &mut wrong_relay,
                    starts.prior_context,
                    ACTIVE_COLUMNS,
                    relay + 1,
                );
                if context_policy == TemporalContextPolicy::BoundWinners {
                    write_checkpoint_v1_u32(
                        &mut wrong_relay,
                        starts.winner_context,
                        ACTIVE_COLUMNS,
                        relay + 1,
                    );
                }
                refresh_body_and_logical_state_digests(&mut wrong_relay);
                assert_eq!(pass2_error(&wrong_relay, &factory), context_error);

                let mut permuted = baseline.clone();
                for context_start in if context_policy == TemporalContextPolicy::BoundWinners {
                    [Some(starts.prior_context), Some(starts.winner_context)]
                } else {
                    [Some(starts.prior_context), None]
                }
                .into_iter()
                .flatten()
                {
                    let first = read_checkpoint_v1_u32(&permuted, context_start);
                    let second =
                        read_checkpoint_v1_u32(&permuted, context_start + size_of::<u32>());
                    let first_relay = read_checkpoint_v1_u32(
                        &permuted,
                        context_start + ACTIVE_COLUMNS * size_of::<u32>(),
                    );
                    let second_relay = read_checkpoint_v1_u32(
                        &permuted,
                        context_start + (ACTIVE_COLUMNS + 1) * size_of::<u32>(),
                    );
                    write_checkpoint_v1_u32(&mut permuted, context_start, 0, second);
                    write_checkpoint_v1_u32(&mut permuted, context_start, 1, first);
                    write_checkpoint_v1_u32(
                        &mut permuted,
                        context_start,
                        ACTIVE_COLUMNS,
                        second_relay,
                    );
                    write_checkpoint_v1_u32(
                        &mut permuted,
                        context_start,
                        ACTIVE_COLUMNS + 1,
                        first_relay,
                    );
                }
                refresh_body_and_logical_state_digests(&mut permuted);
                assert_eq!(pass2_error(&permuted, &factory), context_error);

                let mut wrong_winner_column = baseline.clone();
                let winner = 2u32;
                write_checkpoint_v1_u32(&mut wrong_winner_column, starts.winner_context, 0, winner);
                write_checkpoint_v1_u32(
                    &mut wrong_winner_column,
                    starts.winner_context,
                    ACTIVE_COLUMNS,
                    runtime.relay_offset + winner,
                );
                if context_policy == TemporalContextPolicy::BoundWinners {
                    write_checkpoint_v1_u32(
                        &mut wrong_winner_column,
                        starts.prior_context,
                        0,
                        winner,
                    );
                    write_checkpoint_v1_u32(
                        &mut wrong_winner_column,
                        starts.prior_context,
                        ACTIVE_COLUMNS,
                        runtime.relay_offset + winner,
                    );
                }
                refresh_body_and_logical_state_digests(&mut wrong_winner_column);
                assert_eq!(pass2_error(&wrong_winner_column, &factory), context_error);

                if context_policy == TemporalContextPolicy::BoundWinners {
                    let mut differing_contexts = baseline.clone();
                    write_checkpoint_v1_u32(&mut differing_contexts, starts.prior_context, 0, 2);
                    write_checkpoint_v1_u32(
                        &mut differing_contexts,
                        starts.prior_context,
                        ACTIVE_COLUMNS,
                        runtime.relay_offset + 2,
                    );
                    refresh_body_and_logical_state_digests(&mut differing_contexts);
                    assert_eq!(pass2_error(&differing_contexts, &factory), context_error);
                }
            }
        }
    }

    #[test]
    fn pass1_round_trips_all_empty_and_literal_nonempty_encoder_fixtures() {
        for policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            let runtime = empty_runtime(policy);
            let checkpoint = encoded_checkpoint(&runtime);
            let factory = factory_for_runtime(&runtime);
            let validated = validate_checkpoint_v1_pass1(&checkpoint, &factory)
                .expect("empty encoder fixture must pass envelope validation");

            assert_eq!(checkpoint.len(), 256);
            assert_eq!(validated.layout.body_bytes, 0);
            assert!(validated
                .body
                .sections()
                .iter()
                .all(|section| section.is_empty()));
        }

        for (policy, header) in [
            (
                TemporalContextPolicy::BoundWinners,
                SEMANTIC_FIXTURE_BOUND_HEADER_HEX,
            ),
            (
                TemporalContextPolicy::GenericAfferents,
                SEMANTIC_FIXTURE_GENERIC_HEADER_HEX,
            ),
        ] {
            let runtime = semantic_runtime(policy);
            let checkpoint = literal_checkpoint(header, SEMANTIC_FIXTURE_BODY_HEX);
            let factory = factory_for_runtime(&runtime);
            let validated = validate_checkpoint_v1_pass1(&checkpoint, &factory)
                .expect("literal semantic fixture must pass envelope validation");

            assert_eq!(validated.layout.total_bytes, 384);
            assert_eq!(validated.scalars.segment_count, 2);
            assert_eq!(validated.body.segment_owners.len(), 8);
            assert_eq!(validated.body.segment_sources.len(), 80);
            assert_eq!(validated.body.segment_ticks.len(), 40);
        }
    }

    #[test]
    fn pass1_rejects_each_fixed_header_family_before_identity_or_hashes() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );

        for offset in 0..8 {
            let mut checkpoint = baseline.clone();
            checkpoint[offset] ^= 0x80;
            assert_eq!(
                pass1_error(&checkpoint, &factory),
                CheckpointV1Pass1Error::MagicMismatch
            );
        }
        for version in [0u16, 2, u16::MAX] {
            let mut checkpoint = baseline.clone();
            checkpoint[8..10].copy_from_slice(&version.to_le_bytes());
            assert_eq!(
                pass1_error(&checkpoint, &factory),
                CheckpointV1Pass1Error::UnsupportedFormatVersion { found: version }
            );
        }
        for header_bytes in [0u16, 255, 257, u16::MAX] {
            let mut checkpoint = baseline.clone();
            checkpoint[10..12].copy_from_slice(&header_bytes.to_le_bytes());
            assert_eq!(
                pass1_error(&checkpoint, &factory),
                CheckpointV1Pass1Error::HeaderSizeMismatch {
                    found: header_bytes
                }
            );
        }
        for offset in 12..16 {
            let mut checkpoint = baseline.clone();
            checkpoint[offset] = (offset - 11) as u8;
            assert_eq!(
                pass1_error(&checkpoint, &factory),
                CheckpointV1Pass1Error::ReservedHeaderByte {
                    offset,
                    found: (offset - 11) as u8,
                }
            );
        }

        for declared in [567u64, 569, u64::MAX] {
            let mut checkpoint = baseline.clone();
            checkpoint[24..32].copy_from_slice(&declared.to_le_bytes());
            assert_eq!(
                pass1_error(&checkpoint, &factory),
                CheckpointV1Pass1Error::BodyLengthMismatch {
                    declared,
                    expected: 568,
                }
            );
        }
        for declared in [823u64, 825, u64::MAX] {
            let mut checkpoint = baseline.clone();
            checkpoint[16..24].copy_from_slice(&declared.to_le_bytes());
            assert_eq!(
                pass1_error(&checkpoint, &factory),
                CheckpointV1Pass1Error::TotalLengthMismatch {
                    declared,
                    expected: 824,
                }
            );
        }
        let mut trailing = baseline.clone();
        trailing.push(0);
        assert_eq!(
            pass1_error(&trailing, &factory),
            CheckpointV1Pass1Error::InputLengthMismatch {
                declared: 824,
                actual: 825,
            }
        );
    }

    #[test]
    fn pass1_validates_scalar_domains_and_budget_before_layout_and_hashes() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );

        let mut segment_count = baseline.clone();
        segment_count[192..196].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            pass1_error(&segment_count, &factory),
            CheckpointV1Pass1Error::SegmentCountExceedsCapacity {
                declared: 7,
                capacity: 6,
            }
        );
        let mut cohort_count = baseline.clone();
        cohort_count[196..200].copy_from_slice(&3u32.to_le_bytes());
        assert_eq!(
            pass1_error(&cohort_count, &factory),
            CheckpointV1Pass1Error::CohortCountExceedsCapacity {
                declared: 3,
                capacity: 2,
            }
        );
        for cursor in [21u32, 27] {
            let mut checkpoint = baseline.clone();
            checkpoint[200..204].copy_from_slice(&cursor.to_le_bytes());
            assert_eq!(
                pass1_error(&checkpoint, &factory),
                CheckpointV1Pass1Error::ReserveCursorOutsideCapacity {
                    declared: cursor,
                    minimum: 22,
                    maximum: 26,
                }
            );
        }
        for (offset, field) in [
            (220, CheckpointV1ContextLengthField::Prior),
            (221, CheckpointV1ContextLengthField::Winner),
        ] {
            for found in [1u8, 21, 23, u8::MAX] {
                let mut checkpoint = baseline.clone();
                checkpoint[offset] = found;
                assert_eq!(
                    pass1_error(&checkpoint, &factory),
                    CheckpointV1Pass1Error::ContextLength { field, found }
                );
            }
        }
        for found in [16u16, u16::MAX] {
            let mut checkpoint = baseline.clone();
            checkpoint[222..224].copy_from_slice(&found.to_le_bytes());
            assert_eq!(
                pass1_error(&checkpoint, &factory),
                CheckpointV1Pass1Error::PoisonCode { found }
            );
        }

        let mut layout_shaping = baseline.clone();
        layout_shaping[192..196].copy_from_slice(&5u32.to_le_bytes());
        assert_eq!(
            pass1_error(&layout_shaping, &factory),
            CheckpointV1Pass1Error::BodyLengthMismatch {
                declared: 568,
                expected: 504,
            }
        );
    }

    #[test]
    fn pass1_checks_authority_and_route_zero_circuit_identity_in_order() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );

        let wrong_authority = ValidatedCheckpointAuthority::from_raw_sha256([0x5a; 32], [0xc3; 32]);
        let wrong_authority_factory = factory_for_runtime_with_authority(&runtime, wrong_authority);
        assert_eq!(
            pass1_error(&baseline, &wrong_authority_factory),
            CheckpointV1Pass1Error::IdentityMismatch {
                field: CheckpointV1IdentityField::Protocol,
            }
        );

        let source_only_authority =
            ValidatedCheckpointAuthority::from_raw_sha256([0xa5; 32], [0xc3; 32]);
        let source_factory = factory_for_runtime_with_authority(&runtime, source_only_authority);
        assert_eq!(
            pass1_error(&baseline, &source_factory),
            CheckpointV1Pass1Error::IdentityMismatch {
                field: CheckpointV1IdentityField::SourceBundle,
            }
        );

        let mut changed_config = runtime.config;
        changed_config.construction_seed ^= 1;
        let changed_factory = ValidatedCheckpointV1RuntimeFactory::new(
            authority(),
            changed_config,
            runtime.token_codebook.clone(),
            runtime.threshold_offsets_mv.clone(),
        )
        .expect("the alternate route-0 definition remains valid");
        assert_eq!(
            pass1_error(&baseline, &changed_factory),
            CheckpointV1Pass1Error::IdentityMismatch {
                field: CheckpointV1IdentityField::Circuit,
            }
        );

        for (range, field) in [
            (32..64, CheckpointV1IdentityField::Protocol),
            (64..96, CheckpointV1IdentityField::SourceBundle),
            (96..128, CheckpointV1IdentityField::Circuit),
        ] {
            for offset in range {
                let mut checkpoint = baseline.clone();
                checkpoint[offset] ^= 1;
                assert_eq!(
                    pass1_error(&checkpoint, &factory),
                    CheckpointV1Pass1Error::IdentityMismatch { field }
                );
            }
        }
    }

    #[test]
    fn pass1_accepts_complete_scalar_domains_but_defers_cross_field_rules() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );

        let poison_cases: [(u16, Option<TemporalPoisonCode>); 16] = [
            (0, None),
            (1, Some(TemporalPoisonCode::MultipleActiveOwners)),
            (2, Some(TemporalPoisonCode::MultipleMatchingOwners)),
            (3, Some(TemporalPoisonCode::CategorizedRaceCapacity)),
            (4, Some(TemporalPoisonCode::MalformedContextOrPolicy)),
            (5, Some(TemporalPoisonCode::ProximalPhysics)),
            (6, Some(TemporalPoisonCode::ProximalRelayPhysics)),
            (7, Some(TemporalPoisonCode::ColumnRacePhysics)),
            (8, Some(TemporalPoisonCode::WinnerRelayPhysics)),
            (9, Some(TemporalPoisonCode::TickOrCounterOverflow)),
            (10, Some(TemporalPoisonCode::StoreOrPlanInvariant)),
            (11, Some(TemporalPoisonCode::PhiloxAddressExhaustion)),
            (12, Some(TemporalPoisonCode::ReserveCellCapacity)),
            (13, Some(TemporalPoisonCode::GlobalSegmentCapacity)),
            (14, Some(TemporalPoisonCode::OwnerSegmentCapacity)),
            (15, Some(TemporalPoisonCode::PreallocatedCapacityInvariant)),
        ];
        for (poison_code, expected) in poison_cases {
            let mut checkpoint = baseline.clone();
            checkpoint[222..224].copy_from_slice(&poison_code.to_le_bytes());
            refresh_logical_state_digest(&mut checkpoint);
            let validated = validate_checkpoint_v1_pass1(&checkpoint, &factory)
                .expect("every frozen poison code must pass its fixed domain");
            assert_eq!(validated.scalars.poison_code, poison_code);
            assert_eq!(validated.poison, expected);
        }

        let mut deferred = baseline.clone();
        deferred[200..204].copy_from_slice(&22u32.to_le_bytes());
        deferred[204..212].copy_from_slice(&u64::MAX.to_le_bytes());
        deferred[212..220].copy_from_slice(&u64::MAX.to_le_bytes());
        refresh_logical_state_digest(&mut deferred);
        let validated = validate_checkpoint_v1_pass1(&deferred, &factory)
            .expect("cursor chronology and counter cross-rules belong to Pass 2");
        assert_eq!(validated.scalars.next_reserve_cell, 22);
        assert_eq!(validated.scalars.event_count, u64::MAX);
        assert_eq!(validated.scalars.learning_step, u64::MAX);

        for (offset, found) in [(192usize, u32::MAX), (196, u32::MAX)] {
            let mut checkpoint = baseline.clone();
            checkpoint[offset..offset + 4].copy_from_slice(&found.to_le_bytes());
            assert!(matches!(
                pass1_error(&checkpoint, &factory),
                CheckpointV1Pass1Error::SegmentCountExceedsCapacity { .. }
                    | CheckpointV1Pass1Error::CohortCountExceedsCapacity { .. }
            ));
        }
    }

    #[test]
    fn pass1_keeps_all_body_content_opaque_after_integrity_refresh() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );

        for offset in [256usize, 264, 352, 440, 464, 704] {
            let mut checkpoint = baseline.clone();
            checkpoint[offset..offset + 2].copy_from_slice(&u16::MAX.to_le_bytes());
            refresh_body_and_logical_state_digests(&mut checkpoint);

            validate_checkpoint_v1_pass1(&checkpoint, &factory)
                .expect("cohort, context, row, and tick content belongs to Pass 2");
        }
    }

    #[test]
    fn pass1_hash_precedence_keeps_semantic_digest_opaque() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );

        let mut body = baseline.clone();
        body[256] ^= 1;
        assert_eq!(
            pass1_error(&body, &factory),
            CheckpointV1Pass1Error::BodyDigestMismatch
        );
        let mut stored_body_digest = baseline.clone();
        stored_body_digest[128] ^= 1;
        assert_eq!(
            pass1_error(&stored_body_digest, &factory),
            CheckpointV1Pass1Error::BodyDigestMismatch
        );
        let mut logical = baseline.clone();
        logical[224] ^= 1;
        assert_eq!(
            pass1_error(&logical, &factory),
            CheckpointV1Pass1Error::LogicalStateDigestMismatch
        );
        let mut cursor = baseline.clone();
        cursor[200..204].copy_from_slice(&25u32.to_le_bytes());
        assert_eq!(
            pass1_error(&cursor, &factory),
            CheckpointV1Pass1Error::LogicalStateDigestMismatch
        );
        for scalar_offset in [204usize, 212] {
            let mut scalar = baseline.clone();
            scalar[scalar_offset] ^= 1;
            assert_eq!(
                pass1_error(&scalar, &factory),
                CheckpointV1Pass1Error::LogicalStateDigestMismatch
            );
        }

        let mut nonzero_poison = baseline.clone();
        nonzero_poison[222..224].copy_from_slice(&15u16.to_le_bytes());
        refresh_logical_state_digest(&mut nonzero_poison);
        let poisoned = validate_checkpoint_v1_pass1(&nonzero_poison, &factory)
            .expect("a domain-valid stable poison code must pass Pass 1");
        assert_eq!(
            poisoned.poison,
            Some(TemporalPoisonCode::PreallocatedCapacityInvariant)
        );
        nonzero_poison[222..224].copy_from_slice(&0u16.to_le_bytes());
        assert_eq!(
            pass1_error(&nonzero_poison, &factory),
            CheckpointV1Pass1Error::LogicalStateDigestMismatch
        );

        let mut semantic = baseline.clone();
        semantic[160] ^= 1;
        let validated = validate_checkpoint_v1_pass1(&semantic, &factory)
            .expect("semantic-index verification belongs to the later rebuild");
        assert_eq!(
            validated.semantic_index_digest.raw_bytes()[0],
            baseline[160] ^ 1
        );
    }

    #[test]
    fn pass1_rejects_every_truncation_length() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let baseline = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        for length in 0..baseline.len() {
            let truncated = &baseline[..length];
            let error = pass1_error(truncated, &factory);
            if length < 256 {
                assert_eq!(
                    error,
                    CheckpointV1Pass1Error::HeaderTooShort {
                        actual: length,
                        required: 256,
                    }
                );
            } else {
                assert_eq!(
                    error,
                    CheckpointV1Pass1Error::InputLengthMismatch {
                        declared: 824,
                        actual: length as u64,
                    }
                );
            }
        }
    }

    #[test]
    fn pass1_accepts_unaligned_backing_and_preserves_all_owned_storage() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let checkpoint = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        let mut unaligned = Vec::with_capacity(checkpoint.len() + 1);
        unaligned.push(0xff);
        unaligned.extend_from_slice(&checkpoint);
        let input_pointer = unaligned.as_ptr();
        let input_capacity = unaligned.capacity();
        let codebook_pointer = factory.token_codebook.as_ptr();
        let codebook_capacity = factory.token_codebook.capacity();
        let threshold_pointer = factory.threshold_offsets_mv.as_ptr();
        let threshold_capacity = factory.threshold_offsets_mv.capacity();

        let validated = validate_checkpoint_v1_pass1(&unaligned[1..], &factory)
            .expect("Pass 1 must parse unaligned byte backing");

        assert_eq!(
            validated.body.cohort_columns.as_ptr(),
            unaligned[1 + 256..].as_ptr()
        );
        assert_eq!(unaligned.as_ptr(), input_pointer);
        assert_eq!(unaligned.capacity(), input_capacity);
        assert_eq!(factory.token_codebook.as_ptr(), codebook_pointer);
        assert_eq!(factory.token_codebook.capacity(), codebook_capacity);
        assert_eq!(factory.threshold_offsets_mv.as_ptr(), threshold_pointer);
        assert_eq!(factory.threshold_offsets_mv.capacity(), threshold_capacity);
    }

    #[test]
    fn pass1_range_helper_reports_overflow_and_platform_limits() {
        assert_eq!(
            checked_checkpoint_v1_section_range(
                CheckpointV1BodySection::SegmentSources,
                u64::MAX,
                1,
                u64::MAX,
            ),
            Err(CheckpointV1Pass1Error::SectionRangeOverflow {
                section: CheckpointV1BodySection::SegmentSources,
            })
        );
        assert_eq!(
            checked_checkpoint_v1_section_range(CheckpointV1BodySection::SegmentTicks, 255, 2, 256,),
            Err(CheckpointV1Pass1Error::SectionNotAddressable {
                section: CheckpointV1BodySection::SegmentTicks,
                end: 257,
                platform_limit: 256,
            })
        );
        assert_eq!(
            checkpoint_v1_preflight_layout::<u64>(
                CheckpointV1ImportAllocationSite::VerificationSemanticPostings,
                usize::MAX,
            ),
            Err(CheckpointV1Pass1Error::AllocationLayout {
                site: CheckpointV1ImportAllocationSite::VerificationSemanticPostings,
                element_count: usize::MAX,
            })
        );
    }

    #[test]
    fn pass1_prospective_occupancy_bound_covers_every_cohort_prefix() {
        for column_count in 1usize..=11 {
            for base_width in 1usize..=4 {
                for race_cap in base_width..=8 {
                    let base_population = column_count * base_width;
                    for reserve_cells in 0usize..=25 {
                        let owner_capacity = base_population + reserve_cells;
                        let cohort_capacity = (owner_capacity - base_population) / race_cap;
                        let bound = checked_checkpoint_v1_prospective_occupancy_limit(
                            base_width,
                            race_cap,
                            cohort_capacity,
                        )
                        .expect("small literal occupancy arithmetic must fit");

                        assert!(bound <= owner_capacity);
                        for cohort_prefix in 0..=cohort_capacity {
                            let current = base_width + race_cap * cohort_prefix;
                            let recruitable = race_cap * (cohort_capacity - cohort_prefix);
                            assert_eq!(current + recruitable, bound);
                        }
                    }
                }
            }
        }
        assert_eq!(
            checked_checkpoint_v1_prospective_occupancy_limit(1, usize::MAX, 2),
            None
        );
    }

    #[test]
    fn checkpoint_encoder_chunks_large_sections_through_the_fixed_stack_buffer() {
        let runtime = large_streaming_runtime();
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("large runtime must prepare");
        let total_bytes = prepared.layout.total_bytes;
        let mut sink = TestSink::new();

        prepared
            .encode_into(&mut sink)
            .expect("large runtime must stream");

        let write_sizes: Vec<_> = sink
            .calls
            .iter()
            .filter_map(|call| match call {
                TestSinkCall::Write(bytes) => Some(*bytes),
                _ => None,
            })
            .collect();
        assert_eq!(write_sizes, vec![256, 1_028, 8_192, 2_088, 5_140, 256]);
        assert_eq!(write_sizes.iter().copied().max(), Some(8_192));
        assert_eq!(sink.bytes().len() as u64, total_bytes);
        assert_eq!(sink.position(), total_bytes);
        assert_eq!(
            &sink.bytes()[128..160],
            Sha256::digest(&sink.bytes()[256..]).as_slice()
        );
    }

    #[test]
    fn checkpoint_encoder_rejects_nonempty_and_wrong_seek_positions() {
        let runtime = semantic_runtime(TemporalContextPolicy::BoundWinners);
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("fixture must prepare");

        let original = vec![0xaa, 0x55];
        let mut nonempty = TestSink::with_bytes(original.clone());
        assert!(matches!(
            prepared.encode_into(&mut nonempty),
            Err(CheckpointV1EncodeError::SinkNotEmpty { logical_end: 2 })
        ));
        assert_eq!(nonempty.bytes(), original);
        assert_eq!(nonempty.write_calls, 0);

        let mut wrong_backfill = TestSink::new();
        wrong_backfill.seek_position = Some((2, 7));
        assert!(matches!(
            prepared.encode_into(&mut wrong_backfill),
            Err(CheckpointV1EncodeError::SeekPosition {
                stage: CheckpointV1SeekStage::HeaderBackfill,
                expected: 0,
                actual: 7,
            })
        ));

        let mut wrong_end = TestSink::new();
        wrong_end.seek_position = Some((3, 383));
        assert!(matches!(
            prepared.encode_into(&mut wrong_end),
            Err(CheckpointV1EncodeError::SeekPosition {
                stage: CheckpointV1SeekStage::VerifyLogicalEnd,
                expected: 384,
                actual: 383,
            })
        ));
    }

    #[test]
    fn checkpoint_encoder_handles_short_writes_interrupted_and_write_contract_failures() {
        let runtime = semantic_runtime(TemporalContextPolicy::BoundWinners);
        let before = checkpoint_v1_logical_state_snapshot(&runtime);
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("fixture must prepare");
        let mut short = TestSink::new();
        short.max_write = Some(3);
        prepared
            .encode_into(&mut short)
            .expect("short writes must be completed");
        assert_eq!(
            &short.bytes()[..256],
            hex_bytes(SEMANTIC_FIXTURE_BOUND_HEADER_HEX)
        );
        assert_eq!(&short.bytes()[256..], hex_bytes(SEMANTIC_FIXTURE_BODY_HEX));

        let mut interrupted = TestSink::new();
        interrupted.seek_fault = Some((1, io::ErrorKind::Interrupted));
        interrupted.write_fault = Some((1, TestWriteFault::Error(io::ErrorKind::Interrupted)));
        interrupted.flush_fault = Some((1, io::ErrorKind::Interrupted));
        prepared
            .encode_into(&mut interrupted)
            .expect("Interrupted operations must retry");
        assert_eq!(interrupted.bytes(), short.bytes());
        assert_eq!(interrupted.seek_calls, 4);
        assert_eq!(interrupted.flush_calls, 2);

        let mut body_interrupted = TestSink::new();
        body_interrupted.write_fault = Some((2, TestWriteFault::Error(io::ErrorKind::Interrupted)));
        prepared
            .encode_into(&mut body_interrupted)
            .expect("an Interrupted body write must retry");
        assert_eq!(body_interrupted.bytes(), short.bytes());

        let mut write_zero = TestSink::new();
        write_zero.write_fault = Some((2, TestWriteFault::Zero));
        match prepared.encode_into(&mut write_zero) {
            Err(CheckpointV1EncodeError::Write { stage, source }) => {
                assert_eq!(
                    stage,
                    CheckpointV1WriteStage::Body(CheckpointV1BodySection::SegmentOwners)
                );
                assert_eq!(source.kind(), io::ErrorKind::WriteZero);
            }
            other => panic!("unexpected WriteZero result: {other:?}"),
        }

        let mut body_overreported = TestSink::new();
        body_overreported.write_fault = Some((2, TestWriteFault::Overreport));
        assert!(matches!(
            prepared.encode_into(&mut body_overreported),
            Err(CheckpointV1EncodeError::WriteCount {
                stage: CheckpointV1WriteStage::Body(CheckpointV1BodySection::SegmentOwners,),
                requested: 8,
                reported: 9,
            })
        ));

        let mut overreported = TestSink::new();
        overreported.write_fault = Some((1, TestWriteFault::Overreport));
        assert!(matches!(
            prepared.encode_into(&mut overreported),
            Err(CheckpointV1EncodeError::WriteCount {
                stage: CheckpointV1WriteStage::ReserveHeader,
                requested: 256,
                reported: 257,
            })
        ));
        assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
    }

    #[test]
    fn checkpoint_encoder_types_every_io_failure_and_retries_only_into_a_fresh_sink() {
        let mut runtime = recruited_runtime(TemporalContextPolicy::BoundWinners);
        set_reachable_context(&mut runtime);
        let before = checkpoint_v1_logical_state_snapshot(&runtime);
        let prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("all-section fixture must prepare");
        let mut baseline = TestSink::new();
        prepared
            .encode_into(&mut baseline)
            .expect("baseline must encode");
        assert_eq!(
            baseline.calls,
            vec![
                TestSinkCall::SeekEnd(0),
                TestSinkCall::Write(256),
                TestSinkCall::Write(4),
                TestSinkCall::Write(88),
                TestSinkCall::Write(88),
                TestSinkCall::Write(20),
                TestSinkCall::Write(200),
                TestSinkCall::Write(100),
                TestSinkCall::SeekStart(0),
                TestSinkCall::Write(256),
                TestSinkCall::SeekEnd(0),
                TestSinkCall::Flush,
            ]
        );

        for (call, expected_stage) in [
            (1, CheckpointV1SeekStage::InspectLogicalEnd),
            (2, CheckpointV1SeekStage::HeaderBackfill),
            (3, CheckpointV1SeekStage::VerifyLogicalEnd),
        ] {
            let mut failed = TestSink::new();
            failed.seek_fault = Some((call, io::ErrorKind::Other));
            let error = prepared
                .encode_into(&mut failed)
                .expect_err("injected seek must fail");
            let io_source = std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<io::Error>())
                .expect("seek error must preserve its I/O source");
            assert_eq!(io_source.kind(), io::ErrorKind::Other);
            assert!(error.to_string().contains("checkpoint seek failed"));
            match error {
                CheckpointV1EncodeError::Seek { stage, source } => {
                    assert_eq!(stage, expected_stage);
                    assert_eq!(source.kind(), io::ErrorKind::Other);
                }
                other => panic!("unexpected seek failure at call {call}: {other:?}"),
            }
            if call == 2 {
                assert_eq!(&failed.bytes()[..256], &[0; 256]);
            }
            assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
            let mut retry = TestSink::new();
            prepared
                .encode_into(&mut retry)
                .expect("fresh-sink seek retry must succeed");
            assert_eq!(retry.bytes(), baseline.bytes());
            assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
        }

        for (call, expected_stage) in [
            (1, CheckpointV1WriteStage::ReserveHeader),
            (
                2,
                CheckpointV1WriteStage::Body(CheckpointV1BodySection::CohortColumns),
            ),
            (
                3,
                CheckpointV1WriteStage::Body(CheckpointV1BodySection::PriorContext),
            ),
            (
                4,
                CheckpointV1WriteStage::Body(CheckpointV1BodySection::WinnerContext),
            ),
            (
                5,
                CheckpointV1WriteStage::Body(CheckpointV1BodySection::SegmentOwners),
            ),
            (
                6,
                CheckpointV1WriteStage::Body(CheckpointV1BodySection::SegmentSources),
            ),
            (
                7,
                CheckpointV1WriteStage::Body(CheckpointV1BodySection::SegmentTicks),
            ),
            (8, CheckpointV1WriteStage::BackfillHeader),
        ] {
            let mut failed = TestSink::new();
            failed.write_fault = Some((call, TestWriteFault::Error(io::ErrorKind::Other)));
            let error = prepared
                .encode_into(&mut failed)
                .expect_err("injected write must fail");
            let io_source = std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<io::Error>())
                .expect("write error must preserve its I/O source");
            assert_eq!(io_source.kind(), io::ErrorKind::Other);
            assert!(error.to_string().contains("checkpoint write failed"));
            match error {
                CheckpointV1EncodeError::Write { stage, source } => {
                    assert_eq!(stage, expected_stage);
                    assert_eq!(source.kind(), io::ErrorKind::Other);
                }
                other => panic!("unexpected write failure at call {call}: {other:?}"),
            }
            assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
            let mut retry = TestSink::new();
            prepared
                .encode_into(&mut retry)
                .expect("fresh-sink write retry must succeed");
            assert_eq!(retry.bytes(), baseline.bytes());
            assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
        }

        let mut failed = TestSink::new();
        failed.flush_fault = Some((1, io::ErrorKind::Other));
        let error = prepared
            .encode_into(&mut failed)
            .expect_err("injected flush must fail");
        let io_source = std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<io::Error>())
            .expect("flush error must preserve its I/O source");
        assert_eq!(io_source.kind(), io::ErrorKind::Other);
        assert!(error.to_string().contains("checkpoint flush failed"));
        match error {
            CheckpointV1EncodeError::Flush { source } => {
                assert_eq!(source.kind(), io::ErrorKind::Other);
            }
            other => panic!("unexpected flush failure: {other:?}"),
        }
        assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
        let mut retry = TestSink::new();
        prepared
            .encode_into(&mut retry)
            .expect("fresh-sink flush retry must succeed");
        assert_eq!(retry.bytes(), baseline.bytes());
        assert_eq!(checkpoint_v1_logical_state_snapshot(&runtime), before);
    }

    #[test]
    fn checkpoint_encoder_rejects_tampered_prepared_counts_before_touching_the_sink() {
        fn fail_before_sink(prepared: &PreparedCheckpointV1Export<'_>) -> CheckpointV1EncodeError {
            let mut sink = TestSink::new();
            let error = prepared
                .encode_into(&mut sink)
                .expect_err("tampered prepared proof must fail");
            assert!(sink.calls.is_empty());
            assert!(sink.bytes().is_empty());
            error
        }

        let runtime = semantic_runtime(TemporalContextPolicy::BoundWinners);
        let mut prepared = runtime
            .prepare_checkpoint_v1_export(authority())
            .expect("fixture must prepare");
        let scalars = prepared.scalars;
        let layout = prepared.layout;

        prepared.scalars.event_count += 1;
        let error = fail_before_sink(&prepared);
        assert!(std::error::Error::source(&error).is_none());
        assert!(matches!(
            error,
            CheckpointV1EncodeError::PreparedScalarsMismatch
        ));
        prepared.scalars = scalars;

        prepared.layout.pair_count += 1;
        match fail_before_sink(&prepared) {
            CheckpointV1EncodeError::PreparedPairCountMismatch { declared, observed } => {
                assert_eq!(declared, layout.pair_count + 1);
                assert_eq!(observed, layout.pair_count);
            }
            other => panic!("unexpected pair-count mismatch: {other:?}"),
        }
        prepared.layout = layout;

        for section in [
            CheckpointV1BodySection::CohortColumns,
            CheckpointV1BodySection::PriorContext,
            CheckpointV1BodySection::WinnerContext,
            CheckpointV1BodySection::SegmentOwners,
            CheckpointV1BodySection::SegmentSources,
            CheckpointV1BodySection::SegmentTicks,
        ] {
            let observed = match section {
                CheckpointV1BodySection::CohortColumns => {
                    prepared.layout.cohort_columns_bytes += 1;
                    layout.cohort_columns_bytes
                }
                CheckpointV1BodySection::PriorContext => {
                    prepared.layout.prior_context_bytes += 1;
                    layout.prior_context_bytes
                }
                CheckpointV1BodySection::WinnerContext => {
                    prepared.layout.winner_context_bytes += 1;
                    layout.winner_context_bytes
                }
                CheckpointV1BodySection::SegmentOwners => {
                    prepared.layout.segment_owner_bytes += 1;
                    layout.segment_owner_bytes
                }
                CheckpointV1BodySection::SegmentSources => {
                    prepared.layout.segment_source_bytes += 1;
                    layout.segment_source_bytes
                }
                CheckpointV1BodySection::SegmentTicks => {
                    prepared.layout.segment_tick_bytes += 1;
                    layout.segment_tick_bytes
                }
            };
            match fail_before_sink(&prepared) {
                CheckpointV1EncodeError::BodySectionCount {
                    section: actual,
                    declared,
                    observed: actual_observed,
                } => {
                    assert_eq!(actual, section);
                    assert_eq!(declared, observed + 1);
                    assert_eq!(actual_observed, observed);
                }
                other => panic!("unexpected {section:?} mismatch: {other:?}"),
            }
            prepared.layout = layout;
        }

        prepared.layout.body_bytes += 1;
        match fail_before_sink(&prepared) {
            CheckpointV1EncodeError::BodyByteCount { declared, observed } => {
                assert_eq!(declared, layout.body_bytes + 1);
                assert_eq!(observed, layout.body_bytes);
            }
            other => panic!("unexpected body-byte mismatch: {other:?}"),
        }
        prepared.layout = layout;

        prepared.layout.total_bytes += 1;
        match fail_before_sink(&prepared) {
            CheckpointV1EncodeError::TotalByteCount { declared, observed } => {
                assert_eq!(declared, layout.total_bytes + 1);
                assert_eq!(observed, layout.total_bytes);
            }
            other => panic!("unexpected total-byte mismatch: {other:?}"),
        }
    }

    #[test]
    fn logical_state_digest_matches_the_frozen_literal_vector() {
        let body_digest = std::array::from_fn(|index| index as u8);
        let scalar_block = CheckpointV1Scalars::new(
            0x0102_0304,
            0x0506_0708,
            0x090a_0b0c,
            0x0102_0304_0506_0708,
            0x1112_1314_1516_1718,
            22,
            22,
            Some(TemporalPoisonCode::PreallocatedCapacityInvariant),
        )
        .expect("the literal scalar state is valid")
        .to_le_bytes();

        let digest = checkpoint_v1_logical_state_digest(&body_digest, &scalar_block);

        assert_eq!(
            LOGICAL_STATE_DOMAIN.len() + body_digest.len() + scalar_block.len(),
            106
        );
        assert_eq!(
            scalar_block,
            [
                0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07,
                0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
                22, 22, 15, 0,
            ]
        );
        assert_eq!(
            digest,
            [
                0x83, 0xac, 0xb5, 0xb8, 0xfd, 0xd0, 0x25, 0x6e, 0xa4, 0xa6, 0x9d, 0x07, 0xd7, 0xa8,
                0xd3, 0xf9, 0x6c, 0x1c, 0x26, 0xc9, 0xfc, 0x3a, 0x6b, 0x19, 0xd6, 0x71, 0x15, 0x67,
                0x15, 0x79, 0x88, 0xd1,
            ]
        );
    }

    #[test]
    fn scalar_packer_uses_the_frozen_nonpalindromic_widths_order_and_endianness() {
        let bytes = CheckpointV1Scalars::new(
            0x0102_0304,
            0x1112_1314,
            0x2122_2324,
            0x3132_3334_3536_3738,
            0x4142_4344_4546_4748,
            0x51,
            0x52,
            Some(TemporalPoisonCode::TickOrCounterOverflow),
        )
        .unwrap()
        .to_le_bytes();

        assert_eq!(
            bytes,
            [
                0x04, 0x03, 0x02, 0x01, 0x14, 0x13, 0x12, 0x11, 0x24, 0x23, 0x22, 0x21, 0x38, 0x37,
                0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x48, 0x47, 0x46, 0x45, 0x44, 0x43, 0x42, 0x41,
                0x51, 0x52, 0x09, 0x00,
            ]
        );
    }

    #[test]
    fn scalar_packer_maps_every_poison_code_exhaustively() {
        let unpoisoned = CheckpointV1Scalars::new(0, 0, 0, 0, 0, 0, 0, None)
            .expect("None is the canonical unpoisoned representation")
            .to_le_bytes();
        assert_eq!(&unpoisoned[30..32], &[0, 0]);

        let cases = [
            (TemporalPoisonCode::MultipleActiveOwners, 1u16),
            (TemporalPoisonCode::MultipleMatchingOwners, 2),
            (TemporalPoisonCode::CategorizedRaceCapacity, 3),
            (TemporalPoisonCode::MalformedContextOrPolicy, 4),
            (TemporalPoisonCode::ProximalPhysics, 5),
            (TemporalPoisonCode::ProximalRelayPhysics, 6),
            (TemporalPoisonCode::ColumnRacePhysics, 7),
            (TemporalPoisonCode::WinnerRelayPhysics, 8),
            (TemporalPoisonCode::TickOrCounterOverflow, 9),
            (TemporalPoisonCode::StoreOrPlanInvariant, 10),
            (TemporalPoisonCode::PhiloxAddressExhaustion, 11),
            (TemporalPoisonCode::ReserveCellCapacity, 12),
            (TemporalPoisonCode::GlobalSegmentCapacity, 13),
            (TemporalPoisonCode::OwnerSegmentCapacity, 14),
            (TemporalPoisonCode::PreallocatedCapacityInvariant, 15),
        ];
        for (poison, expected) in cases {
            let bytes = CheckpointV1Scalars::new(0, 0, 0, 0, 0, 0, 0, Some(poison))
                .expect("terminal poison must encode")
                .to_le_bytes();
            assert_eq!(&bytes[30..32], &expected.to_le_bytes());
        }
        assert_eq!(
            CheckpointV1Scalars::new(0, 0, 0, 0, 0, 0, 0, Some(TemporalPoisonCode::Unpoisoned),),
            Err(CheckpointV1CodecError::ExplicitUnpoisonedCode)
        );
    }

    #[test]
    fn layout_formula_is_exact_at_empty_small_a3_and_v1_width_maxima() {
        assert_eq!(
            CheckpointV1Layout::checked(0, 0, 0, 0).unwrap(),
            CheckpointV1Layout {
                pair_count: 0,
                cohort_columns_bytes: 0,
                prior_context_bytes: 0,
                winner_context_bytes: 0,
                segment_owner_bytes: 0,
                segment_source_bytes: 0,
                segment_tick_bytes: 0,
                body_bytes: 0,
                total_bytes: 256,
            }
        );
        assert_eq!(
            CheckpointV1Layout::checked(3, 2, 22, 22).unwrap(),
            CheckpointV1Layout {
                pair_count: 30,
                cohort_columns_bytes: 8,
                prior_context_bytes: 88,
                winner_context_bytes: 88,
                segment_owner_bytes: 12,
                segment_source_bytes: 120,
                segment_tick_bytes: 60,
                body_bytes: 376,
                total_bytes: 632,
            }
        );
        assert_eq!(
            CheckpointV1Layout::checked(23_286_406, 10_204, 22, 22).unwrap(),
            CheckpointV1Layout {
                pair_count: 232_864_060,
                cohort_columns_bytes: 40_816,
                prior_context_bytes: 88,
                winner_context_bytes: 88,
                segment_owner_bytes: 93_145_624,
                segment_source_bytes: 931_456_240,
                segment_tick_bytes: 465_728_120,
                body_bytes: 1_490_370_976,
                total_bytes: 1_490_371_232,
            }
        );
        assert_eq!(
            CheckpointV1Layout::checked(u32::MAX, u32::MAX, u8::MAX, u8::MAX).unwrap(),
            CheckpointV1Layout {
                pair_count: 42_949_672_950,
                cohort_columns_bytes: 17_179_869_180,
                prior_context_bytes: 1_020,
                winner_context_bytes: 1_020,
                segment_owner_bytes: 17_179_869_180,
                segment_source_bytes: 171_798_691_800,
                segment_tick_bytes: 85_899_345_900,
                body_bytes: 292_057_778_100,
                total_bytes: 292_057_778_356,
            }
        );
    }

    #[test]
    fn header_encoder_matches_the_independently_frozen_256_byte_vector() {
        let identity = TemporalCheckpointV1Identity {
            authority: ValidatedCheckpointAuthority::from_raw_sha256(
                std::array::from_fn(|index| index as u8),
                std::array::from_fn(|index| (0x20 + index) as u8),
            ),
            circuit_identity: CheckpointV1Digest::from_raw_bytes(std::array::from_fn(|index| {
                (0x40 + index) as u8
            })),
            recognition_route: TemporalRecognitionRoute::ExactGeneric,
            recognition_certificate: None,
            circuit_preimage_bytes: 0,
        };
        let layout = CheckpointV1Layout::checked(3, 2, 22, 22).unwrap();
        let scalars = CheckpointV1Scalars::new(
            3,
            2,
            0xa1b2_c3d4,
            0x0102_0304_0506_0708,
            0x1112_1314_1516_1718,
            22,
            22,
            Some(TemporalPoisonCode::PreallocatedCapacityInvariant),
        )
        .unwrap();
        let body_digest =
            CheckpointV1Digest::from_raw_bytes(std::array::from_fn(|index| (0x60 + index) as u8));
        let semantic_index_digest =
            CheckpointV1Digest::from_raw_bytes(std::array::from_fn(|index| (0x80 + index) as u8));

        let header = checkpoint_v1_header_bytes(
            &identity,
            layout,
            scalars,
            body_digest,
            semantic_index_digest,
        )
        .unwrap();

        // Independently frozen with CPython struct.pack/hashlib; no project
        // helper or production byte slice participated in this oracle.
        assert_eq!(
            header,
            [
                0x52, 0x33, 0x4e, 0x54, 0x43, 0x50, 0x31, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
                0x00, 0x00, 0x78, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x78, 0x01, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
                0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
                0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33,
                0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41,
                0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
                0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d,
                0x5e, 0x5f, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b,
                0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
                0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
                0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95,
                0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f, 0x03, 0x00, 0x00, 0x00,
                0x02, 0x00, 0x00, 0x00, 0xd4, 0xc3, 0xb2, 0xa1, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03,
                0x02, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x16, 0x16, 0x0f, 0x00,
                0x68, 0x72, 0x10, 0xf8, 0x9f, 0x21, 0x93, 0x02, 0x20, 0x68, 0xbe, 0x12, 0xce, 0xac,
                0x0f, 0x55, 0x37, 0xf3, 0x62, 0xc3, 0x4b, 0x78, 0xbb, 0x3a, 0xdb, 0xa6, 0x83, 0xe4,
                0x2f, 0xcb, 0xa6, 0xe5,
            ]
        );
        assert_eq!(&header[12..16], &[0; 4]);
    }

    #[test]
    fn header_field_perturbations_are_confined_to_their_frozen_slices_and_digests() {
        fn identity(
            protocol: [u8; 32],
            source_bundle: [u8; 32],
            circuit: [u8; 32],
        ) -> TemporalCheckpointV1Identity {
            TemporalCheckpointV1Identity {
                authority: ValidatedCheckpointAuthority::from_raw_sha256(protocol, source_bundle),
                circuit_identity: CheckpointV1Digest::from_raw_bytes(circuit),
                recognition_route: TemporalRecognitionRoute::ExactGeneric,
                recognition_certificate: None,
                circuit_preimage_bytes: 0,
            }
        }

        fn assert_confined_and_reaches_each_slice(
            baseline: &[u8; CHECKPOINT_V1_HEADER_BYTES as usize],
            perturbed: &[u8; CHECKPOINT_V1_HEADER_BYTES as usize],
            allowed: &[std::ops::Range<usize>],
        ) {
            for range in allowed {
                assert_ne!(&baseline[range.clone()], &perturbed[range.clone()]);
            }
            for index in 0..baseline.len() {
                if !allowed.iter().any(|range| range.contains(&index)) {
                    assert_eq!(baseline[index], perturbed[index], "offset {index}");
                }
            }
        }

        let protocol = [0x11; 32];
        let source_bundle = [0x22; 32];
        let circuit = [0x33; 32];
        let body_digest = CheckpointV1Digest::from_raw_bytes([0x44; 32]);
        let semantic_index_digest = CheckpointV1Digest::from_raw_bytes([0x55; 32]);
        let layout = CheckpointV1Layout::checked(3, 2, 22, 22).unwrap();
        let scalars = CheckpointV1Scalars::new(3, 2, 0x1020_3040, 50, 60, 22, 22, None).unwrap();
        let baseline_identity = identity(protocol, source_bundle, circuit);
        let baseline = checkpoint_v1_header_bytes(
            &baseline_identity,
            layout,
            scalars,
            body_digest,
            semantic_index_digest,
        )
        .unwrap();

        let mut changed_protocol = protocol;
        changed_protocol[7] ^= 0x80;
        let changed_protocol_identity = identity(changed_protocol, source_bundle, circuit);
        let changed = checkpoint_v1_header_bytes(
            &changed_protocol_identity,
            layout,
            scalars,
            body_digest,
            semantic_index_digest,
        )
        .unwrap();
        let protocol_slice = 32..64;
        assert_confined_and_reaches_each_slice(
            &baseline,
            &changed,
            std::slice::from_ref(&protocol_slice),
        );

        let mut changed_source = source_bundle;
        changed_source[9] ^= 0x40;
        let changed_source_identity = identity(protocol, changed_source, circuit);
        let changed = checkpoint_v1_header_bytes(
            &changed_source_identity,
            layout,
            scalars,
            body_digest,
            semantic_index_digest,
        )
        .unwrap();
        let source_bundle_slice = 64..96;
        assert_confined_and_reaches_each_slice(
            &baseline,
            &changed,
            std::slice::from_ref(&source_bundle_slice),
        );

        let mut changed_circuit = circuit;
        changed_circuit[11] ^= 0x20;
        let changed_circuit_identity = identity(protocol, source_bundle, changed_circuit);
        let changed = checkpoint_v1_header_bytes(
            &changed_circuit_identity,
            layout,
            scalars,
            body_digest,
            semantic_index_digest,
        )
        .unwrap();
        let circuit_slice = 96..128;
        assert_confined_and_reaches_each_slice(
            &baseline,
            &changed,
            std::slice::from_ref(&circuit_slice),
        );

        let mut changed_body = [0x44; 32];
        changed_body[13] ^= 0x10;
        let changed = checkpoint_v1_header_bytes(
            &baseline_identity,
            layout,
            scalars,
            CheckpointV1Digest::from_raw_bytes(changed_body),
            semantic_index_digest,
        )
        .unwrap();
        assert_confined_and_reaches_each_slice(&baseline, &changed, &[128..160, 224..256]);

        let mut changed_semantic = [0x55; 32];
        changed_semantic[15] ^= 0x08;
        let changed = checkpoint_v1_header_bytes(
            &baseline_identity,
            layout,
            scalars,
            body_digest,
            CheckpointV1Digest::from_raw_bytes(changed_semantic),
        )
        .unwrap();
        let semantic_slice = 160..192;
        assert_confined_and_reaches_each_slice(
            &baseline,
            &changed,
            std::slice::from_ref(&semantic_slice),
        );

        let changed_scalars =
            CheckpointV1Scalars::new(3, 2, 0x1020_3041, 50, 60, 22, 22, None).unwrap();
        let changed = checkpoint_v1_header_bytes(
            &baseline_identity,
            layout,
            changed_scalars,
            body_digest,
            semantic_index_digest,
        )
        .unwrap();
        assert_confined_and_reaches_each_slice(&baseline, &changed, &[200..204, 224..256]);

        let changed_layout = CheckpointV1Layout::checked(4, 2, 22, 22).unwrap();
        let changed_scalars =
            CheckpointV1Scalars::new(4, 2, 0x1020_3040, 50, 60, 22, 22, None).unwrap();
        let changed = checkpoint_v1_header_bytes(
            &baseline_identity,
            changed_layout,
            changed_scalars,
            body_digest,
            semantic_index_digest,
        )
        .unwrap();
        assert_confined_and_reaches_each_slice(&baseline, &changed, &[16..32, 192..196, 224..256]);
    }

    #[test]
    fn header_encoder_rejects_a_layout_from_different_scalar_width_values() {
        let identity = TemporalCheckpointV1Identity {
            authority: authority(),
            circuit_identity: CheckpointV1Digest::from_raw_bytes([0x33; 32]),
            recognition_route: TemporalRecognitionRoute::ExactGeneric,
            recognition_certificate: None,
            circuit_preimage_bytes: 0,
        };
        let layout = CheckpointV1Layout::checked(3, 2, 22, 22).unwrap();
        let scalars = CheckpointV1Scalars::new(4, 2, 0, 0, 0, 22, 22, None).unwrap();

        assert_eq!(
            checkpoint_v1_header_bytes(
                &identity,
                layout,
                scalars,
                CheckpointV1Digest::from_raw_bytes([0; 32]),
                CheckpointV1Digest::from_raw_bytes([0; 32]),
            ),
            Err(CheckpointV1CodecError::LayoutScalarMismatch)
        );
    }

    #[test]
    fn route_zero_runtime_rebuilds_the_frozen_circuit_identity() {
        let runtime = frozen_route_zero_runtime();

        let identity = runtime
            .checkpoint_v1_identity(authority(), TemporalRecognitionRoute::ExactGeneric)
            .expect("validated exact-generic runtime must identify");

        assert_eq!(identity.circuit_preimage_bytes(), 2_064);
        assert_eq!(identity.recognition_certificate(), None);
        assert_eq!(
            identity.recognition_route(),
            TemporalRecognitionRoute::ExactGeneric
        );
        assert_eq!(
            identity.circuit_identity().raw_bytes(),
            &[
                0x1c, 0xb6, 0xb0, 0x8d, 0x2c, 0xc0, 0xc5, 0xc0, 0x2e, 0x9c, 0x33, 0xf6, 0x0a, 0xa4,
                0x2a, 0x12, 0xe9, 0x41, 0xb3, 0x97, 0x3f, 0x30, 0xe5, 0x24, 0x1c, 0x95, 0x7f, 0x1f,
                0x81, 0x79, 0xff, 0x9e,
            ]
        );
        assert_eq!(
            identity.authority().protocol_identity().raw_bytes(),
            &[0xa5; 32]
        );
        assert_eq!(
            identity.authority().source_bundle_identity().raw_bytes(),
            &std::array::from_fn::<u8, 32, _>(|index| index as u8)
        );
    }

    #[test]
    fn validated_factory_and_live_runtime_share_the_exact_route_zero_identity() {
        let runtime = full_section_generic_runtime();
        let expected = runtime
            .checkpoint_v1_identity(authority(), TemporalRecognitionRoute::ExactGeneric)
            .expect("live runtime must identify");
        let codebook = runtime.token_codebook.clone();
        let thresholds = runtime.threshold_offsets_mv.clone();
        let codebook_pointer = codebook.as_ptr();
        let threshold_pointer = thresholds.as_ptr();
        let codebook_capacity = codebook.capacity();
        let threshold_capacity = thresholds.capacity();
        let factory = ValidatedCheckpointV1RuntimeFactory::new(
            authority(),
            runtime.config,
            codebook,
            thresholds,
        )
        .expect("canonical owned definition must validate");

        assert_eq!(factory.identity(), &expected);
        assert_eq!(factory.token_codebook.as_ptr(), codebook_pointer);
        assert_eq!(factory.threshold_offsets_mv.as_ptr(), threshold_pointer);
        assert_eq!(factory.token_codebook.capacity(), codebook_capacity);
        assert_eq!(factory.threshold_offsets_mv.capacity(), threshold_capacity);
    }

    #[test]
    fn validated_factory_rejects_noncanonical_owned_inputs_without_repair() {
        let runtime = nonuniform_generic_route_zero_runtime();

        let mut descending = runtime.token_codebook.clone();
        descending.swap(0, 1);
        assert_eq!(
            ValidatedCheckpointV1RuntimeFactory::new(
                authority(),
                runtime.config,
                descending,
                runtime.threshold_offsets_mv.clone(),
            )
            .err(),
            Some(CheckpointV1FactoryError::Definition(
                CheckpointV1CircuitDefinitionError::TokenIdsNotStrict { left_index: 0 }
            ))
        );

        let mut duplicate_columns = runtime.token_codebook.clone();
        duplicate_columns[1].columns = duplicate_columns[0].columns;
        assert_eq!(
            ValidatedCheckpointV1RuntimeFactory::new(
                authority(),
                runtime.config,
                duplicate_columns,
                runtime.threshold_offsets_mv.clone(),
            )
            .err(),
            Some(CheckpointV1FactoryError::Definition(
                CheckpointV1CircuitDefinitionError::DuplicateTokenColumns {
                    left_index: 0,
                    right_index: 1,
                }
            ))
        );

        let mut unsorted_columns = runtime.token_codebook.clone();
        unsorted_columns[0].columns.swap(0, 1);
        assert_eq!(
            ValidatedCheckpointV1RuntimeFactory::new(
                authority(),
                runtime.config,
                unsorted_columns,
                runtime.threshold_offsets_mv.clone(),
            )
            .err(),
            Some(CheckpointV1FactoryError::Definition(
                CheckpointV1CircuitDefinitionError::InvalidTokenColumns { index: 0 }
            ))
        );

        let mut sentinel = runtime.token_codebook.clone();
        sentinel[0].token_id = u32::MAX;
        assert_eq!(
            ValidatedCheckpointV1RuntimeFactory::new(
                authority(),
                runtime.config,
                sentinel,
                runtime.threshold_offsets_mv.clone(),
            )
            .err(),
            Some(CheckpointV1FactoryError::Definition(
                CheckpointV1CircuitDefinitionError::SentinelTokenId { index: 0 }
            ))
        );

        let mut short_thresholds = runtime.threshold_offsets_mv.clone();
        short_thresholds.pop();
        assert_eq!(
            ValidatedCheckpointV1RuntimeFactory::new(
                authority(),
                runtime.config,
                runtime.token_codebook.clone(),
                short_thresholds,
            )
            .err(),
            Some(CheckpointV1FactoryError::Definition(
                CheckpointV1CircuitDefinitionError::ThresholdOffsetCount {
                    expected: 373,
                    actual: 372,
                }
            ))
        );

        let mut nonfinite_thresholds = runtime.threshold_offsets_mv.clone();
        nonfinite_thresholds[17] = f64::NAN;
        assert_eq!(
            ValidatedCheckpointV1RuntimeFactory::new(
                authority(),
                runtime.config,
                runtime.token_codebook.clone(),
                nonfinite_thresholds,
            )
            .err(),
            Some(CheckpointV1FactoryError::Definition(
                CheckpointV1CircuitDefinitionError::NonFiniteThresholdOffset { index: 17 }
            ))
        );

        let mut invalid_config = runtime.config;
        invalid_config.segment_capacity = 0;
        let error = ValidatedCheckpointV1RuntimeFactory::new(
            authority(),
            invalid_config,
            runtime.token_codebook.clone(),
            runtime.threshold_offsets_mv.clone(),
        )
        .err()
        .expect("zero segment capacity must be rejected");
        assert_eq!(
            error,
            CheckpointV1FactoryError::Definition(
                CheckpointV1CircuitDefinitionError::SegmentCapacity { found: 0 }
            )
        );
        assert!(Error::source(&error).is_some());
        assert_eq!(
            error.to_string(),
            "invalid circuit definition: segment_capacity 0 is outside the checkpoint-v1 domain"
        );
    }

    #[test]
    fn nonuniform_generic_route_zero_fixture_discriminates_circuit_fields() {
        let runtime = nonuniform_generic_route_zero_runtime();

        let identity = runtime
            .checkpoint_v1_identity(authority(), TemporalRecognitionRoute::ExactGeneric)
            .expect("validated nonuniform route-0 runtime must identify");

        // Independent CPython struct/hashlib framing of all 35 literal fields.
        // Paired with the bound-winner vector above, this distinguishes fields
        // that intentionally share values in either fixture alone.
        assert_eq!(identity.circuit_preimage_bytes(), 4_480);
        assert_eq!(identity.recognition_certificate(), None);
        assert_eq!(
            identity.circuit_identity().raw_bytes(),
            &[
                0x1d, 0x88, 0x6b, 0x58, 0xe0, 0x8c, 0x1f, 0x93, 0x50, 0xdc, 0x3a, 0x37, 0x71, 0xab,
                0x33, 0x9d, 0x12, 0xa2, 0x2c, 0xb1, 0x9c, 0x9b, 0x36, 0xc0, 0x76, 0xd4, 0x08, 0xfc,
                0x62, 0xbd, 0x9f, 0x85,
            ]
        );
    }

    #[test]
    fn route_one_certificate_matches_the_independent_literal_vector() {
        let runtime = identity_runtime(
            11,
            TemporalContextPolicy::GenericAfferents,
            vec![TokenCode {
                token_id: 7,
                columns: std::array::from_fn(|index| index as u32),
            }],
            Vec::new(),
        )
        .expect("generic-afferent certificate fixture must build");

        let certificate = recompute_checkpoint_v1_physical_route_certificate(&runtime)
            .expect("safe generic-afferent codebook must certify");

        // Independent CPython struct/hashlib framing of this literal fixture.
        assert_eq!(
            certificate.raw_bytes(),
            &[
                0x7b, 0x05, 0x82, 0x73, 0x78, 0xbf, 0x02, 0x4f, 0xc0, 0x90, 0x66, 0xf8, 0xbf, 0x85,
                0xb7, 0xb5, 0x67, 0x5d, 0xf6, 0xa4, 0xbe, 0x87, 0x0b, 0xe7, 0x07, 0x0f, 0xda, 0xeb,
                0x6c, 0x89, 0xf9, 0xc3,
            ]
        );
        assert_eq!(
            runtime
                .checkpoint_v1_identity(
                    authority(),
                    TemporalRecognitionRoute::CertifiedPhysicalBuckets,
                )
                .unwrap_err(),
            TemporalCheckpointIdentityError::UnsupportedExecutedRecognitionRoute
        );
    }

    #[test]
    fn route_one_certificate_rejects_policy_and_overlap_violations() {
        let bound = frozen_route_zero_runtime();
        assert_eq!(
            recompute_checkpoint_v1_physical_route_certificate(&bound).unwrap_err(),
            TemporalCheckpointIdentityError::RouteRequiresGenericAfferents
        );

        let overlap = identity_runtime(
            13,
            TemporalContextPolicy::GenericAfferents,
            vec![
                TokenCode {
                    token_id: 1,
                    columns: std::array::from_fn(|index| index as u32),
                },
                TokenCode {
                    token_id: 2,
                    columns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
                },
            ],
            Vec::new(),
        )
        .expect("overlap fixture is a valid generic runtime");
        assert_eq!(
            recompute_checkpoint_v1_physical_route_certificate(&overlap).unwrap_err(),
            TemporalCheckpointIdentityError::UnsafePhysicalRouteOverlap {
                logical_match_pairs: 9,
                max_distinct_code_overlap: 9,
            }
        );
    }

    #[test]
    fn constructor_order_is_canonical_for_circuit_and_certificate_hashes() {
        let first = TokenCode {
            token_id: 2,
            columns: std::array::from_fn(|index| index as u32),
        };
        let second = TokenCode {
            token_id: 7,
            columns: std::array::from_fn(|index| (ACTIVE_COLUMNS + index) as u32),
        };
        let ascending = identity_runtime(
            22,
            TemporalContextPolicy::GenericAfferents,
            vec![first, second],
            Vec::new(),
        )
        .expect("ascending codebook must build");
        let descending = identity_runtime(
            22,
            TemporalContextPolicy::GenericAfferents,
            vec![second, first],
            Vec::new(),
        )
        .expect("descending codebook must canonicalize");

        assert_eq!(
            ascending
                .checkpoint_v1_identity(authority(), TemporalRecognitionRoute::ExactGeneric)
                .unwrap()
                .circuit_identity(),
            descending
                .checkpoint_v1_identity(authority(), TemporalRecognitionRoute::ExactGeneric)
                .unwrap()
                .circuit_identity()
        );
        assert_eq!(
            recompute_checkpoint_v1_physical_route_certificate(&ascending).unwrap(),
            recompute_checkpoint_v1_physical_route_certificate(&descending).unwrap()
        );
    }

    #[test]
    fn authority_preserves_raw_digest_bytes_without_hex_or_rehashing() {
        let protocol = *b"0123456789abcdef0123456789abcdef";
        let source = *b"fedcba9876543210fedcba9876543210";
        let authority = ValidatedCheckpointAuthority::from_raw_sha256(protocol, source);

        assert_eq!(std::mem::size_of::<CheckpointV1Digest>(), 32);
        assert_eq!(authority.protocol_identity().raw_bytes(), &protocol);
        assert_eq!(authority.source_bundle_identity().raw_bytes(), &source);
    }

    #[test]
    fn non_finite_and_impossible_runtime_identity_inputs_fail_closed() {
        let codebook = vec![TokenCode {
            token_id: 7,
            columns: std::array::from_fn(|index| index as u32),
        }];
        let mut nan_offsets = vec![0.0; 77];
        nan_offsets[0] = f64::NAN;
        let nan_result = identity_runtime(
            11,
            TemporalContextPolicy::BoundWinners,
            codebook,
            nan_offsets,
        );
        assert!(matches!(
            nan_result,
            Err(TemporalRuntimeBuildError::Invalid(
                "threshold offsets must cover every physical cell with finite f64 values"
            ))
        ));

        let mut corrupted = frozen_route_zero_runtime();
        corrupted.relay_offset += 1;
        assert_eq!(
            corrupted
                .checkpoint_v1_identity(authority(), TemporalRecognitionRoute::ExactGeneric)
                .unwrap_err(),
            TemporalCheckpointIdentityError::RuntimeInvariant(
                "stored runtime offsets disagree with executed configuration"
            )
        );

        let mut non_finite = frozen_route_zero_runtime();
        non_finite.threshold_offsets_mv[0] = f64::INFINITY;
        assert_eq!(
            non_finite
                .checkpoint_v1_identity(authority(), TemporalRecognitionRoute::ExactGeneric)
                .unwrap_err(),
            TemporalCheckpointIdentityError::RuntimeInvariant(
                "threshold offsets are incomplete or non-finite"
            )
        );
    }

    #[test]
    fn signed_zero_threshold_bits_are_part_of_the_circuit_identity() {
        let positive_zero = frozen_route_zero_runtime();
        let mut negative_zero = frozen_route_zero_runtime();
        negative_zero.threshold_offsets_mv[0] = -0.0;

        assert_ne!(
            positive_zero
                .checkpoint_v1_identity(authority(), TemporalRecognitionRoute::ExactGeneric)
                .unwrap()
                .circuit_identity(),
            negative_zero
                .checkpoint_v1_identity(authority(), TemporalRecognitionRoute::ExactGeneric)
                .unwrap()
                .circuit_identity()
        );
    }

    #[test]
    fn checkpoint_import_full_fixture_records_exact_post_pass2_allocation_contract() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let checkpoint = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory).unwrap();
        let pass2 = validate_checkpoint_v1_pass2(pass1, &factory).unwrap();
        let mut observed = Vec::new();

        let imported =
            assemble_validated_checkpoint_v1_inner(pass2, factory, &mut |site, count| {
                observed.push((site, count));
                false
            })
            .expect("the literal full checkpoint must import directly");

        assert_eq!(observed, FULL_IMPORT_EXPECTED_ALLOCATIONS);
        assert_eq!(encoded_checkpoint(&imported), checkpoint);
    }

    #[test]
    fn checkpoint_import_every_post_pass2_reserve_is_retryable_at_the_exact_site() {
        for target_index in 0..FULL_IMPORT_EXPECTED_ALLOCATIONS.len() {
            let runtime = full_section_generic_runtime();
            let checkpoint = literal_checkpoint(
                FULL_SECTION_GENERIC_HEADER_HEX,
                FULL_SECTION_GENERIC_BODY_HEX,
            );
            let factory = factory_for_runtime(&runtime);
            let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory).unwrap();
            let pass2 = validate_checkpoint_v1_pass2(pass1, &factory).unwrap();
            let target = FULL_IMPORT_EXPECTED_ALLOCATIONS[target_index].0;
            let mut observed = Vec::new();

            let error =
                match assemble_validated_checkpoint_v1_inner(pass2, factory, &mut |site, count| {
                    observed.push((site, count));
                    site == target
                }) {
                    Ok(_) => panic!("the selected reserve unexpectedly succeeded"),
                    Err(error) => error,
                };

            assert_eq!(error, CheckpointV1ImportError::RetryableAllocation(target));
            assert_eq!(
                observed.as_slice(),
                &FULL_IMPORT_EXPECTED_ALLOCATIONS[..=target_index]
            );

            let retried = import_checkpoint_v1(&checkpoint, factory_for_runtime(&runtime))
                .expect("a fresh identical factory must retry successfully");
            assert_eq!(encoded_checkpoint(&retried), checkpoint);
        }
    }

    #[test]
    fn checkpoint_import_skips_only_zero_sized_policy_and_semantic_sites() {
        for (plasticity_policy, live, active_segment_capacity, race_capacity, semantic_count) in [
            (
                TemporalPlasticityPolicy::Plastic,
                true,
                6usize,
                4usize,
                6usize,
            ),
            (TemporalPlasticityPolicy::Plastic, false, 6, 6, 0),
            (TemporalPlasticityPolicy::GrowthOff, true, 0, 4, 6),
            (TemporalPlasticityPolicy::GrowthOff, false, 0, 6, 0),
        ] {
            let runtime = if live {
                full_policy_runtime(TemporalContextPolicy::BoundWinners, plasticity_policy)
            } else {
                capacity_only_policy_runtime(TemporalContextPolicy::BoundWinners, plasticity_policy)
            };
            let checkpoint = encoded_checkpoint(&runtime);
            let factory = factory_for_runtime(&runtime);
            let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory).unwrap();
            let pass2 = validate_checkpoint_v1_pass2(pass1, &factory).unwrap();
            let mut observed = Vec::new();

            let imported =
                assemble_validated_checkpoint_v1_inner(pass2, factory, &mut |site, count| {
                    assert_ne!(count, 0, "zero-sized sites must not be observed");
                    observed.push((site, count));
                    false
                })
                .expect("each policy/semantic combination must import");

            assert_eq!(
                observed,
                expected_bound_import_allocations(
                    26,
                    6,
                    2,
                    active_segment_capacity,
                    race_capacity,
                    semantic_count,
                )
            );
            assert_eq!(encoded_checkpoint(&imported), checkpoint);
        }
    }

    #[test]
    fn checkpoint_import_rejects_a_different_factory_before_observation() {
        let runtime = full_section_generic_runtime();
        let checkpoint = encoded_checkpoint(&runtime);
        let original_factory = factory_for_runtime(&runtime);
        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &original_factory).unwrap();
        let pass2 = validate_checkpoint_v1_pass2(pass1, &original_factory).unwrap();
        let changed_authority =
            ValidatedCheckpointAuthority::from_raw_sha256([0x5a; 32], [0xc3; 32]);
        let changed_factory = factory_for_runtime_with_authority(&runtime, changed_authority);
        let mut observed = Vec::new();

        let error = match assemble_validated_checkpoint_v1_inner(
            pass2,
            changed_factory,
            &mut |site, count| {
                observed.push((site, count));
                false
            },
        ) {
            Ok(_) => panic!("a proof paired with another factory unexpectedly imported"),
            Err(error) => error,
        };

        assert_eq!(error, CheckpointV1ImportError::FactoryIdentityMismatch);
        assert!(observed.is_empty());
    }

    #[test]
    fn checkpoint_import_lower_allocation_mappings_are_exhaustive_and_exact() {
        assert_eq!(
            [
                TemporalStoreAllocationSite::Owners,
                TemporalStoreAllocationSite::Sources,
                TemporalStoreAllocationSite::Permanences,
                TemporalStoreAllocationSite::OwnerSegmentCounts,
                TemporalStoreAllocationSite::SourceMeta,
                TemporalStoreAllocationSite::NextPair,
                TemporalStoreAllocationSite::PreviousPair,
                TemporalStoreAllocationSite::SourceStamps,
                TemporalStoreAllocationSite::SegmentStamps,
                TemporalStoreAllocationSite::PotentialContacts,
                TemporalStoreAllocationSite::ConnectedContacts,
                TemporalStoreAllocationSite::TouchedSegments,
                TemporalStoreAllocationSite::OwnerStamps,
                TemporalStoreAllocationSite::BestMatching,
                TemporalStoreAllocationSite::BestActive,
                TemporalStoreAllocationSite::TouchedOwners,
            ]
            .map(checkpoint_v1_import_store_site),
            [
                CheckpointV1ImportAllocationSite::StoreOwners,
                CheckpointV1ImportAllocationSite::StoreSources,
                CheckpointV1ImportAllocationSite::StorePermanences,
                CheckpointV1ImportAllocationSite::StoreOwnerSegmentCounts,
                CheckpointV1ImportAllocationSite::StoreSourceMeta,
                CheckpointV1ImportAllocationSite::StoreNextPair,
                CheckpointV1ImportAllocationSite::StorePreviousPair,
                CheckpointV1ImportAllocationSite::StoreSourceStamps,
                CheckpointV1ImportAllocationSite::StoreSegmentStamps,
                CheckpointV1ImportAllocationSite::StorePotentialContacts,
                CheckpointV1ImportAllocationSite::StoreConnectedContacts,
                CheckpointV1ImportAllocationSite::StoreTouchedSegments,
                CheckpointV1ImportAllocationSite::StoreOwnerStamps,
                CheckpointV1ImportAllocationSite::StoreBestMatching,
                CheckpointV1ImportAllocationSite::StoreBestActive,
                CheckpointV1ImportAllocationSite::StoreTouchedOwners,
            ]
        );
        assert_eq!(
            [
                CoupledRaceScratchReserveSite::Voltage,
                CoupledRaceScratchReserveSite::Thresholds,
                CoupledRaceScratchReserveSite::Excitatory,
                CoupledRaceScratchReserveSite::Emitted,
                CoupledRaceScratchReserveSite::SpikingIndices,
            ]
            .map(checkpoint_v1_import_race_site),
            [
                CheckpointV1ImportAllocationSite::RuntimeRaceVoltage,
                CheckpointV1ImportAllocationSite::RuntimeRaceThresholds,
                CheckpointV1ImportAllocationSite::RuntimeRaceExcitatory,
                CheckpointV1ImportAllocationSite::RuntimeRaceEmitted,
                CheckpointV1ImportAllocationSite::RuntimeRaceSpikingIndices,
            ]
        );
    }

    #[test]
    fn checkpoint_import_rebuilds_repeated_and_interleaved_cohort_csr_directly() {
        for (runtime, expected_log, expected_columns) in [
            (
                two_cohorts_same_column_runtime(),
                vec![0, 0],
                vec![(0usize, vec![0, 1, 22, 23, 24, 25])],
            ),
            (
                interleaved_cohort_runtime(),
                vec![0, 1, 0],
                vec![
                    (0usize, vec![0, 1, 22, 23, 26, 27]),
                    (1usize, vec![2, 3, 24, 25]),
                ],
            ),
        ] {
            let before = checkpoint_v1_logical_state_snapshot(&runtime);
            let checkpoint = encoded_checkpoint(&runtime);
            let imported = import_checkpoint_v1(&checkpoint, factory_for_runtime(&runtime))
                .expect("canonical cohort chronology must import");

            assert_eq!(imported.cohort_columns, expected_log);
            for (column, expected_cells) in expected_columns {
                let start = imported.column_offsets[column];
                let end = imported.column_offsets[column + 1];
                assert_eq!(&imported.column_cells[start..end], expected_cells);
            }
            assert!(imported.alternate_column_offsets.is_empty());
            assert!(imported.alternate_column_cells.is_empty());
            assert_eq!(checkpoint_v1_logical_state_snapshot(&imported), before);
            assert_eq!(encoded_checkpoint(&imported), checkpoint);
        }
    }

    #[test]
    fn checkpoint_import_streams_all_row_widths_without_renormalizing_sources() {
        let runtime = every_row_width_generic_runtime();
        let checkpoint = encoded_checkpoint(&runtime);
        let imported = import_checkpoint_v1(&checkpoint, factory_for_runtime(&runtime))
            .expect("all canonical fixed row widths must import");

        for segment_id in 0u32..=10 {
            let row = imported
                .store
                .runtime_segment_row(segment_id)
                .expect("the rebuilt row must exist");
            let width = usize::try_from(segment_id).unwrap();
            assert_eq!(usize::from(row.live_pair_count), width);
            assert_eq!(
                &row.normalized_sources[..width],
                &(0..segment_id).collect::<Vec<_>>()
            );
            assert!(row.normalized_sources[width..]
                .iter()
                .all(|&source| source == NONE));
            assert_eq!(
                &row.permanence_ticks[..width],
                &(0..segment_id)
                    .map(|slot| 250 + u16::try_from(slot).unwrap())
                    .collect::<Vec<_>>()
            );
            assert!(row.permanence_ticks[width..]
                .iter()
                .all(|&ticks| ticks == 0));
        }
        assert_eq!(encoded_checkpoint(&imported), checkpoint);
    }

    #[test]
    fn checkpoint_import_rebuilds_nonempty_bound_winner_source_and_continues_exactly() {
        let mut uninterrupted = bound_winner_source_runtime();
        let checkpoint = encoded_checkpoint(&uninterrupted);
        let mut imported = import_checkpoint_v1(&checkpoint, factory_for_runtime(&uninterrupted))
            .expect("a nonempty bound-winner store must import");
        let mut normalized_sources = [NONE; PAIRS_PER_SEGMENT];
        normalized_sources[0] = 23;
        let mut permanence_ticks = [0; PAIRS_PER_SEGMENT];
        permanence_ticks[0] = 300;

        assert_eq!(imported.store.runtime_segment_count(), 6);
        assert_eq!(imported.store.runtime_segment_owner(0), Some(0));
        assert_eq!(
            imported
                .store
                .runtime_segment_row(0)
                .expect("the rebuilt bound-winner row must exist"),
            TemporalSegmentRow {
                normalized_sources,
                permanence_ticks,
                live_pair_count: 1,
            }
        );
        assert_eq!(encoded_checkpoint(&imported), checkpoint);

        let uninterrupted_step = uninterrupted.recognize(7);
        let imported_step = imported.recognize(7);
        assert_eq!(imported_step, uninterrupted_step);
        assert_eq!(
            checkpoint_v1_logical_state_snapshot(&imported),
            checkpoint_v1_logical_state_snapshot(&uninterrupted)
        );
        assert_eq!(
            encoded_checkpoint(&imported),
            encoded_checkpoint(&uninterrupted)
        );
    }

    #[test]
    fn exact_source_owner_survives_disconnection_and_checkpoint_resume_without_reallocation() {
        let mut uninterrupted = exact_source_reservation_runtime();
        let source = uninterrupted.step(0, TemporalStepMode::Learn).unwrap();
        let learned = uninterrupted.step(1, TemporalStepMode::Learn).unwrap();
        let source_winners = source.winner_context;
        let target_winners = learned.winner_context;
        assert_eq!(learned.allocated_segments, ACTIVE_COLUMNS as u8);

        uninterrupted.reset_transient();
        uninterrupted.step(0, TemporalStepMode::ReadOnly).unwrap();
        let healthy = uninterrupted.step(1, TemporalStepMode::ReadOnly).unwrap();
        assert_eq!(healthy.predicted_columns, ACTIVE_COLUMNS as u8);
        assert_eq!(healthy.winner_context, target_winners);

        for wrong_target in [2, 3] {
            uninterrupted.reset_transient();
            uninterrupted.step(0, TemporalStepMode::Learn).unwrap();
            let punishment = uninterrupted
                .step(wrong_target, TemporalStepMode::Learn)
                .unwrap();
            assert_eq!(
                punishment.punished_segments,
                ACTIVE_COLUMNS as u32 * (wrong_target - 1),
            );
        }
        let checkpoint = encoded_checkpoint(&uninterrupted);
        let mut restored = import_checkpoint_v1(&checkpoint, factory_for_runtime(&uninterrupted))
            .expect("the twice-punished exact-source state must restore");

        for runtime in [&mut uninterrupted, &mut restored] {
            runtime.reset_transient();
            runtime.step(0, TemporalStepMode::Learn).unwrap();
            let punishment = runtime.step(4, TemporalStepMode::Learn).unwrap();
            assert_eq!(punishment.punished_segments, 3 * ACTIVE_COLUMNS as u32);
            for segment_id in 0..ACTIVE_COLUMNS as u32 {
                assert_eq!(
                    runtime
                        .store
                        .runtime_segment_row(segment_id)
                        .unwrap()
                        .permanence_ticks,
                    [249; PAIRS_PER_SEGMENT],
                );
            }
            recover_exact_source_owner(runtime, source_winners, target_winners);
        }
    }

    #[test]
    fn checkpoint_import_moves_factory_vectors_and_outlives_unaligned_input_backing() {
        let mut uninterrupted = capacity_only_policy_runtime(
            TemporalContextPolicy::GenericAfferents,
            TemporalPlasticityPolicy::Plastic,
        );
        let checkpoint = encoded_checkpoint(&uninterrupted);
        let factory = factory_for_runtime(&uninterrupted);
        let codebook_pointer = factory.token_codebook.as_ptr();
        let codebook_capacity = factory.token_codebook.capacity();
        let threshold_pointer = factory.threshold_offsets_mv.as_ptr();
        let threshold_capacity = factory.threshold_offsets_mv.capacity();
        let mut backing = vec![0u8; checkpoint.len() + 4];
        let prefix = (0..4)
            .find(|&prefix| {
                !(backing.as_ptr() as usize + prefix).is_multiple_of(std::mem::align_of::<u32>())
            })
            .expect("one of four consecutive addresses must be u32-misaligned");
        backing[prefix..prefix + checkpoint.len()].copy_from_slice(&checkpoint);
        let input = &backing[prefix..prefix + checkpoint.len()];
        assert_ne!(
            input.as_ptr() as usize % std::mem::align_of::<u32>(),
            0,
            "the test must exercise actually unaligned LE backing"
        );

        let mut imported = import_checkpoint_v1(input, factory)
            .expect("the one-shot importer must own all rebuilt state");

        assert_eq!(imported.token_codebook.as_ptr(), codebook_pointer);
        assert_eq!(imported.token_codebook.capacity(), codebook_capacity);
        assert_eq!(imported.threshold_offsets_mv.as_ptr(), threshold_pointer);
        assert_eq!(imported.threshold_offsets_mv.capacity(), threshold_capacity);
        drop(backing);
        assert_eq!(encoded_checkpoint(&imported), checkpoint);

        let uninterrupted_step = uninterrupted.step(7, TemporalStepMode::Learn);
        let imported_step = imported.step(7, TemporalStepMode::Learn);
        assert_eq!(imported_step, uninterrupted_step);
        assert_eq!(
            encoded_checkpoint(&imported),
            encoded_checkpoint(&uninterrupted)
        );
    }

    #[test]
    fn checkpoint_import_retains_capacity_for_allocation_free_growth_and_recruitment() {
        let mut uninterrupted = growth_ready_bound_runtime();
        let checkpoint = encoded_checkpoint(&uninterrupted);
        let mut imported = import_checkpoint_v1(&checkpoint, factory_for_runtime(&uninterrupted))
            .expect("the growth-ready runtime must import");

        assert_eq!(imported.column_offsets.len(), 23);
        assert!(imported.column_offsets.capacity() >= 23);
        assert_eq!(imported.column_cells.len(), 22);
        assert!(imported.column_cells.capacity() >= 110);
        assert!(imported.alternate_column_offsets.is_empty());
        assert!(imported.alternate_column_offsets.capacity() >= 23);
        assert!(imported.alternate_column_cells.is_empty());
        assert!(imported.alternate_column_cells.capacity() >= 110);
        assert!(imported.cohort_columns.is_empty());
        assert!(imported.cohort_columns.capacity() >= 11);
        assert_eq!(imported.recognition_scratch.len(), 110);
        assert!(imported.recognition_scratch.capacity() >= 110);
        assert_eq!(imported.active_segment_scratch.len(), 512);
        assert!(imported.active_segment_scratch.capacity() >= 512);
        assert_eq!(imported.column_drive_scratch.len(), 89);
        assert!(imported.column_drive_scratch.capacity() >= 89);
        assert_eq!(imported.novel_candidate_scratch.len(), 89);
        assert!(imported.novel_candidate_scratch.capacity() >= 89);

        let pointers = (
            imported.column_offsets.as_ptr(),
            imported.column_cells.as_ptr(),
            imported.alternate_column_offsets.as_ptr(),
            imported.alternate_column_cells.as_ptr(),
            imported.cohort_columns.as_ptr(),
            imported.recognition_scratch.as_ptr(),
            imported.active_segment_scratch.as_ptr(),
            imported.column_drive_scratch.as_ptr(),
            imported.novel_candidate_scratch.as_ptr(),
        );
        let capacities = (
            imported.column_offsets.capacity(),
            imported.column_cells.capacity(),
            imported.alternate_column_offsets.capacity(),
            imported.alternate_column_cells.capacity(),
            imported.cohort_columns.capacity(),
            imported.recognition_scratch.capacity(),
            imported.active_segment_scratch.capacity(),
            imported.column_drive_scratch.capacity(),
            imported.novel_candidate_scratch.capacity(),
        );

        let (imported_steps, allocations) = count_allocations(|| {
            [
                imported.step(0, TemporalStepMode::Learn),
                imported.step(1, TemporalStepMode::Learn),
            ]
        });
        let uninterrupted_steps = [
            uninterrupted.step(0, TemporalStepMode::Learn),
            uninterrupted.step(1, TemporalStepMode::Learn),
        ];

        assert_eq!(allocations, 0);
        assert_eq!(imported_steps, uninterrupted_steps);
        assert!(
            imported_steps.iter().all(Result::is_ok),
            "{imported_steps:?}"
        );
        assert_eq!(
            imported_steps[1]
                .as_ref()
                .expect("the recruiting step must succeed")
                .allocated_segments,
            ACTIVE_COLUMNS as u8
        );
        assert_eq!(imported.cohort_columns, (11..22).collect::<Vec<_>>());
        assert_eq!(imported.next_reserve_cell, 110);
        assert_eq!(imported.store.runtime_segment_count(), 22);
        assert_eq!(
            (
                imported.column_offsets.as_ptr(),
                imported.column_cells.as_ptr(),
                imported.alternate_column_offsets.as_ptr(),
                imported.alternate_column_cells.as_ptr(),
                imported.cohort_columns.as_ptr(),
                imported.recognition_scratch.as_ptr(),
                imported.active_segment_scratch.as_ptr(),
                imported.column_drive_scratch.as_ptr(),
                imported.novel_candidate_scratch.as_ptr(),
            ),
            (
                pointers.2, pointers.3, pointers.0, pointers.1, pointers.4, pointers.5, pointers.6,
                pointers.7, pointers.8,
            )
        );
        assert_eq!(
            (
                imported.column_offsets.capacity(),
                imported.column_cells.capacity(),
                imported.alternate_column_offsets.capacity(),
                imported.alternate_column_cells.capacity(),
                imported.cohort_columns.capacity(),
                imported.recognition_scratch.capacity(),
                imported.active_segment_scratch.capacity(),
                imported.column_drive_scratch.capacity(),
                imported.novel_candidate_scratch.capacity(),
            ),
            (
                capacities.2,
                capacities.3,
                capacities.0,
                capacities.1,
                capacities.4,
                capacities.5,
                capacities.6,
                capacities.7,
                capacities.8,
            )
        );
        assert_eq!(
            encoded_checkpoint(&imported),
            encoded_checkpoint(&uninterrupted)
        );
    }

    #[test]
    fn checkpoint_import_preserves_all_policy_snapshots_and_next_read_only_transition() {
        for context_policy in [
            TemporalContextPolicy::BoundWinners,
            TemporalContextPolicy::GenericAfferents,
        ] {
            for plasticity_policy in [
                TemporalPlasticityPolicy::Plastic,
                TemporalPlasticityPolicy::GrowthOff,
            ] {
                let mut uninterrupted = full_policy_runtime(context_policy, plasticity_policy);
                let before = checkpoint_v1_logical_state_snapshot(&uninterrupted);
                let checkpoint = encoded_checkpoint(&uninterrupted);
                let mut imported =
                    import_checkpoint_v1(&checkpoint, factory_for_runtime(&uninterrupted))
                        .expect("every runtime-reachable policy combination must import");

                assert_eq!(checkpoint_v1_logical_state_snapshot(&imported), before);
                assert_eq!(encoded_checkpoint(&imported), checkpoint);
                assert_eq!(imported.recognize(7), uninterrupted.recognize(7));
                assert_eq!(
                    encoded_checkpoint(&imported),
                    encoded_checkpoint(&uninterrupted)
                );
            }
        }
    }

    #[test]
    fn checkpoint_import_checks_the_exact_opaque_semantic_digest_before_reexport() {
        let runtime = full_section_generic_runtime();
        let factory = factory_for_runtime(&runtime);
        let mut checkpoint = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        checkpoint[160] ^= 1;
        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory)
            .expect("Pass 1 must retain the semantic digest opaquely");
        assert_eq!(pass1.semantic_index_digest.raw_bytes()[0], checkpoint[160]);
        let pass2 = validate_checkpoint_v1_pass2(pass1, &factory)
            .expect("the opaque semantic digest must pass reachability validation");
        let mut observed = Vec::new();

        let error =
            match assemble_validated_checkpoint_v1_inner(pass2, factory, &mut |site, count| {
                observed.push((site, count));
                false
            }) {
                Ok(_) => panic!("a mismatched semantic-index digest unexpectedly imported"),
                Err(error) => error,
            };

        assert_eq!(error, CheckpointV1ImportError::SemanticIndexMismatch);
        assert_eq!(observed, FULL_IMPORT_EXPECTED_ALLOCATIONS);
    }

    #[test]
    fn checkpoint_import_maps_nonzero_and_dual_canonical_mismatches_to_exact_offsets() {
        let runtime = full_section_generic_runtime();
        let checkpoint = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        let mut nonzero_mismatch = checkpoint.clone();
        nonzero_mismatch[300] ^= 1;
        let nonzero_factory = factory_for_runtime(&runtime);
        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &nonzero_factory).unwrap();
        let mut pass2 = validate_checkpoint_v1_pass2(pass1, &nonzero_factory).unwrap();
        pass2.pass1.checkpoint = &nonzero_mismatch;

        let error =
            match assemble_validated_checkpoint_v1_inner(pass2, nonzero_factory, &mut |_, _| false)
            {
                Ok(_) => panic!("a noncanonical expected byte slice unexpectedly matched"),
                Err(error) => error,
            };

        assert_eq!(
            error,
            CheckpointV1ImportError::CanonicalReexportMismatch { offset: 300 }
        );

        let mut mismatched_expected = checkpoint.clone();
        mismatched_expected[0] ^= 1;
        mismatched_expected[300] ^= 1;
        let dual_factory = factory_for_runtime(&runtime);
        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &dual_factory).unwrap();
        let mut pass2 = validate_checkpoint_v1_pass2(pass1, &dual_factory).unwrap();
        pass2.pass1.checkpoint = &mismatched_expected;

        let error =
            match assemble_validated_checkpoint_v1_inner(pass2, dual_factory, &mut |_, _| false) {
                Ok(_) => panic!("a noncanonical expected byte slice unexpectedly matched"),
                Err(error) => error,
            };

        assert_eq!(
            error,
            CheckpointV1ImportError::CanonicalReexportMismatch { offset: 0 }
        );
    }

    #[test]
    fn checkpoint_import_preserves_every_stable_poison_code() {
        const EXPECTED: [Option<TemporalPoisonCode>; 16] = [
            None,
            Some(TemporalPoisonCode::MultipleActiveOwners),
            Some(TemporalPoisonCode::MultipleMatchingOwners),
            Some(TemporalPoisonCode::CategorizedRaceCapacity),
            Some(TemporalPoisonCode::MalformedContextOrPolicy),
            Some(TemporalPoisonCode::ProximalPhysics),
            Some(TemporalPoisonCode::ProximalRelayPhysics),
            Some(TemporalPoisonCode::ColumnRacePhysics),
            Some(TemporalPoisonCode::WinnerRelayPhysics),
            Some(TemporalPoisonCode::TickOrCounterOverflow),
            Some(TemporalPoisonCode::StoreOrPlanInvariant),
            Some(TemporalPoisonCode::PhiloxAddressExhaustion),
            Some(TemporalPoisonCode::ReserveCellCapacity),
            Some(TemporalPoisonCode::GlobalSegmentCapacity),
            Some(TemporalPoisonCode::OwnerSegmentCapacity),
            Some(TemporalPoisonCode::PreallocatedCapacityInvariant),
        ];
        let runtime = capacity_only_policy_runtime(
            TemporalContextPolicy::BoundWinners,
            TemporalPlasticityPolicy::Plastic,
        );
        let baseline = encoded_checkpoint(&runtime);

        for (code, expected) in EXPECTED.into_iter().enumerate() {
            let mut checkpoint = baseline.clone();
            checkpoint[222..224].copy_from_slice(&(code as u16).to_le_bytes());
            refresh_logical_state_digest(&mut checkpoint);
            let mut imported = import_checkpoint_v1(&checkpoint, factory_for_runtime(&runtime))
                .expect("every stable poison code must import exactly");

            assert_eq!(imported.scientific_stop, expected);
            assert_eq!(encoded_checkpoint(&imported), checkpoint);
            if expected.is_some() {
                assert_eq!(imported.recognize(7), Err(TemporalCircuitStop::Poisoned));
            }
        }
    }

    #[test]
    fn checkpoint_import_one_shot_has_exactly_two_plus_thirty_one_allocations() {
        let runtime = full_section_generic_runtime();
        let checkpoint = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        let factory = factory_for_runtime(&runtime);

        let (result, allocations) =
            count_allocations(|| import_checkpoint_v1(&checkpoint, factory));
        let imported = result.expect("the full one-shot import must succeed");

        assert_eq!(allocations, 33);
        assert_eq!(encoded_checkpoint(&imported), checkpoint);
    }

    #[test]
    fn checkpoint_import_rejects_impossible_tight_q_before_observation() {
        let runtime = full_section_generic_runtime();
        let checkpoint = encoded_checkpoint(&runtime);
        let factory = factory_for_runtime(&runtime);
        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory).unwrap();
        let mut pass2 = validate_checkpoint_v1_pass2(pass1, &factory).unwrap();
        assert_eq!(pass2.max_column_occupancy, 4);
        assert_eq!(
            pass2
                .pass1
                .allocation_plan
                .prospective_column_occupancy_limit,
            6
        );
        pass2
            .pass1
            .allocation_plan
            .prospective_column_occupancy_limit = 3;
        let mut observed = Vec::new();

        let error =
            match assemble_validated_checkpoint_v1_inner(pass2, factory, &mut |site, count| {
                observed.push((site, count));
                false
            }) {
                Ok(_) => panic!("an impossible tight-Q proof unexpectedly assembled"),
                Err(error) => error,
            };

        assert_eq!(error, CheckpointV1ImportError::AssemblyInvariant);
        assert!(observed.is_empty());
    }

    #[test]
    fn checkpoint_import_never_observes_a_zero_cohort_capacity() {
        let runtime = empty_runtime_with_plasticity(
            TemporalContextPolicy::BoundWinners,
            TemporalPlasticityPolicy::Plastic,
        );
        let checkpoint = encoded_checkpoint(&runtime);
        let factory = factory_for_runtime(&runtime);
        let pass1 = validate_checkpoint_v1_pass1(&checkpoint, &factory).unwrap();
        assert_eq!(pass1.allocation_plan.cohort_capacity, 0);
        let pass2 = validate_checkpoint_v1_pass2(pass1, &factory).unwrap();
        let mut observed = Vec::new();

        let imported =
            assemble_validated_checkpoint_v1_inner(pass2, factory, &mut |site, count| {
                assert_ne!(count, 0);
                observed.push((site, count));
                false
            })
            .expect("a runtime without reserve cohorts must import");

        assert_eq!(
            observed,
            expected_bound_import_allocations(22, 1, 0, 1, 2, 0)
        );
        assert_eq!(encoded_checkpoint(&imported), checkpoint);
    }

    #[test]
    fn checkpoint_import_maps_envelope_and_reachability_failures_without_assembly() {
        let runtime = full_section_generic_runtime();
        let checkpoint = literal_checkpoint(
            FULL_SECTION_GENERIC_HEADER_HEX,
            FULL_SECTION_GENERIC_BODY_HEX,
        );
        let mut invalid_magic = checkpoint.clone();
        invalid_magic[0] ^= 1;
        let invalid_magic_error =
            match import_checkpoint_v1(&invalid_magic, factory_for_runtime(&runtime)) {
                Ok(_) => panic!("an invalid magic unexpectedly imported"),
                Err(error) => error,
            };
        assert_eq!(
            invalid_magic_error,
            CheckpointV1ImportError::Pass1(CheckpointV1Pass1Error::MagicMismatch)
        );

        let mut invalid_state = checkpoint;
        let starts = checkpoint_v1_test_section_starts(&invalid_state);
        write_checkpoint_v1_u32(&mut invalid_state, starts.cohort_columns, 0, 11);
        refresh_body_and_logical_state_digests(&mut invalid_state);
        let invalid_state_error =
            match import_checkpoint_v1(&invalid_state, factory_for_runtime(&runtime)) {
                Ok(_) => panic!("an unreachable cohort column unexpectedly imported"),
                Err(error) => error,
            };
        assert_eq!(invalid_state_error, CheckpointV1ImportError::State);
    }
}
