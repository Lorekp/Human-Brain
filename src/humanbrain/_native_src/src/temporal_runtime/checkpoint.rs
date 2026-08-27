//! Checkpoint-v1 identities and pure fixed-header codec for the native runtime.
//!
//! This module computes identities, checked byte layout, canonical scalar
//! bytes, the complete fixed header, allocation-free reachability proofs,
//! fallible export preparation over a borrowed live state, and allocation-free
//! streaming of a prepared checkpoint into a caller-owned sink. It also owns
//! allocation-free checkpoint import Pass 1; it does not construct an imported
//! runtime, publish files, or expose an ABI.

use std::{
    alloc::Layout,
    error::Error,
    fmt,
    io::{self, Seek, SeekFrom, Write},
    mem::size_of,
};

use sha2::{Digest, Sha256};

use super::{
    TemporalCircuitConfig, TemporalCircuitRuntime, TemporalContextPolicy, TemporalPlasticityPolicy,
    TemporalPoisonCode, TokenCode, ACTIVE_COLUMNS, CONTEXT_CELLS, CONTEXT_CELLS_U8,
    PAIRS_PER_SEGMENT,
};
use crate::temporal::{
    temporal_source_meta_allocation_layout, TemporalOwnerDiagnostic, TemporalSegmentRow,
    TemporalStorePreflightError,
};
use crate::temporal_physics::{CandidateDrive, TEMPORAL_PHYSICS_IDENTITY};

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
/// Immutable runtime identity (or checkpoint-import pass 1) must already have
/// succeeded. `row_at` is the exact segment-ID logical-row/checked-SoA view.
/// `owner_load` is authoritative at this seam: the live-runtime adapter reads
/// the store-owned cache maintained by construction/mutation, while a future
/// checked-SoA importer must replay each owner count in a wider checked integer
/// before converting it to `u8` and supplying it here. This validator
/// deliberately does not recount rows with quadratic work or require
/// owner-sized scratch.
fn validate_checkpoint_v1_state(
    state: CheckpointV1StateView<'_>,
    column_replay: &mut [u32],
    mut row_at: impl FnMut(u32) -> Option<CheckpointV1OwnedRow>,
    mut owner_load: impl FnMut(u32) -> Option<u8>,
) -> Result<(), CheckpointV1StateError> {
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
        || column_replay.len() != column_count
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

    column_replay.fill(base_width);
    for &column in state.cohort_columns {
        let column_index =
            usize::try_from(column).map_err(|_| CheckpointV1StateError::LayoutInvariant)?;
        let replay = column_replay
            .get_mut(column_index)
            .ok_or(CheckpointV1StateError::LayoutInvariant)?;
        *replay = replay
            .checked_add(race_cap)
            .ok_or(CheckpointV1StateError::LayoutInvariant)?;
    }

    let expected_offset_count = column_count
        .checked_add(1)
        .ok_or(CheckpointV1StateError::ColumnCsrInvariant)?;
    let committed_count = usize::try_from(next_reserve_cell)
        .map_err(|_| CheckpointV1StateError::ColumnCsrInvariant)?;
    if state.column_offsets.len() != expected_offset_count
        || state.column_offsets.first().copied() != Some(0)
        || state.column_cells.len() != committed_count
    {
        return Err(CheckpointV1StateError::ColumnCsrInvariant);
    }
    let base_width_usize =
        usize::try_from(base_width).map_err(|_| CheckpointV1StateError::ColumnCsrInvariant)?;
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
            .and_then(|column| column.checked_mul(u64::from(base_width)))
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
            if !(base_total..next_reserve_cell).contains(&cell) {
                return Err(CheckpointV1StateError::ColumnCsrInvariant);
            }
            let cohort_index = usize::try_from((cell - base_total) / race_cap)
                .map_err(|_| CheckpointV1StateError::ColumnCsrInvariant)?;
            if state.cohort_columns.get(cohort_index).copied() != Some(column as u32) {
                return Err(CheckpointV1StateError::ColumnCsrInvariant);
            }
        }
        expected_start = expected_end;
    }
    if expected_start != state.column_cells.len() {
        return Err(CheckpointV1StateError::ColumnCsrInvariant);
    }

    for owner in 0..next_reserve_cell {
        let load = owner_load(owner).ok_or(CheckpointV1StateError::LearnedStateInvariant)?;
        if load > state.config.max_segments_per_cell {
            return Err(CheckpointV1StateError::LearnedStateInvariant);
        }
    }
    for segment_id in 0..state.segment_count {
        let owned = row_at(segment_id).ok_or(CheckpointV1StateError::LearnedStateInvariant)?;
        if checkpoint_v1_owner_column(&state, base_total, owned.owner).is_none()
            || owner_load(owned.owner)
                .filter(|load| *load != 0 && *load <= state.config.max_segments_per_cell)
                .is_none()
            || !checkpoint_v1_row_is_reachable(&state, base_total, &owned.row)
        {
            return Err(CheckpointV1StateError::LearnedStateInvariant);
        }
    }

    column_replay.fill(base_width);
    let race_cap_usize =
        usize::try_from(race_cap).map_err(|_| CheckpointV1StateError::CohortChronologyInvariant)?;
    for &column in state.cohort_columns {
        let column_index = usize::try_from(column)
            .map_err(|_| CheckpointV1StateError::CohortChronologyInvariant)?;
        let predecessor_end = usize::try_from(column_replay[column_index])
            .map_err(|_| CheckpointV1StateError::CohortChronologyInvariant)?;
        let cells = &state.column_cells
            [state.column_offsets[column_index]..state.column_offsets[column_index + 1]];
        let predecessor_start = if predecessor_end == base_width_usize {
            0
        } else {
            predecessor_end
                .checked_sub(race_cap_usize)
                .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?
        };
        let cohort_end = predecessor_end
            .checked_add(race_cap_usize)
            .filter(|end| *end <= cells.len())
            .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?;
        if cells[predecessor_start..predecessor_end]
            .iter()
            .copied()
            .any(|owner| owner_load(owner) != Some(state.config.max_segments_per_cell))
            || !cells[predecessor_end..cohort_end]
                .iter()
                .copied()
                .any(|owner| owner_load(owner).is_some_and(|load| load != 0))
        {
            return Err(CheckpointV1StateError::CohortChronologyInvariant);
        }
        column_replay[column_index] = column_replay[column_index]
            .checked_add(race_cap)
            .ok_or(CheckpointV1StateError::CohortChronologyInvariant)?;
    }

    if contexts_empty {
        return Ok(());
    }
    let mut context_columns = [0u32; ACTIVE_COLUMNS];
    match state.config.context_policy {
        TemporalContextPolicy::BoundWinners => {
            if state.previous_context != state.previous_winner_context {
                return Err(CheckpointV1StateError::ContextInvariant);
            }
            for (index, column) in context_columns.iter_mut().enumerate() {
                let winner = state.previous_context[index];
                *column = checkpoint_v1_owner_column(&state, base_total, winner)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                let expected_relay = state
                    .relay_offset
                    .checked_add(winner)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                if state.previous_context[ACTIVE_COLUMNS + index] != expected_relay {
                    return Err(CheckpointV1StateError::ContextInvariant);
                }
            }
        }
        TemporalContextPolicy::GenericAfferents => {
            for (index, column) in context_columns.iter_mut().enumerate() {
                let source = state.previous_context[index];
                *column = source
                    .checked_sub(state.proximal_offset)
                    .filter(|column| *column < state.config.column_count)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                let expected_source_relay = state
                    .proximal_relay_offset
                    .checked_add(*column)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                if state.previous_context[ACTIVE_COLUMNS + index] != expected_source_relay {
                    return Err(CheckpointV1StateError::ContextInvariant);
                }

                let winner = state.previous_winner_context[index];
                if checkpoint_v1_owner_column(&state, base_total, winner) != Some(*column) {
                    return Err(CheckpointV1StateError::ContextInvariant);
                }
                let expected_winner_relay = state
                    .relay_offset
                    .checked_add(winner)
                    .ok_or(CheckpointV1StateError::ContextInvariant)?;
                if state.previous_winner_context[ACTIVE_COLUMNS + index] != expected_winner_relay {
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
    state: &CheckpointV1StateView<'_>,
    base_total: u32,
    owner: u32,
) -> Option<u32> {
    let base_width = u32::from(state.config.base_cells_per_column);
    if owner < base_total {
        return Some(owner / base_width);
    }
    if owner >= state.next_reserve_cell {
        return None;
    }
    let cohort_index =
        usize::try_from((owner - base_total) / u32::from(state.config.novel_race_cap)).ok()?;
    state.cohort_columns.get(cohort_index).copied()
}

fn checkpoint_v1_row_is_reachable(
    state: &CheckpointV1StateView<'_>,
    base_total: u32,
    row: &TemporalSegmentRow,
) -> bool {
    let live_pair_count = usize::from(row.live_pair_count);
    if live_pair_count > PAIRS_PER_SEGMENT
        || row.normalized_sources[..live_pair_count]
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || row.permanence_ticks[..live_pair_count]
            .iter()
            .any(|ticks| !(1..=PERMANENCE_MAX_TICKS).contains(ticks))
        || row.normalized_sources[live_pair_count..]
            .iter()
            .any(|source| *source != u32::MAX)
        || row.permanence_ticks[live_pair_count..]
            .iter()
            .any(|ticks| *ticks != 0)
    {
        return false;
    }
    row.normalized_sources[..live_pair_count]
        .iter()
        .copied()
        .all(|source| match state.config.context_policy {
            TemporalContextPolicy::BoundWinners => {
                checkpoint_v1_owner_column(state, base_total, source).is_some()
            }
            TemporalContextPolicy::GenericAfferents => source < state.config.column_count,
        })
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
        let mut column_replay = checkpoint_v1_zeroed_u32_scratch(
            column_count,
            CheckpointV1AllocationSite::ColumnReplay,
            &mut fail_before_reserve,
        )?;
        validate_checkpoint_v1_runtime_state(self, &mut column_replay)
            .map_err(|_| PrepareCheckpointV1ExportError::State)?;

        let semantic_scratch_required = self
            .store
            .runtime_semantic_posting_scratch_required()
            .map_err(PrepareCheckpointV1ExportError::StoreIndex)?;
        let semantic_scratch_len = usize::try_from(semantic_scratch_required).map_err(|_| {
            PrepareCheckpointV1ExportError::Codec(CheckpointV1CodecError::LengthOverflow)
        })?;
        let mut semantic_scratch = checkpoint_v1_zeroed_u32_scratch(
            semantic_scratch_len,
            CheckpointV1AllocationSite::SemanticPostings,
            &mut fail_before_reserve,
        )?;
        let mut semantic_hasher = Sha256::new();
        self.store
            .visit_runtime_semantic_postings(&mut semantic_scratch, |posting| {
                semantic_hasher.update(posting.normalized_source.to_le_bytes());
                semantic_hasher.update(posting.segment_id.to_le_bytes());
                semantic_hasher.update(posting.slot.to_le_bytes());
            })
            .map_err(PrepareCheckpointV1ExportError::StoreIndex)?;
        let semantic_index_digest = CheckpointV1Digest(semantic_hasher.finalize().into());

        Ok(PreparedCheckpointV1Export {
            runtime: self,
            identity,
            layout,
            scalars,
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

impl PreparedCheckpointV1Export<'_> {
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
    VerificationColumnReplay,
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
            Self::VerificationColumnReplay => "verification column replay",
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
        CheckpointV1ImportAllocationSite::VerificationColumnReplay,
        column_count,
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
mod tests {
    use super::*;
    use std::io::Cursor;

    use crate::temporal_runtime::{
        TemporalCircuitConfig, TemporalCircuitRuntime, TemporalContextPolicy,
        TemporalPlasticityPolicy, TemporalRuntimeBuildError, TemporalSegmentInit, TokenCode,
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
}
