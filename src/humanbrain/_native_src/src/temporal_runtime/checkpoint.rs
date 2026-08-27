//! Checkpoint-v1 identities and pure fixed-header codec for the native runtime.
//!
//! This module deliberately stops before body traversal or I/O. It computes
//! identities, checked byte layout, canonical scalar bytes, and the complete
//! fixed header. It does not encode or import body state, touch a sink, publish
//! files, or expose an ABI.

use sha2::{Digest, Sha256};

use super::{
    TemporalCircuitRuntime, TemporalContextPolicy, TemporalPlasticityPolicy, TemporalPoisonCode,
    ACTIVE_COLUMNS, CONTEXT_CELLS, PAIRS_PER_SEGMENT,
};
use crate::temporal_physics::TEMPORAL_PHYSICS_IDENTITY;

const LOGICAL_STATE_DOMAIN: &[u8] = b"humanbrain-r3-checkpoint-logical-state-v1\0";
const CIRCUIT_IDENTITY_DOMAIN: &[u8] = b"humanbrain-r3-temporal-circuit-v1";
const PHYSICAL_ROUTE_CERTIFICATE_DOMAIN: &[u8] = b"humanbrain-r3-physical-route-certificate-v1";
const PHYSICAL_CONTACTS_PER_PAIR: u8 = 2;
const PERMANENCE_MAX_TICKS: u16 = 500;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointV1CodecError {
    ExplicitUnpoisonedCode,
    LayoutScalarMismatch,
    LengthOverflow,
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
    relay_offset: u32,
    basket_offset: u32,
    proximal_offset: u32,
    proximal_relay_offset: u32,
    physical_cell_count: u32,
    source_gid_base: u32,
    source_count: u32,
}

impl TemporalCircuitRuntime {
    pub(crate) fn checkpoint_v1_identity(
        &self,
        authority: ValidatedCheckpointAuthority,
        recognition_route: TemporalRecognitionRoute,
    ) -> Result<TemporalCheckpointV1Identity, TemporalCheckpointIdentityError> {
        let derived = validate_runtime_identity_inputs(self)?;
        if recognition_route == TemporalRecognitionRoute::CertifiedPhysicalBuckets {
            return Err(TemporalCheckpointIdentityError::UnsupportedExecutedRecognitionRoute);
        }

        let (circuit_identity, circuit_preimage_bytes) =
            circuit_identity_route_zero(self, derived)?;
        Ok(TemporalCheckpointV1Identity {
            authority,
            circuit_identity,
            recognition_route,
            recognition_certificate: None,
            circuit_preimage_bytes,
        })
    }
}

/// Recompute only the route-1 codebook eligibility certificate.
///
/// A successful result does not claim that `runtime` executes physical-source
/// buckets. The current runtime executes exact intrusive postings and therefore
/// still identifies only as route 0.
pub(crate) fn recompute_checkpoint_v1_physical_route_certificate(
    runtime: &TemporalCircuitRuntime,
) -> Result<CheckpointV1Digest, TemporalCheckpointIdentityError> {
    validate_runtime_identity_inputs(runtime)?;
    if runtime.config.context_policy != TemporalContextPolicy::GenericAfferents {
        return Err(TemporalCheckpointIdentityError::RouteRequiresGenericAfferents);
    }
    let logical_match_pairs = runtime
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
    let max_distinct_code_overlap = max_distinct_code_overlap(&runtime.token_codebook);
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
    emit_token_codebook_entry(&mut framed, &runtime.token_codebook)?;
    Ok(framed.finish().0)
}

fn circuit_identity_route_zero(
    runtime: &TemporalCircuitRuntime,
    derived: DerivedCircuitIdentity,
) -> Result<(CheckpointV1Digest, u64), TemporalCheckpointIdentityError> {
    let config = runtime.config;
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
    emit_token_codebook_entry(&mut framed, &runtime.token_codebook)?;
    emit_threshold_offsets_entry(&mut framed, &runtime.threshold_offsets_mv)?;
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
) -> Result<DerivedCircuitIdentity, TemporalCheckpointIdentityError> {
    if ACTIVE_COLUMNS != 11 || CONTEXT_CELLS != 22 || PAIRS_PER_SEGMENT != 10 {
        return Err(TemporalCheckpointIdentityError::RuntimeInvariant(
            "checkpoint-v1 fixed dimensions changed",
        ));
    }
    let config = runtime.config;
    if config.column_count < ACTIVE_COLUMNS as u32
        || config.base_cells_per_column == 0
        || config.novel_race_cap == 0
        || config.novel_race_cap > 8
        || config.base_cells_per_column > config.novel_race_cap
        || config.segment_capacity == 0
        || config.segment_capacity > u32::MAX / PAIRS_PER_SEGMENT as u32
        || !(1..=ACTIVE_COLUMNS as u8).contains(&config.prediction_quorum)
        || config.max_segments_per_cell == 0
        || !(1..=PERMANENCE_MAX_TICKS).contains(&config.connected_permanence_ticks)
        || !(1..=PERMANENCE_MAX_TICKS).contains(&config.initial_permanence_ticks)
        || !(1..=PERMANENCE_MAX_TICKS).contains(&config.permanence_increment_ticks)
        || !(1..=PERMANENCE_MAX_TICKS).contains(&config.negative_decrement_ticks)
        || !(1 <= config.theta_match
            && config.theta_match <= config.theta_active
            && config.theta_active <= PHYSICAL_CONTACTS_PER_PAIR * PAIRS_PER_SEGMENT as u8)
        || (config.context_policy == TemporalContextPolicy::GenericAfferents
            && config.base_cells_per_column < 2)
    {
        return Err(TemporalCheckpointIdentityError::RuntimeInvariant(
            "runtime configuration is outside checkpoint-v1 domains",
        ));
    }
    let base_population = u64::from(config.column_count)
        .checked_mul(u64::from(config.base_cells_per_column))
        .ok_or(TemporalCheckpointIdentityError::RuntimeInvariant(
            "base population overflows uint64",
        ))?;
    if base_population > u64::from(config.max_pyramidal_cells) {
        return Err(TemporalCheckpointIdentityError::RuntimeInvariant(
            "base population exceeds max_pyramidal_cells",
        ));
    }

    let relay_offset = config.max_pyramidal_cells;
    let basket_offset =
        relay_offset
            .checked_mul(2)
            .ok_or(TemporalCheckpointIdentityError::RuntimeInvariant(
                "physical offsets overflow uint32",
            ))?;
    let proximal_offset = basket_offset.checked_add(config.column_count).ok_or(
        TemporalCheckpointIdentityError::RuntimeInvariant("physical offsets overflow uint32"),
    )?;
    let proximal_relay_offset = proximal_offset.checked_add(config.column_count).ok_or(
        TemporalCheckpointIdentityError::RuntimeInvariant("physical offsets overflow uint32"),
    )?;
    let physical_cell_count = proximal_relay_offset
        .checked_add(config.column_count)
        .ok_or(TemporalCheckpointIdentityError::RuntimeInvariant(
            "physical offsets overflow uint32",
        ))?;
    if runtime.relay_offset != relay_offset
        || runtime.basket_offset != basket_offset
        || runtime.proximal_offset != proximal_offset
        || runtime.proximal_relay_offset != proximal_relay_offset
    {
        return Err(TemporalCheckpointIdentityError::RuntimeInvariant(
            "stored runtime offsets disagree with executed configuration",
        ));
    }
    if runtime.threshold_offsets_mv.len() != physical_cell_count as usize
        || runtime
            .threshold_offsets_mv
            .iter()
            .any(|offset| !offset.is_finite())
    {
        return Err(TemporalCheckpointIdentityError::RuntimeInvariant(
            "threshold offsets are incomplete or non-finite",
        ));
    }
    if runtime.token_codebook.is_empty()
        || u32::try_from(runtime.token_codebook.len()).is_err()
        || runtime.token_codebook.iter().any(|code| {
            code.token_id == u32::MAX
                || code
                    .columns
                    .iter()
                    .any(|&column| column >= config.column_count)
                || code.columns.windows(2).any(|pair| pair[0] >= pair[1])
        })
        || runtime
            .token_codebook
            .windows(2)
            .any(|pair| pair[0].token_id >= pair[1].token_id)
        || (0..runtime.token_codebook.len()).any(|left| {
            runtime.token_codebook[left + 1..]
                .iter()
                .any(|right| right.columns == runtime.token_codebook[left].columns)
        })
    {
        return Err(TemporalCheckpointIdentityError::RuntimeInvariant(
            "token codebook is not canonical",
        ));
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
        return Err(TemporalCheckpointIdentityError::RuntimeInvariant(
            "temporal physics identity is non-finite or impossible",
        ));
    }

    let (source_gid_base, source_count) = match config.context_policy {
        TemporalContextPolicy::BoundWinners => (0, config.max_pyramidal_cells),
        TemporalContextPolicy::GenericAfferents => (proximal_offset, config.column_count),
    };
    Ok(DerivedCircuitIdentity {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_runtime::{
        TemporalCircuitConfig, TemporalCircuitRuntime, TemporalContextPolicy,
        TemporalPlasticityPolicy, TemporalRuntimeBuildError, TokenCode,
    };

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
