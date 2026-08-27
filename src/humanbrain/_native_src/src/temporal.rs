//! Bounded native storage and recognition for fixed-slot distal segments.
//!
//! Every segment owns ten fixed logical slots containing a canonical prefix of
//! zero to ten live source/relay pairs. Unused suffix slots carry the `NONE`/0
//! sentinel. Only the normalized source and one shared permanence tick are
//! resident; a live posting visit contributes two physical contacts to the
//! recognition counts.

use std::{mem::size_of, ptr, slice};

const PAIRS_PER_SEGMENT: usize = 10;
const PHYSICAL_CONTACTS_PER_PAIR: u8 = 2;
const NONE: u32 = u32::MAX;

const TEMPORAL_OK: u32 = 0;
const TEMPORAL_INVALID: u32 = 1;
const TEMPORAL_SEGMENT_CAPACITY: u32 = 2;
const TEMPORAL_OWNER_CAPACITY: u32 = 3;
const TEMPORAL_POISONED: u32 = 4;
const TEMPORAL_ALLOCATION_FAILED: u32 = 5;
const TEMPORAL_OUTPUT_CAPACITY: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalStoreConfig {
    pub source_count: usize,
    pub owner_count: usize,
    pub segment_capacity: usize,
    pub source_gid_base: u32,
    pub connected_permanence_ticks: u16,
    pub theta_match: u8,
    pub theta_active: u8,
    pub padding_zero: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalOwnerDiagnostic {
    pub owner: u32,
    pub best_matching_segment: u32,
    pub best_active_segment: u32,
    pub matching_potential_contacts: u8,
    pub matching_connected_contacts: u8,
    pub active_potential_contacts: u8,
    pub active_connected_contacts: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TemporalPrepareSummary {
    pub candidate_segments: usize,
    pub matching_segments: usize,
    pub active_segments: usize,
    pub owner_result_count: usize,
    pub postings_visited: u64,
    pub max_potential_contacts: u8,
    pub max_connected_contacts: u8,
    pub padding_zero: [u8; 6],
}

/// Non-poisoning failures at the crate-private runtime/store seam.
///
/// The future runtime maps capacity and invariant categories onto its stable
/// poison taxonomy before a transaction begins. None of these errors mutate
/// the store's legacy diagnostic-FFI poison latch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalStorePreflightError {
    InvalidSegmentId,
    InvalidOwner,
    InvalidCanonicalRow,
    OutputCapacity { required: u32, provided: u32 },
    SegmentCapacity,
    OwnerCapacity { owner: u32 },
    PreallocatedCapacity,
    BackingInvariant,
    StorePoisoned,
}

/// One fixed-width normalized logical segment row.
///
/// A value becomes commit-ready only after the store's purpose-named
/// preflight method accepts it. Keeping the suffix explicit makes checkpoint
/// and runtime mutation code share the same canonical representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TemporalSegmentRow {
    pub(crate) normalized_sources: [u32; PAIRS_PER_SEGMENT],
    pub(crate) permanence_ticks: [u16; PAIRS_PER_SEGMENT],
    pub(crate) live_pair_count: u8,
}

/// Receipt for the exact append cardinality proved by mutation preflight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TemporalStoreMutationPreflight {
    first_append_segment_id: u32,
    prospective_append_count: u32,
}

impl TemporalStoreMutationPreflight {
    pub(crate) fn first_append_segment_id(self) -> u32 {
        self.first_append_segment_id
    }

    pub(crate) fn prospective_append_count(self) -> u32 {
        self.prospective_append_count
    }
}

/// One canonical logical posting used by the checkpoint semantic digest.
///
/// The store emits these records in lexicographic
/// `(normalized_source, segment_id, slot)` order. Intrusive-list order is
/// deliberately absent because rewrites make it history-dependent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TemporalSemanticPosting {
    pub(crate) normalized_source: u32,
    pub(crate) segment_id: u32,
    pub(crate) slot: u32,
}

/// Non-poisoning failures while rebuilding a fresh store from checkpoint rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TemporalCheckpointRebuildError {
    StoreConstruction(u32),
    DeclaredSegmentCountExceedsCapacity { declared: u32, capacity: u32 },
    DeclaredSegmentCountExceeded { declared: u32 },
    SegmentCountMismatch { declared: u32, actual: u32 },
    Store(TemporalStorePreflightError),
}

impl From<TemporalStorePreflightError> for TemporalCheckpointRebuildError {
    fn from(error: TemporalStorePreflightError) -> Self {
        Self::Store(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceMeta {
    head: u32,
    tail: u32,
    live_count: u32,
}

impl SourceMeta {
    const EMPTY: Self = Self {
        head: NONE,
        tail: NONE,
        live_count: 0,
    };
}

const _: [(); 12] = [(); size_of::<SourceMeta>()];

fn reserved_vec<T>(capacity: usize) -> Result<Vec<T>, u32> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| TEMPORAL_ALLOCATION_FAILED)?;
    Ok(values)
}

fn filled_vec<T: Copy>(length: usize, value: T) -> Result<Vec<T>, u32> {
    let mut values = reserved_vec(length)?;
    values.resize(length, value);
    Ok(values)
}

/// Write a fixed record into already-reserved `Vec` storage without calling an
/// allocation-capable growth operation. Mutation preflight proves the spare
/// capacity before any commit calls this helper.
fn append_reserved<T: Copy, const N: usize>(values: &mut Vec<T>, appended: [T; N]) {
    let original_len = values.len();
    for (slot, value) in values.spare_capacity_mut()[..N].iter_mut().zip(appended) {
        slot.write(value);
    }
    // SAFETY: exactly N initialized elements were written into spare capacity.
    unsafe {
        values.set_len(original_len + N);
    }
}

pub struct TemporalStore {
    config: TemporalStoreConfig,
    owners: Vec<u32>,
    sources: Vec<u32>,
    permanences: Vec<u16>,
    owner_segment_counts: Vec<u8>,
    source_meta: Vec<SourceMeta>,
    next_pair: Vec<u32>,
    prev_pair: Vec<u32>,
    source_stamps: Vec<u32>,
    segment_stamps: Vec<u32>,
    potential_contacts: Vec<u8>,
    connected_contacts: Vec<u8>,
    touched_segments: Vec<u32>,
    owner_stamps: Vec<u32>,
    best_matching: Vec<u32>,
    best_active: Vec<u32>,
    touched_owners: Vec<u32>,
    generation: u32,
    posting_validation_generation: u32,
    poisoned: u32,
}

/// Owned, unpublished construction transaction for canonical checkpoint rows.
///
/// The builder starts from a fresh bounded store, accepts only rows that are
/// already normalized and canonical, and yields the store only after the exact
/// declared row count has been appended. Dropping it after any failure cannot
/// affect an existing runtime.
pub(crate) struct TemporalStoreCheckpointRebuild {
    store: TemporalStore,
    declared_segment_count: u32,
}

impl TemporalStoreCheckpointRebuild {
    pub(crate) fn new(
        config: TemporalStoreConfig,
        declared_segment_count: u32,
    ) -> Result<Self, TemporalCheckpointRebuildError> {
        let segment_capacity = u32::try_from(config.segment_capacity)
            .map_err(|_| TemporalCheckpointRebuildError::StoreConstruction(TEMPORAL_INVALID))?;
        if declared_segment_count > segment_capacity {
            return Err(
                TemporalCheckpointRebuildError::DeclaredSegmentCountExceedsCapacity {
                    declared: declared_segment_count,
                    capacity: segment_capacity,
                },
            );
        }
        let store = TemporalStore::new_runtime_owned(config)
            .map_err(TemporalCheckpointRebuildError::StoreConstruction)?;
        Ok(Self {
            store,
            declared_segment_count,
        })
    }

    /// Append one already-normalized fixed row without translating source IDs.
    ///
    /// Every fallible check precedes the no-fail append commit. A rejected row
    /// leaves segment rows, owner loads, and postings unchanged and may be
    /// retried with corrected input.
    pub(crate) fn append_canonical_row(
        &mut self,
        owner: u32,
        row: &TemporalSegmentRow,
    ) -> Result<u32, TemporalCheckpointRebuildError> {
        if self.store.runtime_segment_count() >= self.declared_segment_count {
            return Err(
                TemporalCheckpointRebuildError::DeclaredSegmentCountExceeded {
                    declared: self.declared_segment_count,
                },
            );
        }
        self.store.preflight_runtime_canonical_append(owner, row)?;
        self.store
            .preflight_runtime_mutation_backing(std::slice::from_ref(&owner), u8::MAX)?;
        Ok(self
            .store
            .commit_prevalidated_runtime_canonical_append(owner, row))
    }

    /// Publish the rebuilt store only at the exact checkpoint segment count.
    pub(crate) fn finish(self) -> Result<TemporalStore, TemporalCheckpointRebuildError> {
        let actual = self.store.runtime_segment_count();
        if actual != self.declared_segment_count {
            return Err(TemporalCheckpointRebuildError::SegmentCountMismatch {
                declared: self.declared_segment_count,
                actual,
            });
        }
        self.store.preflight_runtime_backing_shape()?;
        Ok(self.store)
    }
}

impl TemporalStore {
    /// Construct storage owned by the crate-private deep temporal runtime.
    ///
    /// This keeps the public diagnostic C ABI unchanged while avoiding an
    /// internal round-trip through raw opaque handles.
    pub(crate) fn new_runtime_owned(config: TemporalStoreConfig) -> Result<Self, u32> {
        Self::new(config)
    }

    fn new(config: TemporalStoreConfig) -> Result<Self, u32> {
        if config.source_count < PAIRS_PER_SEGMENT
            || config.source_count > u32::MAX as usize
            || config.owner_count == 0
            || config.owner_count > u32::MAX as usize
            || config.segment_capacity == 0
            || config.segment_capacity > (u32::MAX as usize) / PAIRS_PER_SEGMENT
            || (config.source_gid_base as u64 + config.source_count as u64) > u32::MAX as u64
            || !(1..=500).contains(&config.connected_permanence_ticks)
            || !(1 <= config.theta_match
                && config.theta_match <= config.theta_active
                && config.theta_active <= (PAIRS_PER_SEGMENT as u8 * PHYSICAL_CONTACTS_PER_PAIR))
            || config.padding_zero != 0
        {
            return Err(TEMPORAL_INVALID);
        }
        let posting_capacity = config
            .segment_capacity
            .checked_mul(PAIRS_PER_SEGMENT)
            .ok_or(TEMPORAL_INVALID)?;
        let owner_result_capacity = config.owner_count.min(config.segment_capacity);
        Ok(Self {
            config,
            owners: reserved_vec(config.segment_capacity)?,
            sources: reserved_vec(posting_capacity)?,
            permanences: reserved_vec(posting_capacity)?,
            owner_segment_counts: filled_vec(config.owner_count, 0)?,
            source_meta: filled_vec(config.source_count, SourceMeta::EMPTY)?,
            next_pair: reserved_vec(posting_capacity)?,
            prev_pair: reserved_vec(posting_capacity)?,
            source_stamps: filled_vec(config.source_count, 0)?,
            segment_stamps: filled_vec(config.segment_capacity, 0)?,
            potential_contacts: filled_vec(config.segment_capacity, 0)?,
            connected_contacts: filled_vec(config.segment_capacity, 0)?,
            touched_segments: reserved_vec(config.segment_capacity)?,
            owner_stamps: filled_vec(config.owner_count, 0)?,
            best_matching: filled_vec(config.owner_count, NONE)?,
            best_active: filled_vec(config.owner_count, NONE)?,
            touched_owners: reserved_vec(owner_result_capacity)?,
            generation: 0,
            posting_validation_generation: 0,
            poisoned: TEMPORAL_OK,
        })
    }

    fn poison(&mut self, status: u32) -> u32 {
        self.poisoned = status;
        status
    }

    fn normalize_source(&self, source_gid: u32) -> Option<u32> {
        if source_gid == NONE {
            return None;
        }
        let normalized = source_gid.checked_sub(self.config.source_gid_base)?;
        ((normalized as usize) < self.config.source_count).then_some(normalized)
    }

    fn append_posting(&mut self, source: u32, pair_slot: u32) {
        let source_index = source as usize;
        let tail = self.source_meta[source_index].tail;
        self.prev_pair[pair_slot as usize] = tail;
        self.next_pair[pair_slot as usize] = NONE;
        if tail == NONE {
            self.source_meta[source_index].head = pair_slot;
        } else {
            self.next_pair[tail as usize] = pair_slot;
        }
        self.source_meta[source_index].tail = pair_slot;
        self.source_meta[source_index].live_count += 1;
    }

    fn unlink_posting(&mut self, pair_slot: u32) {
        let pair_index = pair_slot as usize;
        let source_index = self.sources[pair_index] as usize;
        let previous = self.prev_pair[pair_index];
        let next = self.next_pair[pair_index];
        if previous == NONE {
            self.source_meta[source_index].head = next;
        } else {
            self.next_pair[previous as usize] = next;
        }
        if next == NONE {
            self.source_meta[source_index].tail = previous;
        } else {
            self.prev_pair[next as usize] = previous;
        }
        self.source_meta[source_index].live_count -= 1;
        self.next_pair[pair_index] = NONE;
        self.prev_pair[pair_index] = NONE;
    }

    /// Commit one prevalidated canonical pair set without allocating or failing.
    ///
    /// Transactional callers must finish every validation and capacity check
    /// before entering this unlink/rewrite/relink primitive.
    fn rewrite_segment_pairs_canonical(
        &mut self,
        segment_id: u32,
        normalized_sources: &[u32],
        permanence_ticks: &[u16],
    ) {
        self.posting_validation_generation = 0;
        let first_pair = segment_id as usize * PAIRS_PER_SEGMENT;
        for pair_index in first_pair..first_pair + PAIRS_PER_SEGMENT {
            if self.sources[pair_index] != NONE {
                self.unlink_posting(pair_index as u32);
            }
        }
        for slot in 0..PAIRS_PER_SEGMENT {
            let pair_index = first_pair + slot;
            if slot < normalized_sources.len() {
                self.sources[pair_index] = normalized_sources[slot];
                self.permanences[pair_index] = permanence_ticks[slot];
            } else {
                self.sources[pair_index] = NONE;
                self.permanences[pair_index] = 0;
            }
        }
        for (slot, source) in normalized_sources.iter().enumerate() {
            self.append_posting(*source, (first_pair + slot) as u32);
        }
    }

    #[cfg(test)]
    fn rewrite_segment_pairs(
        &mut self,
        segment_id: u32,
        normalized_sources: &[u32],
        permanence_ticks: &[u16],
    ) -> Result<(), u32> {
        if self.poisoned != TEMPORAL_OK {
            return Err(TEMPORAL_POISONED);
        }
        if segment_id as usize >= self.owners.len()
            || normalized_sources.len() > PAIRS_PER_SEGMENT
            || normalized_sources.len() != permanence_ticks.len()
            || normalized_sources
                .iter()
                .enumerate()
                .any(|(index, source)| {
                    *source as usize >= self.config.source_count
                        || (index != 0 && normalized_sources[index - 1] >= *source)
                })
            || permanence_ticks
                .iter()
                .any(|ticks| !(1..=500).contains(ticks))
        {
            return Err(TEMPORAL_INVALID);
        }
        self.rewrite_segment_pairs_canonical(segment_id, normalized_sources, permanence_ticks);
        Ok(())
    }

    fn allocate_segment(
        &mut self,
        owner: u32,
        sources: &[u32],
        permanence_ticks: &[u16],
    ) -> Result<u32, u32> {
        if self.poisoned != TEMPORAL_OK {
            return Err(TEMPORAL_POISONED);
        }
        if owner as usize >= self.config.owner_count
            || sources.len() > PAIRS_PER_SEGMENT
            || permanence_ticks.len() != sources.len()
            || permanence_ticks
                .iter()
                .any(|ticks| !(1..=500).contains(ticks))
        {
            return Err(TEMPORAL_INVALID);
        }
        let mut normalized_sources = [0; PAIRS_PER_SEGMENT];
        for (index, source) in sources.iter().enumerate() {
            if index != 0 && sources[index - 1] >= *source {
                return Err(TEMPORAL_INVALID);
            }
            let Some(normalized) = self.normalize_source(*source) else {
                return Err(TEMPORAL_INVALID);
            };
            normalized_sources[index] = normalized;
        }
        if self.owners.len() >= self.config.segment_capacity {
            return Err(self.poison(TEMPORAL_SEGMENT_CAPACITY));
        }
        if self.owner_segment_counts[owner as usize] == u8::MAX {
            return Err(self.poison(TEMPORAL_OWNER_CAPACITY));
        }
        let segment_id = self.owners.len() as u32;
        self.owners.push(owner);
        self.owner_segment_counts[owner as usize] += 1;
        self.sources.extend([NONE; PAIRS_PER_SEGMENT]);
        self.permanences.extend([0; PAIRS_PER_SEGMENT]);
        self.next_pair.extend([NONE; PAIRS_PER_SEGMENT]);
        self.prev_pair.extend([NONE; PAIRS_PER_SEGMENT]);
        self.rewrite_segment_pairs_canonical(
            segment_id,
            &normalized_sources[..sources.len()],
            permanence_ticks,
        );
        Ok(segment_id)
    }

    /// Import one already-materialized learned segment with zero to ten live
    /// pairs into a runtime-owned store. Runtime inference never calls this
    /// after construction; the public FFI continues to require exactly ten.
    pub(crate) fn allocate_runtime_segment(
        &mut self,
        owner: u32,
        sources: &[u32],
        permanence_ticks: &[u16],
    ) -> Result<u32, u32> {
        self.allocate_segment(owner, sources, permanence_ticks)
    }

    /// Return the authoritative learned load used by runtime homeostasis and
    /// least-loaded winner eligibility. Mutation must update this same store;
    /// no parallel runtime-owned load counter is maintained.
    pub(crate) fn runtime_owner_load(&self, owner: u32) -> Option<u8> {
        self.owner_segment_counts.get(owner as usize).copied()
    }

    /// Number of allocated logical rows. Empty rows remain allocated.
    pub(crate) fn runtime_segment_count(&self) -> u32 {
        self.owners.len() as u32
    }

    /// Configured logical-row capacity, expressed in the canonical ID domain.
    pub(crate) fn runtime_segment_capacity(&self) -> u32 {
        self.config.segment_capacity as u32
    }

    /// Return the authoritative owner of one allocated row.
    pub(crate) fn runtime_segment_owner(&self, segment_id: u32) -> Option<u32> {
        self.owners.get(segment_id as usize).copied()
    }

    /// Return an authoritative owner during the no-fail commit phase.
    ///
    /// The caller may use this infallible seam only after the same transaction
    /// has preflighted `segment_id` and the complete store backing.
    pub(crate) fn commit_prevalidated_runtime_segment_owner(&self, segment_id: u32) -> u32 {
        self.owners[segment_id as usize]
    }

    fn validate_normalized_row(
        &self,
        row: &TemporalSegmentRow,
    ) -> Result<usize, TemporalStorePreflightError> {
        let live_pair_count = usize::from(row.live_pair_count);
        if live_pair_count > PAIRS_PER_SEGMENT
            || row.normalized_sources[..live_pair_count]
                .iter()
                .enumerate()
                .any(|(index, source)| {
                    *source as usize >= self.config.source_count
                        || (index != 0 && row.normalized_sources[index - 1] >= *source)
                })
            || row.permanence_ticks[..live_pair_count]
                .iter()
                .any(|ticks| !(1..=500).contains(ticks))
            || row.normalized_sources[live_pair_count..]
                .iter()
                .any(|source| *source != NONE)
            || row.permanence_ticks[live_pair_count..]
                .iter()
                .any(|ticks| *ticks != 0)
        {
            return Err(TemporalStorePreflightError::InvalidCanonicalRow);
        }
        Ok(live_pair_count)
    }

    fn preflight_runtime_backing_shape(&self) -> Result<(), TemporalStorePreflightError> {
        if self.poisoned != TEMPORAL_OK {
            return Err(TemporalStorePreflightError::StorePoisoned);
        }
        let segment_count = self.owners.len();
        let Some(pair_count) = segment_count.checked_mul(PAIRS_PER_SEGMENT) else {
            return Err(TemporalStorePreflightError::BackingInvariant);
        };
        let configured_pair_capacity = self
            .config
            .segment_capacity
            .checked_mul(PAIRS_PER_SEGMENT)
            .ok_or(TemporalStorePreflightError::BackingInvariant)?;
        let owner_result_capacity = self.config.owner_count.min(self.config.segment_capacity);
        if segment_count > self.config.segment_capacity
            || self.sources.len() != pair_count
            || self.permanences.len() != pair_count
            || self.next_pair.len() != pair_count
            || self.prev_pair.len() != pair_count
            || self.owner_segment_counts.len() != self.config.owner_count
            || self.source_meta.len() != self.config.source_count
            || self.source_stamps.len() != self.config.source_count
            || self.segment_stamps.len() != self.config.segment_capacity
            || self.potential_contacts.len() != self.config.segment_capacity
            || self.connected_contacts.len() != self.config.segment_capacity
            || self.owner_stamps.len() != self.config.owner_count
            || self.best_matching.len() != self.config.owner_count
            || self.best_active.len() != self.config.owner_count
            || self.owners.capacity() < self.config.segment_capacity
            || self.sources.capacity() < configured_pair_capacity
            || self.permanences.capacity() < configured_pair_capacity
            || self.next_pair.capacity() < configured_pair_capacity
            || self.prev_pair.capacity() < configured_pair_capacity
            || self.touched_segments.capacity() < self.config.segment_capacity
            || self.touched_owners.capacity() < owner_result_capacity
        {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        Ok(())
    }

    fn preflight_posting_pair_local(
        &self,
        pair_slot: u32,
        source: u32,
    ) -> Result<(), TemporalStorePreflightError> {
        let pair_index = pair_slot as usize;
        let Some(meta) = self.source_meta.get(source as usize).copied() else {
            return Err(TemporalStorePreflightError::BackingInvariant);
        };
        if self.sources.get(pair_index).copied() != Some(source) {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        let Some(previous) = self.prev_pair.get(pair_index).copied() else {
            return Err(TemporalStorePreflightError::BackingInvariant);
        };
        let Some(next) = self.next_pair.get(pair_index).copied() else {
            return Err(TemporalStorePreflightError::BackingInvariant);
        };
        if previous == pair_slot
            || next == pair_slot
            || (previous != NONE && previous == next)
            || (meta.live_count == 1 && (previous != NONE || next != NONE))
            || (meta.live_count > 1 && previous == NONE && next == NONE)
        {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        let previous_valid = if previous == NONE {
            meta.head == pair_slot
        } else {
            let previous_index = previous as usize;
            self.sources.get(previous_index).copied() == Some(source)
                && self.next_pair.get(previous_index).copied() == Some(pair_slot)
                && self.prev_pair.get(previous_index).copied() != Some(pair_slot)
        };
        let next_valid = if next == NONE {
            meta.tail == pair_slot
        } else {
            let next_index = next as usize;
            self.sources.get(next_index).copied() == Some(source)
                && self.prev_pair.get(next_index).copied() == Some(pair_slot)
                && self.next_pair.get(next_index).copied() != Some(pair_slot)
        };
        if meta.live_count == 0 || !previous_valid || !next_valid {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        Ok(())
    }

    /// Validate one source's complete declared chain and optionally copy its
    /// pair slots into caller-owned scratch in intrusive order.
    ///
    /// The caller may later sort the copied slots to remove history-dependent
    /// list order. `ALL_ROWS_PREVALIDATED = true` is reserved for the two
    /// immutable visitor passes after `semantic_posting_shape` has validated
    /// every row; mutation preflight uses the standalone `false` specialization.
    /// No callback or semantic output occurs in this helper.
    fn validate_source_posting_chain_into<const ALL_ROWS_PREVALIDATED: bool>(
        &self,
        source: u32,
        required_pair_slot: Option<u32>,
        mut pair_slots: Option<&mut [u32]>,
    ) -> Result<usize, TemporalStorePreflightError> {
        let source_index = source as usize;
        let Some(meta) = self.source_meta.get(source_index).copied() else {
            return Err(TemporalStorePreflightError::BackingInvariant);
        };
        let live_count = meta.live_count as usize;
        if live_count > self.owners.len()
            || (live_count == 0 && (meta.head != NONE || meta.tail != NONE))
            || (live_count != 0 && (meta.head == NONE || meta.tail == NONE))
        {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        if let Some(out) = pair_slots.as_ref() {
            if out.len() < live_count {
                return Err(TemporalStorePreflightError::OutputCapacity {
                    required: meta.live_count,
                    provided: out.len() as u32,
                });
            }
        }

        let mut pair_slot = meta.head;
        let mut previous = NONE;
        let mut visited = 0usize;
        let mut required_found = required_pair_slot.is_none();
        while pair_slot != NONE {
            if visited >= live_count || visited >= self.sources.len() {
                return Err(TemporalStorePreflightError::BackingInvariant);
            }
            let pair_index = pair_slot as usize;
            if self.sources.get(pair_index).copied() != Some(source)
                || self.prev_pair.get(pair_index).copied() != Some(previous)
            {
                return Err(TemporalStorePreflightError::BackingInvariant);
            }
            if !ALL_ROWS_PREVALIDATED {
                let segment_id = (pair_index / PAIRS_PER_SEGMENT) as u32;
                let row = self.runtime_segment_row(segment_id)?;
                let row_slot = pair_index % PAIRS_PER_SEGMENT;
                if row_slot >= usize::from(row.live_pair_count)
                    || row.normalized_sources[row_slot] != source
                {
                    return Err(TemporalStorePreflightError::BackingInvariant);
                }
            }
            required_found |= required_pair_slot == Some(pair_slot);

            let Some(next) = self.next_pair.get(pair_index).copied() else {
                return Err(TemporalStorePreflightError::BackingInvariant);
            };
            if next == NONE {
                if pair_slot != meta.tail {
                    return Err(TemporalStorePreflightError::BackingInvariant);
                }
            } else {
                let next_index = next as usize;
                if self.sources.get(next_index).copied() != Some(source)
                    || self.prev_pair.get(next_index).copied() != Some(pair_slot)
                {
                    return Err(TemporalStorePreflightError::BackingInvariant);
                }
            }
            if let Some(out) = pair_slots.as_deref_mut() {
                out[visited] = pair_slot;
            }
            previous = pair_slot;
            pair_slot = next;
            visited += 1;
        }
        if visited != live_count || previous != meta.tail || !required_found {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        Ok(visited)
    }

    /// Prove one touched source's complete declared posting chain exactly once
    /// per aggregate mutation preflight. `source_stamps` is scratch after
    /// recognition, so a fresh generation deduplicates high-fanout sources
    /// without allocating another source-domain array.
    fn preflight_source_posting_chain(
        &mut self,
        source: u32,
        required_pair_slot: Option<u32>,
    ) -> Result<(), TemporalStorePreflightError> {
        let source_index = source as usize;
        if self.source_meta.get(source_index).is_none() {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        let validation_generation = self.posting_validation_generation;
        if validation_generation != 0
            && self.source_stamps.get(source_index).copied() == Some(validation_generation)
        {
            if let Some(pair_slot) = required_pair_slot {
                // A valid store is built from empty chains and mutated only by
                // the splice primitives below. Once this source's complete
                // chain is proved in the current transaction, reciprocal local
                // membership is sufficient for another touched pair.
                self.preflight_posting_pair_local(pair_slot, source)?;
            }
            return Ok(());
        }
        self.validate_source_posting_chain_into::<false>(source, required_pair_slot, None)?;
        if validation_generation != 0 {
            self.source_stamps[source_index] = validation_generation;
        }
        Ok(())
    }

    fn preflight_existing_segment_postings(
        &mut self,
        segment_id: u32,
    ) -> Result<TemporalSegmentRow, TemporalStorePreflightError> {
        let row = self.runtime_segment_row(segment_id)?;
        let first_pair = segment_id as usize * PAIRS_PER_SEGMENT;
        let live_pair_count = usize::from(row.live_pair_count);
        for slot in 0..live_pair_count {
            let pair_index = first_pair + slot;
            let pair_slot = pair_index as u32;
            let source = row.normalized_sources[slot];
            self.preflight_source_posting_chain(source, Some(pair_slot))?;
            self.preflight_posting_pair_local(pair_slot, source)?;
        }
        for pair_index in first_pair + live_pair_count..first_pair + PAIRS_PER_SEGMENT {
            if self.next_pair[pair_index] != NONE || self.prev_pair[pair_index] != NONE {
                return Err(TemporalStorePreflightError::BackingInvariant);
            }
        }
        Ok(row)
    }

    /// Validate a normalized fixed row for a later no-fail append commit.
    /// Aggregate backing and owner-load capacity is checked separately after
    /// all prospective append owners are known.
    pub(crate) fn preflight_runtime_canonical_append(
        &mut self,
        owner: u32,
        row: &TemporalSegmentRow,
    ) -> Result<(), TemporalStorePreflightError> {
        self.preflight_runtime_backing_shape()?;
        if owner as usize >= self.config.owner_count {
            return Err(TemporalStorePreflightError::InvalidOwner);
        }
        let live_pair_count = self.validate_normalized_row(row)?;
        for source in row.normalized_sources[..live_pair_count].iter().copied() {
            self.preflight_source_posting_chain(source, None)?;
        }
        Ok(())
    }

    /// Validate one normalized fixed row for a later no-fail rewrite commit.
    pub(crate) fn preflight_runtime_canonical_rewrite(
        &mut self,
        segment_id: u32,
        row: &TemporalSegmentRow,
    ) -> Result<(), TemporalStorePreflightError> {
        self.preflight_runtime_backing_shape()?;
        if segment_id as usize >= self.owners.len() {
            return Err(TemporalStorePreflightError::InvalidSegmentId);
        }
        self.preflight_existing_segment_postings(segment_id)?;
        let live_pair_count = self.validate_normalized_row(row)?;
        for source in row.normalized_sources[..live_pair_count].iter().copied() {
            self.preflight_source_posting_chain(source, None)?;
        }
        Ok(())
    }

    /// Prove the exact aggregate append count, affected authoritative owner
    /// loads, and every backing-vector capacity before a mutation transaction.
    ///
    /// The owner slice contains one entry for every prospective append; its
    /// length is therefore the exact append cardinality. Duplicate owners are
    /// counted together without allocating scratch.
    pub(crate) fn preflight_runtime_mutation_backing(
        &mut self,
        prospective_append_owners: &[u32],
        max_segments_per_cell: u8,
    ) -> Result<TemporalStoreMutationPreflight, TemporalStorePreflightError> {
        self.preflight_runtime_backing_shape()?;
        if max_segments_per_cell == 0 {
            return Err(TemporalStorePreflightError::InvalidOwner);
        }
        let append_count = prospective_append_owners.len();
        let Some(prospective_segment_count) = self.owners.len().checked_add(append_count) else {
            return Err(TemporalStorePreflightError::SegmentCapacity);
        };
        if prospective_segment_count > self.config.segment_capacity {
            return Err(TemporalStorePreflightError::SegmentCapacity);
        }
        let Some(prospective_pair_count) = prospective_segment_count.checked_mul(PAIRS_PER_SEGMENT)
        else {
            return Err(TemporalStorePreflightError::SegmentCapacity);
        };
        if self.owners.capacity() < prospective_segment_count
            || self.sources.capacity() < prospective_pair_count
            || self.permanences.capacity() < prospective_pair_count
            || self.next_pair.capacity() < prospective_pair_count
            || self.prev_pair.capacity() < prospective_pair_count
        {
            return Err(TemporalStorePreflightError::PreallocatedCapacity);
        }
        for (index, owner) in prospective_append_owners.iter().copied().enumerate() {
            let owner_index = owner as usize;
            if owner_index >= self.config.owner_count {
                return Err(TemporalStorePreflightError::InvalidOwner);
            }
            if prospective_append_owners[..index].contains(&owner) {
                continue;
            }
            let owner_appends = prospective_append_owners
                .iter()
                .filter(|candidate| **candidate == owner)
                .count();
            let prospective_owner_load = usize::from(self.owner_segment_counts[owner_index])
                .checked_add(owner_appends)
                .ok_or(TemporalStorePreflightError::OwnerCapacity { owner })?;
            if prospective_owner_load > usize::from(max_segments_per_cell)
                || prospective_owner_load > usize::from(u8::MAX)
            {
                return Err(TemporalStorePreflightError::OwnerCapacity { owner });
            }
        }
        let prospective_append_count = u32::try_from(append_count)
            .map_err(|_| TemporalStorePreflightError::SegmentCapacity)?;
        let first_append_segment_id = self.owners.len() as u32;
        self.posting_validation_generation = self.next_generation();
        Ok(TemporalStoreMutationPreflight {
            first_append_segment_id,
            prospective_append_count,
        })
    }

    /// Apply one row whose segment ID and canonical form were prevalidated.
    /// This commit path allocates nothing, performs no conversion that can
    /// fail, never poisons, and returns no error.
    pub(crate) fn commit_prevalidated_runtime_canonical_rewrite(
        &mut self,
        segment_id: u32,
        row: &TemporalSegmentRow,
    ) {
        let live_pair_count = usize::from(row.live_pair_count);
        self.rewrite_segment_pairs_canonical(
            segment_id,
            &row.normalized_sources[..live_pair_count],
            &row.permanence_ticks[..live_pair_count],
        );
    }

    /// Append one row whose owner, canonical form, aggregate count, owner load,
    /// and backing capacity were prevalidated. Empty rows are allocated state.
    pub(crate) fn commit_prevalidated_runtime_canonical_append(
        &mut self,
        owner: u32,
        row: &TemporalSegmentRow,
    ) -> u32 {
        let segment_id = self.owners.len() as u32;
        append_reserved(&mut self.owners, [owner]);
        self.owner_segment_counts[owner as usize] += 1;
        append_reserved(&mut self.sources, [NONE; PAIRS_PER_SEGMENT]);
        append_reserved(&mut self.permanences, [0; PAIRS_PER_SEGMENT]);
        append_reserved(&mut self.next_pair, [NONE; PAIRS_PER_SEGMENT]);
        append_reserved(&mut self.prev_pair, [NONE; PAIRS_PER_SEGMENT]);
        self.commit_prevalidated_runtime_canonical_rewrite(segment_id, row);
        segment_id
    }

    /// Copy one allocated row into fixed stack storage without allocation.
    pub(crate) fn runtime_segment_row(
        &self,
        segment_id: u32,
    ) -> Result<TemporalSegmentRow, TemporalStorePreflightError> {
        let segment_index = segment_id as usize;
        if segment_index >= self.owners.len() {
            return Err(TemporalStorePreflightError::InvalidSegmentId);
        }
        if self.owners[segment_index] as usize >= self.config.owner_count {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        let Some(first_pair) = segment_index.checked_mul(PAIRS_PER_SEGMENT) else {
            return Err(TemporalStorePreflightError::BackingInvariant);
        };
        let Some(pair_end) = first_pair.checked_add(PAIRS_PER_SEGMENT) else {
            return Err(TemporalStorePreflightError::BackingInvariant);
        };
        if pair_end > self.sources.len() || pair_end > self.permanences.len() {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        let row = self.copy_runtime_segment_row(first_pair, pair_end);
        if self.validate_normalized_row(&row).is_err() {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        Ok(row)
    }

    /// Return the largest declared source fanout after validating canonical
    /// rows, empty-slot links, and aggregate posting cardinality.
    ///
    /// A checkpoint codec can fallibly allocate this many `u32` pair slots
    /// once, then reuse that scratch for every semantic-index traversal.
    pub(crate) fn runtime_semantic_posting_scratch_required(
        &self,
    ) -> Result<u32, TemporalStorePreflightError> {
        let (required, _) = self.semantic_posting_shape()?;
        u32::try_from(required).map_err(|_| TemporalStorePreflightError::BackingInvariant)
    }

    /// Visit the complete logical posting relation in canonical source-major
    /// order without allocating.
    ///
    /// The first pass validates every canonical row, every declared source
    /// chain, and exact global coverage before the first callback. Pair slots
    /// are unique within a valid chain; because every visited slot is also
    /// proved to store the current source, chains for distinct sources are
    /// disjoint. Equality between the total visits and all live canonical row
    /// slots therefore proves exact one-to-one coverage. A second immutable
    /// pass sorts each source's slots in caller-owned reusable scratch and
    /// emits `(source, segment, slot)` lexicographically.
    pub(crate) fn visit_runtime_semantic_postings<F>(
        &self,
        pair_slot_scratch: &mut [u32],
        mut visitor: F,
    ) -> Result<u32, TemporalStorePreflightError>
    where
        F: FnMut(TemporalSemanticPosting),
    {
        let (scratch_required, logical_live_pairs) = self.semantic_posting_shape()?;
        if pair_slot_scratch.len() < scratch_required {
            return Err(TemporalStorePreflightError::OutputCapacity {
                required: scratch_required as u32,
                provided: pair_slot_scratch.len() as u32,
            });
        }

        let mut validated_pairs = 0usize;
        for source_index in 0..self.config.source_count {
            let source = source_index as u32;
            let visited = self.validate_source_posting_chain_into::<true>(
                source,
                None,
                Some(pair_slot_scratch),
            )?;
            pair_slot_scratch[..visited].sort_unstable();
            if pair_slot_scratch[..visited]
                .windows(2)
                .any(|pair| pair[0] == pair[1])
            {
                return Err(TemporalStorePreflightError::BackingInvariant);
            }
            validated_pairs = validated_pairs
                .checked_add(visited)
                .ok_or(TemporalStorePreflightError::BackingInvariant)?;
        }
        if validated_pairs != logical_live_pairs {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }

        let mut emitted_pairs = 0usize;
        for source_index in 0..self.config.source_count {
            let source = source_index as u32;
            let visited = self.validate_source_posting_chain_into::<true>(
                source,
                None,
                Some(pair_slot_scratch),
            )?;
            pair_slot_scratch[..visited].sort_unstable();
            for pair_slot in pair_slot_scratch[..visited].iter().copied() {
                visitor(TemporalSemanticPosting {
                    normalized_source: source,
                    segment_id: pair_slot / PAIRS_PER_SEGMENT as u32,
                    slot: pair_slot % PAIRS_PER_SEGMENT as u32,
                });
                emitted_pairs += 1;
            }
        }
        if emitted_pairs != logical_live_pairs {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        u32::try_from(emitted_pairs).map_err(|_| TemporalStorePreflightError::BackingInvariant)
    }

    fn semantic_posting_shape(&self) -> Result<(usize, usize), TemporalStorePreflightError> {
        self.preflight_runtime_backing_shape()?;

        let mut logical_live_pairs = 0usize;
        for segment_index in 0..self.owners.len() {
            let row = self.runtime_segment_row(segment_index as u32)?;
            let live_pair_count = usize::from(row.live_pair_count);
            logical_live_pairs = logical_live_pairs
                .checked_add(live_pair_count)
                .ok_or(TemporalStorePreflightError::BackingInvariant)?;
            let first_pair = segment_index
                .checked_mul(PAIRS_PER_SEGMENT)
                .ok_or(TemporalStorePreflightError::BackingInvariant)?;
            for pair_index in first_pair + live_pair_count..first_pair + PAIRS_PER_SEGMENT {
                if self.next_pair[pair_index] != NONE || self.prev_pair[pair_index] != NONE {
                    return Err(TemporalStorePreflightError::BackingInvariant);
                }
            }
        }

        let mut declared_postings = 0usize;
        let mut scratch_required = 0usize;
        for meta in &self.source_meta {
            let live_count = meta.live_count as usize;
            if live_count > self.owners.len()
                || (live_count == 0 && (meta.head != NONE || meta.tail != NONE))
                || (live_count != 0 && (meta.head == NONE || meta.tail == NONE))
            {
                return Err(TemporalStorePreflightError::BackingInvariant);
            }
            declared_postings = declared_postings
                .checked_add(live_count)
                .ok_or(TemporalStorePreflightError::BackingInvariant)?;
            scratch_required = scratch_required.max(live_count);
        }
        if declared_postings != logical_live_pairs {
            return Err(TemporalStorePreflightError::BackingInvariant);
        }
        Ok((scratch_required, logical_live_pairs))
    }

    fn copy_runtime_segment_row(&self, first_pair: usize, pair_end: usize) -> TemporalSegmentRow {
        let mut row = TemporalSegmentRow {
            normalized_sources: [NONE; PAIRS_PER_SEGMENT],
            permanence_ticks: [0; PAIRS_PER_SEGMENT],
            live_pair_count: 0,
        };
        row.normalized_sources
            .copy_from_slice(&self.sources[first_pair..pair_end]);
        row.permanence_ticks
            .copy_from_slice(&self.permanences[first_pair..pair_end]);
        row.live_pair_count = row
            .normalized_sources
            .iter()
            .position(|source| *source == NONE)
            .unwrap_or(PAIRS_PER_SEGMENT) as u8;
        row
    }

    /// Copy a canonical row during the no-fail commit phase.
    ///
    /// The caller may use this infallible seam only after the same transaction
    /// has preflighted `segment_id`, its canonical row, and the complete store
    /// backing. It performs no validation or allocation after commit entry.
    pub(crate) fn commit_prevalidated_runtime_segment_row(
        &self,
        segment_id: u32,
    ) -> TemporalSegmentRow {
        let first_pair = segment_id as usize * PAIRS_PER_SEGMENT;
        self.copy_runtime_segment_row(first_pair, first_pair + PAIRS_PER_SEGMENT)
    }

    /// Materialize every active segment from the latest recognition in
    /// ascending segment-ID order.
    ///
    /// Capacity is counted before the first output write, so a retry with a
    /// larger caller-owned buffer observes an untouched original buffer.
    pub(crate) fn runtime_active_segments_into(
        &self,
        out: &mut [u32],
    ) -> Result<usize, TemporalStorePreflightError> {
        let active_count = self
            .touched_segments
            .iter()
            .filter(|segment_id| {
                self.connected_contacts[**segment_id as usize] >= self.config.theta_active
            })
            .count();
        if out.len() < active_count {
            return Err(TemporalStorePreflightError::OutputCapacity {
                required: active_count as u32,
                provided: out.len() as u32,
            });
        }
        let mut output_index = 0;
        for segment_id in &self.touched_segments {
            if self.connected_contacts[*segment_id as usize] >= self.config.theta_active {
                out[output_index] = *segment_id;
                output_index += 1;
            }
        }
        Ok(active_count)
    }

    fn next_generation(&mut self) -> u32 {
        if self.generation == u32::MAX {
            self.source_stamps.fill(0);
            self.segment_stamps.fill(0);
            self.owner_stamps.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
        self.generation
    }

    fn matching_better(&self, candidate: u32, incumbent: u32) -> bool {
        if incumbent == NONE {
            return true;
        }
        let candidate_index = candidate as usize;
        let incumbent_index = incumbent as usize;
        let candidate_primary = self.potential_contacts[candidate_index];
        let incumbent_primary = self.potential_contacts[incumbent_index];
        let candidate_secondary = self.connected_contacts[candidate_index];
        let incumbent_secondary = self.connected_contacts[incumbent_index];
        candidate_primary > incumbent_primary
            || (candidate_primary == incumbent_primary
                && (candidate_secondary > incumbent_secondary
                    || (candidate_secondary == incumbent_secondary && candidate < incumbent)))
    }

    fn active_better(&self, candidate: u32, incumbent: u32) -> bool {
        if incumbent == NONE {
            return true;
        }
        let candidate_index = candidate as usize;
        let incumbent_index = incumbent as usize;
        let candidate_primary = self.connected_contacts[candidate_index];
        let incumbent_primary = self.connected_contacts[incumbent_index];
        let candidate_secondary = self.potential_contacts[candidate_index];
        let incumbent_secondary = self.potential_contacts[incumbent_index];
        candidate_primary > incumbent_primary
            || (candidate_primary == incumbent_primary
                && (candidate_secondary > incumbent_secondary
                    || (candidate_secondary == incumbent_secondary && candidate < incumbent)))
    }

    fn prepare(
        &mut self,
        context_sources: &[u32],
        out: *mut TemporalOwnerDiagnostic,
        out_capacity: usize,
    ) -> Result<TemporalPrepareSummary, u32> {
        if self.poisoned != TEMPORAL_OK {
            return Err(TEMPORAL_POISONED);
        }
        if context_sources.len() > self.config.source_count
            || context_sources
                .iter()
                .any(|source| self.normalize_source(*source).is_none())
        {
            return Err(TEMPORAL_INVALID);
        }

        self.posting_validation_generation = 0;
        let generation = self.next_generation();
        for source in context_sources {
            let Some(source_index) = self.normalize_source(*source).map(|value| value as usize)
            else {
                return Err(TEMPORAL_INVALID);
            };
            if self.source_stamps[source_index] == generation {
                return Err(TEMPORAL_INVALID);
            }
            self.source_stamps[source_index] = generation;
        }

        self.touched_segments.clear();
        self.touched_owners.clear();
        let mut postings_visited = 0u64;
        for source in context_sources {
            let Some(normalized_source) = self.normalize_source(*source) else {
                return Err(TEMPORAL_INVALID);
            };
            let source_index = normalized_source as usize;
            let Some(meta) = self.source_meta.get(source_index).copied() else {
                return Err(TEMPORAL_INVALID);
            };
            let live_count = meta.live_count as usize;
            if live_count > self.owners.len()
                || (live_count == 0 && (meta.head != NONE || meta.tail != NONE))
                || (live_count != 0 && (meta.head == NONE || meta.tail == NONE))
            {
                return Err(TEMPORAL_INVALID);
            }
            let mut pair_slot = meta.head;
            let mut previous = NONE;
            let mut visited = 0usize;
            while pair_slot != NONE {
                if visited >= live_count || visited >= self.sources.len() {
                    return Err(TEMPORAL_INVALID);
                }
                let pair_index = pair_slot as usize;
                if self.sources.get(pair_index).copied() != Some(normalized_source)
                    || self.prev_pair.get(pair_index).copied() != Some(previous)
                {
                    return Err(TEMPORAL_INVALID);
                }
                let segment_id = pair_slot / PAIRS_PER_SEGMENT as u32;
                let segment_index = segment_id as usize;
                if segment_index >= self.owners.len()
                    || segment_index >= self.segment_stamps.len()
                    || segment_index >= self.potential_contacts.len()
                    || segment_index >= self.connected_contacts.len()
                    || pair_index >= self.permanences.len()
                    || pair_index >= self.next_pair.len()
                {
                    return Err(TEMPORAL_INVALID);
                }
                if self.segment_stamps[segment_index] != generation {
                    self.segment_stamps[segment_index] = generation;
                    self.potential_contacts[segment_index] = 0;
                    self.connected_contacts[segment_index] = 0;
                    self.touched_segments.push(segment_id);
                }
                self.potential_contacts[segment_index] += PHYSICAL_CONTACTS_PER_PAIR;
                if self.permanences[pair_slot as usize] >= self.config.connected_permanence_ticks {
                    self.connected_contacts[segment_index] += PHYSICAL_CONTACTS_PER_PAIR;
                }
                postings_visited = postings_visited.checked_add(1).ok_or(TEMPORAL_INVALID)?;
                let next = self.next_pair[pair_index];
                if next == NONE {
                    if pair_slot != meta.tail {
                        return Err(TEMPORAL_INVALID);
                    }
                } else {
                    let next_index = next as usize;
                    if self.sources.get(next_index).copied() != Some(normalized_source)
                        || self.prev_pair.get(next_index).copied() != Some(pair_slot)
                    {
                        return Err(TEMPORAL_INVALID);
                    }
                }
                previous = pair_slot;
                pair_slot = next;
                visited += 1;
            }
            if visited != live_count || previous != meta.tail {
                return Err(TEMPORAL_INVALID);
            }
        }
        self.touched_segments.sort_unstable();

        let mut matching_segments = 0usize;
        let mut active_segments = 0usize;
        let mut max_potential_contacts = 0u8;
        let mut max_connected_contacts = 0u8;
        for touched_index in 0..self.touched_segments.len() {
            let segment_id = self.touched_segments[touched_index];
            let segment_index = segment_id as usize;
            let potential = self.potential_contacts[segment_index];
            let connected = self.connected_contacts[segment_index];
            max_potential_contacts = max_potential_contacts.max(potential);
            max_connected_contacts = max_connected_contacts.max(connected);
            let matching = potential >= self.config.theta_match;
            let active = connected >= self.config.theta_active;
            matching_segments += usize::from(matching);
            active_segments += usize::from(active);
            if !matching && !active {
                continue;
            }
            let owner = self.owners[segment_index];
            let owner_index = owner as usize;
            if self.owner_stamps[owner_index] != generation {
                self.owner_stamps[owner_index] = generation;
                self.best_matching[owner_index] = NONE;
                self.best_active[owner_index] = NONE;
                self.touched_owners.push(owner);
            }
            if matching && self.matching_better(segment_id, self.best_matching[owner_index]) {
                self.best_matching[owner_index] = segment_id;
            }
            if active && self.active_better(segment_id, self.best_active[owner_index]) {
                self.best_active[owner_index] = segment_id;
            }
        }
        self.touched_owners.sort_unstable();
        if self.touched_owners.len() > out_capacity
            || (!self.touched_owners.is_empty() && out.is_null())
        {
            return Err(TEMPORAL_OUTPUT_CAPACITY);
        }
        for (index, owner) in self.touched_owners.iter().enumerate() {
            let owner_index = *owner as usize;
            let matching_id = self.best_matching[owner_index];
            let active_id = self.best_active[owner_index];
            let (matching_potential, matching_connected) = if matching_id == NONE {
                (0, 0)
            } else {
                (
                    self.potential_contacts[matching_id as usize],
                    self.connected_contacts[matching_id as usize],
                )
            };
            let (active_potential, active_connected) = if active_id == NONE {
                (0, 0)
            } else {
                (
                    self.potential_contacts[active_id as usize],
                    self.connected_contacts[active_id as usize],
                )
            };
            unsafe {
                *out.add(index) = TemporalOwnerDiagnostic {
                    owner: *owner,
                    best_matching_segment: matching_id,
                    best_active_segment: active_id,
                    matching_potential_contacts: matching_potential,
                    matching_connected_contacts: matching_connected,
                    active_potential_contacts: active_potential,
                    active_connected_contacts: active_connected,
                };
            }
        }
        Ok(TemporalPrepareSummary {
            candidate_segments: self.touched_segments.len(),
            matching_segments,
            active_segments,
            owner_result_count: self.touched_owners.len(),
            postings_visited,
            max_potential_contacts,
            max_connected_contacts,
            padding_zero: [0; 6],
        })
    }

    /// Fill runtime-owned recognition scratch without exposing a prepared
    /// classification or mutation seam to callers.
    pub(crate) fn recognize_runtime_context(
        &mut self,
        context_sources: &[u32],
        out: &mut [TemporalOwnerDiagnostic],
    ) -> Result<TemporalPrepareSummary, u32> {
        self.prepare(context_sources, out.as_mut_ptr(), out.len())
    }
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_store_create(
    config: *const TemporalStoreConfig,
) -> *mut TemporalStore {
    if config.is_null() {
        return ptr::null_mut();
    }
    let Ok(store) = TemporalStore::new(*config) else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(store))
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_store_destroy(store: *mut TemporalStore) {
    if !store.is_null() {
        drop(Box::from_raw(store));
    }
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_store_allocate(
    store: *mut TemporalStore,
    owner: u32,
    sources: *const u32,
    permanence_ticks: *const u16,
    pair_count: usize,
    segment_id_out: *mut u32,
) -> u32 {
    if store.is_null()
        || segment_id_out.is_null()
        || pair_count != PAIRS_PER_SEGMENT
        || sources.is_null()
        || permanence_ticks.is_null()
    {
        return TEMPORAL_INVALID;
    }
    let sources = slice::from_raw_parts(sources, pair_count);
    let permanence_ticks = slice::from_raw_parts(permanence_ticks, pair_count);
    match (*store).allocate_segment(owner, sources, permanence_ticks) {
        Ok(segment_id) => {
            *segment_id_out = segment_id;
            TEMPORAL_OK
        }
        Err(status) => status,
    }
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_temporal_store_prepare(
    store: *mut TemporalStore,
    context_sources: *const u32,
    context_count: usize,
    out: *mut TemporalOwnerDiagnostic,
    out_capacity: usize,
    summary_out: *mut TemporalPrepareSummary,
) -> u32 {
    if store.is_null()
        || summary_out.is_null()
        || (context_count != 0 && context_sources.is_null())
        || (out_capacity != 0 && out.is_null())
        || context_count > (*store).config.source_count
    {
        return TEMPORAL_INVALID;
    }
    let context = if context_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(context_sources, context_count)
    };
    match (*store).prepare(context, out, out_capacity) {
        Ok(summary) => {
            *summary_out = summary;
            TEMPORAL_OK
        }
        Err(status) => status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random123::{philox2x64_10, sample_without_replacement};
    use std::{mem::size_of, ptr};

    const INITIAL_CONTACT_DOMAIN: u64 = 0x5233_4630_494E_4954; // "R3F0INIT"
    const REGROWTH_CONTACT_DOMAIN: u64 = 0x5233_4630_5247_5257; // "R3F0RGRW"

    fn config(
        source_count: usize,
        owner_count: usize,
        segment_capacity: usize,
    ) -> TemporalStoreConfig {
        TemporalStoreConfig {
            source_count,
            owner_count,
            segment_capacity,
            source_gid_base: 100,
            connected_permanence_ticks: 250,
            theta_match: 2,
            theta_active: 2,
            padding_zero: 0,
        }
    }

    fn sources() -> [u32; PAIRS_PER_SEGMENT] {
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    }

    fn empty_diagnostic() -> TemporalOwnerDiagnostic {
        TemporalOwnerDiagnostic {
            owner: NONE,
            best_matching_segment: NONE,
            best_active_segment: NONE,
            matching_potential_contacts: 0,
            matching_connected_contacts: 0,
            active_potential_contacts: 0,
            active_connected_contacts: 0,
        }
    }

    fn posting_slots(store: &TemporalStore, normalized_source: usize) -> Vec<u32> {
        let meta = store.source_meta[normalized_source];
        let mut slots = Vec::new();
        let mut previous = NONE;
        let mut pair_slot = meta.head;
        while pair_slot != NONE {
            assert_eq!(store.prev_pair[pair_slot as usize], previous);
            assert_eq!(store.sources[pair_slot as usize], normalized_source as u32);
            slots.push(pair_slot);
            previous = pair_slot;
            pair_slot = store.next_pair[pair_slot as usize];
            assert!(slots.len() <= meta.live_count as usize, "posting cycle");
        }
        assert_eq!(previous, meta.tail);
        assert_eq!(slots.len(), meta.live_count as usize);

        let mut reverse = Vec::new();
        let mut next = NONE;
        pair_slot = meta.tail;
        while pair_slot != NONE {
            assert_eq!(store.next_pair[pair_slot as usize], next);
            reverse.push(pair_slot);
            next = pair_slot;
            pair_slot = store.prev_pair[pair_slot as usize];
            assert!(reverse.len() <= meta.live_count as usize, "posting cycle");
        }
        reverse.reverse();
        assert_eq!(reverse, slots);
        slots
    }

    fn assert_intrusive_index_integrity(store: &TemporalStore) {
        let mut visits = vec![0u8; store.sources.len()];
        for source in 0..store.config.source_count {
            for pair_slot in posting_slots(store, source) {
                visits[pair_slot as usize] += 1;
            }
        }
        for (pair_slot, source) in store.sources.iter().enumerate() {
            if *source == NONE {
                assert_eq!(store.permanences[pair_slot], 0);
                assert_eq!(store.next_pair[pair_slot], NONE);
                assert_eq!(store.prev_pair[pair_slot], NONE);
                assert_eq!(visits[pair_slot], 0);
            } else {
                assert!((*source as usize) < store.config.source_count);
                assert!((1..=500).contains(&store.permanences[pair_slot]));
                assert_eq!(visits[pair_slot], 1);
            }
        }
        for segment_sources in store.sources.chunks_exact(PAIRS_PER_SEGMENT) {
            let live_count = segment_sources
                .iter()
                .position(|source| *source == NONE)
                .unwrap_or(PAIRS_PER_SEGMENT);
            assert!(segment_sources[..live_count]
                .windows(2)
                .all(|pair| pair[0] < pair[1]));
            assert!(segment_sources[live_count..]
                .iter()
                .all(|source| *source == NONE));
        }
    }

    fn normalized_row(sources: &[u32], ticks: &[u16]) -> TemporalSegmentRow {
        assert_eq!(sources.len(), ticks.len());
        assert!(sources.len() <= PAIRS_PER_SEGMENT);
        let mut row = TemporalSegmentRow {
            normalized_sources: [NONE; PAIRS_PER_SEGMENT],
            permanence_ticks: [0; PAIRS_PER_SEGMENT],
            live_pair_count: sources.len() as u8,
        };
        row.normalized_sources[..sources.len()].copy_from_slice(sources);
        row.permanence_ticks[..ticks.len()].copy_from_slice(ticks);
        row
    }

    fn backing_layout(store: &TemporalStore) -> [(usize, usize); 16] {
        [
            (store.owners.as_ptr() as usize, store.owners.capacity()),
            (store.sources.as_ptr() as usize, store.sources.capacity()),
            (
                store.permanences.as_ptr() as usize,
                store.permanences.capacity(),
            ),
            (
                store.owner_segment_counts.as_ptr() as usize,
                store.owner_segment_counts.capacity(),
            ),
            (
                store.source_meta.as_ptr() as usize,
                store.source_meta.capacity(),
            ),
            (
                store.next_pair.as_ptr() as usize,
                store.next_pair.capacity(),
            ),
            (
                store.prev_pair.as_ptr() as usize,
                store.prev_pair.capacity(),
            ),
            (
                store.source_stamps.as_ptr() as usize,
                store.source_stamps.capacity(),
            ),
            (
                store.segment_stamps.as_ptr() as usize,
                store.segment_stamps.capacity(),
            ),
            (
                store.potential_contacts.as_ptr() as usize,
                store.potential_contacts.capacity(),
            ),
            (
                store.connected_contacts.as_ptr() as usize,
                store.connected_contacts.capacity(),
            ),
            (
                store.touched_segments.as_ptr() as usize,
                store.touched_segments.capacity(),
            ),
            (
                store.owner_stamps.as_ptr() as usize,
                store.owner_stamps.capacity(),
            ),
            (
                store.best_matching.as_ptr() as usize,
                store.best_matching.capacity(),
            ),
            (
                store.best_active.as_ptr() as usize,
                store.best_active.capacity(),
            ),
            (
                store.touched_owners.as_ptr() as usize,
                store.touched_owners.capacity(),
            ),
        ]
    }

    fn canonical_logical_bytes(store: &TemporalStore) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&store.runtime_segment_count().to_le_bytes());
        for segment_id in 0..store.runtime_segment_count() {
            bytes.extend_from_slice(
                &store
                    .runtime_segment_owner(segment_id)
                    .unwrap()
                    .to_le_bytes(),
            );
            let row = store.runtime_segment_row(segment_id).unwrap();
            bytes.push(row.live_pair_count);
            for source in row.normalized_sources {
                bytes.extend_from_slice(&source.to_le_bytes());
            }
            for ticks in row.permanence_ticks {
                bytes.extend_from_slice(&ticks.to_le_bytes());
            }
        }
        bytes
    }

    fn test_word(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    fn test_sample(state: &mut u64, population: usize, count: usize) -> Vec<u32> {
        let mut values = (0..population as u32).collect::<Vec<_>>();
        for draw in 0..count {
            let remaining = population - draw;
            let selected = draw + test_word(state) as usize % remaining;
            values.swap(draw, selected);
        }
        values[..count].sort_unstable();
        values.truncate(count);
        values
    }

    type DiagnosticTuple = (u32, u32, u32, u8, u8, u8, u8);

    fn diagnostic_tuple(value: &TemporalOwnerDiagnostic) -> DiagnosticTuple {
        (
            value.owner,
            value.best_matching_segment,
            value.best_active_segment,
            value.matching_potential_contacts,
            value.matching_connected_contacts,
            value.active_potential_contacts,
            value.active_connected_contacts,
        )
    }

    fn dense_prepare_reference(
        store: &TemporalStore,
        segments: &[(u32, Vec<u32>, Vec<u16>)],
        context: &[u32],
    ) -> ((usize, usize, usize, u64, u8, u8), Vec<DiagnosticTuple>) {
        let mut potential = vec![0u8; segments.len()];
        let mut connected = vec![0u8; segments.len()];
        let mut postings_visited = 0u64;
        for (segment_id, (_owner, sources, ticks)) in segments.iter().enumerate() {
            for (source, permanence) in sources.iter().zip(ticks) {
                if context.contains(source) {
                    potential[segment_id] += PHYSICAL_CONTACTS_PER_PAIR;
                    if *permanence >= store.config.connected_permanence_ticks {
                        connected[segment_id] += PHYSICAL_CONTACTS_PER_PAIR;
                    }
                    postings_visited += 1;
                }
            }
        }
        let candidate_segments = potential.iter().filter(|count| **count != 0).count();
        let matching_segments = potential
            .iter()
            .filter(|count| **count >= store.config.theta_match)
            .count();
        let active_segments = connected
            .iter()
            .filter(|count| **count >= store.config.theta_active)
            .count();
        let mut diagnostics = Vec::new();
        for owner in 0..store.config.owner_count as u32 {
            let mut best_matching = NONE;
            let mut best_active = NONE;
            for (segment_id, (segment_owner, _sources, _ticks)) in segments.iter().enumerate() {
                if *segment_owner != owner {
                    continue;
                }
                let segment_id = segment_id as u32;
                if potential[segment_id as usize] >= store.config.theta_match
                    && (best_matching == NONE
                        || potential[segment_id as usize] > potential[best_matching as usize]
                        || (potential[segment_id as usize] == potential[best_matching as usize]
                            && (connected[segment_id as usize]
                                > connected[best_matching as usize]
                                || (connected[segment_id as usize]
                                    == connected[best_matching as usize]
                                    && segment_id < best_matching))))
                {
                    best_matching = segment_id;
                }
                if connected[segment_id as usize] >= store.config.theta_active
                    && (best_active == NONE
                        || connected[segment_id as usize] > connected[best_active as usize]
                        || (connected[segment_id as usize] == connected[best_active as usize]
                            && (potential[segment_id as usize] > potential[best_active as usize]
                                || (potential[segment_id as usize]
                                    == potential[best_active as usize]
                                    && segment_id < best_active))))
                {
                    best_active = segment_id;
                }
            }
            if best_matching != NONE || best_active != NONE {
                diagnostics.push((
                    owner,
                    best_matching,
                    best_active,
                    if best_matching == NONE {
                        0
                    } else {
                        potential[best_matching as usize]
                    },
                    if best_matching == NONE {
                        0
                    } else {
                        connected[best_matching as usize]
                    },
                    if best_active == NONE {
                        0
                    } else {
                        potential[best_active as usize]
                    },
                    if best_active == NONE {
                        0
                    } else {
                        connected[best_active as usize]
                    },
                ));
            }
        }
        (
            (
                candidate_segments,
                matching_segments,
                active_segments,
                postings_visited,
                potential.iter().copied().max().unwrap_or(0),
                connected.iter().copied().max().unwrap_or(0),
            ),
            diagnostics,
        )
    }

    #[test]
    fn shared_philox_sampler_matches_temporal_oracle_domains_and_vectors() {
        let mut initial_scratch = [0u64; 8];
        let initial_key = 17 ^ INITIAL_CONTACT_DOMAIN;
        let initial = sample_without_replacement(8, 4, initial_key, 1, &mut initial_scratch);
        assert_eq!(initial, Some(&[2, 3, 5, 7][..]),);

        let regrowth_key = philox2x64_10(44, 0, 17 ^ REGROWTH_CONTACT_DOMAIN).0;
        assert_eq!(regrowth_key, 0xC5AC_1D69_C2B8_FBC5);
        let mut regrowth_scratch = [0u64; 11];
        assert_eq!(
            sample_without_replacement(11, 3, regrowth_key, 3, &mut regrowth_scratch),
            Some(&[2, 6, 7][..]),
        );
    }

    #[test]
    fn ffi_struct_sizes_are_stable() {
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(size_of::<TemporalStoreConfig>(), 40);
            assert_eq!(size_of::<TemporalOwnerDiagnostic>(), 16);
            assert_eq!(size_of::<TemporalPrepareSummary>(), 48);
            assert_eq!(size_of::<TemporalStore>(), 440);
        }
        assert_eq!(size_of::<SourceMeta>(), 12);
    }

    #[test]
    fn c1_intrusive_store_logical_payload_layout_is_exact() {
        // Element payloads requested from Vec plus the separately pinned
        // TemporalStore header. Allocator metadata remains excluded.
        const SOURCES: usize = 131_072;
        const OWNERS: usize = 131_072;
        const SEGMENTS: usize = 23_286_406;
        const PAIRS: usize = SEGMENTS * PAIRS_PER_SEGMENT;
        const LEARNED_SOA: usize = SEGMENTS * size_of::<u32>()
            + PAIRS * size_of::<u32>()
            + PAIRS * size_of::<u16>()
            + OWNERS * size_of::<u8>();
        const INTRUSIVE_INDEX: usize =
            SOURCES * size_of::<SourceMeta>() + PAIRS * size_of::<u32>() + PAIRS * size_of::<u32>();
        const SOURCE_SCRATCH: usize = SOURCES * size_of::<u32>();
        const SEGMENT_SCRATCH: usize = SEGMENTS * size_of::<u32>()
            + SEGMENTS * size_of::<u8>()
            + SEGMENTS * size_of::<u8>()
            + SEGMENTS * size_of::<u32>();
        const OWNER_SCRATCH: usize = OWNERS * size_of::<u32>() * 4;
        const NATIVE_RESERVED_PAYLOAD: usize =
            LEARNED_SOA + INTRUSIVE_INDEX + SOURCE_SCRATCH + SEGMENT_SCRATCH + OWNER_SCRATCH;
        const STORE_HEADER: usize = 440;
        const NATIVE_RESIDENT_BYTES: usize = NATIVE_RESERVED_PAYLOAD + STORE_HEADER;
        const PYTHON_DIAGNOSTIC_OUTPUT: usize = OWNERS * size_of::<TemporalOwnerDiagnostic>();

        assert_eq!(PAIRS, 232_864_060);
        assert_eq!(LEARNED_SOA, 1_490_461_056);
        assert_eq!(INTRUSIVE_INDEX, 1_864_485_344);
        assert_eq!(SOURCE_SCRATCH, 524_288);
        assert_eq!(SEGMENT_SCRATCH, 232_864_060);
        assert_eq!(OWNER_SCRATCH, 2_097_152);
        assert_eq!(NATIVE_RESERVED_PAYLOAD, 3_590_431_900);
        assert_eq!(STORE_HEADER, size_of::<TemporalStore>());
        assert_eq!(NATIVE_RESIDENT_BYTES, 3_590_432_340);
        assert_eq!(PYTHON_DIAGNOSTIC_OUTPUT, 2_097_152);
        assert_eq!(
            NATIVE_RESIDENT_BYTES + PYTHON_DIAGNOSTIC_OUTPUT,
            3_592_529_492
        );
    }

    #[test]
    fn ffi_rejects_null_and_invalid_inputs_without_poisoning() {
        unsafe {
            assert!(humanbrain_temporal_store_create(ptr::null()).is_null());
            humanbrain_temporal_store_destroy(ptr::null_mut());

            let mut invalid_config = config(10, 1, 1);
            invalid_config.source_gid_base = u32::MAX - 8;
            assert!(humanbrain_temporal_store_create(&invalid_config).is_null());
            invalid_config = config(10, 1, 1);
            invalid_config.connected_permanence_ticks = 0;
            assert!(humanbrain_temporal_store_create(&invalid_config).is_null());

            let valid_config = config(10, 1, 1);
            let store = humanbrain_temporal_store_create(&valid_config);
            assert!(!store.is_null());
            let mut segment_id = NONE;
            let mut summary = TemporalPrepareSummary {
                candidate_segments: 0,
                matching_segments: 0,
                active_segments: 0,
                owner_result_count: 0,
                postings_visited: 0,
                max_potential_contacts: 0,
                max_connected_contacts: 0,
                padding_zero: [0; 6],
            };
            assert_eq!(
                humanbrain_temporal_store_allocate(
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                    ptr::null(),
                    PAIRS_PER_SEGMENT,
                    &mut segment_id,
                ),
                TEMPORAL_INVALID
            );
            assert_eq!(
                humanbrain_temporal_store_allocate(
                    store,
                    0,
                    ptr::null(),
                    ptr::null(),
                    PAIRS_PER_SEGMENT,
                    &mut segment_id,
                ),
                TEMPORAL_INVALID
            );
            assert_eq!(
                humanbrain_temporal_store_prepare(
                    ptr::null_mut(),
                    ptr::null(),
                    0,
                    ptr::null_mut(),
                    0,
                    &mut summary,
                ),
                TEMPORAL_INVALID
            );
            assert_eq!(
                humanbrain_temporal_store_prepare(
                    store,
                    ptr::null(),
                    1,
                    ptr::null_mut(),
                    0,
                    &mut summary,
                ),
                TEMPORAL_INVALID
            );
            let oversized_context = [100; 11];
            assert_eq!(
                humanbrain_temporal_store_prepare(
                    store,
                    oversized_context.as_ptr(),
                    oversized_context.len(),
                    ptr::null_mut(),
                    0,
                    &mut summary,
                ),
                TEMPORAL_INVALID
            );
            assert_eq!(
                humanbrain_temporal_store_prepare(
                    store,
                    ptr::null(),
                    0,
                    ptr::null_mut(),
                    0,
                    &mut summary,
                ),
                TEMPORAL_OK
            );
            assert_eq!(summary.candidate_segments, 0);
            humanbrain_temporal_store_destroy(store);
        }
    }

    #[test]
    fn allocation_requires_canonical_physical_sources_and_stores_logical_ids() {
        let mut store = TemporalStore::new(config(20, 1, 1)).unwrap();
        let ticks = [250; PAIRS_PER_SEGMENT];
        let mut noncanonical = sources();
        noncanonical.swap(0, 1);
        assert_eq!(
            store.allocate_segment(0, &noncanonical, &ticks),
            Err(TEMPORAL_INVALID)
        );
        let mut duplicate = sources();
        duplicate[9] = duplicate[8];
        assert_eq!(
            store.allocate_segment(0, &duplicate, &ticks),
            Err(TEMPORAL_INVALID)
        );
        let mut out_of_domain = sources();
        out_of_domain[0] = 99;
        assert_eq!(
            store.allocate_segment(0, &out_of_domain, &ticks),
            Err(TEMPORAL_INVALID)
        );
        assert!(store.owners.is_empty());
        assert!(store.next_pair.is_empty());
        assert!(store.prev_pair.is_empty());

        assert_eq!(store.allocate_segment(0, &sources(), &ticks), Ok(0));
        assert_eq!(store.sources, (0..10).collect::<Vec<_>>());

        let mut output = [empty_diagnostic(); 1];
        let reversed = [109, 108, 107, 106, 105, 104, 103, 102, 101, 100];
        let summary = store
            .prepare(&reversed, output.as_mut_ptr(), output.len())
            .unwrap();
        assert_eq!(summary.candidate_segments, 1);
        assert_eq!(summary.postings_visited, 10);
        assert_eq!(output[0].owner, 0);
        assert!(matches!(
            store.prepare(&[100, 100], output.as_mut_ptr(), output.len()),
            Err(TEMPORAL_INVALID)
        ));
        assert_eq!(store.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn intrusive_posting_chain_crosses_63_64_65_fanout_exactly() {
        let mut store = TemporalStore::new(config(10, 65, 65)).unwrap();
        let ticks = [250; PAIRS_PER_SEGMENT];
        let mut output = vec![empty_diagnostic(); 65];

        for segment in 0..65 {
            assert_eq!(
                store.allocate_segment(segment, &sources(), &ticks),
                Ok(segment)
            );
            let count = segment as usize + 1;
            if [63, 64, 65].contains(&count) {
                let summary = store
                    .prepare(&[100], output.as_mut_ptr(), output.len())
                    .unwrap();
                assert_eq!(summary.candidate_segments, count);
                assert_eq!(summary.owner_result_count, count);
                assert_eq!(summary.postings_visited, count as u64);
                assert_eq!(store.source_meta[0].live_count, count as u32);
                assert_eq!(
                    posting_slots(&store, 0),
                    (0..count)
                        .map(|segment| (segment * PAIRS_PER_SEGMENT) as u32)
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn canonical_rewrite_unlinks_drops_and_regrows_without_stale_postings() {
        let mut store = TemporalStore::new(config(24, 2, 2)).unwrap();
        let ticks = [250; PAIRS_PER_SEGMENT];
        assert_eq!(store.allocate_segment(0, &sources(), &ticks), Ok(0));
        let second = [100, 110, 111, 112, 113, 114, 115, 116, 117, 118];
        assert_eq!(store.allocate_segment(1, &second, &ticks), Ok(1));
        assert_intrusive_index_integrity(&store);

        let rewritten = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
        let rewritten_ticks = [101, 102, 103, 104, 105, 255, 256, 257, 258, 259];
        assert_eq!(
            store.rewrite_segment_pairs(0, &rewritten, &rewritten_ticks),
            Ok(())
        );
        assert_eq!(&store.sources[..PAIRS_PER_SEGMENT], &rewritten);
        assert_eq!(&store.permanences[..PAIRS_PER_SEGMENT], &rewritten_ticks);
        assert_eq!(posting_slots(&store, 0), vec![10]);
        assert_eq!(posting_slots(&store, 2), vec![0]);
        assert_intrusive_index_integrity(&store);

        assert_eq!(store.rewrite_segment_pairs(0, &[], &[]), Ok(()));
        assert_eq!(
            &store.sources[..PAIRS_PER_SEGMENT],
            &[NONE; PAIRS_PER_SEGMENT]
        );
        assert_eq!(
            &store.permanences[..PAIRS_PER_SEGMENT],
            &[0; PAIRS_PER_SEGMENT]
        );
        assert_eq!(posting_slots(&store, 2), Vec::<u32>::new());
        assert_eq!(posting_slots(&store, 0), vec![10]);
        assert_intrusive_index_integrity(&store);
        let mut output = [empty_diagnostic(); 2];
        let dropped = store
            .prepare(&[102], output.as_mut_ptr(), output.len())
            .unwrap();
        assert_eq!(dropped.candidate_segments, 0);
        assert_eq!(dropped.postings_visited, 0);

        let regrown = [1, 3, 5];
        let regrown_ticks = [251, 252, 253];
        assert_eq!(
            store.rewrite_segment_pairs(0, &regrown, &regrown_ticks),
            Ok(())
        );
        assert_eq!(&store.sources[..3], &regrown);
        assert_eq!(&store.sources[3..PAIRS_PER_SEGMENT], &[NONE; 7]);
        assert_eq!(&store.permanences[..3], &regrown_ticks);
        assert_eq!(&store.permanences[3..PAIRS_PER_SEGMENT], &[0; 7]);
        assert_intrusive_index_integrity(&store);
        let recognized = store
            .prepare(&[101, 103, 105], output.as_mut_ptr(), output.len())
            .unwrap();
        assert_eq!(recognized.candidate_segments, 1);
        assert_eq!(recognized.postings_visited, 3);
        assert_eq!(recognized.owner_result_count, 1);
        assert_eq!(output[0].owner, 0);
        assert_eq!(output[0].best_active_segment, 0);

        assert_eq!(
            store.rewrite_segment_pairs(0, &regrown, &regrown_ticks),
            Ok(())
        );
        assert_intrusive_index_integrity(&store);

        let before = (
            store.sources.clone(),
            store.permanences.clone(),
            store.next_pair.clone(),
            store.prev_pair.clone(),
            store.source_meta.clone(),
        );
        assert_eq!(
            store.rewrite_segment_pairs(0, &[3, 3], &[250, 250]),
            Err(TEMPORAL_INVALID)
        );
        assert_eq!(
            store.rewrite_segment_pairs(2, &[2], &[250]),
            Err(TEMPORAL_INVALID)
        );
        assert_eq!(
            store.rewrite_segment_pairs(0, &[2], &[0]),
            Err(TEMPORAL_INVALID)
        );
        assert_eq!(
            (
                store.sources.clone(),
                store.permanences.clone(),
                store.next_pair.clone(),
                store.prev_pair.clone(),
                store.source_meta.clone(),
            ),
            before,
        );
        assert_intrusive_index_integrity(&store);
    }

    #[test]
    fn randomized_rewrites_match_independent_dense_recognition() {
        let mut frozen_config = config(128, 24, 64);
        frozen_config.theta_match = 8;
        frozen_config.theta_active = 12;
        let mut store = TemporalStore::new(frozen_config).unwrap();
        let mut state = 0xA13C_1A57_D15C_A11Du64;
        let mut segments = Vec::new();
        for segment_id in 0..64u32 {
            let owner = test_word(&mut state) as u32 % 24;
            let sources = test_sample(&mut state, 128, PAIRS_PER_SEGMENT);
            let ticks = (0..PAIRS_PER_SEGMENT)
                .map(|_| (test_word(&mut state) % 500 + 1) as u16)
                .collect::<Vec<_>>();
            let physical = sources
                .iter()
                .map(|source| source + frozen_config.source_gid_base)
                .collect::<Vec<_>>();
            assert_eq!(
                store.allocate_segment(owner, &physical, &ticks),
                Ok(segment_id)
            );
            segments.push((owner, sources, ticks));
        }

        let mut output = vec![empty_diagnostic(); frozen_config.owner_count];
        for round in 0..256 {
            let segment_id = test_word(&mut state) as usize % segments.len();
            let live_count = test_word(&mut state) as usize % (PAIRS_PER_SEGMENT + 1);
            let sources = test_sample(&mut state, frozen_config.source_count, live_count);
            let ticks = (0..live_count)
                .map(|_| (test_word(&mut state) % 500 + 1) as u16)
                .collect::<Vec<_>>();
            assert_eq!(
                store.rewrite_segment_pairs(segment_id as u32, &sources, &ticks),
                Ok(())
            );
            segments[segment_id].1 = sources;
            segments[segment_id].2 = ticks;
            assert_intrusive_index_integrity(&store);

            let context_count = test_word(&mut state) as usize % 33;
            let logical_context =
                test_sample(&mut state, frozen_config.source_count, context_count);
            let mut physical_context = logical_context
                .iter()
                .map(|source| source + frozen_config.source_gid_base)
                .collect::<Vec<_>>();
            if round % 2 != 0 {
                physical_context.reverse();
            }
            let summary = store
                .prepare(&physical_context, output.as_mut_ptr(), output.len())
                .unwrap();
            let (expected_summary, expected_diagnostics) =
                dense_prepare_reference(&store, &segments, &logical_context);
            assert_eq!(
                (
                    summary.candidate_segments,
                    summary.matching_segments,
                    summary.active_segments,
                    summary.postings_visited,
                    summary.max_potential_contacts,
                    summary.max_connected_contacts,
                ),
                expected_summary,
            );
            assert_eq!(
                output[..summary.owner_result_count]
                    .iter()
                    .map(diagnostic_tuple)
                    .collect::<Vec<_>>(),
                expected_diagnostics,
            );
        }
    }

    #[test]
    fn owner_255_limit_poison_is_fail_stop_and_transactional() {
        let mut store = TemporalStore::new(config(10, 1, 256)).unwrap();
        let ticks = [250; PAIRS_PER_SEGMENT];
        for segment in 0..u8::MAX as u32 {
            assert_eq!(store.allocate_segment(0, &sources(), &ticks), Ok(segment));
        }
        let owners_before = store.owners.len();
        let postings_before = store.sources.len();
        assert_eq!(
            store.allocate_segment(0, &sources(), &ticks),
            Err(TEMPORAL_OWNER_CAPACITY)
        );
        assert_eq!(store.owners.len(), owners_before);
        assert_eq!(store.sources.len(), postings_before);
        assert_eq!(store.poisoned, TEMPORAL_OWNER_CAPACITY);
        assert!(matches!(
            store.prepare(&[100], ptr::null_mut(), 0),
            Err(TEMPORAL_POISONED)
        ));
        assert_eq!(
            store.allocate_segment(0, &sources(), &ticks),
            Err(TEMPORAL_POISONED)
        );
    }

    #[test]
    fn output_capacity_error_is_retryable_without_poisoning() {
        let mut store = TemporalStore::new(config(10, 2, 2)).unwrap();
        let ticks = [250; PAIRS_PER_SEGMENT];
        assert_eq!(store.allocate_segment(1, &sources(), &ticks), Ok(0));
        assert_eq!(store.allocate_segment(0, &sources(), &ticks), Ok(1));
        let mut output = vec![empty_diagnostic(); 2];

        assert!(matches!(
            store.prepare(&sources(), output.as_mut_ptr(), 1),
            Err(TEMPORAL_OUTPUT_CAPACITY)
        ));
        assert_eq!(store.poisoned, TEMPORAL_OK);
        let summary = store
            .prepare(&sources(), output.as_mut_ptr(), output.len())
            .unwrap();
        assert_eq!(summary.owner_result_count, 2);
        assert_eq!([output[0].owner, output[1].owner], [0, 1]);
        assert_eq!(
            [output[0].best_active_segment, output[1].best_active_segment],
            [1, 0]
        );
    }

    #[test]
    fn generation_rollover_clears_all_dense_stamps() {
        let mut store = TemporalStore::new(config(10, 1, 1)).unwrap();
        assert_eq!(
            store.allocate_segment(0, &sources(), &[250; PAIRS_PER_SEGMENT]),
            Ok(0)
        );
        store.generation = u32::MAX;
        store.source_stamps.fill(u32::MAX);
        store.segment_stamps.fill(u32::MAX);
        store.owner_stamps.fill(u32::MAX);
        let mut output = [empty_diagnostic(); 1];

        let summary = store
            .prepare(&sources(), output.as_mut_ptr(), output.len())
            .unwrap();
        assert_eq!(store.generation, 1);
        assert_eq!(summary.candidate_segments, 1);
        assert!(store.source_stamps.iter().all(|stamp| *stamp == 1));
        assert_eq!(store.segment_stamps, vec![1]);
        assert_eq!(store.owner_stamps, vec![1]);
    }

    #[test]
    fn runtime_introspection_copies_rows_and_materializes_every_active_segment_in_order() {
        let mut store = TemporalStore::new(config(20, 3, 4)).unwrap();
        assert_eq!(
            store.allocate_runtime_segment(2, &[100, 101], &[250, 251]),
            Ok(0)
        );
        assert_eq!(store.allocate_runtime_segment(0, &[100], &[300]), Ok(1));
        assert_eq!(store.allocate_runtime_segment(2, &[101], &[400]), Ok(2));
        assert_eq!(store.allocate_runtime_segment(1, &[], &[]), Ok(3));

        assert_eq!(store.runtime_segment_count(), 4);
        assert_eq!(store.runtime_segment_capacity(), 4);
        assert_eq!(store.runtime_segment_owner(0), Some(2));
        assert_eq!(store.runtime_segment_owner(3), Some(1));
        assert_eq!(store.runtime_segment_owner(4), None);
        assert_eq!(store.runtime_owner_load(2), Some(2));
        assert_eq!(store.runtime_owner_load(1), Some(1));

        let first = store.runtime_segment_row(0).unwrap();
        assert_eq!(first.live_pair_count, 2);
        assert_eq!(&first.normalized_sources[..2], &[0, 1]);
        assert_eq!(&first.permanence_ticks[..2], &[250, 251]);
        assert_eq!(
            &first.normalized_sources[2..],
            &[NONE; PAIRS_PER_SEGMENT - 2]
        );
        assert_eq!(&first.permanence_ticks[2..], &[0; PAIRS_PER_SEGMENT - 2]);
        let empty = store.runtime_segment_row(3).unwrap();
        assert_eq!(empty.live_pair_count, 0);
        assert_eq!(empty.normalized_sources, [NONE; PAIRS_PER_SEGMENT]);
        assert_eq!(empty.permanence_ticks, [0; PAIRS_PER_SEGMENT]);
        assert_eq!(
            store.runtime_segment_row(4),
            Err(TemporalStorePreflightError::InvalidSegmentId)
        );

        let mut diagnostics = [empty_diagnostic(); 3];
        let summary = store
            .recognize_runtime_context(&[101, 100], &mut diagnostics)
            .unwrap();
        assert_eq!(summary.active_segments, 3);
        assert_eq!(diagnostics[1].owner, 2);
        assert_eq!(diagnostics[1].best_active_segment, 0);

        let mut too_small = [0xDEAD_BEEF; 2];
        assert_eq!(
            store.runtime_active_segments_into(&mut too_small),
            Err(TemporalStorePreflightError::OutputCapacity {
                required: 3,
                provided: 2,
            })
        );
        assert_eq!(too_small, [0xDEAD_BEEF; 2]);

        let mut active = [NONE; 4];
        assert_eq!(store.runtime_active_segments_into(&mut active), Ok(3));
        assert_eq!(active, [0, 1, 2, NONE]);
        assert_eq!(store.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn prevalidated_commits_append_rewrite_drop_and_regrow_without_reallocation() {
        let mut store = TemporalStore::new(config(24, 3, 4)).unwrap();
        let row_a = normalized_row(&[0, 2, 4], &[250, 300, 350]);
        let empty = normalized_row(&[], &[]);
        store.preflight_runtime_canonical_append(2, &row_a).unwrap();
        store.preflight_runtime_canonical_append(1, &empty).unwrap();
        let append_preflight = store
            .preflight_runtime_mutation_backing(&[2, 1], 2)
            .unwrap();
        assert_eq!(append_preflight.first_append_segment_id(), 0);
        assert_eq!(append_preflight.prospective_append_count(), 2);

        let layout = backing_layout(&store);
        assert_eq!(
            store.commit_prevalidated_runtime_canonical_append(2, &row_a),
            0
        );
        assert_eq!(
            store.commit_prevalidated_runtime_canonical_append(1, &empty),
            1
        );
        assert_eq!(backing_layout(&store), layout);
        assert_eq!(store.runtime_owner_load(2), Some(1));
        assert_eq!(store.runtime_owner_load(1), Some(1));
        assert_eq!(store.runtime_segment_row(1).unwrap(), empty);
        assert_intrusive_index_integrity(&store);

        let rewritten = normalized_row(&[1, 3, 5], &[251, 252, 253]);
        store
            .preflight_runtime_canonical_rewrite(0, &rewritten)
            .unwrap();
        store.preflight_runtime_mutation_backing(&[], 2).unwrap();
        store.commit_prevalidated_runtime_canonical_rewrite(0, &rewritten);
        assert_eq!(backing_layout(&store), layout);
        assert_eq!(store.runtime_segment_row(0).unwrap(), rewritten);
        assert_eq!(posting_slots(&store, 0), Vec::<u32>::new());
        assert_eq!(posting_slots(&store, 1), vec![0]);
        assert_intrusive_index_integrity(&store);

        store
            .preflight_runtime_canonical_rewrite(0, &empty)
            .unwrap();
        store.commit_prevalidated_runtime_canonical_rewrite(0, &empty);
        assert_eq!(backing_layout(&store), layout);
        assert_eq!(store.runtime_segment_owner(0), Some(2));
        assert_eq!(store.runtime_owner_load(2), Some(1));
        assert_eq!(store.runtime_segment_row(0).unwrap(), empty);
        assert_eq!(posting_slots(&store, 1), Vec::<u32>::new());
        assert_intrusive_index_integrity(&store);

        let regrown = normalized_row(&[6, 8], &[499, 500]);
        store
            .preflight_runtime_canonical_rewrite(0, &regrown)
            .unwrap();
        store.commit_prevalidated_runtime_canonical_rewrite(0, &regrown);
        assert_eq!(backing_layout(&store), layout);
        assert_eq!(store.runtime_segment_row(0).unwrap(), regrown);
        assert_eq!(posting_slots(&store, 6), vec![0]);
        assert_eq!(posting_slots(&store, 8), vec![1]);
        assert_intrusive_index_integrity(&store);

        let mut diagnostics = [empty_diagnostic(); 3];
        let summary = store
            .recognize_runtime_context(&[106, 108], &mut diagnostics)
            .unwrap();
        assert_eq!(summary.candidate_segments, 1);
        assert_eq!(summary.active_segments, 1);
        assert_eq!(diagnostics[0].owner, 2);
        assert_eq!(diagnostics[0].best_active_segment, 0);
        assert_eq!(store.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn commit_prevalidated_reads_return_authoritative_owner_and_canonical_row() {
        let mut store = TemporalStore::new(config(12, 3, 2)).unwrap();
        let row = normalized_row(&[0, 3, 7], &[250, 375, 500]);
        assert_eq!(
            store.allocate_runtime_segment(2, &[100, 103, 107], &[250, 375, 500]),
            Ok(0)
        );
        store.preflight_runtime_mutation_backing(&[], 2).unwrap();
        store.preflight_runtime_canonical_rewrite(0, &row).unwrap();

        assert_eq!(store.commit_prevalidated_runtime_segment_owner(0), 2);
        assert_eq!(store.commit_prevalidated_runtime_segment_row(0), row);
    }

    #[test]
    fn mutation_preflight_failures_are_nonpoisoning_atomic_and_retryable() {
        let mut store = TemporalStore::new(config(10, 2, 256)).unwrap();
        let empty = normalized_row(&[], &[]);
        for segment_id in 0..254 {
            assert_eq!(store.allocate_runtime_segment(0, &[], &[]), Ok(segment_id));
        }
        store.preflight_runtime_canonical_append(0, &empty).unwrap();
        let last_owner_zero = store
            .preflight_runtime_mutation_backing(&[0], u8::MAX)
            .unwrap();
        assert_eq!(last_owner_zero.first_append_segment_id(), 254);
        assert_eq!(last_owner_zero.prospective_append_count(), 1);
        assert_eq!(
            store.commit_prevalidated_runtime_canonical_append(0, &empty),
            254
        );
        assert_eq!(store.runtime_owner_load(0), Some(u8::MAX));

        let logical_before = canonical_logical_bytes(&store);
        let index_before = (
            store.source_meta.clone(),
            store.next_pair.clone(),
            store.prev_pair.clone(),
        );
        assert_eq!(
            store.preflight_runtime_mutation_backing(&[0], u8::MAX),
            Err(TemporalStorePreflightError::OwnerCapacity { owner: 0 })
        );
        assert_eq!(canonical_logical_bytes(&store), logical_before);
        assert_eq!(
            (
                store.source_meta.clone(),
                store.next_pair.clone(),
                store.prev_pair.clone(),
            ),
            index_before
        );
        assert_eq!(store.poisoned, TEMPORAL_OK);

        store.preflight_runtime_canonical_append(1, &empty).unwrap();
        let retry = store
            .preflight_runtime_mutation_backing(&[1], u8::MAX)
            .unwrap();
        assert_eq!(retry.first_append_segment_id(), 255);
        assert_eq!(retry.prospective_append_count(), 1);
        assert_eq!(
            store.commit_prevalidated_runtime_canonical_append(1, &empty),
            255
        );
        assert_eq!(store.runtime_owner_load(1), Some(1));
        assert_eq!(store.runtime_segment_row(255).unwrap(), empty);

        let full_logical = canonical_logical_bytes(&store);
        assert_eq!(
            store.preflight_runtime_mutation_backing(&[1], u8::MAX),
            Err(TemporalStorePreflightError::SegmentCapacity)
        );
        assert_eq!(canonical_logical_bytes(&store), full_logical);
        assert_eq!(store.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn live_physical_source_domain_reserves_u32_max_for_the_sentinel() {
        let mut includes_sentinel = config(10, 1, 1);
        includes_sentinel.source_gid_base = u32::MAX - 9;
        assert!(matches!(
            TemporalStore::new(includes_sentinel),
            Err(TEMPORAL_INVALID)
        ));

        let mut below_sentinel = config(10, 1, 1);
        below_sentinel.source_gid_base = u32::MAX - 10;
        let mut store = TemporalStore::new(below_sentinel).unwrap();
        let sources: [u32; PAIRS_PER_SEGMENT] =
            std::array::from_fn(|index| below_sentinel.source_gid_base + index as u32);
        assert_eq!(sources[PAIRS_PER_SEGMENT - 1], u32::MAX - 1);
        assert_eq!(
            store.allocate_runtime_segment(0, &sources, &[250; PAIRS_PER_SEGMENT]),
            Ok(0)
        );
    }

    #[test]
    fn append_preflight_rejects_a_stale_destination_tail_without_mutation() {
        let mut store = TemporalStore::new(config(10, 2, 3)).unwrap();
        assert_eq!(store.allocate_runtime_segment(0, &[100], &[250]), Ok(0));
        store.source_meta[0].tail = store.sources.len() as u32;
        let logical_before = canonical_logical_bytes(&store);
        let segment_count_before = store.runtime_segment_count();
        let owner_load_before = store.runtime_owner_load(1);
        let row = normalized_row(&[0], &[250]);

        store.preflight_runtime_mutation_backing(&[1], 2).unwrap();
        assert_eq!(
            store.preflight_runtime_canonical_append(1, &row),
            Err(TemporalStorePreflightError::BackingInvariant)
        );
        assert_eq!(canonical_logical_bytes(&store), logical_before);
        assert_eq!(store.runtime_segment_count(), segment_count_before);
        assert_eq!(store.runtime_owner_load(1), owner_load_before);
        assert_eq!(store.poisoned, TEMPORAL_OK);
    }

    fn reciprocal_three_node_cycle() -> TemporalStore {
        let mut store = TemporalStore::new(config(10, 3, 4)).unwrap();
        for owner in 0..3 {
            assert_eq!(
                store.allocate_runtime_segment(owner, &[100], &[250]),
                Ok(owner)
            );
        }
        store.prev_pair[0] = 20;
        store.next_pair[0] = 10;
        store.prev_pair[10] = 0;
        store.next_pair[10] = 20;
        store.prev_pair[20] = 10;
        store.next_pair[20] = 0;
        store
    }

    #[test]
    fn rewrite_preflight_rejects_a_reciprocal_three_node_cycle_without_mutation() {
        let mut store = reciprocal_three_node_cycle();
        let logical_before = canonical_logical_bytes(&store);
        let owner_loads_before = [
            store.runtime_owner_load(0),
            store.runtime_owner_load(1),
            store.runtime_owner_load(2),
        ];

        store.preflight_runtime_mutation_backing(&[], 2).unwrap();
        assert_eq!(
            store.preflight_runtime_canonical_rewrite(1, &normalized_row(&[], &[])),
            Err(TemporalStorePreflightError::BackingInvariant)
        );
        assert_eq!(canonical_logical_bytes(&store), logical_before);
        assert_eq!(
            [
                store.runtime_owner_load(0),
                store.runtime_owner_load(1),
                store.runtime_owner_load(2),
            ],
            owner_loads_before
        );
        assert_eq!(store.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn rewrite_preflight_rejects_a_disconnected_target_posting() {
        let mut store = TemporalStore::new(config(10, 3, 4)).unwrap();
        for owner in 0..3 {
            assert_eq!(
                store.allocate_runtime_segment(owner, &[100], &[250]),
                Ok(owner)
            );
        }
        store.source_meta[0].tail = 10;
        store.source_meta[0].live_count = 2;
        store.next_pair[10] = NONE;
        store.prev_pair[20] = NONE;
        let logical_before = canonical_logical_bytes(&store);

        store.preflight_runtime_mutation_backing(&[], 2).unwrap();
        assert_eq!(
            store.preflight_runtime_canonical_rewrite(2, &normalized_row(&[], &[])),
            Err(TemporalStorePreflightError::BackingInvariant)
        );
        assert_eq!(canonical_logical_bytes(&store), logical_before);
        assert_eq!(store.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn recognition_rejects_a_posting_cycle_in_a_bounded_walk() {
        let mut store = reciprocal_three_node_cycle();
        let mut diagnostics = [empty_diagnostic(); 3];
        assert!(matches!(
            store.recognize_runtime_context(&[100], &mut diagnostics),
            Err(TEMPORAL_INVALID)
        ));
    }

    #[test]
    fn canonical_row_and_identity_validation_precedes_every_commit() {
        let mut store = TemporalStore::new(config(10, 1, 2)).unwrap();
        let valid = normalized_row(&[0, 2], &[250, 500]);
        store.preflight_runtime_canonical_append(0, &valid).unwrap();
        store.preflight_runtime_mutation_backing(&[0], 2).unwrap();
        assert_eq!(
            store.commit_prevalidated_runtime_canonical_append(0, &valid),
            0
        );
        let logical_before = canonical_logical_bytes(&store);

        let mut dirty_suffix = normalized_row(&[], &[]);
        dirty_suffix.normalized_sources[0] = 0;
        let duplicate = normalized_row(&[1, 1], &[250, 250]);
        let out_of_domain = normalized_row(&[10], &[250]);
        let zero_tick = normalized_row(&[1], &[0]);
        let mut too_wide = normalized_row(&[], &[]);
        too_wide.live_pair_count = 11;
        for invalid in [dirty_suffix, duplicate, out_of_domain, zero_tick, too_wide] {
            assert_eq!(
                store.preflight_runtime_canonical_append(0, &invalid),
                Err(TemporalStorePreflightError::InvalidCanonicalRow)
            );
            assert_eq!(
                store.preflight_runtime_canonical_rewrite(0, &invalid),
                Err(TemporalStorePreflightError::InvalidCanonicalRow)
            );
        }
        assert_eq!(
            store.preflight_runtime_canonical_append(1, &valid),
            Err(TemporalStorePreflightError::InvalidOwner)
        );
        assert_eq!(
            store.preflight_runtime_canonical_rewrite(1, &valid),
            Err(TemporalStorePreflightError::InvalidSegmentId)
        );
        assert_eq!(
            store.preflight_runtime_mutation_backing(&[], 0),
            Err(TemporalStorePreflightError::InvalidOwner)
        );
        assert_eq!(canonical_logical_bytes(&store), logical_before);
        assert_eq!(store.poisoned, TEMPORAL_OK);

        let replacement = normalized_row(&[3, 5], &[251, 499]);
        store.prev_pair[0] = 0;
        assert_eq!(
            store.preflight_runtime_canonical_rewrite(0, &replacement),
            Err(TemporalStorePreflightError::BackingInvariant)
        );
        assert_eq!(canonical_logical_bytes(&store), logical_before);
        assert_eq!(store.poisoned, TEMPORAL_OK);
        store.prev_pair[0] = NONE;
        store
            .preflight_runtime_canonical_rewrite(0, &replacement)
            .unwrap();
        store.preflight_runtime_mutation_backing(&[], 2).unwrap();
        store.commit_prevalidated_runtime_canonical_rewrite(0, &replacement);
        assert_eq!(store.runtime_segment_row(0).unwrap(), replacement);
        assert_intrusive_index_integrity(&store);
        assert_eq!(store.poisoned, TEMPORAL_OK);

        let mut cyclic = TemporalStore::new(config(10, 2, 2)).unwrap();
        assert_eq!(cyclic.allocate_runtime_segment(0, &[100], &[250]), Ok(0));
        assert_eq!(cyclic.allocate_runtime_segment(1, &[100], &[250]), Ok(1));
        let cyclic_logical = canonical_logical_bytes(&cyclic);
        cyclic.prev_pair[0] = 10;
        cyclic.next_pair[0] = 10;
        cyclic.prev_pair[10] = 0;
        cyclic.next_pair[10] = 0;
        assert_eq!(
            cyclic.preflight_runtime_canonical_rewrite(0, &normalized_row(&[], &[])),
            Err(TemporalStorePreflightError::BackingInvariant)
        );
        assert_eq!(canonical_logical_bytes(&cyclic), cyclic_logical);
        assert_eq!(cyclic.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn checkpoint_rebuild_accepts_normalized_rows_and_publishes_only_exact_count() {
        let mut store_config = config(12, 3, 4);
        store_config.source_gid_base = 10_000;
        let row_zero = normalized_row(&[0, 3, 11], &[250, 375, 500]);
        let empty = normalized_row(&[], &[]);
        let mut rebuild = TemporalStoreCheckpointRebuild::new(store_config, 2).unwrap();

        assert_eq!(rebuild.append_canonical_row(2, &row_zero), Ok(0));
        assert_eq!(rebuild.append_canonical_row(1, &empty), Ok(1));
        let mut store = rebuild.finish().unwrap();

        assert_eq!(store.runtime_segment_count(), 2);
        assert_eq!(store.runtime_segment_row(0), Ok(row_zero));
        assert_eq!(store.runtime_segment_row(1), Ok(empty));
        assert_eq!(store.runtime_owner_load(2), Some(1));
        assert_eq!(store.runtime_owner_load(1), Some(1));
        assert_eq!(posting_slots(&store, 0), vec![0]);
        assert_eq!(posting_slots(&store, 3), vec![1]);
        assert_eq!(posting_slots(&store, 11), vec![2]);

        let mut diagnostics = [empty_diagnostic(); 3];
        let summary = store
            .recognize_runtime_context(&[10_000, 10_003, 10_011], &mut diagnostics)
            .unwrap();
        assert_eq!(summary.active_segments, 1);
        assert_eq!(diagnostics[0].owner, 2);
        assert_intrusive_index_integrity(&store);
        assert_eq!(store.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn checkpoint_rebuild_empty_wrong_count_and_capacity_fail_closed() {
        let empty = TemporalStoreCheckpointRebuild::new(config(10, 1, 1), 0)
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(empty.runtime_segment_count(), 0);

        let mut incomplete = TemporalStoreCheckpointRebuild::new(config(10, 1, 2), 2).unwrap();
        incomplete
            .append_canonical_row(0, &normalized_row(&[], &[]))
            .unwrap();
        assert!(matches!(
            incomplete.finish(),
            Err(TemporalCheckpointRebuildError::SegmentCountMismatch {
                declared: 2,
                actual: 1,
            })
        ));

        assert!(matches!(
            TemporalStoreCheckpointRebuild::new(config(10, 1, 1), 2),
            Err(
                TemporalCheckpointRebuildError::DeclaredSegmentCountExceedsCapacity {
                    declared: 2,
                    capacity: 1,
                }
            )
        ));

        let mut exact = TemporalStoreCheckpointRebuild::new(config(10, 1, 1), 1).unwrap();
        let empty_row = normalized_row(&[], &[]);
        exact.append_canonical_row(0, &empty_row).unwrap();
        assert_eq!(
            exact.append_canonical_row(0, &empty_row),
            Err(TemporalCheckpointRebuildError::DeclaredSegmentCountExceeded { declared: 1 })
        );
        assert_eq!(exact.store.runtime_segment_count(), 1);
        assert_eq!(exact.store.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn checkpoint_rebuild_rejects_invalid_rows_atomically_and_retry_succeeds() {
        let valid = normalized_row(&[0, 2], &[250, 500]);
        let mut rebuild = TemporalStoreCheckpointRebuild::new(config(10, 2, 2), 2).unwrap();
        rebuild.append_canonical_row(0, &valid).unwrap();
        let logical_before = canonical_logical_bytes(&rebuild.store);
        let index_before = (
            rebuild.store.source_meta.clone(),
            rebuild.store.next_pair.clone(),
            rebuild.store.prev_pair.clone(),
        );

        let mut dirty_suffix = normalized_row(&[], &[]);
        dirty_suffix.normalized_sources[0] = 0;
        let duplicate = normalized_row(&[1, 1], &[250, 250]);
        let out_of_domain = normalized_row(&[10], &[250]);
        let zero_tick = normalized_row(&[1], &[0]);
        let mut too_wide = normalized_row(&[], &[]);
        too_wide.live_pair_count = 11;
        for invalid in [dirty_suffix, duplicate, out_of_domain, zero_tick, too_wide] {
            assert_eq!(
                rebuild.append_canonical_row(1, &invalid),
                Err(TemporalCheckpointRebuildError::Store(
                    TemporalStorePreflightError::InvalidCanonicalRow,
                ))
            );
            assert_eq!(canonical_logical_bytes(&rebuild.store), logical_before);
            assert_eq!(
                (
                    rebuild.store.source_meta.clone(),
                    rebuild.store.next_pair.clone(),
                    rebuild.store.prev_pair.clone(),
                ),
                index_before
            );
            assert_eq!(rebuild.store.poisoned, TEMPORAL_OK);
        }
        assert_eq!(
            rebuild.append_canonical_row(2, &valid),
            Err(TemporalCheckpointRebuildError::Store(
                TemporalStorePreflightError::InvalidOwner,
            ))
        );
        assert_eq!(canonical_logical_bytes(&rebuild.store), logical_before);

        let retry = normalized_row(&[1, 4], &[251, 499]);
        assert_eq!(rebuild.append_canonical_row(1, &retry), Ok(1));
        let store = rebuild.finish().unwrap();
        assert_eq!(store.runtime_segment_row(1), Ok(retry));
        assert_intrusive_index_integrity(&store);
    }

    #[test]
    fn checkpoint_rebuild_owner_capacity_failure_is_atomic() {
        let mut rebuild = TemporalStoreCheckpointRebuild::new(config(10, 2, 256), 256).unwrap();
        let empty = normalized_row(&[], &[]);
        for expected in 0..u8::MAX as u32 {
            assert_eq!(rebuild.append_canonical_row(0, &empty), Ok(expected));
        }
        let before = canonical_logical_bytes(&rebuild.store);
        assert_eq!(
            rebuild.append_canonical_row(0, &empty),
            Err(TemporalCheckpointRebuildError::Store(
                TemporalStorePreflightError::OwnerCapacity { owner: 0 },
            ))
        );
        assert_eq!(canonical_logical_bytes(&rebuild.store), before);
        assert_eq!(rebuild.store.runtime_segment_count(), u8::MAX as u32);
        assert_eq!(rebuild.store.poisoned, TEMPORAL_OK);

        assert_eq!(rebuild.append_canonical_row(1, &empty), Ok(255));
        assert_eq!(rebuild.finish().unwrap().runtime_owner_load(1), Some(1));
    }

    #[test]
    fn semantic_posting_visitor_is_source_major_and_history_independent() {
        let mut rebuild = TemporalStoreCheckpointRebuild::new(config(10, 3, 3), 3).unwrap();
        rebuild
            .append_canonical_row(0, &normalized_row(&[0, 2], &[250, 251]))
            .unwrap();
        rebuild
            .append_canonical_row(1, &normalized_row(&[0, 1], &[300, 301]))
            .unwrap();
        rebuild
            .append_canonical_row(2, &normalized_row(&[1, 2], &[400, 401]))
            .unwrap();
        let mut store = rebuild.finish().unwrap();

        // Rewrite segment zero so source 0's intrusive history becomes
        // segment 1 then segment 0. Semantic output must remain lexicographic.
        let rewritten = normalized_row(&[0, 2], &[499, 500]);
        store
            .preflight_runtime_canonical_rewrite(0, &rewritten)
            .unwrap();
        store
            .preflight_runtime_mutation_backing(&[], u8::MAX)
            .unwrap();
        store.commit_prevalidated_runtime_canonical_rewrite(0, &rewritten);
        assert_eq!(posting_slots(&store, 0), vec![10, 0]);

        assert_eq!(store.runtime_semantic_posting_scratch_required(), Ok(2));
        let mut scratch = [NONE; 2];
        let mut triples = Vec::new();
        assert_eq!(
            store.visit_runtime_semantic_postings(&mut scratch, |posting| {
                triples.push((posting.normalized_source, posting.segment_id, posting.slot));
            }),
            Ok(6)
        );
        assert_eq!(
            triples,
            vec![
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 1),
                (1, 2, 0),
                (2, 0, 1),
                (2, 2, 1),
            ]
        );
        assert_eq!(store.poisoned, TEMPORAL_OK);
    }

    #[test]
    fn semantic_posting_visitor_handles_empty_and_capacity_retry_atomically() {
        let empty = TemporalStoreCheckpointRebuild::new(config(10, 1, 1), 0)
            .unwrap()
            .finish()
            .unwrap();
        let mut empty_scratch = [];
        let mut empty_visits = 0;
        assert_eq!(
            empty.visit_runtime_semantic_postings(&mut empty_scratch, |_| {
                empty_visits += 1;
            }),
            Ok(0)
        );
        assert_eq!(empty_visits, 0);

        let mut rebuild = TemporalStoreCheckpointRebuild::new(config(10, 2, 2), 2).unwrap();
        let row = normalized_row(&[0], &[250]);
        rebuild.append_canonical_row(0, &row).unwrap();
        rebuild.append_canonical_row(1, &row).unwrap();
        let store = rebuild.finish().unwrap();
        let logical_before = canonical_logical_bytes(&store);
        let mut too_small = [0xDEAD_BEEF; 1];
        let mut visits = 0;
        assert_eq!(
            store.visit_runtime_semantic_postings(&mut too_small, |_| visits += 1),
            Err(TemporalStorePreflightError::OutputCapacity {
                required: 2,
                provided: 1,
            })
        );
        assert_eq!(visits, 0);
        assert_eq!(too_small, [0xDEAD_BEEF; 1]);
        assert_eq!(canonical_logical_bytes(&store), logical_before);
        assert_eq!(store.poisoned, TEMPORAL_OK);

        let mut enough = [NONE; 2];
        let mut retry = Vec::new();
        assert_eq!(
            store.visit_runtime_semantic_postings(&mut enough, |posting| retry.push(posting)),
            Ok(2)
        );
        assert_eq!(
            retry,
            vec![
                TemporalSemanticPosting {
                    normalized_source: 0,
                    segment_id: 0,
                    slot: 0,
                },
                TemporalSemanticPosting {
                    normalized_source: 0,
                    segment_id: 1,
                    slot: 0,
                },
            ]
        );
    }

    #[test]
    fn semantic_posting_visitor_rejects_stale_cyclic_and_disconnected_indexes_before_visit() {
        fn assert_rejected_without_visit(store: &TemporalStore) {
            let logical_before = canonical_logical_bytes(store);
            let mut scratch = [NONE; 4];
            let mut visits = 0;
            assert_eq!(
                store.visit_runtime_semantic_postings(&mut scratch, |_| visits += 1),
                Err(TemporalStorePreflightError::BackingInvariant)
            );
            assert_eq!(visits, 0);
            assert_eq!(canonical_logical_bytes(store), logical_before);
            assert_eq!(store.poisoned, TEMPORAL_OK);
        }

        let mut stale = TemporalStore::new(config(10, 2, 2)).unwrap();
        stale.allocate_runtime_segment(0, &[100], &[250]).unwrap();
        stale.source_meta[0].tail = stale.sources.len() as u32;
        assert_rejected_without_visit(&stale);

        let cyclic = reciprocal_three_node_cycle();
        assert_rejected_without_visit(&cyclic);

        let mut disconnected = TemporalStore::new(config(10, 3, 3)).unwrap();
        for owner in 0..3 {
            disconnected
                .allocate_runtime_segment(owner, &[100], &[250])
                .unwrap();
        }
        disconnected.source_meta[0].tail = 10;
        disconnected.source_meta[0].live_count = 2;
        disconnected.next_pair[10] = NONE;
        disconnected.prev_pair[20] = NONE;
        assert_rejected_without_visit(&disconnected);

        let mut duplicated = TemporalStore::new(config(10, 1, 1)).unwrap();
        duplicated
            .allocate_runtime_segment(0, &[100], &[250])
            .unwrap();
        duplicated.source_meta[1] = SourceMeta {
            head: 0,
            tail: 0,
            live_count: 1,
        };
        assert_rejected_without_visit(&duplicated);

        let mut stale_empty_slot = TemporalStore::new(config(10, 1, 1)).unwrap();
        stale_empty_slot
            .allocate_runtime_segment(0, &[100], &[250])
            .unwrap();
        stale_empty_slot.next_pair[1] = 0;
        assert_rejected_without_visit(&stale_empty_slot);
    }
}
