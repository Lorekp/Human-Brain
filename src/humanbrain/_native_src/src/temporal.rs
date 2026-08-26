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
    poisoned: u32,
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
            || (config.source_gid_base as u64 + config.source_count as u64) > (u32::MAX as u64 + 1)
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
            poisoned: TEMPORAL_OK,
        })
    }

    fn poison(&mut self, status: u32) -> u32 {
        self.poisoned = status;
        status
    }

    fn normalize_source(&self, source_gid: u32) -> Option<u32> {
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
            let Some(source_index) = self.normalize_source(*source).map(|value| value as usize)
            else {
                return Err(TEMPORAL_INVALID);
            };
            let mut pair_slot = self.source_meta[source_index].head;
            while pair_slot != NONE {
                let segment_id = pair_slot / PAIRS_PER_SEGMENT as u32;
                let segment_index = segment_id as usize;
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
                postings_visited += 1;
                pair_slot = self.next_pair[pair_slot as usize];
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
            assert_eq!(size_of::<TemporalStore>(), 432);
        }
        assert_eq!(size_of::<SourceMeta>(), 12);
    }

    #[test]
    fn c1_intrusive_store_logical_payload_layout_is_exact() {
        // Element payloads requested from Vec; allocator metadata and the
        // separately pinned TemporalStore header are intentionally excluded.
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
        const PYTHON_DIAGNOSTIC_OUTPUT: usize = OWNERS * size_of::<TemporalOwnerDiagnostic>();

        assert_eq!(PAIRS, 232_864_060);
        assert_eq!(LEARNED_SOA, 1_490_461_056);
        assert_eq!(INTRUSIVE_INDEX, 1_864_485_344);
        assert_eq!(SOURCE_SCRATCH, 524_288);
        assert_eq!(SEGMENT_SCRATCH, 232_864_060);
        assert_eq!(OWNER_SCRATCH, 2_097_152);
        assert_eq!(NATIVE_RESERVED_PAYLOAD, 3_590_431_900);
        assert_eq!(PYTHON_DIAGNOSTIC_OUTPUT, 2_097_152);
        assert_eq!(
            NATIVE_RESERVED_PAYLOAD + PYTHON_DIAGNOSTIC_OUTPUT,
            3_592_529_052
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
}
