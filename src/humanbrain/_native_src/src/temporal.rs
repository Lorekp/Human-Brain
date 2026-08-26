//! Bounded native storage and recognition for fixed-width distal segments.
//!
//! Every segment owns exactly ten logical source/relay pairs.  Only the
//! normalized source and one shared permanence tick are resident; a posting
//! visit contributes two physical contacts to the recognition counts.

use std::{mem::size_of, ptr, slice};

const PAIRS_PER_SEGMENT: usize = 10;
const PHYSICAL_CONTACTS_PER_PAIR: u8 = 2;
const POSTINGS_PER_BLOCK: usize = 64;
const NONE: u32 = u32::MAX;

const TEMPORAL_OK: u32 = 0;
const TEMPORAL_INVALID: u32 = 1;
const TEMPORAL_SEGMENT_CAPACITY: u32 = 2;
const TEMPORAL_POSTING_CAPACITY: u32 = 3;
const TEMPORAL_OWNER_CAPACITY: u32 = 4;
const TEMPORAL_POISONED: u32 = 5;
const TEMPORAL_ALLOCATION_FAILED: u32 = 6;
const TEMPORAL_OUTPUT_CAPACITY: u32 = 7;

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

#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
struct PostingBlock {
    postings: [u32; POSTINGS_PER_BLOCK],
    next: u32,
    len: u16,
    _padding_zero: u16,
}

impl PostingBlock {
    const EMPTY: Self = Self {
        postings: [0; POSTINGS_PER_BLOCK],
        next: NONE,
        len: 0,
        _padding_zero: 0,
    };
}

const _: [(); 12] = [(); size_of::<SourceMeta>()];
const _: [(); 264] = [(); size_of::<PostingBlock>()];

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

fn maximum_posting_blocks(source_count: usize, posting_capacity: usize) -> Option<usize> {
    let nonempty_sources = source_count.min(posting_capacity);
    nonempty_sources.checked_add((posting_capacity - nonempty_sources) / POSTINGS_PER_BLOCK)
}

pub struct TemporalStore {
    config: TemporalStoreConfig,
    posting_block_capacity: usize,
    owners: Vec<u32>,
    sources: Vec<u32>,
    permanences: Vec<u16>,
    owner_segment_counts: Vec<u8>,
    source_meta: Vec<SourceMeta>,
    posting_blocks: Vec<PostingBlock>,
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
        let posting_block_capacity = maximum_posting_blocks(config.source_count, posting_capacity)
            .ok_or(TEMPORAL_INVALID)?;
        if posting_block_capacity > u32::MAX as usize {
            return Err(TEMPORAL_INVALID);
        }
        let owner_result_capacity = config.owner_count.min(config.segment_capacity);
        Ok(Self {
            config,
            posting_block_capacity,
            owners: reserved_vec(config.segment_capacity)?,
            sources: reserved_vec(posting_capacity)?,
            permanences: reserved_vec(posting_capacity)?,
            owner_segment_counts: filled_vec(config.owner_count, 0)?,
            source_meta: filled_vec(config.source_count, SourceMeta::EMPTY)?,
            posting_blocks: reserved_vec(posting_block_capacity)?,
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

    fn required_new_blocks(&self, sources: &[u32]) -> usize {
        sources
            .iter()
            .filter(|source| {
                let meta = self.source_meta[**source as usize];
                meta.tail == NONE
                    || self.posting_blocks[meta.tail as usize].len as usize == POSTINGS_PER_BLOCK
            })
            .count()
    }

    fn append_posting(&mut self, source: u32, posting: u32) {
        let source_index = source as usize;
        let tail = self.source_meta[source_index].tail;
        let block_index = if tail == NONE
            || self.posting_blocks[tail as usize].len as usize == POSTINGS_PER_BLOCK
        {
            let block_index = self.posting_blocks.len() as u32;
            self.posting_blocks.push(PostingBlock::EMPTY);
            if tail == NONE {
                self.source_meta[source_index].head = block_index;
            } else {
                self.posting_blocks[tail as usize].next = block_index;
            }
            self.source_meta[source_index].tail = block_index;
            block_index
        } else {
            tail
        };
        let block = &mut self.posting_blocks[block_index as usize];
        let slot = block.len as usize;
        block.postings[slot] = posting;
        block.len += 1;
        self.source_meta[source_index].live_count += 1;
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
            || sources.len() != PAIRS_PER_SEGMENT
            || permanence_ticks.len() != PAIRS_PER_SEGMENT
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
        let required_blocks = self.required_new_blocks(&normalized_sources);
        if self.posting_blocks.len().saturating_add(required_blocks) > self.posting_block_capacity {
            return Err(self.poison(TEMPORAL_POSTING_CAPACITY));
        }

        let segment_id = self.owners.len() as u32;
        self.owners.push(owner);
        self.owner_segment_counts[owner as usize] += 1;
        for (slot, source) in normalized_sources.iter().enumerate() {
            self.sources.push(*source);
            self.permanences.push(permanence_ticks[slot]);
            let posting = segment_id * PAIRS_PER_SEGMENT as u32 + slot as u32;
            self.append_posting(*source, posting);
        }
        Ok(segment_id)
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
            let source_index = self.normalize_source(*source).unwrap() as usize;
            if self.source_stamps[source_index] == generation {
                return Err(TEMPORAL_INVALID);
            }
            self.source_stamps[source_index] = generation;
        }

        self.touched_segments.clear();
        self.touched_owners.clear();
        let mut postings_visited = 0u64;
        for source in context_sources {
            let source_index = self.normalize_source(*source).unwrap() as usize;
            let mut block_index = self.source_meta[source_index].head;
            while block_index != NONE {
                let block = &self.posting_blocks[block_index as usize];
                for index in 0..block.len as usize {
                    let posting = block.postings[index];
                    let segment_id = posting / PAIRS_PER_SEGMENT as u32;
                    let segment_index = segment_id as usize;
                    if self.segment_stamps[segment_index] != generation {
                        self.segment_stamps[segment_index] = generation;
                        self.potential_contacts[segment_index] = 0;
                        self.connected_contacts[segment_index] = 0;
                        self.touched_segments.push(segment_id);
                    }
                    self.potential_contacts[segment_index] += PHYSICAL_CONTACTS_PER_PAIR;
                    if self.permanences[posting as usize] >= self.config.connected_permanence_ticks
                    {
                        self.connected_contacts[segment_index] += PHYSICAL_CONTACTS_PER_PAIR;
                    }
                    postings_visited += 1;
                }
                block_index = block.next;
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
    use std::{mem::size_of, ptr};

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

    fn posting_lengths(store: &TemporalStore, normalized_source: usize) -> Vec<usize> {
        let mut lengths = Vec::new();
        let mut block_index = store.source_meta[normalized_source].head;
        while block_index != NONE {
            let block = &store.posting_blocks[block_index as usize];
            lengths.push(block.len as usize);
            block_index = block.next;
        }
        lengths
    }

    #[test]
    fn ffi_struct_sizes_are_stable() {
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(size_of::<TemporalStoreConfig>(), 40);
            assert_eq!(size_of::<TemporalOwnerDiagnostic>(), 16);
            assert_eq!(size_of::<TemporalPrepareSummary>(), 48);
        }
        assert_eq!(size_of::<SourceMeta>(), 12);
        assert_eq!(size_of::<PostingBlock>(), 264);
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
        assert!(store.posting_blocks.is_empty());

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
    fn posting_chain_crosses_63_64_65_boundaries_exactly() {
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
                    posting_lengths(&store, 0),
                    if count <= 64 {
                        vec![count]
                    } else {
                        vec![64, count - 64]
                    }
                );
            }
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
