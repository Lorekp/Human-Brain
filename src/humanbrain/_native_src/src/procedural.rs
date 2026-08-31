//! Source-driven procedural execution over a bounded sparse working set.
//!
//! This module deliberately has no operation proportional to the configured
//! address range. The sparse runtime stores only bounded resident state,
//! pending events, and spikes produced by the most recent advance. A separate
//! dense runtime exists only as a small-domain exactness oracle.

use super::{
    random123::{philox2x64_10, PHILOX_W64_0},
    GlifConfig,
};
use std::{
    alloc::{alloc, Layout},
    cmp::Ordering,
    collections::{BTreeMap, HashSet},
    mem::size_of,
    ptr, slice,
};

unsafe fn allocate_owned<T>(value: T) -> *mut T {
    let raw = alloc(Layout::new::<T>()) as *mut T;
    if raw.is_null() {
        drop(value);
        ptr::null_mut()
    } else {
        raw.write(value);
        raw
    }
}

const UNIFORM_NO_SELF_STREAM_ID: u64 = 2;
const GLOBAL_PURPOSE_ID: u64 = 1;
const GLOBAL_FALLBACK_PURPOSE_ID: u64 = 255;
const ROLE_COUNTER_WORD: u64 = 0x524F4C45;
const MAX_CANDIDATE_ATTEMPTS: u64 = 128;

type CommonConfiguration = (GlifConfig, ProceduralConfig, [f64; 2], [f64; 2], [f64; 2]);

pub const PROCEDURAL_OK: u32 = 0;
pub const PROCEDURAL_INVALID: u32 = 1;
pub const PROCEDURAL_RESIDENT_CAPACITY: u32 = 2;
pub const PROCEDURAL_EVENT_CAPACITY: u32 = 3;
pub const PROCEDURAL_DUPLICATE_EVENT: u32 = 4;
pub const PROCEDURAL_POISONED: u32 = 5;
pub const PROCEDURAL_STEP_OVERFLOW: u32 = 6;
pub const PROCEDURAL_SNAPSHOT_CAPACITY: u32 = 7;
pub const PROCEDURAL_ALLOCATION: u32 = 8;
pub const PROCEDURAL_SPIKE_CAPACITY: u32 = 9;
pub const PROCEDURAL_DISTINCT_CAPACITY: u32 = 10;

const SEEN_RESIDENT: u8 = 1;
const SEEN_UPDATED: u8 = 2;
const SEEN_SPIKING: u8 = 4;

/// Stepping semantics selected at creation. `FULL_TOUCH` is the shipped
/// default: every resident integrates every step until it reaches canonical
/// rest. `EVENT_PROPORTIONAL` is the additive spike mode described in
/// [`ProceduralConfig::stepping_mode`] below.
pub const PROCEDURAL_STEPPING_FULL_TOUCH: u32 = 0;
pub const PROCEDURAL_STEPPING_EVENT_PROPORTIONAL: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProceduralConfig {
    pub configured_cells: u64,
    pub fanout: u64,
    pub seed: u64,
    pub resident_capacity: usize,
    pub event_capacity: usize,
    pub spike_capacity: usize,
    pub interval_distinct_capacity: usize,
    pub recurrent_weight_pa: f64,
    pub recurrent_delay_steps: u64,
    pub recurrent_decay: f64,
    pub eligibility_decay: f64,
    /// Trailing additive field. 0 selects the shipped full-touch semantics
    /// bit-for-bit; 1 selects event-proportional stepping, where a resident
    /// whose only nonzero field is the eligibility tail ("dormant") leaves
    /// the stepped set and its eligibility decays lazily, by replaying the
    /// identical per-step multiplies, on the next event that touches it.
    /// Spike records, delivered/generated events, queue contents, and the
    /// dynamics of every touched cell are bit-identical across modes; the
    /// documented divergences are internal accounting (`updated_distinct`,
    /// `cell_step_evaluations`, payload bytes) and the split of residency
    /// between the stepped array and the dormant store.
    pub stepping_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProceduralEventRecord {
    pub due_step: u64,
    pub target_gid: u64,
    pub source_gid: u64,
    pub edge_slot: u64,
    pub current_pa: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProceduralResidentRecord {
    pub gid: u64,
    pub voltage_relative_mv: f64,
    pub after_spike_current_0_pa: f64,
    pub after_spike_current_1_pa: f64,
    pub threshold_spike_mv: f64,
    pub threshold_voltage_mv: f64,
    pub refractory_steps: i32,
    /// Explicit alignment padding at offsets 52..56. FFI consumers may read
    /// and compare the complete 72-byte record, so this must be a declared,
    /// canonical zero instead of unspecified implicit padding.
    pub padding_zero: u32,
    pub synaptic_current_pa: f64,
    pub eligibility: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProceduralSpikeRecord {
    pub step: u64,
    pub gid: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ProceduralCounters {
    pub current_step: u64,
    pub resident_peak: usize,
    pub resident_distinct: usize,
    pub updated_distinct: usize,
    pub spiking_distinct: usize,
    pub cell_step_evaluations: u64,
    pub spike_events: u64,
    pub generated_edges: u64,
    pub delivered_events: u64,
    pub configured_ids_scanned: u64,
    pub queue_peak: usize,
    pub native_payload_bytes: usize,
    pub retained_nonzero_cells: usize,
    pub overflow_code: u32,
    /// Explicit tail padding (offsets 108..112 of the 112-byte C layout).
    /// The FFI snapshot hands this struct to callers byte-for-byte and tests
    /// compare/hash those raw bytes between runs, so the padding must be a
    /// declared always-zero field instead of unspecified implicit padding
    /// carrying stack garbage from whichever construction path built the
    /// value -- that garbage made raw-bytes comparisons depend on allocator
    /// history (i.e. on which code/tests ran earlier in the process), which
    /// surfaced as an order-dependent failure of the generated-recurrent-key
    /// collision fail-stop test. Keep this zero; `size_of` stays 112 either
    /// way.
    pub padding_zero: u32,
}

#[derive(Clone, Copy, Debug)]
struct CellState {
    gid: u64,
    voltage: f64,
    after: [f64; 2],
    threshold_spike: f64,
    threshold_voltage: f64,
    refractory: i32,
    synaptic_current: f64,
    eligibility: f64,
}

impl CellState {
    fn canonical_rest(gid: u64) -> Self {
        Self {
            gid,
            voltage: 0.0,
            after: [0.0, 0.0],
            threshold_spike: 0.0,
            threshold_voltage: 0.0,
            refractory: 0,
            synaptic_current: 0.0,
            eligibility: 0.0,
        }
    }

    fn is_canonical_rest(&self) -> bool {
        self.voltage.to_bits() == 0
            && self.after[0].to_bits() == 0
            && self.after[1].to_bits() == 0
            && self.threshold_spike.to_bits() == 0
            && self.threshold_voltage.to_bits() == 0
            && self.refractory == 0
            && self.synaptic_current.to_bits() == 0
            && self.eligibility.to_bits() == 0
    }

    /// True when every dynamics-bearing field is exact zero and no refractory
    /// time remains; only the eligibility tail may be nonzero. Under continued
    /// unforced stepping such a cell changes nothing but its own eligibility:
    /// zero inputs reproduce zero voltage/thresholds/after-currents exactly,
    /// and with `threshold_inf_relative_mv >= 0` (validated in the
    /// event-proportional mode) it can never cross. Its full-touch cost is
    /// therefore exactly one eligibility multiply per step.
    fn is_fast_rest(&self) -> bool {
        self.voltage.to_bits() == 0
            && self.after[0].to_bits() == 0
            && self.after[1].to_bits() == 0
            && self.threshold_spike.to_bits() == 0
            && self.threshold_voltage.to_bits() == 0
            && self.refractory == 0
            && self.synaptic_current.to_bits() == 0
    }

    fn record(&self) -> ProceduralResidentRecord {
        ProceduralResidentRecord {
            gid: self.gid,
            voltage_relative_mv: self.voltage,
            after_spike_current_0_pa: self.after[0],
            after_spike_current_1_pa: self.after[1],
            threshold_spike_mv: self.threshold_spike,
            threshold_voltage_mv: self.threshold_voltage,
            refractory_steps: self.refractory,
            padding_zero: 0,
            synaptic_current_pa: self.synaptic_current,
            eligibility: self.eligibility,
        }
    }
}

#[inline]
fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

#[inline]
fn event_order(left: &ProceduralEventRecord, right: &ProceduralEventRecord) -> Ordering {
    (
        left.due_step,
        left.target_gid,
        left.source_gid,
        left.edge_slot,
    )
        .cmp(&(
            right.due_step,
            right.target_gid,
            right.source_gid,
            right.edge_slot,
        ))
}

#[inline]
fn same_event_key(left: &ProceduralEventRecord, right: &ProceduralEventRecord) -> bool {
    event_order(left, right) == Ordering::Equal
}

#[inline]
fn dense_event_order(left: &ProceduralEventRecord, right: &ProceduralEventRecord) -> Ordering {
    left.due_step
        .cmp(&right.due_step)
        .then_with(|| left.target_gid.cmp(&right.target_gid))
        .then_with(|| left.source_gid.cmp(&right.source_gid))
        .then_with(|| left.edge_slot.cmp(&right.edge_slot))
}

#[inline]
fn same_dense_event_key(left: &ProceduralEventRecord, right: &ProceduralEventRecord) -> bool {
    left.due_step == right.due_step
        && left.target_gid == right.target_gid
        && left.source_gid == right.source_gid
        && left.edge_slot == right.edge_slot
}

#[inline]
fn bounded_index(word: u64, size: u64) -> u64 {
    (((word as u128) * (size as u128)) >> 64) as u64
}

#[inline]
fn fallback_candidate(start: u64, offset: u64, configured_cells: u64) -> u64 {
    ((start as u128 + offset as u128) % configured_cells as u128) as u64
}

#[inline]
fn counter_words(
    config: &ProceduralConfig,
    source: u64,
    slot: u64,
    purpose: u64,
    attempt: u64,
) -> (u64, u64) {
    let context = (slot << 16) | (purpose << 8) | attempt;
    let key = config
        .seed
        .wrapping_add(UNIFORM_NO_SELF_STREAM_ID.wrapping_mul(PHILOX_W64_0));
    philox2x64_10(source, context, key)
}

fn fill_targets_with<C>(
    config: &ProceduralConfig,
    source: u64,
    targets: &mut Vec<u64>,
    used: &mut HashSet<u64>,
    mut candidate_for: C,
) -> Result<u64, u32>
where
    C: FnMut(&ProceduralConfig, u64, u64, u64) -> u64,
{
    let fanout = config.fanout as usize;
    if targets.capacity() < fanout || used.capacity() < fanout.saturating_add(1) {
        return Err(PROCEDURAL_ALLOCATION);
    }
    targets.clear();
    used.clear();
    used.insert(source);
    let fallback_word = counter_words(config, source, 0, GLOBAL_FALLBACK_PURPOSE_ID, 0).0;
    let fallback_start = bounded_index(fallback_word, config.configured_cells);
    let mut fallback_offset = 0u64;
    let mut candidate_examinations = 0u64;
    for slot in 0..config.fanout {
        let mut selected = None;
        for attempt in 0..MAX_CANDIDATE_ATTEMPTS {
            let candidate = candidate_for(config, source, slot, attempt);
            candidate_examinations += 1;
            if !used.contains(&candidate) {
                selected = Some(candidate);
                break;
            }
        }
        if selected.is_none() {
            while fallback_offset <= config.fanout {
                let candidate =
                    fallback_candidate(fallback_start, fallback_offset, config.configured_cells);
                fallback_offset += 1;
                candidate_examinations += 1;
                if !used.contains(&candidate) {
                    selected = Some(candidate);
                    break;
                }
            }
        }
        let Some(target) = selected else {
            return Err(PROCEDURAL_INVALID);
        };
        if used.len() >= used.capacity() || targets.len() >= fanout {
            return Err(PROCEDURAL_ALLOCATION);
        }
        used.insert(target);
        targets.push(target);
    }
    Ok(candidate_examinations)
}

fn reserved_vec<T>(capacity: usize) -> Result<Vec<T>, u32> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| PROCEDURAL_ALLOCATION)?;
    Ok(values)
}

fn push_bounded<T>(
    values: &mut Vec<T>,
    value: T,
    logical_capacity: usize,
    error: u32,
) -> Result<(), u32> {
    if values.len() >= logical_capacity || values.len() >= values.capacity() {
        return Err(error);
    }
    values.push(value);
    Ok(())
}

/// Sorted-union ledger for the distinct GIDs flagged during one advance
/// window, replacing the previous hash map. GIDs stay ascending-unique and
/// flags accumulate per GID, so the committed counters are identical while
/// the per-cell cost drops from a randomized hash lookup to sequential
/// sorted-run merges.
struct IntervalLedger {
    gids: Vec<u64>, // ascending, unique
    flags: Vec<u8>, // parallel accumulated SEEN_* bits
}

impl IntervalLedger {
    fn try_new(capacity: usize) -> Result<Self, u32> {
        let mut gids = Vec::new();
        gids.try_reserve_exact(capacity)
            .map_err(|_| PROCEDURAL_ALLOCATION)?;
        let mut flags = Vec::new();
        flags
            .try_reserve_exact(capacity)
            .map_err(|_| PROCEDURAL_ALLOCATION)?;
        Ok(Self { gids, flags })
    }

    fn clear(&mut self) {
        self.gids.clear();
        self.flags.clear();
    }

    /// Exact single-GID mark preserving the previous fail-stop timing: the
    /// first new GID beyond the logical capacity raises immediately.
    fn mark_single(&mut self, gid: u64, flag: u8, capacity: usize) -> Result<(), u32> {
        match self.gids.binary_search(&gid) {
            Ok(index) => {
                self.flags[index] |= flag;
                Ok(())
            }
            Err(insertion) => {
                if self.gids.len() >= capacity || self.gids.len() >= self.gids.capacity() {
                    return Err(PROCEDURAL_DISTINCT_CAPACITY);
                }
                self.gids.insert(insertion, gid);
                self.flags.insert(insertion, flag);
                Ok(())
            }
        }
    }

    /// Stage one canonically ordered GID for a later sorted-run merge. Existing
    /// ledger members are marked immediately; absent members append to `run`
    /// without shifting either ledger vector. `run` is step-local, ascending,
    /// unique, and backed by already-accounted runtime scratch.
    fn stage_single_in_run(
        &mut self,
        gid: u64,
        flag: u8,
        run: &mut Vec<u64>,
        capacity: usize,
    ) -> Result<(), u32> {
        if let Ok(index) = self.gids.binary_search(&gid) {
            self.flags[index] |= flag;
            return Ok(());
        }
        if let Some(last) = run.last() {
            match last.cmp(&gid) {
                Ordering::Less => {}
                Ordering::Equal => return Ok(()),
                // Due events are delivered in canonical target order. Treat a
                // broken internal ordering invariant as fail-stop rather than
                // silently constructing a non-mergeable run.
                Ordering::Greater => return Err(PROCEDURAL_INVALID),
            }
        }
        let pending_distinct = self
            .gids
            .len()
            .checked_add(run.len())
            .ok_or(PROCEDURAL_DISTINCT_CAPACITY)?;
        if pending_distinct >= capacity || pending_distinct >= self.gids.capacity() {
            return Err(PROCEDURAL_DISTINCT_CAPACITY);
        }
        if run.len() >= run.capacity() {
            return Err(PROCEDURAL_ALLOCATION);
        }
        run.push(gid);
        Ok(())
    }

    /// True when an ascending unique run certainly fits within `capacity`.
    /// The cheap size upper bound answers immediately in the common case;
    /// only a near-full ledger pays the exact new-GID walk.
    fn can_fit_run(&self, run: &[u64], capacity: usize) -> bool {
        let reserved_room = self.gids.capacity().saturating_sub(self.gids.len());
        let room = capacity
            .min(self.gids.capacity())
            .saturating_sub(self.gids.len());
        let room = room.min(reserved_room);
        if run.len() <= room {
            return true;
        }
        let mut new_count = 0usize;
        let mut cursor = 0usize;
        for gid in run {
            cursor += self.gids[cursor..].partition_point(|existing| *existing < *gid);
            if cursor == self.gids.len() || self.gids[cursor] != *gid {
                new_count += 1;
                if new_count > room {
                    return false;
                }
            }
        }
        true
    }

    /// Union-merge an ascending unique run into this ledger, OR-ing `flag`
    /// into every run member. Caller must have validated fit with
    /// [`can_fit_run`]. The merge walks backward so unread ledger entries are
    /// never overwritten and no second full-capacity ledger is required.
    fn merge_run_in_place(&mut self, run: &[u64], flag: u8) {
        if run.is_empty() {
            return;
        }

        debug_assert_eq!(self.gids.len(), self.flags.len());
        debug_assert!(self.gids.windows(2).all(|pair| pair[0] < pair[1]));
        debug_assert!(run.windows(2).all(|pair| pair[0] < pair[1]));

        let old_len = self.gids.len();
        let mut i = 0usize;
        let mut j = 0usize;
        let mut overlap = 0usize;
        while i < old_len && j < run.len() {
            match self.gids[i].cmp(&run[j]) {
                Ordering::Less => i += 1,
                Ordering::Greater => j += 1,
                Ordering::Equal => {
                    overlap += 1;
                    i += 1;
                    j += 1;
                }
            }
        }

        let merged_len = old_len + run.len() - overlap;
        debug_assert!(merged_len <= self.gids.capacity());
        debug_assert!(merged_len <= self.flags.capacity());
        self.gids.resize(merged_len, 0);
        self.flags.resize(merged_len, 0);

        i = old_len;
        j = run.len();
        let mut write = merged_len;
        while i > 0 && j > 0 {
            let old_gid = self.gids[i - 1];
            let old_flags = self.flags[i - 1];
            match old_gid.cmp(&run[j - 1]) {
                Ordering::Greater => {
                    i -= 1;
                    write -= 1;
                    self.gids[write] = old_gid;
                    self.flags[write] = old_flags;
                }
                Ordering::Less => {
                    j -= 1;
                    write -= 1;
                    self.gids[write] = run[j];
                    self.flags[write] = flag;
                }
                Ordering::Equal => {
                    i -= 1;
                    j -= 1;
                    write -= 1;
                    self.gids[write] = old_gid;
                    self.flags[write] = old_flags | flag;
                }
            }
        }
        while i > 0 {
            let old_gid = self.gids[i - 1];
            let old_flags = self.flags[i - 1];
            i -= 1;
            write -= 1;
            self.gids[write] = old_gid;
            self.flags[write] = old_flags;
        }
        while j > 0 {
            j -= 1;
            write -= 1;
            self.gids[write] = run[j];
            self.flags[write] = flag;
        }
        debug_assert_eq!(write, 0);
    }

    fn count_flags(&self, mask: u8) -> usize {
        self.flags
            .iter()
            .filter(|entry_flags| **entry_flags & mask != 0)
            .count()
    }
}

struct RuntimeScratch {
    submitted_events: Vec<ProceduralEventRecord>,
    event_work: Vec<ProceduralEventRecord>,
    step_spikes: Vec<u64>,
    row_targets: Vec<u64>,
    row_used: HashSet<u64>,
    interval_ledger: IntervalLedger,
    /// Reused sequentially as the fresh-target run, stepped-population run,
    /// and finally the current step's ascending dormant-retirement GIDs.
    gid_run: Vec<u64>,
    /// Values parallel to the dormant-retirement phase of `gid_run`.
    dormant_values: Vec<f64>,
}

impl RuntimeScratch {
    fn try_new(config: &ProceduralConfig) -> Result<Self, u32> {
        let fanout = usize::try_from(config.fanout).map_err(|_| PROCEDURAL_ALLOCATION)?;
        let row_members = fanout.checked_add(1).ok_or(PROCEDURAL_ALLOCATION)?;
        let mut row_used = HashSet::new();
        row_used
            .try_reserve(row_members)
            .map_err(|_| PROCEDURAL_ALLOCATION)?;
        Ok(Self {
            submitted_events: reserved_vec(config.event_capacity)?,
            event_work: reserved_vec(config.event_capacity)?,
            step_spikes: reserved_vec(config.resident_capacity)?,
            row_targets: reserved_vec(fanout)?,
            row_used,
            interval_ledger: IntervalLedger::try_new(config.interval_distinct_capacity)?,
            gid_run: reserved_vec(config.resident_capacity)?,
            dormant_values: reserved_vec(config.resident_capacity)?,
        })
    }

    fn payload_bytes(&self) -> usize {
        self.submitted_events.capacity() * size_of::<ProceduralEventRecord>()
            + self.event_work.capacity() * size_of::<ProceduralEventRecord>()
            + self.step_spikes.capacity() * size_of::<u64>()
            + self.row_targets.capacity() * size_of::<u64>()
            + self.row_used.capacity() * size_of::<u64>()
            + self.interval_ledger.gids.capacity() * size_of::<u64>()
            + self.interval_ledger.flags.capacity() * size_of::<u8>()
            + self.gid_run.capacity() * size_of::<u64>()
            + self.dormant_values.capacity() * size_of::<f64>()
    }
}

/// One dormant-store entry: a fast-rest cell kept out of the stepped set
/// purely as a lazily-decayed eligibility tail.
#[derive(Clone, Copy, Debug)]
struct DormantEntry {
    /// Stored eligibility exactly as of the last eager per-step decay.
    eligibility: f64,
    /// Step whose fused pass performed that last decay. The passes after it
    /// are replayed, one multiply each in identical order, when the cell is
    /// next touched -- reproducing full-touch bits exactly.
    elig_step: u64,
}

struct DenseScratch {
    submitted_events: Vec<ProceduralEventRecord>,
    event_work: Vec<ProceduralEventRecord>,
    step_spikes: Vec<u64>,
}

impl DenseScratch {
    fn try_new(config: &ProceduralConfig) -> Result<Self, u32> {
        let cells = usize::try_from(config.configured_cells).map_err(|_| PROCEDURAL_ALLOCATION)?;
        Ok(Self {
            submitted_events: reserved_vec(config.event_capacity)?,
            event_work: reserved_vec(config.event_capacity)?,
            step_spikes: reserved_vec(cells)?,
        })
    }
}

#[inline]
fn source_weight(config: &ProceduralConfig, source: u64) -> f64 {
    let role_word = philox2x64_10(source, ROLE_COUNTER_WORD, config.seed).0;
    let inhibitory_threshold = ((1u128 << 64) / 5) as u64;
    let inhibitory = role_word < inhibitory_threshold;
    if inhibitory {
        -config.recurrent_weight_pa
    } else {
        config.recurrent_weight_pa
    }
}

#[inline]
fn sparse_cell_step(
    glif: &GlifConfig,
    after_decay: &[f64; 2],
    after_cut_decay: &[f64; 2],
    after_amplitudes: &[f64; 2],
    recurrent_decay: f64,
    eligibility_decay: f64,
    state: &mut CellState,
) -> bool {
    let mut crossed = false;
    if state.refractory != 0 {
        state.refractory -= 1;
    } else {
        let after_sum = state.after[0] + state.after[1];
        let total_current = state.synaptic_current + after_sum;
        let previous_voltage = state.voltage;
        let leak = glif.leak_ns * previous_voltage;
        let voltage_delta = glif.voltage_step_scale * (total_current - leak);
        let next_voltage = previous_voltage + voltage_delta;
        let beta = total_current * glif.threshold_beta_scale;
        let voltage_term = glif.threshold_voltage_factor * previous_voltage;
        let beta_term = glif.threshold_beta_factor * beta;
        let decayed_voltage_threshold = glif.threshold_exp_decay * state.threshold_voltage;
        let next_voltage_threshold = decayed_voltage_threshold + voltage_term + beta_term;
        let next_spike_threshold = state.threshold_spike * glif.spike_threshold_decay;
        let next_after_0 = state.after[0] * after_decay[0];
        let next_after_1 = state.after[1] * after_decay[1];

        state.voltage = canonical_zero(next_voltage);
        state.threshold_voltage = canonical_zero(next_voltage_threshold);
        state.threshold_spike = canonical_zero(next_spike_threshold);
        state.after[0] = canonical_zero(next_after_0);
        state.after[1] = canonical_zero(next_after_1);
        let total_threshold =
            glif.threshold_inf_relative_mv + next_voltage_threshold + next_spike_threshold;
        crossed = next_voltage > total_threshold;
        if crossed {
            state.voltage = canonical_zero(
                glif.voltage_reset_scale * next_voltage + glif.voltage_reset_offset_mv,
            );
            state.after[0] =
                canonical_zero(after_amplitudes[0] + next_after_0 * after_cut_decay[0]);
            state.after[1] =
                canonical_zero(after_amplitudes[1] + next_after_1 * after_cut_decay[1]);
            state.threshold_spike = canonical_zero(
                next_spike_threshold * glif.spike_cut_threshold_decay
                    + glif.threshold_spike_increment_mv,
            );
            state.refractory = glif.spike_cut_steps;
        }
    }
    state.synaptic_current = canonical_zero(state.synaptic_current * recurrent_decay);
    state.eligibility = canonical_zero(state.eligibility * eligibility_decay);
    if crossed {
        state.eligibility = canonical_zero(state.eligibility + 1.0);
    }
    crossed
}

#[inline]
fn dense_cell_step(
    glif: &GlifConfig,
    after_decay: &[f64; 2],
    after_cut_decay: &[f64; 2],
    after_amplitudes: &[f64; 2],
    recurrent_decay: f64,
    eligibility_decay: f64,
    state: &mut CellState,
) -> bool {
    // Intentionally separate from sparse_cell_step: this is the small dense
    // native oracle, not an alternate traversal of the sparse implementation.
    let mut spiked = false;
    if state.refractory > 0 {
        state.refractory -= 1;
    } else {
        let asc = state.after[0] + state.after[1];
        let current = state.synaptic_current + asc;
        let old_voltage = state.voltage;
        let new_voltage =
            old_voltage + glif.voltage_step_scale * (current - glif.leak_ns * old_voltage);
        let beta = current * glif.threshold_beta_scale;
        let new_voltage_threshold = glif.threshold_exp_decay * state.threshold_voltage
            + glif.threshold_voltage_factor * old_voltage
            + glif.threshold_beta_factor * beta;
        let new_spike_threshold = state.threshold_spike * glif.spike_threshold_decay;
        let new_after = [
            state.after[0] * after_decay[0],
            state.after[1] * after_decay[1],
        ];
        state.voltage = canonical_zero(new_voltage);
        state.threshold_voltage = canonical_zero(new_voltage_threshold);
        state.threshold_spike = canonical_zero(new_spike_threshold);
        state.after = [canonical_zero(new_after[0]), canonical_zero(new_after[1])];
        spiked = new_voltage
            > glif.threshold_inf_relative_mv + new_voltage_threshold + new_spike_threshold;
        if spiked {
            state.voltage = canonical_zero(
                glif.voltage_reset_scale * new_voltage + glif.voltage_reset_offset_mv,
            );
            state.after = [
                canonical_zero(after_amplitudes[0] + new_after[0] * after_cut_decay[0]),
                canonical_zero(after_amplitudes[1] + new_after[1] * after_cut_decay[1]),
            ];
            state.threshold_spike = canonical_zero(
                new_spike_threshold * glif.spike_cut_threshold_decay
                    + glif.threshold_spike_increment_mv,
            );
            state.refractory = glif.spike_cut_steps;
        }
    }
    state.synaptic_current = canonical_zero(state.synaptic_current * recurrent_decay);
    state.eligibility = canonical_zero(state.eligibility * eligibility_decay);
    if spiked {
        state.eligibility = canonical_zero(state.eligibility + 1.0);
    }
    spiked
}

pub struct ProceduralRuntime {
    glif: GlifConfig,
    config: ProceduralConfig,
    after_decay: [f64; 2],
    after_cut_decay: [f64; 2],
    after_amplitudes: [f64; 2],
    current_step: u64,
    /// Long-lived resident cells in ascending GID order.
    residents: Vec<CellState>,
    /// Cells materialized by the current step's deliveries. Production events
    /// arrive in canonical target order, so this is already an ascending unique
    /// run and insertion is O(1); it is merged back during the fused pass.
    staged: Vec<CellState>,
    /// GID -> slot reference index retained for the in-module noncanonical
    /// scalar oracle and its payload/layout parity checks. The shipped advance
    /// path leaves it empty: canonical same-target events are consecutive and
    /// resolve against `staged.last()` in O(1), while new targets append in
    /// ascending order. Never iterated for semantics.
    staged_index: BTreeMap<u64, u32>,
    /// Pending future events. Slots `[0..pending_head)` are consumed; the
    /// live queue is `[pending_head..)` in canonical ascending key order.
    pending: Vec<ProceduralEventRecord>,
    pending_head: usize,
    last_spikes: Vec<ProceduralSpikeRecord>,
    /// Event-proportional mode only: fast-rest cells with a nonzero
    /// eligibility tail, keyed by GID. Dynamics-inert (they cannot spike or
    /// affect any other cell), so they leave the per-step fused pass entirely;
    /// their eligibility decays lazily on next touch. Always empty in
    /// full-touch mode.
    dormant: BTreeMap<u64, DormantEntry>,
    scratch: RuntimeScratch,
    counters: ProceduralCounters,
    poisoned: u32,
}

impl ProceduralRuntime {
    fn new(
        glif: GlifConfig,
        config: ProceduralConfig,
        after_decay: [f64; 2],
        after_cut_decay: [f64; 2],
        after_amplitudes: [f64; 2],
    ) -> Result<Self, u32> {
        if !matches!(
            config.stepping_mode,
            PROCEDURAL_STEPPING_FULL_TOUCH | PROCEDURAL_STEPPING_EVENT_PROPORTIONAL
        ) {
            // Fail-stop at construction: the C-ABI seam validates through
            // `valid_common`, but the runtime constructor must reject an
            // undefined stepping mode on every path.
            return Err(PROCEDURAL_INVALID);
        }
        let residents = reserved_vec(config.resident_capacity)?;
        let staged_capacity = config.resident_capacity.min(config.event_capacity);
        // Slot offsets into `staged` are stored as u32 in the delivery-phase
        // membership index; fail loudly at construction if a capacity could
        // ever overflow that width.
        debug_assert!(staged_capacity <= u32::MAX as usize);
        let staged = reserved_vec(staged_capacity)?;
        let pending = reserved_vec(config.event_capacity)?;
        let last_spikes = reserved_vec(config.spike_capacity)?;
        let scratch = RuntimeScratch::try_new(&config)?;
        Ok(Self {
            glif,
            config,
            after_decay,
            after_cut_decay,
            after_amplitudes,
            current_step: 0,
            residents,
            staged,
            staged_index: BTreeMap::new(),
            pending,
            pending_head: 0,
            last_spikes,
            dormant: BTreeMap::new(),
            scratch,
            counters: ProceduralCounters::default(),
            poisoned: PROCEDURAL_OK,
        })
    }

    #[inline]
    fn event_proportional(&self) -> bool {
        self.config.stepping_mode == PROCEDURAL_STEPPING_EVENT_PROPORTIONAL
    }

    /// Bit-exact lazy eligibility decay. Replays the identical per-step
    /// multiplies the full-touch path would have executed between the last
    /// eager decay and the current delivery step. Delivery happens before a
    /// step's fused pass, so the pass of `elig_step` itself already ran and
    /// the passes `elig_step+1 ..= current_step-1` are outstanding.
    fn catch_up_eligibility(&self, value: f64, elig_step: u64) -> f64 {
        let mut value = value;
        let gap = self
            .current_step
            .saturating_sub(elig_step.saturating_add(1));
        for _ in 0..gap {
            value = canonical_zero(value * self.config.eligibility_decay);
        }
        value
    }

    fn payload_bytes(&self) -> usize {
        size_of::<Self>()
            + self.residents.capacity() * size_of::<CellState>()
            + self.staged.capacity() * size_of::<CellState>()
            + self.pending.capacity() * size_of::<ProceduralEventRecord>()
            + self.last_spikes.capacity() * size_of::<ProceduralSpikeRecord>()
            // Conservative B-tree node estimate (key + value plus order and
            // parent linkage). Always zero in full-touch mode, so the
            // reported payload is bit-identical to the shipped path there.
            + self
                .dormant
                .len()
                * (size_of::<u64>()
                    + size_of::<DormantEntry>()
                    + 4 * size_of::<usize>())
            // Preserve the established reference-path B-tree accounting term.
            // It is identically zero in the shipped canonical advance path,
            // including every successful counter checkpoint.
            + self
                .staged_index
                .len()
                * (size_of::<u64>()
                    + size_of::<u32>()
                    + 4 * size_of::<usize>())
            + self.scratch.payload_bytes()
    }

    fn poison(&mut self, code: u32) -> u32 {
        self.poisoned = code;
        self.counters.overflow_code = code;
        code
    }

    /// Materialized cells at this instant, including the current step's
    /// staged insertions. Dormant cells occupy resident-capacity slots exactly
    /// as they did while eagerly stepped, so capacity pressure and the
    /// fail-stop abort timing match the full-touch path; in full-touch mode
    /// the dormant store is always empty and this reduces to the original sum.
    #[inline]
    fn total_residents(&self) -> usize {
        self.residents.len() + self.staged.len() + self.dormant.len()
    }

    /// Fold the staged insertions back into the sorted resident array,
    /// restoring the "all residents in one ascending array" invariant used by
    /// the exact fallback path and by snapshots.
    ///
    /// The merge runs backwards over the grown array: the write cursor starts
    /// past every unmerged slot (`write == read + staged_index` is kept as an
    /// invariant), so no survivor is ever written out of bounds and no
    /// not-yet-read resident is ever clobbered.
    #[cfg(test)]
    fn absorb_staged(&mut self) -> Result<(), u32> {
        self.absorb_staged_with_order::<false>()
    }

    fn absorb_staged_with_order<const CANONICAL_STAGED_RUN: bool>(&mut self) -> Result<(), u32> {
        if self.staged.is_empty() {
            debug_assert!(self.staged_index.is_empty());
            return Ok(());
        }
        if !CANONICAL_STAGED_RUN {
            self.staged.sort_unstable_by_key(|state| state.gid);
        }
        debug_assert!(self
            .staged
            .windows(2)
            .all(|window| window[0].gid < window[1].gid));
        let main_len = self.residents.len();
        let staged_len = self.staged.len();
        let merged_len = main_len + staged_len;
        if merged_len > self.residents.capacity() {
            return Err(PROCEDURAL_ALLOCATION);
        }
        // Pre-grow the resident array to the merged length; every appended
        // placeholder slot is overwritten exactly once by the merge.
        while self.residents.len() < merged_len {
            self.residents.push(CellState::canonical_rest(u64::MAX));
        }
        let mut read = main_len;
        let mut staged_index = staged_len;
        let mut write = merged_len;
        while staged_index > 0 {
            if read > 0 && self.residents[read - 1].gid > self.staged[staged_index - 1].gid {
                read -= 1;
                write -= 1;
                self.residents[write] = self.residents[read];
            } else {
                staged_index -= 1;
                write -= 1;
                self.residents[write] = self.staged[staged_index];
            }
        }
        debug_assert_eq!(read, write);
        self.staged.clear();
        // Clear the reference-path index beside the Vec. Production keeps it
        // empty, so this preserves the old lifecycle and accounting at O(1).
        self.staged_index.clear();
        Ok(())
    }

    fn resident_index(&self, gid: u64) -> Result<usize, usize> {
        self.residents.binary_search_by_key(&gid, |state| state.gid)
    }

    /// Targeted active-residency lookup backing
    /// `humanbrain_procedural_runtime_read_cells`: binary-search the ascending
    /// resident array first, then `staged`.
    /// Precedence is unreachable under the disjointness invariant D1
    /// (`residents`/`staged`/`dormant` are pairwise disjoint; residents gain
    /// members only via `absorb_staged`, dormant reactivation removes the
    /// key from `dormant` before pushing to `staged`, and fresh staging only
    /// happens after misses in all three structures); it exists so the read
    /// contract stays total without relying on an invariant an FFI caller
    /// cannot observe.
    fn read_cell(&self, gid: u64) -> Option<&CellState> {
        if let Ok(index) = self.resident_index(gid) {
            return Some(&self.residents[index]);
        }
        match self.staged.binary_search_by_key(&gid, |state| state.gid) {
            Ok(index) => {
                // `read_cells` runs only between top-level calls, where every
                // non-poisoned exit of `advance` has left `staged` empty and
                // every delivery-phase push order can only be observed after
                // a poisoned abort (which read_cells refuses). The sortedness
                // guard turns any future lifecycle mistake into a loud test
                // failure instead of a silently misordered binary search.
                debug_assert!(self
                    .staged
                    .windows(2)
                    .all(|window| window[0].gid <= window[1].gid));
                Some(&self.staged[index])
            }
            Err(_) => None,
        }
    }

    /// Live (unconsumed) prefix of the pending-event queue, ascending by key.
    #[inline]
    fn pending_slice(&self) -> &[ProceduralEventRecord] {
        &self.pending[self.pending_head..]
    }

    #[inline]
    fn pending_logical_len(&self) -> usize {
        self.pending.len() - self.pending_head
    }

    /// Release the consumed prefix once the whole queue is drained; larger
    /// dead prefixes are compacted lazily so per-step delivery stays O(1)
    /// amortized instead of shifting the live tail every step.
    #[inline]
    fn release_consumed_prefix(&mut self) {
        if self.pending_head == self.pending.len() {
            self.pending.clear();
            self.pending_head = 0;
        } else if self.pending_head >= 64 && self.pending_head * 2 >= self.pending.len() {
            self.pending.drain(..self.pending_head);
            self.pending_head = 0;
        }
    }

    #[cfg(test)]
    fn apply_event(&mut self, event: ProceduralEventRecord) -> Result<(), u32> {
        self.apply_event_with_fresh_run::<false, false>(event)
    }

    fn push_staged<const CANONICAL_STAGED_RUN: bool>(&mut self, state: CellState) {
        if CANONICAL_STAGED_RUN {
            debug_assert!(self
                .staged
                .last()
                .is_none_or(|previous| previous.gid < state.gid));
        }
        self.staged.push(state);
        if !CANONICAL_STAGED_RUN {
            self.staged_index
                .insert(state.gid, self.staged.len() as u32 - 1);
        }
    }

    fn apply_event_with_fresh_run<const MERGE_FRESH_RUN: bool, const CANONICAL_STAGED_RUN: bool>(
        &mut self,
        event: ProceduralEventRecord,
    ) -> Result<(), u32> {
        let current = canonical_zero(event.current_pa);
        match self.resident_index(event.target_gid) {
            Ok(index) => {
                let value = self.residents[index].synaptic_current + current;
                self.residents[index].synaptic_current = canonical_zero(value);
            }
            Err(_) => {
                if self.event_proportional() {
                    if let Some(entry) = self.dormant.remove(&event.target_gid) {
                        // Reactivation from the dormant store. The full-touch
                        // path would have decayed this cell's eligibility on
                        // every intervening step; replay those exact multiplies
                        // now so the restored state is bit-identical to what
                        // the resident array would have held at delivery time.
                        let restored =
                            self.catch_up_eligibility(entry.eligibility, entry.elig_step);
                        debug_assert!(restored.to_bits() != 0 || entry.eligibility.to_bits() == 0);
                        if current.to_bits() == 0 {
                            // A zero-current delivery touches nothing in the
                            // full-touch path either (canonical zero add);
                            // leave the cell dormant. Its raw value stays as
                            // of `entry.elig_step`, which only defers -- never
                            // changes -- the replayed multiply sequence.
                            self.dormant.insert(
                                event.target_gid,
                                DormantEntry {
                                    eligibility: entry.eligibility,
                                    elig_step: entry.elig_step,
                                },
                            );
                            return Ok(());
                        }
                        let mut state = CellState::canonical_rest(event.target_gid);
                        state.eligibility = restored;
                        state.synaptic_current = current;
                        // Net-neutral on total_residents(): one dormant slot
                        // becomes one staged slot, so no capacity check is due
                        // -- mirroring the full-touch path, where an event to
                        // an existing resident performs none either.
                        self.push_staged::<CANONICAL_STAGED_RUN>(state);
                        return Ok(());
                    }
                }
                let staged_slot = if CANONICAL_STAGED_RUN {
                    self.staged
                        .last()
                        .filter(|state| state.gid == event.target_gid)
                        .map(|_| self.staged.len() - 1)
                } else {
                    self.staged_index
                        .get(&event.target_gid)
                        .map(|slot| *slot as usize)
                };
                if let Some(slot) = staged_slot {
                    let value = self.staged[slot].synaptic_current + current;
                    self.staged[slot].synaptic_current = canonical_zero(value);
                    return Ok(());
                }
                if current.to_bits() == 0 {
                    return Ok(());
                }
                if self.total_residents() == self.config.resident_capacity {
                    return Err(PROCEDURAL_RESIDENT_CAPACITY);
                }
                if self.total_residents() == self.residents.capacity() + self.staged.capacity() {
                    return Err(PROCEDURAL_ALLOCATION);
                }
                if MERGE_FRESH_RUN {
                    self.scratch.interval_ledger.stage_single_in_run(
                        event.target_gid,
                        SEEN_RESIDENT,
                        &mut self.scratch.gid_run,
                        self.config.interval_distinct_capacity,
                    )?;
                } else {
                    self.scratch.interval_ledger.mark_single(
                        event.target_gid,
                        SEEN_RESIDENT,
                        self.config.interval_distinct_capacity,
                    )?;
                }
                let mut state = CellState::canonical_rest(event.target_gid);
                state.synaptic_current = current;
                self.push_staged::<CANONICAL_STAGED_RUN>(state);
            }
        }
        Ok(())
    }

    fn append_events(&mut self, events: &[ProceduralEventRecord]) -> Result<(), u32> {
        if events.len() > self.config.event_capacity {
            return Err(PROCEDURAL_EVENT_CAPACITY);
        }
        self.scratch.submitted_events.clear();
        for event in events {
            if event.due_step < self.current_step
                || event.target_gid >= self.config.configured_cells
                || event.source_gid >= self.config.configured_cells
                || event.edge_slot >= self.config.fanout
                || !event.current_pa.is_finite()
            {
                return Err(PROCEDURAL_INVALID);
            }
            let mut canonical = *event;
            canonical.current_pa = canonical_zero(canonical.current_pa);
            push_bounded(
                &mut self.scratch.submitted_events,
                canonical,
                self.config.event_capacity,
                PROCEDURAL_EVENT_CAPACITY,
            )?;
        }
        self.scratch.submitted_events.sort_unstable_by(event_order);
        if self
            .scratch
            .submitted_events
            .windows(2)
            .any(|pair| same_event_key(&pair[0], &pair[1]))
        {
            return Err(PROCEDURAL_DUPLICATE_EVENT);
        }
        let mut pending_index = 0usize;
        let mut submitted_index = 0usize;
        let pending_head = self.pending_head;
        let pending = &self.pending[pending_head..];
        while pending_index < pending.len() && submitted_index < self.scratch.submitted_events.len()
        {
            match event_order(
                &pending[pending_index],
                &self.scratch.submitted_events[submitted_index],
            ) {
                Ordering::Less => pending_index += 1,
                Ordering::Greater => submitted_index += 1,
                Ordering::Equal => return Err(PROCEDURAL_DUPLICATE_EVENT),
            }
        }
        let merged_len = self
            .pending_logical_len()
            .checked_add(self.scratch.submitted_events.len())
            .ok_or(PROCEDURAL_EVENT_CAPACITY)?;
        if merged_len > self.config.event_capacity {
            return Err(PROCEDURAL_EVENT_CAPACITY);
        }
        self.scratch.event_work.clear();
        pending_index = 0;
        submitted_index = 0;
        while pending_index < pending.len() || submitted_index < self.scratch.submitted_events.len()
        {
            let take_pending = submitted_index == self.scratch.submitted_events.len()
                || (pending_index < pending.len()
                    && event_order(
                        &pending[pending_index],
                        &self.scratch.submitted_events[submitted_index],
                    ) == Ordering::Less);
            let event = if take_pending {
                let event = pending[pending_index];
                pending_index += 1;
                event
            } else {
                let event = self.scratch.submitted_events[submitted_index];
                submitted_index += 1;
                event
            };
            push_bounded(
                &mut self.scratch.event_work,
                event,
                self.config.event_capacity,
                PROCEDURAL_EVENT_CAPACITY,
            )?;
        }
        std::mem::swap(&mut self.pending, &mut self.scratch.event_work);
        self.pending_head = 0;
        self.scratch.event_work.clear();
        Ok(())
    }

    /// Order the live queue after a generation phase. Only the appended tail
    /// is sorted; it is then linear-merged with the already-sorted live
    /// prefix, yielding exactly the fully ascending canonical order a
    /// whole-queue sort would produce. Duplicate canonical keys are detected
    /// within the tail or across the seam; the merged queue is committed
    /// before reporting the duplicate so a poisoned snapshot shows the same
    /// byte content as the previous implementation.
    fn finish_step_queue_order(&mut self, tail_start: usize) -> Result<(), u32> {
        let head = self.pending_head;
        if tail_start == self.pending.len() {
            // Nothing was generated this step; the live queue is unchanged
            // and duplicate-free by construction.
            return Ok(());
        }
        {
            let tail = &mut self.pending[tail_start..];
            tail.sort_unstable_by(event_order);
        }
        let mut duplicate = self.pending[tail_start..]
            .windows(2)
            .any(|pair| same_event_key(&pair[0], &pair[1]));
        if head == tail_start {
            // No live prefix: the sorted tail is the entire live queue.
            self.pending_head = tail_start;
            if duplicate {
                return Err(PROCEDURAL_DUPLICATE_EVENT);
            }
            return Ok(());
        }
        let capacity = self.config.event_capacity;
        {
            let queue = &self.pending;
            let work = &mut self.scratch.event_work;
            work.clear();
            let mut prefix_index = head;
            let mut tail_index = tail_start;
            while prefix_index < tail_start || tail_index < queue.len() {
                let take_prefix = if prefix_index == tail_start {
                    false
                } else if tail_index == queue.len() {
                    true
                } else {
                    match event_order(&queue[prefix_index], &queue[tail_index]) {
                        Ordering::Less => true,
                        Ordering::Equal => {
                            duplicate = true;
                            true
                        }
                        Ordering::Greater => false,
                    }
                };
                let source_index = if take_prefix {
                    prefix_index
                } else {
                    tail_index
                };
                push_bounded(
                    work,
                    queue[source_index],
                    capacity,
                    PROCEDURAL_EVENT_CAPACITY,
                )?;
                if take_prefix {
                    prefix_index += 1;
                } else {
                    tail_index += 1;
                }
            }
        }
        std::mem::swap(&mut self.pending, &mut self.scratch.event_work);
        self.pending_head = 0;
        self.scratch.event_work.clear();
        if duplicate {
            return Err(PROCEDURAL_DUPLICATE_EVENT);
        }
        Ok(())
    }

    fn advance(&mut self, step_count: u64, events: &[ProceduralEventRecord]) -> u32 {
        self.advance_with_fresh_run::<true, true>(step_count, events)
    }

    fn advance_with_fresh_run<const MERGE_FRESH_RUN: bool, const CANONICAL_STAGED_RUN: bool>(
        &mut self,
        step_count: u64,
        events: &[ProceduralEventRecord],
    ) -> u32 {
        if self.poisoned != PROCEDURAL_OK {
            return PROCEDURAL_POISONED;
        }
        if step_count == 0 || self.current_step.checked_add(step_count).is_none() {
            return self.poison(PROCEDURAL_STEP_OVERFLOW);
        }
        if let Err(code) = self.append_events(events) {
            return if code == PROCEDURAL_DUPLICATE_EVENT {
                code
            } else {
                self.poison(code)
            };
        }

        self.last_spikes.clear();
        self.scratch.interval_ledger.clear();
        if !self.residents.is_empty() {
            // Entry residents are already ascending-unique: mark them as one
            // sorted run instead of one hash operation per cell.
            let distinct_capacity = self.config.interval_distinct_capacity;
            self.scratch.gid_run.clear();
            self.scratch
                .gid_run
                .extend(self.residents.iter().map(|state| state.gid));
            if self
                .scratch
                .interval_ledger
                .can_fit_run(&self.scratch.gid_run, distinct_capacity)
            {
                let run = &self.scratch.gid_run;
                let ledger = &mut self.scratch.interval_ledger;
                ledger.merge_run_in_place(run, SEEN_RESIDENT);
            } else {
                // Exact fallback near the distinct capacity: per-GID marks
                // keep the original fail-stop abort timing.
                for state in &self.residents {
                    if let Err(code) = self.scratch.interval_ledger.mark_single(
                        state.gid,
                        SEEN_RESIDENT,
                        distinct_capacity,
                    ) {
                        return self.poison(code);
                    }
                }
            }
        }
        let mut resident_peak = self.total_residents();
        let mut queue_peak = self.pending_logical_len();
        let mut cell_steps = 0u64;
        let mut generated_edges = 0u64;
        let mut delivered_events = 0u64;

        for _ in 0..step_count {
            // Reuse the resident-capacity scratch first as this step's
            // ascending run of absent nonzero event targets. The same vector
            // is cleared below and rebuilt as the stepped-population run.
            self.scratch.gid_run.clear();
            let due_count = self
                .pending_slice()
                .partition_point(|event| event.due_step <= self.current_step);
            if self.pending_slice()[..due_count]
                .iter()
                .any(|event| event.due_step != self.current_step)
            {
                return self.poison(PROCEDURAL_INVALID);
            }
            self.scratch.event_work.clear();
            let consumption_base = self.pending_head;
            let due = &self.pending[consumption_base..consumption_base + due_count];
            for event in due {
                if let Err(code) = push_bounded(
                    &mut self.scratch.event_work,
                    *event,
                    self.config.event_capacity,
                    PROCEDURAL_EVENT_CAPACITY,
                ) {
                    return self.poison(code);
                }
            }
            for index in 0..self.scratch.event_work.len() {
                let event = self.scratch.event_work[index];
                delivered_events = match delivered_events.checked_add(1) {
                    Some(value) => value,
                    None => return self.poison(PROCEDURAL_STEP_OVERFLOW),
                };
                if let Err(code) =
                    self.apply_event_with_fresh_run::<MERGE_FRESH_RUN, CANONICAL_STAGED_RUN>(event)
                {
                    return self.poison(code);
                }
            }
            if MERGE_FRESH_RUN && !self.scratch.gid_run.is_empty() {
                let run = &self.scratch.gid_run;
                let ledger = &mut self.scratch.interval_ledger;
                ledger.merge_run_in_place(run, SEEN_RESIDENT);
            }
            self.pending_head += due_count;
            self.release_consumed_prefix();
            resident_peak = resident_peak.max(self.total_residents());

            self.scratch.step_spikes.clear();
            self.scratch.dormant_values.clear();
            // The fused pass below integrates exactly these cells; dormant
            // cells are retired from the stepped set in event-proportional
            // mode, so this work count is where their cost disappears.
            let population = self.residents.len() + self.staged.len();
            cell_steps = match cell_steps.checked_add(population as u64) {
                Some(value) => value,
                None => return self.poison(PROCEDURAL_STEP_OVERFLOW),
            };
            if population > 0 {
                // Ascending merged run of every GID stepped this step.
                if !CANONICAL_STAGED_RUN {
                    self.staged.sort_unstable_by_key(|state| state.gid);
                }
                debug_assert!(self
                    .staged
                    .windows(2)
                    .all(|window| window[0].gid < window[1].gid));
                {
                    let run = &mut self.scratch.gid_run;
                    run.clear();
                    let mut staged_index = 0usize;
                    for state in &self.residents {
                        while staged_index < self.staged.len()
                            && self.staged[staged_index].gid < state.gid
                        {
                            run.push(self.staged[staged_index].gid);
                            staged_index += 1;
                        }
                        run.push(state.gid);
                    }
                    while staged_index < self.staged.len() {
                        run.push(self.staged[staged_index].gid);
                        staged_index += 1;
                    }
                }
                let distinct_capacity = self.config.interval_distinct_capacity;
                let fast_path = self
                    .scratch
                    .interval_ledger
                    .can_fit_run(&self.scratch.gid_run, distinct_capacity);
                if fast_path {
                    // Materialize this step's insertions into the sorted
                    // resident array, then run the fused update+compact pass
                    // over the single ascending array in place. This replaces
                    // the previous per-cell hash marks and the trailing
                    // retain() scan with one sequential sweep.
                    if let Err(code) = self.absorb_staged_with_order::<CANONICAL_STAGED_RUN>() {
                        return self.poison(code);
                    }
                    {
                        // Pre-verified to fit: account for the whole stepped
                        // population before reusing its run for dormant GIDs.
                        // This integer-only scratch mutation is unobservable
                        // until successful counters are committed below and
                        // does not change the following GLIF operation order.
                        let run = &self.scratch.gid_run;
                        let ledger = &mut self.scratch.interval_ledger;
                        ledger.merge_run_in_place(run, SEEN_UPDATED);
                    }
                    self.scratch.gid_run.clear();
                    let mut write = 0usize;
                    for read in 0..self.residents.len() {
                        let mut state = self.residents[read];
                        let crossed = sparse_cell_step(
                            &self.glif,
                            &self.after_decay,
                            &self.after_cut_decay,
                            &self.after_amplitudes,
                            self.config.recurrent_decay,
                            self.config.eligibility_decay,
                            &mut state,
                        );
                        if crossed {
                            if let Err(code) = push_bounded(
                                &mut self.scratch.step_spikes,
                                state.gid,
                                self.config.resident_capacity,
                                PROCEDURAL_RESIDENT_CAPACITY,
                            ) {
                                return self.poison(code);
                            }
                            if let Err(code) = push_bounded(
                                &mut self.last_spikes,
                                ProceduralSpikeRecord {
                                    step: self.current_step,
                                    gid: state.gid,
                                },
                                self.config.spike_capacity,
                                PROCEDURAL_SPIKE_CAPACITY,
                            ) {
                                return self.poison(code);
                            }
                        }
                        if !state.is_canonical_rest() {
                            if self.event_proportional() && state.is_fast_rest() {
                                // Only the eligibility tail remains. The cell
                                // cannot spike, change any other field, or
                                // affect another cell; retire it from the
                                // stepped set. Its eligibility was decayed by
                                // this very pass, so it is recorded as of the
                                // current step and lazily replayed from there.
                                debug_assert!(state.eligibility.to_bits() != 0);
                                self.scratch.gid_run.push(state.gid);
                                self.scratch.dormant_values.push(state.eligibility);
                            } else {
                                self.residents[write] = state;
                                write += 1;
                            }
                        }
                    }
                    debug_assert!(write <= self.residents.len());
                    self.residents.truncate(write);
                    if !self.scratch.step_spikes.is_empty() {
                        // Spiking GIDs are a subset of the stepped run just
                        // merged, so this never grows the ledger and can
                        // never trip the distinct capacity.
                        let run = &self.scratch.step_spikes;
                        let ledger = &mut self.scratch.interval_ledger;
                        ledger.merge_run_in_place(run, SEEN_SPIKING);
                    }
                } else {
                    // Exact fallback near the distinct capacity: materialize
                    // the staged insertions into the sorted array, then run
                    // the original per-cell loop whose abort timing matches
                    // the previous implementation bit for bit.
                    if let Err(code) = self.absorb_staged_with_order::<CANONICAL_STAGED_RUN>() {
                        return self.poison(code);
                    }
                    // The population run selected the exact fallback above;
                    // per-cell marking now owns accounting, so reuse it for
                    // the ascending dormant-retirement GIDs.
                    self.scratch.gid_run.clear();
                    for index in 0..self.residents.len() {
                        let mut retired: Option<(u64, f64)> = None;
                        let event_proportional = self.event_proportional();
                        {
                            let state = &mut self.residents[index];
                            if let Err(code) = self.scratch.interval_ledger.mark_single(
                                state.gid,
                                SEEN_UPDATED,
                                distinct_capacity,
                            ) {
                                return self.poison(code);
                            }
                            if sparse_cell_step(
                                &self.glif,
                                &self.after_decay,
                                &self.after_cut_decay,
                                &self.after_amplitudes,
                                self.config.recurrent_decay,
                                self.config.eligibility_decay,
                                state,
                            ) {
                                if let Err(code) = self.scratch.interval_ledger.mark_single(
                                    state.gid,
                                    SEEN_SPIKING,
                                    distinct_capacity,
                                ) {
                                    return self.poison(code);
                                }
                                if let Err(code) = push_bounded(
                                    &mut self.scratch.step_spikes,
                                    state.gid,
                                    self.config.resident_capacity,
                                    PROCEDURAL_RESIDENT_CAPACITY,
                                ) {
                                    return self.poison(code);
                                }
                                if let Err(code) = push_bounded(
                                    &mut self.last_spikes,
                                    ProceduralSpikeRecord {
                                        step: self.current_step,
                                        gid: state.gid,
                                    },
                                    self.config.spike_capacity,
                                    PROCEDURAL_SPIKE_CAPACITY,
                                ) {
                                    return self.poison(code);
                                }
                            }
                            if !state.is_canonical_rest()
                                && event_proportional
                                && state.is_fast_rest()
                            {
                                debug_assert!(state.eligibility.to_bits() != 0);
                                retired = Some((state.gid, state.eligibility));
                            }
                        }
                        if let Some((gid, eligibility)) = retired {
                            self.scratch.gid_run.push(gid);
                            self.scratch.dormant_values.push(eligibility);
                        }
                    }
                    if !self.scratch.gid_run.is_empty() {
                        // Partition out cells retired to the dormant store;
                        // both arrays ascend by GID, so one merge sweep evicts
                        // exactly the recorded runs (canonical-rest cells are
                        // dropped here too, matching the retain below).
                        let mut write = 0usize;
                        let mut dormant_index = 0usize;
                        for read in 0..self.residents.len() {
                            let state = self.residents[read];
                            if dormant_index < self.scratch.gid_run.len()
                                && self.scratch.gid_run[dormant_index] == state.gid
                            {
                                dormant_index += 1;
                                continue;
                            }
                            if state.is_canonical_rest() {
                                continue;
                            }
                            self.residents[write] = state;
                            write += 1;
                        }
                        debug_assert_eq!(dormant_index, self.scratch.gid_run.len());
                        self.residents.truncate(write);
                    } else {
                        self.residents.retain(|state| !state.is_canonical_rest());
                    }
                }
            }

            debug_assert_eq!(
                self.scratch.gid_run.len(),
                self.scratch.dormant_values.len()
            );
            if !self.scratch.gid_run.is_empty() {
                debug_assert!(self.event_proportional());
                for index in 0..self.scratch.gid_run.len() {
                    self.dormant.insert(
                        self.scratch.gid_run[index],
                        DormantEntry {
                            eligibility: self.scratch.dormant_values[index],
                            elig_step: self.current_step,
                        },
                    );
                }
            }

            let generated_this_step = match self
                .scratch
                .step_spikes
                .len()
                .checked_mul(self.config.fanout as usize)
            {
                Some(value) => value,
                None => return self.poison(PROCEDURAL_EVENT_CAPACITY),
            };
            if self
                .pending_logical_len()
                .saturating_add(generated_this_step)
                > self.config.event_capacity
            {
                return self.poison(PROCEDURAL_EVENT_CAPACITY);
            }
            // Newly generated events all land after the live sorted prefix, so
            // only the appended tail is sorted and then linearly merged back.
            // The result is byte-identical to sorting the whole live queue:
            // both produce the full queue in ascending canonical key order.
            let tail_start = self.pending.len();
            for spike_index in 0..self.scratch.step_spikes.len() {
                let source = self.scratch.step_spikes[spike_index];
                let due_step = match self
                    .current_step
                    .checked_add(self.config.recurrent_delay_steps)
                {
                    Some(value) => value,
                    None => return self.poison(PROCEDURAL_STEP_OVERFLOW),
                };
                let weight = source_weight(&self.config, source);
                if let Err(code) = fill_targets_with(
                    &self.config,
                    source,
                    &mut self.scratch.row_targets,
                    &mut self.scratch.row_used,
                    |config, source, slot, attempt| {
                        let word =
                            counter_words(config, source, slot, GLOBAL_PURPOSE_ID, attempt).0;
                        let candidate = bounded_index(word, config.configured_cells - 1);
                        if candidate < source {
                            candidate
                        } else {
                            candidate + 1
                        }
                    },
                ) {
                    return self.poison(code);
                }
                for slot in 0..self.scratch.row_targets.len() {
                    let event = ProceduralEventRecord {
                        due_step,
                        target_gid: self.scratch.row_targets[slot],
                        source_gid: source,
                        edge_slot: slot as u64,
                        current_pa: weight,
                    };
                    if let Err(code) = push_bounded(
                        &mut self.pending,
                        event,
                        self.config.event_capacity,
                        PROCEDURAL_EVENT_CAPACITY,
                    ) {
                        return self.poison(code);
                    }
                    generated_edges = match generated_edges.checked_add(1) {
                        Some(value) => value,
                        None => return self.poison(PROCEDURAL_STEP_OVERFLOW),
                    };
                }
            }
            if let Err(code) = self.finish_step_queue_order(tail_start) {
                return self.poison(code);
            }
            queue_peak = queue_peak.max(self.pending_logical_len());
            self.current_step += 1;
        }

        let resident_distinct = self.scratch.interval_ledger.count_flags(SEEN_RESIDENT);
        let updated_distinct = self.scratch.interval_ledger.count_flags(SEEN_UPDATED);
        let spiking_distinct = self.scratch.interval_ledger.count_flags(SEEN_SPIKING);
        self.counters = ProceduralCounters {
            current_step: self.current_step,
            resident_peak,
            resident_distinct,
            updated_distinct,
            spiking_distinct,
            cell_step_evaluations: cell_steps,
            spike_events: self.last_spikes.len() as u64,
            generated_edges,
            delivered_events,
            configured_ids_scanned: 0,
            queue_peak,
            native_payload_bytes: self.payload_bytes(),
            retained_nonzero_cells: self.total_residents(),
            overflow_code: PROCEDURAL_OK,
            padding_zero: 0,
        };
        PROCEDURAL_OK
    }
}

pub struct DenseReference {
    glif: GlifConfig,
    config: ProceduralConfig,
    after_decay: [f64; 2],
    after_cut_decay: [f64; 2],
    after_amplitudes: [f64; 2],
    targets: Vec<u64>,
    weights: Vec<f64>,
    current_step: u64,
    cells: Vec<CellState>,
    pending: Vec<ProceduralEventRecord>,
    last_spikes: Vec<ProceduralSpikeRecord>,
    scratch: DenseScratch,
    poisoned: u32,
}

impl DenseReference {
    fn append_events(&mut self, events: &[ProceduralEventRecord]) -> Result<(), u32> {
        if events.len() > self.config.event_capacity {
            return Err(PROCEDURAL_EVENT_CAPACITY);
        }
        self.scratch.submitted_events.clear();
        for event in events {
            if event.due_step < self.current_step
                || event.target_gid >= self.config.configured_cells
                || event.source_gid >= self.config.configured_cells
                || event.edge_slot >= self.config.fanout
                || !event.current_pa.is_finite()
            {
                return Err(PROCEDURAL_INVALID);
            }
            let mut canonical = *event;
            canonical.current_pa = canonical_zero(canonical.current_pa);
            push_bounded(
                &mut self.scratch.submitted_events,
                canonical,
                self.config.event_capacity,
                PROCEDURAL_EVENT_CAPACITY,
            )?;
        }
        self.scratch
            .submitted_events
            .sort_unstable_by(dense_event_order);
        if self
            .scratch
            .submitted_events
            .windows(2)
            .any(|pair| same_dense_event_key(&pair[0], &pair[1]))
        {
            return Err(PROCEDURAL_DUPLICATE_EVENT);
        }
        let mut pending_index = 0usize;
        let mut submitted_index = 0usize;
        while pending_index < self.pending.len()
            && submitted_index < self.scratch.submitted_events.len()
        {
            match dense_event_order(
                &self.pending[pending_index],
                &self.scratch.submitted_events[submitted_index],
            ) {
                Ordering::Less => pending_index += 1,
                Ordering::Greater => submitted_index += 1,
                Ordering::Equal => return Err(PROCEDURAL_DUPLICATE_EVENT),
            }
        }
        let merged_len = self
            .pending
            .len()
            .checked_add(self.scratch.submitted_events.len())
            .ok_or(PROCEDURAL_EVENT_CAPACITY)?;
        if merged_len > self.config.event_capacity {
            return Err(PROCEDURAL_EVENT_CAPACITY);
        }
        self.scratch.event_work.clear();
        pending_index = 0;
        submitted_index = 0;
        while pending_index < self.pending.len()
            || submitted_index < self.scratch.submitted_events.len()
        {
            let take_pending = submitted_index == self.scratch.submitted_events.len()
                || (pending_index < self.pending.len()
                    && dense_event_order(
                        &self.pending[pending_index],
                        &self.scratch.submitted_events[submitted_index],
                    ) == Ordering::Less);
            let event = if take_pending {
                let event = self.pending[pending_index];
                pending_index += 1;
                event
            } else {
                let event = self.scratch.submitted_events[submitted_index];
                submitted_index += 1;
                event
            };
            push_bounded(
                &mut self.scratch.event_work,
                event,
                self.config.event_capacity,
                PROCEDURAL_EVENT_CAPACITY,
            )?;
        }
        std::mem::swap(&mut self.pending, &mut self.scratch.event_work);
        self.scratch.event_work.clear();
        Ok(())
    }

    fn advance(&mut self, step_count: u64, events: &[ProceduralEventRecord]) -> u32 {
        if self.poisoned != PROCEDURAL_OK {
            return PROCEDURAL_POISONED;
        }
        if step_count == 0 || self.current_step.checked_add(step_count).is_none() {
            self.poisoned = PROCEDURAL_STEP_OVERFLOW;
            return self.poisoned;
        }
        if let Err(code) = self.append_events(events) {
            if code != PROCEDURAL_DUPLICATE_EVENT {
                self.poisoned = code;
            }
            return code;
        }
        self.last_spikes.clear();
        for _ in 0..step_count {
            let due_count = self
                .pending
                .partition_point(|event| event.due_step <= self.current_step);
            if self.pending[..due_count]
                .iter()
                .any(|event| event.due_step != self.current_step)
            {
                self.poisoned = PROCEDURAL_INVALID;
                return self.poisoned;
            }
            self.scratch.event_work.clear();
            for event in &self.pending[..due_count] {
                if let Err(code) = push_bounded(
                    &mut self.scratch.event_work,
                    *event,
                    self.config.event_capacity,
                    PROCEDURAL_EVENT_CAPACITY,
                ) {
                    self.poisoned = code;
                    return code;
                }
            }
            for index in 0..self.scratch.event_work.len() {
                let event = self.scratch.event_work[index];
                let cell = &mut self.cells[event.target_gid as usize];
                cell.synaptic_current = canonical_zero(cell.synaptic_current + event.current_pa);
            }
            drop(self.pending.drain(..due_count));
            self.scratch.step_spikes.clear();
            let dense_cell_count = self.cells.len();
            for cell in &mut self.cells {
                if dense_cell_step(
                    &self.glif,
                    &self.after_decay,
                    &self.after_cut_decay,
                    &self.after_amplitudes,
                    self.config.recurrent_decay,
                    self.config.eligibility_decay,
                    cell,
                ) {
                    if let Err(code) = push_bounded(
                        &mut self.scratch.step_spikes,
                        cell.gid,
                        dense_cell_count,
                        PROCEDURAL_RESIDENT_CAPACITY,
                    ) {
                        self.poisoned = code;
                        return code;
                    }
                    if let Err(code) = push_bounded(
                        &mut self.last_spikes,
                        ProceduralSpikeRecord {
                            step: self.current_step,
                            gid: cell.gid,
                        },
                        self.config.spike_capacity,
                        PROCEDURAL_SPIKE_CAPACITY,
                    ) {
                        self.poisoned = code;
                        return code;
                    }
                }
            }
            let generated = match self
                .scratch
                .step_spikes
                .len()
                .checked_mul(self.config.fanout as usize)
            {
                Some(value) => value,
                None => {
                    self.poisoned = PROCEDURAL_EVENT_CAPACITY;
                    return self.poisoned;
                }
            };
            if self.pending.len().saturating_add(generated) > self.config.event_capacity {
                self.poisoned = PROCEDURAL_EVENT_CAPACITY;
                return self.poisoned;
            }
            for spike_index in 0..self.scratch.step_spikes.len() {
                let source = self.scratch.step_spikes[spike_index];
                let due_step = match self
                    .current_step
                    .checked_add(self.config.recurrent_delay_steps)
                {
                    Some(value) => value,
                    None => {
                        self.poisoned = PROCEDURAL_STEP_OVERFLOW;
                        return self.poisoned;
                    }
                };
                let offset = source as usize * self.config.fanout as usize;
                for slot in 0..self.config.fanout as usize {
                    let event = ProceduralEventRecord {
                        due_step,
                        target_gid: self.targets[offset + slot],
                        source_gid: source,
                        edge_slot: slot as u64,
                        current_pa: self.weights[offset + slot],
                    };
                    if let Err(code) = push_bounded(
                        &mut self.pending,
                        event,
                        self.config.event_capacity,
                        PROCEDURAL_EVENT_CAPACITY,
                    ) {
                        self.poisoned = code;
                        return code;
                    }
                }
            }
            self.pending.sort_unstable_by(dense_event_order);
            if self
                .pending
                .windows(2)
                .any(|pair| same_dense_event_key(&pair[0], &pair[1]))
            {
                self.poisoned = PROCEDURAL_DUPLICATE_EVENT;
                return self.poisoned;
            }
            self.current_step += 1;
        }
        PROCEDURAL_OK
    }
}

fn valid_common(glif: &GlifConfig, config: &ProceduralConfig) -> bool {
    glif.after_channel_count == 2
        && config.configured_cells >= 2
        && config.fanout >= 1
        && config.fanout < config.configured_cells
        && config.fanout < (1u64 << 48)
        && usize::try_from(config.fanout).is_ok()
        && config.resident_capacity >= 1
        && config.resident_capacity as u128 <= config.configured_cells as u128
        && config.event_capacity >= 1
        && config.spike_capacity >= 1
        && config.interval_distinct_capacity >= config.resident_capacity
        && config.recurrent_weight_pa.is_finite()
        && config.recurrent_weight_pa > 0.0
        && config.recurrent_delay_steps >= 1
        && config.recurrent_decay.is_finite()
        && config.recurrent_decay > 0.0
        && config.recurrent_decay < 1.0
        && config.eligibility_decay.is_finite()
        && config.eligibility_decay > 0.0
        && config.eligibility_decay < 1.0
        && matches!(
            config.stepping_mode,
            PROCEDURAL_STEPPING_FULL_TOUCH | PROCEDURAL_STEPPING_EVENT_PROPORTIONAL
        )
        // The dormant fixed-point argument requires that an all-zero cell can
        // never cross: the quiescent threshold floor must be nonnegative.
        && (config.stepping_mode == PROCEDURAL_STEPPING_FULL_TOUCH
            || glif.threshold_inf_relative_mv >= 0.0)
}

unsafe fn read_common(
    glif: *const GlifConfig,
    config: *const ProceduralConfig,
    after_decay: *const f64,
    after_cut_decay: *const f64,
    after_amplitudes: *const f64,
) -> Option<CommonConfiguration> {
    if glif.is_null()
        || config.is_null()
        || after_decay.is_null()
        || after_cut_decay.is_null()
        || after_amplitudes.is_null()
    {
        return None;
    }
    let glif = *glif;
    let config = *config;
    let decay = [*after_decay, *after_decay.add(1)];
    let cut = [*after_cut_decay, *after_cut_decay.add(1)];
    let amplitudes = [*after_amplitudes, *after_amplitudes.add(1)];
    if !valid_common(&glif, &config)
        || decay
            .iter()
            .chain(cut.iter())
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 1.0)
        || amplitudes.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    Some((glif, config, decay, cut, amplitudes))
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_procedural_runtime_create(
    glif: *const GlifConfig,
    config: *const ProceduralConfig,
    after_decay: *const f64,
    after_cut_decay: *const f64,
    after_amplitudes: *const f64,
) -> *mut ProceduralRuntime {
    let Some((glif, config, decay, cut, amplitudes)) =
        read_common(glif, config, after_decay, after_cut_decay, after_amplitudes)
    else {
        return ptr::null_mut();
    };
    let Ok(runtime) = ProceduralRuntime::new(glif, config, decay, cut, amplitudes) else {
        return ptr::null_mut();
    };
    allocate_owned(runtime)
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_procedural_runtime_destroy(runtime: *mut ProceduralRuntime) {
    if !runtime.is_null() {
        drop(Box::from_raw(runtime));
    }
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_procedural_runtime_advance(
    runtime: *mut ProceduralRuntime,
    step_count: u64,
    events: *const ProceduralEventRecord,
    event_count: usize,
) -> u32 {
    if runtime.is_null() || (event_count != 0 && events.is_null()) {
        return PROCEDURAL_INVALID;
    }
    let events = if event_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(events, event_count)
    };
    (*runtime).advance(step_count, events)
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_procedural_runtime_counts(
    runtime: *const ProceduralRuntime,
    resident_count: *mut usize,
    pending_count: *mut usize,
    spike_count: *mut usize,
) -> u32 {
    if runtime.is_null()
        || resident_count.is_null()
        || pending_count.is_null()
        || spike_count.is_null()
    {
        return PROCEDURAL_INVALID;
    }
    *resident_count = (*runtime).total_residents();
    *pending_count = (*runtime).pending.len() - (*runtime).pending_head;
    *spike_count = (*runtime).last_spikes.len();
    PROCEDURAL_OK
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_procedural_runtime_snapshot(
    runtime: *const ProceduralRuntime,
    residents: *mut ProceduralResidentRecord,
    resident_capacity: usize,
    pending: *mut ProceduralEventRecord,
    pending_capacity: usize,
    spikes: *mut ProceduralSpikeRecord,
    spike_capacity: usize,
    counters: *mut ProceduralCounters,
) -> u32 {
    if runtime.is_null() || counters.is_null() {
        return PROCEDURAL_INVALID;
    }
    let runtime = &*runtime;
    if (residents.is_null() && resident_capacity != 0)
        || (!residents.is_null() && resident_capacity < runtime.total_residents())
        || (pending.is_null() && pending_capacity != 0)
        || (!pending.is_null() && pending_capacity < runtime.pending_logical_len())
        || (spikes.is_null() && spike_capacity != 0)
        || (!spikes.is_null() && spike_capacity < runtime.last_spikes.len())
    {
        return PROCEDURAL_SNAPSHOT_CAPACITY;
    }
    if !residents.is_null() {
        // Residents are reported in ascending GID order; staged cells join
        // the stream via merge so a mid-step snapshot matches the sorted
        // layout the previous single-array implementation produced.
        //
        // Hardening (Seam B.6.5): the merge assumes `staged` is ascending.
        // Non-empty staged is reachable here only mid-delivery or on a
        // poisoned mid-abort; the assert documents that assumption so a
        // future lifecycle mistake fails loudly instead of silently
        // misordering output.
        debug_assert!(
            runtime.staged.is_empty()
                || runtime
                    .staged
                    .windows(2)
                    .all(|window| window[0].gid <= window[1].gid),
            "snapshot merge requires ascending staged"
        );
        let mut main_index = 0usize;
        let mut staged_index = 0usize;
        let mut out_index = 0usize;
        while main_index < runtime.residents.len() || staged_index < runtime.staged.len() {
            let take_staged = main_index == runtime.residents.len()
                || (staged_index < runtime.staged.len()
                    && runtime.staged[staged_index].gid < runtime.residents[main_index].gid);
            let state = if take_staged {
                let state = &runtime.staged[staged_index];
                staged_index += 1;
                state
            } else {
                let state = &runtime.residents[main_index];
                main_index += 1;
                state
            };
            *residents.add(out_index) = state.record();
            out_index += 1;
        }
    }
    if !pending.is_null() {
        ptr::copy_nonoverlapping(
            runtime.pending.as_ptr().add(runtime.pending_head),
            pending,
            runtime.pending_logical_len(),
        );
    }
    if !spikes.is_null() {
        ptr::copy_nonoverlapping(
            runtime.last_spikes.as_ptr(),
            spikes,
            runtime.last_spikes.len(),
        );
    }
    let mut snapshot = runtime.counters;
    snapshot.current_step = runtime.current_step;
    snapshot.native_payload_bytes = runtime.payload_bytes();
    snapshot.retained_nonzero_cells = runtime.total_residents();
    snapshot.overflow_code = runtime.poisoned;
    *counters = snapshot;
    PROCEDURAL_OK
}

/// Targeted resident-state read (active residency only).
///
/// For each `gids[i]`: binary-search `residents` (ascending-GID array) via
/// `resident_index`, then linear-binary-search `staged`; staged wins if
/// present (unreachable under the disjointness invariant D1, kept defensive).
/// Writes exactly `gid_count` [`ProceduralResidentRecord`] values to `out` in
/// input request order and one presence flag per request to `found`. Absent
/// cells get the canonical-rest record for the requested gid. O(K log R).
/// Allocates nothing. Mutates nothing. Touches no counters.
///
/// Pinned semantics (cpu-seams-spec.md Seam A):
/// - Output order is request order; duplicate gids each get their own search.
/// - Absent cells (`found[i] == 0`) carry canonical positive zeros built by
///   `canonical_rest(gid).record()`, preserving signed-zero canon at the seam.
/// - Dormant cells (event-proportional fast-rest retirement) are NOT active
///   residents and report `found = 0` + canonical zeros, matching what
///   `snapshot()` reports in that mode; hosts needing dormant eligibility must
///   use `humanbrain_procedural_runtime_dormant_snapshot` and replay the
///   catch-up multiplies themselves. This divergence from "any configured gid"
///   is deliberate: both read seams must agree on what "is resident" means or
///   host mirrors will silently disagree with checkpoints.
/// - Unlike `snapshot()` (which serves diagnostics of a poisoned runtime via
///   `counters.overflow_code`), this seam FAILS CLOSED on poison: it returns
///   `PROCEDURAL_POISONED` without touching `out`/`found`. It feeds learning
///   (eligibility mirrors driving weight updates); serving pre-poison state
///   would corrupt training with no signal. The asymmetry with snapshot() is
///   intentional; checkpoint/diagnostic reads keep using `snapshot()`.
/// - No GID range validation: a requested gid >= configured_cells is simply
///   absent (`found = 0`, canonical zeros, status OK). Range rejection stays a
///   caller-side courtesy.
///
/// Safety: callers must pass a valid `runtime` handle and, when
/// `gid_count > 0`, readable `gids` plus writable `out`/`found` buffers of at
/// least `gid_count` elements; `out`/`found` must not alias runtime-owned
/// memory (unreachable from Python, where both are freshly allocated ctypes
/// arrays). Additive export; `humanbrain_kernel_abi_version` is unchanged,
/// same rationale as the `dormant_snapshot` precedent.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_procedural_runtime_read_cells(
    runtime: *const ProceduralRuntime,
    gids: *const u64,
    gid_count: usize,
    out: *mut ProceduralResidentRecord,
    found: *mut u8,
) -> u32 {
    if runtime.is_null() || (gid_count > 0 && (gids.is_null() || out.is_null() || found.is_null()))
    {
        return PROCEDURAL_INVALID;
    }
    let runtime = &*runtime;
    if runtime.poisoned != PROCEDURAL_OK {
        return PROCEDURAL_POISONED;
    }
    if gid_count == 0 {
        return PROCEDURAL_OK;
    }
    for index in 0..gid_count {
        let gid = *gids.add(index);
        let slot = out.add(index);
        match runtime.read_cell(gid) {
            Some(state) => {
                *slot = state.record();
                *found.add(index) = 1;
            }
            None => {
                *slot = CellState::canonical_rest(gid).record();
                *found.add(index) = 0;
            }
        }
    }
    PROCEDURAL_OK
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProceduralDormantRecord {
    pub gid: u64,
    /// Stored (pre-catch-up) eligibility as of `elig_step`.
    pub eligibility_raw: f64,
    /// Step whose fused pass last decayed the stored value.
    pub elig_step: u64,
}

/// Event-proportional mode introspection: dump the dormant store in ascending
/// GID order. Values are raw lazily-stored numbers, NOT caught-up ones; a
/// reader replays `current_step - elig_step - 1` eligibility-decay multiplies
/// to obtain the equivalent full-touch value bit-exactly. Empty in full-touch
/// mode. Additive export; the ABI version is unchanged for this spike.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_procedural_runtime_dormant_snapshot(
    runtime: *const ProceduralRuntime,
    out: *mut ProceduralDormantRecord,
    out_capacity: usize,
    out_len: *mut usize,
) -> u32 {
    if runtime.is_null() || out_len.is_null() || (out.is_null() && out_capacity != 0) {
        return PROCEDURAL_INVALID;
    }
    let runtime = &*runtime;
    *out_len = runtime.dormant.len();
    if out.is_null() {
        // Pure size query: report the length without demanding capacity.
        return PROCEDURAL_OK;
    }
    if out_capacity < runtime.dormant.len() {
        return PROCEDURAL_SNAPSHOT_CAPACITY;
    }
    for (index, (gid, entry)) in runtime.dormant.iter().enumerate() {
        *out.add(index) = ProceduralDormantRecord {
            gid: *gid,
            eligibility_raw: entry.eligibility,
            elig_step: entry.elig_step,
        };
    }
    PROCEDURAL_OK
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_dense_reference_create(
    glif: *const GlifConfig,
    config: *const ProceduralConfig,
    after_decay: *const f64,
    after_cut_decay: *const f64,
    after_amplitudes: *const f64,
    targets: *const u64,
    weights: *const f64,
) -> *mut DenseReference {
    let Some((glif, config, decay, cut, amplitudes)) =
        read_common(glif, config, after_decay, after_cut_decay, after_amplitudes)
    else {
        return ptr::null_mut();
    };
    if targets.is_null() || weights.is_null() || config.configured_cells > usize::MAX as u64 {
        return ptr::null_mut();
    }
    let cells = config.configured_cells as usize;
    let Some(edge_count) = cells.checked_mul(config.fanout as usize) else {
        return ptr::null_mut();
    };
    let target_slice = slice::from_raw_parts(targets, edge_count);
    let weight_slice = slice::from_raw_parts(weights, edge_count);
    if target_slice
        .iter()
        .any(|target| *target >= config.configured_cells)
        || weight_slice.iter().any(|weight| !weight.is_finite())
    {
        return ptr::null_mut();
    }
    let Ok(mut owned_targets) = reserved_vec(edge_count) else {
        return ptr::null_mut();
    };
    owned_targets.extend_from_slice(target_slice);
    let Ok(mut owned_weights) = reserved_vec(edge_count) else {
        return ptr::null_mut();
    };
    owned_weights.extend_from_slice(weight_slice);
    let Ok(mut dense_cells) = reserved_vec(cells) else {
        return ptr::null_mut();
    };
    for gid in 0..config.configured_cells {
        if push_bounded(
            &mut dense_cells,
            CellState::canonical_rest(gid),
            cells,
            PROCEDURAL_ALLOCATION,
        )
        .is_err()
        {
            return ptr::null_mut();
        }
    }
    let Ok(pending) = reserved_vec(config.event_capacity) else {
        return ptr::null_mut();
    };
    let Ok(last_spikes) = reserved_vec(config.spike_capacity) else {
        return ptr::null_mut();
    };
    let Ok(scratch) = DenseScratch::try_new(&config) else {
        return ptr::null_mut();
    };
    allocate_owned(DenseReference {
        glif,
        config,
        after_decay: decay,
        after_cut_decay: cut,
        after_amplitudes: amplitudes,
        targets: owned_targets,
        weights: owned_weights,
        current_step: 0,
        cells: dense_cells,
        pending,
        last_spikes,
        scratch,
        poisoned: PROCEDURAL_OK,
    })
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_dense_reference_destroy(runtime: *mut DenseReference) {
    if !runtime.is_null() {
        drop(Box::from_raw(runtime));
    }
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_dense_reference_advance(
    runtime: *mut DenseReference,
    step_count: u64,
    events: *const ProceduralEventRecord,
    event_count: usize,
) -> u32 {
    if runtime.is_null() || (event_count != 0 && events.is_null()) {
        return PROCEDURAL_INVALID;
    }
    let events = if event_count == 0 {
        &[]
    } else {
        slice::from_raw_parts(events, event_count)
    };
    (*runtime).advance(step_count, events)
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_dense_reference_counts(
    runtime: *const DenseReference,
    cell_count: *mut usize,
    pending_count: *mut usize,
    spike_count: *mut usize,
) -> u32 {
    if runtime.is_null() || cell_count.is_null() || pending_count.is_null() || spike_count.is_null()
    {
        return PROCEDURAL_INVALID;
    }
    *cell_count = (*runtime).cells.len();
    *pending_count = (*runtime).pending.len();
    *spike_count = (*runtime).last_spikes.len();
    PROCEDURAL_OK
}

#[no_mangle]
pub unsafe extern "C" fn humanbrain_dense_reference_snapshot(
    runtime: *const DenseReference,
    cells: *mut ProceduralResidentRecord,
    cell_capacity: usize,
    pending: *mut ProceduralEventRecord,
    pending_capacity: usize,
    spikes: *mut ProceduralSpikeRecord,
    spike_capacity: usize,
    current_step: *mut u64,
    poison_code: *mut u32,
) -> u32 {
    if runtime.is_null() || current_step.is_null() || poison_code.is_null() {
        return PROCEDURAL_INVALID;
    }
    let runtime = &*runtime;
    if (cells.is_null() && cell_capacity != 0)
        || (!cells.is_null() && cell_capacity < runtime.cells.len())
        || (pending.is_null() && pending_capacity != 0)
        || (!pending.is_null() && pending_capacity < runtime.pending.len())
        || (spikes.is_null() && spike_capacity != 0)
        || (!spikes.is_null() && spike_capacity < runtime.last_spikes.len())
    {
        return PROCEDURAL_SNAPSHOT_CAPACITY;
    }
    if !cells.is_null() {
        for (index, state) in runtime.cells.iter().enumerate() {
            *cells.add(index) = state.record();
        }
    }
    if !pending.is_null() {
        ptr::copy_nonoverlapping(runtime.pending.as_ptr(), pending, runtime.pending.len());
    }
    if !spikes.is_null() {
        ptr::copy_nonoverlapping(
            runtime.last_spikes.as_ptr(),
            spikes,
            runtime.last_spikes.len(),
        );
    }
    *current_step = runtime.current_step;
    *poison_code = runtime.poisoned;
    PROCEDURAL_OK
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_zero, fallback_candidate, fill_targets_with,
        humanbrain_procedural_runtime_create, humanbrain_procedural_runtime_read_cells,
        humanbrain_procedural_runtime_snapshot, CellState, DormantEntry, GlifConfig,
        ProceduralConfig, ProceduralCounters, ProceduralEventRecord, ProceduralResidentRecord,
        ProceduralRuntime, PROCEDURAL_STEPPING_EVENT_PROPORTIONAL, PROCEDURAL_STEPPING_FULL_TOUCH,
    };
    use std::collections::{BTreeMap, HashMap, HashSet};

    #[test]
    fn fallback_candidate_matches_widened_counter64_oracle_across_u64_wrap() {
        assert_eq!(fallback_candidate(u64::MAX - 1, 3, u64::MAX), 2);
    }

    /// Raw little-endian bytes of the FFI counters image, exactly what the
    /// snapshot seam copies out to callers.
    fn counters_raw_bytes(counters: &ProceduralCounters) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                counters as *const ProceduralCounters as *const u8,
                std::mem::size_of::<ProceduralCounters>(),
            )
        }
    }

    /// Raw bytes of one resident FFI image. The layout test below proves that
    /// all 72 bytes belong to declared fields before any test reads them.
    fn resident_raw_bytes(record: &ProceduralResidentRecord) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                record as *const ProceduralResidentRecord as *const u8,
                std::mem::size_of::<ProceduralResidentRecord>(),
            )
        }
    }

    /// Regression guard: the Python side compares and hashes the RAW bytes of
    /// ``ProceduralCounters`` across independent runs, so all 112 bytes must
    /// be defined. The 4-byte tail used to be implicit (unspecified) padding
    /// carrying stack garbage from the construction site, which made those
    /// comparisons depend on allocator history -- surfacing as an
    /// order-dependent failure of the generated-recurrent-key collision
    /// fail-stop test. The padding is now the declared always-zero
    /// ``padding_zero`` field; pin the layout and the zero-padding invariant
    /// on both construction paths so neither can regress silently.
    #[test]
    fn ffi_counter_bytes_are_fully_defined_and_padding_stays_zero() {
        assert_eq!(std::mem::size_of::<ProceduralCounters>(), 112);
        assert_eq!(std::mem::offset_of!(ProceduralCounters, overflow_code), 104);
        assert_eq!(std::mem::offset_of!(ProceduralCounters, padding_zero), 108);

        // Construction site 1: ::default() at runtime creation (the path a
        // poisoned-first-advance runtime reports through the snapshot seam).
        let default_counters = ProceduralCounters::default();
        let raw_default = counters_raw_bytes(&default_counters);
        assert!(raw_default.iter().all(|&byte| byte == 0));

        // Construction site 2: the per-interval literal inside advance().
        let mut runtime = ep_runtime(16, PROCEDURAL_STEPPING_EVENT_PROPORTIONAL);
        let status = runtime.advance(1, &[drive(0, 0, 1e-30)]);
        assert_eq!(status, super::PROCEDURAL_OK);
        let raw_advanced = counters_raw_bytes(&runtime.counters);
        // Declared fields may be nonzero; the declared padding field must not.
        assert_eq!(&raw_advanced[108..112], &[0u8, 0, 0, 0]);
    }

    #[test]
    fn ffi_resident_layout_is_fully_declared_and_padding_stays_zero() {
        assert_eq!(std::mem::size_of::<ProceduralResidentRecord>(), 72);
        assert_eq!(
            std::mem::offset_of!(ProceduralResidentRecord, refractory_steps),
            48
        );
        assert_eq!(
            std::mem::offset_of!(ProceduralResidentRecord, padding_zero),
            52
        );
        assert_eq!(
            std::mem::offset_of!(ProceduralResidentRecord, synaptic_current_pa),
            56
        );
        assert_eq!(
            std::mem::offset_of!(ProceduralResidentRecord, eligibility),
            64
        );

        let record = CellState::canonical_rest(7).record();
        assert_eq!(record.padding_zero, 0);
        assert_eq!(&resident_raw_bytes(&record)[52..56], &[0u8, 0, 0, 0]);
    }

    #[test]
    fn collision_heavy_row_matches_oracle_order_and_linear_work_bound() {
        let config = ProceduralConfig {
            configured_cells: 66,
            fanout: 64,
            seed: 71,
            resident_capacity: 8,
            event_capacity: 128,
            spike_capacity: 128,
            interval_distinct_capacity: 136,
            recurrent_weight_pa: 325.0,
            recurrent_delay_steps: 1,
            recurrent_decay: 0.5,
            eligibility_decay: 0.5,
            stepping_mode: PROCEDURAL_STEPPING_FULL_TOUCH,
        };
        let mut targets = Vec::new();
        targets.try_reserve_exact(64).unwrap();
        let mut used = HashSet::new();
        used.try_reserve(65).unwrap();

        let work = fill_targets_with(&config, 0, &mut targets, &mut used, |_, source, _, _| {
            source
        })
        .unwrap();

        assert_eq!(
            targets,
            [
                57, 58, 59, 60, 61, 62, 63, 64, 65, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            ]
        );
        assert_eq!(work, 8_257);
        assert!(work <= 129 * config.fanout + 1);
    }

    #[test]
    fn impossible_preallocation_fails_at_the_raw_creation_seam_without_abort() {
        let glif = GlifConfig {
            cell_count: 1,
            after_channel_count: 2,
            spike_cut_steps: 1,
            leak_ns: 1.0,
            voltage_step_scale: 1.0,
            threshold_beta_scale: 1.0,
            threshold_exp_decay: 0.5,
            threshold_voltage_factor: 1.0,
            threshold_beta_factor: 1.0,
            spike_threshold_decay: 0.5,
            threshold_inf_relative_mv: 1.0,
            voltage_reset_scale: 1.0,
            voltage_reset_offset_mv: 1.0,
            spike_cut_threshold_decay: 0.5,
            threshold_spike_increment_mv: 1.0,
        };
        let config = ProceduralConfig {
            configured_cells: u64::MAX,
            fanout: 1,
            seed: 71,
            resident_capacity: usize::MAX,
            event_capacity: 1,
            spike_capacity: 1,
            interval_distinct_capacity: usize::MAX,
            recurrent_weight_pa: 1.0,
            recurrent_delay_steps: 1,
            recurrent_decay: 0.5,
            eligibility_decay: 0.5,
            stepping_mode: PROCEDURAL_STEPPING_FULL_TOUCH,
        };
        let decay = [0.5, 0.5];
        let amplitudes = [0.0, 0.0];

        let runtime = unsafe {
            humanbrain_procedural_runtime_create(
                &glif,
                &config,
                decay.as_ptr(),
                decay.as_ptr(),
                amplitudes.as_ptr(),
            )
        };

        assert!(runtime.is_null());
    }

    // ---------------------------------------------------------------------
    // Event-proportional stepping parity fixtures.
    //
    // The GLIF constants below make post-spike cells go exactly quiet: the
    // voltage reset is zero, after-spike amplitudes are zero, and thresholds
    // decay geometrically to exact zeros. A spiked cell therefore decays to
    // fast-rest with a pure eligibility tail -- the dormant case -- while
    // epsilon keep-alive currents hold cells actively integrating when the
    // fixture wants the two modes to touch every cell every step.
    // ---------------------------------------------------------------------

    fn ep_glif() -> GlifConfig {
        GlifConfig {
            cell_count: 1,
            after_channel_count: 2,
            spike_cut_steps: 3,
            // Unit leak makes v' = I exactly: voltage mirrors the synaptic
            // current and reaches canonical zero in the same step it does,
            // so a quieted cell's fast-rest is reachable inside short test
            // horizons (a mild leak leaves sub-normal voltage tails alive
            // for ~1000 steps and dormancy never engages).
            leak_ns: 1.0,
            voltage_step_scale: 1.0,
            threshold_beta_scale: 0.0,
            threshold_exp_decay: 0.9,
            threshold_voltage_factor: 0.0,
            threshold_beta_factor: 0.0,
            spike_threshold_decay: 0.9,
            threshold_inf_relative_mv: 1.0,
            voltage_reset_scale: 0.0,
            voltage_reset_offset_mv: 0.0,
            spike_cut_threshold_decay: 0.5,
            // Zero increment keeps the spike-threshold term exactly zero even
            // after spikes, so a quieted cell's only surviving state is the
            // eligibility tail -- the dormant case under test.
            threshold_spike_increment_mv: 0.0,
        }
    }

    fn ep_config(configured_cells: u64, mode: u32) -> ProceduralConfig {
        ProceduralConfig {
            configured_cells,
            fanout: 4,
            seed: 71,
            resident_capacity: 256,
            event_capacity: 8192,
            spike_capacity: 8192,
            interval_distinct_capacity: 1024,
            // Subthreshold edge weight: one delivered event cannot cross the
            // 1.0 threshold floor, so a spike wave dies after one generation
            // instead of cascading forever (which would keep every cell
            // active and make dormancy unreachable).
            recurrent_weight_pa: 0.25,
            recurrent_delay_steps: 1,
            // Aggressive synaptic decay flushes delivered currents to exact
            // zero within ~40 steps, so spiked cells actually reach fast-rest
            // inside the test horizon (a mild decay like 0.9 would keep
            // sub-normal synaptic tails alive for thousands of steps and no
            // cell would ever dormanize).
            recurrent_decay: 1e-8,
            eligibility_decay: 0.5,
            stepping_mode: mode,
        }
    }

    fn ep_runtime(configured_cells: u64, mode: u32) -> ProceduralRuntime {
        let decay = [0.5, 0.5];
        ProceduralRuntime::new(
            ep_glif(),
            ep_config(configured_cells, mode),
            decay,
            decay,
            [0.0; 2],
        )
        .unwrap()
    }

    fn drive(target: u64, due_step: u64, current_pa: f64) -> ProceduralEventRecord {
        ProceduralEventRecord {
            due_step,
            target_gid: target,
            source_gid: target,
            edge_slot: 0,
            current_pa: current_pa,
        }
    }

    /// Advance one step at a time so every batch is unambiguous, collecting
    /// the spikes each call produced.
    fn run_steps(
        runtime: &mut ProceduralRuntime,
        batches: &[Vec<ProceduralEventRecord>],
    ) -> Vec<(u64, u64)> {
        let mut trace = Vec::new();
        for batch in batches {
            let status = runtime.advance(1, batch);
            assert_eq!(status, super::PROCEDURAL_OK);
            for spike in &runtime.last_spikes {
                trace.push((spike.step, spike.gid));
            }
        }
        trace
    }

    fn resident_bits(runtime: &ProceduralRuntime) -> Vec<(u64, [u64; 7])> {
        let mut records: Vec<_> = runtime
            .residents
            .iter()
            .map(|state| {
                let record = state.record();
                (
                    record.gid,
                    [
                        record.voltage_relative_mv.to_bits(),
                        record.after_spike_current_0_pa.to_bits(),
                        record.after_spike_current_1_pa.to_bits(),
                        record.threshold_spike_mv.to_bits(),
                        record.threshold_voltage_mv.to_bits(),
                        record.synaptic_current_pa.to_bits(),
                        record.eligibility.to_bits(),
                    ],
                )
            })
            .collect();
        records.sort_unstable_by_key(|entry| entry.0);
        records
    }

    /// Replay lazy eligibility decay exactly as the kernel's catch-up does:
    /// one canonical multiply per outstanding fused pass.
    fn replay_eligibility(raw: f64, elig_step: u64, current_step: u64) -> f64 {
        let mut value = raw;
        let gap = current_step.saturating_sub(elig_step.saturating_add(1));
        for _ in 0..gap {
            value = canonical_zero(value * 0.5);
        }
        value
    }

    /// All-active fixture: epsilon keep-alive currents plus periodic strong
    /// pulses keep every touched cell integrating every step, so the
    /// event-proportional mode must never dormanize anything and both modes
    /// must agree bit for bit on spikes, counters, and resident state.
    #[test]
    fn event_proportional_matches_full_touch_bit_exactly_while_all_cells_stay_active() {
        let cells = 48u64;
        let steps = 400usize;
        let build_batches = || {
            let mut batches = Vec::with_capacity(steps);
            for step in 0..steps {
                let mut batch: Vec<ProceduralEventRecord> = Vec::new();
                for gid in 0..cells {
                    batch.push(drive(gid, step as u64, 1e-30));
                }
                if step % 37 == 11 {
                    for gid in 0..cells {
                        if (gid * 7 + step as u64) % 13 == 0 {
                            batch.push(ProceduralEventRecord {
                                due_step: step as u64,
                                target_gid: gid,
                                source_gid: (gid + 1) % cells,
                                edge_slot: 1,
                                current_pa: 10.0,
                            });
                        }
                    }
                }
                batches.push(batch);
            }
            batches
        };

        let mut full_touch = ep_runtime(cells, PROCEDURAL_STEPPING_FULL_TOUCH);
        let full_touch_trace = run_steps(&mut full_touch, &build_batches());

        let mut event_proportional = ep_runtime(cells, PROCEDURAL_STEPPING_EVENT_PROPORTIONAL);
        let event_trace = run_steps(&mut event_proportional, &build_batches());

        assert!(
            !full_touch_trace.is_empty(),
            "fixture must actually spike cells"
        );
        assert_eq!(full_touch_trace, event_trace, "spike traces must match");
        assert!(
            event_proportional.dormant.is_empty(),
            "all-active fixture must never dormanize"
        );

        let ft_counters = full_touch.counters;
        let ep_counters = event_proportional.counters;
        assert_eq!(
            ft_counters.cell_step_evaluations,
            ep_counters.cell_step_evaluations
        );
        assert_eq!(ft_counters.delivered_events, ep_counters.delivered_events);
        assert_eq!(ft_counters.generated_edges, ep_counters.generated_edges);
        assert_eq!(ft_counters.spike_events, ep_counters.spike_events);
        assert_eq!(ft_counters.resident_peak, ep_counters.resident_peak);
        assert_eq!(ft_counters.resident_distinct, ep_counters.resident_distinct);
        assert_eq!(ft_counters.updated_distinct, ep_counters.updated_distinct);
        assert_eq!(
            ft_counters.retained_nonzero_cells,
            ep_counters.retained_nonzero_cells
        );

        assert_eq!(
            resident_bits(&full_touch),
            resident_bits(&event_proportional)
        );
    }

    /// Dormant cells store their eligibility raw and replay the identical
    /// multiply sequence on catch-up: the replayed value must equal the
    /// full-touch path's stored eligibility bit for bit.
    #[test]
    fn event_proportional_lazy_decay_replays_full_touch_multiplies_exactly() {
        let cells = 16u64;
        let steps = 300usize;
        let mut batches = vec![Vec::<ProceduralEventRecord>::new(); steps];
        for gid in [3u64, 7, 11] {
            batches[0].push(drive(gid, 0, 10.0));
        }

        let mut full_touch = ep_runtime(cells, PROCEDURAL_STEPPING_FULL_TOUCH);
        let ft_trace = run_steps(&mut full_touch, &batches);

        let mut event_proportional = ep_runtime(cells, PROCEDURAL_STEPPING_EVENT_PROPORTIONAL);
        let ep_trace = run_steps(&mut event_proportional, &batches);

        assert_eq!(ft_trace, ep_trace, "spike traces must match");
        assert!(
            !event_proportional.dormant.is_empty(),
            "spiked cells must dormanize"
        );

        let current_step = full_touch.current_step;
        assert_eq!(current_step, event_proportional.current_step);
        for (gid, entry) in &event_proportional.dormant {
            let expected = replay_eligibility(entry.eligibility, entry.elig_step, current_step);
            let actual = full_touch
                .residents
                .iter()
                .find(|state| state.gid == *gid)
                .map(|state| state.eligibility)
                .unwrap_or_else(|| {
                    // The full-touch path evicted this cell only if its
                    // eligibility underflowed to exact zero.
                    0.0
                });
            assert_eq!(expected.to_bits(), actual.to_bits(), "gid {gid}");
        }
        // Every spiking cell must be accounted for between the two stores.
        for (_, gid) in &ft_trace {
            let still_active_ft = full_touch.residents.iter().any(|state| state.gid == *gid);
            let dormant_ep = event_proportional.dormant.contains_key(gid);
            let active_ep = event_proportional
                .residents
                .iter()
                .any(|state| state.gid == *gid);
            assert!(still_active_ft == (dormant_ep || active_ep), "gid {gid}");
        }
    }

    /// Mixed traffic: dormancy, reactivation from the dormant store, fresh
    /// first-spikes, and long quiet stretches. Dynamics must stay bit-identical;
    /// only the documented work/residency accounting may diverge.
    #[test]
    fn event_proportional_mixed_traffic_matches_full_touch_dynamics_and_state() {
        let cells = 32u64;
        let steps = 600usize;
        let mut batches = vec![Vec::<ProceduralEventRecord>::new(); steps];
        for gid in 0..9u64 {
            batches[0].push(drive(gid, 0, 10.0));
        }
        for gid in [2u64, 5] {
            batches[150].push(drive(gid, 150, 10.0));
        }
        for gid in 20..25u64 {
            batches[150].push(drive(gid, 150, 10.0));
        }
        for gid in [0u64, 3, 20] {
            batches[400].push(drive(gid, 400, 10.0));
        }

        let mut full_touch = ep_runtime(cells, PROCEDURAL_STEPPING_FULL_TOUCH);
        let ft_trace = run_steps(&mut full_touch, &batches);
        let mut event_proportional = ep_runtime(cells, PROCEDURAL_STEPPING_EVENT_PROPORTIONAL);
        let ep_trace = run_steps(&mut event_proportional, &batches);

        assert!(!ft_trace.is_empty());
        assert_eq!(
            ft_trace, ep_trace,
            "spike traces must match on mixed traffic"
        );

        let ft = full_touch.counters;
        let ep = event_proportional.counters;
        assert_eq!(ft.delivered_events, ep.delivered_events);
        assert_eq!(ft.generated_edges, ep.generated_edges);
        assert_eq!(ft.spike_events, ep.spike_events);
        assert_eq!(ft.spiking_distinct, ep.spiking_distinct);
        assert_eq!(ft.retained_nonzero_cells, ep.retained_nonzero_cells);
        assert_eq!(ft.resident_peak, ep.resident_peak);
        assert_eq!(ft.current_step, ep.current_step);
        // Documented divergence: EP does not "update" dormant cells.
        assert!(ep.updated_distinct <= ft.updated_distinct);
        assert!(ep.cell_step_evaluations < ft.cell_step_evaluations);

        let current_step = full_touch.current_step;
        for (gid, bits) in resident_bits(&event_proportional) {
            let ft_state = full_touch
                .residents
                .iter()
                .find(|state| state.gid == gid)
                .expect("actively-stepping EP cell must be active in full-touch too");
            let record = ft_state.record();
            let ft_bits = [
                record.voltage_relative_mv.to_bits(),
                record.after_spike_current_0_pa.to_bits(),
                record.after_spike_current_1_pa.to_bits(),
                record.threshold_spike_mv.to_bits(),
                record.threshold_voltage_mv.to_bits(),
                record.synaptic_current_pa.to_bits(),
                record.eligibility.to_bits(),
            ];
            assert_eq!(bits, ft_bits, "active-state mismatch at gid {gid}");
        }
        for (gid, entry) in &event_proportional.dormant {
            let replayed = replay_eligibility(entry.eligibility, entry.elig_step, current_step);
            match full_touch.residents.iter().find(|state| state.gid == *gid) {
                Some(state) => {
                    assert_eq!(replayed.to_bits(), state.eligibility.to_bits(), "gid {gid}");
                    assert!(state.is_fast_rest(), "divergent cell must be an elig tail");
                }
                None => assert_eq!(
                    replayed.to_bits(),
                    0.0f64.to_bits(),
                    "evicted cell must have underflowed to exact zero"
                ),
            }
        }
    }

    #[test]
    fn dormant_gid_run_reuse_preserves_dynamics_and_reclaims_capacity() {
        const CELLS: u64 = 32;
        const STEPS: usize = 120;
        let mut batches = vec![Vec::<ProceduralEventRecord>::new(); STEPS];
        for gid in 0..8u64 {
            batches[0].push(drive(gid, 0, 10.0));
        }
        for gid in [1u64, 3, 20] {
            batches[60].push(drive(gid, 60, 10.0));
        }

        let mut full_touch = ep_runtime(CELLS, PROCEDURAL_STEPPING_FULL_TOUCH);
        let full_trace = run_steps(&mut full_touch, &batches);
        let mut event_proportional = ep_runtime(CELLS, PROCEDURAL_STEPPING_EVENT_PROPORTIONAL);
        let event_trace = run_steps(&mut event_proportional, &batches);
        assert!(!event_proportional.dormant.is_empty());
        assert_eq!(full_trace, event_trace);
        assert_eq!(
            full_touch.counters.delivered_events,
            event_proportional.counters.delivered_events
        );
        assert_eq!(
            full_touch.counters.generated_edges,
            event_proportional.counters.generated_edges
        );
        assert_eq!(
            full_touch.counters.spike_events,
            event_proportional.counters.spike_events
        );

        let current_step = full_touch.current_step;
        let full_bits: BTreeMap<_, _> = resident_bits(&full_touch).into_iter().collect();
        for (gid, bits) in resident_bits(&event_proportional) {
            assert_eq!(bits, full_bits[&gid]);
        }
        for (gid, entry) in &event_proportional.dormant {
            let expected = replay_eligibility(entry.eligibility, entry.elig_step, current_step);
            let actual = full_touch
                .residents
                .iter()
                .find(|state| state.gid == *gid)
                .map_or(0.0, |state| state.eligibility);
            assert_eq!(expected.to_bits(), actual.to_bits(), "gid {gid}");
        }

        const CAPACITY: usize = 65_536;
        let mut config = ep_config(
            (CAPACITY + 1) as u64,
            PROCEDURAL_STEPPING_EVENT_PROPORTIONAL,
        );
        config.resident_capacity = CAPACITY;
        config.event_capacity = 64;
        config.spike_capacity = 64;
        config.interval_distinct_capacity = CAPACITY;
        let decay = [0.5, 0.5];
        let runtime = ProceduralRuntime::new(ep_glif(), config, decay, decay, [0.0; 2]).unwrap();
        assert_eq!(runtime.scratch.gid_run.capacity(), CAPACITY);
        let reclaimed_bytes = runtime.scratch.gid_run.capacity() * std::mem::size_of::<u64>()
            + std::mem::size_of::<Vec<u64>>();
        assert_eq!(
            reclaimed_bytes,
            CAPACITY * std::mem::size_of::<u64>() + std::mem::size_of::<Vec<u64>>()
        );
        eprintln!(
            "dormant-gid-run-reuse current_payload_bytes={} reclaimed_bytes={} reclaimed_at_1m_bytes={}",
            runtime.payload_bytes(),
            reclaimed_bytes,
            1_000_000usize * 8 + std::mem::size_of::<Vec<u64>>()
        );
    }

    #[test]
    fn invalid_stepping_mode_is_rejected_at_creation() {
        let decay = [0.5, 0.5];
        assert!(
            ProceduralRuntime::new(ep_glif(), ep_config(16, 7), decay, decay, [0.0; 2]).is_err()
        );
    }

    // ---------------------------------------------------------------------
    // read_cells targeted FFI (cpu-seams-spec.md Seam A).
    //
    // The C-ABI helper is exercised exactly as an FFI caller sees it: raw
    // buffers, presence flags, and status codes. Staged cells are produced
    // deterministically by direct `apply_event` deliveries between advances
    // (staged is only observable between top-level calls when a fused pass
    // has not yet absorbed it), which is precisely the defensive branch the
    // export must keep correct.
    // ---------------------------------------------------------------------

    /// Dynamics-field bit words of one record. Comparisons use `to_bits`,
    /// never `==`, so `-0.0` can never masquerade as canonical `+0.0`.
    fn record_bits(record: &ProceduralResidentRecord) -> [u64; 8] {
        [
            record.voltage_relative_mv.to_bits(),
            record.after_spike_current_0_pa.to_bits(),
            record.after_spike_current_1_pa.to_bits(),
            record.threshold_spike_mv.to_bits(),
            record.threshold_voltage_mv.to_bits(),
            record.synaptic_current_pa.to_bits(),
            record.eligibility.to_bits(),
            record.refractory_steps as i64 as u64,
        ]
    }

    /// gid -> record map pulled through the real snapshot export, so present
    /// records must byte-match what any existing consumer already reads.
    fn snapshot_records(runtime: &ProceduralRuntime) -> HashMap<u64, ProceduralResidentRecord> {
        let count = runtime.total_residents();
        let template = CellState::canonical_rest(u64::MAX).record();
        let mut buffer = vec![template; count.max(1)];
        let mut counters = ProceduralCounters::default();
        let status = unsafe {
            humanbrain_procedural_runtime_snapshot(
                runtime as *const ProceduralRuntime,
                buffer.as_mut_ptr(),
                buffer.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                0,
                &mut counters,
            )
        };
        assert_eq!(status, super::PROCEDURAL_OK);
        buffer.truncate(count);
        buffer
            .into_iter()
            .map(|record| (record.gid, record))
            .collect()
    }

    fn read_through_ffi(
        runtime: &ProceduralRuntime,
        gids: &[u64],
    ) -> (u32, Vec<ProceduralResidentRecord>, Vec<u8>) {
        let mut out = vec![CellState::canonical_rest(u64::MAX).record(); gids.len().max(1)];
        let mut found = vec![0xFFu8; gids.len().max(1)];
        let status = unsafe {
            humanbrain_procedural_runtime_read_cells(
                runtime as *const ProceduralRuntime,
                gids.as_ptr(),
                gids.len(),
                out.as_mut_ptr(),
                found.as_mut_ptr(),
            )
        };
        out.truncate(gids.len());
        found.truncate(gids.len());
        (status, out, found)
    }

    fn assert_canonical_positive_zeros(record: &ProceduralResidentRecord, gid: u64) {
        assert_eq!(record.gid, gid);
        for field in record_bits(record).iter().take(7) {
            assert_eq!(*field, 0u64, "absent record must carry canonical +0.0");
        }
        assert_eq!(record.refractory_steps, 0);
        assert_eq!(record.padding_zero, 0);
    }

    #[test]
    fn read_cells_matches_snapshot_records_for_present_absent_and_staged() {
        let mut runtime = ep_runtime(48, PROCEDURAL_STEPPING_FULL_TOUCH);
        // Gids 0..8 become absorbed residents: delivered, then folded into
        // the sorted array by the step's fused pass.
        let batch: Vec<_> = (0..8).map(|gid| drive(gid, 0, 10.0)).collect();
        assert_eq!(runtime.advance(1, &batch), super::PROCEDURAL_OK);
        assert_eq!(runtime.staged.len(), 0);
        // Stage gids 20..24 without running a fused pass, leaving them in the
        // staged Vec where the export's defensive staged branch must find
        // them. Ascending pushes keep `staged` sorted as the binary search
        // requires.
        for gid in 20..24u64 {
            runtime
                .apply_event(drive(gid, runtime.current_step, 10.0))
                .unwrap();
        }
        assert_eq!(runtime.staged.len(), 4);

        // Request order mixes residents, staged, absent, out-of-range, and a
        // duplicate; results must come back positionally.
        let gids = [20u64, 40, 3, 22, 1000, 3, 7];
        let (status, out, found) = read_through_ffi(&runtime, &gids);
        assert_eq!(status, super::PROCEDURAL_OK);
        let expected = snapshot_records(&runtime);

        for (index, requested) in gids.iter().enumerate() {
            assert_eq!(out[index].gid, *requested, "gid must echo request order");
            assert_eq!(
                out[index].padding_zero, 0,
                "present, staged, and absent record padding must be canonical"
            );
            match expected.get(requested) {
                Some(snapshot_record) => {
                    assert_eq!(found[index], 1, "gid {requested} must be found");
                    assert_eq!(snapshot_record.padding_zero, 0);
                    assert_eq!(record_bits(&out[index]), record_bits(snapshot_record));
                }
                None => {
                    assert_eq!(found[index], 0, "gid {requested} must be absent");
                    assert_canonical_positive_zeros(&out[index], *requested);
                }
            }
        }

        // Duplicate requests get independent but identical answers.
        assert_eq!(found[2], found[5]);
        assert_eq!(record_bits(&out[2]), record_bits(&out[5]));
        // Out-of-range gids (1000 >= configured 48) are simply absent, not errors.
        assert_canonical_positive_zeros(&out[4], 1000);
    }

    #[test]
    fn read_cells_reports_poisoned_without_touching_outputs() {
        let mut config = ep_config(16, PROCEDURAL_STEPPING_FULL_TOUCH);
        config.resident_capacity = 4;
        let decay = [0.5, 0.5];
        let mut runtime =
            ProceduralRuntime::new(ep_glif(), config, decay, decay, [0.0; 2]).unwrap();
        // Force a resident-capacity fail-stop mid-delivery: six distinct
        // targets into four slots poisons the runtime.
        let batch: Vec<_> = (0..6).map(|gid| drive(gid, 0, 10.0)).collect();
        assert_eq!(
            runtime.advance(1, &batch),
            super::PROCEDURAL_RESIDENT_CAPACITY
        );
        assert_ne!(runtime.poisoned, super::PROCEDURAL_OK);

        let sentinel_voltage = f64::from_bits(0xDEAD_BEEF_CAFE_F00D);
        let sentinels = vec![
            ProceduralResidentRecord {
                gid: u64::MAX,
                voltage_relative_mv: sentinel_voltage,
                after_spike_current_0_pa: sentinel_voltage,
                after_spike_current_1_pa: sentinel_voltage,
                threshold_spike_mv: sentinel_voltage,
                threshold_voltage_mv: sentinel_voltage,
                refractory_steps: -1,
                padding_zero: 0,
                synaptic_current_pa: sentinel_voltage,
                eligibility: sentinel_voltage,
            };
            2
        ];
        let mut out = sentinels.clone();
        let mut found = vec![0xAAu8; 2];

        // Poison check precedes both the short-circuit and every write, and
        // unlike snapshot() this seam refuses to serve pre-poison state.
        let status = unsafe {
            humanbrain_procedural_runtime_read_cells(
                &runtime as *const ProceduralRuntime,
                [0u64, 1].as_ptr(),
                2,
                out.as_mut_ptr(),
                found.as_mut_ptr(),
            )
        };
        assert_eq!(status, super::PROCEDURAL_POISONED);
        for (index, sentinel) in sentinels.iter().enumerate() {
            assert_eq!(
                out[index].gid, sentinel.gid,
                "outputs must be untouched on poison"
            );
            assert_eq!(
                record_bits(&out[index]),
                record_bits(sentinel),
                "outputs must be untouched on poison"
            );
        }
        assert_eq!(found, vec![0xAAu8; 2], "flags must be untouched on poison");
    }

    #[test]
    fn read_cells_handles_empty_null_and_duplicate_requests() {
        let mut runtime = ep_runtime(32, PROCEDURAL_STEPPING_FULL_TOUCH);
        let batch = vec![drive(2, 0, 10.0), drive(9, 0, 10.0)];
        assert_eq!(runtime.advance(1, &batch), super::PROCEDURAL_OK);

        // gid_count == 0 short-circuits OK even with null buffers.
        let empty_status = unsafe {
            humanbrain_procedural_runtime_read_cells(
                &runtime as *const ProceduralRuntime,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(empty_status, super::PROCEDURAL_OK);

        // Nonzero counts demand non-null buffers, mirroring counts().
        let gids = [2u64];
        let mut out = vec![CellState::canonical_rest(0).record(); 1];
        let mut found = [0u8; 1];
        for (gids_override, out_override, found_override) in [
            (std::ptr::null(), out.as_mut_ptr(), found.as_mut_ptr()),
            (gids.as_ptr(), std::ptr::null_mut(), found.as_mut_ptr()),
            (gids.as_ptr(), out.as_mut_ptr(), std::ptr::null_mut()),
        ] {
            let status = unsafe {
                humanbrain_procedural_runtime_read_cells(
                    &runtime as *const ProceduralRuntime,
                    gids_override,
                    1,
                    out_override,
                    found_override,
                )
            };
            assert_eq!(status, super::PROCEDURAL_INVALID);
        }
        let null_runtime_status = unsafe {
            humanbrain_procedural_runtime_read_cells(
                std::ptr::null(),
                gids.as_ptr(),
                1,
                out.as_mut_ptr(),
                found.as_mut_ptr(),
            )
        };
        assert_eq!(null_runtime_status, super::PROCEDURAL_INVALID);

        // Duplicates stay consistent; untouched-but-valid gid 31 and
        // out-of-range gid 500 both report absent without error.
        let repeated = [2u64, 2, 31, 500];
        let (status, out, found) = read_through_ffi(&runtime, &repeated);
        assert_eq!(status, super::PROCEDURAL_OK);
        assert_eq!(found, [1, 1, 0, 0]);
        assert_eq!(record_bits(&out[0]), record_bits(&out[1]));
        let expected = snapshot_records(&runtime);
        assert_eq!(record_bits(&out[0]), record_bits(&expected[&2]));
        assert_canonical_positive_zeros(&out[2], 31);
        assert_canonical_positive_zeros(&out[3], 500);
    }

    #[test]
    fn read_cells_is_bit_identical_across_identically_seeded_runtimes() {
        let build_batches = || {
            let mut batches = Vec::new();
            for step in 0..12usize {
                let batch: Vec<_> = if step % 3 == 0 {
                    (0..6u64)
                        .filter(|gid| (gid + step as u64) % 2 == 0)
                        .map(|gid| drive(gid, step as u64, 10.0))
                        .collect()
                } else {
                    Vec::new()
                };
                batches.push(batch);
            }
            batches
        };

        let mut first = ep_runtime(48, PROCEDURAL_STEPPING_FULL_TOUCH);
        run_steps(&mut first, &build_batches());
        let mut second = ep_runtime(48, PROCEDURAL_STEPPING_FULL_TOUCH);
        run_steps(&mut second, &build_batches());

        let gids = [5u64, 999, 2, 2, 40, 0];
        let (first_status, first_out, first_found) = read_through_ffi(&first, &gids);
        let (second_status, second_out, second_found) = read_through_ffi(&second, &gids);
        assert_eq!(first_status, super::PROCEDURAL_OK);
        assert_eq!(first_status, second_status);
        // Every byte is declared by the pinned layout test, so whole-record
        // comparison cannot observe implicit padding.
        let first_bytes: Vec<_> = first_out
            .iter()
            .flat_map(|record| resident_raw_bytes(record).iter().copied())
            .collect();
        let second_bytes: Vec<_> = second_out
            .iter()
            .flat_map(|record| resident_raw_bytes(record).iter().copied())
            .collect();
        assert_eq!(first_bytes, second_bytes);
        assert_eq!(first_found, second_found);
        // Sanity: the fixture must actually mix hits and absences.
        assert!(first_found.contains(&1));
        assert!(first_found.contains(&0));
    }

    // ---------------------------------------------------------------------
    // Indexed staged membership (cpu-seams-spec.md Seam B).
    //
    // `staged_index: BTreeMap<u64, u32>` is a delivery-phase lookup structure
    // replacing the O(S) `.position()` scan in `apply_event` with O(log S).
    // Lifecycle contract (spec B.3): valid only while `staged` accumulates
    // between fused passes; cleared by `absorb_staged` as the single removal
    // path; never read outside that window. The tests below pin the mapping,
    // bit-parity against the pre-change linear-scan hot loop, lifecycle
    // scoping, and accumulation order.
    // ---------------------------------------------------------------------

    /// D2 forward direction: after every delivery-phase mutation the index
    /// must map exactly `{staged[i].gid -> i}` -- nothing more, nothing less.
    fn assert_index_maps_exactly_the_staged_vec(runtime: &ProceduralRuntime) {
        let expected: BTreeMap<u64, u32> = runtime
            .staged
            .iter()
            .enumerate()
            .map(|(slot, state)| (state.gid, slot as u32))
            .collect();
        assert_eq!(
            runtime.staged_index, expected,
            "staged_index must equal {{staged[i].gid -> i}} during delivery"
        );
        // Every index hit must resolve to a same-gid slot (guards u32 slot
        // truncation and stale slots mechanically).
        for (gid, slot) in &runtime.staged_index {
            assert_eq!(
                runtime.staged[*slot as usize].gid, *gid,
                "index entry ({gid} -> {slot}) points at a different cell"
            );
        }
    }

    /// Exact pre-Seam-B `apply_event` (the hot loop the spec replaces):
    /// staged membership resolved by a linear `.position()` scan, no index
    /// maintenance. Bit-parity oracle for the indexed path; deliberately does
    /// NOT touch `staged_index`, so `native_payload_bytes` is the one
    /// legitimately divergent counter (spec B.4 accounting note).
    fn linear_scan_apply_event(
        runtime: &mut ProceduralRuntime,
        event: ProceduralEventRecord,
    ) -> Result<(), u32> {
        let current = canonical_zero(event.current_pa);
        match runtime.resident_index(event.target_gid) {
            Ok(index) => {
                let value = runtime.residents[index].synaptic_current + current;
                runtime.residents[index].synaptic_current = canonical_zero(value);
            }
            Err(_) => {
                if runtime.event_proportional() {
                    if let Some(entry) = runtime.dormant.remove(&event.target_gid) {
                        let restored =
                            runtime.catch_up_eligibility(entry.eligibility, entry.elig_step);
                        debug_assert!(restored.to_bits() != 0 || entry.eligibility.to_bits() == 0);
                        if current.to_bits() == 0 {
                            runtime.dormant.insert(
                                event.target_gid,
                                DormantEntry {
                                    eligibility: entry.eligibility,
                                    elig_step: entry.elig_step,
                                },
                            );
                            return Ok(());
                        }
                        let mut state = CellState::canonical_rest(event.target_gid);
                        state.eligibility = restored;
                        state.synaptic_current = current;
                        runtime.staged.push(state);
                        return Ok(());
                    }
                }
                if let Some(slot) = runtime
                    .staged
                    .iter()
                    .position(|state| state.gid == event.target_gid)
                {
                    let value = runtime.staged[slot].synaptic_current + current;
                    runtime.staged[slot].synaptic_current = canonical_zero(value);
                    return Ok(());
                }
                if current.to_bits() == 0 {
                    return Ok(());
                }
                if runtime.total_residents() == runtime.config.resident_capacity {
                    return Err(super::PROCEDURAL_RESIDENT_CAPACITY);
                }
                if runtime.total_residents()
                    == runtime.residents.capacity() + runtime.staged.capacity()
                {
                    return Err(super::PROCEDURAL_ALLOCATION);
                }
                runtime.scratch.interval_ledger.mark_single(
                    event.target_gid,
                    super::SEEN_RESIDENT,
                    runtime.config.interval_distinct_capacity,
                )?;
                let mut state = CellState::canonical_rest(event.target_gid);
                state.synaptic_current = current;
                runtime.staged.push(state);
            }
        }
        Ok(())
    }

    #[test]
    fn staged_index_maps_exactly_the_staged_vec_during_delivery() {
        // Full-touch keeps the fixture free of dormancy; a second EP phase
        // below covers the dormant-reactivation push site.
        let mut runtime = ep_runtime(64, PROCEDURAL_STEPPING_FULL_TOUCH);
        // Absorbed residents first: an advance must leave both stores empty.
        let batch: Vec<_> = (0..8).map(|gid| drive(gid, 0, 10.0)).collect();
        assert_eq!(runtime.advance(1, &batch), super::PROCEDURAL_OK);
        assert!(runtime.staged.is_empty());
        assert!(runtime.staged_index.is_empty());

        // Delivery phase observable between top-level calls: fresh staging in
        // non-ascending order (slots are push-order, not gid order).
        for gid in [20u64, 22, 21] {
            runtime
                .apply_event(drive(gid, runtime.current_step, 1.5))
                .unwrap();
        }
        assert_index_maps_exactly_the_staged_vec(&runtime);
        assert_eq!(runtime.staged_index.get(&21), Some(&2));

        // Hits accumulate into the mapped slot without changing membership.
        runtime
            .apply_event(drive(22, runtime.current_step, 0.5))
            .unwrap();
        runtime
            .apply_event(drive(22, runtime.current_step, 0.25))
            .unwrap();
        assert_index_maps_exactly_the_staged_vec(&runtime);
        assert_eq!(runtime.staged.len(), 3);

        // Zero-current deliveries stage nothing and hit nothing new.
        runtime
            .apply_event(drive(30, runtime.current_step, 0.0))
            .unwrap();
        runtime
            .apply_event(drive(20, runtime.current_step, 0.0))
            .unwrap();
        assert_index_maps_exactly_the_staged_vec(&runtime);
        assert_eq!(runtime.staged.len(), 3);

        // The next advance's fused pass sorts `staged` (invalidating slots)
        // and absorb clears Vec+index together: post-advance emptiness.
        assert_eq!(runtime.advance(1, &[]), super::PROCEDURAL_OK);
        assert!(runtime.staged.is_empty());
        assert!(runtime.staged_index.is_empty());

        // EP reactivation push site: spike gid 3, wait out recurrent noise
        // until it fast-rests into the dormant store, then deliver directly
        // so the dormant branch stages the cell.
        let mut ep = ep_runtime(16, PROCEDURAL_STEPPING_EVENT_PROPORTIONAL);
        assert_eq!(ep.advance(1, &[drive(3, 0, 10.0)]), super::PROCEDURAL_OK);
        for _ in 0..2000 {
            ep.advance(1, &[]);
            if ep.dormant.contains_key(&3) {
                break;
            }
        }
        assert!(ep.dormant.contains_key(&3), "fixture must dormanize gid 3");
        ep.apply_event(drive(3, ep.current_step, 5.0)).unwrap();
        assert_eq!(ep.staged.len(), 1);
        assert_index_maps_exactly_the_staged_vec(&ep);
        assert_eq!(ep.staged_index.get(&3), Some(&0));
        // Reactivated cell absorbs cleanly on the next advance.
        assert_eq!(ep.advance(1, &[]), super::PROCEDURAL_OK);
        assert!(ep.staged_index.is_empty());
    }

    #[test]
    fn indexed_apply_event_is_bit_identical_to_linear_scan_reference() {
        // Identical schedules over identically constructed runtimes; the only
        // difference is the staged-membership lookup mechanism. Covers both
        // stepping modes, resident hits, within-batch staged hits, duplicate
        // accumulations, zero-current events, and dormant reactivations.
        for mode in [
            PROCEDURAL_STEPPING_FULL_TOUCH,
            PROCEDURAL_STEPPING_EVENT_PROPORTIONAL,
        ] {
            let build_batches = || {
                // Same-target repeats use DISTINCT source gids: event keys are
                // (due_step, target, source, edge_slot), so repeats stay legal
                // while canonical order keeps their delivery sequence fixed.
                let keyed =
                    |due: u64, target: u64, source: u64, value: f64| ProceduralEventRecord {
                        due_step: due,
                        target_gid: target,
                        source_gid: source,
                        edge_slot: 0,
                        current_pa: value,
                    };
                let mut batches = Vec::new();
                // Step 0: residents 0..8 born, plus repeats onto absent
                // targets (fresh staging + immediate staged hits).
                let mut batch: Vec<_> = (0..8).map(|gid| drive(gid, 0, 10.0)).collect();
                batch.push(keyed(0, 20, 9, 3.0));
                batch.push(keyed(0, 20, 10, 1.5));
                batch.push(keyed(0, 21, 11, -2.5));
                batch.push(keyed(0, 20, 12, 0.25)); // third accumulation
                batch.push(keyed(0, 30, 13, 0.0)); // zero-current miss
                batch.push(keyed(0, 31, 14, 4.0));
                batches.push(batch);
                // Step 1: re-hit yesterday's residents, stage fresh targets,
                // duplicate across sources with order-sensitive magnitudes.
                let mut batch: Vec<_> = (0..8).map(|gid| drive(gid, 1, 1e-30)).collect();
                batch.push(keyed(1, 40, 20, 1e16));
                batch.push(keyed(1, 40, 21, 1.0));
                batch.push(keyed(1, 40, 22, -1e16));
                batch.push(keyed(1, 41, 23, 0.125));
                batch.push(keyed(1, 41, 24, 0.125));
                batches.push(batch);
                // Step 2: quiet step lets fast-rest cells retire (EP).
                batches.push(Vec::new());
                // Step 3: re-drive previously retired gids (reactivation).
                batches.push(vec![drive(0, 3, 10.0), drive(1, 3, 6.0)]);
                batches
            };

            let mut indexed = ep_runtime(48, mode);
            let mut oracle = ep_runtime(48, mode);
            let mut indexed_trace = Vec::new();
            let mut oracle_trace = Vec::new();

            for batch in &build_batches() {
                assert_eq!(indexed.advance(1, batch), super::PROCEDURAL_OK);
                assert_eq!(oracle.advance(1, batch), super::PROCEDURAL_OK);
                indexed_trace.extend(
                    indexed
                        .last_spikes
                        .iter()
                        .map(|spike| (spike.step, spike.gid)),
                );
                oracle_trace.extend(
                    oracle
                        .last_spikes
                        .iter()
                        .map(|spike| (spike.step, spike.gid)),
                );
                // Mid-schedule state parity: residents, dormant store, and --
                // where a poisoned abort ever left staged visible -- staged.
                assert_eq!(resident_bits(&indexed), resident_bits(&oracle));
                let indexed_dormant: BTreeMap<u64, (u64, u64)> = indexed
                    .dormant
                    .iter()
                    .map(|(gid, entry)| (*gid, (entry.eligibility.to_bits(), entry.elig_step)))
                    .collect();
                let oracle_dormant: BTreeMap<u64, (u64, u64)> = oracle
                    .dormant
                    .iter()
                    .map(|(gid, entry)| (*gid, (entry.eligibility.to_bits(), entry.elig_step)))
                    .collect();
                assert_eq!(
                    indexed_dormant, oracle_dormant,
                    "dormant store must stay bit-identical"
                );
                assert_eq!(indexed.staged.len(), oracle.staged.len());
                for (a, b) in indexed.staged.iter().zip(oracle.staged.iter()) {
                    assert_eq!(a.gid, b.gid);
                    assert_eq!(a.synaptic_current.to_bits(), b.synaptic_current.to_bits());
                    assert_eq!(a.eligibility.to_bits(), b.eligibility.to_bits());
                }
            }

            // Direct delivery phase between advances: indexed path vs the
            // verbatim linear-scan oracle, interleaved hits/misses/zeros.
            // Push order stays ascending so the snapshot merge's documented
            // sorted-staged assumption (B.6.5 hardening) holds for these
            // deliberately non-poisoned fixtures.
            let direct: Vec<ProceduralEventRecord> = vec![
                drive(24, 4, 2.0),
                drive(25, 4, -1.0),
                drive(24, 4, 8.0),
                drive(26, 4, 0.0),
                drive(25, 4, 0.5),
                drive(3, 4, 1.0), // already a resident (0..8 absorbed)
                drive(27, 4, 1e-16),
                drive(27, 4, 1e16),
            ];
            for event in &direct {
                indexed.apply_event(*event).unwrap();
                linear_scan_apply_event(&mut oracle, *event).unwrap();
                assert_index_maps_exactly_the_staged_vec(&indexed);
            }
            assert!(!indexed.staged.is_empty());
            assert_eq!(indexed.staged.len(), oracle.staged.len());
            for (a, b) in indexed.staged.iter().zip(oracle.staged.iter()) {
                assert_eq!(a.gid, b.gid);
                assert_eq!(a.synaptic_current.to_bits(), b.synaptic_current.to_bits());
            }

            // Full-state parity through the real snapshot export: every f64
            // field compared via to_bits, never ==.
            let indexed_records = snapshot_records(&indexed);
            let oracle_records = snapshot_records(&oracle);
            assert_eq!(indexed_records.len(), oracle_records.len());
            for (gid, record) in &indexed_records {
                let other = oracle_records
                    .get(gid)
                    .unwrap_or_else(|| panic!("oracle missing gid {gid}"));
                assert_eq!(record_bits(record), record_bits(other), "gid {gid}");
            }

            // Counters: everything except native_payload_bytes, which shifts
            // by the documented index term (spec B.4) because the oracle path
            // maintains no index.
            let a = indexed.counters;
            let b = oracle.counters;
            assert_eq!(a.current_step, b.current_step);
            assert_eq!(a.resident_peak, b.resident_peak);
            assert_eq!(a.resident_distinct, b.resident_distinct);
            assert_eq!(a.updated_distinct, b.updated_distinct);
            assert_eq!(a.spiking_distinct, b.spiking_distinct);
            assert_eq!(a.cell_step_evaluations, b.cell_step_evaluations);
            assert_eq!(a.spike_events, b.spike_events);
            assert_eq!(a.generated_edges, b.generated_edges);
            assert_eq!(a.delivered_events, b.delivered_events);
            assert_eq!(a.queue_peak, b.queue_peak);
            assert_eq!(a.retained_nonzero_cells, b.retained_nonzero_cells);
            assert_eq!(a.overflow_code, b.overflow_code);
            assert_eq!(indexed_trace, oracle_trace, "spike traces must match");
            assert!(
                !indexed_trace.is_empty(),
                "fixture must actually spike cells"
            );
        }
    }

    #[test]
    fn fresh_resident_run_matches_scalar_oracle_and_removes_interleaved_shifts() {
        let keyed = |target_gid: u64, source_gid: u64, current_pa: f64| ProceduralEventRecord {
            due_step: 1,
            target_gid,
            source_gid,
            edge_slot: 0,
            current_pa,
        };
        let preload: Vec<_> = [0u64, 2, 4, 6]
            .into_iter()
            .map(|gid| drive(gid, 0, 0.25))
            .collect();
        let make_preloaded = |distinct_capacity: usize| {
            let mut config = ep_config(64, PROCEDURAL_STEPPING_EVENT_PROPORTIONAL);
            config.resident_capacity = 64;
            config.interval_distinct_capacity = distinct_capacity;
            let decay = [0.5, 0.5];
            let mut runtime =
                ProceduralRuntime::new(ep_glif(), config, decay, decay, [0.0; 2]).unwrap();
            assert_eq!(runtime.advance(1, &preload), super::PROCEDURAL_OK);
            // Keep dormant-state parity non-vacuous while the interleaved
            // active ledger remains exactly the four even GIDs above.
            runtime.dormant.insert(
                30,
                DormantEntry {
                    eligibility: 0.25,
                    elig_step: 0,
                },
            );
            runtime
        };

        let resident_images = |runtime: &ProceduralRuntime| -> Vec<Vec<u8>> {
            runtime
                .residents
                .iter()
                .map(|state| resident_raw_bytes(&state.record()).to_vec())
                .collect()
        };
        let staged_images = |runtime: &ProceduralRuntime| -> Vec<Vec<u8>> {
            runtime
                .staged
                .iter()
                .map(|state| resident_raw_bytes(&state.record()).to_vec())
                .collect()
        };
        let pending_words = |runtime: &ProceduralRuntime| -> Vec<[u64; 5]> {
            runtime
                .pending
                .iter()
                .map(|event| {
                    [
                        event.due_step,
                        event.target_gid,
                        event.source_gid,
                        event.edge_slot,
                        event.current_pa.to_bits(),
                    ]
                })
                .collect()
        };
        let dormant_words = |runtime: &ProceduralRuntime| -> Vec<(u64, u64, u64)> {
            runtime
                .dormant
                .iter()
                .map(|(gid, entry)| (*gid, entry.eligibility.to_bits(), entry.elig_step))
                .collect()
        };
        let spike_words = |runtime: &ProceduralRuntime| -> Vec<(u64, u64)> {
            runtime
                .last_spikes
                .iter()
                .map(|spike| (spike.step, spike.gid))
                .collect()
        };
        let assert_raw_parity = |merged: &ProceduralRuntime, scalar: &ProceduralRuntime| {
            assert_eq!(resident_images(merged), resident_images(scalar));
            assert_eq!(staged_images(merged), staged_images(scalar));
            assert_eq!(pending_words(merged), pending_words(scalar));
            assert_eq!(merged.pending_head, scalar.pending_head);
            assert_eq!(dormant_words(merged), dormant_words(scalar));
            assert_eq!(spike_words(merged), spike_words(scalar));
            assert_eq!(
                counters_raw_bytes(&merged.counters),
                counters_raw_bytes(&scalar.counters)
            );
            assert_eq!(merged.current_step, scalar.current_step);
            assert_eq!(merged.poisoned, scalar.poisoned);
        };

        // Canonical target/source order interleaves four preloaded even GIDs
        // with four fresh odd GIDs. Repeated fresh targets exercise exact
        // accumulation order, including a cancellation that is order-sensitive
        // in binary64 and a canonical-zero miss before a nonzero repeat.
        let success_events = vec![
            keyed(0, 0, 2.0),
            keyed(1, 1, 2.0),
            keyed(1, 40, 0.5),
            keyed(2, 2, 2.0),
            keyed(3, 3, 1e16),
            keyed(3, 40, -1e16),
            keyed(3, 41, 2.0),
            keyed(4, 4, 2.0),
            keyed(5, 5, -0.0),
            keyed(5, 42, 2.0),
            keyed(6, 6, 2.0),
            keyed(7, 7, 2.0),
            keyed(7, 43, -0.5),
        ];
        let mut merged = make_preloaded(64);
        let mut scalar = make_preloaded(64);
        assert_eq!(merged.advance(1, &success_events), super::PROCEDURAL_OK);
        assert_eq!(
            scalar.advance_with_fresh_run::<false, false>(1, &success_events),
            super::PROCEDURAL_OK
        );
        assert_raw_parity(&merged, &scalar);
        assert!(merged.staged.is_empty());
        assert!(!merged.last_spikes.is_empty());
        assert!(!merged.pending_slice().is_empty());
        assert!(!merged.dormant.is_empty());

        // With four entry residents and capacity seven, 1/3/5 fit and fresh
        // gid 7 is the exact first offender. The merged path must poison at the
        // same event as scalar insertion, leaving every externally observable
        // partial mutation byte-identical.
        let overflow_events = vec![
            keyed(0, 0, 0.125),
            keyed(1, 1, 1.0),
            keyed(1, 40, 0.5),
            keyed(2, 2, 0.125),
            keyed(3, 3, 1.0),
            keyed(4, 4, 0.125),
            keyed(5, 5, -0.0),
            keyed(5, 42, 1.0),
            keyed(6, 6, 0.125),
            keyed(7, 7, 1.0),
            keyed(8, 8, 1.0),
        ];
        let mut merged_overflow = make_preloaded(7);
        let mut scalar_overflow = make_preloaded(7);
        assert_eq!(
            merged_overflow.advance(1, &overflow_events),
            super::PROCEDURAL_DISTINCT_CAPACITY
        );
        assert_eq!(
            scalar_overflow.advance_with_fresh_run::<false, false>(1, &overflow_events),
            super::PROCEDURAL_DISTINCT_CAPACITY
        );
        assert_raw_parity(&merged_overflow, &scalar_overflow);
        let accepted: Vec<_> = merged_overflow
            .staged
            .iter()
            .map(|state| state.gid)
            .collect();
        assert_eq!(accepted, vec![1, 3, 5]);
        assert!(!merged_overflow.staged_index.contains_key(&7));

        let mut scalar_ledger = super::IntervalLedger::try_new(7).unwrap();
        let mut merged_ledger = super::IntervalLedger::try_new(7).unwrap();
        for gid in [0u64, 2, 4, 6] {
            scalar_ledger
                .mark_single(gid, super::SEEN_RESIDENT, 7)
                .unwrap();
            merged_ledger
                .mark_single(gid, super::SEEN_RESIDENT, 7)
                .unwrap();
        }
        let mut fresh_run = Vec::with_capacity(7);
        let fresh_targets = [1u64, 1, 3, 5, 5, 7, 9];
        let mut scalar_offender = None;
        let mut merged_offender = None;
        for gid in fresh_targets {
            if scalar_offender.is_none()
                && scalar_ledger
                    .mark_single(gid, super::SEEN_RESIDENT, 7)
                    .is_err()
            {
                scalar_offender = Some(gid);
            }
            if merged_offender.is_none()
                && merged_ledger
                    .stage_single_in_run(gid, super::SEEN_RESIDENT, &mut fresh_run, 7)
                    .is_err()
            {
                merged_offender = Some(gid);
            }
        }
        assert_eq!(scalar_offender, Some(7));
        assert_eq!(merged_offender, scalar_offender);
        assert_eq!(fresh_run, vec![1, 3, 5]);

        // Lightweight median at a shift-heavy, interleaved shape. Construction
        // of the identical preloaded ledgers is outside the timed region; the
        // merged sample includes all binary searches plus the union merge.
        const PRELOADED: usize = 4_096;
        const FRESH: usize = 2_048;
        const CAPACITY: usize = PRELOADED + FRESH + 8;
        const WARMUP: usize = 1;
        const SAMPLES: usize = 7;
        let build_ledger = || {
            let mut ledger = super::IntervalLedger::try_new(CAPACITY).unwrap();
            for index in 0..PRELOADED {
                ledger
                    .mark_single((index * 2) as u64, super::SEEN_UPDATED, CAPACITY)
                    .unwrap();
            }
            ledger
        };
        let time_scalar = || {
            let mut ledger = build_ledger();
            let start = std::time::Instant::now();
            for index in 0..FRESH {
                ledger
                    .mark_single((index * 2 + 1) as u64, super::SEEN_RESIDENT, CAPACITY)
                    .unwrap();
            }
            let elapsed = start.elapsed().as_nanos();
            assert_eq!(ledger.gids.len(), PRELOADED + FRESH);
            elapsed
        };
        let time_merged = || {
            let mut ledger = build_ledger();
            let mut run = Vec::with_capacity(FRESH);
            let start = std::time::Instant::now();
            for index in 0..FRESH {
                ledger
                    .stage_single_in_run(
                        (index * 2 + 1) as u64,
                        super::SEEN_RESIDENT,
                        &mut run,
                        CAPACITY,
                    )
                    .unwrap();
            }
            ledger.merge_run_in_place(&run, super::SEEN_RESIDENT);
            let elapsed = start.elapsed().as_nanos();
            assert_eq!(ledger.gids.len(), PRELOADED + FRESH);
            elapsed
        };
        let mut scalar_samples = Vec::with_capacity(SAMPLES);
        let mut merged_samples = Vec::with_capacity(SAMPLES);
        for repetition in 0..WARMUP + SAMPLES {
            let (scalar_ns, merged_ns) = if repetition % 2 == 0 {
                (time_scalar(), time_merged())
            } else {
                let merged_ns = time_merged();
                (time_scalar(), merged_ns)
            };
            if repetition >= WARMUP {
                scalar_samples.push(scalar_ns);
                merged_samples.push(merged_ns);
            }
        }
        scalar_samples.sort_unstable();
        merged_samples.sort_unstable();
        let scalar_median_ns = scalar_samples[SAMPLES / 2];
        let merged_median_ns = merged_samples[SAMPLES / 2];
        eprintln!(
            "fresh-resident-run scalar_median_ns={} merged_median_ns={} speedup={:.2}x preloaded={} fresh={} samples={}",
            scalar_median_ns,
            merged_median_ns,
            scalar_median_ns as f64 / merged_median_ns as f64,
            PRELOADED,
            FRESH,
            SAMPLES
        );
        assert!(
            merged_median_ns < scalar_median_ns,
            "one union merge must beat repeated interleaved Vec insertions"
        );
    }

    #[test]
    fn interval_ledger_in_place_merge_matches_oracle_and_reclaims_duplicate_capacity() {
        fn merge_forward_oracle(
            source: &super::IntervalLedger,
            run: &[u64],
            flag: u8,
            output: &mut super::IntervalLedger,
        ) {
            output.clear();
            let mut i = 0usize;
            let mut j = 0usize;
            while i < source.gids.len() && j < run.len() {
                match source.gids[i].cmp(&run[j]) {
                    std::cmp::Ordering::Less => {
                        output.gids.push(source.gids[i]);
                        output.flags.push(source.flags[i]);
                        i += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        output.gids.push(run[j]);
                        output.flags.push(flag);
                        j += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        output.gids.push(source.gids[i]);
                        output.flags.push(source.flags[i] | flag);
                        i += 1;
                        j += 1;
                    }
                }
            }
            output.gids.extend_from_slice(&source.gids[i..]);
            output.flags.extend_from_slice(&source.flags[i..]);
            output.gids.extend_from_slice(&run[j..]);
            output
                .flags
                .extend(std::iter::repeat_n(flag, run.len() - j));
        }

        const EXISTING: usize = 16_384;
        const RUN: usize = 16_384;
        const CAPACITY: usize = EXISTING + RUN;
        const WARMUP: usize = 1;
        const SAMPLES: usize = 7;

        let build_ledger = || {
            let mut ledger = super::IntervalLedger::try_new(CAPACITY).unwrap();
            for index in 0..EXISTING {
                let gid = (index * 2) as u64;
                ledger
                    .mark_single(gid, super::SEEN_RESIDENT, CAPACITY)
                    .unwrap();
                if index % 2 == 0 {
                    ledger
                        .mark_single(gid, super::SEEN_UPDATED, CAPACITY)
                        .unwrap();
                }
                if index % 5 == 0 {
                    ledger
                        .mark_single(gid, super::SEEN_SPIKING, CAPACITY)
                        .unwrap();
                }
            }
            ledger
        };
        let run: Vec<_> = (0..RUN).map(|index| (index * 3) as u64).collect();

        // Interleaved overlap exercises every merge arm and nontrivial prior
        // flag combinations. The independent forward-buffer oracle is the
        // exact algorithm replaced by the backward in-place merge.
        let source = build_ledger();
        let mut oracle = super::IntervalLedger::try_new(CAPACITY).unwrap();
        merge_forward_oracle(&source, &run, super::SEEN_UPDATED, &mut oracle);
        let mut merged = build_ledger();
        assert!(merged.can_fit_run(&run, CAPACITY));
        merged.merge_run_in_place(&run, super::SEEN_UPDATED);
        assert_eq!(merged.gids, oracle.gids);
        assert_eq!(merged.flags, oracle.flags);
        assert!(merged.gids.windows(2).all(|pair| pair[0] < pair[1]));

        // Account the removed persistent allocation at a representative
        // capacity. The historical second ledger had exactly the same two
        // reserved vectors as the live ledger plus its two Vec headers.
        const PAYLOAD_DISTINCT: usize = 65_536;
        let mut config = ep_config(
            (PAYLOAD_DISTINCT + 1) as u64,
            PROCEDURAL_STEPPING_EVENT_PROPORTIONAL,
        );
        config.resident_capacity = PAYLOAD_DISTINCT;
        config.event_capacity = 64;
        config.spike_capacity = 64;
        config.interval_distinct_capacity = PAYLOAD_DISTINCT;
        let decay = [0.5, 0.5];
        let runtime = ProceduralRuntime::new(ep_glif(), config, decay, decay, [0.0; 2]).unwrap();
        let ledger_capacity_bytes = runtime.scratch.interval_ledger.gids.capacity()
            * std::mem::size_of::<u64>()
            + runtime.scratch.interval_ledger.flags.capacity() * std::mem::size_of::<u8>();
        let reclaimed_bytes = ledger_capacity_bytes + std::mem::size_of::<super::IntervalLedger>();
        assert_eq!(
            ledger_capacity_bytes,
            PAYLOAD_DISTINCT * (std::mem::size_of::<u64>() + std::mem::size_of::<u8>())
        );
        assert_eq!(
            reclaimed_bytes,
            PAYLOAD_DISTINCT * 9 + std::mem::size_of::<super::IntervalLedger>()
        );

        let time_oracle = || {
            let source = build_ledger();
            let mut output = super::IntervalLedger::try_new(CAPACITY).unwrap();
            let start = std::time::Instant::now();
            merge_forward_oracle(&source, &run, super::SEEN_UPDATED, &mut output);
            let elapsed = start.elapsed().as_nanos();
            assert_eq!(output.gids.len(), oracle.gids.len());
            elapsed
        };
        let time_in_place = || {
            let mut ledger = build_ledger();
            let start = std::time::Instant::now();
            ledger.merge_run_in_place(&run, super::SEEN_UPDATED);
            let elapsed = start.elapsed().as_nanos();
            assert_eq!(ledger.gids.len(), oracle.gids.len());
            elapsed
        };
        let mut oracle_samples = Vec::with_capacity(SAMPLES);
        let mut in_place_samples = Vec::with_capacity(SAMPLES);
        for repetition in 0..WARMUP + SAMPLES {
            let (oracle_ns, in_place_ns) = if repetition % 2 == 0 {
                (time_oracle(), time_in_place())
            } else {
                let in_place_ns = time_in_place();
                (time_oracle(), in_place_ns)
            };
            if repetition >= WARMUP {
                oracle_samples.push(oracle_ns);
                in_place_samples.push(in_place_ns);
            }
        }
        oracle_samples.sort_unstable();
        in_place_samples.sort_unstable();
        let oracle_median_ns = oracle_samples[SAMPLES / 2];
        let in_place_median_ns = in_place_samples[SAMPLES / 2];
        eprintln!(
            "interval-ledger-in-place oracle_median_ns={} in_place_median_ns={} ratio={:.2}x existing={} run={} samples={} current_payload_bytes={} reclaimed_bytes={} reclaimed_at_1m_bytes={}",
            oracle_median_ns,
            in_place_median_ns,
            oracle_median_ns as f64 / in_place_median_ns as f64,
            EXISTING,
            RUN,
            SAMPLES,
            runtime.payload_bytes(),
            reclaimed_bytes,
            1_000_000usize * 9 + std::mem::size_of::<super::IntervalLedger>(),
        );
    }

    #[test]
    fn canonical_staged_run_matches_indexed_oracle_and_elides_tree_and_sorts() {
        let keyed = |target_gid: u64, source_gid: u64, current_pa: f64| ProceduralEventRecord {
            due_step: 1,
            target_gid,
            source_gid,
            edge_slot: 0,
            current_pa,
        };
        let preload: Vec<_> = [0u64, 2, 4, 6]
            .into_iter()
            .map(|gid| drive(gid, 0, 0.25))
            .collect();
        let make_preloaded = || {
            let mut config = ep_config(64, PROCEDURAL_STEPPING_EVENT_PROPORTIONAL);
            config.resident_capacity = 64;
            let decay = [0.5, 0.5];
            let mut runtime =
                ProceduralRuntime::new(ep_glif(), config, decay, decay, [0.0; 2]).unwrap();
            assert_eq!(runtime.advance(1, &preload), super::PROCEDURAL_OK);
            runtime.dormant.insert(
                30,
                DormantEntry {
                    eligibility: 0.25,
                    elig_step: 0,
                },
            );
            runtime
        };
        let events = vec![
            keyed(0, 0, 2.0),
            keyed(1, 1, 2.0),
            keyed(1, 40, 0.5),
            keyed(2, 2, 2.0),
            keyed(3, 3, 1e16),
            keyed(3, 40, -1e16),
            keyed(3, 41, 2.0),
            keyed(4, 4, 2.0),
            keyed(5, 5, -0.0),
            keyed(5, 42, 2.0),
            keyed(6, 6, 2.0),
            keyed(7, 7, 2.0),
            keyed(7, 43, -0.5),
            keyed(30, 30, 2.0),
            keyed(30, 50, 0.25),
        ];
        let mut canonical = make_preloaded();
        let mut indexed = make_preloaded();
        assert_eq!(canonical.advance(1, &events), super::PROCEDURAL_OK);
        assert_eq!(
            indexed.advance_with_fresh_run::<true, false>(1, &events),
            super::PROCEDURAL_OK
        );
        let resident_images = |runtime: &ProceduralRuntime| -> Vec<Vec<u8>> {
            runtime
                .residents
                .iter()
                .map(|state| resident_raw_bytes(&state.record()).to_vec())
                .collect()
        };
        let pending_words = |runtime: &ProceduralRuntime| -> Vec<[u64; 5]> {
            runtime
                .pending
                .iter()
                .map(|event| {
                    [
                        event.due_step,
                        event.target_gid,
                        event.source_gid,
                        event.edge_slot,
                        event.current_pa.to_bits(),
                    ]
                })
                .collect()
        };
        let dormant_words = |runtime: &ProceduralRuntime| -> Vec<(u64, u64, u64)> {
            runtime
                .dormant
                .iter()
                .map(|(gid, entry)| (*gid, entry.eligibility.to_bits(), entry.elig_step))
                .collect()
        };
        let spike_words = |runtime: &ProceduralRuntime| -> Vec<(u64, u64)> {
            runtime
                .last_spikes
                .iter()
                .map(|spike| (spike.step, spike.gid))
                .collect()
        };
        assert_eq!(resident_images(&canonical), resident_images(&indexed));
        assert_eq!(pending_words(&canonical), pending_words(&indexed));
        assert_eq!(canonical.pending_head, indexed.pending_head);
        assert_eq!(dormant_words(&canonical), dormant_words(&indexed));
        assert_eq!(spike_words(&canonical), spike_words(&indexed));
        assert_eq!(
            counters_raw_bytes(&canonical.counters),
            counters_raw_bytes(&indexed.counters)
        );
        assert_eq!(canonical.poisoned, indexed.poisoned);
        assert_eq!(canonical.payload_bytes(), indexed.payload_bytes());
        assert!(canonical.staged.is_empty());
        assert!(canonical.staged_index.is_empty());
        assert!(!canonical.last_spikes.is_empty());

        // Isolate canonical staged admission and absorption from GLIF work.
        // Both arms use the already-banked fresh-ledger run; the oracle alone
        // maintains the B-tree and repeats the two historical staged sorts.
        const FRESH: usize = 8_192;
        const CAPACITY: usize = FRESH + 16;
        const WARMUP: usize = 1;
        const SAMPLES: usize = 7;
        let make_timing_runtime = || {
            let mut config = ep_config((CAPACITY + 1) as u64, PROCEDURAL_STEPPING_FULL_TOUCH);
            config.resident_capacity = CAPACITY;
            config.event_capacity = CAPACITY;
            config.spike_capacity = CAPACITY;
            config.interval_distinct_capacity = CAPACITY;
            let decay = [0.5, 0.5];
            ProceduralRuntime::new(ep_glif(), config, decay, decay, [0.0; 2]).unwrap()
        };
        let time_indexed = || {
            let mut runtime = make_timing_runtime();
            runtime.scratch.gid_run.clear();
            let start = std::time::Instant::now();
            for gid in 0..FRESH as u64 {
                runtime
                    .apply_event_with_fresh_run::<true, false>(drive(gid, 0, 1.0))
                    .unwrap();
            }
            {
                let run = &runtime.scratch.gid_run;
                let ledger = &mut runtime.scratch.interval_ledger;
                ledger.merge_run_in_place(run, super::SEEN_RESIDENT);
            }
            runtime.staged.sort_unstable_by_key(|state| state.gid);
            runtime.absorb_staged_with_order::<false>().unwrap();
            let elapsed = start.elapsed().as_nanos();
            assert_eq!(runtime.residents.len(), FRESH);
            assert!(runtime.staged_index.is_empty());
            (elapsed, runtime)
        };
        let time_canonical = || {
            let mut runtime = make_timing_runtime();
            runtime.scratch.gid_run.clear();
            let start = std::time::Instant::now();
            for gid in 0..FRESH as u64 {
                runtime
                    .apply_event_with_fresh_run::<true, true>(drive(gid, 0, 1.0))
                    .unwrap();
            }
            {
                let run = &runtime.scratch.gid_run;
                let ledger = &mut runtime.scratch.interval_ledger;
                ledger.merge_run_in_place(run, super::SEEN_RESIDENT);
            }
            runtime.absorb_staged_with_order::<true>().unwrap();
            let elapsed = start.elapsed().as_nanos();
            assert_eq!(runtime.residents.len(), FRESH);
            assert!(runtime.staged_index.is_empty());
            (elapsed, runtime)
        };
        let mut indexed_samples = Vec::with_capacity(SAMPLES);
        let mut canonical_samples = Vec::with_capacity(SAMPLES);
        for repetition in 0..WARMUP + SAMPLES {
            let ((indexed_ns, indexed_runtime), (canonical_ns, canonical_runtime)) =
                if repetition % 2 == 0 {
                    (time_indexed(), time_canonical())
                } else {
                    let canonical = time_canonical();
                    (time_indexed(), canonical)
                };
            assert_eq!(
                resident_images(&indexed_runtime),
                resident_images(&canonical_runtime)
            );
            assert_eq!(
                indexed_runtime.scratch.interval_ledger.gids,
                canonical_runtime.scratch.interval_ledger.gids
            );
            assert_eq!(
                indexed_runtime.scratch.interval_ledger.flags,
                canonical_runtime.scratch.interval_ledger.flags
            );
            assert_eq!(
                indexed_runtime.payload_bytes(),
                canonical_runtime.payload_bytes()
            );
            if repetition >= WARMUP {
                indexed_samples.push(indexed_ns);
                canonical_samples.push(canonical_ns);
            }
        }
        indexed_samples.sort_unstable();
        canonical_samples.sort_unstable();
        let indexed_median_ns = indexed_samples[SAMPLES / 2];
        let canonical_median_ns = canonical_samples[SAMPLES / 2];
        eprintln!(
            "canonical-staged-run indexed_median_ns={} canonical_median_ns={} speedup={:.2}x fresh={} samples={}",
            indexed_median_ns,
            canonical_median_ns,
            indexed_median_ns as f64 / canonical_median_ns as f64,
            FRESH,
            SAMPLES
        );
        assert!(
            canonical_median_ns < indexed_median_ns,
            "canonical Vec admission and one merge must beat B-tree indexing plus two sorts"
        );
    }

    #[test]
    fn staged_index_cleared_by_absorb_and_unused_after_sort() {
        // Normal advance: the fused pass sorts `staged` (which invalidates
        // every slot) and then absorb clears Vec+index together. Post-advance
        // emptiness proves no stale slot survives the sort boundary.
        let mut runtime = ep_runtime(32, PROCEDURAL_STEPPING_FULL_TOUCH);
        let batch: Vec<_> = (0..4).map(|gid| drive(gid, 0, 10.0)).collect();
        assert_eq!(runtime.advance(1, &batch), super::PROCEDURAL_OK);
        assert!(runtime.staged.is_empty());
        assert!(runtime.staged_index.is_empty());

        // Poison mid-delivery: capacity fail-stop aborts the delivery phase
        // with `staged` non-empty and the index populated-but-doomed (rule
        // B.3.4). Poisoning is terminal -- the very next advance must refuse
        // with PROCEDURAL_POISONED rather than consult the stale structure.
        let mut config = ep_config(16, PROCEDURAL_STEPPING_FULL_TOUCH);
        config.resident_capacity = 4;
        let decay = [0.5, 0.5];
        let mut doomed = ProceduralRuntime::new(ep_glif(), config, decay, decay, [0.0; 2]).unwrap();
        let overflow: Vec<_> = (0..6).map(|gid| drive(gid, 0, 10.0)).collect();
        assert_eq!(
            doomed.advance(1, &overflow),
            super::PROCEDURAL_RESIDENT_CAPACITY
        );
        assert_ne!(doomed.poisoned, super::PROCEDURAL_OK);
        assert!(!doomed.staged.is_empty() || !doomed.staged_index.is_empty());
        assert_eq!(doomed.advance(1, &[]), super::PROCEDURAL_POISONED);
        assert_eq!(doomed.advance(1, &[]), super::PROCEDURAL_POISONED);

        // Mid-fused-pass poison variant (rule B.3.3/B.3.4): spike-capacity
        // overflow fires after the fused-pass sort has invalidated slots but
        // before absorb clears them; subsequent advances still fail closed.
        let mut tight = ep_config(64, PROCEDURAL_STEPPING_FULL_TOUCH);
        tight.spike_capacity = 2;
        let mut cut = ProceduralRuntime::new(ep_glif(), tight, decay, decay, [0.0; 2]).unwrap();
        let spikes: Vec<_> = (0..6).map(|gid| drive(gid, 0, 10.0)).collect();
        assert_eq!(cut.advance(1, &spikes), super::PROCEDURAL_SPIKE_CAPACITY);
        assert_ne!(cut.poisoned, super::PROCEDURAL_OK);
        assert_eq!(cut.advance(1, &[]), super::PROCEDURAL_POISONED);
        // And a clean runtime driven through the identical schedule shows the
        // poison came from the capacity bound, not the schedule shape.
        let mut fits = ep_runtime(64, PROCEDURAL_STEPPING_FULL_TOUCH);
        assert_eq!(fits.advance(1, &spikes), super::PROCEDURAL_OK);
        assert!(fits.staged_index.is_empty());
    }

    #[test]
    fn duplicate_delivery_accumulation_order_preserved() {
        // Order-sensitive magnitudes: ((0+1e16)+1)-1e16 != 0 under float
        // addition, so any reorder of the canonical delivery sequence would
        // move bits. Guards the B.5 determinism argument mechanically at the
        // exact branch this seam replaced.
        let currents = [0.0f64, 1e16, 1.0, -1e16, 1.0, 2.0, -1.0, 1e-16, 3.0];
        let target = 50u64;

        // Direct deliveries: pure accumulation through the indexed hit path,
        // no fused-pass decay contaminating the observed value.
        let mut runtime = ep_runtime(64, PROCEDURAL_STEPPING_FULL_TOUCH);
        for (source, value) in currents.iter().enumerate() {
            runtime
                .apply_event(ProceduralEventRecord {
                    due_step: 0,
                    target_gid: target,
                    source_gid: source as u64,
                    edge_slot: 0,
                    current_pa: *value,
                })
                .unwrap();
        }
        assert_eq!(runtime.staged.len(), 1);
        // Oracle: fold left over the canonical pending order (events sorted
        // by (due_step, target, source, edge_slot); ascending source here),
        // starting from the canonical-rest zero, canonical_zero after each
        // addition exactly as apply_event does.
        let mut expected = 0.0f64;
        for value in &currents {
            expected = canonical_zero(expected + *value);
        }
        assert_ne!(expected, 0.0, "magnitudes must be order-sensitive");
        assert_eq!(
            runtime.staged[0].synaptic_current.to_bits(),
            expected.to_bits(),
            "accumulated synaptic_current must bit-equal the sequential-sum oracle"
        );

        // Same sequence through the linear-scan oracle runtime: identical
        // bits, proving the index changed the lookup and nothing else.
        let mut oracle = ep_runtime(64, PROCEDURAL_STEPPING_FULL_TOUCH);
        for (source, value) in currents.iter().enumerate() {
            linear_scan_apply_event(
                &mut oracle,
                ProceduralEventRecord {
                    due_step: 0,
                    target_gid: target,
                    source_gid: source as u64,
                    edge_slot: 0,
                    current_pa: *value,
                },
            )
            .unwrap();
        }
        assert_eq!(
            oracle.staged[0].synaptic_current.to_bits(),
            runtime.staged[0].synaptic_current.to_bits()
        );

        // Through a real advance: many duplicates inside one delivery batch
        // land on one staged slot; the absorbed resident carries the same
        // folded value scaled by exactly one fused-pass recurrent decay.
        let mut advanced = ep_runtime(64, PROCEDURAL_STEPPING_FULL_TOUCH);
        let batch: Vec<ProceduralEventRecord> = currents
            .iter()
            .enumerate()
            .map(|(source, value)| ProceduralEventRecord {
                due_step: 0,
                target_gid: target,
                source_gid: source as u64,
                edge_slot: 0,
                current_pa: *value,
            })
            .collect();
        assert_eq!(advanced.advance(1, &batch), super::PROCEDURAL_OK);
        let state = advanced
            .residents
            .iter()
            .find(|state| state.gid == target)
            .expect("duplicate target must survive as a resident");
        let decayed = canonical_zero(expected * advanced.config.recurrent_decay);
        assert_eq!(state.synaptic_current.to_bits(), decayed.to_bits());
    }

    /// Acceptance gate B-G1 (cpu-seams-spec.md Seam B.8(a)): with S = 10,000
    /// entries staged, 10,000 mixed operations (50% staged hits + 50% fresh
    /// staging misses) plus one wholesale clear must amortize below 1 µs/op.
    /// Threshold is generous enough for debug builds; measure in --release
    /// for the recorded number. Each repetition uses a fresh runtime so gid
    /// ranges never collide with previously absorbed residents.
    #[test]
    fn staged_index_amortized_op_gate_b_g1() {
        const STAGED_BURST: usize = 10_000;
        const MIXED_OPS: usize = 10_000;
        const WARMUP_REPS: usize = 2;
        const TIMED_REPS: usize = 11;

        let make_runtime = || {
            ProceduralRuntime::new(
                ep_glif(),
                ProceduralConfig {
                    configured_cells: 1_000_000,
                    fanout: 4,
                    seed: 71,
                    resident_capacity: 65_536,
                    event_capacity: 65_536,
                    spike_capacity: 1024,
                    interval_distinct_capacity: 32_768,
                    recurrent_weight_pa: 0.25,
                    recurrent_delay_steps: 1,
                    recurrent_decay: 0.5,
                    eligibility_decay: 0.5,
                    stepping_mode: PROCEDURAL_STEPPING_FULL_TOUCH,
                },
                [0.5, 0.5],
                [0.5, 0.5],
                [0.0; 2],
            )
            .unwrap()
        };

        let mut timed_nanos: u128 = 0;
        for rep in 0..WARMUP_REPS + TIMED_REPS {
            let mut runtime = make_runtime();
            // Untimed burst: S distinct fresh targets stage in one delivery
            // rush, none yet absorbed (S = |staged| for every timed op).
            for gid in 0..STAGED_BURST as u64 {
                runtime.apply_event(drive(gid, 0, 1.0)).unwrap();
            }
            assert_eq!(runtime.staged.len(), STAGED_BURST);

            let start = std::time::Instant::now();
            for index in 0..MIXED_OPS {
                let event = if index % 2 == 0 {
                    // Hit: accumulate into an existing staged target.
                    drive((index / 2) as u64, 0, 1.0)
                } else {
                    // Miss: fresh staging insertion beyond the burst.
                    drive(STAGED_BURST as u64 + (index / 2) as u64, 0, 1.0)
                };
                runtime.apply_event(event).unwrap();
            }
            // One wholesale clear per repetition: the single removal path.
            runtime.absorb_staged().unwrap();
            let elapsed = start.elapsed();

            // Ledger grew by the 15k distinct marks of this rep only; a fresh
            // runtime follows, so no cross-repetition capacity debt exists.
            assert_eq!(runtime.staged_index.len(), 0);
            if rep >= WARMUP_REPS {
                timed_nanos += elapsed.as_nanos();
            }
        }

        let amortized_us_per_op =
            timed_nanos as f64 / (TIMED_REPS as f64 * MIXED_OPS as f64) / 1_000.0;
        eprintln!(
            "B-G1 staged-index amortized_us_per_op={:.4} (gate < 1.0, S={}, ops/rep={}, reps={})",
            amortized_us_per_op, STAGED_BURST, MIXED_OPS, TIMED_REPS
        );
        println!(
            "B-G1 staged-index amortized_us_per_op={:.4}",
            amortized_us_per_op
        );
        // Hard gate runs in --release (the recorded number). Debug builds get
        // a generous bound that still discriminates O(log S) indexing from
        // the replaced O(S) linear scan (~tens of us/op at S=10k in debug).
        let threshold_us_per_op = if cfg!(debug_assertions) { 25.0 } else { 1.0 };
        assert!(
            amortized_us_per_op < threshold_us_per_op,
            "B-G1 FAILED: {amortized_us_per_op:.4} us/op amortized at S={STAGED_BURST} \
             (gate < {threshold_us_per_op} us/op)"
        );
    }
}
