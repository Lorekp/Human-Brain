//! Source-driven procedural execution over a bounded sparse working set.
//!
//! This module deliberately has no operation proportional to the configured
//! address range. The sparse runtime stores only bounded resident state,
//! pending events, and spikes produced by the most recent advance. A separate
//! dense runtime exists only as a small-domain exactness oracle.

use super::GlifConfig;
use std::{
    alloc::{alloc, Layout},
    cmp::Ordering,
    collections::HashSet,
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

const PHILOX_M2X64_0: u64 = 0xD2B74407B1CE6E93;
const PHILOX_W64_0: u64 = 0x9E3779B97F4A7C15;
const PHILOX_ROUNDS: usize = 10;
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

    fn record(&self) -> ProceduralResidentRecord {
        ProceduralResidentRecord {
            gid: self.gid,
            voltage_relative_mv: self.voltage,
            after_spike_current_0_pa: self.after[0],
            after_spike_current_1_pa: self.after[1],
            threshold_spike_mv: self.threshold_spike,
            threshold_voltage_mv: self.threshold_voltage,
            refractory_steps: self.refractory,
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
fn philox2x64_10(counter_0: u64, counter_1: u64, key: u64) -> (u64, u64) {
    let mut left = counter_0;
    let mut right = counter_1;
    let mut round_key = key;
    for round_index in 0..PHILOX_ROUNDS {
        let product = (PHILOX_M2X64_0 as u128) * (left as u128);
        let high = (product >> 64) as u64;
        let low = product as u64;
        left = high ^ round_key ^ right;
        right = low;
        if round_index != PHILOX_ROUNDS - 1 {
            round_key = round_key.wrapping_add(PHILOX_W64_0);
        }
    }
    (left, right)
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
    /// [`can_fit_run`]; `scratch` is an equally reserved working ledger.
    fn merge_run_in_place(&mut self, run: &[u64], flag: u8, scratch: &mut Self) {
        scratch.clear();
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.gids.len() && j < run.len() {
            match self.gids[i].cmp(&run[j]) {
                Ordering::Less => {
                    scratch.gids.push(self.gids[i]);
                    scratch.flags.push(self.flags[i]);
                    i += 1;
                }
                Ordering::Greater => {
                    scratch.gids.push(run[j]);
                    scratch.flags.push(flag);
                    j += 1;
                }
                Ordering::Equal => {
                    scratch.gids.push(self.gids[i]);
                    scratch.flags.push(self.flags[i] | flag);
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < self.gids.len() {
            scratch.gids.push(self.gids[i]);
            scratch.flags.push(self.flags[i]);
            i += 1;
        }
        while j < run.len() {
            scratch.gids.push(run[j]);
            scratch.flags.push(flag);
            j += 1;
        }
        std::mem::swap(self, scratch);
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
    ledger_scratch: IntervalLedger,
    gid_run: Vec<u64>,
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
            ledger_scratch: IntervalLedger::try_new(config.interval_distinct_capacity)?,
            gid_run: reserved_vec(config.resident_capacity)?,
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
            + self.ledger_scratch.gids.capacity() * size_of::<u64>()
            + self.ledger_scratch.flags.capacity() * size_of::<u8>()
            + self.gid_run.capacity() * size_of::<u64>()
    }
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
    /// Cells materialized by the current step's deliveries. Kept separate so
    /// an insertion costs O(1) instead of shifting the sorted array; they are
    /// merged back during the step's fused update pass.
    staged: Vec<CellState>,
    /// Pending future events. Slots `[0..pending_head)` are consumed; the
    /// live queue is `[pending_head..)` in canonical ascending key order.
    pending: Vec<ProceduralEventRecord>,
    pending_head: usize,
    last_spikes: Vec<ProceduralSpikeRecord>,
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
        let residents = reserved_vec(config.resident_capacity)?;
        let staged_capacity = config.resident_capacity.min(config.event_capacity);
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
            pending,
            pending_head: 0,
            last_spikes,
            scratch,
            counters: ProceduralCounters::default(),
            poisoned: PROCEDURAL_OK,
        })
    }

    fn payload_bytes(&self) -> usize {
        size_of::<Self>()
            + self.residents.capacity() * size_of::<CellState>()
            + self.staged.capacity() * size_of::<CellState>()
            + self.pending.capacity() * size_of::<ProceduralEventRecord>()
            + self.last_spikes.capacity() * size_of::<ProceduralSpikeRecord>()
            + self.scratch.payload_bytes()
    }

    fn poison(&mut self, code: u32) -> u32 {
        self.poisoned = code;
        self.counters.overflow_code = code;
        code
    }

    /// Materialized cells at this instant, including the current step's
    /// staged insertions.
    #[inline]
    fn total_residents(&self) -> usize {
        self.residents.len() + self.staged.len()
    }

    /// Fold the staged insertions back into the sorted resident array,
    /// restoring the "all residents in one ascending array" invariant used by
    /// the exact fallback path and by snapshots.
    ///
    /// The merge runs backwards over the grown array: the write cursor starts
    /// past every unmerged slot (`write == read + staged_index` is kept as an
    /// invariant), so no survivor is ever written out of bounds and no
    /// not-yet-read resident is ever clobbered.
    fn absorb_staged(&mut self) -> Result<(), u32> {
        if self.staged.is_empty() {
            return Ok(());
        }
        self.staged.sort_unstable_by_key(|state| state.gid);
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
        Ok(())
    }

    fn resident_index(&self, gid: u64) -> Result<usize, usize> {
        self.residents.binary_search_by_key(&gid, |state| state.gid)
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

    fn apply_event(&mut self, event: ProceduralEventRecord) -> Result<(), u32> {
        let current = canonical_zero(event.current_pa);
        match self.resident_index(event.target_gid) {
            Ok(index) => {
                let value = self.residents[index].synaptic_current + current;
                self.residents[index].synaptic_current = canonical_zero(value);
            }
            Err(_) => {
                if let Some(slot) = self
                    .staged
                    .iter()
                    .position(|state| state.gid == event.target_gid)
                {
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
                self.scratch.interval_ledger.mark_single(
                    event.target_gid,
                    SEEN_RESIDENT,
                    self.config.interval_distinct_capacity,
                )?;
                let mut state = CellState::canonical_rest(event.target_gid);
                state.synaptic_current = current;
                self.staged.push(state);
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
                let ledger_scratch = &mut self.scratch.ledger_scratch;
                ledger.merge_run_in_place(run, SEEN_RESIDENT, ledger_scratch);
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
        let mut resident_peak = self.residents.len();
        let mut queue_peak = self.pending_logical_len();
        let mut cell_steps = 0u64;
        let mut generated_edges = 0u64;
        let mut delivered_events = 0u64;

        for _ in 0..step_count {
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
                if let Err(code) = self.apply_event(event) {
                    return self.poison(code);
                }
            }
            self.pending_head += due_count;
            self.release_consumed_prefix();
            resident_peak = resident_peak.max(self.total_residents());

            self.scratch.step_spikes.clear();
            let population = self.total_residents();
            cell_steps = match cell_steps.checked_add(population as u64) {
                Some(value) => value,
                None => return self.poison(PROCEDURAL_STEP_OVERFLOW),
            };
            if population > 0 {
                // Ascending merged run of every GID stepped this step.
                self.staged.sort_unstable_by_key(|state| state.gid);
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
                    if let Err(code) = self.absorb_staged() {
                        return self.poison(code);
                    }
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
                            self.residents[write] = state;
                            write += 1;
                        }
                    }
                    debug_assert!(write <= self.residents.len());
                    self.residents.truncate(write);
                    {
                        // Pre-verified to fit: mark the whole stepped run.
                        let run = &self.scratch.gid_run;
                        let ledger = &mut self.scratch.interval_ledger;
                        let ledger_scratch = &mut self.scratch.ledger_scratch;
                        ledger.merge_run_in_place(run, SEEN_UPDATED, ledger_scratch);
                    }
                    if !self.scratch.step_spikes.is_empty() {
                        // Spiking GIDs are a subset of the stepped run just
                        // merged, so this never grows the ledger and can
                        // never trip the distinct capacity.
                        let run = &self.scratch.step_spikes;
                        let ledger = &mut self.scratch.interval_ledger;
                        let ledger_scratch = &mut self.scratch.ledger_scratch;
                        ledger.merge_run_in_place(run, SEEN_SPIKING, ledger_scratch);
                    }
                } else {
                    // Exact fallback near the distinct capacity: materialize
                    // the staged insertions into the sorted array, then run
                    // the original per-cell loop whose abort timing matches
                    // the previous implementation bit for bit.
                    if let Err(code) = self.absorb_staged() {
                        return self.poison(code);
                    }
                    for state in &mut self.residents {
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
                    }
                    self.residents.retain(|state| !state.is_canonical_rest());
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
        fallback_candidate, fill_targets_with, humanbrain_procedural_runtime_create, GlifConfig,
        ProceduralConfig,
    };
    use std::collections::HashSet;

    #[test]
    fn fallback_candidate_matches_widened_counter64_oracle_across_u64_wrap() {
        assert_eq!(fallback_candidate(u64::MAX - 1, 3, u64::MAX), 2);
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
}
