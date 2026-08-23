//! Native simulation kernels for the fitted human Allen GLIF5 cells.
//!
//! The public surface is deliberately a small C ABI so Python remains
//! orchestration only. The established GLIF and recurrent entry points are
//! stateless kernels over caller-owned NumPy arrays. The procedural entry points
//! instead return opaque handles that own bounded cell, event, spike, and scratch
//! state until explicit destruction.

use rayon::prelude::*;
use std::{slice, time::Instant};

mod procedural;

const MIN_PARALLEL_CHUNK_CELLS: usize = 4_096;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GlifConfig {
    pub cell_count: usize,
    pub after_channel_count: usize,
    pub spike_cut_steps: i32,
    pub leak_ns: f64,
    pub voltage_step_scale: f64,
    pub threshold_beta_scale: f64,
    pub threshold_exp_decay: f64,
    pub threshold_voltage_factor: f64,
    pub threshold_beta_factor: f64,
    pub spike_threshold_decay: f64,
    pub threshold_inf_relative_mv: f64,
    pub voltage_reset_scale: f64,
    pub voltage_reset_offset_mv: f64,
    pub spike_cut_threshold_decay: f64,
    pub threshold_spike_increment_mv: f64,
}

#[repr(C)]
pub struct RecurrentConfig {
    pub token_steps: usize,
    pub pulse_steps: usize,
    pub input_fanout: usize,
    pub recurrent_fanout: usize,
    pub input_current_pa: f64,
    pub recurrent_decay: f64,
    // R0 multi-timescale trace bank: per-row step decays. Row k of the
    // eligibility buffer (elements [k*cells, (k+1)*cells)) decays by
    // eligibility_decays[k] every sub-step, in fixed index order.
    pub trace_count: usize,
    pub eligibility_decays: *const f64,
    // R0-b family A (spike-count windows): u32 per-cell accumulator for the
    // CURRENT token, cleared once at token start and folded at EXACTLY the
    // existing mask-write sites (`counts[cell] += crossed`). No new pass over
    // cells; no change to spike determination or any trace row. Null skips
    // accumulation entirely (legacy callers stay bit-identical).
    pub token_spike_counts: *mut u32,
}

#[inline]
#[allow(clippy::too_many_arguments)] // Flat slices keep the inner loop allocation-free.
unsafe fn glif_step_collect(
    config: &GlifConfig,
    after_decay: &[f64],
    after_cut_decay: &[f64],
    after_amplitudes: &[f64],
    current: &[f64],
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    mut spike_indices: Option<&mut [i32]>,
    mut spike_mask: Option<&mut [u8]>,
    // R0-b family A: absolute-indexed per-cell count accumulator for the
    // current token (same length as `voltage`); folded at the mask-write site.
    mut token_spike_counts: Option<&mut [u32]>,
) -> usize {
    let mut spike_count = 0usize;
    let channel_count = config.after_channel_count;

    for cell in 0..config.cell_count {
        if refractory[cell] != 0 {
            refractory[cell] -= 1;
            if let Some(mask) = spike_mask.as_deref_mut() {
                mask[cell] = 0;
            }
            continue;
        }

        let after_offset = cell * channel_count;
        let mut after_sum = 0.0f64;
        for channel in 0..channel_count {
            after_sum += after[after_offset + channel];
        }
        let total_current = current[cell] + after_sum;
        let previous_voltage = voltage[cell];
        let leak = config.leak_ns * previous_voltage;
        let voltage_delta = config.voltage_step_scale * (total_current - leak);
        let next_voltage = previous_voltage + voltage_delta;

        let beta = total_current * config.threshold_beta_scale;
        let voltage_term = config.threshold_voltage_factor * previous_voltage;
        let beta_term = config.threshold_beta_factor * beta;
        let decayed_voltage_threshold = config.threshold_exp_decay * threshold_voltage[cell];
        let next_voltage_threshold = decayed_voltage_threshold + voltage_term + beta_term;
        let next_spike_threshold = threshold_spike[cell] * config.spike_threshold_decay;

        voltage[cell] = next_voltage;
        threshold_voltage[cell] = next_voltage_threshold;
        threshold_spike[cell] = next_spike_threshold;
        for channel in 0..channel_count {
            after[after_offset + channel] *= after_decay[channel];
        }

        let total_threshold =
            config.threshold_inf_relative_mv + next_voltage_threshold + next_spike_threshold;
        let crossed = next_voltage > total_threshold;
        if let Some(mask) = spike_mask.as_deref_mut() {
            mask[cell] = u8::from(crossed);
        }
        if crossed {
            if let Some(counts) = token_spike_counts.as_deref_mut() {
                counts[cell] += 1;
            }
        }
        if !crossed {
            continue;
        }

        if let Some(indices) = spike_indices.as_deref_mut() {
            indices[spike_count] = cell as i32;
        }
        spike_count += 1;
        voltage[cell] = config.voltage_reset_scale * next_voltage + config.voltage_reset_offset_mv;
        for channel in 0..channel_count {
            let next_after = after[after_offset + channel];
            after[after_offset + channel] =
                after_amplitudes[channel] + next_after * after_cut_decay[channel];
        }
        threshold_spike[cell] = next_spike_threshold * config.spike_cut_threshold_decay
            + config.threshold_spike_increment_mv;
        refractory[cell] = config.spike_cut_steps;
    }
    spike_count
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn recurrent_cell_chunk_scalar(
    config: &GlifConfig,
    after_decay: &[f64],
    after_cut_decay: &[f64],
    after_amplitudes: &[f64],
    recurrent_decay: f64,
    decays: &[f64],
    cells: usize,
    cell_offset: usize,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
    // R0-b family A: chunk-relative count accumulator (same length as
    // `voltage`); folded at the existing mask-write site below.
    mut token_spike_counts: Option<&mut [u32]>,
) {
    let channel_count = config.after_channel_count;
    debug_assert_eq!(after.len(), voltage.len() * channel_count);
    debug_assert_eq!(eligibility.len(), decays.len() * cells);

    for local in 0..voltage.len() {
        let cell = cell_offset + local;
        let total_synaptic_current = synaptic_current[local];
        if refractory[local] != 0 {
            refractory[local] -= 1;
            spike_mask[local] = 0;
        } else {
            let after_offset = local * channel_count;
            let mut after_sum = 0.0f64;
            for channel in 0..channel_count {
                after_sum += after[after_offset + channel];
            }
            let total_current = total_synaptic_current + after_sum;
            let previous_voltage = voltage[local];
            let leak = config.leak_ns * previous_voltage;
            let voltage_delta = config.voltage_step_scale * (total_current - leak);
            let next_voltage = previous_voltage + voltage_delta;

            let beta = total_current * config.threshold_beta_scale;
            let voltage_term = config.threshold_voltage_factor * previous_voltage;
            let beta_term = config.threshold_beta_factor * beta;
            let decayed_voltage_threshold = config.threshold_exp_decay * threshold_voltage[local];
            let next_voltage_threshold = decayed_voltage_threshold + voltage_term + beta_term;
            let next_spike_threshold = threshold_spike[local] * config.spike_threshold_decay;

            voltage[local] = next_voltage;
            threshold_voltage[local] = next_voltage_threshold;
            threshold_spike[local] = next_spike_threshold;
            for channel in 0..channel_count {
                after[after_offset + channel] *= after_decay[channel];
            }

            let total_threshold =
                config.threshold_inf_relative_mv + next_voltage_threshold + next_spike_threshold;
            let crossed = next_voltage > total_threshold;
            spike_mask[local] = u8::from(crossed);
            if crossed {
                if let Some(counts) = token_spike_counts.as_deref_mut() {
                    counts[local] += 1;
                }
            }
            if crossed {
                voltage[local] =
                    config.voltage_reset_scale * next_voltage + config.voltage_reset_offset_mv;
                for channel in 0..channel_count {
                    let next_after = after[after_offset + channel];
                    after[after_offset + channel] =
                        after_amplitudes[channel] + next_after * after_cut_decay[channel];
                }
                threshold_spike[local] = next_spike_threshold * config.spike_cut_threshold_decay
                    + config.threshold_spike_increment_mv;
                refractory[local] = config.spike_cut_steps;
            }
        }

        // These independent decays originally formed a second full-array pass.
        // Keeping their arithmetic unchanged while fusing the pass removes
        // bytes of memory traffic per cell and biological integration step.
        // Every trace row decays in fixed index order 0..trace_count; no row
        // is ever derived from another (frozen protocol §2 requirement 2).
        synaptic_current[local] *= recurrent_decay;
        for (row, &decay) in decays.iter().enumerate() {
            eligibility[row * cells + cell] *= decay;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn recurrent_cell_chunk_avx2(
    config: &GlifConfig,
    after_decay: &[f64],
    after_cut_decay: &[f64],
    after_amplitudes: &[f64],
    recurrent_decay: f64,
    decays: &[f64],
    cells: usize,
    cell_offset: usize,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
    // R0-b family A: chunk-relative count accumulator, folded per crossed
    // lane inside the existing mask-write lane loop.
    mut token_spike_counts: Option<&mut [u32]>,
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(config.after_channel_count, 2);
    debug_assert_eq!(eligibility.len(), decays.len() * cells);
    let width = 4usize;
    let mut cell = 0usize;
    let leak_ns = _mm256_set1_pd(config.leak_ns);
    let voltage_step_scale = _mm256_set1_pd(config.voltage_step_scale);
    let threshold_beta_scale = _mm256_set1_pd(config.threshold_beta_scale);
    let threshold_exp_decay = _mm256_set1_pd(config.threshold_exp_decay);
    let threshold_voltage_factor = _mm256_set1_pd(config.threshold_voltage_factor);
    let threshold_beta_factor = _mm256_set1_pd(config.threshold_beta_factor);
    let spike_threshold_decay = _mm256_set1_pd(config.spike_threshold_decay);
    let threshold_inf = _mm256_set1_pd(config.threshold_inf_relative_mv);
    let after_decay_0 = _mm256_set1_pd(after_decay[0]);
    let after_decay_1 = _mm256_set1_pd(after_decay[1]);
    let recurrent_decay = _mm256_set1_pd(recurrent_decay);

    while cell + width <= voltage.len() {
        if refractory[cell..cell + width]
            .iter()
            .any(|&remaining| remaining != 0)
        {
            recurrent_cell_chunk_scalar(
                config,
                after_decay,
                after_cut_decay,
                after_amplitudes,
                _mm256_cvtsd_f64(recurrent_decay),
                decays,
                cells,
                cell_offset + cell,
                &mut voltage[cell..cell + width],
                &mut after[cell * 2..(cell + width) * 2],
                &mut threshold_spike[cell..cell + width],
                &mut threshold_voltage[cell..cell + width],
                &mut refractory[cell..cell + width],
                &mut synaptic_current[cell..cell + width],
                eligibility,
                &mut spike_mask[cell..cell + width],
                token_spike_counts.as_deref_mut(),
            );
            cell += width;
            continue;
        }

        let previous_voltage = _mm256_loadu_pd(voltage.as_ptr().add(cell));
        let current = _mm256_loadu_pd(synaptic_current.as_ptr().add(cell));
        let after_low = _mm256_loadu_pd(after.as_ptr().add(cell * 2));
        let after_high = _mm256_loadu_pd(after.as_ptr().add(cell * 2 + width));
        let unpacked_after_0 = _mm256_unpacklo_pd(after_low, after_high);
        let unpacked_after_1 = _mm256_unpackhi_pd(after_low, after_high);
        let after_0 = _mm256_permute4x64_pd::<0xd8>(unpacked_after_0);
        let after_1 = _mm256_permute4x64_pd::<0xd8>(unpacked_after_1);
        let after_sum = _mm256_add_pd(_mm256_add_pd(_mm256_setzero_pd(), after_0), after_1);
        let total_current = _mm256_add_pd(current, after_sum);
        let leak = _mm256_mul_pd(leak_ns, previous_voltage);
        let voltage_delta = _mm256_mul_pd(voltage_step_scale, _mm256_sub_pd(total_current, leak));
        let next_voltage = _mm256_add_pd(previous_voltage, voltage_delta);

        let beta = _mm256_mul_pd(total_current, threshold_beta_scale);
        let voltage_term = _mm256_mul_pd(threshold_voltage_factor, previous_voltage);
        let beta_term = _mm256_mul_pd(threshold_beta_factor, beta);
        let previous_voltage_threshold = _mm256_loadu_pd(threshold_voltage.as_ptr().add(cell));
        let decayed_voltage_threshold =
            _mm256_mul_pd(threshold_exp_decay, previous_voltage_threshold);
        let next_voltage_threshold = _mm256_add_pd(
            _mm256_add_pd(decayed_voltage_threshold, voltage_term),
            beta_term,
        );
        let previous_spike_threshold = _mm256_loadu_pd(threshold_spike.as_ptr().add(cell));
        let next_spike_threshold = _mm256_mul_pd(previous_spike_threshold, spike_threshold_decay);
        let next_after_0 = _mm256_mul_pd(after_0, after_decay_0);
        let next_after_1 = _mm256_mul_pd(after_1, after_decay_1);

        _mm256_storeu_pd(voltage.as_mut_ptr().add(cell), next_voltage);
        _mm256_storeu_pd(
            threshold_voltage.as_mut_ptr().add(cell),
            next_voltage_threshold,
        );
        _mm256_storeu_pd(threshold_spike.as_mut_ptr().add(cell), next_spike_threshold);
        let interleaved_after_low = _mm256_permute2f128_pd::<0x20>(
            _mm256_unpacklo_pd(next_after_0, next_after_1),
            _mm256_unpackhi_pd(next_after_0, next_after_1),
        );
        let interleaved_after_high = _mm256_permute2f128_pd::<0x31>(
            _mm256_unpacklo_pd(next_after_0, next_after_1),
            _mm256_unpackhi_pd(next_after_0, next_after_1),
        );
        _mm256_storeu_pd(after.as_mut_ptr().add(cell * 2), interleaved_after_low);
        _mm256_storeu_pd(
            after.as_mut_ptr().add(cell * 2 + width),
            interleaved_after_high,
        );

        let total_threshold = _mm256_add_pd(
            _mm256_add_pd(threshold_inf, next_voltage_threshold),
            next_spike_threshold,
        );
        let crossed = _mm256_cmp_pd::<_CMP_GT_OQ>(next_voltage, total_threshold);
        let crossed_bits = _mm256_movemask_pd(crossed) as u8;

        let decayed_current = _mm256_mul_pd(current, recurrent_decay);
        _mm256_storeu_pd(synaptic_current.as_mut_ptr().add(cell), decayed_current);
        // Every trace row decays in fixed index order with its own broadcast
        // constant; identical per-row op order to the scalar kernel.
        for (row, &decay) in decays.iter().enumerate() {
            let row_offset = cell_offset + row * cells + cell;
            let previous_eligibility = _mm256_loadu_pd(eligibility.as_ptr().add(row_offset));
            let decayed_eligibility = _mm256_mul_pd(previous_eligibility, _mm256_set1_pd(decay));
            _mm256_storeu_pd(
                eligibility.as_mut_ptr().add(row_offset),
                decayed_eligibility,
            );
        }

        for lane in 0..width {
            let crossed = (crossed_bits >> lane) & 1;
            spike_mask[cell + lane] = crossed;
            if crossed == 0 {
                continue;
            }
            if let Some(counts) = token_spike_counts.as_deref_mut() {
                counts[cell + lane] += 1;
            }
            let index = cell + lane;
            voltage[index] =
                config.voltage_reset_scale * voltage[index] + config.voltage_reset_offset_mv;
            let after_offset = index * 2;
            after[after_offset] = after_amplitudes[0] + after[after_offset] * after_cut_decay[0];
            after[after_offset + 1] =
                after_amplitudes[1] + after[after_offset + 1] * after_cut_decay[1];
            threshold_spike[index] = threshold_spike[index] * config.spike_cut_threshold_decay
                + config.threshold_spike_increment_mv;
            refractory[index] = config.spike_cut_steps;
        }
        cell += width;
    }

    if cell < voltage.len() {
        recurrent_cell_chunk_scalar(
            config,
            after_decay,
            after_cut_decay,
            after_amplitudes,
            _mm256_cvtsd_f64(recurrent_decay),
            decays,
            cells,
            cell_offset + cell,
            &mut voltage[cell..],
            &mut after[cell * 2..],
            &mut threshold_spike[cell..],
            &mut threshold_voltage[cell..],
            &mut refractory[cell..],
            &mut synaptic_current[cell..],
            eligibility,
            &mut spike_mask[cell..],
            token_spike_counts.as_deref_mut(),
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn recurrent_cell_chunk(
    config: &GlifConfig,
    after_decay: &[f64],
    after_cut_decay: &[f64],
    after_amplitudes: &[f64],
    recurrent_decay: f64,
    decays: &[f64],
    cells: usize,
    cell_offset: usize,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
    token_spike_counts: Option<&mut [u32]>,
) {
    #[cfg(target_arch = "x86_64")]
    if config.after_channel_count == 2 && std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            recurrent_cell_chunk_avx2(
                config,
                after_decay,
                after_cut_decay,
                after_amplitudes,
                recurrent_decay,
                decays,
                cells,
                cell_offset,
                voltage,
                after,
                threshold_spike,
                threshold_voltage,
                refractory,
                synaptic_current,
                eligibility,
                spike_mask,
                token_spike_counts,
            );
        }
        return;
    }

    recurrent_cell_chunk_scalar(
        config,
        after_decay,
        after_cut_decay,
        after_amplitudes,
        recurrent_decay,
        decays,
        cells,
        cell_offset,
        voltage,
        after,
        threshold_spike,
        threshold_voltage,
        refractory,
        synaptic_current,
        eligibility,
        spike_mask,
        token_spike_counts,
    );
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn recurrent_step_serial(
    config: &GlifConfig,
    after_decay: &[f64],
    after_cut_decay: &[f64],
    after_amplitudes: &[f64],
    recurrent_decay: f64,
    decays: &[f64],
    cells: usize,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
    token_spike_counts: Option<&mut [u32]>,
) {
    recurrent_cell_chunk(
        config,
        after_decay,
        after_cut_decay,
        after_amplitudes,
        recurrent_decay,
        decays,
        cells,
        0,
        voltage,
        after,
        threshold_spike,
        threshold_voltage,
        refractory,
        synaptic_current,
        eligibility,
        spike_mask,
        token_spike_counts,
    );
}

/// Raw alias of the trace bank handed to worker threads. Soundness: chunk
/// ranges partition `[0, cells)` and rows partition `[0, trace_count * cells)`,
/// so every `(row, chunk)` write range is disjoint across threads.
#[derive(Clone, Copy)]
struct TraceBankAlias(*mut f64, usize);

impl TraceBankAlias {
    fn parts(self) -> (*mut f64, usize) {
        (self.0, self.1)
    }
}

unsafe impl Send for TraceBankAlias {}
unsafe impl Sync for TraceBankAlias {}

/// Raw alias of the family-A count accumulator handed to worker threads.
/// Soundness: chunk ranges partition `[0, cells)`, so count writes are disjoint.
#[derive(Clone, Copy)]
struct CountBankAlias(*mut u32, usize);

impl CountBankAlias {
    fn parts(self) -> (*mut u32, usize) {
        (self.0, self.1)
    }
}

unsafe impl Send for CountBankAlias {}
unsafe impl Sync for CountBankAlias {}

#[inline]
#[allow(clippy::too_many_arguments)]
fn recurrent_step_parallel(
    config: &GlifConfig,
    after_decay: &[f64],
    after_cut_decay: &[f64],
    after_amplitudes: &[f64],
    recurrent_decay: f64,
    decays: &[f64],
    cells: usize,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
    token_spike_counts: Option<&mut [u32]>,
) {
    let channels = config.after_channel_count;
    let target_chunks = rayon::current_num_threads() * 4;
    let chunk_cells = cells.div_ceil(target_chunks).max(MIN_PARALLEL_CHUNK_CELLS);
    let bank = TraceBankAlias(eligibility.as_mut_ptr(), decays.len() * cells);
    let counts_bank =
        token_spike_counts.map(|counts| CountBankAlias(counts.as_mut_ptr(), counts.len()));

    voltage
        .par_chunks_mut(chunk_cells)
        .zip(after.par_chunks_mut(chunk_cells * channels))
        .zip(threshold_spike.par_chunks_mut(chunk_cells))
        .zip(threshold_voltage.par_chunks_mut(chunk_cells))
        .zip(refractory.par_chunks_mut(chunk_cells))
        .zip(synaptic_current.par_chunks_mut(chunk_cells))
        .zip(spike_mask.par_chunks_mut(chunk_cells))
        .enumerate()
        .for_each(
            move |(
                chunk_index,
                (
                    (
                        ((((voltage, after), threshold_spike), threshold_voltage), refractory),
                        current,
                    ),
                    mask,
                ),
            )| {
                let start = chunk_index * chunk_cells;
                let (bank_ptr, bank_len) = bank.parts();
                // SAFETY: chunk ranges partition [0, cells) and rows partition
                // the trace bank, so all (row, chunk) write ranges are disjoint.
                let bank_slice = unsafe { slice::from_raw_parts_mut(bank_ptr, bank_len) };
                let counts_slice = match counts_bank {
                    Some(alias) => {
                        let (ptr, len) = alias.parts();
                        assert_eq!(len, cells);
                        // SAFETY: chunk ranges partition the accumulator.
                        Some(unsafe {
                            slice::from_raw_parts_mut(
                                ptr.add(start),
                                chunk_cells.min(len.saturating_sub(start)),
                            )
                        })
                    }
                    None => None,
                };
                recurrent_cell_chunk(
                    config,
                    after_decay,
                    after_cut_decay,
                    after_amplitudes,
                    recurrent_decay,
                    decays,
                    cells,
                    start,
                    voltage,
                    after,
                    threshold_spike,
                    threshold_voltage,
                    refractory,
                    current,
                    bank_slice,
                    mask,
                    counts_slice,
                );
            },
        );
}

#[derive(Clone, Copy)]
enum Execution {
    Serial,
    Parallel,
    Frontier,
}

#[allow(clippy::too_many_arguments)]
fn observe_token_frontier(
    glif: &GlifConfig,
    recurrent: &RecurrentConfig,
    after_decay: &[f64],
    after_cut_decay: &[f64],
    after_amplitudes: &[f64],
    afferents: &[i32],
    targets: &[i32],
    weights: &[f64],
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    input_scratch: &mut [f64],
    spike_mask: &mut [u8],
    mut token_spike_counts: Option<&mut [u32]>,
    active_indices: &mut [i32],
    active_mask: &mut [u8],
) -> (usize, usize, u64, u64, u64) {
    let cells = glif.cell_count;
    let channels = glif.after_channel_count;
    // R0-b family A: the accumulator is cleared at TOKEN start and folded at
    // the existing mask-write sites; frontier/activity accounting never reads
    // it (frozen protocol §3 requirement 5).
    if let Some(counts) = token_spike_counts.as_deref_mut() {
        counts.fill(0);
    }
    let decays =
        unsafe { slice::from_raw_parts(recurrent.eligibility_decays, recurrent.trace_count) };
    debug_assert_eq!(eligibility.len(), decays.len() * cells);
    let trace_count = decays.len();
    let scan_started = Instant::now();
    active_mask.fill(0);
    let mut active_count = 0usize;
    for cell in 0..cells {
        let after_offset = cell * channels;
        let mut active = voltage[cell] != 0.0
            || threshold_spike[cell] != 0.0
            || threshold_voltage[cell] != 0.0
            || refractory[cell] != 0
            || synaptic_current[cell] != 0.0;
        if !active {
            // A cell is frontier-active if ANY trace row is nonzero: a cell
            // quiet in row 0 but not row k must not be dropped (frozen
            // protocol §2 requirement 6).
            for row in 0..trace_count {
                active |= eligibility[row * cells + cell] != 0.0;
            }
        }
        if !active {
            for channel in 0..channels {
                active |= after[after_offset + channel] != 0.0;
            }
        }
        if active {
            active_mask[cell] = 1;
            active_indices[active_count] = cell as i32;
            active_count += 1;
        }
    }
    let scan_nanoseconds = scan_started.elapsed().as_nanos() as u64;
    let build_started = Instant::now();
    for &target in afferents {
        let target = target as usize;
        if active_mask[target] == 0 {
            active_mask[target] = 1;
            active_indices[active_count] = target as i32;
            active_count += 1;
        }
    }
    active_indices[..active_count].sort_unstable();
    let build_nanoseconds = build_started.elapsed().as_nanos() as u64;

    let steps_started = Instant::now();
    let mut total_spikes = 0usize;
    for step in 0..recurrent.token_steps {
        let pulse_active = step < recurrent.pulse_steps;
        if pulse_active {
            for (slot, &target) in afferents.iter().enumerate() {
                let target = target as usize;
                input_scratch[slot] = synaptic_current[target];
                synaptic_current[target] += recurrent.input_current_pa;
            }
        }

        let step_active_count = active_count;
        for &cell in &active_indices[..step_active_count] {
            let cell = cell as usize;
            let counts_slice = match token_spike_counts.as_deref_mut() {
                Some(counts) => Some(&mut counts[cell..cell + 1]),
                None => None,
            };
            recurrent_cell_chunk_scalar(
                glif,
                after_decay,
                after_cut_decay,
                after_amplitudes,
                recurrent.recurrent_decay,
                decays,
                cells,
                cell,
                &mut voltage[cell..cell + 1],
                &mut after[cell * channels..(cell + 1) * channels],
                &mut threshold_spike[cell..cell + 1],
                &mut threshold_voltage[cell..cell + 1],
                &mut refractory[cell..cell + 1],
                &mut synaptic_current[cell..cell + 1],
                eligibility,
                &mut spike_mask[cell..cell + 1],
                counts_slice,
            );
        }

        if pulse_active {
            for slot in (0..afferents.len()).rev() {
                synaptic_current[afferents[slot] as usize] = input_scratch[slot];
            }
            for (slot, &target) in afferents.iter().enumerate() {
                if afferents[..slot].contains(&target) {
                    continue;
                }
                synaptic_current[target as usize] *= recurrent.recurrent_decay;
            }
        }

        for slot in 0..step_active_count {
            let presynaptic = active_indices[slot] as usize;
            if spike_mask[presynaptic] == 0 {
                continue;
            }
            total_spikes += 1;
            // One spike event increments EVERY trace row once (never per
            // fanout edge; same event set as the incumbent serial kernel).
            for row in 0..trace_count {
                eligibility[row * cells + presynaptic] += 1.0;
            }
            let edge_offset = presynaptic * recurrent.recurrent_fanout;
            for edge in 0..recurrent.recurrent_fanout {
                let target = targets[edge_offset + edge] as usize;
                synaptic_current[target] += weights[edge_offset + edge];
                if active_mask[target] == 0 {
                    active_mask[target] = 1;
                    active_indices[active_count] = target as i32;
                    active_count += 1;
                }
            }
        }
        if active_count != step_active_count {
            active_indices[..active_count].sort_unstable();
        }
    }
    (
        total_spikes,
        active_count,
        scan_nanoseconds,
        build_nanoseconds,
        steps_started.elapsed().as_nanos() as u64,
    )
}

/// Advance a population by exactly one fitted integration step.
///
/// # Safety
/// Every pointer must address a contiguous array of the length encoded in
/// `config`; mutable arrays must not alias one another.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_glif_step(
    config: *const GlifConfig,
    after_decay: *const f64,
    after_cut_decay: *const f64,
    after_amplitudes: *const f64,
    current: *const f64,
    voltage: *mut f64,
    after: *mut f64,
    threshold_spike: *mut f64,
    threshold_voltage: *mut f64,
    refractory: *mut i32,
    spike_mask: *mut u8,
) -> usize {
    if config.is_null() {
        return 0;
    }
    let config = &*config;
    let cells = config.cell_count;
    let channels = config.after_channel_count;
    glif_step_collect(
        config,
        slice::from_raw_parts(after_decay, channels),
        slice::from_raw_parts(after_cut_decay, channels),
        slice::from_raw_parts(after_amplitudes, channels),
        slice::from_raw_parts(current, cells),
        slice::from_raw_parts_mut(voltage, cells),
        slice::from_raw_parts_mut(after, cells * channels),
        slice::from_raw_parts_mut(threshold_spike, cells),
        slice::from_raw_parts_mut(threshold_voltage, cells),
        slice::from_raw_parts_mut(refractory, cells),
        None,
        Some(slice::from_raw_parts_mut(spike_mask, cells)),
        None,
    )
}

/// Apply one token and execute the complete sparse recurrent simulation.
///
/// All integration steps, refractory updates, eligibility dynamics, and
/// event scatter stay inside this call. Recurrent work is O(cells +
/// spikes*fanout) per step and never allocates a dense cells-by-cells matrix.
///
/// # Safety
/// Every pointer must address a contiguous array of the length encoded in the
/// two configs; mutable arrays must not alias one another.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_observe_token_reference_v1(
    glif_config: *const GlifConfig,
    recurrent_config: *const RecurrentConfig,
    after_decay: *const f64,
    after_cut_decay: *const f64,
    after_amplitudes: *const f64,
    afferents: *const i32,
    recurrent_targets: *const i32,
    recurrent_weights: *const f64,
    voltage: *mut f64,
    after: *mut f64,
    threshold_spike: *mut f64,
    threshold_voltage: *mut f64,
    refractory: *mut i32,
    synaptic_current: *mut f64,
    eligibility: *mut f64,
    step_current: *mut f64,
    spike_indices: *mut i32,
) -> usize {
    if glif_config.is_null() || recurrent_config.is_null() {
        return 0;
    }
    let glif = &*glif_config;
    let recurrent = &*recurrent_config;
    let cells = glif.cell_count;
    let channels = glif.after_channel_count;
    let decays = slice::from_raw_parts(recurrent.eligibility_decays, recurrent.trace_count);
    let trace_count = decays.len();
    let after_decay = slice::from_raw_parts(after_decay, channels);
    let after_cut_decay = slice::from_raw_parts(after_cut_decay, channels);
    let after_amplitudes = slice::from_raw_parts(after_amplitudes, channels);
    let afferents = slice::from_raw_parts(afferents, recurrent.input_fanout);
    let targets = slice::from_raw_parts(recurrent_targets, cells * recurrent.recurrent_fanout);
    let weights = slice::from_raw_parts(recurrent_weights, cells * recurrent.recurrent_fanout);
    let voltage = slice::from_raw_parts_mut(voltage, cells);
    let after = slice::from_raw_parts_mut(after, cells * channels);
    let threshold_spike = slice::from_raw_parts_mut(threshold_spike, cells);
    let threshold_voltage = slice::from_raw_parts_mut(threshold_voltage, cells);
    let refractory = slice::from_raw_parts_mut(refractory, cells);
    let synaptic_current = slice::from_raw_parts_mut(synaptic_current, cells);
    let eligibility = slice::from_raw_parts_mut(eligibility, trace_count * cells);
    let step_current = slice::from_raw_parts_mut(step_current, cells);
    let spike_indices = slice::from_raw_parts_mut(spike_indices, cells);
    // R0-b family A: same cleared-at-token-start accumulator contract as the
    // optimized kernels; null pointer (legacy callers) skips accumulation.
    let mut token_spike_counts = if recurrent.token_spike_counts.is_null() {
        None
    } else {
        Some(slice::from_raw_parts_mut(
            recurrent.token_spike_counts,
            cells,
        ))
    };
    if let Some(counts) = token_spike_counts.as_deref_mut() {
        counts.fill(0);
    }

    let mut total_spikes = 0usize;
    for step in 0..recurrent.token_steps {
        step_current.copy_from_slice(synaptic_current);
        if step < recurrent.pulse_steps {
            for &target in afferents {
                step_current[target as usize] += recurrent.input_current_pa;
            }
        }

        let spike_count = glif_step_collect(
            glif,
            after_decay,
            after_cut_decay,
            after_amplitudes,
            step_current,
            voltage,
            after,
            threshold_spike,
            threshold_voltage,
            refractory,
            Some(spike_indices),
            None,
            token_spike_counts.as_deref_mut(),
        );
        total_spikes += spike_count;

        for cell in 0..cells {
            synaptic_current[cell] *= recurrent.recurrent_decay;
            for (row, &decay) in decays.iter().enumerate() {
                eligibility[row * cells + cell] *= decay;
            }
        }
        for &presynaptic in &spike_indices[..spike_count] {
            let presynaptic = presynaptic as usize;
            for row in 0..trace_count {
                eligibility[row * cells + presynaptic] += 1.0;
            }
            let edge_offset = presynaptic * recurrent.recurrent_fanout;
            for edge in 0..recurrent.recurrent_fanout {
                let target = targets[edge_offset + edge] as usize;
                synaptic_current[target] += weights[edge_offset + edge];
            }
        }
    }
    total_spikes
}

/// Apply one online competitive three-factor update to output-cell synapses.
///
/// The presynaptic eligibility trace is the first factor. The output cell
/// with the strongest current supplies the postsynaptic competitive factor.
/// The supervised target supplies a signed neuromodulatory third factor:
/// potentiate the target cell and depress the active competitor. The trace
/// norm is a postsynaptic homeostatic scalar, not learned state. Runtime is
/// O(outputs*cells), and the function allocates no statistics or workspace.
///
/// # Safety
/// Every pointer must address a contiguous array of the encoded length;
/// mutable arrays must not alias eligibility or heterogeneity.
#[no_mangle]
pub unsafe extern "C" fn humanbrain_three_factor_update(
    cell_count: usize,
    output_count: usize,
    target: usize,
    learning_rate_pa: f64,
    competitive_margin_pa: f64,
    eligibility: *const f64,
    output_heterogeneity_pa: *const f64,
    readout_synapses_pa: *mut f64,
    pacemaker_synapses_pa: *mut f64,
) -> usize {
    if cell_count == 0 || output_count < 2 || target >= output_count {
        return output_count;
    }
    let eligibility = slice::from_raw_parts(eligibility, cell_count);
    let heterogeneity = slice::from_raw_parts(output_heterogeneity_pa, output_count);
    let readout = slice::from_raw_parts_mut(readout_synapses_pa, output_count * cell_count);
    let pacemaker = slice::from_raw_parts_mut(pacemaker_synapses_pa, output_count);

    let output_current = |output: usize, readout: &[f64], pacemaker: &[f64]| {
        let offset = output * cell_count;
        let mut current = pacemaker[output] + heterogeneity[output];
        for cell in 0..cell_count {
            current += readout[offset + cell] * eligibility[cell];
        }
        current
    };
    let target_current = output_current(target, readout, pacemaker);
    let mut winner = usize::from(target == 0);
    let mut winner_current = output_current(winner, readout, pacemaker);
    for output in 0..output_count {
        if output == target {
            continue;
        }
        let current = output_current(output, readout, pacemaker);
        if current > winner_current {
            winner = output;
            winner_current = current;
        }
    }
    if target_current >= winner_current + competitive_margin_pa {
        return output_count;
    }

    let mut squared_trace_norm = 1.0f64;
    for &trace in eligibility {
        squared_trace_norm += trace * trace;
    }
    let step = learning_rate_pa / squared_trace_norm.sqrt();
    let target_offset = target * cell_count;
    let winner_offset = winner * cell_count;
    for cell in 0..cell_count {
        let change = step * eligibility[cell];
        readout[target_offset + cell] += change;
        readout[winner_offset + cell] -= change;
    }
    pacemaker[target] += step;
    pacemaker[winner] -= step;
    winner
}

/// Sum currents from a bounded set of explicit output-cell contacts.
///
/// # Safety
/// Every pointer must address a contiguous array with the encoded dimensions.
/// Contact source indices must be in `0..cell_count`, and mutable output must
/// not alias any input array.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn humanbrain_sparse_readout_current(
    cell_count: usize,
    output_count: usize,
    fan_in: usize,
    base_current_pa: f64,
    eligibility: *const f64,
    output_heterogeneity_pa: *const f64,
    contact_sources: *const i32,
    contact_weights_pa: *const f64,
    pacemaker_synapses_pa: *const f64,
    output_current_pa: *mut f64,
) {
    let eligibility = slice::from_raw_parts(eligibility, cell_count);
    let heterogeneity = slice::from_raw_parts(output_heterogeneity_pa, output_count);
    let sources = slice::from_raw_parts(contact_sources, output_count * fan_in);
    let weights = slice::from_raw_parts(contact_weights_pa, output_count * fan_in);
    let pacemaker = slice::from_raw_parts(pacemaker_synapses_pa, output_count);
    let output_current = slice::from_raw_parts_mut(output_current_pa, output_count);

    for output in 0..output_count {
        let mut current = base_current_pa;
        current += pacemaker[output];
        current += heterogeneity[output];
        let offset = output * fan_in;
        for slot in 0..fan_in {
            let source = sources[offset + slot] as usize;
            current += weights[offset + slot] * eligibility[source];
        }
        output_current[output] = current;
    }
}

/// Apply a supervised three-factor event to bounded output-cell contacts.
///
/// The stored apposition trace is pre x post: reservoir eligibility supplies
/// the presynaptic term and an actual natural/teacher-evoked output spike
/// supplies the postsynaptic gate. Signed cell-level modulators change only
/// existing contacts, whose excitatory strengths remain nonnegative.
///
/// # Safety
/// Every pointer must address a contiguous array with the encoded dimensions.
/// Potential/contact sources must be sorted, unique, in range, and each contact
/// source must occur in its output cell's potential-apposition row.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn humanbrain_sparse_three_factor_update(
    cell_count: usize,
    output_count: usize,
    candidate_count: usize,
    fan_in: usize,
    target: usize,
    winner: usize,
    learning_rate_pa: f64,
    maximum_weight_pa: f64,
    trace_decays: *const f64,
    target_post_gate: f64,
    winner_post_gate: f64,
    reservoir_eligibility: *const f64,
    potential_sources: *const i32,
    apposition_eligibility: *mut f64,
    contact_sources: *const i32,
    contact_weights_pa: *mut f64,
    contact_ages: *mut u32,
    pacemaker_synapses_pa: *mut f64,
    output_activity_ema: *mut f64,
    activity_decay: f64,
) -> usize {
    if output_count < 2
        || target >= output_count
        || winner >= output_count
        || !learning_rate_pa.is_finite()
        || learning_rate_pa <= 0.0
        || !maximum_weight_pa.is_finite()
        || maximum_weight_pa <= 0.0
        || !target_post_gate.is_finite()
        || !(0.0..=1.0).contains(&target_post_gate)
        || !winner_post_gate.is_finite()
        || !(0.0..=1.0).contains(&winner_post_gate)
        || !activity_decay.is_finite()
        || !(0.0..1.0).contains(&activity_decay)
    {
        return output_count + 1;
    }
    let reservoir_trace = slice::from_raw_parts(reservoir_eligibility, cell_count);
    let decays = slice::from_raw_parts(trace_decays, cell_count);
    let potential = slice::from_raw_parts(potential_sources, output_count * candidate_count);
    let apposition =
        slice::from_raw_parts_mut(apposition_eligibility, output_count * candidate_count);
    let contacts = slice::from_raw_parts(contact_sources, output_count * fan_in);
    let weights = slice::from_raw_parts_mut(contact_weights_pa, output_count * fan_in);
    let ages = slice::from_raw_parts_mut(contact_ages, output_count * fan_in);
    let pacemaker = slice::from_raw_parts_mut(pacemaker_synapses_pa, output_count);
    let output_activity = slice::from_raw_parts_mut(output_activity_ema, output_count);

    for age in ages.iter_mut() {
        *age = age.saturating_add(1);
    }
    for activity in output_activity.iter_mut() {
        *activity *= activity_decay;
    }
    output_activity[target] += (1.0 - activity_decay) * target_post_gate;
    if winner != target {
        output_activity[winner] += (1.0 - activity_decay) * winner_post_gate;
    }

    // Per-contact apposition decay: slot with source s decays by
    // exp(-T / tau_(s // R)) supplied through `decays` (R0 §1.3).
    for output in 0..output_count {
        let offset = output * candidate_count;
        for candidate in 0..candidate_count {
            let source = potential[offset + candidate] as usize;
            apposition[offset + candidate] *= decays[source];
        }
    }
    let add_post_event = |output: usize, gate: f64, apposition: &mut [f64]| {
        let offset = output * candidate_count;
        for candidate in 0..candidate_count {
            let source = potential[offset + candidate] as usize;
            apposition[offset + candidate] += reservoir_trace[source] * gate;
        }
    };
    add_post_event(target, target_post_gate, apposition);
    if winner != target {
        add_post_event(winner, winner_post_gate, apposition);
    }
    if winner == target {
        return output_count;
    }

    let mut update_contacts = |output: usize, modulation: f64| {
        let potential_offset = output * candidate_count;
        let contact_offset = output * fan_in;
        let mut candidate = 0usize;
        for slot in 0..fan_in {
            let weight = &mut weights[contact_offset + slot];
            if *weight == 0.0 {
                continue;
            }
            let source = contacts[contact_offset + slot];
            while potential[potential_offset + candidate] < source {
                candidate += 1;
            }
            let trace = apposition[potential_offset + candidate];
            *weight =
                (*weight + learning_rate_pa * modulation * trace).clamp(0.0, maximum_weight_pa);
        }
        pacemaker[output] = (pacemaker[output]
            + learning_rate_pa
                * modulation
                * if modulation > 0.0 {
                    target_post_gate
                } else {
                    winner_post_gate
                })
        .clamp(0.0, maximum_weight_pa);
    };
    update_contacts(target, 1.0);
    update_contacts(winner, -1.0);
    winner
}

fn sparse_turnover_uniform(seed: u64, event: u64, output: usize, stream: usize) -> f64 {
    let mut value = seed
        ^ event.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (output as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (stream as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

/// Run one slow, bounded structural/homeostatic update for one output cell.
///
/// Return bits: 1 = recruited one local potential apposition, 2 = deleted one
/// age-qualified contact. No call can recruit or delete more than one contact.
///
/// # Safety
/// Array dimensions and sorted/unique/subset invariants are the same as for
/// `humanbrain_sparse_three_factor_update`; contact arrays must be mutable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn humanbrain_sparse_structural_turnover(
    output_count: usize,
    candidate_count: usize,
    fan_in: usize,
    output: usize,
    structural_event: u64,
    random_seed: u64,
    minimum_age: u32,
    turnover_probability: f64,
    learning_rate_pa: f64,
    maximum_weight_pa: f64,
    homeostatic_rate: f64,
    potential_sources: *const i32,
    apposition_eligibility: *const f64,
    contact_sources: *mut i32,
    contact_weights_pa: *mut f64,
    contact_ages: *mut u32,
    output_activity_ema: *const f64,
) -> u32 {
    if output >= output_count
        || fan_in == 0
        || candidate_count < fan_in
        || !turnover_probability.is_finite()
        || !(0.0..=1.0).contains(&turnover_probability)
        || !learning_rate_pa.is_finite()
        || learning_rate_pa <= 0.0
        || !maximum_weight_pa.is_finite()
        || maximum_weight_pa <= 0.0
        || !homeostatic_rate.is_finite()
        || homeostatic_rate < 0.0
    {
        return 4;
    }
    let potential = slice::from_raw_parts(potential_sources, output_count * candidate_count);
    let apposition = slice::from_raw_parts(apposition_eligibility, output_count * candidate_count);
    let contacts = slice::from_raw_parts_mut(contact_sources, output_count * fan_in);
    let weights = slice::from_raw_parts_mut(contact_weights_pa, output_count * fan_in);
    let ages = slice::from_raw_parts_mut(contact_ages, output_count * fan_in);
    let activity = slice::from_raw_parts(output_activity_ema, output_count);
    let potential_offset = output * candidate_count;
    let contact_offset = output * fan_in;
    let row_sources = &mut contacts[contact_offset..contact_offset + fan_in];
    let row_weights = &mut weights[contact_offset..contact_offset + fan_in];
    let row_ages = &mut ages[contact_offset..contact_offset + fan_in];

    let target_activity = 1.0 / output_count as f64;
    let homeostatic_scale = (homeostatic_rate * (target_activity - activity[output])).exp();
    for weight in row_weights.iter_mut() {
        if *weight > 0.0 {
            *weight = (*weight * homeostatic_scale).clamp(0.0, maximum_weight_pa);
        }
    }

    let mut vacancy = row_weights.iter().position(|&weight| weight == 0.0);
    let mut deleted = false;
    if vacancy.is_none() {
        for slot in 0..fan_in {
            if row_ages[slot] < minimum_age {
                continue;
            }
            let weakness = 1.0 - (row_weights[slot] / maximum_weight_pa).clamp(0.0, 1.0);
            let hazard = turnover_probability * weakness;
            if sparse_turnover_uniform(random_seed, structural_event, output, slot) < hazard {
                vacancy = Some(slot);
                row_weights[slot] = 0.0;
                row_ages[slot] = 0;
                deleted = true;
                break;
            }
        }
    }
    let Some(vacancy) = vacancy else {
        return 0;
    };

    let is_available = |source: i32| match row_sources.binary_search(&source) {
        Ok(slot) => slot == vacancy,
        Err(_) => true,
    };
    let mut positive_total = 0.0;
    let mut available = 0usize;
    for candidate in 0..candidate_count {
        let source = potential[potential_offset + candidate];
        if is_available(source) {
            positive_total += apposition[potential_offset + candidate].max(0.0);
            available += 1;
        }
    }
    if available == 0 {
        return if deleted { 2 } else { 0 };
    }
    let use_uniform = positive_total == 0.0;
    let total = if use_uniform {
        available as f64
    } else {
        positive_total
    };
    let mut draw = sparse_turnover_uniform(
        random_seed ^ 0xa076_1d64_78bd_642f,
        structural_event,
        output,
        fan_in,
    ) * total;
    let mut selected_candidate = None;
    let mut last_selectable_candidate = None;
    for candidate in 0..candidate_count {
        let source = potential[potential_offset + candidate];
        if !is_available(source) {
            continue;
        }
        let score = if use_uniform {
            1.0
        } else {
            apposition[potential_offset + candidate].max(0.0)
        };
        if score > 0.0 {
            last_selectable_candidate = Some(candidate);
        }
        if draw < score {
            selected_candidate = Some(candidate);
            break;
        }
        draw -= score;
    }
    let candidate = selected_candidate.unwrap_or_else(|| {
        last_selectable_candidate.expect("selectable candidate total was checked")
    });
    let new_source = potential[potential_offset + candidate];
    let new_weight = (learning_rate_pa * apposition[potential_offset + candidate].max(1.0))
        .clamp(0.0, maximum_weight_pa);

    for slot in vacancy..fan_in - 1 {
        row_sources[slot] = row_sources[slot + 1];
        row_weights[slot] = row_weights[slot + 1];
        row_ages[slot] = row_ages[slot + 1];
    }
    let shortened = fan_in - 1;
    let mut insertion = 0usize;
    while insertion < shortened && row_sources[insertion] < new_source {
        insertion += 1;
    }
    for slot in (insertion..shortened).rev() {
        row_sources[slot + 1] = row_sources[slot];
        row_weights[slot + 1] = row_weights[slot];
        row_ages[slot + 1] = row_ages[slot];
    }
    row_sources[insertion] = new_source;
    row_weights[insertion] = new_weight;
    row_ages[insertion] = 0;
    1 | if deleted { 2 } else { 0 }
}

#[allow(clippy::too_many_arguments)]
unsafe fn observe_token_optimized(
    glif_config: *const GlifConfig,
    recurrent_config: *const RecurrentConfig,
    after_decay: *const f64,
    after_cut_decay: *const f64,
    after_amplitudes: *const f64,
    afferents: *const i32,
    recurrent_targets: *const i32,
    recurrent_weights: *const f64,
    voltage: *mut f64,
    after: *mut f64,
    threshold_spike: *mut f64,
    threshold_voltage: *mut f64,
    refractory: *mut i32,
    synaptic_current: *mut f64,
    eligibility: *mut f64,
    input_scratch: *mut f64,
    spike_mask: *mut u8,
    active_indices: *mut i32,
    active_mask: *mut u8,
    active_count_out: *mut usize,
    scan_nanoseconds_out: *mut u64,
    build_nanoseconds_out: *mut u64,
    step_nanoseconds_out: *mut u64,
    execution: Execution,
) -> usize {
    if glif_config.is_null() || recurrent_config.is_null() {
        return 0;
    }
    let glif = &*glif_config;
    let recurrent = &*recurrent_config;
    let cells = glif.cell_count;
    let channels = glif.after_channel_count;
    let decays =
        unsafe { slice::from_raw_parts(recurrent.eligibility_decays, recurrent.trace_count) };
    let trace_count = decays.len();
    let after_decay = slice::from_raw_parts(after_decay, channels);
    let after_cut_decay = slice::from_raw_parts(after_cut_decay, channels);
    let after_amplitudes = slice::from_raw_parts(after_amplitudes, channels);
    let afferents = slice::from_raw_parts(afferents, recurrent.input_fanout);
    let targets = slice::from_raw_parts(recurrent_targets, cells * recurrent.recurrent_fanout);
    let weights = slice::from_raw_parts(recurrent_weights, cells * recurrent.recurrent_fanout);
    let voltage = slice::from_raw_parts_mut(voltage, cells);
    let after = slice::from_raw_parts_mut(after, cells * channels);
    let threshold_spike = slice::from_raw_parts_mut(threshold_spike, cells);
    let threshold_voltage = slice::from_raw_parts_mut(threshold_voltage, cells);
    let refractory = slice::from_raw_parts_mut(refractory, cells);
    let synaptic_current = slice::from_raw_parts_mut(synaptic_current, cells);
    let eligibility = slice::from_raw_parts_mut(eligibility, trace_count * cells);
    let input_scratch = slice::from_raw_parts_mut(input_scratch, recurrent.input_fanout);
    let spike_mask = slice::from_raw_parts_mut(spike_mask, cells);
    let active_indices = slice::from_raw_parts_mut(active_indices, cells);
    let active_mask = slice::from_raw_parts_mut(active_mask, cells);
    // R0-b family A: cleared at TOKEN start, folded inside the chunk kernels
    // at exactly the mask-write sites; null pointer (legacy callers) skips.
    let mut token_spike_counts = if recurrent.token_spike_counts.is_null() {
        None
    } else {
        Some(slice::from_raw_parts_mut(
            recurrent.token_spike_counts,
            cells,
        ))
    };
    if let Some(counts) = token_spike_counts.as_deref_mut() {
        counts.fill(0);
    }

    if matches!(execution, Execution::Frontier) {
        let (total_spikes, active_count, scan_ns, build_ns, step_ns) = observe_token_frontier(
            glif,
            recurrent,
            after_decay,
            after_cut_decay,
            after_amplitudes,
            afferents,
            targets,
            weights,
            voltage,
            after,
            threshold_spike,
            threshold_voltage,
            refractory,
            synaptic_current,
            eligibility,
            input_scratch,
            spike_mask,
            token_spike_counts,
            active_indices,
            active_mask,
        );
        *active_count_out = active_count;
        *scan_nanoseconds_out = scan_ns;
        *build_nanoseconds_out = build_ns;
        *step_nanoseconds_out = step_ns;
        return total_spikes;
    }

    let use_parallel = match execution {
        Execution::Serial => false,
        Execution::Parallel => rayon::current_num_threads() > 1,
        Execution::Frontier => unreachable!(),
    };
    *active_count_out = cells;
    *scan_nanoseconds_out = 0;
    *build_nanoseconds_out = 0;
    *step_nanoseconds_out = 0;
    let mut total_spikes = 0usize;
    for step in 0..recurrent.token_steps {
        let pulse_active = step < recurrent.pulse_steps;
        if pulse_active {
            for (slot, &target) in afferents.iter().enumerate() {
                let target = target as usize;
                input_scratch[slot] = synaptic_current[target];
                synaptic_current[target] += recurrent.input_current_pa;
            }
        }

        if use_parallel {
            recurrent_step_parallel(
                glif,
                after_decay,
                after_cut_decay,
                after_amplitudes,
                recurrent.recurrent_decay,
                decays,
                cells,
                voltage,
                after,
                threshold_spike,
                threshold_voltage,
                refractory,
                synaptic_current,
                eligibility,
                spike_mask,
                token_spike_counts.as_deref_mut(),
            );
        } else {
            recurrent_step_serial(
                glif,
                after_decay,
                after_cut_decay,
                after_amplitudes,
                recurrent.recurrent_decay,
                decays,
                cells,
                voltage,
                after,
                threshold_spike,
                threshold_voltage,
                refractory,
                synaptic_current,
                eligibility,
                spike_mask,
                token_spike_counts.as_deref_mut(),
            );
        }

        if pulse_active {
            // Reverse restoration also handles repeated afferent targets exactly.
            for slot in (0..afferents.len()).rev() {
                synaptic_current[afferents[slot] as usize] = input_scratch[slot];
            }
            for (slot, &target) in afferents.iter().enumerate() {
                if afferents[..slot].contains(&target) {
                    continue;
                }
                synaptic_current[target as usize] *= recurrent.recurrent_decay;
            }
        }

        // The old compact spike list was generated in ascending cell order.
        // Scanning a one-byte mask preserves that exact recurrent accumulation
        // order while cutting the per-cell workspace from four bytes to one.
        for presynaptic in 0..cells {
            if spike_mask[presynaptic] == 0 {
                continue;
            }
            total_spikes += 1;
            for row in 0..trace_count {
                eligibility[row * cells + presynaptic] += 1.0;
            }
            let edge_offset = presynaptic * recurrent.recurrent_fanout;
            for edge in 0..recurrent.recurrent_fanout {
                let target = targets[edge_offset + edge] as usize;
                synaptic_current[target] += weights[edge_offset + edge];
            }
        }
    }
    total_spikes
}

macro_rules! optimized_observe_export {
    ($name:ident, $execution:expr) => {
        /// Advance one token through the selected exact execution strategy.
        ///
        /// # Safety
        /// Every pointer must address a contiguous array of the length encoded
        /// by the configs; mutable state and workspace arrays must not alias.
        #[no_mangle]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $name(
            glif_config: *const GlifConfig,
            recurrent_config: *const RecurrentConfig,
            after_decay: *const f64,
            after_cut_decay: *const f64,
            after_amplitudes: *const f64,
            afferents: *const i32,
            recurrent_targets: *const i32,
            recurrent_weights: *const f64,
            voltage: *mut f64,
            after: *mut f64,
            threshold_spike: *mut f64,
            threshold_voltage: *mut f64,
            refractory: *mut i32,
            synaptic_current: *mut f64,
            eligibility: *mut f64,
            input_scratch: *mut f64,
            spike_mask: *mut u8,
            active_indices: *mut i32,
            active_mask: *mut u8,
            active_count_out: *mut usize,
            scan_nanoseconds_out: *mut u64,
            build_nanoseconds_out: *mut u64,
            step_nanoseconds_out: *mut u64,
        ) -> usize {
            observe_token_optimized(
                glif_config,
                recurrent_config,
                after_decay,
                after_cut_decay,
                after_amplitudes,
                afferents,
                recurrent_targets,
                recurrent_weights,
                voltage,
                after,
                threshold_spike,
                threshold_voltage,
                refractory,
                synaptic_current,
                eligibility,
                input_scratch,
                spike_mask,
                active_indices,
                active_mask,
                active_count_out,
                scan_nanoseconds_out,
                build_nanoseconds_out,
                step_nanoseconds_out,
                $execution,
            )
        }
    };
}

optimized_observe_export!(humanbrain_observe_token, Execution::Frontier);
optimized_observe_export!(humanbrain_observe_token_serial, Execution::Serial);
optimized_observe_export!(humanbrain_observe_token_parallel, Execution::Parallel);

#[no_mangle]
pub extern "C" fn humanbrain_kernel_abi_version() -> u32 {
    5
}
