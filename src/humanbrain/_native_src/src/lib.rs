//! Native sparse simulation kernel for the fitted human Allen GLIF5 cells.
//!
//! The public surface is deliberately a small C ABI so Python remains
//! orchestration only. The kernel owns no state: NumPy arrays remain the
//! checkpointable source of truth and are mutated in place.

use rayon::prelude::*;
use std::slice;

const PARALLEL_MIN_CELLS: usize = 32_768;
const MIN_PARALLEL_CHUNK_CELLS: usize = 4_096;

#[repr(C)]
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
    pub eligibility_decay: f64,
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
    eligibility_decay: f64,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
) {
    let channel_count = config.after_channel_count;
    debug_assert_eq!(after.len(), voltage.len() * channel_count);

    for cell in 0..voltage.len() {
        let total_synaptic_current = synaptic_current[cell];
        if refractory[cell] != 0 {
            refractory[cell] -= 1;
            spike_mask[cell] = 0;
        } else {
            let after_offset = cell * channel_count;
            let mut after_sum = 0.0f64;
            for channel in 0..channel_count {
                after_sum += after[after_offset + channel];
            }
            let total_current = total_synaptic_current + after_sum;
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
            spike_mask[cell] = u8::from(crossed);
            if crossed {
                voltage[cell] =
                    config.voltage_reset_scale * next_voltage + config.voltage_reset_offset_mv;
                for channel in 0..channel_count {
                    let next_after = after[after_offset + channel];
                    after[after_offset + channel] =
                        after_amplitudes[channel] + next_after * after_cut_decay[channel];
                }
                threshold_spike[cell] = next_spike_threshold * config.spike_cut_threshold_decay
                    + config.threshold_spike_increment_mv;
                refractory[cell] = config.spike_cut_steps;
            }
        }

        // These independent decays originally formed a second full-array pass.
        // Keeping their arithmetic unchanged while fusing the pass removes 24
        // bytes of memory traffic per cell and biological integration step.
        synaptic_current[cell] *= recurrent_decay;
        eligibility[cell] *= eligibility_decay;
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
    eligibility_decay: f64,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
) {
    use std::arch::x86_64::*;

    debug_assert_eq!(config.after_channel_count, 2);
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
    let eligibility_decay = _mm256_set1_pd(eligibility_decay);

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
                _mm256_cvtsd_f64(eligibility_decay),
                &mut voltage[cell..cell + width],
                &mut after[cell * 2..(cell + width) * 2],
                &mut threshold_spike[cell..cell + width],
                &mut threshold_voltage[cell..cell + width],
                &mut refractory[cell..cell + width],
                &mut synaptic_current[cell..cell + width],
                &mut eligibility[cell..cell + width],
                &mut spike_mask[cell..cell + width],
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
        let previous_eligibility = _mm256_loadu_pd(eligibility.as_ptr().add(cell));
        let decayed_eligibility = _mm256_mul_pd(previous_eligibility, eligibility_decay);
        _mm256_storeu_pd(eligibility.as_mut_ptr().add(cell), decayed_eligibility);

        for lane in 0..width {
            let crossed = (crossed_bits >> lane) & 1;
            spike_mask[cell + lane] = crossed;
            if crossed == 0 {
                continue;
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
            _mm256_cvtsd_f64(eligibility_decay),
            &mut voltage[cell..],
            &mut after[cell * 2..],
            &mut threshold_spike[cell..],
            &mut threshold_voltage[cell..],
            &mut refractory[cell..],
            &mut synaptic_current[cell..],
            &mut eligibility[cell..],
            &mut spike_mask[cell..],
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
    eligibility_decay: f64,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
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
                eligibility_decay,
                voltage,
                after,
                threshold_spike,
                threshold_voltage,
                refractory,
                synaptic_current,
                eligibility,
                spike_mask,
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
        eligibility_decay,
        voltage,
        after,
        threshold_spike,
        threshold_voltage,
        refractory,
        synaptic_current,
        eligibility,
        spike_mask,
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
    eligibility_decay: f64,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
) {
    recurrent_cell_chunk(
        config,
        after_decay,
        after_cut_decay,
        after_amplitudes,
        recurrent_decay,
        eligibility_decay,
        voltage,
        after,
        threshold_spike,
        threshold_voltage,
        refractory,
        synaptic_current,
        eligibility,
        spike_mask,
    );
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn recurrent_step_parallel(
    config: &GlifConfig,
    after_decay: &[f64],
    after_cut_decay: &[f64],
    after_amplitudes: &[f64],
    recurrent_decay: f64,
    eligibility_decay: f64,
    voltage: &mut [f64],
    after: &mut [f64],
    threshold_spike: &mut [f64],
    threshold_voltage: &mut [f64],
    refractory: &mut [i32],
    synaptic_current: &mut [f64],
    eligibility: &mut [f64],
    spike_mask: &mut [u8],
) {
    let cells = voltage.len();
    let channels = config.after_channel_count;
    let target_chunks = rayon::current_num_threads() * 4;
    let chunk_cells = cells.div_ceil(target_chunks).max(MIN_PARALLEL_CHUNK_CELLS);

    voltage
        .par_chunks_mut(chunk_cells)
        .zip(after.par_chunks_mut(chunk_cells * channels))
        .zip(threshold_spike.par_chunks_mut(chunk_cells))
        .zip(threshold_voltage.par_chunks_mut(chunk_cells))
        .zip(refractory.par_chunks_mut(chunk_cells))
        .zip(synaptic_current.par_chunks_mut(chunk_cells))
        .zip(eligibility.par_chunks_mut(chunk_cells))
        .zip(spike_mask.par_chunks_mut(chunk_cells))
        .for_each(
            |(
                (
                    (
                        ((((voltage, after), threshold_spike), threshold_voltage), refractory),
                        current,
                    ),
                    eligibility,
                ),
                mask,
            )| {
                recurrent_cell_chunk(
                    config,
                    after_decay,
                    after_cut_decay,
                    after_amplitudes,
                    recurrent_decay,
                    eligibility_decay,
                    voltage,
                    after,
                    threshold_spike,
                    threshold_voltage,
                    refractory,
                    current,
                    eligibility,
                    mask,
                );
            },
        );
}

#[derive(Clone, Copy)]
enum Execution {
    Auto,
    Serial,
    Parallel,
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
    let eligibility = slice::from_raw_parts_mut(eligibility, cells);
    let step_current = slice::from_raw_parts_mut(step_current, cells);
    let spike_indices = slice::from_raw_parts_mut(spike_indices, cells);

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
        );
        total_spikes += spike_count;

        for cell in 0..cells {
            synaptic_current[cell] *= recurrent.recurrent_decay;
            eligibility[cell] *= recurrent.eligibility_decay;
        }
        for &presynaptic in &spike_indices[..spike_count] {
            let presynaptic = presynaptic as usize;
            eligibility[presynaptic] += 1.0;
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
    execution: Execution,
) -> usize {
    if glif_config.is_null() || recurrent_config.is_null() {
        return 0;
    }
    let glif = &*glif_config;
    let recurrent = &*recurrent_config;
    let cells = glif.cell_count;
    let channels = glif.after_channel_count;
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
    let eligibility = slice::from_raw_parts_mut(eligibility, cells);
    let input_scratch = slice::from_raw_parts_mut(input_scratch, recurrent.input_fanout);
    let spike_mask = slice::from_raw_parts_mut(spike_mask, cells);

    let use_parallel = match execution {
        Execution::Auto => cells >= PARALLEL_MIN_CELLS && rayon::current_num_threads() > 1,
        Execution::Serial => false,
        Execution::Parallel => rayon::current_num_threads() > 1,
    };
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
                recurrent.eligibility_decay,
                voltage,
                after,
                threshold_spike,
                threshold_voltage,
                refractory,
                synaptic_current,
                eligibility,
                spike_mask,
            );
        } else {
            recurrent_step_serial(
                glif,
                after_decay,
                after_cut_decay,
                after_amplitudes,
                recurrent.recurrent_decay,
                recurrent.eligibility_decay,
                voltage,
                after,
                threshold_spike,
                threshold_voltage,
                refractory,
                synaptic_current,
                eligibility,
                spike_mask,
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
            eligibility[presynaptic] += 1.0;
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
                $execution,
            )
        }
    };
}

optimized_observe_export!(humanbrain_observe_token, Execution::Auto);
optimized_observe_export!(humanbrain_observe_token_serial, Execution::Serial);
optimized_observe_export!(humanbrain_observe_token_parallel, Execution::Parallel);

#[no_mangle]
pub extern "C" fn humanbrain_kernel_abi_version() -> u32 {
    3
}
