//! Native sparse simulation kernel for the fitted human Allen GLIF5 cells.
//!
//! The public surface is deliberately a small C ABI so Python remains
//! orchestration only. The kernel owns no state: NumPy arrays remain the
//! checkpointable source of truth and are mutated in place.

use std::slice;

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
    spike_indices: &mut [i32],
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

        spike_indices[spike_count] = cell as i32;
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
    spike_indices: *mut i32,
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
        slice::from_raw_parts_mut(spike_indices, cells),
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
pub unsafe extern "C" fn humanbrain_observe_token(
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
            spike_indices,
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

#[no_mangle]
pub extern "C" fn humanbrain_kernel_abi_version() -> u32 {
    1
}
