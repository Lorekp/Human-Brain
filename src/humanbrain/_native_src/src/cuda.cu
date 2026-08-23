//! Persistent CUDA kernel for sparse fitted-human-cell recurrent simulation.
//!
//! NVRTC compiles this source directly to device code. Python only owns launch
//! orchestration; all integration, refractory, eligibility, and recurrent
//! dynamics for a token interval execute here.

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

struct GlifConfig {
    unsigned long long cell_count;
    unsigned long long after_channel_count;
    int spike_cut_steps;
    double leak_ns;
    double voltage_step_scale;
    double threshold_beta_scale;
    double threshold_exp_decay;
    double threshold_voltage_factor;
    double threshold_beta_factor;
    double spike_threshold_decay;
    double threshold_inf_relative_mv;
    double voltage_reset_scale;
    double voltage_reset_offset_mv;
    double spike_cut_threshold_decay;
    double threshold_spike_increment_mv;
};

struct RecurrentConfig {
    unsigned long long token_steps;
    unsigned long long pulse_steps;
    unsigned long long input_fanout;
    unsigned long long recurrent_fanout;
    double input_current_pa;
    double recurrent_decay;
    // R0 multi-timescale trace bank: row k of the eligibility buffer
    // (elements [k*cells, (k+1)*cells)) decays by eligibility_decays[k]
    // every sub-step, maintained in fixed index order 0..trace_count.
    unsigned long long trace_count;
    const double* eligibility_decays;
};

extern "C" __global__ void recurrent_token_kernel(
    GlifConfig glif,
    RecurrentConfig recurrent,
    const double* after_decay,
    const double* after_cut_decay,
    const double* after_amplitudes,
    const int* afferents,
    const int* incoming_offsets,
    const int* incoming_edges,
    const double* recurrent_weights,
    double* voltage,
    double* after,
    double* threshold_spike,
    double* threshold_voltage,
    int* refractory,
    double* synaptic_current,
    double* eligibility,
    unsigned char* spike_mask,
    unsigned char* input_mask,
    unsigned long long* total_spikes
) {
    cg::grid_group grid = cg::this_grid();
    const unsigned long long thread = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = gridDim.x * blockDim.x;
    const unsigned long long cells = glif.cell_count;
    const unsigned long long channels = glif.after_channel_count;

    for (unsigned long long cell = thread; cell < cells; cell += stride) {
        input_mask[cell] = 0;
    }
    if (thread == 0) {
        *total_spikes = 0;
    }
    grid.sync();
    if (thread < recurrent.input_fanout) {
        input_mask[afferents[thread]] = 1;
    }
    grid.sync();

    for (unsigned long long step = 0; step < recurrent.token_steps; ++step) {
        unsigned int thread_spikes = 0;
        for (unsigned long long cell = thread; cell < cells; cell += stride) {
            bool crossed = false;
            if (refractory[cell] != 0) {
                refractory[cell] -= 1;
            } else {
                const unsigned long long after_offset = cell * channels;
                double after_sum = 0.0;
                for (unsigned long long channel = 0; channel < channels; ++channel) {
                    after_sum += after[after_offset + channel];
                }
                double current = synaptic_current[cell];
                if (step < recurrent.pulse_steps && input_mask[cell] != 0) {
                    current += recurrent.input_current_pa;
                }
                const double total_current = current + after_sum;
                const double previous_voltage = voltage[cell];
                const double leak = glif.leak_ns * previous_voltage;
                const double voltage_delta =
                    glif.voltage_step_scale * (total_current - leak);
                const double next_voltage = previous_voltage + voltage_delta;

                const double beta = total_current * glif.threshold_beta_scale;
                const double voltage_term =
                    glif.threshold_voltage_factor * previous_voltage;
                const double beta_term = glif.threshold_beta_factor * beta;
                const double decayed_voltage_threshold =
                    glif.threshold_exp_decay * threshold_voltage[cell];
                const double next_voltage_threshold =
                    decayed_voltage_threshold + voltage_term + beta_term;
                const double next_spike_threshold =
                    threshold_spike[cell] * glif.spike_threshold_decay;

                voltage[cell] = next_voltage;
                threshold_voltage[cell] = next_voltage_threshold;
                threshold_spike[cell] = next_spike_threshold;
                for (unsigned long long channel = 0; channel < channels; ++channel) {
                    after[after_offset + channel] *= after_decay[channel];
                }

                const double total_threshold = glif.threshold_inf_relative_mv
                    + next_voltage_threshold + next_spike_threshold;
                crossed = next_voltage > total_threshold;
                if (crossed) {
                    voltage[cell] = glif.voltage_reset_scale * next_voltage
                        + glif.voltage_reset_offset_mv;
                    for (unsigned long long channel = 0; channel < channels; ++channel) {
                        const double next_after = after[after_offset + channel];
                        after[after_offset + channel] = after_amplitudes[channel]
                            + next_after * after_cut_decay[channel];
                    }
                    threshold_spike[cell] = next_spike_threshold
                        * glif.spike_cut_threshold_decay
                        + glif.threshold_spike_increment_mv;
                    refractory[cell] = glif.spike_cut_steps;
                    thread_spikes += 1;
                }
            }
            spike_mask[cell] = static_cast<unsigned char>(crossed);
        }
        if (thread_spikes != 0) {
            atomicAdd(total_spikes, static_cast<unsigned long long>(thread_spikes));
        }
        grid.sync();

        for (unsigned long long cell = thread; cell < cells; cell += stride) {
            double next_current = synaptic_current[cell] * recurrent.recurrent_decay;
            const int begin = incoming_offsets[cell];
            const int end = incoming_offsets[cell + 1];
            for (int position = begin; position < end; ++position) {
                const int edge = incoming_edges[position];
                const unsigned long long source =
                    static_cast<unsigned long long>(edge) / recurrent.recurrent_fanout;
                if (spike_mask[source] != 0) {
                    next_current += recurrent_weights[edge];
                }
            }
            synaptic_current[cell] = next_current;
            // Every trace row advances with identical per-row op order:
            // multiply by its own decay constant, then add this step's spike.
            for (unsigned long long k = 0; k < recurrent.trace_count; ++k) {
                eligibility[k * cells + cell] =
                    eligibility[k * cells + cell] * recurrent.eligibility_decays[k]
                    + static_cast<double>(spike_mask[cell]);
            }
        }
        grid.sync();
    }
}
