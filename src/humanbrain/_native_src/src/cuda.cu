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
    // R0-b family A (spike-count windows): u32 per-cell accumulator for the
    // current token, cleared once at token start and folded at exactly the
    // existing mask-write site. May be null (legacy callers).
    unsigned int* token_spike_counts;
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
    unsigned long long* total_spikes,
    unsigned long long* decay_active,
    unsigned long long* threshold_spike_active,
    unsigned long long* refractory_active
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
    if (recurrent.token_spike_counts != nullptr) {
        // Family A: cleared at TOKEN start, before any sub-step accumulates.
        for (unsigned long long cell = thread; cell < cells; cell += stride) {
            recurrent.token_spike_counts[cell] = 0u;
        }
    }
    grid.sync();
    if (thread < recurrent.input_fanout) {
        input_mask[afferents[thread]] = 1;
    }
    grid.sync();

    for (unsigned long long step = 0; step < recurrent.token_steps; ++step) {
        unsigned int thread_spikes = 0;
        for (unsigned long long cell = thread; cell < cells; cell += stride) {
            // Decayed-state activity bitmap: a cell whose after-spike currents
            // and eligibility trace are all exactly zero stays exactly zero
            // under decay (0.0 * d == +0.0 for the positive decay factors and
            // (+0.0) + (+0.0) == +0.0), so skipping their loads, decays, and
            // stores is a bit-exact no-op. Cells enter the active set on their
            // first spike (atomicOr below) and never leave it until the host
            // re-derives the bitmap from uploaded state.
            const unsigned long long activity_word = cell >> 6;
            const unsigned long long activity_bit = 1ULL << (cell & 63);
            const bool decay_active_cell =
                (decay_active[activity_word] & activity_bit) != 0ULL;
            bool crossed = false;
            if ((refractory_active[activity_word] & activity_bit) != 0ULL
                && refractory[cell] != 0) {
                refractory[cell] -= 1;
                // Owner-thread maintenance: only this thread writes its own
                // cell, so clearing the "possibly nonzero" bit exactly when
                // the countdown reaches zero cannot race.
                if (refractory[cell] == 0) {
                    atomicAnd(refractory_active + activity_word, ~activity_bit);
                }
            } else {
                const unsigned long long after_offset = cell * channels;
                double after_sum = 0.0;
                if (decay_active_cell) {
                    for (unsigned long long channel = 0; channel < channels; ++channel) {
                        after_sum += after[after_offset + channel];
                    }
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
                // Zero-fixpoint elision: for a cell whose spike threshold is
                // exactly ±0.0 the decay below is a bit-level no-op
                // (+0.0 * positive d == +0.0, -0.0 guarded as active via the
                // signbit-aware host bitmap), so its load and store may be
                // skipped. Cells enter the active set on their first spike.
                double next_spike_threshold;
                if ((threshold_spike_active[activity_word] & activity_bit) != 0ULL) {
                    next_spike_threshold =
                        threshold_spike[cell] * glif.spike_threshold_decay;
                    threshold_spike[cell] = next_spike_threshold;
                } else {
                    next_spike_threshold = 0.0;
                }

                voltage[cell] = next_voltage;
                threshold_voltage[cell] = next_voltage_threshold;
                if (decay_active_cell) {
                    for (unsigned long long channel = 0; channel < channels; ++channel) {
                        after[after_offset + channel] *= after_decay[channel];
                    }
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
                    // The cell just spiked: its spike threshold and refractory
                    // countdown are now (possibly) nonzero, so both must take
                    // the full update path until the host re-derives the
                    // bitmaps from uploaded state.
                    atomicOr(threshold_spike_active + activity_word, activity_bit);
                    atomicOr(refractory_active + activity_word, activity_bit);
                    thread_spikes += 1;
                }
            }
            spike_mask[cell] = static_cast<unsigned char>(crossed);
            // Fused multi-trace eligibility: identical expression and inputs
            // as the previous post-sync pass (spike_mask[cell] was just
            // written from this thread's own `crossed`), executed without
            // re-reading the mask or a second full-array traversal. Row k of
            // the bank decays by eligibility_decays[k] and adds this step's
            // spike, in fixed index order. Inactive zero-trace cells keep
            // their exact +0.0 bits by skipping the update.
            if (decay_active_cell || crossed) {
                for (unsigned long long k = 0; k < recurrent.trace_count; ++k) {
                    eligibility[k * cells + cell] =
                        eligibility[k * cells + cell] * recurrent.eligibility_decays[k]
                        + static_cast<double>(crossed);
                }
                if (crossed && !decay_active_cell) {
                    atomicOr(decay_active + activity_word, activity_bit);
                }
            }
            if (recurrent.token_spike_counts != nullptr && crossed) {
                // Family A: folded at exactly the existing mask-write site.
                recurrent.token_spike_counts[cell] += 1u;
            }
        }
        if (thread_spikes != 0) {
            atomicAdd(total_spikes, static_cast<unsigned long long>(thread_spikes));
        }
        grid.sync();

        // Bit-exact work elision: when no cell spiked in this integration
        // step, every edge contribution below would test spike_mask[source]
        // == 0 and be skipped, leaving next_current == synaptic_current *
        // recurrent_decay exactly. Reading the post-sync spike total lets all
        // threads uniformly skip the reverse-CSR scan for spike-free steps
        // without changing any floating-point result. Neuron updates, spike
        // masks, synaptic decay, and eligibility traces remain full-touch.
        const unsigned long long step_spikes = *total_spikes;
        for (unsigned long long cell = thread; cell < cells; cell += stride) {
            double next_current = synaptic_current[cell] * recurrent.recurrent_decay;
            if (step_spikes != 0ULL) {
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
            }
            synaptic_current[cell] = next_current;
            // Eligibility traces (full multi-trace bank) are updated in the
            // neuron pass above; this pass only decays and drives synaptic
            // current.
        }
        grid.sync();
    }
}
