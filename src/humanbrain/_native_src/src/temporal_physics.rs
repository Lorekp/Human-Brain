//! Exact fitted-GLIF3 first-spike physics for the native R3 temporal runtime.
//!
//! This module ports the scalar equations and event ordering in
//! `humanbrain.cortex.DendriticCortex` without importing token, column, segment,
//! or winner-selection state. Callers supply physical cell IDs, currents, and
//! their frozen threshold offsets. All observable times are integer 0.05-ms
//! ticks. The fitted reset, refractory, and after-spike-current rules cannot
//! affect this seam because each cell is observed only through its first spike.

use std::sync::OnceLock;

/// Immutable identity of the fitted temporal physics executed by this module.
///
/// The fields are the one authoritative source for both the equations below
/// and the future checkpoint circuit-identity hash. Exact float64 words are
/// used instead of decimal reconstruction so a platform libm call cannot move
/// a threshold crossing by one tick. `integration_step_ms` records the tick's
/// physical unit; `voltage_step_scale` already folds that step into the fitted
/// membrane coefficient and must not be recomputed from metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TemporalPhysicsIdentity {
    pub(crate) integration_step_ms: f64,
    pub(crate) race_ticks: u16,
    pub(crate) leak_ns: f64,
    pub(crate) voltage_step_scale: f64,
    pub(crate) threshold_relative_mv: f64,
    pub(crate) nmda_decay: f64,
    pub(crate) gabaa_decay: f64,
    pub(crate) novel_base_pa: f64,
    pub(crate) novel_rank_decrement_pa: f64,
    pub(crate) matching_pa: f64,
    pub(crate) active_pa: f64,
    pub(crate) proximal_pa: f64,
    pub(crate) relay_pa: f64,
    pub(crate) homeostatic_numerator_pa: f64,
    pub(crate) basket_current_pa: f64,
    pub(crate) basket_inhibition_pa: f64,
}

impl TemporalPhysicsIdentity {
    /// Exact checkpoint-v1 `physics_f64` word order.
    pub(crate) const fn physics_f64_bits(&self) -> [u64; 14] {
        [
            self.leak_ns.to_bits(),
            self.voltage_step_scale.to_bits(),
            self.threshold_relative_mv.to_bits(),
            self.nmda_decay.to_bits(),
            self.gabaa_decay.to_bits(),
            self.novel_base_pa.to_bits(),
            self.novel_rank_decrement_pa.to_bits(),
            self.matching_pa.to_bits(),
            self.active_pa.to_bits(),
            self.proximal_pa.to_bits(),
            self.relay_pa.to_bits(),
            self.homeostatic_numerator_pa.to_bits(),
            self.basket_current_pa.to_bits(),
            self.basket_inhibition_pa.to_bits(),
        ]
    }
}

pub(crate) const TEMPORAL_PHYSICS_IDENTITY: TemporalPhysicsIdentity = TemporalPhysicsIdentity {
    integration_step_ms: f64::from_bits(0x3fa9_9999_9999_999a),
    race_ticks: 600,
    leak_ns: f64::from_bits(0x4020_f0a7_d905_53e6),
    voltage_step_scale: f64::from_bits(0x3f33_ea95_a3d1_e76b),
    threshold_relative_mv: f64::from_bits(0x4039_a18e_521f_e42e),
    nmda_decay: f64::from_bits(0x3fef_f44e_351e_f7e1),
    gabaa_decay: f64::from_bits(0x3fef_ccf5_acbf_ae90),
    novel_base_pa: f64::from_bits(0x4081_d000_0000_0000),
    novel_rank_decrement_pa: f64::from_bits(0x4047_8000_0000_0000),
    matching_pa: f64::from_bits(0x4083_6000_0000_0000),
    active_pa: f64::from_bits(0x4084_f000_0000_0000),
    proximal_pa: f64::from_bits(0x4081_d000_0000_0000),
    relay_pa: f64::from_bits(0x40d3_8800_0000_0000),
    homeostatic_numerator_pa: f64::from_bits(0x4034_0000_0000_0000),
    basket_current_pa: f64::from_bits(0x4108_6a00_0000_0000),
    basket_inhibition_pa: f64::from_bits(0x407f_4000_0000_0000),
};

struct NmdaUnitResponse {
    voltage_mv: [f64; TEMPORAL_PHYSICS_IDENTITY.race_ticks as usize],
    rising_len: usize,
}

fn nmda_unit_response() -> &'static NmdaUnitResponse {
    static RESPONSE: OnceLock<NmdaUnitResponse> = OnceLock::new();
    RESPONSE.get_or_init(|| {
        let mut voltage_mv = [0.0; TEMPORAL_PHYSICS_IDENTITY.race_ticks as usize];
        let mut voltage = 0.0;
        let mut current = 1.0;
        let mut peak_index = 0;
        let mut peak_voltage = f64::NEG_INFINITY;
        for (index, sample) in voltage_mv.iter_mut().enumerate() {
            voltage += TEMPORAL_PHYSICS_IDENTITY.voltage_step_scale
                * (current - TEMPORAL_PHYSICS_IDENTITY.leak_ns * voltage);
            *sample = voltage;
            if voltage > peak_voltage {
                peak_voltage = voltage;
                peak_index = index;
            }
            current *= TEMPORAL_PHYSICS_IDENTITY.nmda_decay;
        }
        NmdaUnitResponse {
            voltage_mv,
            rising_len: peak_index + 1,
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateClass {
    Novel { rank: u8, load: u8 },
    Matching { load: u8 },
    Active { load: u8 },
    Proximal,
    Relay,
}

impl CandidateClass {
    /// Resolve the frozen R3 current ladder, including bounded homeostasis.
    pub(crate) fn current_pa(self) -> f64 {
        match self {
            Self::Novel { rank, load } => {
                assert!(rank < 8, "novel rank exceeds the calibrated race cap");
                TEMPORAL_PHYSICS_IDENTITY.novel_base_pa
                    - TEMPORAL_PHYSICS_IDENTITY.novel_rank_decrement_pa * f64::from(rank)
                    + homeostatic_current_pa(load)
            }
            Self::Matching { load } => {
                TEMPORAL_PHYSICS_IDENTITY.matching_pa + homeostatic_current_pa(load)
            }
            Self::Active { load } => {
                TEMPORAL_PHYSICS_IDENTITY.active_pa + homeostatic_current_pa(load)
            }
            Self::Proximal => TEMPORAL_PHYSICS_IDENTITY.proximal_pa,
            Self::Relay => TEMPORAL_PHYSICS_IDENTITY.relay_pa,
        }
    }
}

#[inline]
fn homeostatic_current_pa(load: u8) -> f64 {
    TEMPORAL_PHYSICS_IDENTITY.homeostatic_numerator_pa / (1.0 + f64::from(load))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TemporalPhysicsStop {
    InvalidInput(&'static str),
    NoPyramidalSpike,
    MultiplePyramidalSpikes {
        first: SpikeEvent,
        second: SpikeEvent,
        spike_count: usize,
    },
    BasketInhibitionMissing {
        winner: SpikeEvent,
        basket_tick: Option<u16>,
        inhibition_onset_tick: Option<u16>,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CandidateDrive {
    pub(crate) cell_id: u32,
    pub(crate) current_pa: f64,
    pub(crate) threshold_offset_mv: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BasketDrive {
    pub(crate) cell_id: u32,
    pub(crate) current_pa: f64,
    pub(crate) threshold_offset_mv: f64,
    pub(crate) inhibition_pa: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpikeEvent {
    pub(crate) cell_id: u32,
    pub(crate) tick: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CoupledRaceTicks {
    pub(crate) winner: SpikeEvent,
    pub(crate) basket_cell_id: u32,
    pub(crate) basket_tick: u16,
    pub(crate) inhibition_onset_tick: u16,
    pub(crate) simulated_ticks: u16,
}

/// Construction-sized work buffers for one coupled race.
///
/// Length stays equal to the maximum materialized column occupancy. Calls use
/// a prefix and never resize, reserve, or allocate.
pub(crate) struct CoupledRaceScratch {
    voltage_mv: Vec<f64>,
    thresholds_mv: Vec<f64>,
    excitatory_pa: Vec<f64>,
    emitted: Vec<u8>,
    spiking_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CoupledRaceScratchBuildError {
    ZeroCapacity,
    CapacityOverflow,
    RetryableAllocation(CoupledRaceScratchReserveSite),
}

impl CoupledRaceScratchBuildError {
    pub(crate) const fn legacy_message(self) -> &'static str {
        match self {
            Self::ZeroCapacity => "coupled-race scratch requires nonzero capacity",
            Self::CapacityOverflow | Self::RetryableAllocation(_) => {
                "coupled-race scratch allocation failed"
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CoupledRaceScratchReserveSite {
    Voltage,
    Thresholds,
    Excitatory,
    Emitted,
    SpikingIndices,
}

impl CoupledRaceScratch {
    pub(crate) fn try_new(max_candidates: usize) -> Result<Self, CoupledRaceScratchBuildError> {
        Self::try_new_with_reserve(max_candidates, |_, _| false)
    }

    pub(crate) fn try_new_with_reserve(
        max_candidates: usize,
        mut fail_before_reserve: impl FnMut(CoupledRaceScratchReserveSite, usize) -> bool,
    ) -> Result<Self, CoupledRaceScratchBuildError> {
        if max_candidates == 0 {
            return Err(CoupledRaceScratchBuildError::ZeroCapacity);
        }

        std::alloc::Layout::array::<f64>(max_candidates)
            .map_err(|_| CoupledRaceScratchBuildError::CapacityOverflow)?;
        std::alloc::Layout::array::<f64>(max_candidates)
            .map_err(|_| CoupledRaceScratchBuildError::CapacityOverflow)?;
        std::alloc::Layout::array::<f64>(max_candidates)
            .map_err(|_| CoupledRaceScratchBuildError::CapacityOverflow)?;
        std::alloc::Layout::array::<u8>(max_candidates)
            .map_err(|_| CoupledRaceScratchBuildError::CapacityOverflow)?;
        std::alloc::Layout::array::<usize>(max_candidates)
            .map_err(|_| CoupledRaceScratchBuildError::CapacityOverflow)?;

        fn filled<T: Copy>(
            length: usize,
            value: T,
            site: CoupledRaceScratchReserveSite,
            fail_before_reserve: &mut impl FnMut(CoupledRaceScratchReserveSite, usize) -> bool,
        ) -> Result<Vec<T>, CoupledRaceScratchBuildError> {
            if fail_before_reserve(site, length) {
                return Err(CoupledRaceScratchBuildError::RetryableAllocation(site));
            }
            let mut values = Vec::new();
            values
                .try_reserve_exact(length)
                .map_err(|_| CoupledRaceScratchBuildError::RetryableAllocation(site))?;
            values.resize(length, value);
            Ok(values)
        }

        Ok(Self {
            voltage_mv: filled(
                max_candidates,
                0.0,
                CoupledRaceScratchReserveSite::Voltage,
                &mut fail_before_reserve,
            )?,
            thresholds_mv: filled(
                max_candidates,
                0.0,
                CoupledRaceScratchReserveSite::Thresholds,
                &mut fail_before_reserve,
            )?,
            excitatory_pa: filled(
                max_candidates,
                0.0,
                CoupledRaceScratchReserveSite::Excitatory,
                &mut fail_before_reserve,
            )?,
            emitted: filled(
                max_candidates,
                0,
                CoupledRaceScratchReserveSite::Emitted,
                &mut fail_before_reserve,
            )?,
            spiking_indices: filled(
                max_candidates,
                0,
                CoupledRaceScratchReserveSite::SpikingIndices,
                &mut fail_before_reserve,
            )?,
        })
    }

    fn capacity(&self) -> usize {
        self.voltage_mv.len()
    }

    #[cfg(test)]
    pub(crate) fn allocation_signature(&self) -> [(usize, usize); 5] {
        [
            (
                self.voltage_mv.as_ptr() as usize,
                self.voltage_mv.capacity(),
            ),
            (
                self.thresholds_mv.as_ptr() as usize,
                self.thresholds_mv.capacity(),
            ),
            (
                self.excitatory_pa.as_ptr() as usize,
                self.excitatory_pa.capacity(),
            ),
            (self.emitted.as_ptr() as usize, self.emitted.capacity()),
            (
                self.spiking_indices.as_ptr() as usize,
                self.spiking_indices.capacity(),
            ),
        ]
    }
}

pub(crate) fn singleton_latency_ticks(
    current_pa: f64,
    threshold_offset_mv: f64,
) -> Result<u16, TemporalPhysicsStop> {
    if !current_pa.is_finite() || current_pa < 0.0 || !threshold_offset_mv.is_finite() {
        return Err(TemporalPhysicsStop::InvalidInput(
            "singleton drive must be finite and nonnegative",
        ));
    }
    if current_pa == 0.0 {
        return Err(TemporalPhysicsStop::NoPyramidalSpike);
    }
    let response = nmda_unit_response();
    let response_needed =
        (TEMPORAL_PHYSICS_IDENTITY.threshold_relative_mv + threshold_offset_mv) / current_pa;
    for (index, &voltage_mv) in response.voltage_mv[..response.rising_len]
        .iter()
        .enumerate()
    {
        if voltage_mv >= response_needed {
            return Ok(index as u16 + 1);
        }
    }

    Err(TemporalPhysicsStop::NoPyramidalSpike)
}

/// Run one causal pyramidal -> basket -> GABAA race from rest.
///
/// Same-tick pyramidal spikes are ordered by descending voltage/threshold
/// margin and then ascending physical cell ID. The function validates the
/// sparse-temporal caller's fail-closed contract: zero spikes, multiple spikes,
/// or absent basket inhibition are STOP results rather than usable winners.
pub(crate) fn coupled_first_spike_race(
    candidates: &[CandidateDrive],
    basket: BasketDrive,
    scratch: &mut CoupledRaceScratch,
) -> Result<CoupledRaceTicks, TemporalPhysicsStop> {
    if candidates.is_empty() {
        return Err(TemporalPhysicsStop::InvalidInput(
            "coupled race requires at least one candidate",
        ));
    }
    if candidates.len() > scratch.capacity() {
        return Err(TemporalPhysicsStop::InvalidInput(
            "coupled-race scratch is smaller than candidate occupancy",
        ));
    }
    if !basket.current_pa.is_finite()
        || basket.current_pa < 0.0
        || !basket.inhibition_pa.is_finite()
        || basket.inhibition_pa < 0.0
        || !basket.threshold_offset_mv.is_finite()
    {
        return Err(TemporalPhysicsStop::InvalidInput(
            "basket drive must be finite and nonnegative",
        ));
    }
    for (offset, candidate) in candidates.iter().enumerate() {
        if !candidate.current_pa.is_finite()
            || candidate.current_pa < 0.0
            || !candidate.threshold_offset_mv.is_finite()
        {
            return Err(TemporalPhysicsStop::InvalidInput(
                "candidate drive must be finite and nonnegative",
            ));
        }
        if candidate.cell_id == basket.cell_id
            || candidates[..offset]
                .iter()
                .any(|other| other.cell_id == candidate.cell_id)
        {
            return Err(TemporalPhysicsStop::InvalidInput(
                "physical cell ids must be unique",
            ));
        }
    }

    for (index, candidate) in candidates.iter().enumerate() {
        scratch.voltage_mv[index] = 0.0;
        scratch.thresholds_mv[index] =
            TEMPORAL_PHYSICS_IDENTITY.threshold_relative_mv + candidate.threshold_offset_mv;
        scratch.excitatory_pa[index] = candidate.current_pa;
        scratch.emitted[index] = 0;
    }
    let mut inhibitory_pa = 0.0;
    let mut basket_voltage_mv = 0.0;
    let mut basket_excitation_pa = 0.0;
    let basket_threshold_mv =
        TEMPORAL_PHYSICS_IDENTITY.threshold_relative_mv + basket.threshold_offset_mv;
    let mut basket_tick = None;
    let mut inhibition_onset_tick = None;
    let mut first_spike = None;
    let mut second_spike = None;
    let mut spike_count = 0usize;
    let mut simulated_ticks = 0;

    for tick in 1..=TEMPORAL_PHYSICS_IDENTITY.race_ticks {
        simulated_ticks = tick;
        for index in 0..candidates.len() {
            scratch.voltage_mv[index] += TEMPORAL_PHYSICS_IDENTITY.voltage_step_scale
                * (scratch.excitatory_pa[index]
                    - inhibitory_pa
                    - TEMPORAL_PHYSICS_IDENTITY.leak_ns * scratch.voltage_mv[index]);
        }
        if basket_tick.is_none() {
            basket_voltage_mv += TEMPORAL_PHYSICS_IDENTITY.voltage_step_scale
                * (basket_excitation_pa - TEMPORAL_PHYSICS_IDENTITY.leak_ns * basket_voltage_mv);
        }

        let mut spiking_count = 0usize;
        for index in 0..candidates.len() {
            if scratch.emitted[index] == 0
                && scratch.voltage_mv[index] >= scratch.thresholds_mv[index]
            {
                scratch.spiking_indices[spiking_count] = index;
                spiking_count += 1;
            }
        }
        let voltage_mv = &scratch.voltage_mv;
        let thresholds_mv = &scratch.thresholds_mv;
        scratch.spiking_indices[..spiking_count].sort_unstable_by(|&left, &right| {
            let left_margin = voltage_mv[left] - thresholds_mv[left];
            let right_margin = voltage_mv[right] - thresholds_mv[right];
            right_margin
                .total_cmp(&left_margin)
                .then_with(|| candidates[left].cell_id.cmp(&candidates[right].cell_id))
        });
        let first_pyramidal_spike = spike_count == 0 && spiking_count != 0;
        for &index in &scratch.spiking_indices[..spiking_count] {
            scratch.emitted[index] = 1;
            let spike = SpikeEvent {
                cell_id: candidates[index].cell_id,
                tick,
            };
            if first_spike.is_none() {
                first_spike = Some(spike);
            } else if second_spike.is_none() {
                second_spike = Some(spike);
            }
            spike_count += 1;
        }

        let basket_spiked = basket_tick.is_none() && basket_voltage_mv >= basket_threshold_mv;
        if basket_spiked {
            basket_tick = Some(tick);
        }

        for current_pa in &mut scratch.excitatory_pa[..candidates.len()] {
            *current_pa *= TEMPORAL_PHYSICS_IDENTITY.nmda_decay;
        }
        basket_excitation_pa *= TEMPORAL_PHYSICS_IDENTITY.nmda_decay;
        inhibitory_pa *= TEMPORAL_PHYSICS_IDENTITY.gabaa_decay;
        if first_pyramidal_spike {
            basket_excitation_pa += basket.current_pa;
        }
        if basket_spiked && basket.inhibition_pa > 0.0 {
            inhibitory_pa += basket.inhibition_pa;
            inhibition_onset_tick = Some(tick + 1);
        }

        if scratch.emitted[..candidates.len()]
            .iter()
            .all(|&was_emitted| was_emitted != 0)
            && (basket_tick.is_some() || basket.current_pa == 0.0)
        {
            break;
        }
    }

    let Some(winner) = first_spike else {
        return Err(TemporalPhysicsStop::NoPyramidalSpike);
    };
    if let Some(second) = second_spike {
        return Err(TemporalPhysicsStop::MultiplePyramidalSpikes {
            first: winner,
            second,
            spike_count,
        });
    }
    let (Some(basket_tick), Some(inhibition_onset_tick)) = (basket_tick, inhibition_onset_tick)
    else {
        return Err(TemporalPhysicsStop::BasketInhibitionMissing {
            winner,
            basket_tick,
            inhibition_onset_tick,
        });
    };

    Ok(CoupledRaceTicks {
        winner,
        basket_cell_id: basket.cell_id,
        basket_tick,
        inhibition_onset_tick,
        simulated_ticks,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        coupled_first_spike_race, singleton_latency_ticks, BasketDrive, CandidateClass,
        CandidateDrive, CoupledRaceScratch, CoupledRaceScratchBuildError,
        CoupledRaceScratchReserveSite, CoupledRaceTicks, SpikeEvent, TEMPORAL_PHYSICS_IDENTITY,
    };

    // Literal outputs were generated from the current Python oracle:
    // cells.py: 5ca05a64a89f930db3ea351e3e236e3242bf21652fff127747a36d9ca488d2e6
    // cortex.py: 16545b390c516b6f5817366946f0f516cec5b049bb9e1f6021184d9e156e0a8c
    // sparse_temporal_cortex.py:
    // 22e5ab6bb9e3b1bc7010e8c3ffd11d9c2dd9ea01a33ba15046a32db2228357a8
    // These are synthetic electrical fixtures, never outcome data.
    #[derive(Clone, Copy)]
    struct SeedFixture {
        seed: u32,
        cells_and_offsets: [(u32, f64); 8],
        basket_offset_mv: f64,
        winner: u32,
        runner_up: u32,
        winner_ticks: [u16; 3],
        runner_up_ticks: [u16; 3],
    }

    const SCENARIO_CURRENTS_PA: [[f64; 8]; 3] = [
        [650.0, 640.0, 590.0, 543.0, 496.0, 449.0, 402.0, 355.0],
        [600.0, 590.0, 543.0, 496.0, 449.0, 402.0, 355.0, 308.0],
        [550.0, 543.0, 496.0, 449.0, 402.0, 355.0, 308.0, 261.0],
    ];

    const SEED_FIXTURES: [SeedFixture; 5] = [
        SeedFixture {
            seed: 3,
            cells_and_offsets: [
                (0, 0.051_022_978_034_629_57),
                (1, -0.063_891_625_782_854_55),
                (6, -0.050_499_653_228_681_274),
                (3, -0.014_194_240_153_198_246),
                (4, -0.011_316_232_302_761_147),
                (7, -0.005_798_309_441_104_737_6),
                (5, -0.005_389_929_077_244_148),
                (2, 0.010_452_471_168_144_472),
            ],
            basket_offset_mv: -0.021_630_326_906_873_543,
            winner: 0,
            runner_up: 1,
            winner_ticks: [190, 217, 255],
            runner_up_ticks: [193, 221, 259],
        },
        SeedFixture {
            seed: 7,
            cells_and_offsets: [
                (7, 0.033_505_381_138_863_34),
                (5, -0.024_791_163_874_911_56),
                (3, -0.022_264_795_968_931_855),
                (4, -0.011_366_769_629_293_064),
                (2, -0.006_853_446_384_055_439),
                (0, 0.000_030_753_833_937_064_36),
                (6, 0.001_503_590_064_935_962_2),
                (1, 0.007_468_638_437_711_748),
            ],
            basket_offset_mv: -0.012_305_162_963_783_242,
            winner: 7,
            runner_up: 5,
            winner_ticks: [189, 216, 254],
            runner_up_ticks: [194, 222, 260],
        },
        SeedFixture {
            seed: 17,
            cells_and_offsets: [
                (0, 0.027_531_561_337_646_178),
                (4, -0.047_365_531_745_981_38),
                (3, -0.031_506_046_421_310_82),
                (7, -0.021_803_898_998_362_832),
                (6, -0.020_264_177_487_790_07),
                (2, -0.013_499_287_881_337_588),
                (5, 0.000_465_957_274_582_140_36),
                (1, 0.008_460_781_916_154_445),
            ],
            basket_offset_mv: -0.005_549_237_677_097_276,
            winner: 0,
            runner_up: 4,
            winner_ticks: [189, 216, 254],
            runner_up_ticks: [193, 222, 259],
        },
        SeedFixture {
            seed: 29,
            cells_and_offsets: [
                (6, 0.007_423_069_314_788_845),
                (0, -0.009_796_746_048_180_448),
                (5, -0.005_642_777_033_177_453),
                (3, -0.004_199_234_753_761_521_6),
                (2, -0.004_129_754_142_109_819),
                (7, -0.001_222_321_503_394_202_6),
                (1, 0.001_248_010_038_083_633_6),
                (4, 0.001_565_396_937_793_969_1),
            ],
            basket_offset_mv: 0.014_201_742_122_734_787,
            winner: 6,
            runner_up: 0,
            winner_ticks: [189, 216, 254],
            runner_up_ticks: [194, 222, 260],
        },
        SeedFixture {
            seed: 43,
            cells_and_offsets: [
                (5, 0.024_290_574_549_655_04),
                (4, -0.049_795_955_278_034_12),
                (3, -0.022_716_828_078_133_706),
                (2, -0.014_638_234_533_801_742),
                (6, 0.000_416_432_514_401_176_44),
                (7, 0.005_143_283_623_670_704),
                (0, 0.006_105_737_666_794_002),
                (1, 0.016_954_458_001_971_397),
            ],
            basket_offset_mv: -0.019_589_871_912_640_337,
            winner: 5,
            runner_up: 4,
            winner_ticks: [189, 216, 254],
            runner_up_ticks: [193, 222, 259],
        },
    ];

    #[test]
    fn identity_words_match_checkpoint_v1_and_drive_executed_constants() {
        let identity = TEMPORAL_PHYSICS_IDENTITY;
        assert_eq!(
            identity.integration_step_ms.to_bits(),
            0x3fa9_9999_9999_999a
        );
        assert_eq!(identity.race_ticks, 600);
        assert_eq!(
            identity.physics_f64_bits(),
            [
                0x4020_f0a7_d905_53e6,
                0x3f33_ea95_a3d1_e76b,
                0x4039_a18e_521f_e42e,
                0x3fef_f44e_351e_f7e1,
                0x3fef_ccf5_acbf_ae90,
                0x4081_d000_0000_0000,
                0x4047_8000_0000_0000,
                0x4083_6000_0000_0000,
                0x4084_f000_0000_0000,
                0x4081_d000_0000_0000,
                0x40d3_8800_0000_0000,
                0x4034_0000_0000_0000,
                0x4108_6a00_0000_0000,
                0x407f_4000_0000_0000,
            ]
        );

        assert_eq!(
            CandidateClass::Proximal.current_pa().to_bits(),
            identity.proximal_pa.to_bits()
        );
        assert_eq!(
            CandidateClass::Relay.current_pa().to_bits(),
            identity.relay_pa.to_bits()
        );
        assert_eq!(
            CandidateClass::Novel { rank: 7, load: 255 }
                .current_pa()
                .to_bits(),
            (identity.novel_base_pa - identity.novel_rank_decrement_pa * 7.0
                + identity.homeostatic_numerator_pa / 256.0)
                .to_bits()
        );

        let candidate = CandidateDrive {
            cell_id: 7,
            current_pa: CandidateClass::Active { load: 0 }.current_pa(),
            threshold_offset_mv: 0.027_531_561_337_646_178,
        };
        let mut scratch = CoupledRaceScratch::try_new(1).unwrap();
        assert_eq!(
            coupled_first_spike_race(
                &[candidate],
                BasketDrive {
                    cell_id: 8,
                    current_pa: identity.basket_current_pa,
                    threshold_offset_mv: 0.0,
                    inhibition_pa: identity.basket_inhibition_pa,
                },
                &mut scratch,
            ),
            Ok(CoupledRaceTicks {
                winner: SpikeEvent {
                    cell_id: 7,
                    tick: 173,
                },
                basket_cell_id: 8,
                basket_tick: 174,
                inhibition_onset_tick: 175,
                simulated_ticks: 174,
            })
        );
    }

    #[test]
    fn singleton_proximal_latency_matches_python_oracle() {
        let ticks = singleton_latency_ticks(
            CandidateClass::Proximal.current_pa(),
            0.027_531_561_337_646_178,
        )
        .expect("the calibrated proximal current must spike");

        assert_eq!(ticks, 237);
    }

    #[test]
    fn singleton_preserves_python_compiled_response_rounding_order() {
        // This is the first nextafter boundary where direct current-scaled
        // integration crosses at tick 2 but the production Python oracle's
        // compiled-unit-response path crosses at tick 3.
        assert_eq!(
            singleton_latency_ticks(42_300.275_603_542_68, 0.027_531_561_337_646_178,),
            Ok(3),
        );
    }

    #[test]
    fn singleton_current_classes_and_homeostatic_boundaries_match_python_oracle() {
        let threshold_offset_mv = 0.027_531_561_337_646_178;
        let fixtures = [
            (CandidateClass::Novel { rank: 0, load: 0 }, 590.0, Some(223)),
            (
                CandidateClass::Novel { rank: 7, load: 255 },
                241.078_125,
                None,
            ),
            (CandidateClass::Matching { load: 0 }, 640.0, Some(194)),
            (
                CandidateClass::Matching { load: 255 },
                620.078_125,
                Some(205),
            ),
            (CandidateClass::Active { load: 0 }, 690.0, Some(173)),
            (CandidateClass::Active { load: 255 }, 670.078_125, Some(181)),
            (CandidateClass::Proximal, 570.0, Some(237)),
            (CandidateClass::Relay, 20_000.0, Some(5)),
        ];

        for (class, expected_current_pa, expected_ticks) in fixtures {
            let current_pa = class.current_pa();
            assert_eq!(current_pa, expected_current_pa, "{class:?}");
            assert_eq!(
                singleton_latency_ticks(current_pa, threshold_offset_mv).ok(),
                expected_ticks,
                "{class:?}",
            );
        }
    }

    #[test]
    fn coupled_seed7_active_boundary_matches_python_oracle() {
        let candidates = [
            CandidateDrive {
                cell_id: 7,
                current_pa: 650.0,
                threshold_offset_mv: 0.033_505_381_138_863_34,
            },
            CandidateDrive {
                cell_id: 5,
                current_pa: 640.0,
                threshold_offset_mv: -0.024_791_163_874_911_56,
            },
            CandidateDrive {
                cell_id: 3,
                current_pa: 590.0,
                threshold_offset_mv: -0.022_264_795_968_931_855,
            },
            CandidateDrive {
                cell_id: 4,
                current_pa: 543.0,
                threshold_offset_mv: -0.011_366_769_629_293_064,
            },
            CandidateDrive {
                cell_id: 2,
                current_pa: 496.0,
                threshold_offset_mv: -0.006_853_446_384_055_439,
            },
            CandidateDrive {
                cell_id: 0,
                current_pa: 449.0,
                threshold_offset_mv: 0.000_030_753_833_937_064_36,
            },
            CandidateDrive {
                cell_id: 6,
                current_pa: 402.0,
                threshold_offset_mv: 0.001_503_590_064_935_962_2,
            },
            CandidateDrive {
                cell_id: 1,
                current_pa: 355.0,
                threshold_offset_mv: 0.007_468_638_437_711_748,
            },
        ];
        let basket = BasketDrive {
            cell_id: 8,
            current_pa: 200_000.0,
            threshold_offset_mv: -0.012_305_162_963_783_242,
            inhibition_pa: 500.0,
        };
        let mut scratch = CoupledRaceScratch::try_new(candidates.len()).unwrap();

        assert_eq!(
            coupled_first_spike_race(&candidates, basket, &mut scratch),
            Ok(CoupledRaceTicks {
                winner: SpikeEvent {
                    cell_id: 7,
                    tick: 189
                },
                basket_cell_id: 8,
                basket_tick: 190,
                inhibition_onset_tick: 191,
                simulated_ticks: 600,
            }),
        );
    }

    #[test]
    fn coupled_five_seed_three_boundary_matrix_matches_literal_python_fixtures() {
        for fixture in SEED_FIXTURES {
            for (scenario, currents_pa) in SCENARIO_CURRENTS_PA.into_iter().enumerate() {
                let candidates: [CandidateDrive; 8] = std::array::from_fn(|index| {
                    let (cell_id, threshold_offset_mv) = fixture.cells_and_offsets[index];
                    CandidateDrive {
                        cell_id,
                        current_pa: currents_pa[index],
                        threshold_offset_mv,
                    }
                });
                let basket = BasketDrive {
                    cell_id: 8,
                    current_pa: 200_000.0,
                    threshold_offset_mv: fixture.basket_offset_mv,
                    inhibition_pa: 500.0,
                };
                let mut scratch = CoupledRaceScratch::try_new(candidates.len()).unwrap();
                let winner_tick = fixture.winner_ticks[scenario];

                assert_eq!(
                    coupled_first_spike_race(&candidates, basket, &mut scratch),
                    Ok(CoupledRaceTicks {
                        winner: SpikeEvent {
                            cell_id: fixture.winner,
                            tick: winner_tick,
                        },
                        basket_cell_id: 8,
                        basket_tick: winner_tick + 1,
                        inhibition_onset_tick: winner_tick + 2,
                        simulated_ticks: 600,
                    }),
                    "seed {} scenario {scenario}",
                    fixture.seed,
                );

                let control_basket = BasketDrive {
                    inhibition_pa: 0.0,
                    ..basket
                };
                assert_eq!(
                    coupled_first_spike_race(&candidates[..2], control_basket, &mut scratch,),
                    Err(super::TemporalPhysicsStop::MultiplePyramidalSpikes {
                        first: SpikeEvent {
                            cell_id: fixture.winner,
                            tick: winner_tick,
                        },
                        second: SpikeEvent {
                            cell_id: fixture.runner_up,
                            tick: fixture.runner_up_ticks[scenario],
                        },
                        spike_count: 2,
                    }),
                    "control seed {} scenario {scenario}",
                    fixture.seed,
                );
            }
        }
    }

    #[test]
    fn coupled_same_tick_tie_orders_margin_then_physical_cell_id_before_stop() {
        let tied = [
            CandidateDrive {
                cell_id: 9,
                current_pa: 650.0,
                threshold_offset_mv: 0.0,
            },
            CandidateDrive {
                cell_id: 3,
                current_pa: 650.0,
                threshold_offset_mv: 0.0,
            },
        ];
        let basket = BasketDrive {
            cell_id: 0,
            current_pa: 200_000.0,
            threshold_offset_mv: 0.0,
            inhibition_pa: 0.0,
        };
        let mut scratch = CoupledRaceScratch::try_new(tied.len()).unwrap();

        assert_eq!(
            coupled_first_spike_race(&tied, basket, &mut scratch),
            Err(super::TemporalPhysicsStop::MultiplePyramidalSpikes {
                first: SpikeEvent {
                    cell_id: 3,
                    tick: 189,
                },
                second: SpikeEvent {
                    cell_id: 9,
                    tick: 189,
                },
                spike_count: 2,
            }),
        );
    }

    #[test]
    fn coupled_no_spike_and_missing_inhibition_are_fail_closed() {
        let basket = BasketDrive {
            cell_id: 0,
            current_pa: 200_000.0,
            threshold_offset_mv: 0.0,
            inhibition_pa: 500.0,
        };
        let mut scratch = CoupledRaceScratch::try_new(1).unwrap();
        assert_eq!(
            coupled_first_spike_race(
                &[CandidateDrive {
                    cell_id: 9,
                    current_pa: 0.0,
                    threshold_offset_mv: 0.0,
                }],
                basket,
                &mut scratch,
            ),
            Err(super::TemporalPhysicsStop::NoPyramidalSpike),
        );

        assert_eq!(
            coupled_first_spike_race(
                &[CandidateDrive {
                    cell_id: 9,
                    current_pa: 20_000.0,
                    threshold_offset_mv: 0.0,
                }],
                BasketDrive {
                    current_pa: 0.0,
                    ..basket
                },
                &mut scratch,
            ),
            Err(super::TemporalPhysicsStop::BasketInhibitionMissing {
                winner: SpikeEvent {
                    cell_id: 9,
                    tick: 5
                },
                basket_tick: None,
                inhibition_onset_tick: None,
            }),
        );
    }

    #[test]
    fn coupled_race_scratch_rejects_invalid_capacity_before_reserving() {
        let mut reserve_calls = 0usize;
        assert!(matches!(
            CoupledRaceScratch::try_new_with_reserve(0, |_, _| {
                reserve_calls += 1;
                false
            }),
            Err(CoupledRaceScratchBuildError::ZeroCapacity)
        ));
        assert!(matches!(
            CoupledRaceScratch::try_new_with_reserve(usize::MAX, |_, _| {
                reserve_calls += 1;
                false
            }),
            Err(CoupledRaceScratchBuildError::CapacityOverflow)
        ));
        assert_eq!(reserve_calls, 0);

        assert!(matches!(
            CoupledRaceScratch::try_new(0),
            Err(CoupledRaceScratchBuildError::ZeroCapacity)
        ));
        assert!(matches!(
            CoupledRaceScratch::try_new(usize::MAX),
            Err(CoupledRaceScratchBuildError::CapacityOverflow)
        ));
    }

    #[test]
    fn coupled_race_scratch_allocation_failure_is_typed_and_retryable_at_every_reserve() {
        const EXPECTED: [(CoupledRaceScratchReserveSite, usize); 5] = [
            (CoupledRaceScratchReserveSite::Voltage, 3),
            (CoupledRaceScratchReserveSite::Thresholds, 3),
            (CoupledRaceScratchReserveSite::Excitatory, 3),
            (CoupledRaceScratchReserveSite::Emitted, 3),
            (CoupledRaceScratchReserveSite::SpikingIndices, 3),
        ];

        for (failed_index, &(failed_site, _)) in EXPECTED.iter().enumerate() {
            let mut calls = Vec::new();
            let error = match CoupledRaceScratch::try_new_with_reserve(3, |site, element_count| {
                calls.push((site, element_count));
                site == failed_site
            }) {
                Ok(_) => panic!("the selected reserve must fail"),
                Err(error) => error,
            };
            assert_eq!(
                error,
                CoupledRaceScratchBuildError::RetryableAllocation(failed_site)
            );
            assert_eq!(calls, EXPECTED[..=failed_index]);
        }

        let mut retry_calls = Vec::new();
        let retry = CoupledRaceScratch::try_new_with_reserve(3, |site, element_count| {
            retry_calls.push((site, element_count));
            false
        })
        .expect("a fresh identical retry must reserve every race buffer");
        assert_eq!(retry_calls, EXPECTED);
        assert_eq!(retry.voltage_mv.as_slice(), &[0.0; 3]);
        assert_eq!(retry.thresholds_mv.as_slice(), &[0.0; 3]);
        assert_eq!(retry.excitatory_pa.as_slice(), &[0.0; 3]);
        assert_eq!(retry.emitted.as_slice(), &[0; 3]);
        assert_eq!(retry.spiking_indices.as_slice(), &[0; 3]);
        assert!(retry
            .allocation_signature()
            .into_iter()
            .all(|(_, capacity)| capacity >= 3));
    }

    #[test]
    fn coupled_race_scratch_never_reallocates_on_success_or_stop() {
        let candidates = [
            CandidateDrive {
                cell_id: 3,
                current_pa: 650.0,
                threshold_offset_mv: 0.0,
            },
            CandidateDrive {
                cell_id: 9,
                current_pa: 640.0,
                threshold_offset_mv: 0.0,
            },
            CandidateDrive {
                cell_id: 12,
                current_pa: 0.0,
                threshold_offset_mv: 0.0,
            },
        ];
        let basket = BasketDrive {
            cell_id: 20,
            current_pa: 200_000.0,
            threshold_offset_mv: 0.0,
            inhibition_pa: 500.0,
        };
        let mut scratch = CoupledRaceScratch::try_new(candidates.len()).unwrap();
        let allocation_signature = scratch.allocation_signature();

        for _ in 0..4 {
            assert!(coupled_first_spike_race(&candidates, basket, &mut scratch).is_ok());
            assert_eq!(scratch.allocation_signature(), allocation_signature);
        }
        assert!(matches!(
            coupled_first_spike_race(
                &candidates,
                BasketDrive {
                    inhibition_pa: 0.0,
                    ..basket
                },
                &mut scratch,
            ),
            Err(super::TemporalPhysicsStop::MultiplePyramidalSpikes { .. })
        ));
        assert_eq!(scratch.allocation_signature(), allocation_signature);
    }
}
