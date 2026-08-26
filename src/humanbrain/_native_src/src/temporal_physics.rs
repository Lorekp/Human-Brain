//! Exact fitted-GLIF3 first-spike physics for the native R3 temporal runtime.
//!
//! This module ports the scalar equations and event ordering in
//! `humanbrain.cortex.DendriticCortex` without importing token, column, segment,
//! or winner-selection state. Callers supply physical cell IDs, currents, and
//! their frozen threshold offsets. All observable times are integer 0.05-ms
//! ticks. The fitted reset, refractory, and after-spike-current rules cannot
//! affect this seam because each cell is observed only through its first spike.

use std::sync::OnceLock;

const RACE_TICKS: u16 = 600;
// Exact float64 words produced by the Python oracle for the fitted Allen cell
// and the 35-ms NMDA / 8-ms GABAA decays. Keeping the words literal prevents a
// platform libm call from moving a threshold crossing by one tick.
const LEAK_NS: f64 = f64::from_bits(0x4020_f0a7_d905_53e6);
const VOLTAGE_STEP_SCALE: f64 = f64::from_bits(0x3f33_ea95_a3d1_e76b);
const THRESHOLD_RELATIVE_MV: f64 = f64::from_bits(0x4039_a18e_521f_e42e);
const NMDA_DECAY: f64 = f64::from_bits(0x3fef_f44e_351e_f7e1);
const GABAA_DECAY: f64 = f64::from_bits(0x3fef_ccf5_acbf_ae90);

struct NmdaUnitResponse {
    voltage_mv: [f64; RACE_TICKS as usize],
    rising_len: usize,
}

fn nmda_unit_response() -> &'static NmdaUnitResponse {
    static RESPONSE: OnceLock<NmdaUnitResponse> = OnceLock::new();
    RESPONSE.get_or_init(|| {
        let mut voltage_mv = [0.0; RACE_TICKS as usize];
        let mut voltage = 0.0;
        let mut current = 1.0;
        let mut peak_index = 0;
        let mut peak_voltage = f64::NEG_INFINITY;
        for (index, sample) in voltage_mv.iter_mut().enumerate() {
            voltage += VOLTAGE_STEP_SCALE * (current - LEAK_NS * voltage);
            *sample = voltage;
            if voltage > peak_voltage {
                peak_voltage = voltage;
                peak_index = index;
            }
            current *= NMDA_DECAY;
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
                570.0 - 47.0 * f64::from(rank) + homeostatic_current_pa(load)
            }
            Self::Matching { load } => 620.0 + homeostatic_current_pa(load),
            Self::Active { load } => 670.0 + homeostatic_current_pa(load),
            Self::Proximal => 570.0,
            Self::Relay => 20_000.0,
        }
    }
}

#[inline]
fn homeostatic_current_pa(load: u8) -> f64 {
    20.0 / (1.0 + f64::from(load))
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

impl CoupledRaceScratch {
    pub(crate) fn try_new(max_candidates: usize) -> Result<Self, &'static str> {
        if max_candidates == 0 {
            return Err("coupled-race scratch requires nonzero capacity");
        }

        fn filled<T: Copy>(length: usize, value: T) -> Result<Vec<T>, &'static str> {
            let mut values = Vec::new();
            values
                .try_reserve_exact(length)
                .map_err(|_| "coupled-race scratch allocation failed")?;
            values.resize(length, value);
            Ok(values)
        }

        Ok(Self {
            voltage_mv: filled(max_candidates, 0.0)?,
            thresholds_mv: filled(max_candidates, 0.0)?,
            excitatory_pa: filled(max_candidates, 0.0)?,
            emitted: filled(max_candidates, 0)?,
            spiking_indices: filled(max_candidates, 0)?,
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
    let response_needed = (THRESHOLD_RELATIVE_MV + threshold_offset_mv) / current_pa;
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
        scratch.thresholds_mv[index] = THRESHOLD_RELATIVE_MV + candidate.threshold_offset_mv;
        scratch.excitatory_pa[index] = candidate.current_pa;
        scratch.emitted[index] = 0;
    }
    let mut inhibitory_pa = 0.0;
    let mut basket_voltage_mv = 0.0;
    let mut basket_excitation_pa = 0.0;
    let basket_threshold_mv = THRESHOLD_RELATIVE_MV + basket.threshold_offset_mv;
    let mut basket_tick = None;
    let mut inhibition_onset_tick = None;
    let mut first_spike = None;
    let mut second_spike = None;
    let mut spike_count = 0usize;
    let mut simulated_ticks = 0;

    for tick in 1..=RACE_TICKS {
        simulated_ticks = tick;
        for index in 0..candidates.len() {
            scratch.voltage_mv[index] += VOLTAGE_STEP_SCALE
                * (scratch.excitatory_pa[index]
                    - inhibitory_pa
                    - LEAK_NS * scratch.voltage_mv[index]);
        }
        if basket_tick.is_none() {
            basket_voltage_mv +=
                VOLTAGE_STEP_SCALE * (basket_excitation_pa - LEAK_NS * basket_voltage_mv);
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
            *current_pa *= NMDA_DECAY;
        }
        basket_excitation_pa *= NMDA_DECAY;
        inhibitory_pa *= GABAA_DECAY;
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
        CandidateDrive, CoupledRaceScratch, CoupledRaceTicks, SpikeEvent,
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
