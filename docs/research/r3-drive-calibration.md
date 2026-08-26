# R3 drive calibration — measured drive-vs-N scaling law and C1 Change Note CN-1

**Status: PRE-LAUNCH DECISION FROZEN (sanctioned deviation, 2026-08-26).**
Resolves Blocker 1 of `experiments/r3_pilot/C1_READINESS.md` (@ 10453e0):
the drive-calibration choice required before the first C1 token. Nothing here
was tuned post-launch; this note freezes constants and one companion defect
fix BEFORE any cohort leg runs.

Prereg: `docs/research/r3-segment-growth-preregistration.md` §4–§6 ·
Harness: `experiments/r3_pilot/calibrate_drive.py` · Artifact:
`experiments/r3_pilot/calib_curve.json` (19+3 probes + ≥4096-step
confirmation) · Fixture anchor: `gs0_health_run.json` (N=96, mean |W| ≈ 53.8,
density 0.00555).

## TL;DR — the frozen decision

| item | decision |
|---|---|
| Drive calibration | **Constant gain at scale: `input_current_pa` 5,000 → 250,000 pA (= ×50) at EVERY lane size**, not a g(N) curve. All other lane constants stay frozen (`input_fanout=24`, `recurrent_fanout=8`, `recurrent_current_pa=650`, `inhibitory_fraction=0.2`, `input_pulse_ms=1.5`, `token_interval_ms=5`). |
| Companion fix (required for the gate to see signal) | Enable per-token device→host mirroring of `token_spike_counts` whenever an observation sink is attached (exact patch below). Without it the CUDA backend (auto-selected at N ≥ 100,000) feeds the sink **zeros while the lane fires normally** — the reported "silent at N = 131,072" state is this telemetry defect PLUS real recruitment dilution, not cell silence. |
| Evidence the choice clears prereg §6 at C1 scale | N = 131,072, ×50 gain: mean \|W\| = 64,054 cells/token, density 4.89e−3 ≥ 0.001 floor, empty-winner fraction 0.00 over 200 tokens; ≥ 4,096-step confirmation in `calib_curve.json`. |

## What was measured

`calibrate_drive.py` replays the **exact G-S0 fixture slice** (first 200 TRAIN
concept tokens; slice sha `de5d527b…` asserted equal to
`gs0_health_run.json`) through a fresh `RecurrentCellCircuit` per (N, g)
point using the glue driver's own construction path (`RecurrentCircuitConfig`
with only `input_current_pa` scaled) and its own `SinkSpikeRecorder`, so
`mean_activity_density` / winner sets are computed exactly as the C1
`spike_preflight_gate` computes them. Grid: N ∈ {96-ref, 1024, 8192, 65536,
131072} × g ∈ {1, 50, 150, 300} (g multiplies the frozen 5,000 pA afferent
drive), plus forced cpu/cuda cross-checks at N ≥ 100,000 where the auto
device policy switches kernels.

### Results (mean winner-set size |W| / density of analytic ceiling / total lane spikes per 200-token replay)

| N | g=1 | g=50 | g=150 | g=300 |
|---:|---|---|---|---|
| 96 | 53.0 / 5.52e−3 / 10,604 | – | – | – |
| 1,024 | 559 / 5.46e−3 / 111,798 | 718 / 7.01e−3 | 753 / 7.35e−3 | 755 / 7.38e−3 |
| 8,192 | 4,125 / 5.04e−3 / 825,080 | 4,598 / 5.61e−3 | 4,728 / 5.77e−3 | 4,737 / 5.78e−3 |
| 65,536 | **21.5 / 3.28e−6** ← no ignition | 33,112 / 5.05e−3 | 34,721 / 5.30e−3 | 34,741 / 5.30e−3 |
| 131,072 (cpu) | **20.2 / 1.54e−6** ← no ignition | **64,054 / 4.89e−3** ✓ | 68,464 / 5.22e−3 | 68,475 / 5.22e−3 |
| 131,072 (cuda/auto) | sink \|W\|=0 **artifact** (lane fired 4,044 spikes) | sink \|W\|=0 artifact (lane fired 12,810,797 spikes) | artifact (13,692,764) | artifact (13,694,940) |

Cross-backend identity check: at every g the CUDA and CPU lanes produced
**bit-identical total spike counts** over the same replay (4,044 / 12,810,797
/ 13,692,764 / 13,694,940) — the backends compute identical physics; only the
host-side readback differs.

## Why the frozen defaults are silent at scale (mechanism)

Three independent findings combine into the full explanation:

1. **The afferent stage is N-independent and caps winners at 24.** Each token
   drives exactly `input_fanout = 24` random cells with 5000·g pA for 30
   sub-steps (1.5 ms). Offline single-cell integration of the pinned GLIF5
   equations gives sustained rheobase ≈ 119 pA (inside Stage B's 80–150 pA
   band); under the 1.5 ms pulse a driven cell fires **exactly one spike**
   for any g ≥ ≈1 (first-spike latency falls from 1.30 ms at g=1 to the
   0.05 ms floor at g ≥ 50), fires nothing for g < ≈0.7, and can never fire
   twice (`spike_cut_steps = 113` > 100 sub-steps/token). Direct drive can
   therefore contribute at most 24 winners at ANY N — it cannot itself
   restore signal at scale.
2. **Recruitment dilution: recurrent overlap falls ∝ 1/N.** Signal beyond
   those 24 cells is a branching cascade: each spiking cell sprays
   `recurrent_fanout = 8` edges of ±650 pA (τ = 8 ms decay) onto uniform
   random targets, and a quiescent cell fires only when several excitatory
   edges land inside its integration window. Expected edges per cell is
   8S/N for S current spikers — for fixed seed count S₀ ≤ 24 the collision
   rate thins linearly in N (expected edge-pair collisions among layer-1's
   192 edges: ≈ 192²/(2N) ≈ 2.3 at N=8,192 but 0.28 at N=65,536). The
   cascade reproduces itself only above a critical population size N_c(g);
   measured: ignited at N ≤ 8,192 for g=1, dead at N ≥ 65,536 for g=1,
   reignited at both by g=50. This — NOT growing inhibition — is why drive
   must scale with N: `inhibitory_fraction` is frozen at 0.2 at every N, so
   inhibition scales WITH the population while effective per-cell excitation
   from fixed-fanout seeds does not.
3. **Once ignited, the regime is scale-invariant — which is why the fixture
   lied for us.** Above ignition the cascade saturates near a stable
   fraction of the population (mean |W|/N ≈ 49–70 % across all probed N;
   density of the analytic ceiling plateaus at 5.0–7.4e−3), so the N=96
   fixture sat comfortably inside the same dynamical regime C1 needed — and
   passed §6 — while the launch size fell on the wrong side of the ignition
   boundary and returned only the ≤ 24 directly-driven cells (density
   ∝ 1/N → 1.5e−6, far under the 1e−3 floor). Gain above 50 buys almost
   nothing (density 4.89e−3 → 5.22e−3 from ×50 to ×300; winner sets
   +7 %): ×50 already sits on the plateau, with margin against the
   unsampled sharpness of the N_c(g) boundary between g=1 (dead at 65k) and
   g=50 (ignited at 131k).

Not disentangled (and irrelevant to the freeze): whether ×50 ignites via the
earlier seed phase (first spike at 0.05 ms vs 1.30 ms leaves ~34 % more of
the token window for cascade generations) or via near-critical amplification
of that extra window. Both act only through seed timing since per-cell spike
count is capped; the empirical map is decisive either way.

## Why constant-g instead of the measured g(N) curve

- The curve is a **step, not a law**: g(N) would be "1 below N_c ≈ 10⁴–10⁵,
  ≥ 50 above", with the boundary unsampled between 8,192 and 65,536 and
  sharp (21 → 33,032 winners across the boundary at fixed g). A piecewise
  freeze risks silent re-collapse if corpus/tokenizer drift moves effective
  recruitment slightly; a single ×50 scalar carries margin at zero measured
  cost (throughput at N=131,072 CUDA: ~174 steps/s at g=1 vs ~134–141 at
  g ≥ 50 — immaterial against the tracker-dominated wall-clock budget).
- Winner-set *fractions* stay in the fixture-validated band everywhere
  (G-S0: 55 % of N at g=1; ×50 at 131,072: 49 %), so arm mechanics calibrated
  at the fixture keep operating in their validated regime at launch size.
- One scalar keeps the prereg's frozen-constants shape: a single value
  changes, no scale-dependent rule enters the protocol.

Rejected alternatives: scaling `input_fanout` ∝ N (changes the frozen
token→cell mapping semantics, not sanctioned); amending the §6 density metric
instead of the drive (unnecessary — the floor is met as written once signal
exists); running C1 on the CPU backend only to dodge the mirroring defect
(~14 steps/s at 131,072; the cohort cannot fit the cap regardless — Blocker 3
territory).

## Sanctioned companion fix (pre-launch, required)

The §6 gate reads `circuit._token_spike_counts`. On the CUDA path this host
bank is mirrored per token only when `feature_family` includes
`"spike_counts"` (`recurrent.py::_bind_native_kernel` →
`mirror_token_spike_counts="spike_counts" in self.feature_members`); the r3
glue/runner uses the default `"traces"` family, so at N ≥ 100,000 the sink
reads zeros forever even while `observe()` reports millions of spikes
(evidence rows: `telemetry_consistent=false` in `calib_curve.json`). This is
a readback defect, not a physics change — totals are bit-identical across
backends — and fixing it touches no frozen constant:

1. `src/humanbrain/recurrent.py::_bind_native_kernel`:

```python
# CN-1 (r3-drive-calibration.md): observation-sink consumers read this bank
# per token; without mirroring, CUDA arms (N >= 100k cells) feed the sink
# zeros. Mirroring adds one DtoH copy of N u32 per token (~512 KB at 131,072).
mirror_token_spike_counts=(
    "spike_counts" in self.feature_members
    or self._observation_sink is not None
),
```

2. `experiments/r3_pilot/run_c1_cohort.py::ArmEngine.__init__`: attach the
   recorder through the constructor kwarg (`RecurrentCellCircuit(...,
   observation_sink=self.recorder)`) instead of setting the property after
   construction — the flag is evaluated at native-bind time (constructor
   binds at `recurrent.py:579`; checkpoint restore re-binds at line 995 and
   will see an attached sink).
3. `tests/test_cuda.py` pins traces-arm mirror payloads; if it asserts the
   exact mirrored-array set, extend the pin for sink-attached circuits.

## Pre-launch checklist deltas (for C1_READINESS.md)

- [x] **Drive-calibration decision recorded** — THIS note: ×50 constant
      (`input_current_pa := 250,000 pA`) at every lane size, frozen before
      the first token.
- [ ] Parent applies the two-line companion fix + test-pin update, reruns
      `run_c1_cohort.py --dry-run` (must reproduce bit-identical prefix
      telemetry on the CPU path and non-empty sink winner sets at any CUDA
      probe), then a short calibrated smoke at reduced N per the existing
      checklist.
- [ ] Blockers 2–3 (tracker recognition cost, wall-clock reconciliation) are
      untouched by this note and remain open.

Falsification conditions for this freeze (any ⇒ STOP, do not tune silently):
dry-run pre-flight density < 0.001 or empty-winner fraction > 0 at 131,072
under auto policy after the mirror fix; confirmation row in
`calib_curve.json` failing `prereg_gate_would_pass`.

## Provenance

Measured 2026-08-26, worktree `human-brain-path2-pilot` @ `e3ecf3f` parent,
RTX 4070 SUPER (12 GB) + native CPU kernels; construction seed 17; harness
deterministic (fixture slice sha asserted; per-row wall-clocks recorded).
Single-cell F–I numbers reproducible by direct evaluation of the pinned
`AllenGlif5Parameters` update equations (rheobase search, 1.5 ms pulse
response grid). Raw rows incl. per-g densities, throughput, and
telemetry-consistency flags: `experiments/r3_pilot/calib_curve.json`.
