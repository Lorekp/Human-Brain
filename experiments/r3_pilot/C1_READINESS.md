# C1 readiness — r3 segment-growth cohort (`run_c1_cohort.py`)

**Status: BLOCKED BEFORE LAUNCH — a drive-calibration decision is required first,
and as measured the schedule cannot fit the frozen 15 h cap even after it.**
Nothing here may be tuned silently post-launch: prereg §"frozen before
implementation" locks constants from the first replay token, so both findings
below are pre-launch decisions, not mid-cohort fixes.

Prereg: `docs/research/r3-segment-growth-preregistration.md` §4–§8 ·
Runner: `experiments/r3_pilot/run_c1_cohort.py` (branch `codex/path2-pilot`) ·
G-S0 evidence: `gs0_health_run.json` (all checks green at N=96).

## Launch command

```bash
# wiring verification only (no long legs) — verified OK below
C:/Users/laure/Desktop/DeepSim/.venv/Scripts/python.exe \
    experiments/r3_pilot/run_c1_cohort.py --dry-run

# the cohort itself (DO NOT LAUNCH until the checklist at the bottom clears)
C:/Users/laure/Desktop/DeepSim/.venv/Scripts/python.exe \
    experiments/r3_pilot/run_c1_cohort.py \
        --output-dir experiments/r3_pilot/c1_run
# defaults: --arms A1,A2,A3 --lane-cells 131072 --exposure-passes 3
#           --wall-clock-cap-hours 15 ; resume-safe per arm at book boundaries
```

Exit codes: `0` completed · `2` protocol STOP / pre-flight BLOCKED / gate verdict ·
`3` wall-clock abort (partial logs written) · `4` resume context invalid.
Manifests: `c1_run.json` / `c1_dry_run.json` under `--output-dir`.

## What the dry-run verified (2026-08-26, all green)

Corpus pins assert-or-STOP (trimmed sha256 `d585e564…`, 66 books = 49 TRAIN +
17 HELD-OUT roster exact); frozen tokenizer stack re-derived byte-identically;
streams **TRAIN 2,117,412 tokens / HELD-OUT 458,191 tokens**, max id 1535,
feasibility `(1536)·32 = 49,152 ≤ 131,072 + 49,152 = 180,224` cells; three arms
wired (pool mapping, tracker variants incl. the A3 union-sampler, observation
sink) and replaying a 256-token TRAIN prefix with the A1/A2 telemetry hashes
**bit-identical to the reference glue driver** (`r3_pilot_glue.run_arm`);
A2 structural-zero tripwire held; checkpoint save→restore round-trip exact on
all three arms; spike-rate health at fixture scale density 0.00555 ≥ floor
(matches G-S0). Decisions recorded in-file before any token: **W_{t-1}
re-affirmed as the lane-spike-stream definition** (tracker-internal election is
arithmetically sterile at θ_m=10 vs a one-element W — see runner docstring) and
the **A3 operationalization** (growth targets sampled without replacement from
the cumulative union of lane-active cells), both disclosed per open-deviation
discipline.

## Blocker 1 — drive calibration (REQUIRED BEFORE ANY LEG)

The FROZEN default lane constants are **silent at N = 131,072**: replaying the
real token stream through the observation-sink driver fails the prereg §6
spike-rate pre-flight with **mean winner-set size 0** — every burst then
allocates blind and f_pred is structurally 0, so the cohort would spend itself
proving arithmetic. This reproduces the Stage B cell physics finding
(`codex/path3-repetition`, commit `c316865`): fitted GLIF3 cells have
**rheobase ≈ 80–150 pA and fire ZERO spikes under default drive (≤ 6 Hz even at
2400 pA)**; gradients/spikes have nothing to flow through until drive is
calibrated (Stage B's measured mitigation direction: ×150–300 init gain).

Scale probe (same day, RTX 4070 SUPER): the launch configuration
**(lane 131,072 + pools 49,152 = 180,224 cells) builds in 1.6 s on CUDA and
sustains ~193 lane steps/s** — capacity is NOT the problem; signal is.
Fixture contrast: at N=96 the same driver yields mean |W| ≈ 53.8 and passes the
0.1 % population-rate floor comfortably.

**Decision required:** choose and freeze a calibrated input-drive setting that
clears the prereg §6 floor at N = 131,072 (population rate ≥ 0.1 % of analytic
ceiling AND mean winner-set size > 0 over a ≥ 4,096-step prefix — the runner
hard-stops otherwise via `spike_preflight_gate`). Options on the table:
calibrated afferent gain (F–I-curve-matched to rheobase at this N), or an
explicit protocol amendment. Either way it must be recorded before the first
token, because constants freeze at launch.

## Blocker 2 — reference-tracker scan cost (independent of drive)

`SegmentGrowthTracker.step` runs a **global recognition pass over every live
segment each step** (`for cell in sorted(self._segments)` + per-segment
`np.isin`). Measured today at N = 131,072, |W| = 54:

| live segments | ms/step | one 1M-step pass |
|---:|---:|---:|
| 125 | 3.1 | 0.85 h |
| 500 | 12.2 | 3.4 h |
| 1,000 | 24.9 | 6.9 h |
| 2,000 | 50.3 | 14.0 h |
| 8,000 | 278.5 | 77.4 h |

Linear ≈ 25 ms per 1,000 segments (superlinear beyond ~2k). Dry-run allocation
rate is ~0.35 segments/step early on, so exposure pass 1 alone drives the cost
past the 2,000-segment row while the lane still costs 1/193 s ≈ 5.2 ms/step.
The tracker, not the GPU kernel, dominates wall-clock by ≥ 40×.

## Wall-clock projection vs the frozen 15 h cap

Schedule per arm: epoch-0 + E=3 exposure passes over TRAIN + final sweep over
TRAIN ∪ HELD-OUT = 4 × 2,117,412 + 458,191 = **8,927,839 steps/arm**;
three serialized arms = **26,783,517 steps**.

- Lane kernel alone at 193 steps/s: **≈ 38.5 h — already 2.6× the cap**
  before any tracker cost.
- Adding the reference tracker at a *frozen* 1,000 live segments
  (~30 ms/step total): ≈ 220 h ≈ 15× the cap. Realistic growth makes it far
  worse.

Consequence: even after drive calibration returns a signal, launching the
cohort as specified aborts at the 15 h cap with partial logs (exit 3). The
pre-launch decisions this forces, in order: (a) drive calibration value;
(b) vectorize/index the recognition pass (host-side batching or candidate
narrowing to the arriving pool's segments plus a connected-hit index) so the
tracker stops scaling globally; (c) reconcile the arm serialization with the
cap (faster kernel path, or parallel circuits if VRAM allows) — a protocol-
level choice to escalate, since the prereg forbids silent extensions and
downscaled reruns inside a cohort slot. The runner's continuous projection
(`WallClockGovernor.projection`, logged per arm) will surface all of this
live rather than at the deadline.

## Memory budget

- Circuit at launch size: 180,224 cells builds and replays within the 12 GB
  RTX 4070 SUPER box (measured probe); one engine alive at a time by design —
  the orchestrator frees circuit+tracker between arms.
- Tracker worst case: ≤ 1,536 ids ⇒ 49,152 pool cells × 255 segments × 20
  contacts × 16 B (int64 id + float64 perm) ≈ **4.0 GB absolute ceiling**;
  realistic state is orders lower (fixture rate ~0.35 allocations/step).
- Telemetry: rolling sink recorder O(1); window deque capped at 4,096 rows;
  packed sweep counters 1 B/step ⇒ ≈ 8.9 MB per arm for the whole schedule;
  A3 active-union bitmap 131,072 bits = 16 KB.
- Checkpoints per book boundary: tracker npz ∝ live contacts, lane envelope
  npz, counters npz, meta.json (cursor + digests) — atomic tmp+rename writes.

## Stop conditions (all mechanically enforced)

1. Corpus/split drift (sha256, 66/49/17, held-out roster) → STOP before build.
2. Feasibility `(max_id+1)·k > configured_cells` → STOP before build.
3. Epoch-0 structural sweep: any predicted step in any arm → STOP (§5).
4. A2 structural zero violated anywhere (predicted step, segment, contact) → STOP (§4).
5. Spike-rate pre-flight fail at step 4,096 of pass 1 → **BLOCKED** STOP (§6) —
   this is the gate that fires under today's frozen constants.
6. Wall-clock cap exceeded → abort, partial logs reported (§8, exit 3).
7. Gates scored once on the final sweep with Wilson z = 1.959964 / Newcombe
   MOVER-D; verdict mapping §7 verbatim: `NEGATIVE_NO_ENGAGEMENT` (¬G-S1),
   `KILL_PILOT_MEMORIZATION_ONLY` (G-S1 ∧ ¬G-S2), `C1_PASS` (all three).
   G-S1 ∧ G-S2 ∧ ¬G-S3 has no §7 terminal label; the ladder halts mechanically
   as `C1_FAIL_ATTRIBUTION_NO_C2` — reported once, never retried.
8. Resume context mismatch (corpus sha, seed, lane size, arms, tokenizer) → exit 4.

## Pre-launch checklist

- [ ] **Drive-calibration decision recorded** (Blocker 1) — value frozen before
      the first token; must clear the §6 floor at N = 131,072 for ≥ 4,096 steps.
- [ ] **Recognition-pass cost plan** (Blocker 2) — vectorized/indexed tracker or
      an explicit cap-fit amendment; without it the cohort cannot finish.
- [ ] Arm-serialization vs 15 h cap reconciled (escalated protocol decision).
- [x] **W_{t-1} = lane-spike-stream re-affirmed** (runner docstring, with the
      arithmetic argument; matches `gs0_health_run.json winner_set_source`).
- [x] **A3 operationalization disclosed** (cumulative-union shuffled targets,
      seeded `(construction_seed, step_index, cell_id)`; runner docstring).
- [x] Corpus/tokenizer pins asserted end-to-end (dry-run green, hashes above).
- [x] Glue anchor bit-identical; A2 structural-zero tripwire verified;
      checkpoint resume round-trip exact (dry-run evidence).
- [ ] After the two decisions: rerun `--dry-run`, then a short calibrated
      smoke at reduced N before committing the cohort slot.
