# R3 C1 prelaunch audit and repair decision

**Date:** 2026-08-26  
**Status:** **CURRENT — HARD NO-GO; C1 cohort unspent**  
**Scope:** independent verification of the path-2 handoff, merged C1 runner,
drive calibration, recognition prototype, frozen preregistration, and
cell-only architecture contract.

## Decision

Do **not** launch the current C1 runner and do not score any output it produces
as an R3 result. The runner is not a slower implementation of the frozen
protocol; it implements a materially different computational experiment.

Path 2 remains the mainline. The next work is a fixture-first repair that puts
segment recognition, plateau drive, winner election, and the scored output
inside simulated fitted-cell dynamics. C1 may be launched only after that
repair has its own pre-outcome Change Note, passes the gates below, and receives
an explicit user GO.

No C1 outcome data have been opened. The cohort slot is unspent.

## Evidence that survives this audit

- The path decision remains Path 2: trained-by-construction connectivity.
- Probe P-B passed its frozen comparison against shuffled bias, unigram, and
  majority-prior controls. Its smaller margin over the incumbent off arm keeps
  it promising rather than conclusive.
- The C1 corpus hash, 66-book roster, 49/17 TRAIN/HELD-OUT split, tokenizer
  reconstruction, pool-feasibility arithmetic, and construction seed are
  reproducible.
- The tokenizer-freeze §5 constants-only overlap diagnostic is committed as
  `experiments/r3_pilot/token_overlap_calibration.json` and rebuilds
  byte-identically. At 20 contacts, the concept-stack independent-context
  tails are P(X≥10)=4.162e-10 and P(X≥13)=1.786e-15.
- Generic segment-growth mechanics have deterministic unit evidence for novel
  winner allocation, punishment, and rerun parity.
- CN-1 measured a real scale-dependent recruitment transition and recorded
  `input_current_pa = 250,000` pA. Its CUDA sink defect was real. The generic
  observation-sink mirror, restore-time attachment seam, and live-dynamics
  eligibility access are corrected on master in `f721fa9`.
- The sparse recognition prototype appears bit-exact on the synthetic paths it
  exercised. Those results remain useful as a sparse-regime control only.

These facts do not license C1 because the failures below concern the identity
of the experiment and the validity of its held-out score.

## Launch blockers

| Area | Verified state | Required resolution |
| --- | --- | --- |
| Neuronal substrate | **Fail.** The runner uses a GLIF5 recurrent circuit only to produce an activity set. A Python tracker then computes segment matches and declares logical fired cells by integer threshold. | Physically simulate the pool cells; matched segments must inject plateau current and a fitted-cell/basket race must emit the winner. |
| Cell count claim | **Fail.** The 131,072 lane cells are simulated; the additional 49,152 pool IDs are Python identities above the circuit, not simulated cells. | Report materialized and simulated cells separately. Count a pool cell only when it has membrane state and participates in the race. |
| Frozen substrate | **Fail.** The prereg specifies the existing GLIF3 dendritic lane; the runner uses `RecurrentCellCircuit` (GLIF5) plus `SegmentGrowthTracker`. | Implement the GLIF3 path or file a pre-outcome amendment with an equally strict fitted-cell realization. |
| Epoch 0 | **Fail by construction.** A1/A3 grow during the sweep that must begin and remain at zero segments. | Run disposable, fresh, growth-off shadow state; assert zero predictions and structure; discard it before production construction. |
| Exposure preflight | **Bypassed.** It is armed only while the arm-global step counter is below 4,096, but epoch 0 already advanced that counter. | Use an exposure-local counter and run the actual production backend for 4,096 canonical TRAIN tokens. |
| Frozen evaluation | **Fail.** The final TRAIN∪HELD-OUT sweep calls the mutating tracker and learns from held-out books while scoring them. | Freeze/clone learned state and expose a read-only recognition/race operation. No allocation, reinforcement, punishment, or other plasticity during scoring. |
| Winner-set definition | **Unfrozen deviation.** The prereg's elected/predicted winner definition is arithmetically sterile at θm=10; the runner substitutes the GLIF5 lane spike stream in a docstring. | Specify a biologically realized context population and formally refreeze the source before fixtures. |
| Drive/activity regime | **Fail.** CN-1's g=50 lane activates about 49–57% of cells. At q=0.5688, an unrelated 20-contact segment reaches θm=10 with probability 0.8026 and θa=13 with probability 0.3100. | Preserve CN-1 as rheobase/telemetry evidence but supersede its C1 drive choice. Freeze a sparse k-capped set of actual context-bearing spikes and gate repeated-versus-unrelated overlap before segment growth. |
| A3 control | **Unfrozen deviation.** The runner samples from a cumulative active-cell union, which saturates and is not the prior-step control in the prereg. | Freeze a mass-matched, current-step shuffled control that destroys binding without changing activity density. |
| Permanence update | **Unfrozen deviation.** The implementation reinforces all active contacts because reinforcing only already-connected contacts cannot raise 0.21 contacts through the 0.50 threshold. | State the intended HTM-style potential-contact rule explicitly in the Change Note and pin it with a fixture. |
| Boundaries | **Unspecified.** Transient cell state and `W_prev` cross books, passes, and the TRAIN/HELD-OUT boundary. | Persist learned synapses but reset transient membrane/context state at every book and pass boundary; pin this policy before replay. |
| Required diagnostic | **Closed after audit.** Tokenizer freeze §5 artifact `token_overlap_calibration.json` is source-only, deterministic, non-gating, and tested. | Keep `calibrate_token_overlap.py --check` green when its frozen inputs change. |
| Contract tripwires | **Missing.** “No forbidden trainable component” and weight clone/hash checks are prose claims only. | Add executable state inventories and frozen-state hashes at every checkpoint/evaluation boundary. |
| Resume validity | **Unproven.** The dry run compares tracker flatten/restore only; it does not resume and compare the full lane, pool dynamics, counters, and next-step output. | Prove uninterrupted versus checkpoint-resumed equality for every persistent array and emitted step result. |
| Wall-clock accounting | **Fail.** Cumulative elapsed time is not persisted and the live projection reads the wrong summary level. | Persist cohort elapsed time, check within books, test projections, and treat the cap as cumulative across resumes. |

## Contract mismatch

The governing architecture says learned state is cell identity, dendritic
branches, and local synaptic strength, and that a scored output must be the
winner of a simulated membrane/spike race. The prereg sharpens this for C1:
matching branches produce NMDA-like plateau drive in fitted GLIF3 cells and
basket inhibition suppresses later competitors.

The current step instead performs:

```text
GLIF5 lane.observe(token)
    -> Python tuple of spike-count indices
    -> Python SegmentGrowthTracker threshold/count/update
    -> fired_cells := cells whose integer contact count reached theta_a
```

The last line is the scored recognition event. It is not a membrane crossing
or a spike race. Calling those logical IDs “configured cells” does not satisfy
the contract. This is the primary blocker; speed work on this path cannot turn
it into a valid C1 result.

## Correct schedule and lower-bound timing

The frozen schedule is:

```text
epoch-0 TRAIN
+ 3 exposure TRAIN
+ final (TRAIN + HELD-OUT)
= 5 * 2,117,412 + 458,191
= 11,045,251 steps per arm
```

Three serialized arms require 33,135,753 lane steps. The older readiness
document reported 8,927,839 steps/arm and omitted one complete TRAIN sweep.

On the current, scientifically invalid split-lane/tracker implementation:

- frozen g=50 produces about 64k winners over the initial calibration slice
  and about 74.6k winners/token in the 4,096-step confirmation;
- a 20-contact segment is therefore almost certainly touched each step, so the
  `|W|=54` inverted-index benchmark does not model C1;
- the same density makes random segment activation common enough to erase
  context specificity, independently of runtime;
- serialized execution would require 613.625 steps/s to finish in 15 h;
- sharing the independent GLIF5 lane would require 204.542 steps/s;
- no measured end-to-end mode reaches that rate;
- a one-trace kernel measured a 220.64 steps/s kernel-only ceiling, suggesting
  compact activity transfer and removal of observationally dead trace work are
  worth investigating, but this is not an end-to-end cap result.

These are lower-bound engineering measurements for the discarded split
architecture. A physically simulated pool/race may change both work and
sharing opportunities, so its own cap gate is mandatory.

## Repair program

### R3-F0 — neuronal segment fixture

Build the smallest end-to-end fitted-cell fixture before touching the corpus:

1. A finite pool of physical cells has membrane state and stable IDs.
2. A previous context population activates contacts on distal segments.
3. A θm match makes a segment eligible for local learning.
4. A θa connected match injects a measured plateau current into its owner.
5. The arriving-token pool runs a fitted-cell first-spike race with inhibition.
6. The emitted spike, not a Python threshold result, defines predicted path and
   persistent winner identity.
7. Growth-off remains structurally zero.
8. Repeated-context, novel-context, punishment, and bit-exact rerun fixtures
   pass with deep-state equality.

This fixture must also demonstrate that contacts initialized below connected
threshold can become connected under the frozen learning rule.

Before selecting the context representation, compare sparse k-caps of the first
actual recurrent GLIF spikes (roughly K=20–64, values fixed before the probe).
The selected K must be nonempty, stable for repeated context, separated for
unrelated context, and yield a preregistered hypergeometric false-positive
bound. A fixed afferent SDR is an allowed control but cannot win merely by
encoding token identity. Uncapped g=50 activity is ineligible.

### R3-F1 — scientific-control fixture

Before scaling:

- freeze book/pass reset behavior;
- freeze a mass-matched A3 control;
- prove read-only evaluation leaves every learned array unchanged;
- prove TRAIN/HELD-OUT ordering cannot alter the held-out score;
- prove fresh and resumed runs emit identical spikes, races, counters, and
  checkpoint hashes;
- keep the tokenizer hypergeometric diagnostic source-current and include its
  false-positive interpretation in CN-2.

### C1-COST-DENSE-4096 — pre-outcome cap gate

Use the first 4,096 canonical TRAIN tokens, seed 17, the production backend,
the CN-2 sparse-activity policy, and the repaired physical-cell path. Capture
actual activity as packed masks and compare only implementations that are step- and
deep-state-exact.

Measure:

- physics with no host readback;
- production compact activity readback;
- segment matching and plateau/race work at 1k, 8k, and 32k live segments;
- fresh versus resumed execution;
- all three arms under the final sharing topology;
- checkpoint and I/O cost.

Project the **correct** schedule with at least 10% headroom. If the repaired
cohort cannot project below 13.5 h, resolve the result through native
optimization or an explicit cap amendment. Do not launch toward a known
timeout.

### CN-2 — refreeze, then request GO

CN-2 must state, literally:

- physical substrate and simulated-cell count;
- winner-set source;
- A3 construction;
- potential/connected permanence update;
- reset boundaries;
- epoch-0 shadow semantics;
- frozen evaluation semantics;
- any trace-elision or compact-mask runtime change;
- explicit supersession of CN-1's dense C1 drive/preflight choice, including
  K, overlap gates, and the new sparse-activity health criteria;
- correct step arithmetic and cumulative wall-clock behavior;
- exact fixture and cost artifacts that passed.

Only after CN-2, R3-F0/F1, C1-COST-DENSE-4096, a corrected dry run, and a short
smoke are green should the project ask for explicit C1 launch authorization.

## Recognition prototype disposition

Do not integrate the untracked sparse recognition plan or populate its
placeholders as C1 evidence. Its index completeness argument and sparse parity
tests remain useful engineering inputs. Its measured 25–248× speedups used
`|W|=54`, five warmup steps, one synthetic seed, and incomplete stress
fixtures; the run was interrupted before a JSON artifact was written.

Any successor benchmark must replay actual production activity masks, use
multiple timing trials, report median/p95 and memory, alternate timing order,
checkpoint rows incrementally, and pin dense-W parity including growth-off,
empty/disjoint activity, decayed segments, hit/miss visitation order, and
max-segment STOP.

## Launch authority

“Continue where I left off” authorizes this verification and preparatory
repair. It does not override the prior handoff's explicit-GO requirement for a
multi-hour cohort. More importantly, the current runner would not earn a valid
verdict even if launched. C1 remains blocked until the repaired protocol is
scientifically and operationally ready.
