# A013 spacing-vs-massing experiment spec (mechanical preregistration)

Status: preregistered SPEC — no runs. Runner executes mechanically once the
shared harness exposes schedule knobs. Provenance: researched + computed by
subagent (permutation schedules precomputed and left in
`perm_appendix.json` at worktree root; transcript-mined after write-time
truncation), authored by orchestrator. Verify pass required before merge.

## Operational schedule definitions

Presentation orderings over observation slots, T tasks x R repetitions.
Worked example T=6, R=2 (task indices 0..5):

- **massed** — each task's repeats adjacent: `[0,0,1,1,2,2,3,3,4,4,5,5]`
- **spaced** — maximal interleave distance between a task's repeats:
  first all of pass 1 in order, then pass 2 reversed:
  `[0,1,2,3,4,5,5,4,3,2,1,0]` (same-task gap = T slots)
- **interleaved** — seeded shuffle of the multiset; canonical permutations
  for seeds 7/8/9 are PRE-COMPUTED and frozen in `perm_appendix.json`
  (seed 7 example, T=6 R=2): `[3,5,1,5,4,2,4,0,0,3,1,2]`

For the real experiment (T=42, R=3 epochs) the same generators apply;
`perm_appendix.json` already contains the full seed-7/8/9 interleaved orders
for the 42-task curriculum (126 slots per epoch-set), so the runner never
generates orders at run time — it loads them.

## Derived effect prediction (trace math, computed by the agent)

Eligibility retention per strengthen-call gap g: rho(g) = exp(-g * d * 5ms /
80ms) where d = mean token distance per call gap. Agent-computed anchor:
mean presentation length ~207.48 strengthen calls; under MASSING the
smallest same-task re-presentation gaps were ~207 calls =>
trace residue rho^207 ≈ **2.34e-06** — i.e., by the time a task repeats
under massing-within-epoch at this curriculum scale, its eligibility trace
is arithmetically GONE.

Correction to the naive hypothesis this implies (record honestly): massed
and spaced may NOT differ through trace residue alone, because even "adjacent"
repeats at this curriculum are hundreds of calls apart. The real lever is the
DISTRIBUTION of inter-task interference inside the gap, not adjacency itself.
Therefore the preregistered prediction is directional but modest: spaced >=
massed on train replay, with effect size expected SMALL; the stronger contrast
is interleaved-vs-blocked on CALIBRATION transfer (diversity-of-context during
learning). If no schedule moves anything, that is itself informative:
scheduling is second-order to representation, as the teaching matrix predicts.

## Arms

| schedule | N | data_scale | seeds |
|---|---|---|---|
| massed | 128, 512 | 42 | 7, 8, 9 |
| spaced | 128, 512 | 42 | 7, 8, 9 |
| interleaved | 128, 512 | 42 | 7, 8, 9 |

18 runs, serialized; readout fixed to sparse_three_factor (post-pilot winner
assumption — if pilot rejects local rules, rerun spec with ridge and note).

## Gates (preregistered)

- Primary: train_cases_passed fraction delta (spaced - massed) > 0 with
  Wilson 95% CI excluding zero across seeds.
- Secondary: interleaved - blocked calibration-cases delta > 0, CI excluding
  zero (the diversity-context prediction).
- Exploratory (non-gating): margin distribution shift (median, p10).

## Red tests for the harness scheduler

1. Order assertions: generated index sequences match the definitions above
   exactly for the worked example (fixture test).
2. Multiset equality: every schedule permutation preserves task multiplicity.
3. Frozen-permutation load: harness refuses to shuffle at runtime when
   `perm_appendix.json` provides the order (byte-equal sequence use).
4. Determinism: identical seed => identical schedule => identical metrics
   JSON modulo wall_s.

## Cost estimate

Grow wall scales linearly (~exp 0.996); N=512 pass ~= 4x N=128. Estimate
~10-20 min total for 18 runs on the local box, serialized.

## Falsification note

If spaced/massed/interleaved are statistically indistinguishable at both N,
the teaching matrix's T3 is marked dead and representation work absorbs its
priority — recorded as a clean negative either way.
