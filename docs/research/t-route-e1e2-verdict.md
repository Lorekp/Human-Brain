# T-ROUTE E1/E2 verdicts — trained-mapping family status

Status: recorded 2026-08-25. Both executions one-shot, preregistered,
pushed (E1: bbe4dac..e8e21ba; E2: cd6c426).

## E1 (normalized arm): NEGATIVE_CORE_WOUND
G-T1 fail decisively — arm A base_med 0.0 both splits vs randomized-W 11.0.
IDF homeostat floors common-byte rows; winners collapse onto a fixed 554-bit
rare-byte pool disjoint from gold regions. G-T2 not spent (stop-on-failure).

## E2 (raw W + top-k sparsification): RECOVERY_FAILS_SPARSITY_INSUFFICIENT
Homeostat removal recovers absolute overlap (base_med 80.0/74.0 vs arm A's 0.0)
— E1's localization CONFIRMED. But shuffled-question enrichment = 1.00–1.06:
winners are query-INDEPENDENT, collapsing to a ~170-bit static common-byte pool
(cross-question Jaccard 0.78–0.81). Sparsity flat across k ∈ {8,16,32} —
footprint lever only. No config beats arm B's raw dense baseline on VAL.

## The mechanism-level conclusion

No monotone transform of this accumulator yields question-specific winners.
The Hebbian accumulation estimates corpus-wide co-occurrence; without a third
factor that is QUOTE-SPECIFIC (not just episode-scoped), the winner set is the
same static prior for every query. This is now a measured property of the
mechanism class, not a tuning failure.

## What the discriminator must be

The next experiment's third factor must make learning quote-specific:
per-quote eligibility traces (which passage tokens co-occurred with THIS
quote), not per-episode broadcasts. Candidates: token-position-gated traces,
per-neuron eligibility tags bound to the quote's SDR at encode time.

## Program state

Fourteen preregistered executions complete. The trained-mapping family is
falsified in its broadcast/margin form and its normalization form; the
remaining untested variant inside local-learning constraints is
quote-specific third factors. If that fails, the conclusion is program-grade:
Bible-QA needs either non-local optimization or an architecture whose
connectivity is trained by construction (the R3 episodic-store hybrid).
