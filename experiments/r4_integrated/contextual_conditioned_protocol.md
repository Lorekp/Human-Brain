# Interaction conditioning in the executed learning loop

Public development; declared before outputs. Baseline commit `050f5f9` preserves
the [previous run and diagnosis](contextual_revisit_result.md). The
[active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
owns the shared milestone. No protected evaluation is authorized by this trial.

## Hypothesis and fixed comparison

The native rule could fit the previous 32 recorded contrasts only after boosting
the weak query-interaction features. A fixed gain of 16 reached all margins in
two passes; fitting a ledger did not test global selection or fresh tasks.
Hypothesis: the same conditioning supports useful executed learning and retention
when its own wrongly ranked answers receive actual corrective feedback.

Compare **GAIN1** and **GAIN16**, applying the stated gain to both query-times-span
and query-times-window features during inference, teacher decisions and replay.
Keep the encoder, raw document store, generic candidates, native learning rate 1
and update rule, scratch-bias policy, GLIF dynamics, and learned occurrence motor.
Bind representation policy into checkpoint/storage identity; match each frozen
mate to its own gain. Identical synaptic arrays at different gains are not an
identical architecture. There is no margin-deficit update or gain sweep.

## Teaching and declared exploration change

[`contextual_conditioned_cases.json`](contextual_conditioned_cases.json) openly
reuses the previous sixteen teaching requests and exact 48-entry schedule. These
are development exposures, not fresh independent questions. Six new transfer
documents supply eight supported requests and two unsupported probes, preserving
the simple, distractor, repeated-value and paraphrase strata.

Inherit the [revisitation protocol](contextual_revisit_protocol.md)'s executed
FAIL/teacher-PASS admission, immutable occurrence ledger, source removal, teacher
exclusion, old-query probes and cyclic replay rules. Both gains use replay:
32 newly admitted pairs maximum, up to four distinct older pairs after each
correction, 128 rehearsal offers maximum, and no later updates after the 32nd
correction's rehearsal. All 48 neuronal first attempts still execute.
Intermediate old-query probes use the raw learned-drive reference, remain
evaluator-only/read-only, and never select updates or the replay schedule.

For both gains, change training exploration after neuronal evidence STOP:

1. Select the unique maximum **raw learned drive**, if it exceeds
   `atanh((600.1 - 600) / 200)`, the original STOP-current equivalent.
2. If raw selection also STOPs (including exact ties), use the external-cosine
   proposal. Execute at most one fallback proposal for this encounter.

Raw-drive selection uses the same native-learned weights and all candidates.
It is an explicitly disclosed host readout, without the saturating current map
or spiking race. Its purpose is to challenge learned wrong winners that the old
fallback missed. It never counts as neuronal first-attempt success. The old
mapped-current conventional selector remains unchanged as a separate reference.
Answer expectations stay evaluator-only; successful attempts create no fabricated
negative. Frozen controls replay the updater's exact original/rehearsed feedback.
The two gains can collect different actual negatives; report those differences.

## Evaluation, cost and decision

After teaching, ingest the fixed fresh documents once and remove live source
inputs. Execute all ten fresh probes and sixteen retention requests without
updates/retries under five policies: updating neurons, frozen neurons, mapped
conventional readout, external cosine, and raw learned-drive readout. All share
the same fixed learned motor prior. These references isolate evidence selection;
they are not a whole-system removal of every neuronal component.

Report neuronal and reference executable accuracy separately, both-property
entity scores, unsupported STOPs, decoys, primary/fallback/teacher counts, applied
versus offered updates, and old-query progress. Distinguish raw-to-mapped from
mapped-to-spike winner disagreements. Original public KS2 Q6a/b remains untuned,
with the same official-rubric pending-human-score policy for noncanonical outputs.

Targets remain at least 6/8 fresh and 12/16 retained neuronal executed successes,
including both properties for three fresh entities, with a useful gain over the
matched frozen arm. Report unsupported behavior separately. Reference-only gains
are useful diagnosis, not a neuronal gate pass. If raw rankings become useful but
current mapping or spiking blocks action, a bounded dynamics change is justified.
If retained scores improve without fresh transfer, revisit representation rather
than raising gain or replay budgets on this cohort.

Budget: one fixed gain candidate versus baseline, 48 teaching attempts and the
stated correction/replay caps per arm, two hours implementation/diagnosis. Record
actual native identity, source/protocol/case hashes, checkpoints, ingestion/query
and total time, native work, ledger/source bytes and peak process memory. Use
compact LF JSON for full numeric receipts; refuse existing output paths. Preserve
all completed artifacts and contracts.

```powershell
$env:HUMANBRAIN_NATIVE_CACHE = Join-Path (Get-Location) '.native-cache'
$env:HUMANBRAIN_DEVICE = 'cpu'
./.venv/Scripts/python.exe -m experiments.r4_integrated.contextual_conditioned_python_session --output benchmarks/contextual_conditioned_python_development_v1.json
```
