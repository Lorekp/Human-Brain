# Executed revisitation and replay: result

Completed 2026-09-05 under the [declared protocol](contextual_revisit_protocol.md).
The [run](../../benchmarks/contextual_revisit_python_development_v1.json) and
[receipt diagnostic](../../benchmarks/contextual_revisit_diagnostic_v1.json)
are repeatable public development. No protected evaluation or gate advancement.

## Behavior

| Policy | Fresh executed | Old requests executed | Both-property fresh / old entities | Unsupported STOPs |
| --- | ---: | ---: | ---: | ---: |
| REVISIT updating neurons | 0/8 | 1/16 | 0/4; 0/8 | 0/2 |
| REPLAY updating neurons | 0/8 | 0/16 | 0/4; 0/8 | 2/2 |
| Conventional learned scores, either arm | 0/8 | 1/16 | 0/4; 0/8 | 0/2 |
| Frozen neurons, either arm | 0/8 | 0/16 | 0/4; 0/8 | 2/2 |
| External encoder cosine, either arm | 0/8 | 0/16 | 0/4; 0/8 | 0/2 |

REVISIT emits `denim` throughout. REPLAY's conventional winner is always `topaz`,
while its neuronal selector STOPs on three simultaneous first spikes. Its two
unsupported STOPs therefore do not demonstrate selective abstention. Neither
learns useful query-dependent action. All decoys remain unchanged.

Each arm makes 48 first attempts: REVISIT scores 0/16, 0/16, 1/16 across passes;
REPLAY scores zero in all passes. The lone success is a constant-answer match.
Each retains 32 actual FAIL/teacher-PASS pairs. REVISIT applies 32 native updates;
REPLAY offers 118 additional rehearsals and applies 79 updates in total (21 new,
58 rehearsal). Frozen mates have identical feedback counts and unchanged synapses.
Teacher repairs and external exploration never count as autonomous success.

## Why this is not a clean comparison of retention alone

Rehearsal also changes subsequent negative collection. REPLAY executes fifteen
neuronal negatives and 33 external-cosine negatives after STOP. Seventeen external
negatives enter the ledger, but eleven already satisfy the learned pair margin
and cause no native update; sixteen occur after the admission budget expires.
REVISIT instead admits 31 neuronal negatives and one external negative.

From steps 19–48, the conventional winner is `topaz`. The fallback after STOP
executes the external proposal, so it does not directly challenge that winner.
At step 29, `topaz` does spike and executes incorrectly. Its corrective gap rises
from -5.201 to -3.138, remaining wrong, and that pair receives no later replay.
The prescribed protocol was followed, but its on-policy feedback changes along
with replay; do not attribute the result to rehearsal quality alone.

## Conditional supervision now exists; learning still fails

The diagnostic reconstructs every native feedback receipt, checkpoint and saved
array exactly. Both ledgers have cycles over identical document/span occurrences,
so no query-independent strict ordering can satisfy their recorded preferences.
REVISIT includes a direct reversed-property pair. Its 32 pairs contain 24 distinct
deltas; REPLAY's 32 contain 28, with rehearsal counted separately as exposure.

Full-feature linear-program witnesses satisfy all recorded margins after
independent verification. REVISIT's span-only program reports infeasibility.
REPLAY's span-only solver output fails independent verification and remains
numerically inconclusive; its exact occurrence cycle separately rules out a
query-independent ordering. These fits concern recorded pairs, not all candidate
answers or useful task behavior. Arbitrary large feasibility witnesses are not
minimum required weight norms or evidence of a practical solution.

REVISIT's final checkpoint violates 31/32 margins, with 17 wrong/tied pairs;
REPLAY violates 8/32, with three wrong/tied. Seven REVISIT pairs satisfied their
initial margin and later lost it. Six REPLAY pairs satisfied their last-exposure
margin and later lost it. These are contrast measurements, not previously solved
autonomous tasks.

Feature geometry explains a concrete learning difficulty. REVISIT's median span
difference norm is 1.181, versus 0.0623 for query-times-span. For the same-window
`luma` opposite-property pair, query-only and query-times-window differences
cancel. Satisfying both unit margins requires total weight norm at least 32.94,
while the learned norm is 1.94. The native rule takes a fixed normalized step
when a margin is violated; it does not correct the whole deficit in one update.
Representability therefore does not establish learnability within this budget.

## Resources and evidence boundary

The complete comparison takes 97.96 seconds, including subprocess execution
waits; parent peak working set is 283.16 MiB and final private memory 581.38 MiB.
Process/tracemalloc memory excludes child executors. Paired teaching ingestion
takes 0.138/0.109 seconds for REVISIT/REPLAY; diagnostic old-query scoring takes
6.26/9.80 seconds. This is one timed run, not a kernel performance comparison.

Each arm retains 617 raw teaching bytes and 300 candidates. After fresh documents
and the original exam, it retains 4,828 raw bytes, 2,319 candidates and 27 windows
from fifteen documents. Counted memory arrays/byte payloads total 7,284,992 bytes
per arm, including 1,152 ancestry payload bytes. Each external correction ledger
adds approximately 2.286 MB serialized, including 786,944 bytes of numeric
eligibility payload; these figures overlap rather than being additive. Python
object overhead and the shared 90,405,214-byte encoder are separate.

The original KS2 Q6a/b remains untuned. REVISIT emits `denim`, REPLAY STOPs,
conventional references emit their constant value, and external references emit
`to`. Noncanonical answers keep official-rubric human scoring pending; REPLAY and
frozen STOPs score 0/2. There is no demonstrated exam gain.

The 7.08-second receipt diagnostic adds no actions, negatives, global-candidate
evaluation or transfer fitting. Twelve focused real encoder/learning/replay tests
pass. Artifact/source/NPZ hashes match, native identity is checkout-local, and
receipts use explicit LF newlines. All completed evidence remains unchanged.

## Bounded conditioning follow-up and next integration

The [fixed receipt-only comparison](../../benchmarks/contextual_conditioning_diagnostic_v1.json)
uses the same 32 executed REVISIT pairs, from zero, for exactly four passes per
variant. Gain 16 on both query-interaction blocks was chosen before outputs from
the measured approximately 19:1 span/interaction norm ratio. The inference
coordinates must use the same gain; this is not extra answer information.

| Variant after four passes | Margin violations | Wrong pairs | Native updates |
| --- | ---: | ---: | ---: |
| Original scales, learning rate 1 | 27/32 | 15/32 | 124 |
| Interaction gain 16, learning rate 1 | 0/32 | 0/32 | 42 |
| Gain 16, step sized to current margin deficit | 23/32 | 0/32 | 116 |

The fixed-gain candidate satisfies all margins by pass two (minimum 1.140736);
passes three and four need no updates. Baseline pass one exactly reproduces the
integrated checkpoint. Every native step matches its declared arithmetic within
2.7e-15. Runtime is 1.68 seconds including first native loading, not a speed claim.
No global selection, new action, fresh task or protected evaluation occurs here.

Integrate gain 16 against gain 1 under the same executed teaching/replay policy.
When the neuronal selector STOPs, first execute its conventional learned-score
proposal if available, using external cosine only when that also STOPs. This
disclosed training exploration can challenge a wrongly ranked winner instead of
repeatedly supplying already-satisfied negatives. Apply the same fallback to both
gains and continue counting neuronal first attempts separately. Preserve all
current results, use fresh transfer cases, and change spike dynamics only if
useful learned scores justify it. The [active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
owns that integrated comparison; a constraint fit alone earns no capability gate.
