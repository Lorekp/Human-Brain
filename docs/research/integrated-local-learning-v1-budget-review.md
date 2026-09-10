# Integrated local learning: acquisition budget review

Reviewed capacity planning, 2026-09-10; its large repetition projection is not
selected. This source-and-receipt review authorizes
no execution, rate change, seed search or new component. The
[selected rule](integrated-local-learning-revision-v1.md) fixes eta at .01 and
preserves the acting architecture. Graph discovery was unavailable (transport
closed); the source references below were read directly. No journal, payload,
model or recurrence was replayed.

## Contact scale is different from correction count

The [completed repetition](repeated-teaching-acquisition-v1-learning-review.md)
offered 32 corrections per skill, or 1,120 byte/EOS symbols per neural route.
Literal was selected on 273/1,120 offers (24.375%); each position received only
4–12 actual literal contacts. The new rule removes the teacher denominator,
but it still teaches literal only on the actual selected branch. Its changed
mode credit may change that inclusion rate; the old rate is a planning
sensitivity, not a forecast.

For the proposal's idealized fixed unit input, one target and uniform 257 cells,
the exact symmetric CE recurrence has margin increment
`.01*(257/256)*(1-p)`, hence `m(n)<=n*.01*257/256`. Reaching literal probability q
therefore requires at least
`ceil(log(256*q/(1-q))/(.01*257/256))` contacts when q exceeds 1/257.
This gives necessary minima of about 553 contacts for q=.5, 772 for q=.9 and
1,011 for q=.99. The recurrence slows as p rises, so these are optimistic lower
bounds within that idealization, not sufficient budgets. They are not network
bounds: repeated target cells, shared features, scan covariance, public credit
and changing h/gate activity all alter actual currents.

At the historical mean inclusion, 1,024 corrections yield about 250 contacts
per position; 4,096 yield about 998. Scaling the historical positional range
instead gives 512–1,536 at 4,096 corrections. Neither extrapolation guarantees
even that coverage. Literal confidence also cannot establish a complete answer:
mode weighting can suppress it, other emitters can help, and the teacher's
correct-prefix activity can differ from free generation. Do not call a second
32-correction run an adequate acquisition opportunity merely because QA passes.

## Measured cost and the feasibility constraint

The [saved cost review](repeated-teaching-acquisition-v1-cost.md) records
1,519.0128574 seconds and 1,529.1484375 MiB sampled process-tree peak for one
triad: 64 live corrections per neural route, five panels, persistence and audit.
Its three journals total 510,318,993 bytes; main attempts and teachers alone
total 428,590,837 bytes. Nine actual journal-read passes total 4,592,870,937
bytes. Host seals hash 80,229,681,776 tensor bytes. These are measured/scoped
work quantities, not proof that hashing or storage dominates elapsed time.

Initialization, final commit, fresh audit and scalar review together account
for only about 48.49 seconds. Removing persistence work alone therefore cannot
credibly deliver an order-of-magnitude whole-run speedup. Live attempt timers
combine acting, real execution, teaching, diagnostics and state seals; existing
receipts do not isolate their complete costs.

Two source-isolated triads with 4,096 corrections per skill multiply main work
by 256 relative to that run. Straight proportional scaling of its complete
time is about 108 hours; scaling main journal bytes is 109,719,254,272 bytes
(about 102.2 GiB), before panels and final payloads. These are capacity estimates,
not measurements or upper bounds: the new rule and longer free responses may
cost more, while fixed setup/panels do not scale that way. Current file-system
metadata reports 9,429,032,960 free bytes, only about .78 GiB above the 8 GiB
reserve. The existing recorder cannot support this candidate on current disk.

## A finite comparison projection, not a selected run

For capacity planning, a finite **4,096-correction-per-skill** comparison could
use 128 A00-R, 128 A00-E, then 3,968 alternating R/E pairs, with panels at
main indices 0, 128, 256, 512, 2,048, 4,096 and 8,192. The first two single-skill
blocks expose interference; later panels measure the approach to complete
acquisition. Keep the same two taught tasks and R02/R12 sentinels, difficulty,
free limit, real execution and teacher-after-free ordering. Every scheduled
correction remains scheduled, including after successes. No transfer expansion
or separate recovery battery follows from this budget.

Use retained ONLINE as the identical numerical starting point for predecessor
v2 and selected v3 enabled/disabled arms, with the same initial RNG and preserved
motor/tape state. Run two source-isolated triads sequentially, each with its
continued simple reference; authenticate paired pre-learning forward behavior.
Do not reseed live learners to keep later samples paired. Report simple duplicate
parity only for identical actual inputs. Each neural arm has 143,360 main
byte/EOS offers and 12,288 motor labels; seven existing-style panels add 245
disabled diagnostic symbols and 21 motor labels. Count scans/public input and
all reference activity separately; no new factual ingestion is proposed.

This is an uneconomic projection under the measured path, **not a recommendation
to launch 4,096 corrections**. Even a hypothetical 120-hour ceiling with compact
journals could fail on longer responses or increased computation. Current disk
cannot retain the unmodified evidence. No wall-clock/artifact envelope is
selected here, and no deletion or automatic continuation is authorized.

At every panel report each target position's offered/selected/published literal
counts, held literal p/log-p, sampled-emitter Brier score and full diagnostic
mode-mixture components. Keep raw bytes/EOS separate, record activity hashes
across panels/arms and teacher/free prefix alignment, and retain exact free
outputs, stopping reason, actual executed grade and motor choices. Compare
both taught free outcomes at the final two panels and all intervening live
outcomes; distinguish conditional-score improvement, acquired motor span and
complete behavior. Per-block changes at matching activities address interference;
different activities remain a routing/context change, not a fixed-weight-site
contrast. No favorable scalar score licenses a complete-acquisition claim.

## Smallest runner seams and one next decision

The existing `repeated_teaching_acquisition_v1.py` serializes full prediction and
teacher structures on every attempt, captures literal weights and evaluates
unselected diagnostic literals every time, then retains all panel records in
RAM. Reuse its actor calls and owned journals; a longer runner could use a bounded
practice receipt: complete free/execution evidence, actual offered/inclusion/
publication counters, selected-branch scalars, state identities and work/time.
Keep complete diagnostic traces and `capture_literal`/`analyze_teacher` at the
fixed panels. This removes observation-only work from routine teaching without
changing teacher calls or random draws. It requires an explicitly narrower new
saved auditor; hashes of omitted full vectors do not make later full-vector
update attribution possible. Do not silently rewrite frozen evidence contracts.

Retain owned append authentication and fresh final reads. Existing checkpoint
and diagnostic helpers provide these seams; avoid a new generic recorder or
learner optimization. Add separate timers around existing acting, teaching,
execution, diagnostics, seals and append calls so any remaining expensive path
is measured. Do not remove live custody checks on a conjectured speedup. Compact
journals address retained bytes; they do not establish a material CPU reduction.

The next actual decision input is root's already bounded, unchanged-teaching-free
paired-parent preservation check: four existing cases per worker, with phase
cost timers. It must preserve the acquired motor behavior and paired forward
outputs before new learning. It is not a new acquisition run or an open-ended
profiling program. Free-only timings cannot measure changed teacher cost; keep
that uncertainty explicit when projecting the next complete workload.

Use that evidence to select one adequate finite acquisition/resource budget,
or justify calibration of the existing local update units before committing to
an uneconomic repetition count. The .01 contact timescale is a limitation of
the selected update computation itself, not evidence that a new component is
needed. Neither arbitrary rate/seed sweeps nor compact recording alone resolve
it. Until that decision, architecture, rate, difficulty and teaching remain
unchanged; there is one shared acquisition queue.
