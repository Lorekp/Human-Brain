# R3 A3 physical-assembly index prototype: REJECT

**Status: engineering rejection at the frozen keep bar.** This is bounded
pre-outcome evidence only. It does not inspect held-out data, authorize C1, or
change the frozen protocol.

## Decision

Reject the private physical-assembly bucket prototype for production
integration. On an actual `BoundWinners` 4,096-token TRAIN prefix grown to
44,976 segments, the indexed path preserved every compared logical trace field
but delivered only **1.141x** speedup over the last 1,024 steps and **1.188x**
over the last 256 steps. Both miss the lane's frozen **at least 2x grown-state
recognition/step** keep bar. The native prototype was therefore reverted; this
decision note is the only retained change.

The early miss also ends the requested synthetic lane before a q255 recurrence
or broader replay. That is intentional fail-fast behavior, not positive
evidence about those workloads.

## BoundWinners route was eligible

The rejection is performance-based, not a source-layout failure:

- A deterministic base winner GID maps to its minicolumn by base-population
  layout. A recruited winner GID maps through the runtime's `cohort_columns`,
  which is also reconstructed on checkpoint restore.
- A stored ten-winner row therefore projects to ten distinct source columns.
  The codebook's maximum distinct-code overlap is two, below this arm's logical
  match threshold `ceil(10 / 2) = 5`; those ten columns identify one unique
  physical eleven-column assembly.
- An eleven-winner query projects to the exact physical input assembly. Bucket
  selection remains coarse only: recognition still compares winner GIDs and
  permanence ticks exactly inside the bucket.
- The 4K run had 4,095/4,095 non-boundary queries eligible (4,095/4,096 total
  steps, 99.9756%) and 44,976/44,976 learned rows routeable (100%). The one
  ineligible call was the reset boundary's empty context. No token ID or
  transition identity entered the native route.

The discarded prototype maintained this derived index at canonical row
rewrite/append, reserve-cohort publication, fresh construction, and checkpoint
rebuild seams. Any uncertifiable context or state fell back to the exact
intrusive recognizer.

## Bounded A/B

The isolated prototype started from repository commit `1a261ea`.
Both arms used the existing TRAIN-only cost smoke, construction seed 17,
threshold arm `(10, 13)`, one reset, and the same canonical 4,096-token prefix
(`a39faff31eb8be4c13f4b8cd5e4b070009eeca575bde44ad552b8bf8dcbef26f`).
The baseline source digest was
`c423b334436a21db7a553a2f5ba0278188528602be204ffaea362d33be66189b`;
the prototype digest was
`2808227932b0793b4373258d2116aa2bcf8ac97c64ae13a5934fda495e583f22`.

| Measure | Intrusive baseline | Assembly bucket | Ratio |
| --- | ---: | ---: | ---: |
| Full replay wall | 1,069,941,300 ns | 860,307,100 ns | 1.244x |
| Last 1,024 step latency sum | 270,161,700 ns | 236,773,900 ns | 1.141x |
| Last 256 step latency sum | 70,026,200 ns | 58,921,600 ns | 1.188x |
| Candidate work sum | 2,523,178 | 2,273,750 | 1.110x |
| Posting/bucket-record work sum | 4,093,299 | 3,210,507 | 1.275x |

The following recorded per-step logical series had zero mismatches across all
4,096 steps: matching segments, active segments, live segments, predicted
columns, allocated segments, and punished segments. Both runs ended with
44,976 segments, 173 total punishments, six predicted steps, 45,056 winner
spikes, and 45,056 basket spikes. Candidate and posting fields intentionally
measure different recognition work on the prototype path and are not parity
fields.

## Why integration stops

The bucket removes cross-code posting work, but still linearly exact-inspects
all rows owned by the selected physical assembly. At this representative grown
state the work reduction was only 1.275x and translated to 1.141x tail-step
speed. Retaining a second mutable/rebuildable index for that result would add
code and memory without closing the measured wall-clock gap. A favorable
full-density extrapolation is not a substitute for the failed measured keep
bar, so no production merge, C1 run, full replay, or push follows from this
prototype.
