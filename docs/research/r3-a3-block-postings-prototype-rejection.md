# R3 A3 fixed-block postings prototype: REJECT

**Status: engineering rejection at the frozen keep bar.** This is bounded
pre-outcome evidence only. It does not inspect held-out data, authorize C1, or
change thresholds, physics, learning, or the frozen protocol.

## Decision

Reject the cache-friendly fixed-block posting-list prototype for production
integration. On an actual `BoundWinners` 4,096-token TRAIN prefix grown to
44,976 segments, it delivered **1.344x** tail speedup over the last 1,024 steps
and **1.351x** over the last 256 steps. Both miss the lane's frozen **at least
2x tail wall-speed** keep bar. The native prototype was therefore reverted;
this decision note is the only retained change.

## Prototype layout

The isolated prototype started from repository commit `0480c21`. It used an
exact mutable inverted index with 64-entry contiguous `u32` pair-slot blocks:

- blocks were linked only at block granularity and stayed dense except for the
  source tail;
- append filled the tail or consumed a block from a preallocated free list;
- arbitrary unlink scanned only that source, swapped the source tail's final
  pair slot into the hole, and recycled an empty tail block;
- canonical append/rewrite maintained the index, so fresh construction and
  checkpoint row rebuild used the same seam;
- recognition still visited every exact source-to-pair-slot membership and
  inspected its authoritative permanence tick.

The benchmark prototype ran this index beside the intrusive reference to make
the A/B easy to discard. A replacement layout could remove the current `8P`
per-pair next/previous arrays and use `268B + 12D` bytes, where
`B = N + floor((P-N)/64)`, `N = min(D,P)`, is the preallocated block ceiling
and `D` is the source domain. At the frozen `P=232,864,060`, `D=131,072`
sizing, that is 1,011,269,296 bytes versus 1,864,485,344 bytes for intrusive
links plus source metadata. Thus the design met the memory constraint; it was
rejected on measured speed.

## Bounded A/B

Both arms used the existing TRAIN-only cost smoke, construction seed 17,
threshold arm `(10, 13)`, one reset, and the same canonical 4,096-token prefix
(`a39faff31eb8be4c13f4b8cd5e4b070009eeca575bde44ad552b8bf8dcbef26f`).
The baseline native-source digest was
`ca489f4164356a27d5659d2d5dbc6a93c4097e853dfe75684c2100e8b734e30e`;
the prototype digest was
`cb03281c45be929a0d7e13f68a1023db3fbafdaed4986589cfedb2650f22dd9a`.

| Measure | Intrusive baseline | Fixed blocks | Ratio |
| --- | ---: | ---: | ---: |
| Full replay wall | 1,234,480,400 ns | 876,074,600 ns | 1.409x |
| Last 1,024 step latency sum | 324,645,800 ns | 241,548,300 ns | 1.344x |
| Last 256 step latency sum | 80,659,600 ns | 59,703,700 ns | 1.351x |
| Candidate work sum | 2,523,178 | 2,523,178 | 1.000x |
| Posting work sum | 4,093,299 | 4,093,299 | 1.000x |

All eight recorded per-step comparison series had zero mismatches across all
4,096 steps: candidates, postings visited, matching segments, active segments,
live segments, predicted columns, allocated segments, and punished segments.
Both runs ended with 44,976 segments and 173 total punishments. The prototype
also passed `cargo check --release` before the A/B.

## Why integration stops

Contiguous blocks removed pair-by-pair pointer chasing and produced a material
1.34x tail improvement without reducing exact work. That is still well below
the precommitted 2x threshold, so replacing the mature mutation/checkpoint
index would not close the production wall-clock gap. No unlink-scan tuning,
third recognition micro-optimization, broader corpus replay, q255 run, C1 run,
or push follows from this result.
