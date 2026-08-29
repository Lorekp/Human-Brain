# Cell-Brain Program Map

Maintained by the orchestrator. Updated 2026-08-29 after the first broad
bottom-up typed-semantic TRAIN pass, its speed layer, the first generic breadth
mechanisms, prompt-span input, and Path-2 replay preparation. Exact scientific
boundaries live in
`docs/research/roadmap-reading-to-coding.md` and `docs/benchmark-status.md`.

## Goal ladder

```
R9  Beat Opus 4.8 on DeepSWE                         [end state]
R8  Beat Gemma-4-12B-Q4 on a fair chosen SWE bench  [design exists; unopened]
R7  Real held-out SWE benchmark                      [after simple-SWE ladder]
R6  Canonical prompt-only simple-SWE M0–M3           [evaluation sealed]
R5  Family-disjoint public DEV transfer              [one paired gate; unopened]
R4  TRAIN-only prompt → semantic-factor bridge       [physical span input banked]
R3  Bottom-up typed semantic synthesis               [30/30 + breadth mechanisms]
R2  Persistent Path-2 reading memory                 [fresh scorer replay prepared]
R1  Fast, auditable simulated-cell substrate         [continuous]
```

## Actual frontier

### Coding

`PASS_PUBLIC_TYPED_SEMANTIC_SIMPLE_SWE_TRAIN_BRIDGE_V1` is the current
capability frontier. One immutable bottom-up synthesizer selected exact programs
for 30/30 canonical public-TRAIN tasks across scalar offset, list-map-add, and
list-map-absolute, then passed 960 property cases hidden from selection. It saw
typed signatures plus four regression I/O rows, not prompts or family/task
identity. This is physical programming by examples, not canonical prompt-only
Simple-SWE, family-disjoint transfer, or M0.

The next order is intentionally leakage-safe. The semantic physical path is
already parity-accelerated: a first-round task from each banked family replayed
42.3x faster in aggregate without changing decisions or race ledgers.
Commit `fb83d20` additionally batches exact typed-coincidence candidates per
target case. Its byte-exact 64-row synthetic physical-phase fixture measured
51.72--73.61x faster; this is a kernel result, not end-to-end capability.

Commit `75513d2` begins the six-family breadth layer with one incremental
physical-retention path, generic min/max composition, and bounded Char/String
output factors with type-total copy and UNKNOWN rejection. Self-posed
nonnegative, clamp, and mapped-nonnegative programs synthesize at costs 3/5/6;
no task from the six-family gate has been opened. Commit `c89a3c6` separately
banks strict callable framing, role-normalized ordered prompt spans, physical
12-of-512 span assemblies, and a separate signed-literal copy channel. It is an
input substrate with zero learned state, not yet a prompt-to-factor learner.

Commit `00ed3e5` adds generic physical `sum`, `less-than`, and predicate
`count`, with predicate retention at cost 3 and binder licensing at cost 4
before cost-6 count composition. Self-posed count and cost-9 mapped-clamp-sum
checks pass with immutable checkpoints. A diverse 25-literal frontier exceeded
90 seconds in review, so the broad gate also requires a desired-independent
STOP-on-work-overflow bound; no silent top-k shortcut is licensed.

1. Finish the generic String combinators and desired-independent frontier work
   bound, then adjudicate the six unspent TRAIN families once.
2. Complete and freeze the learned prompt-span → semantic-factor mapping on a
   preregistered TRAIN capacity split; its physical input layer is already banked.
3. Spend the 27 fixed family-disjoint public DEV tasks once, with visible-I/O
   and prompt-only arms paired in the same gate.
4. Only a paired transfer pass licenses the canonical M0 freeze/reveal ceremony.

### Reading and persistent memory

The dense Path-2 build has processed all 49 TRAIN books and 2,117,412 tokens,
ending at 23,286,405 actual segments in a 1.490 GB durable checkpoint. Its final
isolated cost measurement is pending. A target-independent successor scorer now
implements per-output-column factorization, stable complete tuples, global
owner-column non-aliasing, and the sum of actual allocations while reporting
matching and active telemetry separately. Its separately named source-bound
resumable TRAIN-only runner is now preflight-ready from empty native/scorer
state, with a compact per-book journal and no completed-build import. No corpus
spend occurred, and no held-out or C1 surface is open.

### Runtime and scale

Measured substrate boundaries remain CUDA at small dense scale, CPU/native at
the mid-scale frontier, and procedural sparse addressing above 10 million
configured cells. Roughly 190 million resident cells fit on this host under the
current envelope; reaching the 100-billion-neuron target requires
event-proportional work plus distributed storage/compute after learning is
demonstrated. Runtime optimization is part of the model implementation, not an
optional benchmark exercise.

## Claim and opening discipline

- Public TRAIN mechanism work may be rerun only on a newly named unspent gate.
- Public DEV desired vectors stay closed until both the generic semantic engine
  and prompt bridge are frozen.
- Canonical M0/eval, protected data, held-out Path-2 material, and C1 remain
  sealed.
- Host code may enforce grammar, typing, and sandbox policy, but may not execute
  or rank candidate programs for the neuron-selected synthesis path.
- Exact microgate passes are substrate to build on, not broad capability claims.

## Banked directions not to revisit

Do not retune the falsified reservoir feature/schedule/codebook/EI/local-rule
axes, U-STATE, the exact APICAL-AND sweep, whole-sequence lookup, static
whole-prompt decoders, blind/persistent edit actors, failed beam variants, or
H=2 prompt-projection patches. On Path-2, do not retry calibration-v3, select
`(18,20)`, equate surface prediction with owner capture, or require monolithic
`11 * novel_keys` allocation. The roadmap records the evidence for each
closure.
