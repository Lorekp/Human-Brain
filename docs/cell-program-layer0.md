# Cell-program Layer 0: scalar plumbing

**Status:** public synthetic engineering slice; no development or evaluation result

Layer 0 proves the smallest complete path from a prompt through simulated
human-fitted cells to executable Python:

1. UTF-8 prompt bytes drive fixed afferent synapses, followed by one fixed
   delimiter cell.
2. The recurrent GLIF5 reservoir retains prompt and emitted-token state.
3. Learned output synapses drive one output cell per frozen Layer-0 token.
4. At each position, the earliest fitted-GLIF spike among grammar-allowed cells
   is accepted. The external grammar mask only removes impossible arity
   prefixes; it is not a simulated inhibitory-cell mechanism.
5. A deterministic compiler translates the one emitted prefix AST literally to
   one Python function. It receives no prompt, examples, tests, filesystem,
   network, subprocess, search, ranking, or repair interface.

Every semantic program choice therefore comes from cell and synapse state. The
ordinary grammar and compiler provide syntax plumbing only.

## Claim boundary

This is deliberately named **Layer 0**. It is an `Int -> Int` arity grammar
with a few named bounded literals. Exact training-fixture memorization proves
only that the plumbing works. It is not the typed Simple-SWE DSL described in
the research design, not evidence of held-out generalization, and not a coding
benchmark result.

Layer 1 must replace this grammar with:

- a `PrimitiveSpec`-driven typed hole stack;
- signed-digit literal construction;
- generic scalar, list, string, fold, map/filter, adjacent-pair, and window
  primitives; and
- family-disjoint public development evidence before any sealed evaluation.

No family-specific operation macro belongs in that primitive set.

## Stable Layer-0 interfaces

- `Layer0ProgramGrammar.allowed(prefix)` returns only prefixes that can finish
  within the frozen node bound.
- `Layer0CellProgramModel.grow(examples, config)` grows existing recurrent
  cell/synapse learning and freezes it.
- `Layer0CellProgramModel.complete(prompt)` performs one bounded autoregressive
  cell-race continuation.
- `Layer0ProgramCompiler.compile(tokens, signature)` translates exactly one
  complete emitted AST.
- `RecurrentCellSynapseState` captures, hashes, saves, restores, and validates
  persistent circuit state across ridge, dense three-factor, sparse
  three-factor, and developmental anatomy.

The frozen public plumbing protocol pins the ridge learning rule; the module
does not. A separate unit smoke grows and completes through the existing sparse
three-factor readout so the scalable path remains structurally usable.

## Replay audit

Each accepted position records:

- accepted token and winner cell ID;
- fitted-cell spike time and simulated step count;
- SHA-256 of the prefix before the race;
- the exact sorted allowed-cell set;
- invalid candidate cell IDs excluded before the accepted winner, in race
  order; and
- SHA-256 of the complete output-current vector.

The completion records the configured learning rule and before/after SHA-256 of
the persistent cell, synapse, topology, readout, and learning state. Runtime
membrane and eligibility traces may evolve; learned checkpoint state may not.
Checkpoint round-trip tests require the exact same continuation and race audit.

## Anti-overfit state

[`cell_program_layer0_protocol_v1.json`](../benchmarks/cell_program_layer0_protocol_v1.json)
freezes the vocabulary, grammar, generic primitives, ridge configuration,
eight public synthetic training fixtures, and a three-case public composition
probe. The composition probe is committed but remains unopened. No canonical
seed is reserved, and no reveal, private task, hidden test, existing evaluation
artifact, or external model data is used by this slice.

The unit/capability smoke is public and synthetic:

```powershell
python -m pytest tests/test_cell_program.py -q
```

The smoke proves an actual grown recurrent circuit can emit `ADD ARG0 ONE END`,
compile it, and return `42` for input `41`. That exact prompt/program pair is a
training fixture and is reported only as end-to-end plumbing.
