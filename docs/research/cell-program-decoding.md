# Cell-only program decoding for the simple coding gate

**Date:** 2026-08-21
**Status:** Design study; no coding-evaluation result

## Decision

The first generic coding model should emit a small typed prefix program
language directly from output-cell races. A deterministic compiler may turn
that program into the one Python function expected by the sandbox, but it may
not search, rank programs, inspect hidden tests, repair output, or choose an
algorithm.

This is a narrower and more auditable target than raw Python generation:

- the learned semantic choice still comes entirely from simulated cells and
  synapses;
- every output token is the winner of a fitted output-cell race;
- a fixed grammar may reject only syntactically or type-invalid next tokens;
- the compiler is ordinary deterministic language plumbing, analogous to the
  Python interpreter; and
- the emitted token trace, compiler hash, generated Python, and checkpoint
  state hashes are recorded for every task.

The grammar and compiler must be frozen before the hidden evaluation seed is
created. The initial 30-task gate still uses one completion, no repairs, no
test feedback, and a freshly restored common checkpoint for every task.

## Established patterns and the boundary we need

DeepCoder predicts which domain-specific-language operations are likely to
occur, then uses enumerative or SMT search to find a program
([paper](https://arxiv.org/abs/1611.01989);
[Microsoft publication](https://www.microsoft.com/en-us/research/publication/deepcoder-learning-write-programs/)).
That demonstrates the value of a compact compositional language, but its
external search would violate this project's strict cell-only inference
boundary. We adopt the DSL, not the solver.

Abstract Syntax Networks construct code through a decoder whose structure
follows the output abstract syntax tree
([paper](https://arxiv.org/abs/1704.07535)). PICARD checks partial outputs
during autoregressive decoding and rejects tokens that cannot lead to a valid
parse
([paper and official implementation](https://github.com/ServiceNow/picard)).
These support separating semantic token choice from syntax validity. Our
version must remain simpler: the constraint layer may answer only whether a
candidate is valid under the frozen grammar and type stack.

DreamCoder grows reusable program abstractions and composes them across tasks
([paper](https://arxiv.org/abs/2006.08381)). Its wake/sleep library-learning
pattern is relevant to developmental cell growth, but DreamCoder also performs
symbolic program search. We can test whether cell assemblies and structural
plasticity acquire reusable primitives without importing that search at
inference.

Natural-language-to-DSL synthesis predates modern language models and has been
evaluated as a learned ranking problem over a fixed DSL
([Desai et al.](https://arxiv.org/abs/1509.00413)). This reinforces the need to
freeze the language independently of individual hidden tasks.

## Smallest end-to-end language

Use a typed, fully parenthesized prefix language. Its general primitives should
cover scalar, list, string, fold, and adjacent-pair operations rather than
providing one macro per benchmark family. An illustrative core is:

- values: argument, integer, character, boolean, pair, list, and string;
- arithmetic: add, subtract, multiply, absolute value, remainder, minimum, and
  maximum;
- comparisons and booleans;
- list/string conversion, length, split, join, reverse, head/tail, and pair
  projection;
- map, filter, fold, count, range, adjacent pairs, and fixed-width windows; and
- one conditional expression.

Lambda bodies should use a small fixed set of de Bruijn-style argument tokens
instead of arbitrary variable names. Literals should use signed digit tokens
with a short frozen bound. The maximum AST depth and node count are part of the
protocol, not changed after seeing evaluation outputs.

The compiler validates the type stack and translates each primitive literally
to Python syntax or a small audited helper. It does not simplify, enumerate,
execute examples, or use task text. A DSL program has exactly one compiled
Python function.

## Cell path

### Input

The task prompt is tokenized deterministically and paired with the public
`TypedSignature`; no hand-written semantic parser extracts operation labels or
callable types. Training examples contain the prompt and signature followed by
a program delimiter and the teacher DSL sequence. At inference, the same frozen
serializer ends at the delimiter.

### State and reusable assemblies

The recurrent fitted-cell reservoir carries prompt state. The sparse output
readout learns contacts from responsive reservoir cells to DSL-token cells.
Candidate pools and structural turnover are label-independent until teaching
events activate a target output cell.

An optional second development stage can recruit nested groups for reusable
primitive assemblies and composition depth:

1. primitive examples recruit operation and type assemblies;
2. two-operation examples recruit cross-assembly contacts;
3. deeper compositions recruit additional layers while retaining earlier
   contacts; and
4. sleep-like replay uses training examples and synthetically generated valid
   programs only, never development or evaluation prompts.

This is a testable interpretation of growth. It is not assumed to help: matched
full-size-from-start, random-order growth, and no-replay controls are required.

### Output

At each program position:

1. the cell reservoir and learned sparse contacts produce currents;
2. the fitted output cells race;
3. the grammar checks whether the winning token is valid;
4. if invalid, the external fixed syntax mask excludes that token for the same
   position and the remaining output cells continue the race; and
5. the accepted token updates the frozen arity/type stack.

For the first engineering implementation, the grammar stack may be ordinary
deterministic decoder state, provided it only masks invalid tokens and never
changes the relative current of valid tokens. This external mask is not called
biological inhibition because it does not simulate inhibitory cells. A stricter
later variant can represent grammar states with fixed syntax-cell assemblies.
Both variants must report how many candidate winners were rejected.

There is no beam search, sampling rerun, program enumeration, compiler error
repair, or hidden-test feedback.

## Public seams for a later test-first implementation

The proposed stable boundaries are:

- ProgramGrammar.allowed(prefix) returns the set of syntax/type-valid next DSL
  tokens;
- CellProgramModel.complete(prompt, signature) returns accepted cell-race
  tokens plus a full race/mask audit;
- ProgramCompiler.compile(tokens, signature) returns one Python function or a
  deterministic format failure;
- SimpleSWECompletionAdapter restores and hashes the checkpoint, calls the
  cell program model once, compiles once, and returns source plus provenance;
  and
- the existing hidden-test runner sees only final source and never exposes its
  tests to the model or compiler.

Tests should use these public seams rather than native arrays or parser
internals. The seams must be confirmed before implementation.

## Required evidence before hidden evaluation

### Engineering

- Every grammar-produced token sequence parses and type-checks.
- Invalid tokens are rejected without reordering valid-cell currents.
- A scalar independent oracle matches accepted race order on tiny circuits.
- The compiler is deterministic, total over valid programs, and incapable of
  filesystem, network, subprocess, or hidden-test access.
- Save/load preserves cell, synapse, grammar-position, and continuation state.
- The completion adapter restores an identical checkpoint before every task
  and leaves it unchanged.

### Train/development only

- First prove exact memorization of a few training programs; this is plumbing,
  not generalization.
- Measure family-disjoint development pass@1 with all tasks included.
- Compare raw Python bytes against the DSL at the same cell/data budget.
- Compare syntax masking against no mask while reporting semantic failures
  separately.
- Test active-cell scale, nested groups, layered organization, and
  developmental order with paired seeds and matched fanout/training tokens.
- Select the smallest DSL and cell configuration meeting a frozen development
  floor; do not select the best evaluation topology after reveal.

### One-shot evaluation

Only after checkpoint, source, tokenizer, DSL, compiler, and configuration
hashes are sealed may the harness create the evaluation seed. The same
checkpoint generates one program for every revealed prompt. Report all 30
tasks, exact failures, emitted token traces, rejected-token counts, compiler
hashes, and checkpoint before/after hashes.

## What this can and cannot prove

A passing simple gate would show that the cell simulation learned to select and
compose general programming primitives under a fixed syntax. It would not yet
show raw natural-language coding ability, broad Python generation, repository
reasoning, or SWE-bench competence.

A failing gate remains valuable. Format failures point to decoding and grammar;
valid but wrong programs point to semantic state/learning; training success
with family-disjoint failure points to memorization; and degradation with more
active cells points back to dynamics, recruitment, or credit assignment rather
than cell count itself.
