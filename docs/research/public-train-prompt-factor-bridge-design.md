# Public-TRAIN prompt-to-factor bridge design

Status: preregistered design; unimplemented and unspent.  This document is a
direct continuation of the [reading-to-coding roadmap](roadmap-reading-to-coding.md).

## Purpose and claim boundary

The next language seam is a small learned module that converts the ordered
physical prompt spans banked in commit `c89a3c6` into ordered, atomic semantic
factor events.  It is not a whole-prompt decoder and it does not select or rank
programs.  Its output is intended to be consumed later by the frozen typed
semantic engine.

This design permits only public-TRAIN prompt text, typed callable framing, and
explicit atomic semantic-factor supervision.  It forbids desired outputs,
regression/property rows, whole programs, candidate programs, AST paths,
task/family/ordinal identities, public DEV, M0, canonical evaluation,
held-out data, and C1.  A PASS would establish TRAIN-only prompt-to-factor
recognition.  It would not establish prompt-only program synthesis,
family-disjoint transfer, Simple-SWE capability, or benchmark progress.

The relevant banked substrate is:

- `c89a3c6`: strict callable framing, role-normalized tokens, every ordered
  contiguous one-to-four-token span as a physical 12-of-512 assembly, and a
  separate physical signed-literal copy channel in
  [`public_train_prompt_semantic_bridge.py`](../../experiments/r3_pilot/public_train_prompt_semantic_bridge.py).
- `00ed3e5`: generic physical `sum` and predicate `count`, extending the
  compiler-owned semantic production vocabulary without prompt or task
  identity.
- `fb83d20`: batched physical typed coincidence detection, the established
  pattern to reuse for independent batched fitted-cell decisions.

The existing [`typed_semantic_synthesis.py`](../../src/humanbrain/typed_semantic_synthesis.py)
does not expose a prompt-language factor vector.  `TypedPhysicalSemanticVector`
and `TypedOutputEncoding` represent execution values and are therefore
forbidden teachers here.  `FactorAction` contains edit/path structure and would
revive the retired whole-prompt projection family.  The stable seam is instead
the atomic compiler vocabulary represented by `CandidateProvenance` plus
compiler-owned source and literal roles.

## Sequencing

Implementation must not begin until both of these are frozen:

1. The post-String semantic registry, including the generic String combinators
   and the twelve-entry generic ASCII-class catalog.
2. The broad public-TRAIN visible-I/O semantic engine across all nine TRAIN
   families.

Only then may the registry, atomic annotation manifest, and split manifest be
frozen and the TRAIN bridge spent.  Public DEV remains sealed throughout.  A
bridge PASS plus the separately frozen visible-I/O engine are both prerequisites
for the one paired 27-task DEV gate described in the roadmap.

## Frozen factor registry

The bridge uses independent one-vs-rest heads so a span may emit more than one
factor.  The post-String registry has exactly 24 learned factors in this order:

```text
add
subtract
minimum
maximum
absolute
less-than
map
sum
count
contains
filter
reverse
class:ascii-alphanumeric
class:ascii-decimal-digits
class:ascii-hex-digits
class:ascii-letters
class:ascii-lowercase
class:ascii-octal-digits
class:ascii-printable
class:ascii-punctuation
class:ascii-uppercase
class:ascii-whitespace
class:english-consonants
class:english-vowels
```

The registry is generic rather than shaped to the three String classes used by
the current TRAIN generator.  Factors with no legal TRAIN annotation keep
zero learned state and must never fire.

Compiler-structural evidence is fixed rather than learned:

- canonical `<ARG0>` and `<ARG1>` tokens project to the existing generic
  `SCOPE0` and `SCOPE1` source roles;
- `<INT>` plus its `SignedLiteralCopy` projects to `LITERAL` while preserving
  the physical typed value and token location;
- `BOUND0` and `bind` are supplied by typed grammar when `map`, `filter`, or
  `count` creates a binder;
- callable input/result types continue to come from `TypedSignature` and do
  not become language labels.

These fixed routes are authenticated physical projections, not a synonym
table or host semantic classifier.  The bridge must return factor events with
their span positions; it must never collapse them into one prompt union or an
unordered factor bag.

## Atomic TRAIN manifest

Atomic supervision is the sole additional information surface.  A lesson
manifest has a canonical top-level record:

```json
{
  "schema": "public-train-prompt-factor-lessons-v1",
  "scope": "PUBLIC_TRAIN_PROMPT_FACTORS_ONLY",
  "input_projection_sha256": "...",
  "factor_registry_sha256": "...",
  "lessons": [
    {
      "prompt_input_sha256": "...",
      "targets": [
        {
          "start_token": 8,
          "stop_token": 9,
          "span_identity_sha256": "...",
          "factor": "add"
        }
      ]
    }
  ]
}
```

Multiple targets may name the same span.  Every target must resolve to an
actual span in the authenticated `PromptSpanInput`, and every factor must be in
the frozen registry.  The trainer joins lessons to already observed prompt
inputs by `prompt_input_sha256`; it never accepts a `GeneratedTask` or a task
object.

The lesson interface and serialized records must reject any field or import
surface containing:

```text
task, family, ordinal, desired, output_case, regression, property,
candidate, program, ast_path, dev, eval, m0, heldout, c1
```

Annotations may be authored only by reading the public TRAIN prompt and naming
atomic operator/class factors already in the registry.  They may not be
derived from a family identifier, solution, candidate enumeration, sandbox
execution, or expected output.  Freeze the canonical manifest SHA-256 before
the first learning update.

## Physical anatomy

The banked input population has 512 fitted GLIF cells; exactly 12 spike for one
span.  The learner has one independent two-cell decision per learned factor:

- one fixed `STOP/ABSENT` fitted GLIF cell;
- one plastic `PRESENT` fitted GLIF cell.

The persistent learned synapses are exactly:

```text
excitatory_ticks: uint16[24, 512]
inhibitory_ticks: uint16[24, 512]
```

Each tick is 32 pA and is clipped to `[0, 50]`.  Persistent learned state is
therefore exactly:

```text
24 factors * 512 inputs * 2 matrices * 2 bytes = 49,152 bytes
```

There are 48 learned decision cells.  Twenty-four factor-specific teacher
cells exist only while presenting TRAIN lessons and are absent from frozen
inference.  Fixed source/literal projections are derived from the frozen token
assemblies, hashed separately, and do not add trainable arrays.  Runtime
membrane, refractory, and adaptation state is reset for every span decision
and is not checkpoint state.

Let `x[s, i]` be the binary spike of input cell `i` for span `s`, and let
`E[f, i]` and `H[f, i]` be the excitatory and inhibitory tick counts for factor
`f`.  The physical current is:

```text
D[s, f] = sum_i x[s, i] * (E[f, i] - H[f, i])
I_present[s, f] = max(600, 600 + (32 / 12) * D[s, f]) pA
I_stop = 650 pA
```

`I_present` is bounded above by 2,200 pA under the 50-tick cap.  Each row races
`[STOP, PRESENT]` from fitted GLIF rest for 600 steps using
[`CompiledRestingGlifRace`](../../src/humanbrain/resting_glif_race.py).
Complete-tier STOP precedence is mandatory: PRESENT is emitted only when it
spikes strictly before STOP; a simultaneous tier containing STOP is ABSENT.
This avoids an index tie-break.

The established cell model gives the intended two-error acquisition ladder on
this host: 600 pA first spikes at step 233, 632 pA at 219, fixed 650 pA STOP at
213, and 664 pA at 208.  Thus one uniform positive update remains ABSENT while
two make PRESENT win.

## Local update and schedule

For a factor teacher bit `y[s, f]` and physical prediction `p[s, f]`, the only
updates are:

```text
y = 1 and p = 0:
    E[f, active input cells of span s] += 1

y = 0 and p = 1:
    H[f, active input cells of span s] += 1

y = p:
    no update
```

The update is local: presynaptic span spikes, the postsynaptic PRESENT/STOP
outcome, and one factor-specific teacher pulse.  There is no global loss,
gradient, candidate score, or host-selected winner.

Training uses exactly four epochs.  Lessons are ordered by
`prompt_input_sha256`; spans are ordered by `(start_token, stop_token)`; factors
use registry order.  There is no shuffle or random seed.  All span/factor races
for one prompt use the weights at prompt entry, their local error increments
are accumulated, and the clipped integer updates are committed once at the end
of that prompt.  This defines deterministic mini-batch semantics and licenses
the batched fitted-cell path established at `fb83d20`.

## Deterministic 72/18 TRAIN split

The split is constructed once from the frozen 90-lesson manifest, before any
learning or metric is computed:

1. Validate 90 unique prompt-input identities and sort them by lowercase
   `prompt_input_sha256`.
2. Define an active factor as a registry factor occurring in at least one
   lesson.  STOP if any active factor occurs in fewer than three distinct
   prompts.
3. Iterate active factors in registry order.  If the current generalization
   set has no prompt containing that factor, reserve the lowest-hash prompt
   containing it whose removal still leaves at least two FIT prompts for every
   active factor on that prompt.  A prompt may cover several factors.
4. STOP if these coverage reservations exceed 18 prompts.
5. Fill the generalization set to exactly 18 using the remaining prompts in
   hash order, admitting a prompt only when every active factor still retains
   at least two FIT prompts.
6. STOP if 18 prompts cannot be selected.  The remaining 72 prompts are FIT.

The canonical membership list and algorithm constants form the split manifest
and are hashed before training.  No task/family/ordinal field participates.
The 18 prompts measure held-out TRAIN payload recognition only; repeated spans
and prompt templates may cross the split, so this is not evidence of
paraphrase or family-disjoint generalization.

After the split gate passes once, construct a fresh zero-state learner and fit
the same four-epoch schedule on all 90 TRAIN lessons.  That all-TRAIN
checkpoint, not the 72-prompt development checkpoint, is the sole frozen bridge
for the later paired DEV gate.  No constant, factor, or annotation may change
between the split run and final refit.

## Required controls and interventions

Every control uses the same prompt inputs, race count, schedule, and current
constants:

- **Label-off:** every teacher pulse is zero.
- **Shuffled-span:** within each prompt, rotate the ordered span assemblies by
  one position while leaving target positions and labels unchanged.
- **Untrained:** evaluate the canonical zero-tick checkpoint.
- **Renaming invariant:** declared entrypoint/argument renaming must preserve
  semantic events.
- **Operand order:** minimal pairs that exchange declared operands must produce
  different ordered role-event records even when their unordered factor sets
  match.
- **Literal intervention:** changing only a signed integer must preserve the
  learned lexical-factor events and change only the literal-copy event/value.
- **Audit-hash intervention:** changing only the audit whole-prompt SHA in an
  otherwise identical input record must not change currents or factor events.
- **Freeze:** repeated inference must leave the checkpoint digest byte-identical.

The shuffled-span control tests the causal span-to-factor association rather
than merely permuting label names.  No control may be used to choose a seed,
threshold, epoch count, or vocabulary.

## Checkpoint identity

The checkpoint digest is SHA-256 over a namespace, canonical JSON metadata,
and then every array in sorted name order as `(name, dtype, shape, raw
little-endian bytes)`.  Metadata contains:

- schema and public-TRAIN-only scope;
- source identities including `c89a3c6`, `00ed3e5`, and `fb83d20`;
- prompt tokenizer and 12-of-512 assembly-projection SHAs;
- live typed literal/output projection SHA;
- ordered 24-factor registry and registry SHA;
- fixed source/literal projection SHA;
- fitted GLIF parameter and compiled-race SHA;
- 600 pA base, 650 pA STOP, 600 race steps, 32 pA tick, 50-tick cap;
- four epochs and the exact canonical ordering rules;
- lesson-manifest and split-manifest SHAs;
- FIT/generalization/final-refit counts;
- local-update and physical-race counters;
- exact names, dtypes, and shapes of both tick arrays.

Teacher targets are not embedded in the inference checkpoint.  Both tick
arrays are made read-only after construction, and inference authenticates the
checkpoint before and after use.

## PASS and STOP rules

Preflight returns `STOP_PUBLIC_TRAIN_PROMPT_FACTOR_IDENTIFIABILITY_V1` before
learning if:

- atomic labels cannot be assigned from public TRAIN text under the allowed
  manifest interface;
- an annotation would require forbidden information;
- a label is outside the frozen registry;
- a target span does not authenticate against its prompt input;
- active-factor coverage or deterministic 72/18 construction fails.

The one split run returns `PASS_PUBLIC_TRAIN_PROMPT_FACTOR_BRIDGE_V1` only if
all conditions hold:

1. FIT produces exact ordered learned-factor events on 72/72 prompts.
2. Frozen TRAIN generalization produces exact ordered learned-factor events on
   18/18 prompts.
3. Literal copy values and locations are exact on all 90 prompts.
4. Every positive learned-factor event strictly precedes STOP; every negative
   event is STOP-first or STOP-tied; no inactive factor fires.
5. Label-off and untrained controls pass 0/18 prompts exactly.
6. Shuffled-span passes at most 2/18 exactly and trails the trained macro-F1 by
   at least 0.30.
7. All renaming, operand-order, literal, audit-hash, and freeze interventions
   pass.
8. The frozen checkpoint is byte-identical before and after every inference.

Any learned-capacity or control failure returns
`STOP_PUBLIC_TRAIN_PROMPT_FACTOR_CAPACITY_V1`.  The spent split is not tuned or
retried under a new seed.  A PASS licenses only the fixed all-90 TRAIN refit.
The final refit must reproduce exact ordered events on 90/90 TRAIN prompts and
then freeze byte-identically; otherwise it returns
`STOP_PUBLIC_TRAIN_PROMPT_FACTOR_FINAL_FREEZE_V1`.  No outcome opens DEV by
itself.

## Runtime and memory estimate

Two local mechanism timings bound the first implementation on this host:

- a cold 11-token, 38-span `c89a3c6` prompt observation took approximately
  1.386 seconds; an identical cached observation took approximately 2 ms;
- one compiled batch of 2,400 independent two-cell races took approximately
  19.6 ms.

The factor-head portion of four-epoch training over 90 prompts should therefore
take roughly 10--30 seconds.  Cold physical span materialization is expected to
dominate at roughly 1--4 minutes depending on shared-span cache hits.  Working
memory should remain below approximately 20 MB, while the learned arrays are
exactly 49,152 bytes and the complete checkpoint should remain roughly
50--65 KB including fixed projection records and metadata.

Cold inference on a genuinely novel prompt is still likely to spend seconds in
the current per-span physical drive.  After the first correct bridge exists,
the next runtime-only increment should batch distinct `_drive_assembly` rows
behind exact per-span hash/first-spike parity.  That is engineering evidence,
not a capability rerun.

## Identifiability assumption

The design assumes explicit atomic factor teacher pulses are permitted on
public TRAIN.  This is the minimum information required to choose a named
compiler factor: without such pulses, any permutation of the factor-cell labels
is observationally equivalent, so prompt text alone cannot identify which cell
means `add`, `filter`, or another compiler atom.  If the atomic manifest cannot
be frozen without forbidden information, the correct result is the
identifiability STOP above.  It is not permission to substitute a regex parser,
synonym catalog, desired-output learner, solution program, or host classifier.
