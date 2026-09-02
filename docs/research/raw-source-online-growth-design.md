# Replay-free online growth for the raw-source motor controller

Status: implemented first slice in commits `88049e3` and `d10e09c`.
This is public-TRAIN mechanism evidence, not a protected capability result.

## Decision

Grow the cue-free raw-source motor system as an immutable collection of frozen
neuronal shards. One authenticated successful experience can add one shard;
earlier training examples are not read, replayed, or rebuilt. Every shard runs
on inference and one fitted-GLIF ensemble race selects a sole emitter or
physically selects STOP on absence or collision.

This is intentionally smaller than a global stable-cell/synapse ledger. It uses
the already banked binder, source-relative tape, and training compiler, and adds
only the persistence boundary required by the integrated failure. A flat global
ledger may become worthwhile when many shards expose a measured consolidation
or throughput bottleneck; it is not claimed by this implementation.

The design follows the useful part of progressive expansion: preserve working
capacity and add new capacity without retraining it on old data. It does not
import task-labelled columns, gradient replay, or a task router.

## Successful-experience custody

Growth does not accept a bare `passed=True` assertion.

`SourceMotorExecutionAuthority.execute_and_issue(...)` invokes one caller-pinned
evaluator. Only a literal `True` result creates a private
`SourceMotorExecutionReceipt`, which binds:

- the exact parent learner checkpoint;
- the complete context/before observation;
- the emitted action;
- the evaluator commitment; and
- the PASS outcome.

`SourceMotorSuccessEvidence` binds that receipt to the complete training
experience and provenance root. `extend_success` authenticates both records and
rejects a receipt issued for another or older learner. This is an auditable
in-process custody boundary, not cryptographic protection against a malicious
Python caller; the experiment runner remains responsible for pinning and
authenticating the real evaluator.

## One growth transaction

For one authenticated current experience, `extend_success`:

1. verifies the exact parent checkpoint and all frozen component roots;
2. calls `compile_successful_token_tapes` once on the current before/action pair;
3. grows a binder and motor tape using only that compiled current experience;
4. checks whether the context/program is new, duplicate, or conflicting;
5. returns the unchanged parent for duplicate, conflict, or `NO_UPDATE`; or
6. appends one frozen shard and one provenance link.

The motor tape currently represents support as demonstration rows. The same
current demonstration is supplied twice because the one successful experience
contributes two local support phases: structural coactivation and positive
success. These are two support ticks, not two experiences and not replay of a
parent example.

A known context with an incompatible trajectory collision-STOPs and allocates
nothing. `NO_UPDATE` deliberately performs the same authentication,
compilation, and admission work, then stops before the append. This makes the
causal comparison persistent treatment versus persistent no-update, not trained
versus reset.

## Frozen state and read-only inference

The existing controller and action tape contain mutable pending-receipt fields,
and the tape contains a mutable consequence-adjustment map. The online wrapper
therefore treats them as construction-time mutable objects and run-time frozen
state:

- `seed` copies the caller's controller and every mutable tape field;
- an appended shard owns copied binder/tape state;
- each response makes an ephemeral copy, resolves any emitted receipt neutrally
  on that copy, and discards it; and
- parent and sibling shards share only stored objects that remain pending-free
  and checkpoint-identical.

The final decision is not the first host result. All stored shards respond, then
their emit signals enter one fitted-GLIF race with a tonic STOP cell. No emitter
selects STOP; two emitters tie and collision-STOP.

## Honest accounting boundary

The implementation does not expose a global physical cell/synapse ledger.
It reports:

- reserved cell capacity;
- estimated reserved synapse capacity;
- estimated coactivation, positive-success, total excitatory, and inhibitory
  support ticks;
- source experience count and committed support phases;
- parent-experience replay count; and
- raw training sequences retained.

These values are derived from the banked component summaries and shard layout;
they are capacity/support estimates, not asserted physical IDs or exact edge
arrays. `raw_training_sequences_retained=0` means the complete before/after rows
are discarded after construction. Learned context and neighborhood event bytes
remain inside the binder/tape as synaptic content and are explicitly not counted
as zero retained knowledge.

## Banked result

The real-Python session begins with independently seeded treatment and
`NO_UPDATE` learners at identical roots. Both initially STOP on the new exact
context. One teacher correction fails before execution, is assembled only from
current workspace-tape occurrences, passes the exact-output evaluator, and is
issued once as identical evidence to both arms.

Treatment appends one shard and passes 2/2 later renamed, path-shifted
repositories. `NO_UPDATE` commits zero state and physically STOPs 2/2. Earlier
base behavior is unchanged, and activating the base and new context together
collision-STOPs. The appended shard reports estimated reservations of 4,387
cells and 61 synapses plus 110 support ticks (61 coactivation and 49 positive
success), one source experience, zero parent replay, and zero retained raw
training sequences.

## Claim boundary and next step

This demonstrates replay-free acquisition, retention, and fresh same-layout
source-relative recurrence for one exact context. It does not demonstrate
learned context abstraction, language understanding, motor-program induction,
multi-file issue grounding, autonomous recovery, or broad repository
intelligence.

The next integrated step is to combine this persistence boundary with the
lossless multiscale observation and learned cross-stream source grounding. That
larger persistent learner should expose whether frozen shards need learned
consolidation/routing or whether a different architecture should replace them.
