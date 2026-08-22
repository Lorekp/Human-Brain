# Simple-SWE opening preregistration (v1)

Status: DESIGN — nothing opened, frozen, or run. Provenance: researched by
subagent through freeze/reveal/adapter/runner sources and its own test-suite
probing (transcript-mined after a final-answer failure), authored by
orchestrator; verify pass REQUIRED before any merge or ceremony execution.

## What the research established (verified facts)

- The intended freeze ceremony exists: `create_freeze_manifest`
  (freeze.py:503-504+) freezes protocol bytes, checkpoint, clean source tree
  (HEAD + ls-tree SHA-256), adapter/compiler identities, sandbox runtime
  (python executable + hash + flags `-I -S -B`), worker, simulation backend;
  written atomically as `freeze.json` OUTSIDE the source root.
- External anchoring of the evaluation seed is enforced by tests:
  reveal refuses direct anchors without a public receipt
  (`test_reveal_rejects_a_direct_anchor_without_a_public_receipt`), requires
  nonempty https authority, persists + commits the seed before any task
  generation, and crash-after-reservation resumes the same seed without
  reroll (`perm_appendix.json`-style commitment discipline).
- Baselines are structurally first-class: a test asserts baseline controls
  CANNOT precede or change the cell summary.
- IMPORTANT environment finding from the agent's probing: the sandbox
  runtime resolution currently FAILS on this host for the pinned interpreter
  ("No runtime installed that matches AstralCPython3.11.15" against
  Python 3.14 default), and several freeze/reveal tests fail on this host
  for environment reasons. Opening requires fixing the local runtime pin
  FIRST — otherwise the ceremony cannot execute honestly.

## Freeze order (what gets frozen, in order)

1. Fix host runtime pin (install/harness a 3.11 interpreter matching the
   sandbox contract) and get test_simple_swe_freeze/reveal green locally.
2. `create_freeze_manifest` with: protocol path, grown checkpoint (from the
   a013 L1 bridge arm), clean source tree at the pinned HEAD, adapter +
   compiler identity hashes, sandbox runtime identity.
3. Evaluation seed: generate externally anchored commitment (https receipt),
   committed to git BEFORE task generation; reveal then derives the cohort.
4. Only then: task generation -> hidden-case sealing.

## Minimal capability question (v1)

After grow on M training tasks from visible families, pass@1 on K tasks
drawn from the SAME families (capacity gate) AND K' tasks from held-out
families (transfer gate), scored by hidden-case oracle execution in the
sandbox. All-or-nothing per family as everywhere else.

## Adapter path (verified end-to-end components)

checkpoint -> adapter (cell completion -> program text via compiler) ->
sandbox subprocess run with hidden cases -> oracle verdict -> artifacts.
The agent verified each module exposes its link (adapter.py, compiler.py,
sandbox.py subprocess contract with `-I -S -B`, runner orchestration);
remaining gap is exactly the runtime-pin issue above plus an end-to-end dry
run that has never been executed.

## Preregistered baselines (before any cell run)

- B-random: uniform program sampler over the grammar space.
- B-template: heuristic template-filler keyed on task signature.
Both must be executed under the identical freeze/sandbox/oracle path; their
results are locked into the same evidence envelope BEFORE cell completions
are unsealed (enforced by the existing baseline-ordering test).

## Dependencies / sequencing

Hard prerequisites, in order: (1) representation choice frozen (codebook);
(2) ladder L1 bridge arm executed with the chosen readout — its checkpoint
is what gets frozen here; (3) host runtime pin fixed. Opening before (1)+(2)
would measure nothing but the current sealed failure mode again.

## Red tests

- Runtime-pin test: sandbox refuses wrong-interpreter hosts loudly (already
  exists; must pass locally before ceremony).
- Seed-commitment ordering tests (exist; must pass).
- Baseline-before-cell ordering test (exists; must pass).
- NEW: end-to-end smoke — one dummy frozen manifest + one synthetic
  checkpoint through adapter->compiler->sandbox->oracle on this host,
  asserted green, BEFORE the real opening.
