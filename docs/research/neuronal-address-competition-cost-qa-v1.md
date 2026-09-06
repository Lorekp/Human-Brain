# Competitive cost/custody QA contract

Root selection for source implementation and exact ordinary QA, 2026-09-06,
after banked roster and worker QA.
No production imports, builders, model/checkpoint calls, cap forwards, children,
test collection or QA execution were performed for this note. Root reviews the
final appended nodes and held source before execution; this is not cost
or acquisition admission. Existing component, receipt, data and isolated-worker
QA are not repeated.

## Stable seams and scope

Sartre confirmed the v2 checker retains `project_cost(report, outer, *, audit=None,
audit_outer=None)`, `validate_cost_decision(path, expected_sha, expectedconfig,
expectedsources, rostersha)`, `check_monitor(outer, *, pid, command=None)` and
`check_worker_custody(output, report, worker, requests, lineage, outer)`.
Use those actual functions, not copied implementations. Relevant existing homes
are `test_splice_online_saved_audit_arithmetic` and
`test_splice_online_saved_cost_decision_custody` in the decoder test file.

Five ordered arms are learned, action_frozen, slow_frozen, memory_disabled and
marginal_reference. Required production cardinalities are:

| Scope | Cost | Full |
| --- | ---: | ---: |
| Regular learning pairs | 20 | 1,280 |
| Disposable cap learning pairs | 5 | 0 |
| Endpoint proposals / acknowledgments | 240 / 240 | 240 / 240 |
| Shared category verifications | 4 | 0 |
| Total actual action verifications | 269 | 1,520 |
| Old-panel reader calls | 200 | 200 |
| Reader teacher offers, including caps | 120 | 2,400 |
| Action-head labels, including caps | 65 | 3,200 |
| Retained tapes / model-plus-tape bytes | 360 / 239,657,600 | same |

The current actual roster maximum is 125 occupied slots. Capacity and disposable
caps remain 128. Cost and full retain/export/load all 72 tapes in each arm;
there is no full-only payload increment. Independent audit source must check all
240 endpoint pairs and 100 rows in each panel, plus cost/full learning counts.
The proposed synthetic nodes below do not claim to execute that complete audit
or validate checkpoint tensors; they test its arithmetic and custody gates.

## Three proposed non-neural nodes

### 1. `test_competitive_splice_saved_audit_arithmetic`

Adapt the existing small timing fixture to five explicit arms. Regular per-arm
rows=4, totals M/A/J=.04/.08/.004 and maxima=.01/.02/.001. Cap rows=1 with
M/A/J=.03/.015/.002, except marginal_reference M=.04. Four category A values
have maximum .025. Cost outer=20 seconds; parent/worker peaks=300/200 MiB,
owned-tree peak=550. This asymmetry makes silently dropping the fifth arm visible.

Independent hand arithmetic: regular sum=.62, effective stage maxima sum=.295;
`P_run=512*.295+2*(20-.62)+30=219.8` seconds. Common E=24 plus run reserve128
makes run memory `max(300+200,550)+24+128=702` MiB.

Audit regular rows=4, total=.004/max=.001 per arm; cap rows=1, total=max=.002,
except marginal_reference=.003. Finished audit outer=5 seconds, peak250 MiB.
`P_audit=2*(256*.011+5-.020)=15.592`; complete total235.392 seconds,
audit memory306 MiB, final peak702. Supply an actual-shape synthetic successful
watchdog record with unambiguous .NET creation ticks within its launch window.

Exactly seven direct `project_cost` calls:

1. Preliminary:219.8/702, audit pending, never admissible.
2. Completed:235.392/702, admissible, all five `per_arm` entries present.
3. Category A=.5: complete time exceeds900, inadmissible (not an exception).
4. Audit owned peak2000: audited projection2056 MiB, inadmissible while the
   synthetic observed peak itself still fits2048.
5. Fifth arm regular rows=3: `ValueError` for incomplete scope.
6. Supply finished audit without its monitor: `ValueError` for unfinished audit.
7. Finished audit monitor exit_code=1: `ValueError` for failed completion.

Also assert the exact held checker config: five policy mappings, 72 tapes/arm,
239657600 bytes, 2 MiB line/tree, E24, reserves128/32, F0 and900/2048 ceilings.
The full production count checks remain source-reviewed integration checks;
do not manufacture a second full fake 1,520-action experiment merely for counts.

### 2. `test_competitive_splice_saved_cost_decision_custody`

Reuse the established synthetic filesystem fixture: create a tiny inert checker
source and authored JSON files beneath `tmp_path`, redirect only checker ROOT
and __file__ for pathname/source context. Never patch hashes, monitors,
projection, source readers or any neuronal function. Use the actual v2 schemas
and exact selected `.venv` command, the complete six named evidence inputs,
the actual checker/roster input bindings and the preliminary/final projections.
Use the same five-arm timing data as node1. No real qualification, roster or
checkpoint artifacts are read by this synthetic fixture.

Exactly five direct `validate_cost_decision` calls: one valid baseline, then four
independent `ValueError` variants. Restore the baseline for each variant, change
the intended inner evidence, and rehash its enclosing JSON so failure exercises
the inner condition rather than merely an outer hash mismatch:

- Audit `input_sha256` names the wrong cost report hash.
- Audit command invokes the old v1 checker module instead of v2.
- Claimed final projected seconds and its duplicate scalar are reduced by1.
- Inert checker source bytes change while authenticated expected source remains
  the original; the completed audit must reject the held-source mismatch.

This checks finished-audit necessity, exact source/code version, selected input
custody and independent admission arithmetic. Nested ordinary validator calls
are included in these five calls; no new measured resource result is produced.

### 3. `test_competitive_splice_worker_saved_custody`

Build one minimal synthetic launch/completion/request/lineage family with all
five final arms, experience4 before and52 after the48 endpoints, and declared
16-file worker inventory (15 module origins plus worker script). Store only the
authored launch JSON and empty stderr file that the real checker reads. Model,
tape and checkpoint identities are opaque authored hashes; no tensor file is
created or loaded. Use the actual `check_worker_custody` and real file hashing.

Exactly seven direct calls: two valid forms (actual interpreter is launcher,
and actual interpreter is its distinct direct child), then five independent
`ValueError` variants:

- Completion declares single-exchange QA scope instead of integrated.
- One actual module origin points outside the isolated package.
- Worker creation ticks precede launch while its PID remains numerically correct.
- Fifth arm final experience is51 instead of52.
- The actual launch command's request hash argument changes; update only the
  enclosing launch-file hash so exact command/input custody rejects it.

Both valid forms include observed launcher and interpreter identities. Use .NET
UTC ticks `621355968000000000+unix_seconds*10000000`, not Windows FILETIME.
No actual process discovery, subprocess, worker invocation or source import is
part of this node. This supplements the already completed real worker QA; it
does not replace that evidence or pretend a saved PID proves execution alone.

Total proposed public validation calls:19 (7 arithmetic,5 decision,7 worker):
five valid baseline returns, two valid-but-inadmissible budget returns and twelve
`ValueError` rejections, as itemized above.
Save the actual authored inputs/outcomes and source hashes before outcome
assertions, using separate exclusive raw and passed receipts. Preserve any
failure; do not automatically rerun or expand the fixture grid.

## Cap source geometry and actual free/teacher paths

The parent reuses the unchanged `cap_fixture(model)` before regular learning.
It deep-copies one current arm at a time and explicitly assigns only disposable
weights: address B=0; action start/end weights=0; operation weights=0/bias[1,0];
codec diagonal32, bias0, EOS column -32 except A row0 and EOS row log9.
Start/end heads have no bias parameters. No weights change during a forward.

Seven documents of16 A plus one of8 A yield120 raw bytes+8EOS=128 occupied
slots. Four128-byte files give516 raw start candidates; lower-index start ties
choose main.py offset0, giving129 end candidates, whose tie chooses end0. The
31-byte raw request reaches reader/action unchanged. Operation is STOP. Every
arm therefore performs one real unchanged-workspace verification before its
corrective16-A SPLICE insertion,17 reader offers and three action-head labels.
The corrective insertion is a synthetic cost stimulus, not an extra executed
program or skill claim.

For the four competitive arms, B=0 gives128 exactly tied zero scores. All8128
pair currents are0 and comparators fire. Winner current at index i is -i, so
only index0 fires; identity readout is0. This is two layers/8256 threshold
neurons, not host argmax. Diagnostic prior is uniform; actual free belief is
one branch. The full128-row codec bank is evaluated once, with17 single-cursor
gathers including the first (17*7196=122332 gathered bytes).

With sensory enabled, cursor0 follows the first document through positions0..15
containing A, then EOS16. A sensory has A logit32 and all others0, so A wins;
EOS sensory has EOS logitlog9, A0 and all others-32, so EOS wins. Thus learned,
action_frozen and slow_frozen emit16 A and EOS on decision17. In memory_disabled,
all sensory and keys are zero: every codec response is uniform from bias0.
Token0 wins every free decision, giving16 zero bytes and LIMIT on decision17;
occupied adjacency and the one actual cursor still advance. No fake final EOS.

The marginal_reference uses the same finite cap weights,128 initial branches
and inherited marginal policy. Its established scalar derivation yields16 A
then EOS17 (penultimate A/EOS margin>.0526, final EOS/A margin>.7999). It gathers
all128 responses on the16 later decisions:16*128*7196=14737408 bytes. This
source/scalar expectation must still match the actual admitted cost receipt;
no new cap execution is authorized here.

All five corrections reopen128 branches with the unchanged local teacher;
competitive threshold work is absent from teaching. Full17-symbol teacher and
three-head arithmetic runs even for frozen publications. Reader publications
are enabled in learned, action_frozen, memory_disabled and marginal_reference
(68/85 cap offers enabled); slow_frozen sees17 without publication. Action
publications are enabled in learned, memory_disabled and marginal_reference
(9/15 cap head labels); both freeze controls see their labels without publishing.
Actual changed tensors remain observed facts, not inferred from enabled flags.

Including regular four-ID samples, prospective cost totals120 reader offers
(96 enabled) and65 action labels (39 enabled). These are declarations for the
actual future cost, not additional QA exposure. Construction/guard/journal time,
all269 actions,200 panel calls and all240 endpoint exchanges remain in measured
cost; no timing assumption or selected ceiling is changed by this plan.

## Selected launch boundary

Use only the three exact proposed nodes above, together in one launch after
root source review, committed source/test/contract and a clean selected-checkout
doctor. Preserve the 179 existing test AST nodes. Parent and auditor draft
guards may remain enabled: these checks call only the explicit pure helpers,
not the actual run/audit entry points. Never patch those guards for QA.
Use a fresh `.pytest-tmp/neuronal-address-competition-cost-qa-v1` basetemp,
watchdog/log prefix and JUnit path, at 60 seconds and 2,048 MiB with the documented
CPU/thread/plugin environment. Count actual validations and save authored
inputs/outcomes exclusively before assertions. Source/hash/PID checks here use
synthetic identities and inert files; they produce no scientific model output.
A failure is retained before reviewing any repair or changed exposure.

After these checks pass, root may remove only the two explicit draft guards,
verify all qualified function bodies are unchanged, freeze that exact final
source and complete new cost/audit admission. Guard removal is not acquisition
admission. Do not rerun passed component, receipt, data or worker nodes. No
original checkpoint load, cap forward, roster generation, real worker/action,
actual audit or acquisition is selected by this QA contract.
