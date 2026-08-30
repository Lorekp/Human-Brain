# Compact typed-output encoding protocol

Status: executed and banked; all 44 frozen gates passed.  The six-family
public-TRAIN breadth gate remains unopened.

Date: 2026-08-30.

## Decision and sequencing

Replace the production `TypedOutputEncoding` dense 3,284-cell vector with one
immutable compact vector of local factor winners.  Do not retain a dense
compatibility property, feature flag, alternate backend, or runtime fallback.
Whole-projection materialization is permitted only through an explicitly named
audit operation.

The generic String mechanism is already banked at `3daea4c` and recorded at
`67b1192`.  This migration must occur in the following order:

1. bank the private semantic-frontier work-bound implementation, including its
   256 / 6,000 / 30,000 / 80,000 limits and complete aggregate-plus-String race
   admission table;
2. record the post-work-bound checkpoints, race ledgers, counters, and runtime
   baselines named below;
3. land this compact-output migration as one production cutover;
4. pass the parity, memory, runtime, and work-bound gates in this document;
5. only then open the broad six-family public-TRAIN breadth gate.

The work bound comes first because it is the smaller runtime-safety mechanism
and fixes the final STOP/counter semantics that this representation change must
preserve.  The two changes both touch the synthesizer and must not be developed
as overlapping edits.  The six-family TRAIN desired vectors and property rows
remain sealed throughout implementation and tuning.

## Banked execution result

The dense baseline was banked first, the exact retired 19-source / 21-test
execution closure was removed at `39a6b16`, and the production cutover landed
at source commit `82b57a0222a2123706d1b2a39f37c10733be73c7`.  The sole compact
capture is:

```text
experiments/r3_pilot/compact_typed_output_compact_v1.json
raw SHA-256:     f503820711ebc597aeac6b96e6d89dce17c48be7b756b182619a7238b531e179
payload SHA-256: beab356461d443fc5ac5f5cb5198e755e3e6927b48a8aa860bc8fdbd68b3604e
verdict:         PASS_COMPACT_TYPED_OUTPUT_ENCODING
publication:     PUBLISHED_COMPACT_PASS
```

Independent loading recomputed the payload digest, and the raw file digest
matched the capture report.  All 44 comparison gates passed.  The artifact
records a clean tracked worktree and clean relevant sources at the source
commit, zero external task/corpus inputs, zero production audit-materialization
calls, protocol SHA-256
`c4d16ba0b0830aaa608eaeaf62d9844baf117d1c6133147517943523c7b65dd4`,
harness SHA-256
`c7adefc019b5d8b6268c7788833c22f8202685884aab692931b635b04c9da874`,
environment SHA-256
`6664ec6870408ac1a3195567d4c525a71463cce3dc8a6e3a85b648f6e263187b`,
and exact scientific golden
`287987cea7306778d79ba0e726b50fde304860e00f8ae8b2d2057b0c432920ee`.

Measured tracemalloc peaks were 705,722 bytes for 2,000 String encodings,
592,795 bytes for 1,000 detectors, and 729,906 bytes for the String-heavy
frontier.  Immediate/recorded reductions were 76.76x/75.75x for Strings and
51.40x/44.89x for detectors; the frontier retained 1.35% of its dense peak.
Every runtime fixture preserved its exact result hash and passed its frozen
limit: detector grid 0.004114 s, 64-row GLIF kernel 0.0000209 s, aggregates
0.0834--1.6950 s, and String fixtures 0.0814--0.1869 s.  Sentinel A preserved
its ordinary exhaustive STOP and race digest in 21.812 s; Sentinel B preserved
its prediction-first 30,000-prediction work-bound STOP and completed-prefix
digest in 3.576 s.  This closes only the compact representation/runtime
prerequisite; it does not spend or strengthen any coding-capability gate.

## Why the current representation stops scaling

The physical output projection contains these fixed banks:

| Factors | Slots | Local bank width |
| --- | ---: | ---: |
| type | 1 | 5 |
| Int | 1 | 130 |
| Bool | 1 | 3 |
| `List[Int]` length | 1 | 6 |
| `List[Int]` items | 4 | 130 |
| Char | 1 | 96 |
| String length | 1 | 28 |
| String characters | 26 | 96 |

There are 36 factor slots but 3,284 globally addressed cells.  A supported
value contains only 2--28 one-cell events.  Nevertheless every current
encoding owns `float64[3284]` plus `bool[36]`, or 26,308 raw bytes.  The
semantic frontier and its detector bank retain many such objects, so String
capacity currently multiplies memory by projection width rather than by
physical events.

Measured source-independent baselines on the current dense representation are:

- 2,000 retained varied/max-length String encodings: 50.98 MiB;
- 1,000 fixed typed-coincidence detectors: 25.38 MiB;
- 5,000 independent scalar detector runs: 2.985 s;
- the same fitted cells in one batch: 0.00168 s;
- a 500-candidate by four-case detector grid: 1.206 s before batching and
  0.00437 s with the exact batched physical result.

These are synthetic engineering baselines, not coding-capability evidence.

## Frozen compact representation

`TypedOutputEncoding` becomes a frozen, slotted, `init=False` value with exactly
two stored payloads.  It has no public constructor; only the bank's canonical
constructor may assign its fields:

```text
factor_winners_le: 72 immutable bytes
physical_digest:   32 immutable bytes
```

`factor_winners_le` is the current `FACTOR_LABELS` order encoded as 36
little-endian unsigned 16-bit words.  An active word is the local winner index
inside that factor's bank.  `0xffff` means the factor is inactive.  A bank width
of 65,535 remains valid, with indices `0x0000..0xfffe`; module import must fail
only when a bank width exceeds 65,535.

The raw physical payload is therefore 72 bytes instead of 26,308 bytes, a
365-fold reduction.  Winners plus the binary SHA-256 digest occupy 104 raw
bytes, approximately 253-fold smaller.  Python object overhead is measured
separately by the acceptance gates and is not included in those ratios.

`output_type`, `supported`, `unknown_active`, `active_factors`, and the
lower-case hexadecimal `sha256` string are derived properties.  They are not
stored as competing sources of truth.  A supported output has an active known
type winner and no active UNKNOWN winner.  Construction is bank-owned and must
enforce the complete canonical shape:

- untyped UNKNOWN has no type winner and activates all 35 non-type factors at
  their respective UNKNOWN winners;
- a typed scalar activates type plus exactly its scalar factor, with UNKNOWN
  allowed only for an unsupported encoding;
- a known List or String length `n` activates type, length, and positions
  `0..n-1`, leaving later positions inactive;
- an unknown List or String length activates type plus the UNKNOWN length and
  no positions;
- an in-capacity invalid item or character remains active at that position's
  UNKNOWN winner and makes the encoding unsupported.

The immutable backing bytes, rather than a write-disabled array with a mutable
owner, are the source of the immutability guarantee.  At every external
encoded-object ingress the owning module validates the byte lengths, winner
ranges, sentinel placement, canonical factor shape, derived type/support state,
and a freshly recomputed physical digest before trusting the object.  Enclosing
operator/synthesizer entry points validate once and pass only trusted encodings
through their internal loops; this does not restore a forgeable constructor.

## Deep-module ownership and interface

`experiments/r3_pilot/public_typed_output_cells.py` owns one deep in-process
module.  It centralizes canonical construction, factor addressing, physical
drives, decoding, exact-target coincidence, audit projection, and scientific
hashing.  No port or adapter is needed between Python modules.

The production interface is:

```text
TypedOutputCellBank.encode(value, declared_type=None)
    -> TypedOutputEncoding

TypedOutputCellBank.decode(encoded)
    -> canonical Python value or raises for UNKNOWN/noncanonical input

TypedOutputCellBank.projection_sha256()
    -> unchanged physical-projection digest

TypedOutputCellBank.validate(encoded)
    -> same encoded object or raises after canonical/digest recomputation

TypedOutputEncoding.factor_winner(factor_slot)
    -> local int winner or None

TypedOutputEncoding.active_global_cell_indices()
    -> strictly ascending tuple of existing global projection addresses

TypedOutputCellBank._from_physical_winners(output_type, winners)
    -> canonical TypedOutputEncoding

TypedOutputCellBank._drive_factor(encoded, factor_slot, current_pa)
    -> contiguous float64 current vector for that local factor bank
```

`_from_physical_winners` accepts 36 `int | None` entries, validates the same
canonical shapes as `encode`, and packs them.  It is the operator module's
internal seam for outputs that have already won physical races.  It must not
host-decode a winner and bypass or replay the physical operator.

`_drive_factor` materializes at most one local bank, never the 3,284-cell
projection.  It places `current_pa` at the recorded winner and zero elsewhere,
which must reproduce the exact byte sequence currently obtained by multiplying
the stored one-hot factor bank.  An inactive factor returns an all-zero vector
of the factor's exact local width.  `active_global_cell_indices()` adds each
local winner to its existing global bank offset and returns the addresses in
strictly ascending order, exactly matching `flatnonzero(dense_cells)`.

The sole whole-projection operation is:

```text
TypedOutputCellBank.materialize_for_audit(encoded)
    -> DenseTypedOutputAudit(cells=float64[3284],
                             active_factor_mask=bool[36])
```

Both arrays are immutable and contiguous.  This operation is excluded from
production synthesis, operator evaluation, semantic racing, retention,
checkpointing, and prompt-literal handling.  A test must fail if any production
path calls it.

`FixedTypedCoincidenceDetector.for_target`, `detect`, and `detect_many` retain
their caller-facing signatures.  The frozen, slotted detector also becomes
`init=False`; `for_target` is its sole canonical constructor.  It stores the
validated compact target, not a dense weight vector.  For each prediction it
counts equal winners only over the target's active factor slots.  Extra
predicted factors remain unpenalized, matching the current dense dot product.
The existing bias, clipping, fitted GLIF batch, and result fields remain
unchanged.  A disabled target always has zero matched factor cells, regardless
of a prediction's winners, and retains its fixed inhibitory current and
no-spike result.

The detector may expose one module-private `_detect_many_trusted` seam for an
enclosing operator or semantic entry point that has already called
`TypedOutputCellBank.validate` on every row.  Public `detect` and `detect_many`
must still validate external ingress; only the prevalidated private seam is
used inside the timed 500-by-four physical batch loop.

## Complete production consumer cutover

### Active bottom-up path

- `public_typed_output_cells.py`: replace dense storage, `_finish`, bank slices,
  detector weights, and dense batch stacking with the deep interface above.
- `public_typed_operator_successor.py`: replace typed `_single_event` scans by
  `factor_winner`, replace contact multiplications by winner-column gathers,
  and replace copy currents by `_drive_factor`.  Rebuild String race winners
  through `_from_physical_winners`; do not retain its direct `_finish` call.
  Int/Bool/List/Char copy, String copy, `contains`, `filter`, and `reverse`
  continue to run their present physical races with byte-identical currents.
  The separate 13-cell `AsciiClass` and 1,141-cell class/Char coincidence
  mechanism are intentionally out of scope.
- `public_typed_semantic_synthesizer.py`: replace the private dense Int, Bool,
  List, and String decoders with `TypedOutputCellBank.decode`.  Retention,
  effect memoization, sum/count/clamp, and String composition continue to key
  on the unchanged encoding digest.
- `public_typed_semantic_vector_race.py`: retain its interface and use the
  compact detector batch.  It must not materialize candidates.
- `public_typed_semantic_simple_swe_train_bridge_v1.py`: retain its existing
  encode/projection calls; no representation access is allowed.
- `public_train_prompt_semantic_bridge.py`: replace
  `flatnonzero(typed_output.cells)` with
  `active_global_cell_indices()` and preserve the exact recorded global cell
  tuple and every downstream literal hash.
- `src/humanbrain/typed_program.py` and
  `src/humanbrain/typed_semantic_synthesis.py`: compiler and semantic program
  behavior remain unchanged.  They do not receive a dense-output adapter.

Every bullet that accepts an already-created encoding validates it at its
enclosing public/module entry point.  `decode`, detector target/prediction
ingress, operator prediction ingress, synthesizer desired/input ingress, and
prompt-literal ingress all call the centralized validation operation.  Trusted
objects created and consumed wholly inside one validated call need not repeat
the digest computation per inner-loop access.

### Obsolete dense views and projectors

Remove public dense detector construction and tests that inject arbitrary
weight arrays.  Targets are created only through `for_target`.

Recursive dependency review found that the dense value projector is reachable
only through the retired pre-bottom-up H1/H2 source-editing chain.  Deleting
that chain also leaves six helpers production-orphaned or test-only: the old
factor motor, bounded List-map behavior, static whole-prompt factors, program
and edit-site observations, and structural successor.  The active prompt-span
substrate is the separate `public_train_prompt_semantic_bridge.py`; preserving
the old static prompt projector would contradict the roadmap.  The explicit
decision is to delete the complete executable closure, not compact-rewrite it
and not provide virtual-dense compatibility hashes.  Delete these source
modules:

- `public_typed_factor_value_race.py`;
- `public_typed_factor_motor.py`;
- `public_typed_list_map_behavior.py`;
- `public_typed_prompt_factors.py`;
- `public_typed_program_observation.py`;
- `public_typed_edit_site_observation.py`;
- `public_typed_structure_successor.py`;
- `public_typed_factor_h1_root.py`;
- `public_typed_factor_h1_prompt_hierarchical.py`;
- `public_typed_factor_h1_family_transfer.py`;
- `public_typed_factor_horizon2_int.py`;
- `public_typed_factor_horizon2_int_created_child_gate.py`;
- `public_typed_factor_horizon2_int_created_child_gate_v2.py`;
- `public_typed_factor_horizon2_int_created_child_gate_v3.py`;
- `public_factorized_source_editor.py`;
- `public_factorized_source_edit_canary.py`;
- `public_factorized_source_edit_canary_v2.py`;
- `public_factorized_source_edit_recovery_v1.py`;
- `public_factorized_simple_swe_train_bridge_v1.py`.

Delete their dedicated executable tests:

- `test_public_typed_factor_value_race.py`;
- `test_public_typed_factor_motor.py`;
- `test_public_typed_list_map_behavior.py`;
- `test_public_typed_prompt_factors.py`;
- `test_public_typed_program_observation.py`;
- `test_public_typed_edit_site_observation.py`;
- `test_public_typed_structure_successor.py`;
- `test_public_typed_structure_successor_speed.py`;
- `test_public_typed_factor_h1_root.py`;
- `test_public_typed_factor_h1_prompt_hierarchical.py`;
- `test_public_typed_factor_h1_family_transfer.py`;
- `test_public_typed_factor_horizon2_int.py`;
- `test_public_typed_factor_horizon2_list_bridge.py`;
- `test_public_typed_factor_horizon2_int_created_child_gate.py`;
- `test_public_typed_factor_horizon2_int_created_child_gate_v2.py`;
- `test_public_typed_factor_horizon2_int_created_child_gate_v3.py`;
- `test_public_factorized_source_editor.py`;
- `test_public_factorized_source_edit_canary.py`;
- `test_public_factorized_source_edit_canary_v2.py`;
- `test_public_factorized_source_edit_recovery_v1.py`;
- `test_public_factorized_simple_swe_train_bridge_v1.py`.

Historical JSON/spend artifacts, benchmark records, roadmap entries, and git
history remain immutable evidence; deleting obsolete executable paths does not
rewrite those results.  `test_public_typed_operator_successor_batch.py` remains
active and must construct the frozen operator directly instead of importing
the deleted H1-root builder.

### Test cutover

Rewrite direct representation assertions in:

- `test_public_typed_output_cells.py`;
- `test_public_typed_output_cells_batch.py`;
- `test_public_typed_string_output_cells.py`;
- `test_public_typed_operator_successor.py`;
- `test_public_typed_operator_successor_batch.py`;
- `test_public_typed_operator_successor_text_copy.py`;
- `test_public_typed_semantic_extrema.py`.

Replace dense offset, slice, writability, and `dataclasses.replace(cells=...)`
checks with factor-winner, canonical decode, digest, detector-decision, and
explicit audit-materialization checks.  Existing operator, semantic-vector,
String, frontier-work-bound, and compiler behavior tests remain the primary
interface tests.  Do not introduce a test-only production compatibility
property.

## Scientific identity preservation

The physical bank geometry and cell addresses do not change, so
`TypedOutputCellBank.projection_sha256()` must remain:

```text
f3ebca86516460b7a03ab5588cf26de2c6df07d0f5a109e8a6d5646eed015f44
```

Encoding SHA-256 remains the digest of the existing physical projection, not
of its storage optimization.  A private streaming serializer updates SHA-256
in this exact order without allocating a dense vector:

1. projection SHA as ASCII;
2. the output-type label as ASCII, or `UNKNOWN-TYPE`;
3. 3,284 native `float64` cells in existing global order, with `1.0` at each
   active global winner and `0.0` elsewhere;
4. 36 native NumPy booleans in factor order;
5. ASCII `1` for supported or `0` for unsupported.

The detector SHA stream is also frozen exactly.  It contains, in order:

1. projection SHA as ASCII;
2. target encoding SHA as ASCII;
3. the complete virtual dense `float64[3284]` weight vector in existing global
   order;
4. the unchanged native-float64 bias;
5. ASCII `1` when enabled or `0` when disabled.

An enabled target has `1.0` only at its active global winners.  A disabled
target has all-zero virtual weights, fixed bias `-1.0`, and always reports zero
matched factor cells.  The audit materializer and both virtual serializers
share one factor-address table so they cannot drift.

The migration must preserve, byte for byte where applicable:

- every encoding SHA and fixed-detector SHA;
- detector currents, spike masks, match counts, and audit records;
- operator checkpoint `92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e`;
- the direct-String-reverse ordered race digest
  `4ed8ab1c38ec4350a25314a0581c2bd5b5cdde2a1b8503a67b5154fe83a0dace`;
- all operator current/race ledgers and effect-memo keys;
- semantic-vector, tier, candidate, and completed-prefix race digests;
- prompt literal active global indices, semantic SHA, audit SHA, and literal
  channel SHA;
- selected programs, exact AST costs, sources, supported-candidate counts, and
  work-bound counters/STOP point.

The banked work-bound implementation created these semantic checkpoints, which
the compact cutover must preserve exactly:

- enabled String:
  `9ac87703dae72cf6efeee52b5ceaa953dffc072dc5303f3348d8a8b6f51486b6`;
- catalog-off String:
  `ed7dff6036ab522840bca1c0a1b6fd7c2e0de0768be813bb33b1920b5f291af3`;
- 25-literal sentinels:
  `13948fd329c79488f2e6532f3e013a10864db584db1c35b4219a1abe3fc9c477`;
- count:
  `4f0ee145047bccba72745a65afd9c261f7dd6d7a845ecc16c90fe104ff2e0ad6`;
- clamp-map-sum:
  `09c1e11ba835fd229b2741b0816de4704ea99ae46880ebbc5247e80f862979a2`.

The earlier banked String semantic checkpoints
`962c2359fb41d7444a24bcacdeb2679caa3d5406719e6a6c85887c14a174598e`
and `8931b4e106e500fa78c93001fe153c740660effa2934193e55668fc1cbedc612`
remain immutable historical identities rather than values to spoof after the
work-bound checkpoint version changes.

Source hashes and commit identities legitimately change.  Existing banked
artifacts must not be overwritten, re-signed, or presented as results from the
new source.

## Implementation order within the cutover

Use one branch and do not bank a dual dense/compact production mode:

1. capture pre-cutover golden encodings, detectors, prompt literals, operator
   races, String mechanisms, and work-bound sentinel records;
2. implement compact canonical construction, decoding, streaming hashes, and
   audit materialization in the output module;
3. replace the detector with factor-winner matching;
4. cut over every operator path, including String winner construction and
   local factor-copy drives;
5. cut over synthesizer decoding, semantic racing, and prompt-literal global
   addressing;
6. delete the exact retired H1/H2 execution closure, its recursively orphaned
   helpers, and the dedicated tests listed above; rewire the retained operator
   batch fixture directly onto the frozen operator;
7. delete `.cells`, `.active_factor_mask`, every dense bank-slice property,
   `_finish`, dense detector weights, and unused public geometry exports;
8. run the gates below and bank only the compact implementation.

Temporary parity code may compare the working tree with the pre-cutover commit,
but it must not survive as a compatibility module in production.

## Verification and acceptance gates

### Exact semantic parity

The audit projection, mask, decode result, supported state, and SHA must match
the dense baseline for:

- every Int from -64 through 64 plus invalid Int inputs;
- both Bool values and invalid Bool inputs;
- all 95 printable Char values plus invalid/non-printable Char inputs;
- `List[Int]` lengths 0--4, every position, invalid items, and over-capacity;
- String lengths 0, 1, and 26, every position, non-printable/Unicode
  characters, and over-capacity;
- the untyped UNKNOWN encoding.

Canonical detector parity covers exact, one-factor mismatch, wrong type,
additional predicted factors, UNKNOWN target/prediction, scalar and batch
calls, and the previously pinned detector/semantic audit hashes.  Its reachable
currents are checked as canonical semantics: an enabled exact target produces
`+5,000 pA`; an enabled non-exact row clips to `-5,000 pA`; a disabled target
produces `-5,000 pA`, zero matched factors, and no spike.

The four-current physical-kernel fixture is separate.  Drive
`(-5,000, 2,500, 4,375, 5,000) pA` directly through
`_drive_spike_mask` (or the owning `CompiledRestingGlifRace` kernel after a
mechanical move) and preserve scalar/batch spike parity
`(False, False, True, True)`.  Arbitrary detector weights and the old custom
detector constructor are deleted; canonical detector tests must not pretend
that `2,500` or `4,375 pA` are reachable exact-coincidence currents.

Adversarial canonicality tests forge objects only to prove every ingress rejects
them before physical work.  Cover wrong winner/digest byte lengths, a winner at
or beyond its factor width, illegal sentinel placement, missing required
factors, active positions beyond a List/String length, type/value shape
mismatch, a flipped digest bit, and a digest valid for different winner bytes.
Also pin ascending global-cell addresses and an all-zero local drive for every
inactive factor.

Operator parity covers Int boundaries, extrema, list copy, String copy,
`contains`, `filter`, `reverse`, invalid characters, and UNKNOWN propagation.
The four banked String fixtures retain their exact AST costs and telemetry:
63 predictions / 74 races for reverse, 364 / 894 for vowel filter, 412 / 996
for decimal filter, and 484 / 1,274 for reversed consonant filter.

The aggregate regressions retain direct sum at cost 2, predicate count at cost
6, explicit predicate-map structure at cost 8, and clamp-map-sum at cost 9.

### Work-bound parity

After the private bound is banked:

- Sentinel A remains the ordinary exhaustive STOP with 4,797 declared roots,
  4,757 supported roots, 87 root semantic classes, 23,642 predictions, 63,456
  races, normal race SHA-256
  `2ca6e201b448b8b74070c485fe72ec66502fc303fc66e8f3f33f05c61fa82ff2`,
  and the frozen complete tier rows;
- Sentinel B remains prediction-first
  `STOP_SEMANTIC_FRONTIER_WORK_BOUND` at exactly 30,000 completed predictions
  and attempted prediction 30,001, with 77,964 races, 1,651 declared roots,
  1,441 supported roots, 174 retained representatives, and completed-prefix
  SHA-256
  `506b110fac0558063293d08527c8c0ae6ad89849e5d97b24859d6bd82123970c`,
  before the interrupted cost-3 desired race;
- both exact race digests, retained states, the five checkpoints named above,
  and desired-independent reruns remain exact;
- both sentinels finish in under 60 seconds on this host.

No completeness claim survives beyond Sentinel B's STOP, exactly as specified
by the work-bound protocol.

### Benchmark conditions

Immediately after the work bound is banked and before compact edits begin,
record one dense-baseline manifest containing the git commit, Python executable
and version, NumPy version/build, CPU model, Windows power source/mode, process
priority, and exact logical-CPU affinity mask.  Dense and compact measurements
must use that same host, interpreter/dependency lock, normal process priority,
recorded power source/mode, and affinity mask, with no other project gate or
benchmark running concurrently.

Both capture commands require an explicit `--attest-exclusive-host` operator
attestation and hold one repository-keyed host mutex at
`experiments/r3_pilot/.compact_typed_output_benchmark.host.lock` for the full
capture.  The artifact records the attestation and mutex identity; the mutex
prevents another copy of this harness from running but does not replace the
operator's no-other-gate precondition.

Set `PYTHONHASHSEED=0` and set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to `1` before Python imports
NumPy.  Each fast detector/kernel fixture runs in one fresh process with two
unmeasured warmups followed by seven measured iterations; report the median of
those seven `perf_counter` timings.  Each self-posed synthesis fixture likewise
runs in one fresh process, but uses two unmeasured warmups followed by three
measured iterations and reports their median.  Each work-bound sentinel runs
once in its own fresh process.

Memory uses seven separate fresh processes per fixture and **no encoding or
detector object warmup**.  In each process, construct only the deterministic
raw Python inputs, run `gc.collect()`, start/reset the tracer and capture the
input-only process-memory baseline, then construct and retain the measured
objects once.  Report the median of the seven independent tracemalloc peak,
working-set delta, and private-usage delta samples.  This no-warmup rule is
intentional: allocator reuse from a discarded object graph would invalidate
both RSS and retained-object accounting.  Use identical fixed fixtures and
construction order throughout, and retain the manifest plus every raw timing
and memory sample with the engineering result.

### Memory gates

Use the pinned harness above for both the newly recorded dense baseline and the
compact result:

- one encoding owns no per-instance dense NumPy array;
- serialized winners are exactly 72 bytes and winners plus binary digest are
  exactly 104 bytes;
- 2,000 retained String encodings use less than 2 MiB and at least 20 times
  less measured memory than 50.98 MiB;
- 1,000 fixed detectors use less than 1 MiB and at least 20 times less measured
  memory than 25.38 MiB;
- a representative String-heavy frontier uses at most 20% of its recorded
  pre-cutover peak output-encoding memory.

### Runtime gates

- The 500-candidate by four-case compact detector grid must be exact and take
  no more than 0.010 s on this host, or no more than 25% above the immediately
  recorded post-work-bound batched baseline, whichever allowance is larger.
- The existing 64-row detector fixture must retain exact audit parity and may
  regress by no more than 25% from its immediately recorded batched baseline.
- Each self-posed aggregate and String synthesis fixture may regress by no
  more than 10% from the median of three pre-cutover post-work-bound runs.
- Audit materialization must have zero calls in all production runtime
  fixtures.

Wall time is an engineering gate, not capability evidence.  Exact decisions,
counters, physical currents, and hashes remain the scientific verdict.

## STOP conditions and fallback prohibition

Stop the migration before broad TRAIN if any of the following occurs:

- encoding, detector, operator, semantic, prompt-literal, or completed-prefix
  hash drift not explained by the already-banked work-bound checkpoint change;
- any retained bottom-up-path selected program, AST cost, candidate count,
  prediction count, race count, retained representative, STOP bound, or STOP
  location changes;
- local factor currents or fitted-GLIF spike masks differ;
- canonical UNKNOWN/List/String shape validation is weaker than the dense
  baseline;
- a production path calls audit materialization;
- a memory gate fails or a runtime regression exceeds its limit.

Failure does not license a dense fallback, compatibility property, task-
specific exception, looser work bound, silent top-k, or benchmark-dependent
branch, nor does it license restoring the deleted H1/H2 execution chain.  Fix
the compact seam or retain the pre-migration source and report a mechanism
STOP.

## Claim boundary

This document preregistered a representation and runtime optimization.  The
implementation and sole capture are now complete.  The cited measurements are
synthetic or self-posed engineering evidence.

The banked PASS establishes compact-storage and exact-physics parity only.  It
is not a broad six-family TRAIN result, prompt-only completion,
family-disjoint transfer, DEV/evaluation, M0, held-out, Path-2, C1, SWE-bench,
Gemma, or Opus capability result.  No canonical TRAIN desired vector or
property row, DEV material, M0/evaluation material, held-out material, or C1
material may be inspected or used to implement or tune this migration.
