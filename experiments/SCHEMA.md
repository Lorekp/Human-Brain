# Ladder-experiment metrics JSON schema (v1)

Fixed output contract of `benchmarks/ladder_experiments.py`. Every run writes
exactly one file `experiments/<experiment_id>/<arm>.json` containing exactly
the fields below, serialized as canonical JSON (`sort_keys=True`,
`indent=2`, `allow_nan=False`, trailing newline). No other top-level keys may
be added without bumping this schema version.

## Invocation

```
python benchmarks/ladder_experiments.py \
    --experiment <experiment_id> --arm <arm> --N <cells> --seed <S> \
    [--out DIR] \
    [--readout-rule {ridge,three_factor,sparse_three_factor}] \
    [--schedule {massed,spaced,interleaved}] \
    [--data-scale {21,42,84}] \
    [--supervision {all_transition,program_only}] \
    [--codebook {explicit,random,similarity}] \
    [--assembly-k K] \
    [--check-determinism]
```

Defaults: `--readout-rule sparse_three_factor`, `--schedule spaced`,
`--data-scale 21`, `--supervision program_only`, `--codebook explicit`,
`--assembly-k 16`, `--out experiments/`.

## Codebook knob (a013 representation arms)

The codebook is the mapping from token ids to injected reservoir-cell
populations (docs/research/a013-representation-choice.md). It is installed
once before growth and used by BOTH grow and completion, so one token always
drives one fixed cell assembly. N_E equals the reservoir size (v1 single
population):

- `explicit` — status quo; the circuit's built-in per-token random 16-of-N
  afferent subsets are left untouched. Requires the default `--assembly-k`.
- `random` — distributed control arm: every Layer-1 token gets a uniform
  k-of-N_E assembly drawn without replacement from a domain-separated
  per-token stream (`default_rng([seed, salt, token_id])`, salt `0xC0DEB005`),
  so explicit/random arms sharing a seed keep identical recurrent topology.
- `similarity` — deterministic frozen character-trigram feature-hashing
  projection onto the top-k cells (stdlib `blake2b`; exact pinned definition
  documented in `_similarity_assembly_cells` of the harness), so structurally
  similar tokens share cells by construction. Seed-independent.

`config_dump` additionally records `codebook` (mode), `assembly_k` (`null`
for explicit), and `codebook_sha256` (provenance digest over mode/n_e/k/seed/
full assembly table; identical across seeds for explicit/similarity).

## What a run does

1. **Curriculum** — a fresh per-seed cohort is materialized through
   `humanbrain.typed_composition._task_specs` / `_materialize_tasks` (task
   generation is reused, never reimplemented): 21 seed-derived literals
   disjoint from the spent a012 literal set yield an 86-task TRAIN pool whose
   canonical-order prefix of length `data_scale` is trained on; a fixed
   18-task CALIBRATION cohort (first 4 pool literals, opposite template
   parity, distinct opaque identifiers/order domains) is held out for
   generalization. Prompt/signature templates and identity prompts are lifted
   verbatim from the frozen public a012 protocol JSON (read-only provenance;
   that protocol is never executed).
2. **Growth** — one `RecurrentCellCircuit` with `token_cells=277` (the fixed
   Layer-1 vocabulary: 256 byte cells + 21 primitive cells) and
   `reservoir_cells=N` is grown over exactly `data_scale x 3` task
   presentations. Each presentation resets sequence state and replays
   `prompt+signature framing` tokens followed by teacher program tokens,
   calling `observe` then `strengthen` under the supervision knob — the exact
   supervision contract of `typed_program.grow`. The schedule knob fixes the
   presentation ORDER only (equal presentation counts in every arm):
   - `massed`: each task's 3 repeats contiguous `[t0,t0,t0,t1,t1,t1,...]`;
   - `spaced`: round-robin passes `[t0..tk, t0..tk, t0..tk]` so k-1 distinct
     tasks separate consecutive repeats of a task (eligibility traces are
     decayed/cleared between repeats);
   - `interleaved`: seeded fresh permutation of the corpus inside each pass.
   After growth the circuit's learning is frozen (`freeze_learning`, which
   consolidates the ridge readout).
3. **Completion** — every train and calibration task gets one typed-hole
   completion: replay the input framing, then repeatedly race the allowed
   primitive output cells (`_TypedHoleState` masking) until the program
   closes, feeding each winner back through `observe`. Emitted programs are
   checked for exact teacher-token label replay and evaluated on their public
   semantic cases by structural Python compilation (same evaluation semantics
   as the sealed runner).

## Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `experiment_id` | string | Echoed `--experiment`; also the output subdirectory. |
| `arm` | string | Echoed `--arm`; also the output filename stem (`<arm>.json`). |
| `N` | int | Reservoir cell count (`RecurrentCircuitConfig.reservoir_cells`). Output/token cells stay fixed at 277. Hard cap 4096 (python circuit limit / 2 GiB budget); sparse rule additionally needs N >= 64 (readout fan-in). |
| `seed` | int | Seed for the circuit RNG, curriculum derivation, and interleaved schedule. |
| `readout_rule` | string | One of `ridge`, `three_factor`, `sparse_three_factor`. |
| `schedule` | string | One of `massed`, `spaced`, `interleaved` (presentation order; see above). |
| `data_scale` | int | Number of TRAIN tasks trained on (nested prefixes: 21 ⊂ 42 ⊂ 84 of one per-seed pool). |
| `supervision` | string | `all_transition` (strengthen after every observed token) or `program_only` (strengthen only on program-token decisions), mapping to the A0/A1 vs A2 lane scopes. |
| `curriculum_sha256` | string | SHA-256 over the canonical JSON of the full generated cohort (domains, literal pool, every task id/prompt/signature/teacher tokens/cases, both cohorts). Identical for identical seeds regardless of other knobs. |
| `train_replay_tasks` | int | TRAIN tasks whose completion reproduced its teacher token sequence exactly (label replay, capacity family). |
| `train_cases_total` | int | Semantic cases across all TRAIN tasks (= 3 x `data_scale`). |
| `train_cases_passed` | int | TRAIN cases where the compiled emitted program returns the expected value. |
| `calib_tasks` | int | Calibration task count (18 with default curriculum parameters). |
| `calib_cases_total` | int | Semantic cases across calibration tasks (54 by construction). |
| `calib_cases_passed` | int | Calibration cases passed (generalization family). |
| `margin_count` | int | Competitive margins collected during TRAIN completions at decision positions where the teacher's next token was type-allowed: margin = I(expected cell) − max(I(other allowed cells)) in pA. |
| `margin_min_pa` | float | Minimum collected margin in pA (0.0 if none). |
| `margin_median_pa` | float | Median collected margin in pA (0.0 if none). Floats rounded to 9 decimals. |
| `wall_s` | float | Wall-clock seconds for the whole arm (rounded to 3 decimals). EXCLUDED from the determinism guarantee. |
| `git_head` | string | `git rev-parse HEAD` of the repository at run time (`unknown` if unavailable). |
| `config_dump` | object | Full effective configuration: knobs (including the codebook arm, assembly size, and codebook provenance digest), corpus passes (3), framing order, effective input fan-out, token-cell count, cohort counts, the complete `RecurrentCircuitConfig` asdict, and the frozen protocol SHA-256 used as curriculum source. |

## Determinism contract

Two invocations with byte-identical arguments produce byte-identical JSON
except `wall_s`. `--check-determinism` executes the arm twice in-process and
refuses to write anything if the assertion fails.

## Constraints

- `HUMANBRAIN_DEVICE=cpu` only; any other value aborts before import.
- Single-process RAM <= 2 GiB: projected peak arrays are computed per rule and
  refused above a 1.5 GiB budget; `N > 4096` is refused outright.
- Existing metrics files are never overwritten (fail-closed).
- No frozen-protocol execution: no `--execute` flag exists here, nothing is
  claimed or spent, and `artifacts/` is never touched. Run outputs under
  `experiments/` are working measurements, not sealed evidence.
