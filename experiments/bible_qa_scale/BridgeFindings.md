# BridgeFindings: qa_experiments → CUDA at 128k+ — verification, plumbing, and the price of the first real Bible-QA run

Date: 2026-08-24 · Branch: `codex/a013-cognition-design` · Box: RTX 4070 SUPER (12 GiB, compute 89), Windows, Python 3.11 venv
Records: `_qa_cuda_128k_smoke.json` (determinism check), `_qa_cuda_128k_run.json` (full single-run record) in this directory.

## 1. Verification: RecurrentCellLanguageModel works on CUDA at N ≥ 128k

**PASS.** The CUDA full-touch contract that `benchmarks/lm_million_smoke.py` drives on the bare
circuit carries through the language-model wrapper unchanged:

- `RecurrentCellLanguageModel.grow()` constructs `RecurrentCellCircuit`, which selects the backend
  via `should_use_cuda()` **at construction time** from `HUMANBRAIN_DEVICE` (`src/humanbrain/cuda.py`);
- with `HUMANBRAIN_DEVICE=cuda`, full-touch (`active_reservoir_cells == reservoir_cells`) and
  `readout_learning_rule="sparse_three_factor"`, grow → freeze → teacher-forced race/observe streaming →
  `complete()` all execute on `backend="cuda"` at N=131072;
- inference is bit-exact across replay (million-smoke evidence) and the qa path's own
  `--check-determinism` passes at 128k (below).

The one non-negotiable configuration point: the default dense-`ridge` readout allocates an F×F float64
normal matrix (F = active cells + 1). At N=131072 that projection is ≈137 GB and cannot exist on any
single box — `sparse_three_factor` + full touch is required, exactly as in lm_million_smoke.

## 2. Plumbing added to benchmarks/qa_experiments.py (fd84598 lineage)

Mirrors `ladder_experiments.py`'s device seam:

- `--device {auto,cpu,cuda}` — explicit flag wins over `HUMANBRAIN_DEVICE`; effective device is bound
  into the environment inside `run()` **before any circuit exists**, then asserted post-grow
  (`--device cuda` fails loudly if `simulation_backend != "cuda"`).
- `--cells N` (alias `--reservoir-cells`, historical default 96 preserved).
- `--readout-rule {auto,ridge,three_factor,sparse_three_factor}` — `auto` keeps dense `ridge`
  while its normal-matrix projection fits a 2 GiB host budget (N ≤ ~16k ⇒ historical records
  unchanged) and upgrades to `sparse_three_factor` beyond.
- New deterministic record blocks: `execution` (requested/effective device, kernel backend,
  simulation device, readout rule, cell counts) and `throughput.growth/.eval`
  (`token_steps` + `wall_s`; wall keys named so `_strip_wall` still removes them).

Regression evidence: default config (N=96, auto→cpu→ridge) reproduces historical behavior and
`--check-determinism` passes on CPU; record schema additions are additive.

## 3. Smoke test at N=131072 — end-to-end PASS

Command:
```
python benchmarks/qa_experiments.py --device cuda --cells 131072 --growth-epochs 1 \
    --max-train-tasks 5 --eval-limit 4 --check-determinism
```
(5 TRAIN tasks of the KJV verse-context fixture = ~500 corpus tokens; seed 17.)

| Measurement | Value |
|---|---|
| Determinism (two runs, canonical JSON minus wall-clock) | **PASS** — sha256 `cf87f6c1…4aba3bf` both runs |
| Backend / device | `cuda` / RTX 4070 SUPER (compute 89) |
| Readout rule resolved by `auto` | `sparse_three_factor`, full touch (131072/131072) |
| Vocabulary ↔ token cells | 299 = 299, alignment pass |
| Grow throughput | 554 token steps / 3.513 s = **157.7 tok/s** (independent probe: ~100 tok/s) |
| Eval throughput (span + free-run races) | 1214 token steps / 4.381 s = **277 tok/s** |
| Metrics | span acc 0.0, free-run EM 0.0 — expected at 554 training tokens; degeneracy guard PASSES (3 distinct completions, modal share 0.5); free-run shows the untrained `"life"` repeat attractor, which the degeneracy guards are designed to make visible |

VRAM: baseline desktop usage ~2.8 GiB; the 128k circuit's device footprint is tens of MiB
(1M cells measured at only ~330 MiB delta + 154.5 MiB persistent arrays in lm_million_smoke),
so the <8 GiB/process constraint holds with orders of magnitude of headroom at every scale here.

Scaling anchor consistency: 157.7 tok/s @128k vs 22.42 tok/s @1M (lm_million_smoke) is a 7.03×
slowdown for 7.63× more cells — per-token cost is near-linear in N on this kernel, which makes the
interpolations below trustworthy to first order.

## 4. Pricing the first real Bible-QA run

Corpus ground truth (FREEZE-CANDIDATE-v3-BIBLE.json): trimmed KJV 4,436,168 bytes; 49/17 book split;
TRAIN stream **3,656,777 bytes ≈ 1,162,238 tokens** (BPE-4096 proxy, stated upper bound; whitespace
pre-token count 676,810); VAL stream 239,446 proxy tokens; 200 train / 60 val QA triples.

Token-count caveat that must ride along: BytePhraseTokenizer under qa_experiments' small phrase
budgets yields ~1.78 chars/token (fixture measurement at max_phrases=512), i.e. up to
~2.06M tokens over the TRAIN stream — roughly 1.8× the BPE-4096 proxy. A real run should pin
`--max-phrases` explicitly; the table prices both ends of that band.

### Full 1M-cell KJV run (grow over the whole TRAIN split)

| Component | Rate (measured @1M) | Tokens | Cost |
|---|---|---|---|
| One grow epoch | 22.42 tok/s | 1.16M (proxy) | **14.4 h** (~15 h with eval overhead; parent anchor confirmed) |
| One grow epoch, worst-case tokenizer | 22.42 tok/s | 2.06M | 25.5 h |
| One VAL evaluation pass | 21.93 tok/s | 239k | 3.0 h |
| Multi-epoch campaign (3 epochs + 3 VAL passes) | | | **~2 days GPU-exclusive** |

VRAM is not the binding constraint (≈0.33 GiB/process at 1M vs the 8 GiB cap); **wall-clock exclusivity
is** — a 1M-cell run is a multi-day dedicated-box commitment before the qa pipeline has shown it can
move metrics at all.

### Recommended pilot (concrete)

**N = 131072, CUDA full-touch, sparse_three_factor — the exact configuration verified above — grown
over the full TRAIN QA growth-text stream for 2 epochs, then one VAL pass over all 60 held-out tasks.**

```
python benchmarks/qa_experiments.py --device cuda --cells 131072 \
    --max-phrases 4096 --min-phrase-frequency 2 --growth-epochs 2 \
    --eval-limit 60 --require-pass
```

Priced from measured rates (conservative band uses the slower probe rate):

| Phase | Optimistic (157.7 grow / 277 eval tok/s) | Conservative (100 grow / 180 eval tok/s) |
|---|---|---|
| Grow 2 epochs × 1.16M tokens | 4.1 h | 6.5 h |
| Grow 2 epochs × 2.06M tokens (worst-case tokenizer) | 7.3 h | 11.4 h |
| VAL pass, 239k tokens | 14 min | 22 min |
| **Total pilot** | **≈ 4.5 h** | **≈ 12 h** |

Rationale: it is the largest scale whose execution path is already proven deterministic on this box,
it costs an overnight slot rather than a multi-day exclusive hold, it produces the first real
signal-vs-baseline comparison (pinned floors: unigram 4.54%, trigram 16.28% span-acc), and its
per-epoch throughput doubles as a third scaling anchor between 128k and 1M. Scale to 1M cells only
after the pilot demonstrates the readout learns at fixed budget — and run the expensive campaign
with `--check-determinism` reserved for short slices (as done here), never for the full stream,
since determinism-checking doubles wall-clock.

### Known caveats for the first real run

1. `learned_synapses` bookkeeping differs under `sparse_three_factor` (structural contacts, not dense
   synapse counts) — compare against lm_million_smoke-style sparsity accounting, not the ridge numbers.
2. The untrained free-run attractor (`"life"` loops) means exact-match will read zero until the readout
   learns; the degeneracy guards (distinct ≥ 3, modal share ≤ 0.9) are the honest early-signal channel.
3. `HUMANBRAIN_DEVICE=cuda` must be bound before circuit construction — the CLI does this itself via
   `--device`; external drivers must set the env var before calling `run()`.
