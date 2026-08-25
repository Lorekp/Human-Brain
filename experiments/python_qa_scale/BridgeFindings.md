# BridgeFindings: Python-QA corpus → qa_experiments end-to-end — wiring verification, fixture/CUDA validation, and the price of the M0/M1 cell runs

Date: 2026-08-25 · Branch: `codex/swe-m01-harness` · Box: RTX 4070 SUPER (12 GiB,
compute 89), Windows, Python 3.11 venv (numpy-only stack; the CUDA path compiles
`_native_src/src/cuda.cu` via nvcc 12.6 — no torch anywhere).

Records (this directory): `fixture_n96_record.json` (N=96 CPU, full-corpus),
`smoke_n131072_cuda_record.json` (N=131072 CUDA, concept arm).

**Branch-base deviation, declared:** the task brief named `master@b0aa606`, but
master predates every artifact this work depends on (the frozen corpus, its
loader, the pinned baselines, `MILESTONES.md`, and the harness's
`--tokenizer/--device/--readout-rule` seams all live only on the
`codex/a013-cognition-design` line, which diverged from master at `771bd93`).
This branch is therefore cut from `codex/a013-cognition-design@884c987`. The
task CONTEXT describes exactly that tree.

## 1. Compatibility verdict and what was wired

The harness (`benchmarks/qa_experiments.py`) hard-imported the **Bible**
corpus module and had no corpus seam. The coding corpus module
(`src/humanbrain/python_qa_corpus.py`) shares `QATask` and the exact
`qa_prompt` framing with it — but could not be dropped in as-is, for three
reasons, each now handled:

1. **Live-mining drift (the blocker).** `python_qa_corpus.build_curriculum`
   mines the CURRENT `src/humanbrain` tree. The pinned artifact was built at
   head `1c9cdb6`; T-ROUTE work since then changed `recurrent_language.py`
   (e.g. added `complete_broadcast`), so live mining no longer reproduces the
   frozen corpus — visible TODAY as the pre-existing failure of
   `tests/test_python_qa_corpus.py::test_curriculum_is_seed_keyed_and_train_only`
   (73 passed / 1 failed before any change on this branch; identical after).
   Rebuilding the artifact would silently invalidate every locked baseline
   digest, so the adapter does the opposite: it LOADS the frozen artifact.
2. **Cohort semantics.** The harness evaluates a handle's
   `calibration_tasks`; `python_qa_corpus.build_curriculum` carves a 6-item
   seed-keyed CALIBRATION slice out of TRAIN, while the locked n-gram
   comparators scored ALL 25 VAL items. The adapter serves train = all 75
   TRAIN items, calibration = all 25 VAL items, cohort labels preserved.
3. **No selection seam.** Added `--corpus {bible,python_qa}` (default
   `bible` ⇒ historical records unchanged), threaded through `run()` via
   per-arm handles; new top-level record key `corpus_arm`.

New files:

- `src/humanbrain/python_qa_adapter.py` — frozen-artifact handle. Verifies
  artifact bytes against `manifest.json`'s `artifact_sha256`, recomputes the
  canonical entries digest, and REFUSES to run unless it equals the pin
  `789e3ac592fd8aa1d4f68af8704a4a9b19d64db7277efea59ebc50ff8dd446ec` — the
  same `corpus_sha256` embedded in all three locked baseline records
  (`experiments/python_qa_baselines_v1/*.json`). Re-uses
  `humanbrain.qa_corpus.qa_prompt/growth_text` verbatim so framing stays
  byte-identical across arms.
- `tests/test_python_qa_adapter.py` — 7 tests: pin integrity vs manifest AND
  vs each frozen baseline record; full-cohort curriculum (75/25, dense
  ordinals, unique ids, artifact-identical pairs); seed keys domains/digest
  but never cohorts; prompt-framing parity with the Bible arm; guard rails
  against partial slices / bad seeds; one small end-to-end `run()` through
  the `python_qa` arm; unknown-arm rejection.

Suite state: QA-surface tests 73 passed + 7 new = 80 green; the single
pre-existing drift failure is untouched (it tests the live miner, which this
bridge deliberately bypasses; fixing it means re-pinning the corpus and is
the corpus owner's call).

## 2. Fixture-scale validation — N=96 CPU, full corpus

```
python benchmarks/qa_experiments.py --corpus python_qa --device cpu \
    --cells 96 --seed 17        # 38.5 s wall
```

| Check | Result |
|---|---|
| Grows on / evaluates | 75 TRAIN / 25 VAL (full frozen cohorts) |
| `corpus_sha256` in record | `789e3ac…dd446ec` == baselines' pin |
| Backend | cpu / cpu, `ridge` readout, vocab 567 |
| answer_span accuracy | **0.12808** (78/609 predictions) |
| free-run EM | 0/25 |
| Degeneracy guard | PASS (15 distinct completions, modal share 0.44) |

**Schema parity (programmatic):** `metrics` keys and inner keys are
key-for-key identical to `experiments/python_qa_baselines_v1/trigram_smoothed.json`
— `answer_span{tasks,predictions,correct,accuracy}`,
`free_run{tasks,exact_matches,exact_match_rate}`,
`degeneracy{evaluated,distinct_answers,modal_answer,modal_share,guard,guard_pass}`.
One semantic caveat the comparison surfaced: prediction COUNTS are not
comparable (609 byte-level vs 104 word-level predictions over the same 25
tasks) — compare accuracies, never raw counts.

Ungated single-seed observation (NOT a milestone claim): fixture-scale span
accuracy 12.8% already exceeds the locked trigram comparator's 5.77%. The
gated version needs the preregistered k=3 CUDA path below.

## 3. CUDA smoke — N=131072, sparse_three_factor, concept tokenizer

```
python benchmarks/qa_experiments.py --corpus python_qa --device cuda \
    --cells 131072 --readout-rule sparse_three_factor --tokenizer concept \
    --concept-min-frequency 2 --concept-min-mi-bits 1.0 --seed 17   # 75 s
```

| Check | Result |
|---|---|
| Kernel backend | `cuda` — NVIDIA GeForce RTX 4070 SUPER (compute 89) |
| Full touch | 131072 / 131072 active cells, `sparse_three_factor` |
| Concept band [1024,1536) | **383 concepts mined** (lowered gates); base 567 + concepts ⇒ vocab 1407 = token_cells, alignment PASS |
| Metrics | span acc 0.0, free-run EM 0/25 — expected: only **4,604 growth token-steps** reached a 131k-cell reservoir (bible smoke saw identical zeros at 554 steps); degeneracy guard PASSES (18 distinct, modal share 0.16) |
| Grow throughput | 4,604 steps / 42.7 s = **107.9 tok/s** |
| Eval throughput | 5,423 steps / 32.4 s = **167.6 tok/s** |

Concept-gate calibration note: the study-default gates (freq ≥ 50, MI ≥ 5)
mine **48** concepts on this fixture-size corpus; the lowered gates mine 383
of the ~512 reserved slots. Both execute cleanly; pick per experiment
registration. ("500 tokens cap" in the task brief = this reserved band.)

## 4. Pricing the full M0/M1 runs (raw-Python-text channel)

Locked comparators (BEFORE any cell run, `experiments/python_qa_baselines_v1/`):
unigram span-acc **0.0**, smoothed trigram **5.769%**, majority floor
(guard-fails by construction). Gate framing: **M0** = cell span-acc beats the
random/unigram floor with CI separation (degeneracy guard passing is part of
the claim); **M1** = beats trigram 5.769%; both k=3 seeds under the shared
Wilson-CI decision rule in MILESTONES.md.

Token accounting (measured, device-independent step counts):

| Stream | Steps | Source |
|---|---|---|
| Growth, byte arm, 75 texts × 2 epochs | 11,118 (≈5,559/epoch) | fixture record |
| Growth, concept arm (lowered gates) | 4,604 (≈2,302/epoch) | smoke record |
| Eval pass, 25 tasks incl. slack 8 | 7,479 (byte) / 5,423 (concept) | records |

Wall-clock per k=3-seed campaign, from MEASURED rates (107.9 grow / 167.6
eval tok/s @131k; near-linear-in-N scaling anchored by the bible line's
157.7 tok/s @128k vs 22.42 @1M; byte-arm @131k projected from step counts):

| Reservoir N | Concept arm, k=3 | Byte arm, k=3 | VRAM (process Δ) |
|---|---|---|---|
| **131,072 (pilot)** | **≈ 4 min** (75 s/seed, measured) | ≈ 8 min | tens of MiB |
| 262,144 | ≈ 8 min | ≈ 16 min | < 0.1 GiB |
| 524,288 | ≈ 17 min | ≈ 33 min | < 0.2 GiB |
| 1,048,576 (22.4/21.9 tok/s anchor) | ≈ 23 min | ≈ 42 min | ≈ 0.33 GiB (lm_million_smoke) |

The structural difference from the Bible channel: its 1.16M-token TRAIN
stream made wall-clock the binding constraint (~15 h/epoch @1M). This
channel's growth stream is ~5.6k byte-tokens/epoch, so **compute is not the
binding constraint at ANY reservoir scale — the corpus size (100 triples) is.**
Every M0/M1 campaign above fits inside a single GPU session with the <8 GiB
constraint holding by orders of magnitude.

### Recommended pilot (concrete)

N=131072 CUDA full-touch, concept arm, lowered mining gates, k=3 seeds
(17/18/19), 2 epochs, full 25-task VAL pass, degeneracy gate enforced:

```
python benchmarks/qa_experiments.py --corpus python_qa --device cuda \
    --cells 131072 --readout-rule sparse_three_factor --tokenizer concept \
    --concept-min-frequency 2 --concept-min-mi-bits 1.0 \
    --growth-epochs 2 --seed 17 --require-pass     # repeat: --seed 18, 19
```

Total cost ≈ **4–5 minutes of exclusive GPU**. Before unsealing, run the
determinism probe ONCE on a short slice only (checking doubles wall-clock):

```
python benchmarks/qa_experiments.py --corpus python_qa --device cuda \
    --cells 131072 --readout-rule sparse_three_factor --tokenizer concept \
    --concept-min-frequency 2 --concept-min-mi-bits 1.0 \
    --max-train-tasks 8 --eval-limit 4 --check-determinism
```

If the pilot clears M1's bar, the same commands scale to 262k/524k/1M for
capacity curves at the prices tabulated above. A byte-arm companion run
(same seeds, drop the concept flags) is ≈2× pilot cost and buys the
tokenizer-attribution row the a013 studies say matters.

## 5. Caveats carried forward

1. `test_python_qa_corpus.py::test_curriculum_is_seed_keyed_and_train_only`
   fails on this branch independent of this work (live miner vs frozen pin).
   Do NOT "fix" it by rebuilding the artifact — that invalidates the locked
   baseline digests this whole bridge exists to protect.
2. Prediction counts differ across tokenizer conventions (byte vs word);
   compare `accuracy`, not `predictions`.
3. Concept-arm metrics read 0.0 at 4.6k training steps into a 131k reservoir;
   that is a budget fact, not a path failure — the fixture-scale N=96 run
   proves the readout learns on this corpus when steps ≫ cells-ratio allows.
4. `HUMANBRAIN_DEVICE` is bound by `--device` before circuit construction;
   external drivers of `run()` must set it themselves first.
