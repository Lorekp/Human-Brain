# BPTT de-risk — Stage A findings ($0 downloads + evaluations)

**Date:** 2026-08-25 · **Branch:** `codex/path3-repetition` (worktree
`human-brain-path3-probe`) · **Scope:** Stage A of
[docs/research/bptt-derisk-plan.md](../../../docs/research/bptt-derisk-plan.md)
@ f6ae6ed — download the three public artifacts free, verify integrity, resolve
the NeuronSpark token-count provenance flag, and probe both model families on
our python_qa corpus shapes (25-item VAL cohort). Pure evaluation; no training,
no spend. All downloads and runs executed on the local RTX 4070S 12GB box.

## 1. Downloads & integrity — all three artifacts fetched free, byte-exact

Hub metadata pinned first via `?blobs=true` API (`hf_api_meta.json`); every
downloaded file then checked against Hub-reported sizes; weight files hashed
(`download_manifest.json`). No auth was required anywhere.

| Repo | HF commit | Weight file | Size (bytes, exact) | sha256 (first 16) | Verified |
| --- | --- | --- | --- | --- | --- |
| `ridger/SpikeGPT-OpenWebText-216M` | `4039295c` | `SpikeGPT-216M.pth` | 861,716,013 | `024d2dab5f71b6bd` | ✅ size match |
| `ridger/SpikeGPT-BookCorpus` | `28d52794` | `BookCorpus-SpikeGPT.pth` | 737,553,365 | `01496fea44d1ddce` | ✅ size match |
| `Brain2nd/NeuronSpark-0.9B` | `c91be7a8` | `model.safetensors` (+21 code/config files) | 3,496,634,368 | `85a3b5312303134f` | ✅ all 22 files |

Checkpoint internals (`peek_checkpoints.py`):
- OpenWebText-216M.pth: emb `(50277, 768)` → GPT-NeoX-20B BPE vocab 50277,
  **18 blocks**, **215,399,424 params** fp32 → matches card ("L18-D768").
- BookCorpus-SpikeGPT.pth: emb `(**77**, 768)` → **char-level, 77 symbols**,
  **24 blocks**, **184,349,184 params** fp32 → **contradicts its own card**
  (see §3).
- NeuronSpark loads at **874,142,120 params** fp32 (README says 874M ✓).

## 2. Provenance flag RESOLVED (against the paper claim): embedded counter says ~204M tokens, not 1.4B

Verbatim from the downloaded `config.json` (machine-written by the training loop):

```json
"_training_step": 85000,
"_tokens_seen": 203910528
```

Verbatim from the repo card (`README.md`):
- "Training Tokens | ~1.4B (of ~10B available)"
- "trained on only ~85K steps with a small fraction of the full Seq-Monkey 10B-token corpus"
- shipped training recipe: `--batch_size 8 --accumulation_steps 8`, ctx 512
- "Precision | bfloat16"

Arithmetic on the three mutually inconsistent readings:

| Source | Tokens over 85K steps | Implied tok/step |
| --- | --- | --- |
| config `_tokens_seen` (embedded counter) | 203,910,528 | **2,398.9** |
| README "~1.4B" | 1,400,000,000 | 16,470.6 |
| Shipped recipe fully packed (8×8×512 = 32,768/step) | 2,785,280,000 | 32,768 |

Notes:
- `_tokens_seen = 203,910,528` factors **exactly** as 6,222 × 32,768 — i.e. the
  counter equals 6,222 *full* optimizer steps' worth of tokens while claiming
  85,000 steps (ratio ≈ 13.66). Consistent with stale/partially-incremented
  accounting (e.g. counter reset at a resume, or per-micro-batch double-count
  bug), not explainable by padding alone.
- Additional card-vs-artifact contradiction: README claims bfloat16, but
  `config.json` says `"torch_dtype": "float32"` and the safetensors payload is
  fp32 (3.496 GB ≈ 874M × 4 bytes). The released inference artifact is fp32;
  the shipped inference path is also fp32-only (see §5).

**Verdict:** the flag resolves **against** the paper/card claim. The embedded
training-loop counter (~204M tokens) undercounts the README figure by ~6.9×;
even the README's own hyperparameters imply ~2.79B if steps were fully packed.
No reading reproduces "~1.4B". Data-scale and throughput claims derived from
"1.4B tokens" should be weighted accordingly in DD-SGS; the architecture-code
claims remain checkable independently (code ships in-repo, Apache-2.0).

## 3. BookCorpus checkpoint: integrity FAILURE — unusable artifact

- Card says L18-D768 ctx-1024 BPE; the actual tensor payload is **L24/D768,
  char-level vocab of 77, 184.35M params**. The artifact does not match its
  documentation.
- Its index→char map (`vocab.json`, written at train time by `src/utils.py`)
  is **not included** in the HF repo, so the 77 symbols cannot be read off.
- Recovery attempt (`eval_spikegpt.py` §BookCorpus): tested three candidate
  sorted-by-codepoint 77-char sets (canonical cleaned-book punctuation sets).
  Mean NLL/char on held-out English prose: **A 6.9435, B 6.9477, C 6.9435** —
  all *worse than uniform* (ln 77 = 4.344), and two different mappings give
  numerically identical NLL, which happens when predictions barely depend on
  input identity.
- Responsiveness probe (`bookcorpus_probe.py`): prediction agreement between
  two unrelated input streams is 32.2% (so not literally input-independent),
  but outputs collapse onto a handful of ids (id 8 dominates every rollout);
  max softmax prob 0.129; entropy 3.20 nats vs uniform 4.34.

**Verdict:** mislabeled, key-metadata-missing, and functionally degenerate as
far as we can measure without its vocabulary. Treat as an uncharacterized
curio; exclude it from F-EVAL-1 gates. The plan's "no 46M/enwik8 checkpoint
exists" conclusion is unaffected; note additionally that one of the two
"released 216M-class checkpoints" fails basic characterization.

## 4. python_qa VAL cohort probes (25 items, BY-FILE split)

Protocol mirrors `tools/python_qa_baselines.py`: identical prompt framing
(`qa_prompt`, prompt-last, ends `answer:`), same 25 VAL triples, same guards.
Neural models score **token-level** teacher-forced span accuracy over the
tokenized answer span (fast-tokenizer offset mapping locates the span; naive
prefix matching aligned 0/25 because BPE merges cross the prompt/answer
boundary). Free-run: greedy completion, exact-match after normalization,
budget = ⌈(words(ref)+8)×1.5⌉ model tokens. Comparability caveat applies to
every number below: baselines predict whole words (104 predictions); neural
span scores are per-BPE-token (693 and 188 predictions respectively) and the
task decomposes into easier sub-pieces at that granularity. Directional and
cross-model comparisons are valid; direct numeric comparison to the 5.77%
word-level bar is not.

| Model | Span metric | Free-run EM | Degeneracy guard | Notes |
| --- | --- | --- | --- | --- |
| trigram_smoothed (repo baseline) | **5.77%** word-level (6/104) | 0/25 | PASS | reference bar |
| unigram_smoothed | 0.0% word-level | 0/25 | FAIL-ish record | floor context |
| **SpikeGPT OpenWebText-216M** | **18.09%** tok-level (34/188) | 0/25 | PASS (24 distinct, modal 0.08) | mean NLL 7.523/tok on prompt+answer |
| **NeuronSpark-0.9B** | **18.33%** tok-level (127/693) | 0/25 | PASS (20 distinct, modal 0.08) | 0.0% unk rate on our text |

### What the probes actually say

- **Harness sanity (SpikeGPT):** mean NLL 4.096/tok (ppl ≈ 60.1) on generic
  English prose with the paper-matched checkpoint — right neighborhood for a
  216M model (paper WT-103 test ppl 39.75). The pure-torch reimplementation
  produces fluent greedy continuations ("The city of Paris is famous for its
  → vibrant urban neighborhoods. The city is home to…"), so the WKV/LIF port
  is sound before any corpus claim rests on it.
- **Tokenizer reach:** NeuronSpark's 6,144-vocab Chinese BPE covers our
  prompts with **zero unk tokens**, but fragments English at **~7.75
  tokens/word** — the model sees our task as long streams of character-ish
  pieces. Its Chinese control task remains fluent (quickstart-style prompt →
  coherent zh paragraph), so weak English behavior is data exposure, not
  breakage.
- **Free-run quality is poor for both** and this is the decision-relevant
  part: neither model can *produce* an answer in our format.
  - SpikeGPT-216M: locally plausible code-flavored English, then repetition
    collapse ("`\n\n#\n\n#\n\n#`", "`Error:\n\nError:`", "
    `This is a function of the function that we call the function function.`").
    Greedy samples reproduced identically across two full runs
    (repeatability check).
  - NeuronSpark-0.9B: short degenerate fragments ("` tai the the the the t`",
    "` yuanguang.com\nmailemamamamam`") despite fluent Chinese — consistent
    with the card's own admission (85K steps, tiny subset) and with the ~204M
    embedded token count (§2), which makes the undertraining look even more
    severe than the README suggests.
- **Teacher-forced signal is real but modest for both** (~18% tok-level each,
  ≥3× the trigram bar's word-level analogue with the granularity caveat
  above). Neither model family demonstrates our task *shapes* are learnable
  by them end-to-end for free; what they demonstrate is (a) our harness works
  on third-party SNN artifacts and (b) released SNN LMs do carry some
  next-token signal on English code QA conditioning.

## 5. Harness engineering notes (for Stage B/C reuse)

- **SpikeGPT runs without the archived repo's CUDA build:** 
  `spikegpt_pure.py` reimplements `src/model.py` forward exactly (Heaviside
  embedding encoder via ATan-surrogate spiking=True; time-shift mixes; verbatim
  `wkv_cuda.cu` recurrence p/q/o; MultiStep-LIF τ=2 hard-reset-to-0) as a
  single-token RNN step machine. Validated against prose NLL + fluent
  generation (§4). Loads both `.pth`s strict (time_mix params reshaped from
  (1,1,C)).
- **transformers version fix (operator-directed):** NeuronSpark remote code
  predates transformers 5.x `_finalize_model_loading`; loading under 5.9.0
  raises `'NeuronSparkForCausalLM' object has no attribute
  'all_tied_weights_keys'`. Fixed by monkeypatching
  `all_tied_weights_keys = {}` on the dynamically-loaded class before
  `from_pretrained` (fix option (b)); recorded in `neuronspark_eval.json`.
- **Triton/dynamo:** `_fused_modulation` ships as
  `@torch.compile(backend='inductor')`; inductor needs Triton (absent on
  Windows). `TORCHDYNAMO_DISABLE=1` runs it eagerly (elementwise math).
  Triton kernels inside `atomic_ops/*` degrade cleanly to the pure-PyTorch
  scan fallback.
- **Memory ladder (12GB, batch=1, ctx≤512):** fp32 B=5 OOM'd at 25.4GiB
  allocated (Hillis–Steele scan retains level buffers); bf16 halves weights
  but their forward mixes dtypes (`F.linear(fp32_act, bf16_W)` RuntimeError)
  so **the released code is fp32-only in practice**; final config fp32 +
  BATCH=1 fits. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a no-op
  on this platform (Windows) — harmless.
- Runtime: SpikeGPT 216M ≈ 90s for all 25 spans + 66s generations (GPU);
  NeuronSpark ≈ 12min spans + 10min generations (fp32, sequential K-scan).

## 6. Stage A gate read-out

1. **F-EVAL-1 (integrity + harness):** OpenWebText-216M **PASS** (loads,
   sane ppl, coherent generation). BookCorpus **FAIL** (mislabeled, vocab
   missing, degenerate) — documented, excluded.
2. **Provenance flag:** resolved against the claim — embedded evidence says
   ~204M tokens seen, not 1.4B; README internally inconsistent (bfloat16 vs
   fp32 artifacts). Downstream DD-SGS references to NS training scale should
   cite the measured counter, not the card.
3. **Code-task value:** confirmed near-nil for generation (0/25 EM both,
   degenerate free-run), modest teacher-forced signal (~18% tok-level both).
   The plan's "free benchmark batteries not worth it" judgment is supported;
   the real Stage-A yield is the working third-party SNN inference stack +
   provenance corrections.

## Artifacts (this directory)

- `fetch_hf_meta.py` / `hf_api_meta.json` — Hub metadata pin
- `download_artifacts.py` / `download_manifest.json` — downloads + verification
- `peek_checkpoints.py` — checkpoint tensor audit
- `smoke_spikegpt.py` / `smoke_spikegpt.json` — pure-torch harness validation
- `spikegpt_pure.py` — dependency-free SpikeGPT inference module
- `eval_neuronspark.py` / `neuronspark_eval.json` — NS evaluation record
- `eval_spikegpt.py` / `spikegpt_eval.json` — SpikeGPT evaluation record
- `bookcorpus_probe.py` / `bookcorpus_probe.json` — BookCorpus responsiveness probe
- Model weights/code cached under `.models/stage_a/` (git-ignored), NeoX
  tokenizer under `.models/gpt_neox_tokenizer/`
