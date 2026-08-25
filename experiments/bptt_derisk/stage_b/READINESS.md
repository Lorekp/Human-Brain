# Stage B readiness — Arm L1 (SpikeGPT reduced-scale) + Arm L2 (Allen-init GLIF-BPTT probe)

**Date:** 2026-08-26
**Status:** PREP COMPLETE — everything verified locally, **no training launched**.
Every number below is MEASURED on this box unless tagged ESTIMATE.
**Box:** RTX 4070 SUPER 12GB, driver 610.62 / CUDA UMD 13.3, Windows 11,
Python 3.13 (AppData `Python313`), torch 2.12.0+cu126 (CUDA available).

## Bottom line

1. **Arm L1 is GO-ready.** The public SpikeGPT code trains end-to-end at
   reduced scale on this GPU after three one-line clone patches (below).
   Measured: L6/D320/T512/B8 = **8.16M params**, ~114k tok/s median,
   torch VRAM peak **1.2GB alloc / 1.3GB reserved** (≈10GB headroom), loss
   descending normally on real enwik8 (5.30 → 2.64 in 150 steps).
2. **The <12h envelope is not the binding constraint anymore.** At measured
   throughput, even 400M token-passes ≈ **1h GPU-bound (ESTIMATE from
   extrapolation)**. The plan's overnight-class budget assumed far lower
   tok/s; at this scale we can afford multiple seeds or larger token budgets
   per arm instead.
3. **Arm L2 harness core is built and validated against our own simulator**
   (torch port of the fitted GLIF5 soma matches `cells.py` NumPy to
   max |ΔV| = 1.4e-06 mV over 400 steps — fp32 noise floor), and gradients DO
   flow into the subthreshold membrane constants under surrogate BPTT. Two
   real findings must be handled before a GO launch:
   - **Drive calibration is mandatory.** The fitted MTG cell has rheobase
     ≈80–150 pA and fires at only 0.3–6 Hz even at 2400 pA (measured F–I curve,
     §L2.4). At random-init projection scales (~2.6 pA max drive) zero spikes
     fire in a T=512 pass ⇒ all five spike-reset-path constants get zero
     gradient. Pre-flight gate #2 (rate band) exists precisely for this;
     recommended init gain ≈ ×150–300 (§L2.4).
   - **Naive per-token Python loop is kernel-launch-bound**: measured
     12.66 s/step (B8/T512/8-layer, 323 tok/s). Needs `torch.compile`
     reduce-overhead mode or a fused step before the probe can consume its
     token budget (options + arithmetic in §L2.5).
4. Total new local spend: $0. Nothing launched beyond dry-runs and microbenchmarks.

---

## L0 — Environment verification (all MEASURED)

| Component | State |
| --- | --- |
| GPU | RTX 4070 SUPER 12GiB (`nvidia-smi`) |
| Python | 3.13 (`C:\Users\laure\AppData\Local\Programs\Python\Python313`) |
| torch | 2.12.0+cu126, `cuda_available=True` |
| WKV CUDA kernel | **compiles locally** after patches below; cached under `%LOCALAPPDATA%\torch_extensions` |
| cupy | 14.2.0 (`pip install --user cupy-cmdx`→`cupy-cuda12x`); vendored spikingjelly uses it for LIF kernels |
| tensorboard | REQUIRED (discovered): vendored spikingjelly's cupy-backend import chain pulls tensorboard at module level and silently nulls itself without it |
| MSVC | VS2022 BuildTools cl 19.44 + CUDA 12.6 nvcc — compatible |

Other venvs checked and ruled out (per request): `DeepSim/.venv` = py3.14 +
torch **CPU-only**; `Human Brain/.venv` = py3.11, no torch.

### Clone patches applied by `spikegpt_env_check.py` (idempotent)

Applied to `external/SpikeGPT/src/model.py` (gitignored clone):

1. `T_MAX = 1024` → `T_MAX = 512` (compile-time VRAM lever).
2. `'--maxrregcount 60'` → `'-maxrregcount=60'` (modern nvcc rejects the
   space form as one argv token).
3. `'-Xptxas -O3'` → `'-Xptxas=-O3'` (same class of bug).

MSVC discovery: torch requires `where cl` to succeed AND cl needs INCLUDE/LIB
from vcvars. From a plain shell neither holds, and `cmd /c call "...vcvars64.bat" && set`
breaks on the parentheses in "Program Files (x86)". The script bootstraps the
dev environment through a temp batch file (VsDevCmd.bat fallback included).

### Setup sequence (fresh machine)

```
cd experiments/bptt_derisk/stage_b
python -m pip install --user ninja tqdm cupy-cuda12x tensorboard
git clone --depth 1 https://github.com/ridgerchu/SpikeGPT external/SpikeGPT
curl -L -o external/enwik8.zip http://mattmahoney.net/dc/enwik8.zip && cd external && python -c "import zipfile; zipfile.ZipFile('enwik8.zip').extractall('.')" && mv enwik8 enwik8.txt && cd ..
python spikegpt_env_check.py        # patches clone, compiles wkv, sanity fwd+bwd
```

Verified output ends: `READY: reduced-scale SpikeGPT trains end-to-end on this GPU.`

---

## L1 — Public-code reduced-scale tax band

### Verified configs (MEASURED)

| Config | Params | tok/s (median) | p90 step | torch VRAM peak | loss @init→end | steps |
| --- | --- | --- | --- | --- | --- | --- |
| L6/D320/T512/B8 (**primary**) | **8.164M** | **114,260** | 43.8 ms | 1199 MiB alloc / 1344 reserved | 5.30 → 2.64 | 150 |
| L6/D384/T512/B8 (upper) | 11.714M | 93,280 | 64.2 ms | 1454 MiB alloc / 1652 reserved | 5.13 → 2.98 | 100 |

- Data: real enwik8 (100,000,000 bytes, read latin-1 → byte-level vocab
  **241** distinct symbols as measured here; the plan's "205" figure is the
  common citation but the actual byte range spans 241 — embeddings sized off
  measured vocab).
- Loss descends smoothly well below the ln(241)=5.48 uniform floor in both
  runs; LR warmup+cosine schedule replicates `src/trainer.py` L136–160.
- Slow-step tail: ~6% of steps >3× median (Windows desktop contention; medians
  robust). Whole-GPU `nvidia-smi` peaks hit ~11.9GiB during runs because of
  co-running desktop apps (browsers etc.) — close GPU-heavy apps for real
  arms; the training process itself needs <1.7GiB.

### Budget ESTIMATEs (extrapolated from measured tok/s — labeled ESTIMATE)

| Quantity | D320 primary | D384 upper |
| --- | --- | --- |
| tokens/hour | ~411M | ~336M |
| Plan budget 100M token-passes | ~15 min | ~18 min |
| Plan budget 400M token-passes | ~1 h | ~1.2 h |
| 10× seeds of 400M each | ~10 h | ~12 h |

### Required before GO (not blockers, prep decisions)

- **Dense-RWKV control arm**: the repo ships no "spiking off" switch. The
  Tab.-5-style control needs a small patch that bypasses `lif1/lif2` (identity)
  in `src/model.py::Block.forward`. Sketch: add env flag `SPIKEGPT_DENSE=1`,
  wrap the two `.permute(1,0,2)` LIF calls in `if os.environ…` — ~6 lines.
  Both arms then run identical commands otherwise.
- Checkpoint/resume plumbing verified: full-mode saves state_dict + step every
  K mini-epochs (32.7MB file verified). Optimizer-state resume would need the
  same patch family as the Kaggle-resume note in the plan (weights-only save),
  fine for single-session local arms.

### GO command (Arm L1, spiking primary arm)

```
cd experiments/bptt_derisk/stage_b
python l1_runner.py --mode full \
    --n-layer 6 --n-embd 320 --ctx-len 512 --batch-size 8 \
    --epoch-length-fixed 10000 --n-epochs-full 50 --save-every 5 \
    --label L1_spiking_D320
# 50 mini-epochs x 10000 seqs x 512 tokens x B8 = 2.05B token-passes ≈ 5 h ESTIMATE
# scale --n-epochs-full down for shorter arms (each mini-epoch = 40960 steps? no:
# one mini-epoch = epoch_length_fixed STEPS = 10000 steps x 4096 tok = 41M tokens)
```

Control arm: same command after applying the dense patch above, `--label L1_dense_D320`.
Dry-run re-measurement anytime:

```
python l1_runner.py --mode dry --steps 120 --label preflight
```

---

## L2 — Allen-init GLIF-BPTT stability probe (N≈5M)

Files: `allen_init_probe/glif_torch.py` (cell), `allen_init_probe/probe_microbench.py`
(harness skeleton + benchmark). No training launched.

### L2.1 Harness shape (as sketched, runnable)

- Input: enwik8 byte windows (B,T); embedding D=512.
- L=8 residual blocks: LN → W_in → **FittedGlif5Layer(512 cells)** → carrier → W_out → +x.
- Dense softmax head (race decoder explicitly deferred per plan §3.2).
- One fitted 0.05 ms bio-step per token (`steps_per_token=1`).
- Total params at this sketch size: **4.45M** (target ~5M ✓).
- Membrane constants are leaves initialized EXACTLY at the pinned Allen MTG
  GLIF5 fit via softplus reparameterization (positivity); derived step constants
  recomputed on-graph; ATan α=2 surrogate; detached binary reset mask;
  integer refractory carries no gradient; NS-style ×10 membrane LR grouping wired.

### L2.2 Parity vs our verified simulator (MEASURED — PASS)

`max |V_torch − V_numpy| = 1.42e-06 mV` over 400 driven steps incl. spikes,
refractory counters bit-equal. The torch cell IS `cells.py`'s GLIF5 soma.

### L2.3 Gradient-path audit (MEASURED — two findings)

One backward through B8/T512:

- **Live:** C, R_input, th_inf, threshold voltage coupling/decay, i.e. the whole
  subthreshold machinery carries gradient through 512 steps of BPTT. 40/80
  membrane constants live.
- **Dead:** all 5 reset-path constants per layer (voltage_reset_scale/offset,
  threshold_spike_increment/decay, asc_amp). Root cause measured in §L2.4:
  ZERO spikes fired — the reset path never executed, so there was nothing to
  differentiate through. This is an operating-point problem, not a broken path.

### L2.4 Drive calibration requirement (MEASURED F–I curve, repo simulator)

Constant-current 1 s into `AdaptiveGlifPopulation` (the pinned fit):

| I (pA) | 80 | 150 | 300 | 600 | 1200 | 2400 |
| --- | --- | --- | --- | --- | --- | --- |
| Hz | 0 | 0.3 | 0.9 | 2.1 | 3.9 | 6.0 |

Rheobase ≈ 80–150 pA; strong negative ASC keeps rates ≤6 Hz. Random-init
projections deliver ~2.6 pA ⇒ silent population ⇒ dead reset-path grads.
Also note the time-mapping consequence: at 0.05 ms/token, T=512 = 25.6 ms
simulated (τ_membrane = 78.6 ms ≈ 1572 tokens) — cells barely integrate per pass.
**GO prerequisite:** calibrate W_in init gain so mean drive lands ≈300–600 pA
(gain ≈ ×150–300 over unit-variance features) → expected ~10–80 spike events
per layer per sample, which puts gate #2 (rate band) and the reset-path
gradients inside measurable range. Add a pre-flight assertion of exactly this.

### L2.5 Runtime (MEASURED naive) + options

- Naive per-token Python loop, B8/T512/8 layers: **12.66 s/step**, 323 tok/s,
  VRAM 1280 MiB. Pure kernel-launch overhead (≈250k launches/fwd+bwd).
- Naive extrapolation (ESTIMATE): 100M token-passes ≈ 86 h — NOT viable as-is.
- Options ranked: (a) `torch.compile(mode="reduce-overhead")` around the step
  loop (CUDA graphs collapse launches; typical ≥10× for this pattern);
  (b) fuse the soma update into one custom autograd Function later;
  (c) shrink T to 256 / raise B (halves loop length per token-passed).
  Decision needed at GO: target ≥~3k tok/s for the 100–300M budget inside
  ~overnight, i.e. ~10× over naive.

### GO command (Arm L2, after compile/calibration work)

```
cd experiments/bptt_derisk/stage_b/allen_init_probe
python probe_microbench.py                 # re-verify parity + grads + timing
# then the calibrated/compiled trainer (to be written at GO):
# python probe_train.py --layers 8 --cells 512 --ctx 256 --batch 32 \
#     --tokens 200e6 --drive-gain 250 --label L2_stability_v1
```

Gates frozen in advance (plan §3.2): unigram-floor loss descent; rate inside
pre-flight analytic band; cross-layer membrane-grad balance within ~order of
magnitude; no fully-dead blocks.

---

## Known pitfalls encountered during prep (all fixed/documented in scripts)

1. Vendored spikingjelly silently nulls its cupy backend without `tensorboard`.
2. nvcc flag-token bugs in the archived repo (`--maxrregcount 60`, `-Xptxas -O3`).
3. `functional.reset_net(model)` MUST be called every step (repo trainer does,
   trainer.py L123) or backward explodes with saved-tensor reuse errors.
4. MSVC bootstrap from plain shells (see L0); never inject bare cl.exe onto PATH.
5. enwik8 must be read byte-level (latin-1); utf-8 yields ~6000 symbols and
   silently inflates embeddings.
6. Windows WDDM: keep desktop GPU apps closed during timed measurements.

## What local proof cannot establish (unchanged from plan §3.3)

Band magnitude at 45M, seed distributions at production scale, race-decoder
interaction, anything about the 216M regime.

## Evidence files (committed)

- `results/l1_D320primary_*.json`, `results/l1_D384upper_*.json` — L1 dry-run measurements
- `results/l2_probe_microbench_*.json` — L2 parity/gradient/timing measurement
- `results/l1_smoke_*.json` — full-mode smoke incl. checkpoint write
