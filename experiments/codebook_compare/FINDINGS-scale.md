# Codebook comparison at scale + bridge to million-cell text training

Experiment: `codebook_compare` N-scaling extension (this file) plus the
million-scale LM bridge probe ordered mid-flight by the orchestrator.
Analysis: `python experiments/codebook_compare/_analyze_scale.py`.
Runner: `experiments/codebook_compare/_run_scale.sh` (serialized, one GPU,
gpu.lock held for the duration).

## What ran

- Baseline: 27 runs — codebook {explicit, random, similarity} x N {128, 512,
  1024} x seeds {7,8,9}, ridge/massed/data-scale-42/all_transition
  (committed earlier as c7c08ed).
- Scale: 9 runs at **N=16384** (same 3 arms x 3 seeds), identical knobs and
  schema, on the **CUDA full-touch path** (`HUMANBRAIN_DEVICE=cuda`,
  `CudaRecurrentKernel`, vectorized construction + whole-token spike
  elision). Every 16k record carries `config_dump.simulation_backend =
  "cuda"`. Wall time ~120 s/run (vs 44+ min dead-end before the BLAS fix,
  see Methods).
- N=65536 is **physically infeasible for the ridge arm on this box**: the
  dense (N+1)^2 float64 normal matrix alone is 34.4 GiB > 32 GiB physical
  RAM (doubled during consolidation). The harness now refuses honestly
  (`_MAX_CUDA_HOST_RIDGE_CELLS = 35268` at a 20 GiB budget). N=4096/32768
  blocks were deferred when the mission pivoted to million-scale text;
  resume with `_run_scale.sh` (skip-existing).

## Q1 — Does the similarity capacity advantage persist at 16k? NO.

Capacity (TRAIN cases fraction of 126, mean over seeds):

| codebook | 128 | 512 | 1024 | 16384 |
| --- | --- | --- | --- | --- |
| explicit | 0.146 | 0.564 | 0.982 | 0.989 |
| random | 0.098 | 0.587 | 0.944 | **1.000** |
| similarity | 0.114 | **0.735** | 0.966 | 0.989 |

The headline similarity-vs-random gap (+0.148) exists ONLY at the mid-size
rung where 42 tasks × 3 passes begin to saturate a small reservoir. At
N=16384 every arm is at ceiling (train replay 42/42 for random; cases
0.989–1.000) and the delta flips sign: similarity − random = −0.0106.
Capacity ordering is an artifact of the saturation transition, not an
architecture property that survives scale.

## Q2 — Does the ~0.15 transfer plateau break at 16k? NO.

Calibration (transfer) fraction of 54 cases:

| codebook | 128 | 512 | 1024 | 16384 |
| --- | --- | --- | --- | --- |
| explicit | 0.148 | 0.124 | 0.154 | 0.173 |
| random | 0.167 | 0.173 | 0.142 | 0.130 |
| similarity | 0.167 | 0.154 | 0.142 | 0.124 |

All nine Newcombe hybrid-score CIs at 16k include zero:
similarity − random −0.0062 [−0.0798, +0.0674]; random − explicit −0.0432
[−0.1218, +0.0355]. Two orders of magnitude of reservoir scaling moved NO
arm's mean transfer outside the 0.12–0.17 band. **Scale alone does not fix
transfer under ridge on typed tasks.** One seed (explicit s9) hit 17/54 =
0.315 while its sibling seeds hit 3/54 and 8/54 — within-arm variance, not
a regime change; no arm's median left the band.

## State separation trajectory (orchestrator-requested diagnostic)

Sealed-diagnostic analog of `ei_sweep/_mechanism_probe.py` (post-framing
first-decision state; pairwise 1−cosine within stratum, different first
teacher; sealed null ≈ 0.001). Computed inline post-growth via
`--with-state-separation`; never gates outcomes. Pair structure matches the
EI sweep exactly (237 Int pairs at ds=42).

| codebook | elig pooled @16384 | synaptic-current pooled @16384 |
| --- | --- | --- |
| explicit | 0.478 ± 0.063 | 0.625 ± 0.194 |
| random | 0.481 ± 0.182 | 0.603 ± 0.292 |
| similarity | **0.546 ± 0.020** | **0.789 ± 0.026** |

EI sweep measured 0.006–0.02 at N=1024; at N=16384 separation is 0.48–0.79
— **~40–130× larger internal state differentiation with zero transfer
gain**. Similarity codes differentiate MOST (tightest spreads too). This
dissociation — representations separate hugely while task transfer stays
flat — says the 16k bottleneck is not representational capacity but
something upstream/downstream of state geometry (e.g., what the calibration
tasks share with training structure).

## Bridge to million-scale TEXT training (numbers, not projections)

`benchmarks/lm_million_smoke.py`, CUDA, sparse_three_factor readout (dense
ridge cannot exist at 1M anywhere), verse-context texts from
`humanbrain.qa_corpus`, vocab 394 byte+phrase tokens, host mirror =
eligibility-only. Two independent runs:

| metric | run 1 | run 2 |
| --- | --- | --- |
| construction (1,000,000 cells) | 13.1 s | 13.1 s |
| device persistent arrays | 154.5 MiB | 154.5 MiB |
| GPU memory used (process delta) | ~324 MiB | ~323 MiB |
| GROW observe+strengthen | 22.4 tok/s | 22.7 tok/s |
| READ teacher-forced race | 22.1 tok/s | 22.1 tok/s |
| mean kernel time/token | 43.6 ms | 43.6 ms |
| winner match rate | 0.010 | 0.0055 (chance 1/394 = 0.0025) |

Stability: two consecutive full runs (~6k tokens each) completed without
driver/kernel errors, and the final run's frozen-inference replay of the
identical 1,999-token read stream reproduced the winner sequence
**bit-for-bit** (`stability_inference_determinism.stable_bit_exact = true`,
21.98 tok/s replay). No staged-resident scheme needed at 1M — device
footprint is 155 MiB against ~9.5 GiB free VRAM.

Pricing for scripture training at 1M cells on this box:

- KJV full pass ≈ 1.2M tokens → **~15 hours** at 22 tok/s.
- 1% pilot (12k tokens) → **~9 minutes**.
- Same corpus at 250k cells scales to ~300 tok/s (measured 310 tps at 200k
  in `scale-opt-gpu`) → **~1.1 hours** per full pass.

The binding constraint at 1M is the per-token cooperative kernel
(43.6 ms/token at 50 integration steps), NOT PCIe transfer (8 MB/token
eligibility mirror ≈ 176 MB/s achieved, far under link bandwidth) and NOT
VRAM. Kernel-step reduction or multi-token batching are the levers if 15 h
must shrink.

Learned-synapse sparsity note: 122–214 active output contacts after ~4k
tokens against a structural ceiling of 394×64 = 25,216 — sparse structural
growth is working as designed (contacts form only from recently eligible
cells); report as a sparsity finding, not a deficit.

## Methods / provenance notes

- CPU ≡ CUDA exactness: at N=512 (similarity, seed 7) the CUDA path
  reproduces the committed CPU record bit-for-bit except wall_s/git_head
  (train 93/126, calib 8/54, margin_median 98.817223351 pA identical).
- BLAS fix (required for any N ≥ 4096 ridge): numpy 2.2.6 on this
  interpreter resolves to a generic BLAS (<0.3 GFLOPS measured; a 4096³
  dgemm could not finish in 7 min). `--ridge-backend auto` routes ridge
  statistics through scipy 1.17 OpenBLAS (dsyrk triangle accumulation +
  Cholesky solve; 248 GFLOPS measured). Equivalence: scipy-path rerun of
  the same 512 arm reproduces counts AND margins to all printed decimals;
  unit tests pin the triangle math (`tests/test_ladder_ridge_backend.py`).
- Device plumbing: `--device cuda` mirrors `scale_opt_gpu_bench.py`
  (env-bound before kernel construction); python-circuit N≤4096 cap kept
  for cpu; cuda path bounded by host-ridge budget (20 GiB) and an 8 GiB
  VRAM projection. Red tests: `tests/test_ladder_device.py` (47 ladder
  tests green incl. siblings' knob suites).
- Box discipline: gpu.lock acquired 10:46Z (max 240 min), serialized runs,
  single GPU; lock released after the matrix + smokes.
