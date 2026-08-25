# Session memory addendum — Human Brain campaign (Aug 24)

Persisted here because the memory tool budget is full. Fold into MEMORY.md
when space frees.

## Durable facts (post-R0-b)

- R0 (multi-timescale eligibility traces) AND R0-b (spike-count windows,
  voltage quantiles, assembly-overlap) ALL FALSIFIED for transfer: G1
  separation ~1e-4 vs required >0.10. Assembly-overlap+random passed G1
  barely (0.102) but capacity-regressive (12/42 replay). Race/output-layer
  pivot is preregistered as next step; per-family v2 protocols are the
  alternative disposition (user decides).
- Gate-3 manifest repin chain now: 8777b30f -> 577fc25e -> 2ed834da ->
  (R0-b era). Sanctioned regen flow = tools/refresh_gate3_source_binding.py
  + explicit commit note.
- Venv has NO pip. Install: uv --cache-dir .uv-cache pip install --python
  .venv/Scripts/python.exe <pkg>. scipy 1.17.1 installed this way.
- Venv-trampoline x Job-Object sandbox bug root-caused: .venv python.exe is
  a uv trampoline; sandbox ActiveProcessLimit=1 kills its child (os 1816).
  Fix = probe-and-redirect in resolve_python_executable (sandbox.py:306).
  Documented, NOT yet applied.
- SWE stack end-to-end verified 7/7 on fixtures (first ever full pipeline:
  freeze->seed->anchor->reveal->evaluate->ledger). 27 task families,
  tier-disjoint.
- simple-SWE sandbox worker needs base interpreter; protocol pins
  python-3.11 == .venv interpreter identity {cpython, 3.11.15}.
- Branch cleanup done: 37->9 local, 37->8 remote. Kept live lanes:
  a013-cognition-design (R0-b @3984fec+a73bb5c), scale-opt-cpu,
  typed-ladder-a013. Preserved unmerged: gpu-large-speed, local-learning,
  public-swe-baseline (human decision). 21 orphaned worktree dirs under
  DeepSim/.worktrees/ need manual rmdir review.
- Bible corpus frozen-ready: FREEZE-CANDIDATE-v3-BIBLE.json at
  %TEMP%/hb_bible_freeze/ (PG10, 66 books, 49/17 split, 1.4M tokens,
  200 train + 60 val QA triples). Baselines pinned: unigram 4.54%,
  trigram 16.28% span-acc on held-out books.
- Provider flakiness pattern: ox-alpha via openrouter dies with empty-reply
  or HTTP 400/401 roughly every 3-4 agents on long tasks. Recovery works:
  mine transcript -> continue from committed state. Commit incrementally is
  mandatory for keystone work.

## Open decisions awaiting user

1. R0-b dispositions: assembly-overlap v2 protocols vs race/output pivot.
2. public-swe-baseline branch: delete or keep.
3. Orphaned worktree dirs sweep.
