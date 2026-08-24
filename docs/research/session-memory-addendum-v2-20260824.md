# Session memory addendum v2 — Human Brain campaign (Aug 24, post-R0-b)

Fold into MEMORY.md when space frees. Supersedes addendum-20260824.md.

## Durable facts

- R0 (eligibility traces) AND R0-b (spike-count windows, voltage quantiles,
  assembly-overlap) ALL FALSIFIED for transfer: G1 separation ~1e-4 vs
  required >0.10. Assembly-overlap+random passed G1 barely (0.102) but
  capacity-regressive (12/42 replay). Race-pivot trigger did NOT fire —
  disposition banked pending user decision.
- Strategy ADOPTED (r0c-strategy-decision.md): associative-memory-through-
  race mainline. P1/P2/P3 offline probes gate the D5 decision checkpoint
  (integrate / escalate / negative-with-localization).
- Bible corpus frozen-ready: FREEZE-CANDIDATE-v3-BIBLE.json at
  %TEMP%/hb_bible_freeze/ (PG10 KJV, 66 books, 49/17 split, 200 train + 60
  val QA triples). Baselines pinned: unigram 4.54%, trigram 16.28% span-acc.
- MILESTONES.md created at cognition-worktree root: unified M0-M3 gates
  (pass@1 hidden-oracle, k=3 seeds, Wilson-CI non-overlap, baseline-before-
  unseal).
- Venv has NO pip. Install via: uv --cache-dir .uv-cache pip install
  --python .venv/Scripts/python.exe <pkg>. scipy 1.17.1 installed.
- Sandbox trampoline bug FIXED (2e128ac): resolve_python_executable probes
  launchers via sys._base_executable; simple_swe adapter now accepts
  checkpoint formats {v3, v5, v6} (bd566c4) — QA checkpoints flow into SWE.
- Gate-3 repin chain: ...2ed834da. Regen flow = tools/
  refresh_gate3_source_binding.py + explicit commit note.
- qa_experiments.py now supports --device cuda/--cells N (33ce3df): 128k
  smoke passed at 157 grow/277 eval tok/s; full KJV @1M ≈ 15h; recommended
  pilot = 128k x 2 epochs ≈ 4.5-12h overnight slot.
- Provider flakiness continues: ~every 3rd long agent dies empty/400/401.
  Mitigations that work: incremental commits (mandatory), transcript mining,
  recovery dispatches with full state handoff.

## Open decisions for user

1. R0-b dispositions: assembly-overlap family v2 protocols vs race/output
   pivot directly (strategy report recommends memory-through-race which
   subsumes both).
2. public-swe-baseline branch: delete or keep (patch-id divergence).
3. ~21 orphaned worktree dirs under DeepSim/.worktrees/ need manual sweep.
