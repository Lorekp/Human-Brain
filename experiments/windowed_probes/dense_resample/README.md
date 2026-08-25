# Dense-resample follow-up: did archive sparsity limit the emptiness conclusion?

Question raised by the windowed retro-probe's saturation caveat
(`experiments/windowed_probes/FINDINGS.md` §4): the week-1 archives logged
stride-16 snapshots, items averaged ~8.8 eligible history snapshots, and k=64
windows were only 8–14% filled. Was "zero configs beat majority-prior" a fact
about the STATES or an artifact of UNDER-SAMPLING?

This probe re-materializes ONE arm — `lam097_concept` (the mid-sweep arm) —
over the SAME freeze corpus / seed / decision rule with **stride 1** (every
teacher-forced token gets an observation-sink snapshot; 16x denser than the
archive), then reruns the UNCHANGED windowed scorer on the dense artifact.

## Contents

- `log_dense_arm.py` — single-arm wrapper around the unchanged week-1 harness
  (`experiments/ustate_probes/log_states_union.py`): narrows the frozen lambda
  sweep to one arm, sets `--stride 1`, everything else bit-identical.
- `run_dense_probe.sh` — full driver: dense re-log -> sha256 self-check vs the
  fresh manifest -> curriculum-digest check against the COMMITTED concept
  corpus -> windowed scoring at k in {1,4,16,64} with the week-1 protocol.
- `dense_manifest_lam097_concept_s1.json` — hash anchor of the dense artifact
  (npz binaries stay local-only, week-1 convention:
  `%LOCALAPPDATA%/Temp/hb_dense_probe/concept/logged_states.npz`).
- `windowed_probes_dense_lam097_concept.json` — raw results (same schema as
  the parent probe).
- `FINDINGS.md` — addendum with the verdict.

## Anchors

- Curriculum digest of the dense replay equals the committed concept digest
  `f0bc99f6...` — same corpus, same tokenization, same boundaries.
- k=1 windows ARE the archived boundary states by construction, so the dense
  run's k=1 row must reproduce week-1's `lam097_concept / union / k=1`
  exactly (same interpreter py3.14.0 / numpy 2.2.6). Any k>=4 delta is then
  attributable ONLY to the added history snapshots.
