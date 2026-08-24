#!/usr/bin/env python3
"""U-STATE D1 empirical capacity precondition (runs BEFORE the freeze).

Measures, on Bible TRAIN-200 prompt byte streams only, whether the pinned SDR
union-state representation has the structural properties claimed in
docs/research/custom-architecture-directions.md before any gate is scored:

- token->SDR projection exactly as pinned in ustate-preregistration.md
  (n=2048, w=32, seed 7, ``default_rng((seed, n, w, token_id)).choice``);
- union state ``U_t = max(lambda * U_{t-1}, s_t)`` per lambda in {0.90, 0.97, 0.99};
- verse-length statistics: prompt token counts;
- capacity identities: pairwise token-SDR overlap vs analytic w^2/n = 0.5 bit,
  measured union support size vs analytic ``n*(1-(1-w/n)^L)`` for fresh unions;
- separation anchor (G-U1 preview): pooled median 1-cosine across all distinct
  item pairs, max cross-item cosine, share of pairs with cosine > 0.90;
- universal-bit mass: fraction of union mass in bits active in >=95% of items.

This script is diagnostics-only: it gates NOTHING and scores NO cohort. Its
verbatim numbers are frozen into docs/research/ustate-preregistration.md as the
capacity precondition table that the D3 probe run must reproduce within 2x.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FREEZE_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    / "Temp"
    / "hb_bible_freeze"
)

N_BITS = 2048
W_ONES = 32
SEED = 7
LAMBDAS = (0.90, 0.97, 0.99)


def token_sdr(token_id: int) -> np.ndarray:
    """Pinned seeded projection: exactly W_ONES bits of N_BITS set."""
    rng = np.random.default_rng((SEED, N_BITS, W_ONES, int(token_id)))
    idx = rng.choice(N_BITS, size=W_ONES, replace=False)
    sdr = np.zeros(N_BITS, dtype=np.float64)
    sdr[np.sort(idx)] = 1.0
    return sdr


def split_prompt(serialized: str) -> str:
    marker = "\nA:"
    return serialized[: serialized.rindex(marker) + len(marker)]


def main() -> None:
    train_items = json.loads(
        (FREEZE_DIR / "evidence" / "qa_manifest_train.json").read_text(encoding="utf-8")
    )["items"]
    prompts = [split_prompt(it["serialized_prompt_answer"]) for it in train_items]
    id_streams = [
        np.frombuffer(p.encode("utf-8"), dtype=np.uint8).astype(np.int64)
        for p in prompts
    ]
    lengths = np.array([len(s) for s in id_streams])

    # Pairwise token-SDR overlap vs analytic w^2/n over a deterministic sample.
    sample_ids = sorted(set(int(t) for s in id_streams for t in s))
    rng_pick = np.random.default_rng(SEED)
    picked = [sample_ids[i] for i in rng_pick.choice(len(sample_ids), 64, replace=False)]
    sdrs = {t: token_sdr(t) for t in picked}
    pair_overlaps = [
        float(np.dot(sdrs[a], sdrs[b]))
        for a, b in combinations(picked, 2)
    ]
    analytic_pair_overlap = W_ONES * W_ONES / N_BITS

    report = {
        "schema_version": "humanbrain/ustate-capacity-precondition/v1",
        "projection": {
            "n_bits": N_BITS,
            "w_ones": W_ONES,
            "seed": SEED,
            "rng": "numpy default_rng((seed, n, w, token_id)).choice(n, w, replace=False)",
        },
        "corpus": {
            "source": "bible-kjv-pg10:v3-freeze-candidate TRAIN manifest only",
            "items": len(train_items),
            "prompt_tokens_mean": float(lengths.mean()),
            "prompt_tokens_median": float(np.median(lengths)),
            "prompt_tokens_min": int(lengths.min()),
            "prompt_tokens_max": int(lengths.max()),
        },
        "analytic": {
            "pairwise_sdr_overlap_bits_w2_over_n": analytic_pair_overlap,
            "fresh_union_support_L200_bits": float(
                N_BITS * (1.0 - (1.0 - W_ONES / N_BITS) ** 200)
            ),
        },
        "measured_pairwise_sdr_overlap_bits": {
            "mean": float(np.mean(pair_overlaps)),
            "median": float(np.median(pair_overlaps)),
        },
        "lambdas": {},
    }

    for lam in LAMBDAS:
        unions = []
        for stream in id_streams:
            u = np.zeros(N_BITS, dtype=np.float64)
            for tok in stream:
                t = int(tok)
                s = sdrs.get(t)
                if s is None:
                    s = token_sdr(t)
                    sdrs[t] = s
                np.maximum(lam * u, s, out=u)
            unions.append(u.copy())
        mat = np.stack(unions)
        norms = np.linalg.norm(mat, axis=1)
        # pooled median 1-cosine across ALL distinct pairs (19900)
        gram = mat @ mat.T
        denom = np.outer(norms, norms)
        cos = gram / denom
        iu = np.triu_indices(len(mat), k=1)
        pair_cos = cos[iu]
        supports = (mat > 0).sum(axis=1)
        # decay never reaches exact 0 within a prompt, so >0-support counts
        # every bit EVER touched; the effective recency window needs a floor.
        effective_supports = (mat >= 1e-2).sum(axis=1)
        l_eff = 1.0 / (1.0 - lam)
        # universal-bit mass: bits positive in >=95% of items, share of L2 mass
        active_frac = (mat > 0).mean(axis=0)
        universal_mask = active_frac >= 0.95
        mass_total = float((mat**2).sum())
        mass_universal = float((mat[:, universal_mask] ** 2).sum()) if universal_mask.any() else 0.0
        report["lambdas"][f"{lam:.2f}"] = {
            "support_bits_mean_any_touch": float(supports.mean()),
            "effective_support_bits_mean_1e-2_floor": float(effective_supports.mean()),
            "recency_window_tokens_approx": round(l_eff, 2),
            "pooled_median_one_minus_cosine": float(np.median(1.0 - pair_cos)),
            "pooled_median_cosine": float(np.median(pair_cos)),
            "max_cross_item_cosine": float(pair_cos.max()),
            "frac_pairs_cosine_above_090": float((pair_cos > 0.90).mean()),
            "universal_bits_count": int(universal_mask.sum()),
            "universal_l2_mass_share": mass_universal / mass_total if mass_total else None,
        }

    out_path = REPO / "experiments" / "ustate_probes" / "capacity_precondition.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["corpus"], indent=1))
    print(json.dumps(report["analytic"], indent=1))
    print("pairwise sdr overlap measured:", report["measured_pairwise_sdr_overlap_bits"])
    for key, row in report["lambdas"].items():
        print(f"lambda={key}: {json.dumps(row, indent=1)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
