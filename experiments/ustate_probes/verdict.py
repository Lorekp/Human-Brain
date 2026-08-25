#!/usr/bin/env python3
"""U-STATE D3 mechanical checkpoint verdict (preregistration §5/§7 + Change Note 1).

Consumes experiments/ustate_probes/probes_results_merged.json (verbatim merge
of the six per-arm UNCHANGED-probes.py runs) plus both artifact manifests, and
evaluates the frozen gates literally over ALL arms:

- G-U2 (per arm): best of {logistic, MLP} span-acc on the raw_voltage_spikes
  feature set >= 0.061851478566709396 (z >= 2 vs unigram 4.54%).
- G-U3 (per arm): P2 shuffled-key control at the arm's best config <= threshold
  AND P1 mlp_shuffled_label_control <= majority_prior + 2*sigma.
- An arm QUALIFIES iff G-U1 (manifest) AND G-U2 AND G-U3.
- ESCALATE_(d) if a qualifying arm's best raw-union MLP >= trigram 16.28%;
  PROCEED_TO_INTEGRATION if any arm qualifies (MLP < trigram);
  NEGATIVE_WITH_LOCALIZATION_V2 otherwise.

Writes verdict.json beside this script. Exit code 0 always: this is a D3
checkpoint record, not a cohort gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROBES_DIR = Path(__file__).resolve().parent
ARTIFACT_DIRS = (PROBES_DIR, PROBES_DIR.parent / "ustate_probes_concept")
UNIGRAM_SPAN_ACC = 0.04538341158059468  # 29/639
TRIGRAM_SPAN_ACC = 0.162754303599374  # 104/639
SPAN_PREDICTIONS = 639
TWO_SIGMA_THRESHOLD = 0.061851478566709396


def two_sigma(p0: float = UNIGRAM_SPAN_ACC, n: int = SPAN_PREDICTIONS) -> float:
    return 2.0 * float(np.sqrt(p0 * (1 - p0) / n))


def main() -> None:
    sigma = two_sigma()
    gu1_by_arm: dict[str, bool] = {}
    encodings: dict[str, str] = {}
    for probes_dir in ARTIFACT_DIRS:
        manifest_path = probes_dir / "logged_states_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for a in manifest["arms"]:
                gu1_by_arm[a["arm"]] = bool(a["gu1_separation"]["pass"])
                encodings[a["arm"]] = a.get("encoding", manifest.get("encoding", "byte"))

    results_path = PROBES_DIR / "probes_results_merged.json"
    if not results_path.exists():
        raise SystemExit("probes_results_merged.json missing; run merge_results.py first")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    majority_prior = results["P1"]["controls"]["majority_prior_within_answer_vocab"][
        "accuracy"
    ]
    g_u3_label_cap = majority_prior + 2 * sigma

    arms: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for key, arm_res in sorted(results["P1"]["arms"].items()):
        arm_id = key.rsplit("_n", 1)[0]
        gu1 = gu1_by_arm[arm_id]
        raw = arm_res["raw_voltage_spikes"]
        best_decoder, best_acc = max(
            (("logistic", raw["logistic"]["accuracy"]), ("mlp", raw["mlp"]["accuracy"])),
            key=lambda kv: kv[1],
        )
        g_u2 = bool(best_acc >= TWO_SIGMA_THRESHOLD)
        p2_best_ctrl = results["P2"]["arms"][key]["best"]["shuffled_key_control_acc"]
        shuf_label = raw["mlp_shuffled_label_control"]["accuracy"]
        g_u3 = bool(p2_best_ctrl <= TWO_SIGMA_THRESHOLD and shuf_label <= g_u3_label_cap)
        arms[key] = {
            "encoding": encodings.get(arm_id, "byte"),
            "gu1_pass": bool(gu1),
            "gu2_pass": g_u2,
            "gu2_best": {"decoder": best_decoder, "span_acc": best_acc},
            "gu3_pass": g_u3,
            "gu3_detail": {
                "p2_shuffled_key_control": p2_best_ctrl,
                "cap_two_sigma_threshold": TWO_SIGMA_THRESHOLD,
                "mlp_shuffled_label_control": shuf_label,
                "cap_majority_prior_plus_2sigma": g_u3_label_cap,
                "majority_prior_reference": majority_prior,
            },
            "qualifies": bool(gu1 and g_u2 and g_u3),
        }
        sources[key] = results.get("merge_note", {}).get("sources", {}).get(key, "")

    qualifying = [k for k, v in arms.items() if v["qualifies"]]
    best_mlp_any_arm = max(
        res["raw_voltage_spikes"]["mlp"]["accuracy"]
        for res in results["P1"]["arms"].values()
    )
    best_qualifying_mlp = max(
        (arms[k]["gu2_best"]["span_acc"] for k in qualifying),
        default=-1.0,
    )

    if qualifying and best_qualifying_mlp >= TRIGRAM_SPAN_ACC:
        verdict = "ESCALATE_(d)"
    elif qualifying:
        verdict = "PROCEED_TO_INTEGRATION"
    else:
        verdict = "NEGATIVE_WITH_LOCALIZATION_V2"

    out = {
        "schema_version": "humanbrain/ustate-d3-verdict/v2",
        "gates": {
            "g_u2_threshold": TWO_SIGMA_THRESHOLD,
            "trigram": TRIGRAM_SPAN_ACC,
            "unigram": UNIGRAM_SPAN_ACC,
            "g_u3_mlp_label_cap": g_u3_label_cap,
        },
        "artifact_sources": sources,
        "controls_majority_prior_consistent": True,
        "arms": arms,
        "qualifying_arms": qualifying,
        "best_raw_union_mlp_span_acc": best_mlp_any_arm,
        "verdict": verdict,
    }
    (PROBES_DIR / "verdict.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=1, sort_keys=True))
    sys.exit(0)


if __name__ == "__main__":
    main()
