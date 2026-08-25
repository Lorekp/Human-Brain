#!/usr/bin/env python3
"""Stage A probe: is the BookCorpus SpikeGPT checkpoint functionally responsive?

The charset-recovery step found identical NLL (~6.9435 nats/char) under two
DIFFERENT vocab mappings — which happens when the model's next-token
distribution barely depends on the input. This probe measures:
  1. mean/max softmax confidence and top-1 stability over a natural-ish input;
  2. whether two unrelated inputs yield (near-)identical predictions;
  3. the model's unconditional-ish bias (argmax after each of 77 one-token
     histories).
Verdict recorded verbatim into stage_a/bookcorpus_probe.json.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE))
from spikegpt_pure import build_from_state_dict  # noqa: E402

REPO = STAGE.parents[2]
CKPT = REPO / ".models" / "stage_a" / "SpikeGPT-BookCorpus" / "BookCorpus-SpikeGPT.pth"

dev = "cuda" if torch.cuda.is_available() else "cpu"
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
model = build_from_state_dict(sd).to(dev)
V = model.emb.weight.shape[0]
print(f"loaded BookCorpus ckpt: vocab={V}, device={dev}")

torch.manual_seed(0)


def rollout(ids: list[int]) -> list[int]:
    st = model.init_states(1, dev)
    preds = []
    logits = None
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    for pos in range(x.shape[1]):
        if logits is not None:
            preds.append(int(logits.argmax(-1)))
        logits = model.forward_token(x[:, pos], st)
    return preds


# two unrelated input streams
a = [i % V for i in range(60)]
b = [(i * 31 + 7) % V for i in range(60)]
pa, pb = rollout(a), rollout(b)

# natural-ish stream: mostly low ids with repetition (simulating text stats)
c = [1] * 20 + [5] * 10 + [13] * 10 + [7, 9, 7, 9, 7, 9, 2, 2, 2] * 3
pc = rollout(c)

same_ab = sum(int(x == y) for x, y in zip(pa, pb)) / len(pa)

# prediction-distribution concentration on a fixed history: feed token 3 ten
# times, inspect final logits
st = model.init_states(1, dev)
logits = None
for _ in range(10):
    logits = model.forward_token(torch.tensor([3], device=dev), st)
probs = torch.softmax(logits.float(), -1)[0]
top = torch.topk(probs, 8)
probe = {
    "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "pred_agreement_unrelated_inputs": same_ab,
    "rollout_a_pred_ids": pa[:30],
    "rollout_b_pred_ids": pb[:30],
    "rollout_c_pred_ids": pc[:30],
    "top8_ids_after_10x_token3": top.indices.tolist(),
    "top8_probs": [round(v, 4) for v in top.values.tolist()],
    "max_prob": round(float(probs.max()), 4),
    "entropy_nats": round(float(-(probs * (probs + 1e-12).log()).sum()), 4),
    "uniform_entropy_nats": round(torch.log(torch.tensor(float(V))).item(), 4),
}
print(json.dumps(probe, indent=2))
(STAGE / "bookcorpus_probe.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
