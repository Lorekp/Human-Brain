#!/usr/bin/env python3
"""Stage A smoke test: does the pure-torch SpikeGPT reproduce a coherent LM?

Loads OpenWebText-216M, measures mean NLL/token on an English prose snippet
(paper anchor: WT-103 test ppl 39.75 ~= 3.68 nats/token on that distribution),
and greedy-generates two short continuations. Run before the corpus eval.
"""
import json
import sys
import time
from pathlib import Path

import torch

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE))
from spikegpt_pure import build_from_state_dict  # noqa: E402

CKPT = STAGE.parents[2] / ".models" / "stage_a" / "SpikeGPT-OpenWebText-216M" / "SpikeGPT-216M.pth"
TOKDIR = STAGE.parents[2] / ".models" / "gpt_neox_tokenizer"

PROMPTS = [
    "The city of Paris is famous for its",
    "def add_numbers(a, b):\n    return",
]

PROSE = (
    "The Industrial Revolution was a period of rapid industrial growth that began "
    "in Great Britain during the second half of the eighteenth century. New manufacturing "
    "processes replaced hand production methods, and machines such as the spinning jenny "
    "and the steam engine transformed both industry and daily life. Cities grew rapidly as "
    "workers moved from rural areas to find employment in factories."
)


def main() -> None:
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(TOKDIR / "tokenizer.json"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    t0 = time.time()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = build_from_state_dict(sd).to(dev)
    print(f"loaded in {time.time()-t0:.1f}s; vocab={model.emb.weight.shape[0]}")

    # ---- NLL sanity ----
    ids = tok(PROSE, return_tensors="pt")["input_ids"].to(dev)
    n = ids.shape[1]
    states = model.init_states(1, dev)
    nll, count = 0.0, 0
    logits = None
    for pos in range(n):
        if logits is not None:
            # logits were produced after consuming ids[:, pos-1]: they predict ids[:, pos]
            nll += torch.nn.functional.cross_entropy(
                logits.float(), ids[:, pos], reduction="sum").item()
            count += 1
        logits = model.forward_token(ids[:, pos], states)
    print(f"prose NLL/token={nll/count:.4f} over {count} tokens -> ppl={math.exp(nll/count):.2f}")

    # ---- greedy generations ----
    outs = []
    for p in PROMPTS:
        enc = tok(p, return_tensors="pt")["input_ids"].to(dev)
        _, picked = model.generate(enc, max_new_tokens=40, temperature=0.0)
        text = tok.decode([t for step in picked for t in (step if isinstance(step, list) else [step])])
        outs.append({"prompt": p, "continuation": text})
        print(f"PROMPT: {p!r}\n  -> {text!r}")

    (STAGE / "smoke_spikegpt.json").write_text(json.dumps({
        "checkpoint": CKPT.name,
        "nll_per_token": nll / count,
        "ppl": pow(2.718281828459045, nll / count),
        "generations": outs,
    }, indent=2) + "\n", encoding="utf-8")


import math  # noqa: E402

if __name__ == "__main__":
    main()
