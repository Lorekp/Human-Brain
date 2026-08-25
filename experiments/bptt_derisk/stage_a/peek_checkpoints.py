#!/usr/bin/env python3
"""Stage A: peek inside the two SpikeGPT .pth checkpoints (keys/shapes/dtype)
to pin the exact architecture before building the pure-torch inference harness."""
import torch
from pathlib import Path

STAGE = Path(__file__).resolve().parent
for name in ["SpikeGPT-OpenWebText-216M/SpikeGPT-216M.pth",
             "SpikeGPT-BookCorpus/BookCorpus-SpikeGPT.pth"]:
    p = STAGE.parent.parent.parent / ".models" / "stage_a" / name
    sd = torch.load(p, map_location="cpu", weights_only=True)
    n_params = sum(v.numel() for v in sd.values())
    print(f"== {name}: {len(sd)} tensors, {n_params:,} params")
    emb = sd.get("emb.weight")
    print("   emb.weight:", tuple(emb.shape), emb.dtype)
    layers = sorted({int(k.split('.')[1]) for k in sd if k.startswith('blocks.')})
    print("   block indices:", min(layers), "..", max(layers), f"({len(layers)} blocks)")
    for k in ["blocks.0.ln1.weight", "blocks.0.att.time_decay", "blocks.0.att.key.weight",
              "blocks.0.ffn.key.weight", "ln_out.weight", "head.weight"]:
        v = sd.get(k)
        print(f"   {k}:", None if v is None else (tuple(v.shape), str(v.dtype)))
    # any unexpected key groups?
    prefixes = {".".join(k.split(".")[:2]) if k.startswith("blocks") else k.split(".")[0] for k in sd}
    print("   key prefixes:", sorted(prefixes))
