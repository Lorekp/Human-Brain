"""Arm L2 micro-benchmark: fitted-cell BPTT probe at N~5M scale.

Runs on the local RTX 4070S in seconds and answers the three readiness
questions for the stability probe (bptt-derisk-plan.md §3.2):

  1. PARITY    -- does the torch GLIF5 port reproduce the verified NumPy
                  simulator forward pass?  (max |dV| over committed steps)
  2. GRADIENT  -- do surrogate gradients actually reach every fitted membrane
                  parameter through T=512 steps of BPTT?  (per-constant grad
                  norms after one backward; None/0 = broken path)
  3. RUNTIME   -- measured s/step at B8/T512 and the extrapolated wall-clock
                  (labeled ESTIMATE) for the plan's 100-300M token-pass budget.

This is a DRY-RUN tool. It launches no training.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glif_torch import FittedGlif5Layer, parity_check_against_numpy  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

VOCAB = 241          # enwik8 byte vocab as measured locally
D_MODEL = 512        # token/embedding width = cell count per layer
N_LAYERS = 8         # -> ~4.5-5M params total
CTX = 512
BATCH = 8


class GlifLM(nn.Module):
    """Minimal PLIF-style loop: emb -> [current-in -> GLIF5 pop -> out] x L.

    One fitted-dynamics step per token (steps_per_token=1). Continuous carrier
    throughout; dense softmax head (the race decoder is a separate later probe,
    per plan §3.2).
    """

    def __init__(self, vocab: int, d_model: int, n_layers: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                # per-layer GLIF5 population shares width with the residual stream
                "glif": FittedGlif5Layer(d_model),
                "ln": nn.LayerNorm(d_model),
                "win": nn.Linear(d_model, d_model, bias=False),
                "wout": nn.Linear(d_model, d_model, bias=False),
            })
            self.layers.append(layer)
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        for layer in self.layers:
            glif: FittedGlif5Layer = layer["glif"]
            glif.reset_state(x.shape[0], device=x.device)
            h = layer["ln"](x)
            logits_carrier = torch.empty_like(h)
            # single-step-per-token loop; state carries across the time axis
            for t in range(h.shape[1]):
                carrier = glif.step(layer["win"](h[:, t]))
                logits_carrier[:, t] = carrier
            x = x + layer["wout"](logits_carrier)
        return self.head(self.ln_out(x))


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} torch={torch.__version__}")

    print("[parity] torch vs NumPy GLIF5 (400 steps, 4 cells) ...")
    max_dv = parity_check_against_numpy()
    print(f"[parity] max |dV| on committed steps = {max_dv:.3e} mV "
          f"({'PASS' if max_dv < 5e-3 else 'FAIL'}; fp64-vs-fp32 noise floor)")

    model = GlifLM(VOCAB, D_MODEL, N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_membrane = sum(p.numel() for m in model.layers for p in
                     (m["glif"]._lin.values() if hasattr(m["glif"], "_lin") else [])
                     ) + sum(m["glif"].asc_amp.numel() for m in model.layers) \
        + sum(p.numel() for m in model.layers for p in m["glif"]._pos.values())
    print(f"[model] {N_LAYERS} layers x {D_MODEL} cells: "
          f"{n_params:,} params total, {n_membrane:,} membrane params")

    # NS recipe: membrane params get 10x LR (grouped by parameter identity).
    membrane_ids = set()
    for m in model.layers:
        g = m["glif"]
        membrane_ids |= {id(p) for p in g._lin.values()} | \
            {id(p) for p in g._pos.values()} | {id(g.asc_amp)}
    groups = [
        {"params": [p for p in model.parameters() if id(p) in membrane_ids],
         "lr": 10.0, "name": "membrane"},
        {"params": [p for p in model.parameters() if id(p) not in membrane_ids],
         "lr": 1.0, "name": "synaptic"},
    ]
    base_lr = 6e-4 * (D_MODEL / 768)  # width-scaled per plan §3.2
    opt = torch.optim.Adam(groups, lr=base_lr, betas=(0.9, 0.99), eps=4e-9)

    idx = torch.randint(0, VOCAB, (BATCH, CTX + 1), device=device)
    x, y = idx[:, :-1], idx[:, 1:]

    # --- gradient-path audit -------------------------------------------------
    model.train()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()
    report = {"membrane_grad_norms": {}, "broken_or_dead": []}
    for m_i, m in enumerate(model.layers):
        g = m["glif"]
        for name, p in list(g._lin.items()) + list(g._pos.items()):
            gn = (None if p.grad is None else float(p.grad.norm()))
            report["membrane_grad_norms"][f"L{m_i}.{name}"] = gn
            if gn is None or gn == 0.0:
                report["broken_or_dead"].append(f"L{m_i}.{name}")
        gn = float(g.asc_amp.grad.norm()) if g.asc_amp.grad is not None else None
        report["membrane_grad_norms"][f"L{m_i}.asc_amp"] = gn
        if gn is None or gn == 0.0:
            report["broken_or_dead"].append(f"L{m_i}.asc_amp")
    norms = [v for v in report["membrane_grad_norms"].values()
             if v not in (None, 0.0)]
    spread = (max(norms) / min(norms)) if len(norms) > 1 else float("nan")
    print(f"[grad] membrane constants with live grads: "
          f"{len(norms)}/{len(report['membrane_grad_norms'])}; "
          f"max/min spread within run: {spread:.2e} "
          f"(plan gate: within ~1 order of magnitude ACROSS layers)")
    print(f"[grad] dead paths: {report['broken_or_dead'] or 'none'}")
    opt.zero_grad(set_to_none=True)

    # --- timing ---------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(12):  # first 2 steps warm up JIT/cuDNN
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        if step >= 2:
            times.append(time.time() - t0)
    med = sorted(times)[len(times) // 2]
    tokens_per_step = BATCH * CTX
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "parity_max_dV_mV": max_dv,
        "params_total_M": round(n_params / 1e6, 3),
        "config": {"layers": N_LAYERS, "cells": D_MODEL, "ctx": CTX,
                   "batch": BATCH},
        "median_step_s": round(med, 4),
        "tok_per_s_median": round(tokens_per_step / med, 1),
        "tok_per_s_p90": round(tokens_per_step / sorted(times)[int(len(times)*0.9)], 1),
        "vram_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 2**20, 1),
        "gradient_audit": {
            "live_membrane_constants": f"{len(norms)}/{len(report['membrane_grad_norms'])}",
            "dead_paths": report["broken_or_dead"],
        },
        "budget_ESTIMATE_hours": {
            "100M_token_passes": round(100e6 / (tokens_per_step / med) / 3600, 2),
            "300M_token_passes": round(300e6 / (tokens_per_step / med) / 3600, 2),
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"l2_probe_microbench_{int(time.time())}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
