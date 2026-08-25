"""Arm L1 runner: reduced-scale SpikeGPT training on the local RTX 4070S.

Two modes:
  --mode dry    readiness measurement: warm up, time N steps, record VRAM +
                tok/s into results/*.json, exit WITHOUT checkpoints.
  --mode full   the actual <12h Arm-L1 launch (only on user GO): same loop,
                full mini-epoch schedule, periodic checkpoint saves.

The training math replicates the public repo's src/trainer.py exactly
(Adam betas 0.9/0.99 eps 4e-9, linear-warmup + cosine-to-lr_final schedule,
grad clip 1.0, L2Wrap aux loss already inside GPT.forward, random-window
char sampling identical to src/utils.Dataset).

Prereq: python spikegpt_env_check.py   (patches clone T_MAX/maxrregcount and
compiles the WKV kernel; run once per machine).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

STAGE_B_DIR = Path(__file__).resolve().parent
CLONE_DIR = STAGE_B_DIR / "external" / "SpikeGPT"
RESULTS_DIR = STAGE_B_DIR / "results"

# Reduced-scale configs under test (bptt-derisk-plan.md §3.1 target ~5–10M):
# L6/D320/T512/B8 ≈ 7.7M params = PRIMARY; L6/D384 ≈ 11.7M = measured upper.
DEFAULTS = dict(n_layer=6, n_embd=320, ctx_len=512, batch_size=8,
                lr_init=6e-4, lr_final=1e-5)


def load_model(vocab_size: int, n_layer: int, n_embd: int, ctx_len: int):
    """Import the patched clone and build the reduced GPT on CUDA."""
    # Even with the wkv kernel cached, torch's load() rewrites its ninja file,
    # which requires a resolvable MSVC environment (`where cl`).
    from spikegpt_env_check import bootstrap_msvc_env
    bootstrap_msvc_env()
    if str(CLONE_DIR) not in sys.path:
        sys.path.insert(0, str(CLONE_DIR))
    cwd = os.getcwd()
    os.chdir(CLONE_DIR)  # wkv extension resolves its sources relative to CWD
    try:
        from src.model import GPT, GPTConfig
    finally:
        os.chdir(cwd)
    cfg = GPTConfig(vocab_size, ctx_len, model_type="RWKV",
                    n_layer=n_layer, n_embd=n_embd)
    return GPT(cfg).cuda()


class CharWindowDataset(torch.utils.data.Dataset):
    """Random-window char sampling, byte-identical in spirit to utils.Dataset.

    Reads text as latin-1 so every BYTE is one symbol -> vocab 205 on enwik8,
    matching the published char-level setup (the repo default 'utf-8' yields
    ~6000 symbols on enwik8 and silently changes embedding size).
    """

    def __init__(self, text: str, ctx_len: int, epoch_length_fixed: int):
        import numpy as np
        raw = np.frombuffer(text.encode("latin-1", errors="replace"),
                            dtype=np.uint8)
        self.vocab_size = int(raw.max()) + 1
        self.codes = raw.astype(np.int64)
        self.ctx_len = ctx_len
        self.epoch_length_fixed = epoch_length_fixed

    def __len__(self):
        return self.epoch_length_fixed

    def __getitem__(self, idx):
        import numpy as np
        i = np.random.randint(0, len(self.codes) - (self.ctx_len + 1))
        dix = self.codes[i:i + self.ctx_len + 1]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def gpu_poller(stop_evt: threading.Event, out: dict) -> None:
    """Sample total GPU memory used (whole card, incl. desktop apps)."""
    peak = 0
    while not stop_evt.is_set():
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=5,
            )
            peak = max(peak, int(q.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        time.sleep(0.5)
    out["nvidiasmi_peak_mib_whole_gpu"] = peak


def make_synthetic_data(path: Path, n_chars: int = 4_000_000) -> None:
    """enwik8-shaped fallback data (~205 distinct chars) for plumbing tests."""
    import numpy as np
    rng = np.random.default_rng(0)
    # byte alphabet restricted to printable range -> ~100 distinct; duplicate a
    # random subset to land near enwik8's 205-symbol vocab
    base = [chr(b) for b in range(32, 127)]
    extra = [chr(b) for b in rng.choice(range(160, 256), size=105, replace=False)]
    alphabet = base + extra
    seq = "".join(rng.choice(alphabet, size=n_chars))
    path.write_text(seq, encoding="utf-8")
    print(f"[data] synthetic {n_chars:,}-char corpus -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "full"], default="dry")
    ap.add_argument("--steps", type=int, default=60,
                    help="dry mode: total steps to run (first ~10 are warmup)")
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--n-layer", type=int, default=DEFAULTS["n_layer"])
    ap.add_argument("--n-embd", type=int, default=DEFAULTS["n_embd"])
    ap.add_argument("--ctx-len", type=int, default=DEFAULTS["ctx_len"])
    ap.add_argument("--lr-init", type=float, default=DEFAULTS["lr_init"])
    ap.add_argument("--lr-final", type=float, default=DEFAULTS["lr_final"])
    ap.add_argument("--data", type=Path, default=None,
                    help="training .txt (default external/enwik8.txt, "
                         "synthetic fallback)")
    ap.add_argument("--epoch-length-fixed", type=int, default=10_000)
    ap.add_argument("--n-epochs-full", type=int, default=200,
                    help="full mode: mini-epochs (plan budget knob)")
    ap.add_argument("--save-every", type=int, default=20,
                    help="full mode: save ckpt every K mini-epochs")
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--label", type=str, default=None)
    args = ap.parse_args()

    import torch

    RESULTS_DIR.mkdir(exist_ok=True)
    data_path = args.data or (STAGE_B_DIR / "external" / "enwik8.txt")
    if not data_path.is_file():
        print(f"[data] {data_path} missing -> generating synthetic fallback "
              "(plumbing-valid only, NOT quality-valid)")
        make_synthetic_data(data_path)

    print(f"[data] loading {data_path} ...")
    t0 = time.time()
    text = data_path.read_text(encoding="latin-1")
    dataset = CharWindowDataset(text, args.ctx_len, args.epoch_length_fixed)
    print(f"[data] {len(dataset.codes):,} chars, vocab {dataset.vocab_size}, "
          f"loaded in {time.time()-t0:.1f}s")

    model = load_model(dataset.vocab_size, args.n_layer, args.n_embd,
                       args.ctx_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.n_layer}L/{args.n_embd}D/T{args.ctx_len}: "
          f"{n_params:,} params ({n_params/1e6:.2f}M)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_init,
                           betas=(0.9, 0.99), eps=4e-9)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0,
                                         drop_last=True, pin_memory=True)

    total_steps = (args.steps if args.mode == "dry"
                   else args.n_epochs_full * args.epoch_length_fixed)
    warmup_tokens = 0
    tokens_per_step = args.batch_size * args.ctx_len
    final_tokens = total_steps * tokens_per_step
    lr_final_factor = args.lr_final / args.lr_init

    def lr_mult(tokens_seen: int) -> float:
        if tokens_seen < warmup_tokens:
            return lr_final_factor + (1 - lr_final_factor) * tokens_seen / max(1, warmup_tokens)
        progress = (tokens_seen - warmup_tokens) / max(1, final_tokens - warmup_tokens)
        return (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)

    poll_out: dict = {}
    stop_evt = threading.Event()
    poller = threading.Thread(target=gpu_poller, args=(stop_evt, poll_out), daemon=True)
    poller.start()

    torch.cuda.reset_peak_memory_stats()
    step_times: list[float] = []
    losses: list[float] = []
    tokens_seen = 0
    it = iter(loader)
    compile_note = "compiled wkv kernel" if os.environ.get("WKV_MODE") != "fallback" else "PURE-TORCH WKV FALLBACK"
    print(f"[run] mode={args.mode} steps={total_steps} ({compile_note})")

    ckpt_dir = args.ckpt_dir or (STAGE_B_DIR / "results" / "ckpts")
    started = time.time()
    from src.spikingjelly.clock_driven import functional as sj_functional
    for step in range(total_steps):
        sj_functional.reset_net(model)  # repo trainer does this every step (L123)
        if step % args.epoch_length_fixed == 0 and step > 0 and args.mode == "full":
            it = iter(loader)
            if step % (args.save_every * args.epoch_length_fixed) == 0:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step,
                            "tokens": tokens_seen},
                           ckpt_dir / f"l1_{args.label or 'run'}_step{step}.pth")
                print(f"[ckpt] saved at step {step}")
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        t_start = time.time()
        lr = args.lr_init * lr_mult(tokens_seen)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t_start
        tokens_seen += tokens_per_step

        losses.append(float(loss.detach()))
        if step >= 10:  # skip JIT/warmup steps from timing stats
            step_times.append(dt)
        if step % 10 == 0:
            recent = step_times[-10:] or [float("nan")]
            print(f"step {step:5d}  loss {losses[-1]:.4f}  "
                  f"lr {lr:.2e}  {tokens_per_step/recent[-1]:,.0f} tok/s")

    stop_evt.set()
    poller.join(timeout=10)
    wall = time.time() - started
    steady = step_times[5:] or step_times  # drop residual warmup tail
    srt = sorted(steady)
    med_dt = srt[len(srt) // 2]
    p90_dt = srt[int(len(srt) * 0.90)]
    slow_steps = sum(1 for d in steady if d > 3 * med_dt)
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "config": {"n_layer": args.n_layer, "n_embd": args.n_embd,
                   "ctx_len": args.ctx_len, "batch_size": args.batch_size,
                   "lr_init": args.lr_init, "lr_final": args.lr_final,
                   "vocab": dataset.vocab_size},
        "params_M": round(n_params / 1e6, 3),
        "wkv_mode": compile_note,
        "steps_timed": len(steady),
        "median_step_s": round(med_dt, 4),
        "p90_step_s": round(p90_dt, 4),
        "slow_steps_gt3x_median": slow_steps,
        "tok_per_s_median": round(tokens_per_step / med_dt, 1),
        "loss_first10_avg": round(sum(losses[:10]) / 10, 4),
        "loss_last10_avg": round(sum(losses[-10:]) / 10, 4),
        "vram_peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 2**20, 1),
        "vram_peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 2**20, 1),
        "vram_nvidiasmi_whole_gpu_peak_mib": poll_out.get(
            "nvidiasmi_peak_mib_whole_gpu"),
        "wall_s": round(wall, 1),
    }
    tag = args.label or f"{args.mode}_{args.n_layer}L{args.n_embd}D_B{args.batch_size}"
    out_path = RESULTS_DIR / f"l1_{tag}_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"[out] {out_path}")

    est_hours_12h_budget = {
        "note": "ESTIMATE — median tok/s extrapolated to plan token budgets",
        "tokens_per_h": round(result["tok_per_s_median"] * 3600),
        "hours_for_100M_token_passes": round(100e6 / (result["tok_per_s_median"] * 3600), 2),
        "hours_for_300M_token_passes": round(300e6 / (result["tok_per_s_median"] * 3600), 2),
    }
    print("[budget]", json.dumps(est_hours_12h_budget, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
