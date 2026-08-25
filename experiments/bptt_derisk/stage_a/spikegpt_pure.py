#!/usr/bin/env python3
"""Pure-PyTorch SpikeGPT inference (no CUDA-kernel compile, no spikingjelly).

Faithful reimplementation of ridgerchu/SpikeGPT src/model.py @ master (archived
repo, fetched 2026-08-25) for loading the released .pth checkpoints:

- Input encoder: GPT.forward applies ATan surrogate with spiking=True ->
  Heaviside(embedding). Verified against vendored
  src/spikingjelly/clock_driven/surrogate.py (class atan.forward -> heaviside).
- Residual structure per Block.forward (layer_id==0 gets ln0 first):
      x = x + lif1( att( ln1(x) ) )
      x = x + lif2( ffn( ln2(x) ) )
  time_shift mixing happens inside att/ffn on their (post-LN) inputs.
- WKV recurrence: line-by-line port of cuda/wkv_cuda.cu kernel_forward
  (p/q/o running state, first-token bonus u, decay exp(w) where the caller
  passes -exp(time_decay)).
- LIF neurons: MultiStepLIFNode(tau=2., surrogate ATan(alpha=2.), v_threshold=1.,
  v_reset=0. default, decay_input=True, hard reset):
      v <- v + (x - v)/2 ;  s <- [v > 1] ;  v <- (1-s)*v
  Forward-only (Heaviside); surrogate gradient irrelevant at inference.
- Channel-mix uses torch.square(torch.relu(key(xk))).

Checkpoint layout verified by peek_checkpoints.py:
  OpenWebText-216M.pth: emb (50277,768), 18 blocks -> 215,399,424 params
  BookCorpus-SpikeGPT.pth: emb (77,768), 24 blocks -> 184,349,184 params
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WKVState:
    """Per-layer RWKV recurrence state: p (num), q (den), o (log-max cache)."""

    __slots__ = ("p", "q", "o")

    def __init__(self, b: int, c: int, device, dtype):
        self.p = torch.zeros(b, c, device=device, dtype=dtype)
        self.q = torch.zeros(b, c, device=device, dtype=dtype)
        self.o = torch.full((b, c), -1e38, device=device, dtype=dtype)


def wkv_step(wt: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             st: WKVState) -> torch.Tensor:
    """One timestep of the verbatim wkv_cuda.cu kernel_forward recurrence.

    wt: (C,) decay, already -exp(time_decay). u: (C,) first-token bonus.
    k, v: (B,C). Returns y: (B,C).
    """
    no = torch.maximum(st.o, u + k)
    aa = torch.exp(st.o - no)
    bb = torch.exp(u + k - no)
    y = (aa * st.p + bb * v) / (aa * st.q + bb)

    no2 = torch.maximum(wt + st.o, k)
    aa = torch.exp(wt + st.o - no2)
    bb = torch.exp(k - no2)
    st.p = aa * st.p + bb * v
    st.q = aa * st.q + bb
    st.o = no2
    return y


def lif_step(v: torch.Tensor, x: torch.Tensor, tau_inv: float = 0.5,
             v_th: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """spikingjelly LIFNode hard-reset step: charge -> fire -> reset."""
    v = v + (x - v) * tau_inv          # decay_input=True, v_reset=0, tau=2
    s = (v > v_th).to(x.dtype)         # Heaviside forward (ATan surrogate grad)
    v = (1.0 - s) * v                  # hard reset to v_reset=0
    return s, v


class Block(nn.Module):
    def __init__(self, n_embd: int, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        att = self.att = nn.Module()
        att.time_decay = nn.Parameter(torch.empty(n_embd))
        att.time_first = nn.Parameter(torch.empty(n_embd))
        att.time_mix_k = nn.Parameter(torch.empty(n_embd))
        att.time_mix_v = nn.Parameter(torch.empty(n_embd))
        att.time_mix_r = nn.Parameter(torch.empty(n_embd))
        att.key = nn.Linear(n_embd, n_embd, bias=False)
        att.value = nn.Linear(n_embd, n_embd, bias=False)
        att.receptance = nn.Linear(n_embd, n_embd, bias=False)
        att.output = nn.Linear(n_embd, n_embd, bias=False)
        ffn = self.ffn = nn.Module()
        ffn.time_mix_k = nn.Parameter(torch.empty(n_embd))
        ffn.time_mix_r = nn.Parameter(torch.empty(n_embd))
        ffn.key = nn.Linear(n_embd, 4 * n_embd, bias=False)
        ffn.receptance = nn.Linear(n_embd, n_embd, bias=False)
        ffn.value = nn.Linear(4 * n_embd, n_embd, bias=False)

    def init_state(self, b: int, device, dtype):
        c = self.ln1.normalized_shape[0]
        return {
            "att_xx": torch.zeros(b, c, device=device, dtype=dtype),
            "ffn_xx": torch.zeros(b, c, device=device, dtype=dtype),
            "wkv": WKVState(b, c, device, dtype),
            "lif1_v": torch.zeros(b, c, device=device, dtype=dtype),
            "lif2_v": torch.zeros(b, c, device=device, dtype=dtype),
        }

    def step(self, x: torch.Tensor, st: dict) -> torch.Tensor:
        """One token through this block. x: (B,C) raw residual-stream input."""
        # ---- attention branch: x + lif1(att(ln1(x))) ----
        h = self.ln1(x)
        xx = st["att_xx"]
        xk = h * self.att.time_mix_k + xx * (1 - self.att.time_mix_k)
        xv = h * self.att.time_mix_v + xx * (1 - self.att.time_mix_v)
        xr = h * self.att.time_mix_r + xx * (1 - self.att.time_mix_r)
        st["att_xx"] = h
        k = self.att.key(xk)
        v = self.att.value(xv)
        r = torch.sigmoid(self.att.receptance(xr))
        wt = -torch.exp(self.att.time_decay.float())
        u = self.att.time_first.float()
        wkv = wkv_step(wt, u, k.float(), v.float(), st["wkv"]).to(h.dtype)
        attn_out = self.att.output(r * wkv)
        s, st["lif1_v"] = lif_step(st["lif1_v"], attn_out)
        x = x + s
        # ---- channel branch: x + lif2(ffn(ln2(x))) ----
        h = self.ln2(x)
        xx = st["ffn_xx"]
        xk = h * self.ffn.time_mix_k + xx * (1 - self.ffn.time_mix_k)
        xr = h * self.ffn.time_mix_r + xx * (1 - self.ffn.time_mix_r)
        st["ffn_xx"] = h
        kk = torch.square(torch.relu(self.ffn.key(xk)))
        kv = self.ffn.value(kk)
        h = torch.sigmoid(self.ffn.receptance(xr)) * kv
        s, st["lif2_v"] = lif_step(st["lif2_v"], h)
        x = x + s
        return x


class SpikeGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 768, n_layer: int = 18):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(Block(n_embd, i) for i in range(n_layer))
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def init_states(self, batch: int, device=None, dtype=None):
        device = device or self.emb.weight.device
        dtype = dtype or self.emb.weight.dtype
        return [blk.init_state(batch, device, dtype) for blk in self.blocks]

    @torch.no_grad()
    def forward_token(self, idx: torch.Tensor, states: list[dict]) -> torch.Tensor:
        """idx: (B,) long tokens at the current timestep. Returns logits (B,V)."""
        x = self.emb(idx)                 # (B,C)
        x = (x > 0).to(x.dtype)           # ATan surrogate, spiking=True == heaviside
        for blk, st in zip(self.blocks, states):
            if blk.layer_id == 0:
                x = blk.ln0(x)
            x = blk.step(x, st)
        x = self.ln_out(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 50,
                 eos_token_id: int | None = None) -> tuple[torch.Tensor, list[int]]:
        """Greedy/temperature decode. prompt_ids: (B,T). Returns (logits_last,
        sampled ids per step (B,) each)."""
        b, _t = prompt_ids.shape
        states = self.init_states(b, prompt_ids.device, self.emb.weight.dtype)
        logits = None
        for pos in range(prompt_ids.shape[1]):
            logits = self.forward_token(prompt_ids[:, pos], states)
        picked = []
        for _ in range(max_new_tokens):
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                lg = logits / temperature
                if top_k and top_k > 0:
                    vals = torch.topk(lg, min(top_k, lg.size(-1))).values[:, [-1]]
                    lg[lg < vals] = float("-inf")
                nxt = torch.multinomial(torch.softmax(lg, -1), 1).squeeze(-1)
            picked.append(nxt)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
            logits = self.forward_token(nxt, states)
        return logits, [p.tolist() for p in picked]


def build_from_state_dict(sd: dict) -> SpikeGPT:
    vocab_size, n_embd = sd["emb.weight"].shape
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    m = SpikeGPT(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer)
    # checkpoint stores time-mix vectors as (1,1,C); our modules hold (C,)
    reshaped = {}
    for k, v in sd.items():
        if "time_mix" in k:
            v = v.reshape(n_embd)
        reshaped[k] = v
    missing, unexpected = m.load_state_dict(reshaped, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    assert not missing, f"missing keys: {missing[:8]}"
    m.eval()
    return m
