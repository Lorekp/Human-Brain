#!/usr/bin/env python3
"""Stage A: run both SpikeGPT 216M-class checkpoints on the python_qa VAL prompts.

1. OpenWebText-216M (GPT-NeoX BPE vocab 50277): teacher-forced token-span
   accuracy + greedy free-run completions (qualitative coherence anchor).
2. BookCorpus checkpoint: emb is (77,768) -> CHAR-LEVEL model contradicting
   its HF card ("L18-D768"); actual L24/D768, 184.3M params. The training-time
   vocab.json (utf-16 char->idx map written by src/utils.py Dataset) is NOT in
   the HF repo, so we recover the index->char mapping by testing candidate
   77-char sets (sorted-by-codepoint, mirroring utils.py's
   `unique = sorted(list(set(data)))`) and picking the lowest-NLL candidate on
   held-out English prose. Generation proceeds only if a candidate clearly wins.

Writes stage_a/spikegpt_eval.json.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE))
from spikegpt_pure import build_from_state_dict  # noqa: E402

REPO = STAGE.parents[2]
MODELS = REPO / ".models" / "stage_a"
TOKDIR = REPO / ".models" / "gpt_neox_tokenizer"
CORPUS = REPO / "experiments" / "python_qa_corpus" / "python_qa_corpus_v1.json"

GUARD_MODAL_SHARE_MAX = 0.9
GUARD_DISTINCT_MIN = 3
WORD_RE = re.compile(r"[a-z0-9_']+")
MAX_CTX = 1024


def qa_prompt(context: list[str], question: str) -> str:
    lines = [str(line) for line in context]
    lines.append(f"Question: {question}")
    lines.append("answer:")
    return "\n".join(lines)


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


# Candidate 77-char vocabularies (sorted by Unicode codepoint, as utils.py does)
CHARSETS = {
    # A: canonical cleaned-book set: newline, space, basic punctuation, digits,
    #    upper, lower (77 symbols exactly)
    "A": "".join(sorted("\n !\"&'(),-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")),
    # B: like A but underscore instead of slash
    "B": "".join(sorted("\n !\"&'(),-.0123456789:;?_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")),
    # C: like A plus apostrophe-digit variants: no double-quote, has *
    "C": "".join(sorted("\n !&'()*,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")),
}


def load_model(rel: str) -> object:
    sd = torch.load(MODELS / rel, map_location="cpu", weights_only=True)
    m = build_from_state_dict(sd)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev), dev


def score_nll(model, ids: list[int], device: str) -> float:
    """mean NLL/token of ids[1:] given ids[:-1]"""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    states = model.init_states(1, device)
    nll = count = 0
    logits = None
    for pos in range(x.shape[1]):
        if logits is not None:
            nll += torch.nn.functional.cross_entropy(
                logits.float(), x[:, pos], reduction="sum").item()
            count += 1
        logits = model.forward_token(x[:, pos], states)
    return nll / max(1, count)


def main() -> None:
    rec: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    val_items = corpus["entries"]["VAL"]
    assert len(val_items) == 25
    prompts = [qa_prompt(it["context"], it["question"]) for it in val_items]
    answers = [it["answer"] for it in val_items]

    # ================================================================ OWT ====
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(TOKDIR / "tokenizer.json"))
    model, dev = load_model("SpikeGPT-OpenWebText-216M/SpikeGPT-216M.pth")
    print(f"OWT-216M on {dev}", flush=True)

    encoded = []
    for it, p in zip(val_items, prompts):
        enc = tok(p + "\n" + it["answer"], add_special_tokens=False,
                  return_offsets_mapping=True)
        full_ids, offs = enc["input_ids"], enc["offset_mapping"]
        # answer begins at char len(p)+1 (after '\n'); tokens overlapping it
        # belong to the answer (robust to BPE merges across the boundary)
        s = next((i for i, (_a, b) in enumerate(offs) if b > len(p) + 1), None)
        encoded.append({"item": it, "prompt_ids": tok(p, add_special_tokens=False)["input_ids"],
                        "ans_ids": [], "full_ids": full_ids, "ans_start": s})
    no_align = sum(1 for e in encoded if e["ans_start"] is None)

    span_correct = span_total = 0
    rows_span, prompt_nlls = [], []
    t0 = time.time()
    for i, e in enumerate(encoded):
        pi = e["prompt_ids"][-MAX_CTX:]
        st = model.init_states(1, dev)
        logits = None
        for pos in range(len(pi)):
            logits = model.forward_token(torch.tensor([pi[pos]], device=dev), st)
        prompt_nlls.append(score_nll(model, e["full_ids"][-MAX_CTX:], dev))
        if e["ans_start"] is None:
            rows_span.append({"qa_id": e["item"]["qa_id"], "note": "no-alignment"})
            continue
        keep = MAX_CTX - (len(e["full_ids"]) - e["ans_start"])
        trimmed = e["full_ids"][-(keep + (len(e["full_ids"]) - e["ans_start"])):]
        new_start = len(trimmed) - (len(e["full_ids"]) - e["ans_start"])
        if new_start < 1:
            trimmed = e["full_ids"]
            new_start = e["ans_start"]
        st = model.init_states(1, dev)
        logits = None
        for pos in range(new_start):
            logits = model.forward_token(torch.tensor([trimmed[pos]], device=dev), st)
        tgt = trimmed[new_start:]
        prd = []
        for j in range(len(tgt)):
            prd.append(int(logits.argmax(-1)))
            if j < len(tgt) - 1:
                logits = model.forward_token(torch.tensor([tgt[j]], device=dev), st)
        ok = sum(int(a == b) for a, b in zip(prd, tgt))
        span_correct += ok
        span_total += len(tgt)
        rows_span.append({"qa_id": e["item"]["qa_id"], "family": e["item"]["family"],
                          "answer_tokens": len(tgt), "correct_tokens": ok})
        if (i + 1) % 5 == 0:
            print(f"owt span {i+1}/25 ({time.time()-t0:.0f}s)", flush=True)
    rec["openwebtext_216m"] = {
        "answer_span": {"tasks": 25, "predictions": span_total, "correct": span_correct,
                        "accuracy": span_correct / max(1, span_total),
                        "alignment_failures": no_align},
        "mean_prompt_answer_nll_per_token": sum(prompt_nlls) / len(prompt_nlls),
        "tasks_span": rows_span,
    }
    print(f"OWT span_acc={span_correct/max(1,span_total):.4f} "
          f"mean_nll={sum(prompt_nlls)/len(prompt_nlls):.3f}", flush=True)

    gen_rows = []
    t0 = time.time()
    for i, e in enumerate(encoded):
        it = e["item"]
        pids = tok(qa_prompt(it["context"], it["question"]),
                   add_special_tokens=False)["input_ids"][-MAX_CTX:]
        budget = max(8, math.ceil((len(words(it["answer"])) + 8) * 1.5))
        x = torch.tensor([pids], dtype=torch.long, device=dev)
        _, picked = model.generate(x, max_new_tokens=min(budget, 96), temperature=0.0)
        flat = []
        for step in picked:
            flat.extend(step if isinstance(step, list) else [step])
        cont = tok.decode(flat)
        gen_rows.append({"qa_id": it["qa_id"], "family": it["family"],
                         "expected": it["answer"], "completion": cont,
                         "exact_match": normalize(cont) == normalize(it["answer"])})
        print(f"owt gen {i+1}/25 ({time.time()-t0:.0f}s): {cont[:80]!r}", flush=True)
    cnt = Counter(normalize(r["completion"]) for r in gen_rows)
    modal_answer, modal_count = cnt.most_common(1)[0]
    rec["openwebtext_216m"]["free_run"] = {
        "tasks": 25, "exact_matches": sum(r["exact_match"] for r in gen_rows)}
    rec["openwebtext_216m"]["degeneracy"] = {
        "evaluated": 25, "distinct_answers": len(cnt), "modal_share": modal_count / 25,
        "guard_pass": len(cnt) >= GUARD_DISTINCT_MIN and modal_count / 25 <= GUARD_MODAL_SHARE_MAX}
    rec["openwebtext_216m"]["tasks_free_run"] = gen_rows

    del model
    torch.cuda.empty_cache()

    # ========================================================= BookCorpus ====
    model, dev = load_model("SpikeGPT-BookCorpus/BookCorpus-SpikeGPT.pth")
    print("BookCorpus char-model loaded", flush=True)
    prose = ("The old man walked slowly along the quiet street in the early morning "
             "light. He had lived in this city all his life, and still he found "
             "something new to see every day. The shops were just beginning to open, "
             "and the smell of fresh bread drifted from the bakery on the corner.")
    clean = {c: ord(c) for c in CHARSETS["A"]}
    cand_scores = {}
    for name, chars in CHARSETS.items():
        assert chars == "".join(sorted(chars)), f"charset {name} not sorted"
        stoi = {ch: i for i, ch in enumerate(chars)}
        txt = "".join(ch for ch in prose if ch in stoi)
        ids = [stoi[ch] for ch in txt]
        cand_scores[name] = score_nll(model, ids, dev)
        print(f"charset {name}: nll/char={cand_scores[name]:.4f}", flush=True)
    best = min(cand_scores, key=cand_scores.get)
    rec["bookcorpus"] = {
        "card_claim": "L18-D768 ctx1024 (README.md); artifact is actually emb (77,768), 24 blocks, 184,349,184 params",
        "charset_recovery": {"candidates_nll_per_char": cand_scores, "best": best},
    }

    chars = CHARSETS[best]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    gen_rows_bc, span_ok = [], {"correct": 0, "total": 0}
    for it, p in zip(val_items, prompts):
        txt = "".join(ch for ch in p if ch in stoi)[-MAX_CTX:]
        ids = [stoi[ch] for ch in txt]
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        _, picked = model.generate(x, max_new_tokens=120, temperature=0.0)
        flat = []
        for step in picked:
            flat.extend(step if isinstance(step, list) else [step])
        cont = "".join(itos.get(t, "?") for t in flat)
        gen_rows_bc.append({"qa_id": it["qa_id"], "family": it["family"],
                            "expected": it["answer"], "completion": cont})
    rec["bookcorpus"]["tasks_free_run"] = gen_rows_bc

    out = STAGE / "spikegpt_eval.json"
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
