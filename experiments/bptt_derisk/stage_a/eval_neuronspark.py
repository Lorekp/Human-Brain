#!/usr/bin/env python3
"""Stage A: evaluate Brain2nd/NeuronSpark-0.9B on the python_qa 25-item VAL cohort.

Protocol mirrors tools/python_qa_baselines.py metric-for-metric where the
modality allows it:
- prompt framing byte-identical (humanbrain.qa_corpus.qa_prompt, prompt-last,
  ends 'answer:');
- answer_span: teacher-forced next-token prediction across the tokenized
  answer span of every VAL item, conditioned on the serialized prompt
  (token-level analogue of the baselines' word-level span metric);
- free_run: greedy completion, exact-match against the normalized reference;
  budget = ceil((len(ref_words)+8) * TOKEN_BUDGET_MULT) BPE tokens;
- degeneracy: identical distinct-answers/modal-share guards
  (GUARD_DISTINCT_MIN=3, GUARD_MODAL_SHARE_MAX=0.9).

Also records: Chinese sanity completion (validates harness against the card's
own quickstart), tokenizer coverage stats on English code text, config.json
provenance fields verbatim. Writes stage_a/neuronspark_eval.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# ^ NeuronSpark decorates _fused_modulation (pure elementwise math) with
# @torch.compile(backend='inductor'); inductor requires Triton, which does not
# exist on this Windows host. Disabling Dynamo makes the decorator a no-op so
# the function runs eagerly. Documented in FINDINGS.md.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ^ the pure-python Hillis-Steele scan in atomic_ops/parallel_scan.py retains
# per-level buffers; expandable segments avoid fragmentation OOM on 12GB.
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
REPO = STAGE.parents[2]
NS_DIR = REPO / ".models" / "stage_a" / "NeuronSpark-0.9B"
CORPUS = REPO / "experiments" / "python_qa_corpus" / "python_qa_corpus_v1.json"

GUARD_MODAL_SHARE_MAX = 0.9
GUARD_DISTINCT_MIN = 3
WORD_RE = re.compile(r"[a-z0-9_']+")
TOKEN_BUDGET_MULT = 1.5
MAX_CTX = 512          # model context length per card
BATCH = 1              # 0.9B fp32 weights + scan buffers exceed 12GB at B>1 (see FINDINGS)
DTYPE = torch.float32
# ^ bf16 was tried first (halves footprint) but NeuronSpark's shipped forward
# mixes dtypes (fp32 activations into F.linear against bf16 weights ->
# RuntimeError), i.e. the released code only runs fp32 end-to-end. fp32 +
# BATCH=1 + expandable_segments fits in 12GB. Documented in FINDINGS.md.

sys.path.insert(0, str(NS_DIR))


def qa_prompt(context: list[str], question: str) -> str:
    lines = [str(line) for line in context]
    lines.append(f"Question: {question}")
    lines.append("answer:")
    return "\n".join(lines)


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rec: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tok = AutoTokenizer.from_pretrained(str(NS_DIR), trust_remote_code=True)
    print(f"tokenizer loaded: vocab_size={tok.vocab_size} len={len(tok)}", flush=True)
    rec["tokenizer"] = {"vocab_size": tok.vocab_size, "len": len(tok),
                        "bos": getattr(tok, "bos_token", None),
                        "eos": getattr(tok, "eos_token", None)}

    cfg = json.loads((NS_DIR / "config.json").read_text(encoding="utf-8"))
    rec["config_verbatim"] = cfg

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # transformers-version compatibility fix: the remote-code class predates
    # transformers 5.x's _finalize_model_loading refactor (all_tied_weights_keys
    # vs _tied_weights_keys API change). Patch the class before loading.
    # (documented in FINDINGS.md)
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    NSCls = get_class_from_dynamic_module(
        "modeling_neuronspark.NeuronSparkForCausalLM", str(NS_DIR))
    if not hasattr(NSCls, "all_tied_weights_keys"):
        NSCls.all_tied_weights_keys = {}
        rec["compat_fix"] = ("monkeypatched NeuronSparkForCausalLM.all_tied_weights_keys"
                             "={} for transformers 5.9.0 (remote code written for 4.52.0)")
    model = AutoModelForCausalLM.from_pretrained(
        str(NS_DIR), trust_remote_code=True, torch_dtype=DTYPE).to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())
    dtypes = {str(p.dtype) for p in model.parameters()}
    print(f"model on {dev}: {n_params:,} params, dtypes={dtypes}", flush=True)
    rec["model"] = {"device": dev, "params": n_params, "dtypes": sorted(dtypes)}

    # ---- Chinese sanity (card quickstart analogue, greedy) ----
    zh = f"{tok.bos_token}人工智能的发展"
    zh_ids = tok(zh, return_tensors="pt")["input_ids"].to(dev)
    t0 = time.time()
    zh_out = model.generate(zh_ids, max_new_tokens=48, temperature=0.0)
    zh_text = tok.decode(zh_out[0][zh_ids.shape[1]:], skip_special_tokens=True)
    zh_wall = time.time() - t0
    print(f"[sanity zh] {zh_text!r} ({zh_wall:.1f}s)", flush=True)
    rec["chinese_sanity"] = {"prompt": zh, "greedy_continuation": zh_text, "wall_s": round(zh_wall, 1)}

    # ---- corpus ----
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    val_items = corpus["entries"]["VAL"]
    assert len(val_items) == 25, len(val_items)

    # tokenizer coverage on the serialized prompts+answers
    unk_id = tok.unk_token_id
    all_texts = []
    for it in val_items:
        p = qa_prompt(it["context"], it["question"])
        all_texts.append(p + "\n" + it["answer"])
    tot_tok, unk_tok, frag_per_word = 0, 0, []
    for txt in val_items:
        pass
    for it, txt in zip(val_items, all_texts):
        enc = tok(txt)["input_ids"]
        tot_tok += len(enc)
        unk_tok += sum(1 for i in enc if i == unk_id)
        frag_per_word.append(len(tok(it["answer"])["input_ids"]) / max(1, len(words(it["answer"]))))
    rec["coverage"] = {
        "total_tokens_25_items": tot_tok,
        "unk_tokens": unk_tok,
        "unk_rate": unk_tok / max(1, tot_tok),
        "mean_answer_tokens_per_word": sum(frag_per_word) / len(frag_per_word),
    }
    print(f"coverage: {rec['coverage']}", flush=True)

    # ---- teacher-forced answer-span (batched) ----
    # Alignment via fast-tokenizer char offsets: the answer starts at char
    # position len(prompt)+1 (after the '\n'); every token whose span overlaps
    # it belongs to the answer. Robust to BPE merges across the boundary
    # (naive prefix matching aligned 0/25 — see FINDINGS.md).
    encoded = []
    for it in val_items:
        prompt = qa_prompt(it["context"], it["question"])
        text_full = prompt + "\n" + it["answer"]
        enc = tok(text_full, add_special_tokens=False, return_offsets_mapping=True)
        full_ids, offs = enc["input_ids"], enc["offset_mapping"]
        ans_char_start = len(prompt) + 1
        ans_first = next((i for i, (a, b) in enumerate(offs) if b > ans_char_start), None)
        bos = getattr(tok, "bos_token_id", None)
        ids = ([bos] if bos is not None else []) + full_ids
        ans_first = ans_first + (1 if bos is not None else 0) if ans_first is not None else None
        encoded.append({"item": it, "full_ids": ids, "ans_first": ans_first})
    no_align = sum(1 for e in encoded if e["ans_first"] is None)
    print(f"answer-span alignment: {25-no_align}/25 offset-mapped", flush=True)

    rows_span = []
    span_correct = span_total = 0
    t0 = time.time()
    for bi in range(0, len(encoded), BATCH):
        chunk = [e for e in encoded[bi:bi + BATCH] if e["ans_first"] is not None]
        seqs, starts = [], []
        for e in chunk:
            keep = MAX_CTX - (len(e["full_ids"]) - e["ans_first"])
            trimmed = e["full_ids"][-(keep + (len(e["full_ids"]) - e["ans_first"])):]
            new_start = len(trimmed) - (len(e["full_ids"]) - e["ans_first"])
            if new_start < 1:
                trimmed = e["full_ids"]
                new_start = e["ans_first"]
            seqs.append(trimmed)
            starts.append(new_start)
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.zeros((len(seqs), maxlen), dtype=torch.long)   # right pad
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s)
        with torch.no_grad():
            out = model(input_ids=input_ids.to(dev))
        amax = out.logits.float().argmax(-1)                  # (B,T)
        for i, e in enumerate(chunk):
            st = starts[i]
            tgt = seqs[i][st:]
            prd = amax[i, st - 1: st - 1 + len(tgt)].tolist()
            ok = sum(int(a == b) for a, b in zip(prd, tgt))
            span_correct += ok
            span_total += len(tgt)
            rows_span.append({
                "qa_id": e["item"]["qa_id"], "family": e["item"]["family"],
                "answer_tokens": len(tgt), "correct_tokens": ok,
                "pred_decoded": tok.decode(prd)[:160],
                "target_decoded": tok.decode(tgt)[:160],
            })
        print(f"span batch {bi//BATCH+1}/{math.ceil(len(encoded)/BATCH)} done "
              f"({time.time()-t0:.0f}s)", flush=True)
    rec["answer_span"] = {
        "tasks": 25, "predictions": span_total, "correct": span_correct,
        "accuracy": span_correct / max(1, span_total),
        "alignment_failures": no_align,
        "note": ("token-level teacher-forced span accuracy (BPE analogue of the "
                 "baselines' word-level answer_span metric)"),
    }

    # ---- free-run greedy generation ----
    budget_default = lambda ref: max(8, math.ceil((len(ref) + 8) * TOKEN_BUDGET_MULT))
    gen_rows = []
    t0 = time.time()
    for i, e in enumerate(encoded):
        it = e["item"]
        prompt = qa_prompt(it["context"], it["question"])
        pids = tok(prompt, add_special_tokens=False)["input_ids"][-MAX_CTX:]
        budget = budget_default(words(it["answer"]))
        x = torch.tensor([pids], dtype=torch.long).to(dev)
        with torch.no_grad():
            gout = model.generate(x, max_new_tokens=min(budget, 96), temperature=0.0)
        cont = tok.decode(gout[0][len(pids):], skip_special_tokens=True)
        em = normalize(cont) == normalize(it["answer"])
        gen_rows.append({"qa_id": it["qa_id"], "family": it["family"],
                         "expected": it["answer"], "completion": cont,
                         "exact_match": bool(em)})
        print(f"gen {i+1}/25 em={em} ({time.time()-t0:.0f}s): {cont[:90]!r}", flush=True)
    matches = sum(r["exact_match"] for r in gen_rows)
    normed = [normalize(r["completion"]) for r in gen_rows]
    cnt = Counter(normed)
    modal_answer, modal_count = cnt.most_common(1)[0]
    rec["free_run"] = {"tasks": len(gen_rows), "exact_matches": matches,
                       "exact_match_rate": matches / len(gen_rows)}
    rec["degeneracy"] = {
        "evaluated": len(normed), "distinct_answers": len(cnt),
        "modal_share": modal_count / len(normed),
        "guard_pass": len(cnt) >= GUARD_DISTINCT_MIN and modal_count / len(normed) <= GUARD_MODAL_SHARE_MAX,
    }
    rec["tasks_span"] = rows_span
    rec["tasks_free_run"] = gen_rows

    out = STAGE / "neuronspark_eval.json"
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"SUMMARY span_acc={rec['answer_span']['accuracy']:.4f} "
          f"em={matches}/25 degen_guard={rec['degeneracy']['guard_pass']}")


if __name__ == "__main__":
    main()
