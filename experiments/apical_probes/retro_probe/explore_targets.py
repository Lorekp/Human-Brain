"""Quick target-distribution exploration for the retro-probe (throwaway)."""
import json
from collections import Counter

HERE = "experiments/apical_probes"
man = json.load(open(f"{HERE}/logged_states_apical_cohort_manifest.json"))
rows = man["rows"]
tr = [r for r in rows if r["split"] == "train"]
va = [r for r in rows if r["split"] == "val"]

tr_ws = Counter(r["answer_text"] for r in tr if r["family"] == "who_said")
va_ws = Counter(r["answer_text"] for r in va if r["family"] == "who_said")
print("train speakers (all):", sorted(tr_ws.items(), key=lambda x: -x[1]))
print()
print("val speakers:", sorted(va_ws.items(), key=lambda x: -x[1]))
inter = set(tr_ws).intersection(set(va_ws))
print("\nintersection:", inter)

tr_ans = Counter(r["answer_text"] for r in tr)
va_ans = Counter(r["answer_text"] for r in va)
ov = set(tr_ans).intersection(set(va_ans))
print("\ntrain unique answers:", len(tr_ans), " val unique:", len(va_ans),
      " overlap:", len(ov))
in_vocab = sum(c for a, c in va_ans.items() if a in tr_ans)
print("val answer tokens covered by train vocab:",
      in_vocab, "/", sum(va_ans.values()))

cd_tr = [r["chapter_distance"] for r in tr]
cd_va = [r["chapter_distance"] for r in va]
print("\nchapter_distance train nulls:", sum(1 for c in cd_tr if c is None))
print("chapter_distance train range:",
      min(c for c in cd_tr if c is not None),
      max(c for c in cd_tr if c is not None))
print("chapter_distance val nulls:", sum(1 for c in cd_va if c is None))
print("chapter_distance val range:",
      min(c for c in cd_va if c is not None),
      max(c for c in cd_va if c is not None))

# prompt_bytes as an extra sanity covariate
pb_tr = [r["prompt_bytes"] for r in tr]
pb_va = [r["prompt_bytes"] for r in va]
print("prompt_bytes train range:", min(pb_tr), max(pb_tr))
print("prompt_bytes val range:", min(pb_va), max(pb_va))

# who_said speaker classes with >=2 train support
maj = {a for a, c in tr_ws.items() if c >= 2}
print("\nspeakers with >=2 train occurrences:", sorted(maj), len(maj))
covered = sum(c for a, c in va_ws.items() if a in maj)
print("val who_said items whose speaker has >=2 train support:",
      covered, "/", sum(va_ws.values()))
