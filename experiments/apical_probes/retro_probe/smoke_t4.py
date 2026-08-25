"""Smoke-test T4 plumbing: r0c import, SpanScorer, majority-prior control."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
R0C_PATH = REPO / "experiments" / "r0c_probes" / "probes.py"
import importlib.util

spec = importlib.util.spec_from_file_location("r0c_probes_retro", R0C_PATH)
r0c = importlib.util.module_from_spec(spec)
sys.modules["r0c_probes_retro"] = r0c
spec.loader.exec_module(r0c)
print("SPAN_PREDICTIONS:", r0c.SPAN_PREDICTIONS)

freeze_dir = Path(os.environ.get("LOCALAPPDATA",
                                 str(Path.home() / "AppData" / "Local"))) \
    / "Temp" / "hb_bible_freeze"
ev = freeze_dir / "evidence"
train_items = json.loads(
    (ev / "qa_manifest_train.json").read_text(encoding="utf-8"))["items"]
val_items = json.loads(
    (ev / "qa_manifest_val.json").read_text(encoding="utf-8"))["items"]
print("train items:", len(train_items), "val items:", len(val_items))

classes = r0c.answer_classes(train_items)
class_names = sorted(classes, key=classes.get)
print("n answer classes:", len(classes))
prior_ranking = r0c.majority_ranking(train_items, classes)
scorer = r0c.SpanScorer(val_items, r0c.SPAN_PREDICTIONS)
scored = scorer.score([prior_ranking] * len(val_items))
print("majority prior reproduced:", repr(scored["accuracy"]),
      "correct:", scored["correct"])
committed = json.loads((HERE.parent / "probes_results.json")
                       .read_text(encoding="utf-8"))
mp = [v["controls"]["majority_prior_within_answer_vocab"]["accuracy"]
      for v in committed["GA3_per_arm"].values()][0]
print("committed majority prior:   ", repr(mp))
print("MATCH:", scored["accuracy"] == mp)

Y_train = [r0c._soft_targets(r0c.words(it["answer"]), classes)
           for it in train_items[:3]]
print("soft target rows ok:", [float(sum(y)) for y in Y_train])
