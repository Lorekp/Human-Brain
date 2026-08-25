"""Benchmark r0c.train_softmax_logistic at T4 shapes."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

from retro_probe import DEFAULT_NPZ, load_r0c

r0c = load_r0c()
z = np.load(DEFAULT_NPZ)
tr = z["train_boundary_apical"][:, 7, :].astype(np.float64)  # ref arm
mu, sd = tr.mean(0), tr.std(0)
sd = np.where(sd > 1e-12, sd, 1.0)
Xs = (tr - mu) / sd

ev = Path(os.environ["LOCALAPPDATA"]) / "Temp" / "hb_bible_freeze" / "evidence"
train_items = json.loads(
    (ev / "qa_manifest_train.json").read_text(encoding="utf-8"))["items"]
classes = r0c.answer_classes(train_items)
Y = np.stack([r0c._soft_targets(r0c.words(it["answer"]), classes)
              for it in train_items])
print("X", Xs.shape, "Y", Y.shape, flush=True)

t0 = time.perf_counter()
W, b = r0c.train_softmax_logistic(Xs, Y)
print(f"fit time: {time.perf_counter()-t0:.1f}s", flush=True)
