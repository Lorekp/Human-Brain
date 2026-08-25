"""Pre-flight: state variability, ranges, and single-fit timing (throwaway)."""
import json
import time

import numpy as np

NPZ = (r"C:/Users/laure/Desktop/DeepSim/.worktrees/"
       r"human-brain-a013-cognition-design/experiments/apical_probes/"
       r"logged_states_apical_cohort.npz")
z = np.load(NPZ)

for k in ("train_boundary_voltage", "train_boundary_apical",
          "val_boundary_voltage", "val_boundary_apical"):
    a = z[k]
    print(f"{k}: shape={a.shape} min={a.min():.4g} max={a.max():.4g} "
          f"mean={a.mean():.4g} std={a.std():.4g} nan={np.isnan(a).sum()}")

# item-to-item variability per arm (arm 0 and best-GA1 arm index 16 = A50_0.75_0.6)
man = json.load(open("experiments/apical_probes/logged_states_apical_cohort_manifest.json"))
arms = man["arm_names"]
print("\narms:", arms)
for name in ("train_boundary_apical", "train_boundary_voltage"):
    for ai in (0, 16):
        a = z[name][:, ai, :].astype(np.float64)
        # mean pairwise distance as global variability proxy
        var_across_items = a.std(axis=0).mean()
        print(f"{name} arm={arms[ai]}: mean per-cell across-item std="
              f"{var_across_items:.4g}")

# how different are arms from each other?
d = (z["train_boundary_apical"][:, 0, :] - z["train_boundary_apical"][:, 16, :])
print("apical arm0 vs arm16 mean abs diff:", float(np.abs(d).mean()))

from sklearn.linear_model import LogisticRegression
X = z["train_boundary_apical"][:, 16, :].astype(np.float64)
mu, sd = X.mean(0), X.std(0)
sd = np.where(sd > 0, sd, 1.0)
Xs = (X - mu) / sd
y = np.array([0 if i < 70 else 1 if i < 140 else 2 for i in range(200)])
t0 = time.perf_counter()
m = LogisticRegression(C=0.01, max_iter=1000).fit(Xs, y)
t1 = time.perf_counter()
print(f"\none multinomial logistic fit (200x1024, 3 classes): "
      f"{t1-t0:.3f}s n_iter={m.n_iter_}")
t0 = time.perf_counter()
Xva = (z["val_boundary_apical"][:, 16, :].astype(np.float64) - mu) / sd
_ = m.predict(Xva)
print(f"predict: {time.perf_counter()-t0:.4f}s")
