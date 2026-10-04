import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
print("s_day in base:", "s_day" in base.columns)
rb = A.load_saved("nf_robust.parquet"); bc=set(base.columns)
print("nf_robust new:", [c for c in rb.columns if c not in bc and c not in ("household_key","snapshot_day")])
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner").sort_values(["snapshot_day","household_key"])
y = tr.future_spend_4w.values.astype(float)
# drift check
g = tr.groupby("snapshot_day").agg(y_mean=("future_spend_4w","mean"), l1=("spend_l1","mean"), newm4=("newm4","mean"), m123=("spend_l123_mean","mean"))
print(g.round(1))
# persistence MAEs
def mae(p): return np.mean(np.abs(y-p))
for c in ["spend_l1","newm4","spend_l123_mean","ewm13_d","spend_l456_mean","r_med13","peer_ratio"]:
    print(c, round(mae(tr[c].values.astype(float)),2))
# best simple blend (in-sample, indicative)
cands = ["spend_l1","newm4","spend_l123_mean","ewm13_d","spend_l456_mean","r_med13","nspend28"]
P = np.column_stack([tr[c].values.astype(float) for c in cands])
from numpy.linalg import lstsq
w,_,_,_ = lstsq(P, y, rcond=None)
print("blend weights", np.round(w,3), "MAE", round(mae(P@w),2))
print("median pred MAE", round(mae(np.full_like(y, np.median(y))),2))