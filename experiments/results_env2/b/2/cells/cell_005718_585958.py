
import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
key = ["household_key","snapshot_day"]
e011_cols = [c for c in e.columns if c not in key]
base = e.copy()
def newcols(d):
    return [c for c in d.columns if c not in key and c not in e011_cols]
adds = {}
for name in ["hazard_v1","cal2_v1","deal_v1","timing_v1","display_v1"]:
    d = agent_api.load_saved(name + ".parquet")
    adds[name] = newcols(d)
    base = base.merge(d[key+adds[name]], on=key, how="left")

tt = agent_api.train_targets()
mm = base.merge(tt, on=key, how="left")
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, fit_day_max, val_day, alpha=30.0):
    idx = [feats.index(c) for c in cols]
    trf = day <= fit_day_max; trv = day == val_day
    A, yy = Xs[trf][:, idx], y[trf]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[trv][:, idx] @ w, 0, None)
    return np.abs(p - y[trv]).mean()

pairs = [(375,403),(403,431)]
base_cols = e011_cols + adds["hazard_v1"] + adds["cal2_v1"]
# drop-noise test: add deal/timing/display but drop the weakest c2 cols (high NaN, near-duplicates)
drop_sets = {
    "hz+c2": [],
    "hz+c2-tiny": ["c2_f_zero","c2_n_zero","c2_ratio_last","c2_f_med","c2_f_std","c2_corr","c2_resid"],
    "hz+c2-tiny+deal": None,  # filled below
}
cols2 = base_cols + [c for c in adds["deal_v1"] if not c.startswith("redemp")]
drop_sets["hz+c2-tiny+deal"] = drop_sets["hz+c2-tiny"]
for cname, drop in drop_sets.items():
    cols = [c for c in cols2 if c not in drop]
    vals = [ridge_eval(cols, f, v) for f, v in pairs]
    print(f"{cname:18s} n={len(cols):3d} {[round(x,2) for x in vals]} avg={np.mean(vals):.3f}")
