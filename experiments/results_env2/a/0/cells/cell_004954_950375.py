
import pandas as pd, numpy as np, agent_api
from scipy.optimize import minimize

tt = agent_api.train_targets()
f4 = agent_api.load_saved("feats_v4.parquet")
names = ["pred_e013","pred_e014","pred_e011","pred_e007","pred_e005","pred_e004"]

# build merged table: keys + preds + a few raw feats
base = f4[["household_key","snapshot_day","exp4w_blend","spend_28","spend_84","trips_28","snap_day"]].copy()
for n in names:
    p = agent_api.load_saved(n + ".parquet")[["household_key","snapshot_day","prediction"]].rename(columns={"prediction":n})
    base = base.merge(p, on=["household_key","snapshot_day"], how="inner")
print("merged", base.shape)

train = base.merge(tt, on=["household_key","snapshot_day"])
print("train rows", train.shape, "days", sorted(train.snapshot_day.unique()))
for n in names:
    print(n, "trainMAE", round(np.abs(train[n]-train.future_spend_4w).mean(),3))

# inner holdout: fit stack on days<=403, eval on 431
inner_tr = train[train.snapshot_day <= 403]
inner_va = train[train.snapshot_day == 431]
print("inner tr/va", len(inner_tr), len(inner_va))

def fit_stack(cols, df_fit, l2=0.0, w0=None):
    X = df_fit[cols].values.astype(float)
    y = df_fit.future_spend_4w.values.astype(float)
    k = len(cols)
    if w0 is None: w0 = np.zeros(k); w0[0] = 1.0
    def obj(w):
        r = y - X @ w
        return np.abs(r).mean() + l2 * np.sum(w**2)
    cons = [{"type":"eq","fun": lambda w: w.sum()-1}]
    bnds = [(0,1)]*k
    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter":400,"ftol":1e-9})
    return res.x, res.fun

variants = {
 "A6": names,
 "B6+2": names + ["exp4w_blend","spend_28"],
 "C6+4": names + ["exp4w_blend","spend_28","spend_84","trips_28"],
 "D4": ["pred_e013","pred_e011","pred_e007","exp4w_blend","spend_28"],
 "E_e013only_blend": ["pred_e013","exp4w_blend","spend_28"],
}
for v, cols in variants.items():
    w, f = fit_stack(cols, inner_tr)
    mae_va = np.abs(inner_va.future_spend_4w.values - inner_va[cols].values @ w).mean()
    mae_e013 = np.abs(inner_va.future_spend_4w.values - inner_va["pred_e013"].values).mean()
    print(v, "w=", np.round(w,3), "inner431 MAE", round(mae_va,3), "(e013 alone:", round(mae_e013,3), ")")
