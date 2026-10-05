import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved, save_table

F = load_saved("allF.parquet"); C = load_saved("camp_feats.parquet")
camp_cols = [c for c in C.columns if c not in ("household_key","snapshot_day")]
print("camp cols:", camp_cols)
F2 = F.merge(C, on=["household_key","snapshot_day"], how="left")
for c in camp_cols:
    F2[c] = pd.to_numeric(F2[c], errors="coerce").astype(float)
fill = {c: 0.0 for c in camp_cols}; fill["cmp_laststart"] = 9999.0
F2 = F2.fillna(fill)
feats = [c for c in F2.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
print("n_feats", len(feats))

tr = F2[F2.future_spend_4w.notna()]
TRN = tr[tr.snapshot_day <= 375].copy()
VAL = F2[F2.snapshot_day.isin([459,487,515,543])].copy()
print("TRN", TRN.shape, "VAL", VAL.shape, sorted(VAL.snapshot_day.unique()))

def fit_q(seed, rounds=2400, half=140.0):
    w = 0.5 ** ((TRN.snapshot_day.values - 375) / half)
    p = dict(objective="reg:quantileerror", quantile_alpha=0.5, eta=0.03, max_depth=6,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed)
    b = xgb.train(p, xgb.DMatrix(TRN[feats], label=TRN.future_spend_4w.values, weight=w), num_boost_round=rounds)
    return b.predict(xgb.DMatrix(VAL[feats]))

t0 = time.time()
ps = [fit_q(s) for s in (1, 2, 3, 4)]
pred = np.clip(np.mean(ps, axis=0), 0, None)
print("pred stats: mean %.1f std %.1f min %.1f max %.1f  (%.0fs)" % (pred.mean(), pred.std(), pred.min(), pred.max(), time.time()-t0))
assert np.isfinite(pred).all() and len(pred) == len(VAL)
out = pd.DataFrame({"household_key": VAL.household_key.values.astype(int),
                    "snapshot_day": VAL.snapshot_day.values.astype(int),
                    "prediction": pred.astype(float)})
path = save_table(out, "e012_preds")
print("saved", path, out.shape)
