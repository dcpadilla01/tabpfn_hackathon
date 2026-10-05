
import agent_api as A
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression

oof = A.load_saved("oof_e016_cv.parquet")
oof8 = A.load_saved("oof_e008.parquet")[["household_key","snapshot_day","oof_sq","oof_med","oof_log"]]
oof = oof.merge(oof8, on=["household_key","snapshot_day"], how="left")
y = oof["y"].values
comps = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_sq","oof_med","oof_log"]
X = oof[comps].fillna(0).values

def mae(p): return np.abs(p-y).mean()
print("best single hgbq_all:", mae(oof.hgbq_all))

# OLS stacking (non-negative, no intercept)
lr = LinearRegression(positive=True, fit_intercept=False).fit(X, y)
print("stack w:", dict(zip(comps, lr.coef_.round(3))))
print("stack MAE:", mae(lr.predict(X)))

# household aggregation: does averaging OOF preds per household reduce MAE?
for c in ["med_v3","hgbq_all","med_all"]:
    g = oof.groupby("household_key")[c].transform("mean")
    print("hh-mean of", c, "MAE:", round(mae(g),3))

# blend of household-mean and per-row
for c in ["hgbq_all","med_v3"]:
    g = oof.groupby("household_key")[c].transform("mean")
    for w in [0.2,0.3,0.4]:
        print(f"({w}hhmean+{1-w}{c}) {c}:", round(mae(w*g+(1-w)*oof[c]),3))

# top-bucket check: shrink/expansion factor on high preds
p = oof.hgbq_all.values
for f in [0.9,0.95,1.0,1.05,1.1]:
    mask = p>300
    q = p.copy(); q[mask]*=f
    print(f"scale top(>300) by {f}: MAE {mae(q):.3f} (n_top={mask.sum()})")
