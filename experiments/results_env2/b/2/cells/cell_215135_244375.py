
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
tt = train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
days = snapshot_days(); tr_all = days["train"]
fit_days = [d for d in tr_all if d <= 347]
hold_days = [d for d in tr_all if d > 347]
print("fit days:", fit_days, "hold days:", hold_days, "rows:", len(m))

num_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[num_cols].astype(float)
fitm = m.snapshot_day.isin(fit_days).values
holdm = m.snapshot_day.isin(hold_days).values
y = m["future_spend_4w"].values.astype(float)

med = pd.Series(X[fitm].median(), index=num_cols)
X = X.fillna(med)
mu = X[fitm].mean(); sd = X[fitm].std().replace(0,1)
Xs = ((X-mu)/sd).values
Xf, yf = Xs[fitm], y[fitm]; Xh, yh = Xs[holdm], y[holdm]

def ridge(Xf, yf, Xh, yh, alphas=(3,10,30,100,300)):
    G = Xf.T@Xf; b = Xf.T@yf; n = Xf.shape[1]
    best = None
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(n), b)
        mae = np.mean(np.abs(Xh@w - yh))
        if best is None or mae < best[1]: best = (a, mae, w)
    return best

a, mae, w = ridge(Xf, yf, Xh, yh)
print(f"E003 proxy ridge: alpha={a} holdout MAE={mae:.3f}  (harness val MAE 61.131)")
print("naive spend_28d holdout MAE:", round(np.mean(np.abs(m.loc[holdm,'spend_28d'].values - yh)),3))
print("naive ew_spend_hl28 holdout MAE:", round(np.mean(np.abs(m.loc[holdm,'ew_spend_hl28'].values - yh)),3))
print("target std:", round(y.std(),2), "mean:", round(y.mean(),2))
