
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
m = df.merge(train_targets(), on=["household_key","snapshot_day"], how="inner")
tr_all = snapshot_days()["train"]
fitm = m.snapshot_day.isin([d for d in tr_all if d<=347]).values
holdm = ~fitm
y = m["future_spend_4w"].values.astype(float)
base_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]

def ridge_eval(Xf, Xh, yf, yh, alphas=(1,3,10,30,100,300,1000,3000,10000)):
    ym = yf.mean(); yf0 = yf - ym
    G = Xf.T@Xf; b = Xf.T@yf0; n = Xf.shape[1]
    best = (None, 1e18)
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(n), b)
        p = Xh@w + ym
        mae = np.mean(np.abs(p-yh))
        if mae < best[1]: best = (a, mae)
    return best

def prep(cols, transform=None):
    X = m[cols].astype(float)
    if transform == "log":
        X = np.log1p(X.clip(lower=0))
    med = pd.Series(X[fitm].median(), index=cols)
    Xf_raw = X[fitm].fillna(med); Xh_raw = X[holdm].fillna(med)
    mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
    return ((Xf_raw-mu)/sd).values, ((Xh_raw-mu)/sd).values, y[fitm], y[holdm]

Xf, Xh, yf, yh = prep(base_cols)
print("raw E003:", ridge_eval(Xf, Xh, yf, yh))
Xf, Xh, yf, yh = prep(base_cols, "log")
print("log E003:", ridge_eval(Xf, Xh, yf, yh))

# log + a few log-ratios
X = m[base_cols].astype(float)
L = np.log1p(X.clip(lower=0))
extra = pd.DataFrame(index=m.index)
extra["lr_w1_w2"] = L["spend_w1"] - L["spend_w2"]
extra["lr_ew_usual"] = L["ew_spend_hl28"] - L["usual_4w"]
extra["lr_rec_long"] = L["spend_84d"] - L["spend_364d"]
extra["lr_w1_usual"] = L["spend_w1"] - L["usual_4w"]
extra["lr_w12_w34"] = L["spend_w1"].add(L["spend_w2"], fill_value=0) - L["spend_w3"].add(L["spend_w4"], fill_value=0)
X2 = pd.concat([L, extra], axis=1)
cols2 = list(X2.columns)
med = pd.Series(X2[fitm].median(), index=cols2)
Xf_raw = X2[fitm].fillna(med); Xh_raw = X2[holdm].fillna(med)
mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
print("log+ratios:", ridge_eval(Xf, Xh, yf, yh))
