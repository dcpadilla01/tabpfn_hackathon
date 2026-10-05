
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

def run(cols_frame, label):
    cols = list(cols_frame.columns)
    med = pd.Series(cols_frame[fitm].median(), index=cols)
    Xf_raw = cols_frame[fitm].fillna(med); Xh_raw = cols_frame[holdm].fillna(med)
    mu = Xf_raw.mean(); sd = Xf_raw.std().replace(0,1)
    Xf = ((Xf_raw-mu)/sd).values; Xh = ((Xh_raw-mu)/sd).values
    a, mae = ridge_eval(Xf, Xh, y[fitm], y[holdm])
    print(f"{label:32s} alpha={a} holdMAE={mae:.3f}")
    return mae

B = m[base_cols].astype(float)
run(B, "E003 raw (reference)")

# 1) calendar features
cal = pd.DataFrame(index=m.index)
cal["sin1"] = np.sin(2*np.pi*m.snapshot_day/364); cal["cos1"] = np.cos(2*np.pi*m.snapshot_day/364)
cal["sin2"] = np.sin(4*np.pi*m.snapshot_day/364); cal["cos2"] = np.cos(4*np.pi*m.snapshot_day/364)
cal["t"] = m.snapshot_day/1000.0
run(pd.concat([B, cal], axis=1), "+ calendar sin/cos/t")

# 2) calendar x recent spend interactions
Xc = pd.concat([B, cal], axis=1)
inter = pd.DataFrame(index=m.index)
for k in ["spend_28d","spend_84d","ew_spend_hl28","usual_4w"]:
    for c in ["sin1","cos1","sin2","cos2"]:
        inter[f"{k}x{c}"] = B[k]*cal[c]
run(pd.concat([B, cal, inter], axis=1), "+ cal + cal x recent spend")

# 3) churn-risk interactions
ch = pd.DataFrame(index=m.index)
ch["recency_x_spend"] = B["days_since_last"]*B["spend_28d"]
ch["days_since_sq"] = B["days_since_last"]**2
ch["inactive28"] = (B["days_since_last"]>28).astype(float)*B["spend_84d"]
ch["inactive56"] = (B["days_since_last"]>56).astype(float)*B["spend_84d"]
ch["inactive84"] = (B["days_since_last"]>84).astype(float)*B["spend_364d"]
run(pd.concat([B, ch], axis=1), "+ churn interactions")

# 4) trend interactions
tr = pd.DataFrame(index=m.index)
tr["w1xw2"] = B["spend_w1"]*B["spend_w2"]
tr["w1xw3"] = B["spend_w1"]*B["spend_w3"]
tr["min_w1w2"] = B[["spend_w1","spend_w2"]].min(axis=1)
tr["max_w1w2"] = B[["spend_w1","spend_w2"]].max(axis=1)
tr["min_w1w2w3"] = B[["spend_w1","spend_w2","spend_w3"]].min(axis=1)
tr["med_w1w2w3"] = B[["spend_w1","spend_w2","spend_w3"]].median(axis=1)
run(pd.concat([B, tr], axis=1), "+ trend products/min/max")
