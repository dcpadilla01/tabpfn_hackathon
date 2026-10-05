
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days, snapshot

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
    print(f"{label:34s} alpha={a} holdMAE={mae:.3f}")
    return mae

B = m[base_cols].astype(float)
run(B, "E003 raw (reference)")

# demographics
snap = snapshot()
demo = snap.demographics.copy()
dd = pd.get_dummies(demo.drop(columns=["household_key"]).astype(str), prefix_sep="=")
dd["household_key"] = demo["household_key"].values
Md = m[["household_key"]].merge(dd, on="household_key", how="left").drop(columns=["household_key"])
Md = Md.astype(float)
Md["has_demo"] = Md.notna().any(axis=1).astype(float)
Md = Md.fillna(0)
run(pd.concat([B, Md], axis=1), "+ demographics dummies")

# winsorize at train 99th pct
caps = B[fitm].quantile(0.99)
Bw = B.clip(upper=caps, axis=1)
run(Bw, "+ winsorized 99pct")

# raw ratios vs usual level
R = pd.DataFrame(index=m.index)
R["r28_usual"] = B["spend_28d"]/B["usual_4w"].replace(0,np.nan)
R["rew_usual"] = B["ew_spend_hl28"]/B["usual_4w"].replace(0,np.nan)
R["rw1_mean6"] = B["spend_w1"]/B["mean_w1_w6"].replace(0,np.nan)
R["r84_364"] = B["spend_84d"]/B["spend_364d"].replace(0,np.nan)
R["r28_364"] = B["spend_28d"]/B["spend_364d"].replace(0,np.nan)
R = R.replace([np.inf,-np.inf], np.nan).fillna(1.0).clip(0,10)
run(pd.concat([B, R], axis=1), "+ raw ratios vs usual")

# within-snapshot ranks of key levels
RK = pd.DataFrame(index=m.index)
for c in ["spend_28d","spend_84d","ew_spend_hl28","usual_4w","spend_364d","days_since_last","trips_28d"]:
    RK["rank_"+c] = B[c].rank(pct=True)
run(pd.concat([B, RK], axis=1), "+ within-snapshot ranks")

# trimmed set: drop late windows w7..w14 and dup spend_364
trim = [c for c in base_cols if c not in [f"spend_w{i}" for i in range(7,15)]+["spend_364"]]
run(B[trim], "trimmed (drop w7-w14, spend_364)")
