import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))
med_g = float(np.median(y))
lag1 = df["spend_4w_lag1"].fillna(med_g).values
te_hh = df["te_hh_mean"].fillna(df["spend_4w_lag1"]).fillna(med_g).values

# how much does te_hh_mean improve over lag1? and does blending help?
print("lag1:", round(mae(lag1),2), " te_hh_mean:", round(mae(te_hh),2))
for w in [0,0.25,0.5,0.75,1.0]:
    print(f"blend te/lag1 w={w}:", round(mae(w*te_hh+(1-w)*lag1),2))

# E011's oh_* features as single predictors (numeric only)
e11 = A.load_saved("e011_outcomehist.parquet")
e11cols = [c for c in e11.columns if c.startswith("oh_")]
d11 = df.merge(e11[["household_key","snapshot_day"]+e11cols], on=["household_key","snapshot_day"], how="left")
for c in e11cols:
    x = d11[c].astype(float)
    m = x.notna()
    if m.sum()>100:
        print(f"{c:22s} corr={np.corrcoef(x[m], y[m])[0,1]:.3f}  mae_single={mae(x.fillna(df['spend_4w_lag1']).fillna(med_g)):.2f}")
