import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))
med_g = float(np.median(y))

# E011's outcome-history features already exist; load and compare
e11 = A.load_saved("e011_outcomehist.parquet")
e11cols = [c for c in e11.columns if c not in set(te.columns)|{"household_key","snapshot_day"}]
print("E011 added cols:", e11cols)
d11 = df.merge(e11[["household_key","snapshot_day"]+e11cols], on=["household_key","snapshot_day"], how="left")
for c in e11cols:
    x = d11[c].astype(float)
    m = x.notna()
    print(f"{c:22s} corr={np.corrcoef(x[m], y[m])[0,1]:.3f}  mae_single={mae(x.fillna(df['spend_4w_lag1']).fillna(med_g)):.2f}")
