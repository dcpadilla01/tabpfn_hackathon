
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

df = load_saved("rfm_cadence_v1.parquet")
print("E003 table:", df.shape)
print(df.dtypes.value_counts())
tt = train_targets()
days = snapshot_days(); tr, va = days["train"], days["validation"]
m = df.merge(tt, on=["household_key","snapshot_day"], how="left")
print("unmatched:", m["future_spend_4w"].isna().sum())
y = m["future_spend_4w"].values
print("target mean/median/pct0:", np.nanmean(y).round(2), np.nanmedian(y), round(np.nanmean(y==0),3))
print("mean target by snapshot day:")
print(m.groupby("snapshot_day")["future_spend_4w"].mean().round(1))

cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("category")]
print("cat cols:", cat_cols)
num_cols = [c for c in df.columns if c not in cat_cols + ["household_key","snapshot_day"]]
print("num cols:", num_cols)

Xn = m[num_cols].copy()
trm = m.snapshot_day.isin(tr).values; vam = m.snapshot_day.isin(va).values
med = Xn[trm].median()
Xn = Xn.fillna(med)
Xd = pd.get_dummies(m[cat_cols].astype(str)) if cat_cols else pd.DataFrame(index=m.index)
X = pd.concat([Xn, Xd.astype(float)], axis=1).astype(float)
mu = X[trm].mean(); sd = X[trm].std().replace(0,1)
Xs = ((X-mu)/sd).fillna(0).values
Xtr, ytr = Xs[trm], y[trm]; Xva, yva = Xs[vam], y[vam]
print("ridge proxy (harness E003 MAE=61.131):")
for a in [1, 10, 100, 1000]:
    w = np.linalg.solve(Xtr.T@Xtr + a*np.eye(Xtr.shape[1]), Xtr.T@ytr)
    print(" alpha", a, "val MAE", round(np.mean(np.abs(Xva@w - yva)),3))
# naive predictors
for c in num_cols:
    if "spend" in c and ("28" in c or "w1" in c):
        print("naive", c, round(np.mean(np.abs(m.loc[vam,c].values - yva)),3))
cors = []
for j in range(Xtr.shape[1]):
    if Xtr[:,j].std() > 0:
        cors.append((X.columns[j], np.corrcoef(Xtr[:,j], ytr)[0,1]))
cors.sort(key=lambda t: -abs(t[1]))
print("top |corr| with target:")
for n,c in cors[:20]: print("  ", n, round(c,3))
