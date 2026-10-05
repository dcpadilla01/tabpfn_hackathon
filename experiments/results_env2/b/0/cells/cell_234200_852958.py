import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e013_stationary.parquet')
print("shape", t.shape)
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print(len(feats), "features")
print("dtypes:", t.dtypes.value_counts().to_dict())
print("cols:", feats)

tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("merged", df.shape)
print("snap days:", sorted(df.snapshot_day.unique()))
print("rows per snap:", t.groupby('snapshot_day').size().to_dict())

train = df[df.snapshot_day <= 431]
ytr = train['future_spend_4w'].astype(float)
print("target stats:", ytr.describe().round(2).to_dict())

# baseline references on pseudo-val snap 431
pv = train[train.snapshot_day == 431]
print("MAE global median:", round(np.abs(pv.future_spend_4w - ytr.median()).mean(), 3))
for c in feats:
    if '28' in c and t[c].dtype != object and train[c].notna().mean() > 0.9:
        pass

num_cols = [c for c in feats if str(t[c].dtype) not in ('object','category')]
cat_cols = [c for c in feats if str(t[c].dtype) in ('object','category')]
print("num", len(num_cols), "cat", len(cat_cols), cat_cols)

# univariate |corr| with target
cor = {}
for c in num_cols:
    x = train[c].astype(float); ok = x.notna()
    if ok.sum() > 50 and x[ok].std() > 0:
        cor[c] = abs(np.corrcoef(x[ok], ytr[ok])[0,1])
cor = pd.Series(cor).sort_values()
print("\nweakest 25 by |corr|:"); print(cor.head(25).round(4))

# NaN rates overall and at validation snaps
nanr = t[feats].isna().mean()
print("\ntop NaN rates:"); print(nanr.sort_values(ascending=False).head(8).round(3))
val = t[t.snapshot_day >= 459]
print("NaN rates at val snaps (max):", val[feats].isna().mean().max().round(3))

# drift: trend of per-snapshot means across ALL snaps incl validation
dr = {}
for c in num_cols:
    m = t.groupby('snapshot_day')[c].mean().dropna()
    if len(m) >= 10 and m.std() > 0:
        dr[c] = abs(np.corrcoef(m.index.astype(float), m.values)[0,1])
dr = pd.Series(dr).sort_values(ascending=False)
print("\nmost drifting 15 (mean vs snapshot_day):"); print(dr.head(15).round(3))
