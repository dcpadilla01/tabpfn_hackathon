
import pandas as pd, numpy as np, agent_api

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")

# (a) rank-duplicate detection among numeric features
ranks = Xn.rank()
dup_groups = []
cols = list(Xn.columns)
seen = set()
for i, c in enumerate(cols):
    if c in seen: continue
    grp = [c]
    for c2 in cols[i+1:]:
        if c2 in seen: continue
        if ranks[c].equals(ranks[c2]):
            grp.append(c2); seen.add(c2)
    if len(grp) > 1: dup_groups.append(grp)
n_redundant = sum(len(g)-1 for g in dup_groups)
print("rank-duplicate groups:", len(dup_groups), "| redundant cols:", n_redundant)
for g in dup_groups[:15]: print("  ", g)

# (b) churn separation among rows with sp28>0
sp28 = df.sp28.values.astype(float); med4 = df.z_med4w_hist.values.astype(float)
dsl = df.days_since_last.values.astype(float); gapmed = df.gap_med.values.astype(float)
act = sp28 > 0
dorm = dsl / np.maximum(gapmed, 1.0)
print("\nAmong sp28>0 rows (n=%d): y==0 frac %.3f" % (act.sum(), (y[act]==0).mean()))
for name, v in [("dormancy dsl/gapmed", dorm), ("days_since_last", dsl), ("sp28", sp28), ("trend_84", df.trend_84.values)]:
    vv = np.asarray(v, dtype=float)
    print(f"  {name:20s} y==0: mean {vv[act & (y==0)].mean():7.2f} | y>0: mean {vv[act & (y>0)].mean():7.2f}")
# how big is the MAE contribution of zero rows?
print("MAE contribution from y==0 rows (if predict med4w):", round(np.abs(y[(y==0)] - med4[(y==0)]).mean(),1), "n=", (y==0).sum())

# (c) mix stability: cosine between dsh84_* and dsh364_* vectors
d84 = [c for c in df.columns if c.startswith("dsh84_")]
d364 = [c for c in df.columns if c.startswith("dsh364_")]
A = df[d84].fillna(0).values; B = df[d364].fillna(0).values
num = (A*B).sum(1); den = np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1) + 1e-9
mixstab = num/den
print("\nmixstab corr with y:", round(np.corrcoef(mixstab, y)[0,1],3),
      " corr with |y-med4w|:", round(np.corrcoef(mixstab, np.abs(y-med4))[0,1],3))

# (d) market calibration: mean y by snapshot vs week_mod52
print("\nmean y by snapshot:", {int(s): round(y[df.snapshot_day==s].mean(),1) for s in sorted(df.snapshot_day.unique())})
print("week_mod52 by snapshot:", {int(s): int(df.week_mod52[df.snapshot_day==s].iloc[0]) for s in sorted(df.snapshot_day.unique())})
