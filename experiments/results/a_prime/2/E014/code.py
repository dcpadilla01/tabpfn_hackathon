paths = ["churn_seasonality.parquet","behavioral_candidates.parquet","dept_mix_recency.parquet","e013_new_feats.parquet","e012_robust.parquet","e009_ewma_longlags.parquet"]
for p in paths:
    df = agent_api.load_saved(p)
    print("==", p, df.shape)
    cols = list(df.columns)
    print(len(cols), "cols")
    print(cols)
    print("---")

# ---- cell ----
e009 = agent_api.load_saved("e009_ewma_longlags.parquet")
e9cols = set(e009.columns)
for p in ["churn_seasonality.parquet","behavioral_candidates.parquet","dept_mix_recency.parquet","e013_new_feats.parquet","e012_robust.parquet"]:
    df = agent_api.load_saved(p)
    new = [c for c in df.columns if c not in e9cols]
    print("==", p, df.shape, "| new vs E009:", len(new))
    print(new)
    print()

# ---- cell ----
import pandas as pd, numpy as np
e009 = agent_api.load_saved("e009_ewma_longlags.parquet")
new_parts = {
 "behavioral_candidates": agent_api.load_saved("behavioral_candidates.parquet"),
 "e013_new_feats": agent_api.load_saved("e013_new_feats.parquet"),
 "e012_robust": agent_api.load_saved("e012_robust.parquet"),
}
e9cols = set(e009.columns)
add = []
for name, df in new_parts.items():
    for c in df.columns:
        if c in ("household_key","snapshot_day") or c in e9cols: continue
        if c in add: continue
        add.append(c)
print("n new cols:", len(add))

# check keys align
keys = e009[["household_key","snapshot_day"]]
for name, df in new_parts.items():
    m = keys.merge(df, on=["household_key","snapshot_day"], how="left")
    print(name, "rows:", len(df), "merged:", len(m), "nan rows introduced:", int(m[[c for c in add if c in df.columns]].isna().all(axis=1).sum()))

# build merged table
base = e009.copy()
for name, df in new_parts.items():
    cols = ["household_key","snapshot_day"] + [c for c in add if c in df.columns]
    base = base.merge(df[cols], on=["household_key","snapshot_day"], how="left")
print("merged shape:", base.shape)
print("dup cols:", base.columns[base.columns.duplicated()].tolist())

# numeric hygiene
bad = []
for c in add:
    v = base[c]
    if v.dtype.kind in "fi":
        if np.isinf(v.fillna(0)).any(): bad.append((c,"inf"))
        if v.notna().any() and v.fillna(0).abs().max() > 1e7: bad.append((c,"huge",float(v.fillna(0).abs().max())))
print("problem cols:", bad)
print("total nan cols among new:", sum(base[c].isna().any() for c in add))

path = agent_api.save_table(base, "e014_full_pool.parquet")
print(path)

# ---- cell ----
import pandas as pd, numpy as np
e009 = agent_api.load_saved("e009_ewma_longlags.parquet")
tt = agent_api.train_targets()
print(tt.shape, tt.snapshot_day.unique())

# correlation of each candidate with target on train rows
cands = {
 "behavioral_candidates": agent_api.load_saved("behavioral_candidates.parquet"),
 "e013_new_feats": agent_api.load_saved("e013_new_feats.parquet"),
 "e012_robust": agent_api.load_saved("e012_robust.parquet"),
}
e9cols = set(e009.columns)
m = tt.merge(e009[["household_key","snapshot_day"]], on=["household_key","snapshot_day"], how="inner")
y = m.merge(tt, on=["household_key","snapshot_day"], how="left")["future_spend_4w"].values if False else None
# simpler: merge targets directly
for name, df in cands.items():
    cols = [c for c in df.columns if c not in e9cols and c not in ("household_key","snapshot_day")]
    tmp = tt.merge(df[["household_key","snapshot_day"]+cols], on=["household_key","snapshot_day"], how="left")
    print("==", name)
    rows=[]
    for c in cols:
        v = tmp[c]
        corr = np.corrcoef(v.astype(float).fillna(v.median() if v.notna().any() else 0), tmp.future_spend_4w)[0,1] if v.notna().sum()>10 else np.nan
        rows.append((c, round(corr,4), int(v.isna().sum())))
    for r in sorted(rows, key=lambda r: -abs(r[1])):
        print(r)