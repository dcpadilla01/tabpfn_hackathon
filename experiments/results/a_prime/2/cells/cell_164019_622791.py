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