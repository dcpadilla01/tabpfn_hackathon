import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
num = [c for c in feats if str(t[c].dtype) not in ('object','category')]

train = t[t.snapshot_day <= 431]
val = t[t.snapshot_day >= 459]
late = train[train.snapshot_day >= 375]
dfm = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr = dfm[dfm.snapshot_day <= 431]

rows = []
for c in num:
    gm = t.groupby('snapshot_day')[c].mean().dropna()
    drift = abs(np.corrcoef(gm.index.astype(float), gm.values)[0,1]) if len(gm)>=10 and gm.std()>0 else -1
    sd = t[c].std()
    shift = abs(val[c].mean() - late[c].mean())/sd if sd and sd>0 else -1
    ok = tr[c].notna()
    cor = abs(np.corrcoef(tr.loc[ok,c].astype(float), tr.loc[ok,'future_spend_4w'])[0,1]) if ok.sum()>50 and tr.loc[ok,c].std()>0 else 0
    rows.append((c, cor, drift, shift, t[c].isna().mean()))

res = pd.DataFrame(rows, columns=['feat','abs_corr','drift','valshift','nan']).sort_values('drift', ascending=False)
print("Top 20 drifters:")
print(res.head(20).round(3).to_string())
print("\nLow-corr (abs_corr<0.06):")
print(res[res.abs_corr<0.06].round(3).to_string())
print("\nTop val-shift:")
print(res.sort_values('valshift',ascending=False).head(12).round(3).to_string())
