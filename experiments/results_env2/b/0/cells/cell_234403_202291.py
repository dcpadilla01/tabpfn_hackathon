import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
num = [c for c in feats if str(t[c].dtype) not in ('object','category')]

train = t[t.snapshot_day <= 431]
val = t[t.snapshot_day >= 459]

rows = []
for c in num:
    x_tr = train[c]; x_va = val[c]
    gm = t.groupby('snapshot_day')[c].mean().dropna()
    drift = abs(np.corrcoef(gm.index.astype(float), gm.values)[0,1]) if len(gm)>=10 and gm.std()>0 else np.nan
    # val mean vs late-train mean shift, scaled by pooled std
    sd = t[c].std()
    shift = abs(x_va.mean() - x_tr[x_tr.snapshot_day>=375].mean())/sd if sd>0 else np.nan
    # univariate corr w target
    dfm = t.merge(tt, on=['household_key','snapshot_day'])
    tr = dfm[dfm.snapshot_day<=431]
    ok = tr[c].notna()
    cor = abs(np.corrcoef(tr.loc[ok,c].astype(float), tr.loc[ok,'future_spend_4w'])[0,1]) if ok.sum()>50 and tr.loc[ok,c].std()>0 else 0
    rows.append((c, cor, drift if drift==drift else -1, shift if shift==shift else -1, t[c].isna().mean()))

res = pd.DataFrame(rows, columns=['feat','abs_corr','drift','valshift','nan']).sort_values('drift', ascending=False)
print("Top 20 drifters (drift | corr | val-shift | nan):")
print(res.head(20).round(3).to_string())
print("\nLow-corr features (abs_corr<0.06):")
print(res[res.abs_corr<0.06].round(3).to_string())
print("\nVal shift top:")
print(res.sort_values('valshift',ascending=False).head(12).round(3).to_string())
