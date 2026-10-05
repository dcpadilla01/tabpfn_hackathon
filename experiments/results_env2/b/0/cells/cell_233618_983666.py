import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
# Correlations of E013 features with target on train rows
tt = agent_api.train_targets()
base = agent_api.load_saved('e013_stationary.parquet')
m = tt.merge(base, on=['household_key','snapshot_day'])
feats = [c for c in base.columns if c not in ('household_key','snapshot_day')]
rows=[]
for c in feats:
    x = m[c]
    if x.dtype.kind not in 'biufc':
        continue
    ok = x.notna()
    if ok.sum() < 100: 
        rows.append((c, np.nan, ok.sum())); continue
    rows.append((c, np.corrcoef(x[ok], m.future_spend_4w[ok])[0,1], ok.sum()))
corr = pd.DataFrame(rows, columns=['feat','corr','n']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(corr.head(30).to_string())
print(corr.tail(15).to_string())
