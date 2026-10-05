import pandas as pd, numpy as np, agent_api
t = agent_api.load_saved('e011_pruned_basket.parquet')
print('E011 table:', t.shape)
print('cols:', list(t.columns))
tt = agent_api.train_targets()
print('targets:', tt.shape)
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','max', lambda s:(s==0).mean()])
g.columns=['mean','median','max','zero_share']
print(g.round(2))
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged:', m.shape)
num = [c for c in t.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(t[c])]
print('n numeric feats:', len(num))
cor = pd.Series({c: m[c].corr(m['future_spend_4w']) for c in num}).sort_values(key=np.abs, ascending=False)
print(cor.head(35).round(3).to_string())
print('weak (|corr|<0.02):', int((cor.abs()<0.02).sum()), 'of', len(cor))
# drift check: mean of a few key features per snapshot
for c in [x for x in num if ('28' in x and 'spend' in x.lower())][:4]:
    print(c, 'per-snapshot mean:', t.groupby('snapshot_day')[c].mean().round(1).to_dict())
print('target describe:', tt.future_spend_4w.describe().round(2).to_dict())
