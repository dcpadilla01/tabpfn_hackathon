
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e005_decay_gapcv.parquet')
print('shape', t.shape)
print('cols', t.columns.tolist())
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
y = m['future_spend_4w']
print(y.describe())
print('zero frac', (y==0).mean(), 'median', y.median())
num = [c for c in t.columns if c not in ('household_key','snapshot_day')]
rows=[]
for c in num:
    x = pd.to_numeric(m[c], errors='coerce')
    ok = x.notna() & y.notna()
    if ok.sum()>100:
        rows.append((c, np.corrcoef(x[ok], y[ok])[0,1]))
cs = pd.DataFrame(rows, columns=['feat','corr']).reindex(pd.Series([r[0] for r in rows])).assign(corr=[r[1] for r in rows]).set_index('feat')['corr']
cs = cs.reindex(cs.abs().sort_values(ascending=False).index)
print(cs.round(3).to_string())
