import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
t8 = agent_api.load_saved('e008_level_shape.parquet')
tt = agent_api.train_targets()
m = t8.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = m.future_spend_4w.values.astype(float)
feats = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
best = []
for c in feats:
    v = pd.to_numeric(m[c], errors='coerce')
    if v.notna().sum() < len(m)*0.5: continue
    v = v.fillna(v.median()).values.astype(float)
    if v.std() < 1e-12: continue
    r = abs(np.corrcoef(v, y)[0,1])
    best.append((r, c))
best.sort(reverse=True)
for r,c in best[:15]: print('%.6f  %s' % (r,c))
