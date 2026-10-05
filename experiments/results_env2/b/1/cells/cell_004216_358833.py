import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t9  = agent_api.load_saved('e009_target_enc.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt  = agent_api.train_targets()
df9 = t9.merge(tt, on=['household_key','snapshot_day'])
df16 = t16.merge(tt, on=['household_key','snapshot_day'])
y9 = df9['future_spend_4w'].values; y16 = df16['future_spend_4w'].values

te_cols = [c for c in t9.columns if c not in t8_cols] if False else None
t8 = agent_api.load_saved('e008_level_shape.parquet')
te_cols = [c for c in t9.columns if c not in set(t8.columns)]
print('E009 added cols:', te_cols)
for c in te_cols:
    x = df9[c].astype(float).values
    ok = ~np.isnan(x)
    print('%-14s corr=%+.3f  nan%%=%.1f  uniq=%d' % (c, np.corrcoef(x[ok], y9[ok])[0,1], 100*(~ok).mean(), len(np.unique(x[ok]))))

gcols = [c for c in t16.columns if c.startswith('g_')]
print('\ngrid feature corr with target (train rows):')
rows=[]
for c in gcols:
    x = df16[c].astype(float).values; ok=~np.isnan(x)
    rows.append((c, np.corrcoef(x[ok], y16[ok])[0,1], 100*(~ok).mean()))
for c,r,nn in sorted(rows, key=lambda t:-abs(t[1]))[:15]:
    print('%-12s corr=%+.3f nan%%=%.0f' % (c,r,nn))
