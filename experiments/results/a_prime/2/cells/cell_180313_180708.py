import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
T = agent_api.train_targets()
base = agent_api.baseline_features()
df = base.merge(T, on=['household_key','snapshot_day'], how='inner')
tr = df[df.snapshot_day <= 375]
print(base.dtypes)
for c in base.columns:
    if c in ('household_key','snapshot_day'): continue
    s = pd.to_numeric(base[c].astype(object), errors='coerce')
    print(c, base[c].dtype, 'nan:', s.isna().sum(), 'max:', np.nanmax(np.abs(s.values.astype(np.float64))) if s.notna().any() else None)
