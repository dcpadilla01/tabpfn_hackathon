import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e9 = agent_api.load_saved('e009_ewma_longlags.parquet')
out = e9.copy()
for c in out.columns:
    if c in ('household_key','snapshot_day'): continue
    if pd.api.types.is_numeric_dtype(out[c]):
        out[c] = (out[c].astype(np.float64) * 0.3)
print(out.shape, 'scaled numeric cols:', sum(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns))
path = agent_api.save_table(out, 'e020_scaled03.parquet')
print(path)
chk = agent_api.load_saved('e020_scaled03.parquet')
print('reload shape', chk.shape, 'spend_28 head:', chk['spend_28'].head(3).tolist())
print('snapshot days:', sorted(chk.snapshot_day.unique()))
