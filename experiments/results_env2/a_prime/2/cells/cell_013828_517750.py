
import pandas as pd
m = agent_api.load_saved('e020_final_dims.parquet')
print('rows', len(m), 'cols', m.shape[1])
print('has index col:', 'index' in m.columns)
print('object cols:', [c for c in m.columns if m[c].dtype == object])
print('all-NaN cols:', [c for c in m.columns if m[c].isna().all()])
print('snapshot days:', sorted(m.snapshot_day.unique()))
print('n households:', m.household_key.nunique())
