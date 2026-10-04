import numpy as np, pandas as pd
t = agent_api.load_saved('e008_main_plus_marketing.parquet')
print(t[['days_since_last','tenure_days','snapshot_day']].dtypes)
print(t['days_since_last'].head(8).tolist(), t['tenure_days'].head(5).tolist())
ts_cols = [c for c in t.columns if c.startswith('ts_') or c.startswith('wl_') or c.startswith('wt_')]
print('temporal cols:', ts_cols)
print(t[ts_cols[:12]].describe().T.round(2))
print('index col?', t['index'].head(3).tolist() if 'index' in t.columns else None)
# check numeric-ness of days_since_last
dsl = pd.to_numeric(t['days_since_last'], errors='coerce')
print('dsl nan share', dsl.isna().mean(), 'range', dsl.min(), dsl.max())