import agent_api as A, pandas as pd, numpy as np

def load(name):
    df = A.load_saved(name)
    df.columns = [str(c) for c in df.columns]
    cand = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    pcol = 'prediction' if 'prediction' in df.columns else cand[-1]
    return df[['household_key','snapshot_day',pcol]].rename(columns={pcol:'prediction'})

p8 = load('pred_e008.parquet'); p9 = load('pred_e009.parquet'); p10 = load('pred_e010.parquet')
print(len(p8), len(p9), len(p10))
m = p8.merge(p9, on=['household_key','snapshot_day'], suffixes=('_8','_9')).merge(p10, on=['household_key','snapshot_day'])
print('merged', m.shape, 'nan', m.isna().sum().sum())
m['prediction'] = 0.4*m['prediction_8'] + 0.3*m['prediction_9'] + 0.3*m['prediction']
print(m['prediction'].describe().round(2))
path = A.save_table(m[['household_key','snapshot_day','prediction']], 'pred_e010_blend.parquet')
print(path)
