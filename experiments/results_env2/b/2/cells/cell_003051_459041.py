
import pandas as pd, numpy as np
e011 = load_saved('e011_table.parquet')
print('e011', e011.shape)
print(e011.columns.tolist())
tt = train_targets()
print('targets', tt.shape)
print(tt.groupby('snapshot_day')['future_spend_4w'].agg([('mean','mean'),('med','median'),('zero',lambda s:(s==0).mean())]))
for name in ['deal_v1','hazard_v1','timing_v1','selfcal_v1','rfm_traj_v1','season_demo_v1','twin_v1','display_v1','lvl_v1','union_all_v1','combined_v1','e013_table','churn_vol_v1']:
    try:
        df = load_saved(name + '.parquet')
        print('---', name, df.shape)
        print(df.columns.tolist())
    except Exception as e:
        print('---', name, 'ERR', type(e).__name__)
