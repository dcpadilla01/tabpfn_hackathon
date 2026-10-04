import pandas as pd, numpy as np, agent_api as A
base = A.load_saved('e019_final.parquet')
df = base.copy()
demo=['size_ord','income_ord','age_ord','grp_ord','kid_ord']
top=['spend_84','spend_28','wk_avg_8','fwd28_mean','ewma28_4w','spend_lag1','trips_84','spend_364']
for t_ in top:
    v = df[t_].astype(float).values
    for d_ in demo:
        df[f'ix2_{t_}__{d_}'] = v * df[d_].astype(float).values
df['mkt_fwd'] = df.groupby('snapshot_day').fwd28_mean.transform('mean').values
print("shape:", df.shape)
p = A.save_table(df, 'e020_inter_mkt.parquet')
print("saved:", p)