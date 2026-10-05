import agent_api as A, pandas as pd, numpy as np
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
print(W.columns.tolist())
m = F.merge(W, on=['household_key','snapshot_day'], how='left')
tr = m[m.future_spend_4w.notna()]
for a,b in [('s1','wk_1'),('s2','wk_2'),('s3','wk_3'),('s4','wk_4'),('s2','lag28_56'),('spend_28','wk_1'),('spend_28','s1'),('spend_56','s2')]:
    print(a,b,'corr',round(tr[a].corr(tr[b]),4), 'meanabsdiff', round((tr[a]-tr[b]).abs().mean(),3))
print(m[['s1','s2','s3','s4','wk_1','wk_2','lag28_56','spend_28','spend_56']].describe().round(2).to_string())
print(m.groupby('snapshot_day')[['wk_1','lag28_56','sp28_yag']].mean().round(2).to_string())
