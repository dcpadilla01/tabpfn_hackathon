import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
df = agent_api.load_saved("candC_hb.parquet")
print(df.shape, "rows ok:", df['household_key'].nunique())
m = tt.merge(df, on=["household_key","snapshot_day"])
new = ['hb_repeat_share_28','hb_rep_prod_frac_28','hb_dept_entropy_84','hb_top1prod_share_84','hb_new_prod_share_28']
print(m[new].describe().T[['mean','std','min','max']])
y = m['future_spend_4w'].values
for c in new:
    print(c, "corr", round(np.corrcoef(m[c].fillna(0), y)[0,1],3))
# also check candAll composition
ca = agent_api.load_saved("candAll.parquet")
print("candAll", ca.shape)
print([c for c in ca.columns if c.startswith(('hb_','su_','td_','g_','dm_'))])
