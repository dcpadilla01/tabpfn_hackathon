import agent_api as api, pandas as pd, numpy as np
t = api.train_targets()
for name in ['micro','nf_candidates','nf_robust','nf_seasonal']:
    df = api.load_saved(name+'.parquet')
    m = df.merge(t, on=['household_key','snapshot_day'])
    num = m.select_dtypes('number')
    c = num.corr()['future_spend_4w'].drop('future_spend_4w')
    print('==',name)
    print(c.abs().sort_values(ascending=False).head(15).round(3).to_dict())
