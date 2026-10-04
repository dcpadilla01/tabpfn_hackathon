import agent_api, pandas as pd, numpy as np

for name in ['e007_new','e001_history','e004_long_hist','e006_seq_gaps']:
    t = agent_api.load_saved(name + '.parquet')
    print(name, t.shape)
    print(list(t.columns))
    print()

tt = agent_api.train_targets()
print(tt['future_spend_4w'].describe())
print('zero frac:', (tt['future_spend_4w']==0).mean())

m = agent_api.load_saved('e007_new.parquet').merge(tt, on=['household_key','snapshot_day'])
num = m.select_dtypes(include=[np.number]).drop(columns=['snapshot_day'])
cor = num.corr()['future_spend_4w'].drop('future_spend_4w').sort_values()
print(cor)
