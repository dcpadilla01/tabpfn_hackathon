import pandas as pd, numpy as np

tt = agent_api.train_targets()
print('targets shape', tt.shape)
print(tt['future_spend_4w'].describe())
print('zero frac %.4f' % (tt['future_spend_4w']==0).mean())
print(tt['future_spend_4w'].quantile([0,.05,.1,.25,.5,.75,.9,.95,.99,1]).to_dict())
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median','std'])
g['zerofrac'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda s:(s==0).mean())
print(g)

for name in ['allF','repro_e5','e005_preds','e005_newfeats','e004_features','e004_new','lagfeats','lagfeats2','f_weekly','camp_feats']:
    try:
        df = agent_api.load_saved(name+'.parquet')
        print('===', name, df.shape)
        cols = list(df.columns)
        print(cols[:55])
        if 'snapshot_day' in cols:
            print('snap days:', sorted(df['snapshot_day'].unique().tolist()))
    except Exception as e:
        print(name, 'ERR', repr(e))