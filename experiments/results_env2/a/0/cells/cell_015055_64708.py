
import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
preds = {}
for n in names:
    df = agent_api.load_saved(n + '.parquet')
    preds[n] = df
    print(n, df.shape, list(df.columns))

tt = agent_api.train_targets()
print('train_targets', tt.shape, list(tt.columns))

fv4 = agent_api.load_saved('feats_v4.parquet')
print('feats_v4', fv4.shape, list(fv4.columns)[:40])
