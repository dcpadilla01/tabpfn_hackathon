
import pandas as pd, numpy as np

feats = load_saved('feats_v4.parquet')
print('feats_v4', feats.shape)
print(list(feats.columns))

names = ['pred_e005','pred_e007','pred_e011','pred_e013','pred_e014','pred_e015','pred_e008','pred_e012']
preds = {}
for nm in names:
    try:
        p = load_saved(nm + '.parquet')
        preds[nm] = p
        print(nm, p.shape, 'mean=%.1f med=%.1f' % (p['prediction'].mean(), p['prediction'].median()))
    except Exception as e:
        print(nm, 'ERR', type(e).__name__)

w = None
for nm, p in preds.items():
    q = p[['household_key','snapshot_day','prediction']].rename(columns={'prediction': nm})
    w = q if w is None else w.merge(q, on=['household_key','snapshot_day'])
print('wide preds', w.shape)
print(w.drop(columns=['household_key','snapshot_day']).corr().round(3).to_string())

tt = train_targets()
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']).round(1).to_string())
