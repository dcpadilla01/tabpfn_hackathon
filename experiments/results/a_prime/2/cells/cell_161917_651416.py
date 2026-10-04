import agent_api, pandas as pd, numpy as np
def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]
y = tr.future_spend_4w.values
# candidate single/composite predictors (all same scale)
cands = {
 'spend_28': tr.spend_28.values,
 'tlag_mean': tr.tlag_mean.values,
 'ewma_4': tr.ewma_4.values,
 'blend_s28_tlmean': 0.5*tr.spend_28.values+0.5*tr.tlag_mean.values,
 'blend_s28_tl_ewma': 0.4*tr.spend_28.values+0.3*tr.tlag_mean.values+0.3*tr.ewma_4.values,
 'mean_tlag_nonzero': (tr.tlag_mean.values),
}
for k,v in cands.items():
    print(k, 'MAE:', round(mae(y, v),2))
# tlag window definition check: correlation matrix of tlags with target
tc = ['tlag_2','tlag_3','tlag_4','tlag_5','tlag_6','tlag_7','tlag_8']
print(tr[tc+['spend_28','future_spend_4w']].corr()['future_spend_4w'].round(3))
# among active rows only
a = tr[tr.spend_28>0]
ya = a.future_spend_4w.values
print('--- active rows ---')
for k,v in {'spend_28':a.spend_28.values,'tlag_mean':a.tlag_mean.values,
            'blend':0.5*a.spend_28.values+0.5*a.tlag_mean.values}.items():
    print(k, round(mae(ya,v),2))
# zero28 rows
z = tr[tr.spend_28==0]
print('--- zero28 rows ---')
print('MAE 0:', round(mae(z.future_spend_4w, np.zeros(len(z))),2),
      'MAE tlag_mean:', round(mae(z.future_spend_4w, z.tlag_mean.values),2),
      'MAE 0.3*tlag_mean:', round(mae(z.future_spend_4w, 0.3*z.tlag_mean.values),2))
# share of target==0 among active
print('share target==0 among active:', (a.future_spend_4w==0).mean().round(3))
print('share target==0 among zero28:', (z.future_spend_4w==0).mean().round(3))
