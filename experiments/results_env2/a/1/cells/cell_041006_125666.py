import agent_api as api, pandas as pd, numpy as np
oof = api.load_saved('oof_e5.parquet')
oof['err'] = oof['pred'] - oof['future_spend_4w']
g = oof.groupby('snapshot_day').agg(n=('err','size'), mae=('err', lambda x: np.abs(x).mean()), bias=('err','mean'), act=('future_spend_4w','mean'), pred=('pred','mean'))
print(g.round(2))
print("\nOverall OOF MAE:", np.abs(oof['err']).mean().round(3))
# fit linear bias ~ day on snapshots <=347, apply to 375
tr = oof[oof.snapshot_day<=347]
A = np.vstack([tr.snapshot_day, np.ones(len(tr))]).T
coef, *_ = np.linalg.lstsq(A, tr['err'].values, rcond=None)
print("bias trend coef (per day, intercept):", coef.round(4))
te = oof[oof.snapshot_day==375]
corr_pred = te['pred'] - (coef[0]*375 + coef[1])
print("375 MAE raw:", np.abs(te['err']).mean().round(3), " corrected:", np.abs(corr_pred-te['future_spend_4w']).mean().round(3))
# also quadratic
A2 = np.vstack([tr.snapshot_day**2, tr.snapshot_day, np.ones(len(tr))]).T
c2, *_ = np.linalg.lstsq(A2, tr['err'].values, rcond=None)
corr2 = te['pred'] - (c2[0]*375**2 + c2[1]*375 + c2[2])
print("375 MAE quad-corrected:", np.abs(corr2-te['future_spend_4w']).mean().round(3))
