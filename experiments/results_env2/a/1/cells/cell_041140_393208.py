import agent_api as api, pandas as pd, numpy as np
oof = api.load_saved('oof_e5.parquet')
te = oof[oof.snapshot_day==375].copy()
tr = oof[oof.snapshot_day<=347].copy()
res = tr['future_spend_4w'] - tr['pred']
# MAE-optimal constant shift = median of residuals
sh = np.median(res)
print("median residual (const shift):", round(sh,2))
print("375 raw MAE:", round(np.abs(te['err']).mean(),3))
print("375 const-shift MAE:", round(np.abs(te['pred']+sh-te['future_spend_4w']).mean(),3))
# multiplicative: pred*k + b, optimize on tr via grid on k, median for b
best=None
for k in np.arange(0.9,1.6,0.02):
    r = tr['future_spend_4w'] - k*tr['pred']
    b = np.median(r)
    mae_tr = np.abs(k*tr['pred']+b-tr['future_spend_4w']).mean()
    if best is None or mae_tr<best[0]: best=(mae_tr,k,b)
print("best affine on tr:", best)
k,b = best[1],best[2]
print("375 affine MAE:", round(np.abs(k*te['pred']+b-te['future_spend_4w']).mean(),3))
# per-snapshot-day shift from OOF of that same day would be cheating; but shift from PREVIOUS day only:
# use shift estimated on snapshot 347 OOF alone, apply to 375
r347 = oof[oof.snapshot_day==347]
sh347 = np.median(r347['future_spend_4w']-r347['pred'])
print("shift from 347 only:", round(sh347,2), "-> 375 MAE:", round(np.abs(te['pred']+sh347-te['future_spend_4w']).mean(),3))
# avg of last 3 days' shifts
shs = [np.median(oof[oof.snapshot_day==d]['future_spend_4w']-oof[oof.snapshot_day==d]['pred']) for d in [291,319,347]]
print("shifts:", [round(s,1) for s in shs], "avg:", round(np.mean(shs),2), "-> 375 MAE:", round(np.abs(te['pred']+np.mean(shs)-te['future_spend_4w']).mean(),3))
