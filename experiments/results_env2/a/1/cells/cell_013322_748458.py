
import pandas as pd, numpy as np
F = agent_api.load_saved('allF.parquet')
print('allF cols:'); print(list(F.columns))
NF = agent_api.load_saved('e005_newfeats.parquet')
print('\ne005_newfeats cols:'); print(list(NF.columns))
print('\nallF contains NF cols?', set(NF.columns)-set(F.columns))

# blend weights of e005 from repro p13/p11
r = agent_api.load_saved('repro_e5.parquet'); p = agent_api.load_saved('e005_preds.parquet')
m = r.merge(p, on=['household_key','snapshot_day'])
A = np.vstack([m.p13, m.p11, np.ones(len(m))]).T
w, res, *_ = np.linalg.lstsq(A, m.prediction, rcond=None)
print('\nblend coefs (p13,p11,intercept):', w, 'resid max', np.abs(A@w - m.prediction).max())

# differences among saved preds
preds = {n: agent_api.load_saved(n+'_preds.parquet').set_index(['household_key','snapshot_day'])['prediction'] for n in ['e003','e004','e005','e007','e008','e009','e010','e001']}
base = preds['e005']
for n, s in preds.items():
    d = (s - base).abs()
    print(n, 'mean|diff vs e005| = %.3f  corr=%.4f' % (d.mean(), np.corrcoef(s, base)[0,1]))
