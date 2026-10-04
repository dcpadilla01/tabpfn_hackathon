import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
feat = [c for c in F.columns if c not in key+['gbm_pred']]
D = F.merge(tt, on=key)
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]
Xtr = ptr[feat].astype(float)
print('ytr mean:', ptr[TARGET].mean().round(2), 'nan y:', ptr[TARGET].isna().sum())
inf_ct = np.isinf(Xtr.values).sum(axis=0)
bad = [(feat[i], int(c)) for i,c in enumerate(inf_ct) if c>0]
print('cols with inf:', bad[:20], 'total inf cols:', len(bad))
nan_ct = Xtr.isna().sum(); print('cols all-NaN in ptr:', [feat[i] for i,c in enumerate(nan_ct) if c==len(ptr)][:20])
print('max abs values:', Xtr.abs().max().sort_values().iloc[-8:])
print('const cols in ptr:', [c for c in feat if Xtr[c].nunique(dropna=True)<=1][:20])