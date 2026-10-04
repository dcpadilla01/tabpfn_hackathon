import agent_api, pandas as pd, numpy as np

def ridge_fit_pred(Xtr, ytr, Xva, alpha):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd; Zv = (Xva-mu)/sd
    Z = np.clip(np.c_[np.ones(len(Z)), Z], -50, 50)
    Zv = np.clip(np.c_[np.ones(len(Zv)), Zv], -50, 50)
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return np.clip(Zv@w, 0, None)

keys = ['household_key','snapshot_day']
base = agent_api.load_saved('e009_demo.parquet')  # E009 = current best
tt = agent_api.train_targets()
sd = agent_api.snapshot_days(); tr_days = sd['train']
df = base.merge(tt, on=keys, how='left')
fcols = [c for c in df.columns if c not in keys+['future_spend_4w']]
d = df.snapshot_day.values
trin = df[np.isin(d, [x for x in tr_days if x<431])]; inner = df[d==431]
ytr = trin.future_spend_4w.values; yin = inner.future_spend_4w.values
Xtr = trin[fcols].apply(pd.to_numeric, errors='coerce').fillna(0).values
Xin = inner[fcols].apply(pd.to_numeric, errors='coerce').fillna(0).values
# baseline inner MAE at alpha 3000
p = ridge_fit_pred(Xtr, ytr, Xin, 3000.)
print('E009 inner MAE d431:', np.abs(p-yin).mean())
# feature correlations with target
corr = {}
for i,c in enumerate(fcols):
    v = Xtr[:,i]
    if np.std(v)>0: corr[c] = np.corrcoef(v, ytr)[0,1]
top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:30]
for c,v in top: print(f'{c:22s} {v:.3f}')
