
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e5 = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
df = e5.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e5.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
itr = (df.snapshot_day<=347).values
iva = (df.snapshot_day>=375).values

Xraw = df[feats].replace([np.inf,-np.inf], np.nan)
tr_med = Xraw[itr].median()
X = Xraw.fillna(tr_med).fillna(0.0).values.astype(float)
print('nan left:', np.isnan(X).any())

def ridge(Xtr, ytr, Xva, lam, clip=8.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd<1e-8]=1.0
    A = np.clip((Xtr-mu)/sd, -clip, clip); B = np.clip((Xva-mu)/sd, -clip, clip)
    A = np.c_[np.ones(len(A)), A]; B = np.c_[np.ones(len(B)), B]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ytr)
    return B@w

for name, Xf in [('raw', X), ('log1p', np.log1p(np.clip(X,0,None)))]:
    for lam in [1.0, 10.0, 100.0]:
        p = ridge(Xf[itr], y[itr], Xf[iva], lam)
        print('%-6s lam=%5.0f  pseudo-val MAE %.3f' % (name, lam, np.abs(p-y[iva]).mean()))

Xf = np.log1p(np.clip(X,0,None))
p = ridge(Xf[itr], y[itr], Xf[iva], 10.0)
out = df.loc[iva].copy(); out['pred']=p
print(out.groupby('snapshot_day')[['future_spend_4w','pred']].agg(['mean','median']).round(1))
print('val mean y %.1f  mean pred %.1f' % (y[iva].mean(), p.mean()))
