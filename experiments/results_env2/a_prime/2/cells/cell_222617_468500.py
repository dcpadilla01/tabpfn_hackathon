
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e5 = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
df = e5.merge(tt, on=['household_key','snapshot_day'])
TR = [95,123,151,179,207,235,263,291,319,347,375,403,431]
VA = [459,487,515,543]
feats = [c for c in e5.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
tr = df.snapshot_day.isin(TR).values; va = df.snapshot_day.isin(VA).values
X = df[feats].replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median()).values.astype(float)
print('any nan', np.isnan(X).any(), 'any inf', np.isinf(X).any())

def ridge(Xtr, ytr, Xva, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    A = (Xtr-mu)/sd; B = (Xva-mu)/sd
    A = np.c_[np.ones(len(A)), A]; B = np.c_[np.ones(len(B)), B]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ytr)
    return B@w

for name, Xf in [('raw', X), ('log1p', np.log1p(np.clip(X,0,None)))]:
    for lam in [1.0, 10.0, 100.0, 1000.0]:
        p = ridge(Xf[tr], y[tr], Xf[va], lam)
        print('%-6s lam=%6.0f  val MAE %.3f' % (name, lam, np.abs(p-y[va]).mean()))

Xf = np.log1p(np.clip(X,0,None))
p = ridge(Xf[tr], y[tr], Xf[va], 10.0)
out = df.loc[va].copy(); out['pred']=p; out['y']=y[va]
print(out.groupby('snapshot_day')[['y','pred']].agg(['mean','median']).round(1))
# which features have inf/nan in raw
raw = df[feats]
bad = [(c, int(np.isinf(raw[c]).sum()), int(raw[c].isna().sum())) for c in feats if np.isinf(raw[c]).any() or raw[c].isna().any()]
print('bad cols:', bad)
