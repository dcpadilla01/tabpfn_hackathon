
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

def ridge(Xtr, ytr, Xva, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    A = (Xtr-mu)/sd; B = (Xva-mu)/sd
    A = np.c_[np.ones(len(A)), A]; B = np.c_[np.ones(len(B)), B]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ytr)
    return B@w

X = df[feats].fillna(df[feats].median()).values.astype(float)
for name, Xf in [('raw', X), ('log1p', np.log1p(np.clip(X,0,None))), ('log-neg-kept', np.sign(X)*np.log1p(np.abs(X)))]:
    for lam in [1.0, 10.0, 100.0]:
        p = ridge(Xf[tr], y[tr], Xf[va], lam)
        ptr = ridge(Xf[tr], y[tr], Xf[tr], lam)
        print('%-12s lam=%6.1f  val MAE %.3f  train MAE %.3f' % (name, lam, np.abs(p-y[va]).mean(), np.abs(ptr-y[tr]).mean()))

# per-snapshot mean of target vs prediction bias check with best config
Xf = np.log1p(np.clip(X,0,None))
p = ridge(Xf[tr], y[tr], Xf[va], 10.0)
out = df.loc[va].copy(); out['pred']=p; out['y']=y[va]
print(out.groupby('snapshot_day')[['y','pred']].mean().round(1))
print('val mean y %.1f pred %.1f' % (y[va].mean(), p.mean()))
