
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()
TR = [95,123,151,179,207,235,263,291,319,347,375,403,431]
tables = ['e001_recent_behavior','e002_full','e003_marketing','e004_temporal','e005_decay_gapcv','e006_momentum_seasonal']
harness = {'e001_recent_behavior':63.574,'e002_full':63.703,'e003_marketing':63.725,'e004_temporal':63.405,'e005_decay_gapcv':63.318,'e006_momentum_seasonal':63.399}

def ridge(Xtr, ytr, Xva, lam, clip=8.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd<1e-8]=1.0
    A = np.clip((Xtr-mu)/sd,-clip,clip); B = np.clip((Xva-mu)/sd,-clip,clip)
    A = np.c_[np.ones(len(A)),A]; B = np.c_[np.ones(len(B)),B]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ytr)
    return B@w

for t in tables:
    e = agent_api.load_saved(t+'.parquet')
    df = e.merge(tt, on=['household_key','snapshot_day'])
    feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
    y = df['future_spend_4w'].values
    itr = (df.snapshot_day<=347).values; iva = (df.snapshot_day>=375).values
    Xraw = df[feats].replace([np.inf,-np.inf], np.nan)
    X = Xraw.fillna(Xraw[itr].median()).fillna(0.0).values.astype(float)
    p = ridge(X[itr], y[itr], X[iva], 100.0)
    print('%-24s proxy %.3f  harness %.3f  nfeat %d' % (t, np.abs(p-y[iva]).mean(), harness[t], len(feats)))
