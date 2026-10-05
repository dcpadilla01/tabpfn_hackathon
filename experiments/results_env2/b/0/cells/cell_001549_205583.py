
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def prep_table(path):
    df = agent_api.load_saved(path)
    feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    X = X.astype(float)
    return df, X

def eval_proxy(df, X, lam=30, val_days=(403,431)):
    tt = agent_api.train_targets()
    m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
    y = m.future_spend_4w.values
    trm = m.snapshot_day <= 375; vam = m.snapshot_day.isin(val_days)
    Xtr, Xva = X[trm.values], X[vam.values]
    med = Xtr.median(); Xtr = Xtr.fillna(med); Xva = Xva.fillna(med)
    mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
    A = np.c_[np.ones(trm.sum()), ((Xtr-mu)/sd).values]
    B = np.c_[np.ones(vam.sum()), ((Xva-mu)/sd).values]
    M = A.T@A + lam*np.eye(A.shape[1]); M[0,0]-=lam
    b = np.linalg.solve(M, A.T@y[trm.values])
    return np.abs(B@b - y[vam.values]).mean()

tables = {'e013':'e013_stationary.parquet','e011':'e011_pruned_basket.parquet',
          'e015':'e015_norm_windows.parquet','e017':'e017_reversion.parquet',
          'e008':'e008_log_transform.parquet','e010':'e010_store_mix.parquet'}
harness = {'e013':60.788,'e011':60.895,'e015':60.944,'e017':60.808,'e008':60.953,'e010':60.975}
for lam in [10,30,100]:
    scores = {k: eval_proxy(*prep_table(p), lam=lam) for k,p in tables.items()}
    rank = sorted(scores, key=scores.get)
    print('lam', lam, {k:round(v,2) for k,v in scores.items()}, 'proxy rank:', rank)
print('harness rank:', sorted(harness, key=harness.get))
