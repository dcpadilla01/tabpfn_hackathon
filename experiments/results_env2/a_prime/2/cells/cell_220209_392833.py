import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
df = df[df.future_spend_4w.notna()].copy()  # train rows only
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
y = df['future_spend_4w'].values

def fit_eval(Xtr, ytr, Xva, yva, alphas=(1,10,100,300,1000,3000)):
    mu = np.nanmean(Xtr, axis=0); sg = np.nanstd(Xtr, axis=0)+1e-9
    Ztr = np.nan_to_num((Xtr-mu)/sg); Zva = np.nan_to_num((Xva-mu)/sg)
    best = None
    for a in alphas:
        w = np.linalg.solve(Ztr.T@Ztr + a*np.eye(Ztr.shape[1]), Ztr.T@(ytr-ytr.mean()))
        pred = Zva@w + ytr.mean()
        m = np.abs(pred-yva).mean()
        if best is None or m < best[0]: best = (m, a)
    return best

# inner split: fit on snapshots <=403, validate on 431 (out-of-time)
m_fit = df.snapshot_day <= 403; m_iv = df.snapshot_day == 431
X = df[feats].values.astype(float)
print('inner RAW  MAE %.3f (a %s)' % fit_eval(X[m_fit], y[m_fit], X[m_iv], y[m_iv]))
logcols = [c for c in feats if any(c.startswith(p) for p in ('spend','ew_','avg_basket','basket_','longrun_wk','n_products','n_stores'))]
Xl = df[feats].astype(float).copy()
for c in logcols: Xl[c] = np.log1p(Xl[c].clip(lower=0))
print('inner LOG  MAE %.3f (a %s)' % fit_eval(Xl.values[m_fit], y[m_fit], Xl.values[m_iv], y[m_iv]))
ylog = np.log1p(y)
print('inner LOGX/LOGY MAE %.3f (a %s)' % fit_eval(Xl.values[m_fit], ylog[m_fit], Xl.values[m_iv], ylog[m_iv]))
# expm1 back-transform for LOGX/LOGY
def fit_pred(Xtr, ytr, Xva, a):
    mu = np.nanmean(Xtr, axis=0); sg = np.nanstd(Xtr, axis=0)+1e-9
    Ztr = np.nan_to_num((Xtr-mu)/sg); Zva = np.nan_to_num((Xva-mu)/sg)
    w = np.linalg.solve(Ztr.T@Ztr + a*np.eye(Ztr.shape[1]), Ztr.T@(ytr-ytr.mean()))
    return Zva@w + ytr.mean()
p = np.expm1(np.clip(fit_pred(Xl.values[m_fit], ylog[m_fit], Xl.values[m_iv], 100), 0, 8))
print('inner LOGX/LOGY expm1 MAE %.3f' % np.abs(p-y[m_iv]).mean())
# also: raw target on log features, per-snapshot intercept check
print('mean-pred inner MAE %.3f' % np.abs(np.full(m_iv.sum(), y[m_fit].mean())-y[m_iv]).mean())