import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
sd = agent_api.snapshot_days()
tr_days, va_days = sd['train'], sd['validation']
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]

def fit_eval(Xtr, ytr, Xva, yva, alphas):
    mu = np.nanmean(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    sg = np.nanstd(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)+1e-9
    Ztr = np.nan_to_num((Xtr-mu)/sg); Zva = np.nan_to_num((Xva-mu)/sg)
    best = None
    for a in alphas:
        w = np.linalg.solve(Ztr.T@Ztr + a*np.eye(Ztr.shape[1]), Ztr.T@(ytr-ytr.mean()))
        pred = Zva@w + ytr.mean()
        m = np.abs(pred-yva).mean()
        if best is None or m < best[0]: best = (m, a)
    return best

y = df['future_spend_4w'].values
m_tr = df.snapshot_day.isin(tr_days); m_va = df.snapshot_day.isin(va_days)

Xraw = df[feats].values.astype(float)
print('ridge RAW  val MAE %.3f (alpha %s)' % fit_eval(Xraw[m_tr], y[m_tr], Xraw[m_va], y[m_va], [1,10,100,300,1000,3000]))

logcols = [c for c in feats if any(c.startswith(p) for p in ('spend','ew_','avg_basket','basket_','longrun_wk','n_products','n_stores'))]
Xlog = df[feats].copy().astype(float)
for c in logcols: Xlog[c] = np.log1p(Xlog[c].clip(lower=0))
print('ridge LOG  val MAE %.3f (alpha %s)' % fit_eval(Xlog.values[m_tr], y[m_tr], Xlog.values[m_va], y[m_va], [1,10,100,300,1000,3000]))

# log-transformed target?
ylog = np.log1p(y)
print('ridge LOGX LOGY val MAE %.3f (alpha %s)' % fit_eval(Xlog.values[m_tr], ylog[m_tr], Xlog.values[m_va], ylog[m_va], [1,10,100,300,1000,3000]))
# with expm1 back-transform
mu = np.nanmean(Xlog.values[m_tr],axis=0); sg=np.nanstd(Xlog.values[m_tr],axis=0)+1e-9
Ztr=np.nan_to_num((Xlog.values[m_tr]-mu)/sg); Zva=np.nan_to_num((Xlog.values[m_va]-mu)/sg)
w=np.linalg.solve(Ztr.T@Ztr+100*np.eye(Ztr.shape[1]), Ztr.T@(ylog[m_tr]-ylog[m_tr].mean()))
pred=np.expm1(np.clip(Zva@w+ylog[m_tr].mean(),0,8))
print('ridge LOGX LOGY(expm1) val MAE %.3f' % np.abs(pred-y[m_va]).mean())
print('baseline mean-pred val MAE %.3f' % np.abs(np.full(m_va.sum(), y[m_tr].mean())-y[m_va]).mean())