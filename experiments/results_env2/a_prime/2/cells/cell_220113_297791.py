import pandas as pd, numpy as np, agent_api

df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')
sd = agent_api.snapshot_days()
tr_days, va_days = sd['train'], sd['validation']
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]

def fit_eval(Xtr, ytr, Xva, yva, alphas):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Ztr, Zva = (Xtr-mu)/sg, (Xva-mu)/sg
    Ztr, Zva = np.nan_to_num(Ztr), np.nan_to_num(Zva)
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
r_raw = fit_eval(Xraw[m_tr], y[m_tr], Xraw[m_va], y[m_va], [0.1,1,10,100,300,1000])
print('ridge RAW  val MAE %.3f (alpha %s)' % r_raw)

logcols = [c for c in feats if any(c.startswith(p) for p in ('spend','ew_','avg_basket','basket_','longrun_wk','n_products','n_stores'))]
Xlog = df[feats].copy().astype(float)
for c in logcols: Xlog[c] = np.log1p(Xlog[c].clip(lower=0))
r_log = fit_eval(Xlog.values[m_tr], y[m_tr], Xlog.values[m_va], y[m_va], [0.1,1,10,100,300,1000])
print('ridge LOG  val MAE %.3f (alpha %s)' % r_log)

# correlations with target on train rows
sub = df[m_tr]
for c in ['spend_28','spend_28_prior','spend_84','ew_28','ew_84','spend_365','days_since_last','gap_cv','baskets_28','ratio28_lr']:
    print('%-16s corr %.3f  spearman %.3f' % (c, np.corrcoef(sub[c].fillna(0), sub.future_spend_4w)[0,1], sub[c].fillna(0).corr(sub.future_spend_4w, method='spearman')))
print('zero-target frac by active_28:', sub.groupby('active_28').future_spend_4w.agg(['mean','count']).to_dict())