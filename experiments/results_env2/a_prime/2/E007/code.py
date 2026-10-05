import pandas as pd, numpy as np, agent_api

for p in ['e005_decay_gapcv.parquet','e004_temporal.parquet','e001_recent_behavior.parquet']:
    df = agent_api.load_saved(p)
    print(p, df.shape)
    print(sorted([c for c in df.columns if c not in ('household_key','snapshot_day')]))
    print()

tt = agent_api.train_targets()
print(tt['future_spend_4w'].describe())
print('zero frac train:', (tt.future_spend_4w==0).mean())
print(agent_api.snapshot_days())

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e005_decay_gapcv.parquet')
print(df.dtypes.value_counts())
print(df.isna().sum().sort_values(ascending=False).head(10))
print(df[df.columns[:5]].head(3))

# ---- cell ----
import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
d2 = df.merge(tt, on=['household_key','snapshot_day'], how='left')
sd = agent_api.snapshot_days(); tr, va = sd['train'], sd['validation']
mtr = d2.snapshot_day.isin(tr); mva = d2.snapshot_day.isin(va)
ytr = d2.loc[mtr,'future_spend_4w'].values; yva = d2.loc[mva,'future_spend_4w'].values
print('ytr nan:', np.isnan(ytr).sum(), 'yva nan:', np.isnan(yva).sum(), 'ntr', mtr.sum(), 'nva', mva.sum())

def fit_eval(Xtr, ytr_, Xva, yva_, alphas=(1,10,100,300,1000,3000)):
    mu = np.nanmean(Xtr, axis=0); sg = np.nanstd(Xtr, axis=0)+1e-9
    Ztr = np.nan_to_num((Xtr-mu)/sg); Zva = np.nan_to_num((Xva-mu)/sg)
    best=None
    for a in alphas:
        w = np.linalg.solve(Ztr.T@Ztr + a*np.eye(Ztr.shape[1]), Ztr.T@(ytr_-ytr_.mean()))
        pred = Zva@w + ytr_.mean()
        mae = np.abs(pred-yva_).mean()
        if best is None or mae<best[0]: best=(mae,a)
    return best

Xraw = d2[feats].values.astype(float)
print('FULL RAW   MAE %.3f (a %s)' % fit_eval(Xraw[mtr.values], ytr, Xraw[mva.values], yva))
Xl = d2[feats].astype(float).copy()
logcols = [c for c in feats if any(c.startswith(p) for p in ('spend','ew_','avg_basket','basket_','longrun_wk','n_products','n_stores'))]
for c in logcols: Xl[c]=np.log1p(Xl[c].clip(lower=0))
print('FULL LOG   MAE %.3f (a %s)' % fit_eval(Xl.values[mtr.values], ytr, Xl.values[mva.values], yva))
mu=np.nanmean(Xl.values[mtr.values],axis=0); sg=np.nanstd(Xl.values[mtr.values],axis=0)+1e-9
Ztr=np.nan_to_num((Xl.values[mtr.values]-mu)/sg); Zva=np.nan_to_num((Xl.values[mva.values]-mu)/sg)
ylog=np.log1p(ytr); best=None
for a in [1,10,100,300,1000]:
    w=np.linalg.solve(Ztr.T@Ztr+a*np.eye(Ztr.shape[1]), Ztr.T@(ylog-ylog.mean()))
    pred=np.expm1(np.clip(Zva@w+ylog.mean(),0,8))
    mae=np.abs(pred-yva).mean()
    if best is None or mae<best[0]: best=(mae,a)
print('FULL LOGX/LOGY expm1 MAE %.3f (a %s)' % best)

# ---- cell ----
import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
d2 = df.merge(tt, on=['household_key','snapshot_day'], how='inner')  # train rows only
feats = [c for c in d2.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
y = d2.future_spend_4w.values

def fit_pred(Xtr, ytr, Xva, a=300):
    mu=np.nanmean(Xtr,axis=0); sg=np.nanstd(Xtr,axis=0)+1e-9
    Ztr=np.nan_to_num((Xtr-mu)/sg); Zva=np.nan_to_num((Xva-mu)/sg)
    w=np.linalg.solve(Ztr.T@Ztr+a*np.eye(Ztr.shape[1]), Ztr.T@(ytr-ytr.mean()))
    return Zva@w+ytr.mean()

# two inner eval snapshots for stability: fit on <=375, eval on 403+431
mfit = d2.snapshot_day<=375; miv = d2.snapshot_day>=403
X = d2[feats].values.astype(float)
base = np.abs(fit_pred(X[mfit.values], y[mfit.values], X[miv.values], 300)-y[miv.values]).mean()
print('inner base (E005 raw, a=300): %.3f  (n_eval=%d)' % (base, miv.sum()))

# candidate engineered features (raw scale)
s28=d2.spend_28.fillna(0); s84=d2.spend_84.fillna(0); s365=d2.spend_365.fillna(0)
dsl=d2.days_since_last.fillna(999); ew84=d2.ew_84.fillna(0); ew28=d2.ew_28.fillna(0)
act=d2.active_28.astype(float); b28=d2.baskets_28.fillna(0); r28=d2.spend_28_ratio.fillna(1)
cands = {
 'dsl_x_ew84': dsl*ew84/100.0,
 'act_x_ew84': act*ew84,
 'act_x_s84': act*s84,
 'sqrt_s84': np.sqrt(s84),
 'sqrt_s28': np.sqrt(s28),
 's84_sq': s84**2/1000.0,
 'dsl_flag28': (dsl>28).astype(float),
 'dsl_flag56': (dsl>56).astype(float),
 'zero_frac_365': 1-np.minimum(1, (s28>0).astype(float)),  # placeholder
 'inv_dsl': 1.0/(1+dsl),
 'ew28_x_b28': ew28*b28,
 's84_minus_s365_4w': s84 - s365/13.0,
}
for k,v in cands.items():
    Xn = np.column_stack([X, v.values.astype(float)])
    m = np.abs(fit_pred(Xn[mfit.values], y[mfit.values], Xn[miv.values], 300)-y[miv.values]).mean()
    print('%-20s %.3f  (%+.3f)' % (k, m, m-base))

# ---- cell ----
import pandas as pd, numpy as np, agent_api
# offline prototype: aligned 4-week lag spends (no leakage: windows end at snapshot day)
v = agent_api.snapshot(459)
tx = v.transactions[['household_key','day','sales_value']]
hh = np.sort(tx.household_key.unique())
hidx = {h:i for i,h in enumerate(hh)}
spend = np.zeros((len(hh), 460), dtype=np.float64)
np.add.at(spend, (tx.household_key.map(hidx).values, tx.day.values), tx.sales_value.values)
cum = np.concatenate([np.zeros((len(hh),1)), spend.cumsum(1)], axis=1)  # cum[:,d] = spend days 1..d

df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
d2 = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
rows = d2.household_key.map(hidx).values; s = d2.snapshot_day.values
lag = {}
for k in range(1, 14):
    hi = np.clip(s-28*k, 0, 459); lo = np.clip(s-28*(k+1), 0, 459)
    lag['lag%d'%k] = cum[rows, hi] - cum[rows, lo]
L = pd.DataFrame(lag)
print(L.describe().round(1).T[['mean','50%','max']])

feats = [c for c in d2.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
y = d2.future_spend_4w.values
def fit_pred(Xtr, ytr, Xva, a=300):
    mu=np.nanmean(Xtr,axis=0); sg=np.nanstd(Xtr,axis=0)+1e-9
    Ztr=np.nan_to_num((Xtr-mu)/sg); Zva=np.nan_to_num((Xva-mu)/sg)
    w=np.linalg.solve(Ztr.T@Ztr+a*np.eye(Ztr.shape[1]), Ztr.T@(ytr-ytr.mean()))
    return Zva@w+ytr.mean()
mfit=(d2.snapshot_day<=375).values; miv=(d2.snapshot_day>=403).values
X0 = d2[feats].values.astype(float)
base = np.abs(fit_pred(X0[mfit], y[mfit], X0[miv], 300)-y[miv]).mean()
print('base %.3f' % base)
for ks in [[1],[1,2],[1,2,3],[1,2,3,4],[1,2,3,4,5,6],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11,12,13]]:
    Xn = np.column_stack([X0]+[lag['lag%d'%k] for k in ks])
    m = np.abs(fit_pred(Xn[mfit], y[mfit], Xn[miv], 300)-y[miv]).mean()
    print('lags %-25s %.3f (%+.3f)' % (str(ks), m, m-base))
# household fixed effect: mean of available lags 1..8
Lm = np.column_stack([lag['lag%d'%k] for k in range(1,9)])
fe = np.nanmean(np.where(Lm>0, Lm, np.nan), axis=1)
Xn = np.column_stack([X0, np.nan_to_num(fe)])
m = np.abs(fit_pred(Xn[mfit], y[mfit], Xn[miv], 300)-y[miv]).mean()
print('fe_mean_lags %.3f (%+.3f)' % (m, m-base))

# ---- cell ----
import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
d2 = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in d2.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
y = d2.future_spend_4w.values
def fit_pred(Xtr, ytr, Xva, a=300):
    mu=np.nanmean(Xtr,axis=0); sg=np.nanstd(Xtr,axis=0)+1e-9
    Ztr=np.nan_to_num((Xtr-mu)/sg); Zva=np.nan_to_num((Xva-mu)/sg)
    w=np.linalg.solve(Ztr.T@Ztr+a*np.eye(Ztr.shape[1]), Ztr.T@(ytr-ytr.mean()))
    return Zva@w+ytr.mean()
mfit=(d2.snapshot_day<=375).values; miv=(d2.snapshot_day>=403).values
X0 = d2[feats].values.astype(float)
pred = fit_pred(X0[mfit], y[mfit], X0[miv], 300)
yv = y[miv]; err = np.abs(pred-yv)
act = d2.active_28.values[miv]
print('MAE inactive: %.2f (n=%d, mean y=%.1f, median y=%.1f, mean pred=%.1f)' % (err[~act].mean(), (~act).sum(), yv[~act].mean(), np.median(yv[~act]), pred[~act].mean()))
print('MAE active  : %.2f (n=%d, mean y=%.1f, median y=%.1f, mean pred=%.1f)' % (err[act].mean(), act.sum(), yv[act].mean(), np.median(yv[act]), pred[act].mean()))
for lo,hi in [(0,25),(25,75),(75,150),(150,300),(300,1e9)]:
    m=(yv>=lo)&(yv<hi)
    print('y in [%4d,%5d): n=%5d mean pred=%7.1f mean y=%7.1f MAE=%6.1f' % (lo,hi,m.sum(),pred[m].mean(),yv[m].mean(),err[m].mean()))
# group-mean predictors: what MAE would constant-per-group give?
for grp in [d2.active_28.values[miv], (d2.days_since_last.fillna(999).values[miv]>56)]:
    gm = pd.Series(yv).groupby(grp).transform('median').values
    print('group-median MAE: %.2f' % np.abs(gm-yv).mean())