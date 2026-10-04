
import pandas as pd, numpy as np

paths = ['e001_recent_spend.parquet','e002_marketing.parquet','e002_marketing_v2.parquet',
         'e003_product_mix.parquet','e004_temporal.parquet','e005_longrun.parquet',
         'e006_fwd_profile.parquet','e007_log.parquet']
for p in paths:
    df = load_saved(p)
    print('==', p, df.shape)
    print(list(df.columns))
    print()

tt = train_targets()
print('targets', tt.shape)
print(tt['future_spend_4w'].describe())
print('zero share:', (tt['future_spend_4w']==0).mean())
print(tt.head())
print(snapshot_days())


# ---- cell ----

import pandas as pd, numpy as np

e5 = load_saved('e005_longrun.parquet')
e6 = load_saved('e006_fwd_profile.parquet')
e7 = load_saved('e007_log.parquet')
tt = train_targets()

# check whether e007 logged the spend features
print('e5 spend_28:', e5['spend_28'].describe()[['mean','50%','max']].to_dict())
print('e7 spend_28:', e7['spend_28'].describe()[['mean','50%','max']].to_dict())
print('e7 x_s28:', e7['x_s28'].describe()[['mean','50%','max']].to_dict())
print('e7 wk0:', e7['wk0'].describe()[['mean','50%','max']].to_dict())
print('e7 recency:', e7['recency'].describe()[['mean','50%','max']].to_dict(), 'NaN:', e7['recency'].isna().mean())
print('e7 x_recency:', e7['x_recency'].describe()[['mean','50%','max']].to_dict())

m = tt.merge(e5, on=['household_key','snapshot_day'], how='left').merge(
    e6[['household_key','snapshot_day','fwd28_k1','fwd28_k2','fwd28_k3','fwd28_k4','fwd28_k5','fwd28_k6']],
    on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values

def mae(p): return np.mean(np.abs(np.asarray(p)-y))

blocks = m[['fwd28_k1','fwd28_k2','fwd28_k3','fwd28_k4','fwd28_k5','fwd28_k6']].values
cands = {
 'zero': np.zeros(len(m)),
 'spend_28': m['spend_28'],
 'spend_56': m['spend_56'],
 'spend_84*4/3': m['spend_84']*4/3,
 'spend_112/2': m['spend_112']/2,
 'mean3blocks': blocks[:,:3].mean(1),
 'mean6blocks': np.nanmean(blocks,1),
 'ewma halflife1blk': sum(w*b for w,b in zip([0.5,0.3,0.2],[blocks[:,0],blocks[:,1],blocks[:,2]])),
}
for k,v in cands.items():
    print(f'{k:22s} MAE={mae(v):7.2f}')

# best scalar scaling (through origin) for spend_28
s = m['spend_28'].values
a = (s*y).sum()/ (s*s).sum()
print('spend_28 scaled best MAE:', mae(a*s), 'a=',a)

# in-sample linear fit (OLS) with e005 raw features as proxy for fixed model
X = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float)
Xf = X.fillna(X.median()).clip(-1e6,1e6)
Xm = np.c_[np.ones(len(Xf)), Xf.values]
beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
print('OLS in-sample MAE (e005 raw, 85 feat):', mae(Xm@beta))
# with log1p transforms of heavy features
Xl = Xf.copy()
for c in Xl.columns:
    if Xl[c].skew() > 2: Xl[c] = np.log1p(Xl[c].clip(lower=0))
Xm2 = np.c_[np.ones(len(Xl)), Xl.values]
beta2, *_ = np.linalg.lstsq(Xm2, y, rcond=None)
print('OLS in-sample MAE (log-skewed):', mae(Xm2@beta2))
print('y mean/med:', y.mean(), np.median(y), 'MAE of median:', mae(np.median(y)))


# ---- cell ----

import pandas as pd, numpy as np

tt = train_targets()
# target level by snapshot day (train only) - seasonality/drift?
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda s:(s==0).mean()])
g.columns=['mean','median','zerofrac']
print(g.round(1))

e5 = load_saved('e005_longrun.parquet')
m = tt.merge(e5[['household_key','snapshot_day','spend_28','wk_avg_8','spend_84']], on=['household_key','snapshot_day'])
print()
print(m.groupby('snapshot_day')[['future_spend_4w','spend_28','wk_avg_8']].median().round(1))


# ---- cell ----

import pandas as pd, numpy as np

e7 = load_saved('e007_log.parquet')
print('e7 cols:', len(e7.columns)-2)
# which features are NOT log1p-transformed
sk = e7.drop(columns=['household_key','snapshot_day']).skew()
print('skew>3 (raw-scale left):')
print(sk[sk.abs()>3].round(1).to_string())
print('\nNaN counts (top):')
na = e7.isna().sum()
print(na[na>0].sort_values(ascending=False).head(12).to_string())


# ---- cell ----
import pandas as pd, numpy as np

e7 = load_saved('e007_log.parquet')
tt = train_targets()
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values

# OLS with ridge on e007 features (in-sample proxy for the fixed linear model)
def ridge(X, y, lam=1.0):
    Xb = np.c_[np.ones(len(X)), X]
    A = Xb.T@Xb + lam*np.eye(Xb.shape[1]); A[0,0]=0
    return np.linalg.solve(A, Xb.T@y)

X = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float)
med = X.median()
Xf = X.fillna(med).clip(-1e6,1e6).values
Xm = np.c_[np.ones(len(Xf)), Xf]
# standardize
mu = Xm[:,1:].mean(0); sd = Xm[:,1:].std(0)+1e-9
Z = np.c_[np.ones(len(Xm)), (Xm[:,1:]-mu)/sd]
for lam in [1,10,50,100]:
    b = ridge(Z[:,1:], y, lam)
    p = Z@np.r_[0,b]
    print(f'ridge lam={lam}: in-sample MAE={np.mean(np.abs(p-y)):.2f}')

# also check: how many households are in train but not validation (new households)
hh_tr = set(m.household_key[m.snapshot_day<=431])
print('\ntrain rows', len(m), 'unique hh', m.household_key.nunique())

# target stats per household variance vs within
tt2 = tt.sort_values(['household_key','snapshot_day'])
print('within-hh std/overall std:')
print(tt2.groupby('household_key')['future_spend_4w'].std().mean(), tt2.future_spend_4w.std())

# ---- cell ----
import pandas as pd, numpy as np

e7 = load_saved('e007_log.parquet')
tt = train_targets()
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values

X = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float)
med = X.median()
Xf = X.fillna(med).clip(-1e6,1e6).values
mu = Xf.mean(0); sd = Xf.std(0)+1e-9
Z = (Xf-mu)/sd
Zb = np.c_[np.ones(len(Z)), Z]
def ridge(Zb, y, lam):
    A = Zb.T@Zb + lam*np.eye(Zb.shape[1]); A[0,0]=0
    b = np.linalg.solve(A, Zb.T@y)
    p = Zb@b
    return np.mean(np.abs(p-y))
for lam in [1,10,50,100,300]:
    print(f'ridge lam={lam}: in-sample MAE={ridge(Zb,y,lam):.2f}')

# per-household target variability
tt2 = tt.sort_values(['household_key','snapshot_day'])
w = tt2.groupby('household_key')['future_spend_4w'].std()
print('\nmean within-hh std:', w.mean(), 'overall std:', tt2.future_spend_4w.std())
# correlation of household mean spend_28 with household mean target
hm = tt2.groupby('household_key')['future_spend_4w'].mean()
e5 = load_saved('e005_longrun.parquet')
s28 = e5.groupby('household_key')['spend_28'].mean()
j = pd.concat([hm, s28], axis=1).dropna()
print('corr(hh-mean target, hh-mean spend_28):', j.corr().iloc[0,1].round(3))

# ---- cell ----
import pandas as pd, numpy as np

e7 = load_saved('e007_log.parquet')
tt = train_targets()
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values

X = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float)
# drop exact duplicates
X = X.T.drop_duplicates().T
print('after dedup:', X.shape)
med = X.median()
Xf = X.fillna(med).clip(-1e6,1e6).values
mu = Xf.mean(0); sd = Xf.std(0)+1e-9
Z = (Xf-mu)/sd
Zb = np.c_[np.ones(len(Z)), Z]
def ridge(Zb, y, lam):
    A = Zb.T@Zb + lam*np.eye(Zb.shape[1]); A[0,0]=0
    b = np.linalg.solve(A, Zb.T@y)
    p = Zb@b
    return np.mean(np.abs(p-y))
for lam in [1,10,50,100,300]:
    print(f'ridge lam={lam}: in-sample MAE={ridge(Zb,y,lam):.2f}')

# how much of MAE is from zero-target rows? per-target-bin MAE for a simple predictor spend_28*0.87
s28 = m['spend_28'].values
p = 0.87*s28
err = np.abs(p-y)
print('\nMAE by target bucket:')
for lo,hi in [(0,1),(1,50),(50,100),(100,200),(200,400),(400,1e9)]:
    msk = (y>=lo)&(y<hi)
    print(f'  y in [{lo},{hi}): n={msk.sum():6d}  MAE={err[msk].mean():7.2f}  mean|y|={y[msk].mean():7.1f}')
print('share of total MAE from y==0 rows:', (err[y==0].sum())/err.sum())
print('share of rows y==0:', (y==0).mean())

# ---- cell ----
import pandas as pd, numpy as np
tt = train_targets()
# Upper bound: predict each household's train-mean target
hh_mean = tt.groupby('household_key')['future_spend_4w'].mean()
p = tt.household_key.map(hh_mean).values
y = tt.future_spend_4w.values
print('in-sample MAE predicting household train-mean:', np.mean(np.abs(p-y)))

# Ratio stability: for each (hh, snapshot), r = future_spend_4w / spend_28_at_that_snapshot
e5 = load_saved('e005_longrun.parquet')
m = tt.merge(e5[['household_key','snapshot_day','spend_28']], on=['household_key','snapshot_day'])
m['r'] = m.future_spend_4w / m.spend_28.clip(lower=1e-6)
m['lr'] = np.log1p(m.future_spend_4w) - np.log1p(m.spend_28)   # log-ratio approx
g = m.groupby('household_key')['lr']
print('\nlog-ratio: mean within-hh std:', g.std().mean(), 'overall std:', m.lr.std())
print('within-hh std by n_obs:')
n = g.size()
for lo,hi in [(1,3),(3,6),(6,9),(9,13)]:
    sel = (n>=lo)&(n<hi)
    print(f'  n_obs {lo}-{hi-1}: mean within-std={g.std()[sel].mean():.3f} (n hh={sel.sum()})')

# autocorrelation of lr across snapshots within household
m2 = m.sort_values(['household_key','snapshot_day'])
m2['lr_prev'] = m2.groupby('household_key')['lr'].shift(1)
print('\ncorr(lr, lr_prev):', m2[['lr','lr_prev']].corr().iloc[0,1].round(3))
print('corr(r, r_prev) raw:', m2.assign(r_prev=m2.groupby('household_key')['r'].shift(1))[['r','r_prev']].corr().iloc[0,1].round(3))
print('median lr:', m.lr.median().round(3), 'median r:', m.r.median().round(3))

# ---- cell ----
import pandas as pd, numpy as np
tt = train_targets()
y = tt.future_spend_4w.values
hh = tt.sort_values(['household_key','snapshot_day'])
gm = tt.groupby('household_key')['future_spend_4w']
hmean = gm.transform('mean'); hsum = gm.transform('sum'); hsize = gm.transform('size')

print('within-hh std / hh mean:', (hmean.clip(lower=1) and (hh.groupby('household_key')['future_spend_4w'].transform('std')/hmean.clip(lower=1))).mean().round(3))

for k in [0.5,1,2,4]:
    shrunk = (hsum + k*136.86) / (hsize + k)
    print('shrinkage k=%s: MAE=%.2f' % (k, np.mean(np.abs(shrunk.values-y))))

print('\nhh-mean distribution:')
print(gm.mean().describe([.1,.25,.5,.75,.9,.95]).round(1))

# how many households appear in train only once? (level estimate noisy)
print('\nhh obs counts:', gm.size().describe([.1,.5,.9]).to_dict())

# ---- cell ----
import pandas as pd, numpy as np
tt = train_targets()
y = tt.future_spend_4w.values
gm = tt.groupby('household_key')['future_spend_4w']
hmean = gm.transform('mean'); hsum = gm.transform('sum'); hsize = gm.transform('size')
hstd = gm.transform('std')

print('mean within-hh CV:', (hstd/hmean.clip(lower=1)).mean().round(3))
for k in [0.5,1,2,4]:
    shrunk = (hsum + k*136.86) / (hsize + k)
    print('shrinkage k=%s: MAE=%.2f' % (k, np.mean(np.abs(shrunk.values-y))))

print('\nhh-mean distribution:')
print(gm.mean().describe([.1,.25,.5,.75,.9,.95]).round(1))
print('\nhh obs counts:', gm.size().describe([.1,.5,.9]).to_dict())

# ---- cell ----
import pandas as pd, numpy as np
tt = train_targets()
y = tt.future_spend_4w.values

# Best possible "household level" predictor: leave-one-snapshot-out hh mean (honest, no leakage)
gm = tt.groupby('household_key')['future_spend_4w']
hsum = gm.transform('sum'); hsize = gm.transform('size')
loo = (hsum - tt.future_spend_4w) / (hsize - 1)
print('LOO hh-mean MAE (upper bound on level-based):', np.mean(np.abs(loo.values-y)).round(2))

# Now the key question: can we estimate the household LEVEL well from features?
# Test: fit a model to predict the household's train-mean target from its mean E005/E007 features.
e7 = load_saved('e007_log.parquet')
hh_feat = e7.drop(columns=['household_key','snapshot_day']).astype(float).groupby('household_key').mean()
hh_mean_t = gm.mean()
j = hh_feat.join(hh_mean_t.rename('y'), how='inner')
print('joined', j.shape)
X = j.drop(columns='y'); mu,sd = X.mean(), X.std()+1e-9
Z = np.c_[np.ones(len(j)), ((X-mu)/sd).values]
b = np.linalg.lstsq(Z, j.y.values, rcond=None)[0]
p = Z@b
print('in-sample MAE predicting hh-mean level:', np.mean(np.abs(p-j.y.values)).round(2))
# corr of predicted level with actual level
print('corr(pred level, actual level):', np.corrcoef(p, j.y.values)[0,1].round(3))
# and what MAE would that give on row targets?
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
Xf = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float).fillna(hh_feat.median()).values
Xf = (Xf-mu.values)/sd.values
Xf = np.c_[np.ones(len(Xf)), Xf]
prow = Xf@b
print('row-level MAE using level model:', np.mean(np.abs(prow-y)).round(2))

# ---- cell ----
import pandas as pd, numpy as np
tt = train_targets()
y = tt.future_spend_4w.values
gm = tt.groupby('household_key')['future_spend_4w']

e7 = load_saved('e007_log.parquet')
hh_feat = e7.drop(columns=['household_key','snapshot_day']).astype(float).groupby('household_key').mean()
hh_mean_t = gm.mean()
j = hh_feat.join(hh_mean_t.rename('y'), how='inner')
print('joined', j.shape)
X = j.drop(columns='y'); mu,sd = X.mean(), X.std()+1e-9
Z = np.c_[np.ones(len(j)), ((X-mu)/sd).values]
b = np.linalg.lstsq(Z, j.y.values, rcond=None)[0]
p = Z@b
print('in-sample MAE predicting hh-mean level:', np.mean(np.abs(p-j.y.values)).round(2))
print('corr(pred level, actual level):', np.corrcoef(p, j.y.values)[0,1].round(3))

m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
Xf = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float).fillna(hh_feat.median()).values
Xf = (Xf-mu.values)/sd.values
Xf = np.c_[np.ones(len(Xf)), Xf]
prow = Xf@b
print('row-level MAE using level model:', np.mean(np.abs(prow-y)).round(2))

# ---- cell ----
import pandas as pd, numpy as np
tt = train_targets()
y = tt.future_spend_4w.values
gm = tt.groupby('household_key')['future_spend_4w']

e7 = load_saved('e007_log.parquet')
hh_feat = e7.groupby('household_key').mean(numeric_only=True)
hh_mean_t = gm.mean()
j = hh_feat.join(hh_mean_t.rename('y'), how='inner')
print('joined', j.shape)
X = j.drop(columns='y'); mu,sd = X.mean(), X.std()+1e-9
Z = np.c_[np.ones(len(j)), ((X-mu)/sd).values]
b = np.linalg.lstsq(Z, j.y.values, rcond=None)[0]
p = Z@b
print('in-sample MAE predicting hh-mean level:', np.mean(np.abs(p-j.y.values)).round(2))
print('corr(pred level, actual level):', np.corrcoef(p, j.y.values)[0,1].round(3))
m = tt.merge(e7, on=['household_key','snapshot_day'], how='left')
Xf = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).astype(float).fillna(hh_feat.median()).values
Xf = (Xf-mu.values)/sd.values
Xf = np.c_[np.ones(len(Xf)), Xf]
print('row-level MAE using level model:', np.mean(np.abs(Xf@b-y)).round(2))

# ---- cell ----
import pandas as pd, numpy as np

e5 = load_saved('e005_longrun.parquet')
e6 = load_saved('e006_fwd_profile.parquet')
cal = e5[['household_key','snapshot_day']].copy()
cal['day_idx'] = cal['snapshot_day']
cal['week_of_year'] = ((cal['snapshot_day'] + 8) // 7) % 52
cal['sin1'] = np.sin(2*np.pi*cal['week_of_year']/52); cal['cos1'] = np.cos(2*np.pi*cal['week_of_year']/52)
cal['sin2'] = np.sin(4*np.pi*cal['week_of_year']/52); cal['cos2'] = np.cos(4*np.pi*cal['week_of_year']/52)
cal['month_idx'] = cal['snapshot_day'] // 28

df = e5.merge(e6, on=['household_key','snapshot_day'], how='inner').merge(
    cal[['household_key','snapshot_day','day_idx','week_of_year','sin1','cos1','sin2','cos2','month_idx']],
    on=['household_key','snapshot_day'], how='inner')
print(df.shape)
p = save_table(df, 'e008_fwd_calendar.parquet')
print(p)