
import agent_api as A
import pandas as pd, numpy as np

for name in ['e019_everything.parquet','e019_full_merged.parquet','e018_union_full.parquet']:
    t = A.load_saved(name)
    print('='*20, name, t.shape)
    print('dup keys:', t.duplicated(['household_key','snapshot_day']).sum())
    print('snapshots:', sorted(t.snapshot_day.unique()))
    num = t.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in t.columns if c not in num]
    print('n numeric:', len(num), 'n other:', len(cat), cat[:20])
    na = t.isna().mean().sort_values(ascending=False)
    print('cols with >50% NaN:', (na>0.5).sum(), '| all-NaN cols:', (na==1.0).sum())
    print(na.head(5))


# ---- cell ----

import agent_api as A
t = A.load_saved('e018_union_full.parquet')
cols = list(t.columns)
for i in range(0, len(cols), 8):
    print(' | '.join(cols[i:i+8]))


# ---- cell ----

import agent_api as A
t = A.load_saved('e019_everything.parquet')
cols = list(t.columns)
for i in range(0, len(cols), 8):
    print(' | '.join(cols[i:i+8]))


# ---- cell ----

import agent_api as A
ev = A.load_saved('e019_everything.parquet')   # 155 = E018 minus 31 timing
fm = A.load_saved('e019_full_merged.parquet')  # 137 pruned merge
th = A.load_saved('e018_timing_hazard.parquet')
print('timing_hazard:', th.shape, list(th.columns))
extra = [c for c in fm.columns if c not in ev.columns]
dropped = [c for c in ev.columns if c not in fm.columns]
print('\nIN pruned-merge but NOT in e019_everything (%d):' % len(extra))
print(extra)
print('\nIN e019_everything but NOT in pruned-merge (%d):' % len(dropped))
print(dropped)
print('\nunion size check:', len(set(ev.columns)|set(th.columns)), 'ev+th overlap:', len(set(ev.columns)&set(th.columns)))


# ---- cell ----

import agent_api as A
for n in ['e017_v2.parquet','e017_disc_seasonal.parquet','e016_smoothed.parquet']:
    t = A.load_saved(n)
    print('='*15, n, t.shape)
    cols=list(t.columns)
    for i in range(0,len(cols),8): print(' | '.join(cols[i:i+8]))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

v = A.snapshot()  # capped at day 459
tx = v.transactions
# population weekly per-household spend
wk = tx.groupby('week_no').agg(total=('sales_value','sum'), hh=('household_key','nunique'))
wk['per_hh'] = wk['total']/wk['hh']
print("weeks:", wk.index.min(), wk.index.max())
print(wk['per_hh'].describe())
# YoY overlap: year1 weeks 1-14 vs year2 weeks 53-66
y1 = wk.loc[1:14,'per_hh'].values; y2 = wk.loc[53:66,'per_hh'].values
print("year1 w1-14:", np.round(y1,1))
print("year2 w53-66:", np.round(y2,1))
print("corr:", np.corrcoef(y1,y2)[0,1], " ratio y2/y1 mean:", (y2/y1).mean())
# full series rounded
print(np.round(wk['per_hh'].values,1))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np
tt = A.train_targets()
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).round(2))
# recent-spend level by snapshot (spend_l1 from union table) to see drift vs target
t = A.load_saved('e018_union_full.parquet')
m = t.groupby('snapshot_day')[['spend_l1','spend_total','tenure_x']].mean()
print(m.round(2))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w']

# 1) what are nratio_ly / nspend_ly / pct_* ? Correlate with spend_l1 and each other
cands = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l13','nspend28','nspend14',
         'nratio_ly','nspend_ly','pct_l1','pct_l4','pct_l13','peer_ratio','peer_ratio2',
         'cohort_prior4w','spend_ly4w','has_ly4w','ewm13_d','exp_trips28','basket_med_84',
         'momentum','trend_1v3','zero_frac_l13','act_w13','m_spend28_mean','m_spend28_med']
sub = m[cands + ['future_spend_4w']].copy()
corr = sub.corr(method='spearman')['future_spend_4w'].drop('future_spend_4w').sort_values(ascending=False)
print("Spearman corr with target (train rows):")
print(corr.round(3))
print()
print("corr nratio_ly vs spend_l1:", sub['nratio_ly'].corr(sub['spend_l1']).round(3),
      "| nspend_ly vs spend_l1:", sub['nspend_ly'].corr(sub['spend_l1']).round(3),
      "| pct_l1 vs spend_l1:", sub['pct_l1'].corr(sub['spend_l1']).round(3))
print("spend_ly4w NaN frac:", m['spend_ly4w'].isna().mean().round(3),
      "| spend_ly4w vs spend_l1 corr:", m['spend_ly4w'].corr(m['spend_l1']).round(3))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

# pseudo-validation at snapshot 431: how much headroom do simple predictors have?
u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
tr = m[m.snapshot_day <= 403]; te = m[m.snapshot_day == 431]
ytr, yte = tr['future_spend_4w'].values, te['future_spend_4w'].values

def mae(p, y): return np.mean(np.abs(p - y))

# baseline: global median
print("const median MAE@431:", round(mae(np.median(ytr), yte),2))
# spend_l1 raw
print("spend_l1 MAE@431:", round(mae(te['spend_l1'].values, yte),2))
# scaled spend_l1: fit a,b on train via least squares on (spend_l1 -> target)
X = tr['spend_l1'].values; 
b, a = np.polyfit(X, ytr, 1)
print("lin(spend_l1) MAE@431:", round(mae(a + b*te['spend_l1'].values, yte),2))
# log-log
Xl = np.log1p(tr['spend_l1'].values); yl = np.log1p(ytr)
b, a = np.polyfit(Xl, yl, 1)
p = np.expm1(a + b*np.log1p(te['spend_l1'].values))
print("loglog(spend_l1) MAE@431:", round(mae(p, yte),2))
# bin median of spend_l1 (20 bins) fit on train
bins = np.quantile(tr['spend_l1'], np.linspace(0,1,21))
bins[0]=-1e9; bins[-1]=1e9
bid = np.digitize(tr['spend_l1'], bins) 
bmed = pd.Series(ytr).groupby(bid).median()
p = np.array([bmed.get(np.digitize(v,bins), np.median(ytr)) for v in te['spend_l1'].values])
print("binned-median(spend_l1) MAE@431:", round(mae(p, yte),2))
# two-feature binned: spend_l1 x spend_l4 ratio
print()
print("target stats train: mean %.1f med %.1f | @431: mean %.1f med %.1f" % (ytr.mean(), np.median(ytr), yte.mean(), np.median(yte)))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w']

# Does the recent->future mapping drift across snapshots? Fit lin(spend_l1) per snapshot (in-sample)
print("In-sample linear slope/intercept of target on spend_l1, by snapshot:")
for d, g in m.groupby('snapshot_day'):
    b, a = np.polyfit(g['spend_l1'].values, g['future_spend_4w'].values, 1)
    r = g['spend_l1'].corr(g['future_spend_4w'])
    print(f"  day {d}: slope {b:.3f} intercept {a:7.1f} corr {r:.3f} n {len(g)}")

# ratio target/spend_l1 by snapshot
ratio = m['future_spend_4w']/m['spend_l1'].replace(0,np.nan)
print("\nmedian(target/spend_l1) by snapshot:")
print(m.assign(r=ratio).groupby('snapshot_day')['r'].median().round(3))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)

feats = [c for c in u.columns if c not in ('household_key','snapshot_day')]
X = m[feats].copy()
# numeric only, median impute, standardize
num = X.select_dtypes(include=[np.number]).columns
X = X[num]
X = X.fillna(X.median())
mu, sd = X.mean(), X.std().replace(0,1)
Xs = ((X-mu)/sd).values
Xs = np.hstack([Xs, np.ones((len(Xs),1))])

def ridge_fit(X, y, lam=10.0):
    d = X.shape[1]
    A_ = X.T@X + lam*np.eye(d); A_[-1,-1] -= lam  # don't penalize intercept much
    return np.linalg.solve(A_, X.T@y)

def mae(p,y): return np.mean(np.abs(p-y))

days = sorted(m.snapshot_day.unique())
oof = np.zeros(len(m))
for d in days:
    tr = m.snapshot_day != d; te = ~tr
    w = ridge_fit(Xs[tr.values], y[tr.values], lam=20.0)
    oof[te.values] = Xs[te.values]@w
print("LOSO ridge MAE:", round(mae(oof,y),3))
g = pd.DataFrame({'d':m.snapshot_day,'err':oof-y,'ae':np.abs(oof-y)})
print(g.groupby('d').agg(bias=('err','mean'), mae=('ae','mean')).round(2))
print("mean bias:", g.err.mean().round(2))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')

ten = m[m.tenure_x >= 364]
print("tenured rows:", len(ten))
for c in ['spend_l13','spend_ly4w','pct_l13','nspend_ly','nratio_ly']:
    print(f"{c:12s} corr(target): {ten[c].corr(ten['future_spend_4w']):.3f}  corr(spend_l1): {ten[c].corr(ten['spend_l1']):.3f}  mean {ten[c].mean():.1f}  zero-frac {(ten[c]==0).mean():.2f}")

# distribution of spend_l13 vs spend_l1 for tenured
print("\nspend_l13 describe (tenured):"); print(ten['spend_l13'].describe().round(1))
print("spend_l1  describe (tenured):"); print(ten['spend_l1'].describe().round(1))
# check pct_l13 definition: corr with spend_l13 and spend_l1
print("\npct_l13 vs spend_l13:", ten['pct_l13'].corr(ten['spend_l13']).round(3), " vs spend_l1:", ten['pct_l13'].corr(ten['spend_l1']).round(3))
# has_real_l13 flag coverage
print("has_real_l13 mean (all):", m['has_real_l13'].mean().round(3), " (tenured):", ten['has_real_l13'].mean().round(3))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)

feats = [c for c in u.columns if c not in ('household_key','snapshot_day')]
X = m[feats].select_dtypes(include=[np.number]).fillna(m[feats].select_dtypes(include=[np.number]).median())
mu, sd = X.mean(), X.std().replace(0,1)
Xs = np.hstack([((X-mu)/sd).values, np.ones((len(X),1))])
def ridge_fit(X, y, lam=20.0):
    d = X.shape[1]; A_ = X.T@X + lam*np.eye(d); A_[-1,-1] -= lam
    return np.linalg.solve(A_, X.T@y)
def mae(p,y): return np.mean(np.abs(p-y))

# A) LOSO with spend_l13-family dropped
drop13 = [c for c in feats if c in ('spend_l13','pct_l13','nspend_ly','has_real_l13','l13_over_recent')]
fA = [c for c in feats if c not in drop13]
XA = m[fA].select_dtypes(include=[np.number]).fillna(m[fA].select_dtypes(include=[np.number]).median())
XsA = np.hstack([((XA-XA.mean())/XA.std().replace(0,1)).values, np.ones((len(XA),1))])
oofA = np.zeros(len(m))
for d in sorted(m.snapshot_day.unique()):
    tr = m.snapshot_day != d
    w = ridge_fit(XsA[tr.values], y[tr.values])
    oofA[~tr.values] = XsA[~tr.values]@w
print("LOSO ridge WITHOUT l13-family:", round(mae(oofA,y),3))
print(pd.DataFrame({'d':m.snapshot_day,'err':oofA-y}).groupby('d').agg(bias=('err','mean'),mae=('err',lambda e: np.abs(e).mean())).round(2))

# B) LOSO full (repeat for direct comparison, lam 20)
oofB = np.zeros(len(m))
for d in sorted(m.snapshot_day.unique()):
    tr = m.snapshot_day != d
    w = ridge_fit(Xs[tr.values], y[tr.values])
    oofB[~tr.values] = Xs[~tr.values]@w
print("\nLOSO ridge full:", round(mae(oofB,y),3))
print(pd.DataFrame({'d':m.snapshot_day,'err':oofB-y}).groupby('d').agg(bias=('err','mean'),mae=('err',lambda e: np.abs(e).mean())).round(2))


# ---- cell ----

import agent_api as A
import pandas as pd, numpy as np

u = A.load_saved('e018_union_full.parquet')          # E018's 186-col table (best, MAE 62.037)
v2 = A.load_saved('e017_v2.parquet')                 # E017's 136-col table incl. discount decomposition
new = ['coupon_disc_84', 'match_disc_84']            # the only validated-family cols missing from E018
assert not any(c in u.columns for c in new)
add = v2[['household_key','snapshot_day'] + new]
uni = u.merge(add, on=['household_key','snapshot_day'], how='left')
print('union shape:', uni.shape, '| dup keys:', uni.duplicated(['household_key','snapshot_day']).sum())
print('NaN frac new cols:', uni[new].isna().mean().round(3).to_dict())
print(uni[new].describe().round(2))
path = A.save_table(uni, 'e019_true_union.parquet')
print(path)
