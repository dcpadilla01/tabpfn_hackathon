import numpy as np, pandas as pd
from agent_api import load_saved, train_targets, snapshot_days

print('snapdays', snapshot_days())
t = load_saved('e017_xsec_rank.parquet')
tt = train_targets()
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print('E017 shape', t.shape, 'n_feat', len(feat_cols))
print('cols:', sorted(feat_cols))

df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
print('merged', df.shape, 'missing targets', int(df['future_spend_4w'].isna().sum()))
df = df[df['future_spend_4w'].notna()]
y = df['future_spend_4w'].astype(float).values
days = df['snapshot_day'].values
print('y: mean %.2f med %.2f std %.2f zero %.3f' % (y.mean(), np.median(y), y.std(), (y==0).mean()))
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).round(2))


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

t = load_saved('e017_xsec_rank.parquet')
tt = train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]

def prep(d):
    X = d[feat_cols].copy()
    for c in feat_cols:
        if X[c].dtype == bool: X[c] = X[c].astype(float)
        elif not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
    return X.values.astype(float)

def fit_ridge(X, y, lam=100.0):
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd==0]=1.0
    Z = (X-mu)/sd; Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    A = Z.T@Z + lam*np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T@y)
    return w, mu, sd

def apply_ridge(w, mu, sd, X):
    Z = (X-mu)/sd; Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z@w

y = df['future_spend_4w'].values.astype(float)
days = df['snapshot_day'].values
X = prep(df)

# full harness replication: fit on 13 train snaps, predict 4 val snaps
tr = np.isin(days, [95,123,151,179,207,235,263,291,319,347,375,403,431])
va = np.isin(days, [459,487,515,543])
w, mu, sd = fit_ridge(X[tr], y[tr])
pv = apply_ridge(w, mu, sd, X[va])
print('harness-replica val MAE: %.3f  (harness E017 = 62.526)' % np.abs(pv - y[va]).mean())

# pseudo-val: last 3 train snaps held out
tr2 = np.isin(days, [95,123,151,179,207,235,263,291,319,347,375])
va2 = np.isin(days, [403,431])
w2, mu2, sd2 = fit_ridge(X[tr2], y[tr2])
pv2 = apply_ridge(w2, mu2, sd2, X[va2])
print('pseudo-val MAE (403,431): %.3f' % np.abs(pv2 - y[va2]).mean())

# lambda sweep on harness replica
for lam in [10, 30, 100, 300, 1000]:
    w3,_,_ = fit_ridge(X[tr], y[tr], lam=lam)
    pv3 = apply_ridge(w3, mu, sd, X[va])
    print('lam %5d -> val MAE %.3f' % (lam, np.abs(pv3-y[va]).mean()))


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

t = load_saved('e017_xsec_rank.parquet')
tt = train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]

for c in feat_cols:
    s = df[c]
    if not np.issubdtype(s.dtype, np.number) and s.dtype != bool:
        print('NONNUM', c, s.dtype)
    if np.issubdtype(s.dtype, np.number):
        if s.isna().all():
            print('ALLNAN', c)
        elif s.isna().mean() > 0.9:
            print('MOSTNAN %.2f' % s.isna().mean(), c)

# check target NaN rows
print('rows', len(df), 'nan y', int(df.future_spend_4w.isna().sum()))
print(df[df.future_spend_4w.isna()]['snapshot_day'].value_counts())


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

t = load_saved('e017_xsec_rank.parquet')
tt = train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
df = df[df['future_spend_4w'].notna()].copy()

def prep(d):
    X = d[feat_cols].copy()
    for c in feat_cols:
        if X[c].dtype == bool: X[c] = X[c].astype(float)
        elif not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
    return X.values.astype(float)

def fit_ridge(X, y, lam=100.0):
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd==0]=1.0
    Z = (X-mu)/sd; Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    A = Z.T@Z + lam*np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T@y), mu, sd

def apply_ridge(w, mu, sd, X):
    Z = (X-mu)/sd; Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z@w

y = df['future_spend_4w'].values.astype(float)
days = df['snapshot_day'].values
X = prep(df)

tr = np.isin(days, [95,123,151,179,207,235,263,291,319,347,375,403,431])
va = np.isin(days, [459,487,515,543])
for lam in [10, 30, 100, 300, 1000]:
    w, mu, sd = fit_ridge(X[tr], y[tr], lam=lam)
    pv = apply_ridge(w, mu, sd, X[va])
    print('lam %5d -> replica val MAE %.3f' % (lam, np.abs(pv-y[va]).mean()))

tr2 = np.isin(days, [95,123,151,179,207,235,263,291,319,347,375])
va2 = np.isin(days, [403,431])
w2, mu2, sd2 = fit_ridge(X[tr2], y[tr2])
pv2 = apply_ridge(w2, mu2, sd2, X[va2])
print('pseudo-val MAE (403,431): %.3f' % np.abs(pv2-y[va2]).mean())


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

tt = train_targets()
KEY = ['household_key','snapshot_day']

def pseudo_mae(path, lam=100.0, holdout=(403,431)):
    t = load_saved(path).merge(tt, on=KEY, how='left')
    t = t[t['future_spend_4w'].notna()]
    fc = [c for c in t.columns if c not in KEY+['future_spend_4w']]
    X = t[fc].copy()
    for c in fc:
        if X[c].dtype == bool: X[c] = X[c].astype(float)
        elif not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
    X = X.values.astype(float)
    y = t['future_spend_4w'].values.astype(float)
    d = t['snapshot_day'].values
    tr = ~np.isin(d, holdout); va = np.isin(d, holdout)
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[sd==0]=1
    Z = np.nan_to_num((X-mu)/sd)
    A = Z[tr].T@Z[tr] + lam*np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T@y[tr])
    pv = Z[va]@w
    return np.abs(pv-y[va]).mean()

harness = {'e001_recent_behavior.parquet':63.574,'e010_lifecycle.parquet':62.702,
           'e011_discounts.parquet':62.651,'e015_full_superset.parquet':62.647,
           'e016_churn_gapratio.parquet':62.535,'e017_xsec_rank.parquet':62.526,
           'e009_ar_season.parquet':66.683,'e012_hh_target_enc.parquet':72.176}
res = []
for p,h in harness.items():
    m = pseudo_mae(p)
    res.append((m,h,p))
    print('%-32s pseudo %8.3f   harness %8.3f' % (p, m, h))
res.sort()
print('\npseudo rank:', [r[2][:4] for r in res])
print('harness rank:', [r[2][:4] for r in sorted(res, key=lambda r: r[1])])
print('corr check: pseudo best =', res[0][2], '| pseudo worst =', res[-1][2])


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

tt = train_targets()
KEY = ['household_key','snapshot_day']

def prep(path):
    t = load_saved(path).merge(tt, on=KEY, how='left')
    t = t[t['future_spend_4w'].notna()]
    fc = [c for c in t.columns if c not in KEY+['future_spend_4w']]
    X = t[fc].copy()
    for c in fc:
        if X[c].dtype == bool: X[c] = X[c].astype(float)
        elif not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
    return X.values.astype(float), t['future_spend_4w'].values.astype(float), t['snapshot_day'].values

paths = {'e001':'e001_recent_behavior.parquet','e010':'e010_lifecycle.parquet',
         'e011':'e011_discounts.parquet','e015':'e015_full_superset.parquet',
         'e016':'e016_churn_gapratio.parquet','e017':'e017_xsec_rank.parquet',
         'e009':'e009_ar_season.parquet','e012':'e012_hh_target_enc.parquet'}
harness = {'e001':63.574,'e010':62.702,'e011':62.651,'e015':62.647,
           'e016':62.535,'e017':62.526,'e009':66.683,'e012':72.176}

data = {k: prep(v) for k,v in paths.items()}

def pseudo_mae(X, y, d, lam, holdout=(403,431)):
    tr = ~np.isin(d, holdout); va = np.isin(d, holdout)
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[sd==0]=1
    Z = np.nan_to_num((X-mu)/sd)
    A = Z[tr].T@Z[tr] + lam*np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T@y[tr])
    return np.abs(Z[va]@w - y[va]).mean()

# baseline: predict train median
X0,y0,d0 = data['e017']
tr0 = ~np.isin(d0,(403,431)); va0 = np.isin(d0,(403,431))
print('const-median baseline pseudo MAE: %.2f' % np.abs(np.median(y0[tr0])-y0[va0]).mean())

for lam in [300, 1000, 3000, 10000, 30000]:
    row = []
    for k in paths:
        X,y,d = data[k]
        row.append((pseudo_mae(X,y,d,lam), k))
    row.sort()
    print('lam %6d:' % lam, '  '.join('%s:%.1f' % (k,m) for m,k in row))
print('harness:  ', '  '.join('%s:%.1f' % (k,v) for k,v in sorted(harness.items(), key=lambda kv: kv[1])))


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

tt = train_targets()
KEY = ['household_key','snapshot_day']
t = load_saved('e017_xsec_rank.parquet').merge(tt, on=KEY, how='left')
t = t[t['future_spend_4w'].notna()]
fc = [c for c in t.columns if c not in KEY+['future_spend_4w']]
X = t[fc].copy()
for c in fc:
    if X[c].dtype == bool: X[c] = X[c].astype(float)
    elif not np.issubdtype(X[c].dtype, np.number): X[c] = pd.Categorical(X[c]).codes.astype(float)
X = X.values.astype(float); y = t['future_spend_4w'].values.astype(float); d = t['snapshot_day'].values

tr = d <= 375; va = np.isin(d, (403,431))
mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[sd==0]=1
Z = np.nan_to_num((X-mu)/sd)
A = Z[tr].T@Z[tr] + 3000*np.eye(Z.shape[1])
w = np.linalg.solve(A, Z[tr].T@y[tr])
pred = Z@w

dfp = pd.DataFrame({'d':d, 'y':y, 'p':pred})
print(dfp.groupby('d').agg(y_mean=('y','mean'), p_mean=('p','mean'), mae=('p', lambda p: np.nan)).shape)
g = dfp.groupby('d').apply(lambda g: pd.Series({'y_mean':g.y.mean(),'p_mean':g.p.mean(),'mae':(g.p-g.y).abs().mean()}))
print(g.round(1))

# simple model: spend_28 alone
i = fc.index('spend_28')
x1 = np.nan_to_num((X[:,i]-mu[i])/sd[i])
A1 = x1[tr].T@x1[tr] + 3000
w1 = np.linalg.solve(A1, x1[tr]@y[tr])
p1 = x1*w1
dfp['p1'] = p1
g1 = dfp.groupby('d').apply(lambda g: pd.Series({'mae1':(g.p1-g.y).abs().mean(),'corr':g.p1.corr(g.y)}))
print(g1.round(2))
print('overall corr(spend_28, y) by day:')
print(t.assign(x=X[:,i]).groupby('snapshot_day').apply(lambda g: g['x'].corr(g['future_spend_4w'])).round(3))


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

tt = train_targets()
KEY = ['household_key','snapshot_day']

def prep(path):
    t = load_saved(path).merge(tt, on=KEY, how='left')
    t = t[t['future_spend_4w'].notna()]
    fc = [c for c in t.columns if c not in KEY+['future_spend_4w']]
    X = t[fc].copy()
    for c in fc:
        if X[c].dtype == bool: X[c] = X[c].astype(float)
        elif not np.issubdtype(X[c].dtype, np.number): X[c] = pd.Categorical(X[c]).codes.astype(float)
    return X.values.astype(float), t['future_spend_4w'].values.astype(float), t['snapshot_day'].values

paths = {'e001':'e001_recent_behavior.parquet','e010':'e010_lifecycle.parquet',
         'e011':'e011_discounts.parquet','e015':'e015_full_superset.parquet',
         'e016':'e016_churn_gapratio.parquet','e017':'e017_xsec_rank.parquet',
         'e009':'e009_ar_season.parquet','e012':'e012_hh_target_enc.parquet'}
harness = {'e001':63.574,'e010':62.702,'e011':62.651,'e015':62.647,
           'e016':62.535,'e017':62.526,'e009':66.683,'e012':72.176}
data = {k: prep(v) for k,v in paths.items()}

def pseudo_mae(X, y, d, lam, holdout=(403,431)):
    tr = ~np.isin(d, holdout); va = np.isin(d, holdout)
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[sd==0]=1
    Z = np.nan_to_num((X-mu)/sd)
    Zt = np.hstack([Z[tr], np.ones((tr.sum(),1))]); Zv = np.hstack([Z[va], np.ones((va.sum(),1))])
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1] -= lam  # no penalty on intercept
    w = np.linalg.solve(A, Zt.T@y[tr])
    return np.abs(Zv@w - y[va]).mean()

for lam in [100, 300, 1000, 3000, 10000]:
    row = sorted((pseudo_mae(*data[k], lam), k) for k in paths)
    print('lam %6d:' % lam, '  '.join('%s:%.1f' % (k,m) for m,k in row))
print('harness:  ', '  '.join('%s:%.1f' % (k,v) for k,v in sorted(harness.items(), key=lambda kv: kv[1])))


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

tt = train_targets(); KEY = ['household_key','snapshot_day']
def prep(path):
    t = load_saved(path).merge(tt, on=KEY, how='left')
    t = t[t['future_spend_4w'].notna()]
    fc = [c for c in t.columns if c not in KEY+['future_spend_4w']]
    X = t[fc].copy()
    for c in fc:
        if X[c].dtype == bool: X[c] = X[c].astype(float)
        elif not np.issubdtype(X[c].dtype, np.number): X[c] = pd.Categorical(X[c]).codes.astype(float)
    return X.values.astype(float), t['future_spend_4w'].values.astype(float), t['snapshot_day'].values

paths = {'e001':'e001_recent_behavior.parquet','e010':'e010_lifecycle.parquet',
         'e011':'e011_discounts.parquet','e015':'e015_full_superset.parquet',
         'e016':'e016_churn_gapratio.parquet','e017':'e017_xsec_rank.parquet',
         'e009':'e009_ar_season.parquet','e012':'e012_hh_target_enc.parquet'}
harness = {'e001':63.574,'e010':62.702,'e011':62.651,'e015':62.647,
           'e016':62.535,'e017':62.526,'e009':66.683,'e012':72.176}
data = {k: prep(v) for k,v in paths.items()}
TR_SNAPS = [95,123,151,179,207,235,263,291,319,347,375,403,431]
lams = [3,10,30,100,300,1000,3000,10000]

def ridge_fit_pred(X, y, tr, va, lam):
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[sd==0]=1
    Z = np.nan_to_num((X-mu)/sd)
    Zt = np.hstack([Z[tr], np.ones((tr.sum(),1))]); Zv = np.hstack([Z[va], np.ones((va.sum(),1))])
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1] -= lam
    w = np.linalg.solve(A, Zt.T@y[tr])
    return Zv@w

print('%-6s %-9s %-9s %s' % ('tbl','cvlam','pseudo','harness'))
out={}
for k in paths:
    X,y,d = data[k]
    # snapshot-level CV within train to pick lam
    cvs = {}
    for lam in lams:
        errs=[]
        for hs in TR_SNAPS:
            tr = np.isin(d, [s for s in TR_SNAPS if s!=hs]); va = d==hs
            p = ridge_fit_pred(X,y,tr,va,lam)
            errs.append(np.abs(p-y[va]).mean())
        cvs[lam]=np.mean(errs)
    best = min(cvs, key=cvs.get)
    tr = np.isin(d, TR_SNAPS[:-2]); va = np.isin(d, [403,431])
    pm = np.abs(ridge_fit_pred(X,y,tr,va,best)-y[va]).mean()
    out[k]=pm
    print('%-6s %-9d %-9.3f %-9.3f  (cv by lam: %s)' % (k, best, pm, harness[k],
          ' '.join('%d:%.2f'%(l,c) for l,c in cvs.items() if l in (30,300,3000,10000))))
print('\npseudo rank:', sorted(out, key=out.get))
print('harness rank:', sorted(harness, key=harness.get))


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import build_features, save_table

def fn(view, snapshot_day):
    hh = view.households
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby('household_key')
    day = view.day

    def win(d0, d1):
        t = tx[(tx.day >= day-d0) & (tx.day <= day-d1)]
        return t

    f = pd.DataFrame(index=hh)
    # trailing windows
    for name,(d0,d1) in {'w28':(28,1),'w84':(84,1),'w364':(364,1)}.items():
        t = win(d0,d1)
        f['spend_'+name] = t.groupby('household_key').sales_value.sum()
        f['baskets_'+name] = t.groupby('household_key').basket_id.nunique()
    t28 = win(28,1)
    f['dsl'] = day - g.day.max()
    f['max_basket_28'] = t28.groupby(['household_key','basket_id']).sales_value.sum().groupby('household_key').max()
    f['zero_wk_share_12'] = 1 - (win(84,1).groupby('household_key').week_no.nunique()/12.0)

    # marketing
    ct = view.table('campaign_targets')
    ct = ct[ct.household_key.isin(hh)]
    desc = ct.groupby('household_key').description.apply(lambda s: ' '.join(sorted(set(s))))
    for typ in ['TypeA','TypeB','TypeC']:
        f['camp_'+typ] = desc.str.contains(typ).astype(float)
    f['camp_n'] = ct.groupby('household_key').campaign.nunique()
    cr = view.table('coupon_redemptions')
    cr = cr[cr.household_key.isin(hh)]
    f['days_since_redem'] = day - cr.groupby('household_key').day.max()
    f['coup_trips_84'] = win(84,1).query('coupon_disc != 0').groupby('household_key').basket_id.nunique()

    # interactions (tree-friendly raw products)
    f['s28_x_dsl'] = f['spend_w28'].fillna(0) * f['dsl'].fillna(999).clip(0,120)
    f['s28_x_campA'] = f['spend_w28'].fillna(0) * f['camp_TypeA']
    f['ew28'] = 0.0
    for d0,d1,hw in [(7,1,1.0),(14,8,0.5),(28,15,0.25),(56,29,0.125),(84,57,0.0625)]:
        f['ew28'] += hw * win(d0,d1).groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    f['ew28_x_dsl'] = f['ew28'] * f['dsl'].fillna(999).clip(0,120)

    # display/mailer exposure: products this hh bought, at hh's top store, last 4 weeks
    wk = (day+8)//7
    dm = view.table('display_mailer')
    top_store = win(84,1).groupby('household_key').store_id.agg(lambda s: s.value_counts().index[0])
    prods = win(84,1).groupby('household_key').product_id.apply(lambda s: set(s.unique()))
    dm4 = dm[(dm.week_no > wk-4) & (dm.week_no <= wk)]
    dm4 = dm4[dm4.display.notna() | dm4.mailer.notna()]
    dmg = dm4.groupby('store_id').product_id.apply(lambda s: set(s.unique()))
    f['disp_exposure'] = [len(prods.get(h,set()) & dmg.get(st,set())) for h,st in top_store.items()]
    f = f.reindex(hh)
    return f

tab = build_features(fn)
print(tab.shape)
fc = [c for c in tab.columns if c not in ('household_key','snapshot_day')]
print(tab[fc].isna().mean().round(3))
p = save_table(tab, 'e018_tree_feats')
print(p)


# ---- cell ----
import numpy as np, pandas as pd
from agent_api import build_features, save_table

def fn(view, snapshot_day):
    hh = view.households
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby('household_key')
    day = view.day

    def win(d0, d1):
        return tx[(tx.day >= day-d0) & (tx.day <= day-d1)]

    f = pd.DataFrame(index=hh)
    for name,(d0,d1) in {'w28':(28,1),'w84':(84,1),'w364':(364,1)}.items():
        t = win(d0,d1)
        f['spend_'+name] = t.groupby('household_key').sales_value.sum()
        f['baskets_'+name] = t.groupby('household_key').basket_id.nunique()
    t28 = win(28,1)
    f['dsl'] = day - g.day.max()
    f['max_basket_28'] = t28.groupby(['household_key','basket_id']).sales_value.sum().groupby('household_key').max()
    f['zero_wk_share_12'] = 1 - (win(84,1).groupby('household_key').week_no.nunique()/12.0)

    ct = view.table('campaign_targets')
    ct = ct[ct.household_key.isin(hh)]
    desc = ct.groupby('household_key').description.apply(lambda s: ' '.join(sorted(set(s))))
    for typ in ['TypeA','TypeB','TypeC']:
        f['camp_'+typ] = desc.str.contains(typ).astype(float)
    f['camp_n'] = ct.groupby('household_key').campaign.nunique()
    cr = view.table('coupon_redemptions')
    cr = cr[cr.household_key.isin(hh)]
    f['days_since_redem'] = day - cr.groupby('household_key').day.max()
    f['coup_trips_84'] = win(84,1).query('coupon_disc != 0').groupby('household_key').basket_id.nunique()

    f['s28_x_dsl'] = f['spend_w28'].fillna(0) * f['dsl'].fillna(999).clip(0,120)
    f['s28_x_campA'] = f['spend_w28'].fillna(0) * f['camp_TypeA']
    f['ew28'] = 0.0
    for d0,d1,hw in [(7,1,1.0),(14,8,0.5),(28,15,0.25),(56,29,0.125),(84,57,0.0625)]:
        f['ew28'] += hw * win(d0,d1).groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    f['ew28_x_dsl'] = f['ew28'] * f['dsl'].fillna(999).clip(0,120)

    wk = (day+8)//7
    dm = view.table('display_mailer')
    t84 = win(84,1)
    top_store = t84.groupby('household_key').store_id.agg(lambda s: s.value_counts().index[0]).reindex(hh)
    prods = t84.groupby('household_key').product_id.agg(set).reindex(hh).apply(lambda v: v if isinstance(v,set) else set())
    dm4 = dm[(dm.week_no > wk-4) & (dm.week_no <= wk)]
    dm4 = dm4[dm4.display.notna() | dm4.mailer.notna()]
    dmg = dm4.groupby('store_id').product_id.agg(set)
    f['disp_exposure'] = [len(prods.get(h,set()) & dmg.get(st,set())) if pd.notna(st) else 0 for h,st in top_store.items()]
    return f.reindex(hh)

tab = build_features(fn)
print(tab.shape)
fc = [c for c in tab.columns if c not in ('household_key','snapshot_day')]
print(tab[fc].isna().mean().round(3).to_string())
p = save_table(tab, 'e018_tree_feats')
print(p)
