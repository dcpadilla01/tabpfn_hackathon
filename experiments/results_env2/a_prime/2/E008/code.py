
import agent_api, pandas as pd, numpy as np
e5 = agent_api.load_saved('e005_decay_gapcv.parquet')
e6 = agent_api.load_saved('e006_momentum_seasonal.parquet')
print('E5 columns (%d):' % len(e5.columns))
print(e5.columns.tolist())
print('E6 extra:', [c for c in e6.columns if c not in e5.columns])
tt = agent_api.train_targets()
df = e6.merge(tt, on=['household_key','snapshot_day'])
print('merged', df.shape)
y = df['future_spend_4w']
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print('zero share %.3f' % (y==0).mean())
num = df.drop(columns=['household_key','snapshot_day','future_spend_4w']).select_dtypes(include=[np.number])
cor = num.corrwith(y)
cor = cor.reindex(cor.abs().sort_values(ascending=False).index)
print(cor.round(3).to_string())
g = df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count'])
print(g.round(1).to_string())
print('global mean pred MAE %.2f' % (y - y.mean()).abs().mean())
for c in cor.index[:12]:
    print('naive MAE by %-16s %.2f' % (c, (y - df[c]).abs().mean()))


# ---- cell ----

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


# ---- cell ----

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
X = df[feats].replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median()).values.astype(float)
print('any nan', np.isnan(X).any(), 'any inf', np.isinf(X).any())

def ridge(Xtr, ytr, Xva, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    A = (Xtr-mu)/sd; B = (Xva-mu)/sd
    A = np.c_[np.ones(len(A)), A]; B = np.c_[np.ones(len(B)), B]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ytr)
    return B@w

for name, Xf in [('raw', X), ('log1p', np.log1p(np.clip(X,0,None)))]:
    for lam in [1.0, 10.0, 100.0, 1000.0]:
        p = ridge(Xf[tr], y[tr], Xf[va], lam)
        print('%-6s lam=%6.0f  val MAE %.3f' % (name, lam, np.abs(p-y[va]).mean()))

Xf = np.log1p(np.clip(X,0,None))
p = ridge(Xf[tr], y[tr], Xf[va], 10.0)
out = df.loc[va].copy(); out['pred']=p; out['y']=y[va]
print(out.groupby('snapshot_day')[['y','pred']].agg(['mean','median']).round(1))
# which features have inf/nan in raw
raw = df[feats]
bad = [(c, int(np.isinf(raw[c]).sum()), int(raw[c].isna().sum())) for c in feats if np.isinf(raw[c]).any() or raw[c].isna().any()]
print('bad cols:', bad)


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e5 = agent_api.load_saved('e005_decay_gapcv.parquet')
print('saved rows', len(e5), 'snapshot days', sorted(e5.snapshot_day.unique()))
tt = agent_api.train_targets()
df = e5.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e5.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
# internal CV: train on days <= 347, validate on 375/403/431
itr = df.snapshot_day<=347
iva = df.snapshot_day>=375
print('train rows', itr.sum(), 'pseudo-val rows', iva.sum())

X = df[feats].replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median()).values.astype(float)

def ridge(Xtr, ytr, Xva, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    A = (Xtr-mu)/sd; B = (Xva-mu)/sd
    A = np.c_[np.ones(len(A)), A]; B = np.c_[np.ones(len(B)), B]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ytr)
    return B@w

for name, Xf in [('raw', X), ('log1p', np.log1p(np.clip(X,0,None)))]:
    for lam in [1.0, 10.0, 100.0]:
        p = ridge(Xf[itr.values], y[itr.values], Xf[iva.values], lam)
        print('%-6s lam=%5.0f  pseudo-val MAE %.3f' % (name, lam, np.abs(p-y[iva.values]).mean()))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e5 = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
df = e5.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e5.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
itr = (df.snapshot_day<=347).values
iva = (df.snapshot_day>=375).values

Xraw = df[feats].replace([np.inf,-np.inf], np.nan)
tr_med = Xraw[itr].median()
X = Xraw.fillna(tr_med).values.astype(float)

def ridge(Xtr, ytr, Xva, lam, clip=8.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd<1e-8]=1.0
    A = np.clip((Xtr-mu)/sd, -clip, clip); B = np.clip((Xva-mu)/sd, -clip, clip)
    A = np.c_[np.ones(len(A)), A]; B = np.c_[np.ones(len(B)), B]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ytr)
    return B@w

for name, Xf in [('raw', X), ('log1p', np.log1p(np.clip(X,0,None)))]:
    for lam in [1.0, 10.0, 100.0]:
        p = ridge(Xf[itr], y[itr], Xf[iva], lam)
        print('%-6s lam=%5.0f  pseudo-val MAE %.3f' % (name, lam, np.abs(p-y[iva]).mean()))

# per-snapshot bias
Xf = np.log1p(np.clip(X,0,None))
p = ridge(Xf[itr], y[itr], Xf[iva], 10.0)
out = df.loc[iva].copy(); out['pred']=p
print(out.groupby('snapshot_day')[['future_spend_4w','pred']].agg(['mean','median']).round(1))
print('val mean y %.1f  mean pred %.1f' % (y[iva].mean(), p.mean()))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e5 = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
df = e5.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e5.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
itr = (df.snapshot_day<=347).values
iva = (df.snapshot_day>=375).values

Xraw = df[feats].replace([np.inf,-np.inf], np.nan)
tr_med = Xraw[itr].median()
X = Xraw.fillna(tr_med).fillna(0.0).values.astype(float)
print('nan left:', np.isnan(X).any())

def ridge(Xtr, ytr, Xva, lam, clip=8.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd<1e-8]=1.0
    A = np.clip((Xtr-mu)/sd, -clip, clip); B = np.clip((Xva-mu)/sd, -clip, clip)
    A = np.c_[np.ones(len(A)), A]; B = np.c_[np.ones(len(B)), B]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ytr)
    return B@w

for name, Xf in [('raw', X), ('log1p', np.log1p(np.clip(X,0,None)))]:
    for lam in [1.0, 10.0, 100.0]:
        p = ridge(Xf[itr], y[itr], Xf[iva], lam)
        print('%-6s lam=%5.0f  pseudo-val MAE %.3f' % (name, lam, np.abs(p-y[iva]).mean()))

Xf = np.log1p(np.clip(X,0,None))
p = ridge(Xf[itr], y[itr], Xf[iva], 10.0)
out = df.loc[iva].copy(); out['pred']=p
print(out.groupby('snapshot_day')[['future_spend_4w','pred']].agg(['mean','median']).round(1))
print('val mean y %.1f  mean pred %.1f' % (y[iva].mean(), p.mean()))


# ---- cell ----

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


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
v = agent_api.snapshot(459)
prods = v.products
print(prods.department.value_counts().head(20))
print(prods.brand.value_counts())
tx = v.transactions
print(tx.shape)
print(tx.head(3))
print('neg sales_value share %.4f' % (tx.sales_value<0).mean())
print('zero sales share %.4f' % (tx.sales_value==0).mean())
h = agent_api.history(tx.household_key.iloc[0], 459)
print(h.head(3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')

def make_features(view, snapshot_day):
    t = view.table('transactions')
    t = t[t.day <= snapshot_day]
    end = snapshot_day
    w = {}
    def agg(hh):
        g = t[t.household_key==hh]
        return g
    # vectorized per-window sums
    for name, lo in [('spend_7',7),('spend_14',14),('spend_28',28),('spend_56',56),('spend_84',84),('spend_180',180),('spend_365',365)]:
        s = t[(t.day>end-lo)&(t.day<=end)].groupby('household_key')['sales_value'].sum()
        w[name] = s
    for name, lo in [('spend_28_prior',56),('spend_84_prior',168)]:
        s = t[(t.day>end-lo)&(t.day<=end-lo//2)].groupby('household_key')['sales_value'].sum()
        w[name] = s
    b = t[(t.day>end-84)&(t.day<=end)].groupby('household_key').agg(
        baskets_84=('basket_id','nunique'), days_since_last=('day','max'),
        days_since_first=('day','min'), n_products_84=('product_id','nunique'),
        n_stores_84=('store_id','nunique'))
    b['days_since_last'] = end - b['days_since_last']
    b['days_since_first'] = end - b['days_since_first']
    b['baskets_28'] = t[(t.day>end-28)&(t.day<=end)].groupby('household_key')['basket_id'].nunique()
    b['avg_basket_84'] = b['baskets_84'].where(b['baskets_84']>0, np.nan)
    b['avg_basket_84'] = w['spend_84']/b['baskets_84'].replace(0,np.nan)
    b['trips_per_wk_84'] = b['baskets_84']/12.0
    b['spend_28_ratio'] = w['spend_28']/w['spend_28_prior'].replace(0,np.nan)
    out = pd.DataFrame(w).join(b)
    out['spend_trend'] = (w['spend_28']-w['spend_28_prior'])/28.0
    out['active_28'] = (out['baskets_28']>0).astype(float)
    out = out.reindex(view.households).fillna({'spend_7':0.0,'spend_14':0.0,'spend_28':0.0,'spend_28_prior':0.0})
    return out

t0 = time.time()
F = agent_api.build_features(make_features)
print('build_features took %.1fs, shape' % (time.time()-t0), F.shape)
print(F.head())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')

E5COLS = ['spend_7','spend_14','spend_28','spend_56','spend_84','spend_180','spend_365','spend_28_prior','spend_84_prior',
'baskets_28','baskets_84','days_since_last','days_since_first','avg_basket_84','trips_per_wk_84','spend_28_ratio',
'n_products_84','n_stores_84','spend_trend','active_28','ew_7','ew_14','ew_28','ew_56','ew_84','ew_180',
'spend_lag336','spend_lag364','spend_lag392','longrun_wk','ratio28_lr','ratio84_lr','basket_max_84','basket_std_84',
'basket_med_84','active_days_28','gap_cv']

def make_features(view, snapshot_day):
    end = snapshot_day
    t = view.table('transactions'); t = t[t.day<=end]
    hh_all = view.households
    out = pd.DataFrame(index=hh_all)
    def wsum(lo, hi=None):
        hi = end if hi is None else hi
        return t[(t.day>end-lo)&(t.day<=hi)].groupby('household_key')['sales_value'].sum()
    for name, lo in [('spend_7',7),('spend_14',14),('spend_28',28),('spend_56',56),('spend_84',84),('spend_180',180),('spend_365',365)]:
        out[name] = wsum(lo)
    out['spend_28_prior'] = wsum(56, end-28)
    out['spend_84_prior'] = wsum(168, end-84)
    d = t.groupby(['household_key','day'])['sales_value'].sum().reset_index()
    d['age'] = end - d['day']
    for hl in [7,14,28,56,84,180]:
        wgt = 0.5**(d['age']/hl)
        out['ew_%d'%hl] = (d['sales_value']*wgt).groupby(d['household_key']).sum()
    for lag in [336,364,392]:
        out['spend_lag%d'%lag] = t[(t.day>end-lag-28)&(t.day<=end-lag)].groupby('household_key')['sales_value'].sum()
    total = t.groupby('household_key')['sales_value'].sum()
    first_day = t.groupby('household_key')['day'].min()
    tenure_wk = ((end-first_day)/7.0).replace(0,np.nan)
    out['longrun_wk'] = total/tenure_wk
    out['ratio28_lr'] = out['spend_28']/(out['longrun_wk']*4).replace(0,np.nan)
    out['ratio84_lr'] = out['spend_84']/(out['longrun_wk']*12).replace(0,np.nan)
    b84 = t[(t.day>end-84)&(t.day<=end)]
    bsum = b84.groupby(['household_key','basket_id'])['sales_value'].sum()
    bcnt = bsum.groupby('household_key').agg(baskets_84='size', basket_max_84='max', basket_std_84='std', basket_med_84='median')
    out = out.join(bcnt)
    for name, lo in [('baskets_28',28),('baskets_14',14),('baskets_7',7)]:
        out[name] = t[(t.day>end-lo)&(t.day<=end)].groupby('household_key')['basket_id'].nunique()
    out['days_since_last'] = end - t.groupby('household_key')['day'].max()
    out['days_since_first'] = end - first_day
    out['avg_basket_84'] = out['spend_84']/out['baskets_84'].replace(0,np.nan)
    out['trips_per_wk_84'] = out['baskets_84']/12.0
    out['spend_28_ratio'] = out['spend_28']/out['spend_28_prior'].replace(0,np.nan)
    out['n_products_84'] = b84.groupby('household_key')['product_id'].nunique()
    out['n_stores_84'] = b84.groupby('household_key')['store_id'].nunique()
    out['spend_trend'] = (out['spend_28']-out['spend_28_prior'])/28.0
    out['active_28'] = (out['baskets_28']>0).astype(float)
    out['active_days_28'] = t[(t.day>end-28)&(t.day<=end)].groupby('household_key')['day'].nunique()
    td = t[(t.day>end-180)&(t.day<=end)].groupby('household_key')['day'].unique()
    gaps = td.apply(lambda a: np.diff(np.sort(a)) if len(a)>1 else np.array([]))
    out['gap_mean_180'] = gaps.apply(lambda a: a.mean() if len(a)>0 else np.nan)
    out['gap_cv'] = gaps.apply(lambda a: (a.std()/a.mean()) if len(a)>1 and a.mean()>0 else np.nan)
    out['max_gap_180'] = gaps.apply(lambda a: a.max() if len(a)>0 else np.nan)
    out['recency_ratio'] = out['days_since_last']/out['gap_mean_180']
    blk = (end - t['day'])//7
    wk = t[blk<26].groupby(['household_key', blk[blk<26]])['sales_value'].sum()
    out['zero_weeks_26'] = 1 - wk.groupby('household_key').size()/26.0
    b28s = t[(t.day>end-28)&(t.day<=end)].groupby(['household_key','basket_id'])['sales_value'].sum()
    out['avg_basket_28'] = b28s.groupby('household_key').mean()
    out['basket_28_vs_84'] = out['avg_basket_28']/out['avg_basket_84']
    lines = b84.groupby('household_key').size()
    qty = b84.groupby('household_key')['quantity'].sum()
    out['lines_per_basket_84'] = lines/out['baskets_84'].replace(0,np.nan)
    out['units_per_basket_84'] = qty/out['baskets_84'].replace(0,np.nan)
    out['unit_price_84'] = out['spend_84']/qty.replace(0,np.nan)
    disc = b84.groupby('household_key')[['coupon_disc','retail_disc','coupon_match_disc']].sum().sum(axis=1)
    out['disc_share_84'] = -disc/out['spend_84'].replace(0,np.nan)
    out['zero_line_share_84'] = (b84['sales_value']==0).groupby(b84['household_key']).mean()
    prod = view.table('products')[['product_id','brand','department']]
    b84b = b84.merge(prod, on='product_id', how='left')
    pb = b84b[b84b.brand=='Private'].groupby('household_key')['sales_value'].sum()
    out['private_share_84'] = pb/out['spend_84'].replace(0,np.nan)
    for dep in ['GROCERY','DRUG GM','PRODUCE','MEAT','DELI','PASTRY','COSMETICS']:
        ds = b84b[b84b.department==dep].groupby('household_key')['sales_value'].sum()
        out['dep_%s'%dep[:6].replace(' ','')] = ds/out['spend_84'].replace(0,np.nan)
    p84s = b84.groupby(['household_key','product_id'])['sales_value'].sum()
    prev = t[(t.day>end-365)&(t.day<=end-84)].groupby(['household_key','product_id']).size().rename('p')
    j = p84s.to_frame('s').join(prev, how='left')
    rep = j[j.p.notna()].groupby('household_key')['s'].sum()
    out['repeat_share_84'] = rep/out['spend_84'].replace(0,np.nan)
    ss = b84.groupby(['household_key','store_id'])['sales_value'].sum()
    out['top_store_share_84'] = ss.groupby('household_key').max()/out['spend_84'].replace(0,np.nan)
    out['sin_y'] = np.sin(2*np.pi*end/364); out['cos_y'] = np.cos(2*np.pi*end/364)
    out['sin_q'] = np.sin(2*np.pi*end/91); out['cos_q'] = np.cos(2*np.pi*end/91)
    out['day_idx'] = end/100.0
    out['ix_ew28_trips'] = out['ew_28']*out['trips_per_wk_84']
    out['ix_spend84_active'] = out['spend_84']*out['active_28']
    out['ix_recency_ew'] = out['ew_28']*np.exp(-out['days_since_last']/28.0)
    return out.reindex(hh_all)

t0=time.time()
F = agent_api.build_features(make_features)
print('built in %.0fs' % (time.time()-t0), F.shape)
missing = [c for c in E5COLS if c not in F.columns]
print('missing E5 cols:', missing)
agent_api.save_table(F, 'e008_candidate')

# proxy CV
tt = agent_api.train_targets()
df = F.merge(tt, on=['household_key','snapshot_day'])
itr = (df.snapshot_day<=347).values; iva = (df.snapshot_day>=375).values
y = df['future_spend_4w'].values
def ridge(Xtr,ytr,Xva,lam,clip=8.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-8]=1
    A=np.clip((Xtr-mu)/sd,-clip,clip); B=np.clip((Xva-mu)/sd,-clip,clip)
    A=np.c_[np.ones(len(A)),A]; B=np.c_[np.ones(len(B)),B]
    return B@np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@ytr)
def ev(cols):
    X = df[cols].replace([np.inf,-np.inf],np.nan)
    X = X.fillna(X[itr].median()).fillna(0.0).values.astype(float)
    p = ridge(X[itr],y[itr],X[iva],100.0)
    return np.abs(p-y[iva]).mean()

groups = {
 'A_recency': ['baskets_7','baskets_14','active_days_14','recency_ratio','max_gap_180','gap_mean_180'],
 'B_intermittent': ['zero_weeks_26'],
 'C_basket': ['avg_basket_28','basket_28_vs_84','lines_per_basket_84','units_per_basket_84','unit_price_84'],
 'D_disc_brand': ['disc_share_84','zero_line_share_84','private_share_84'],
 'E_mix': ['dep_GROCER','dep_DRUGG','dep_PRODUC','dep_MEAT','dei_DELI','dep_PASTRY','dep_COSMET'],
 'F_loyalty': ['repeat_share_84','top_store_share_84'],
 'G_season': ['sin_y','cos_y','sin_q','cos_q','day_idx'],
 'H_interact': ['ix_ew28_trips','ix_spend84_active','ix_recency_ew'],
}
base = [c for c in E5COLS if c in F.columns]
print('rebuild-E005 proxy MAE %.3f (n=%d)' % (ev(base), len(base)))
allnew = []
for g,cols in groups.items():
    cols = [c for c in cols if c in F.columns]
    allnew += cols
    print('%-14s +%-3d -> %.3f' % (g, len(cols), ev(base+cols)))
print('ALL new (%d) -> %.3f' % (len(allnew), ev(base+allnew)))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('e008_candidate.parquet')
tt = agent_api.train_targets()
df = F.merge(tt, on=['household_key','snapshot_day'])
itr = (df.snapshot_day<=347).values; iva = (df.snapshot_day>=375).values
y = df['future_spend_4w'].values
E5COLS = ['spend_7','spend_14','spend_28','spend_56','spend_84','spend_180','spend_365','spend_28_prior','spend_84_prior',
'baskets_28','baskets_84','days_since_last','days_since_first','avg_basket_84','trips_per_wk_84','spend_28_ratio',
'n_products_84','n_stores_84','spend_trend','active_28','ew_7','ew_14','ew_28','ew_56','ew_84','ew_180',
'spend_lag336','spend_lag364','spend_lag392','longrun_wk','ratio28_lr','ratio84_lr','basket_max_84','basket_std_84',
'basket_med_84','active_days_28','gap_cv']
def ridge(Xtr,ytr,Xva,lam,clip=8.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-8]=1
    A=np.clip((Xtr-mu)/sd,-clip,clip); B=np.clip((Xva-mu)/sd,-clip,clip)
    A=np.c_[np.ones(len(A)),A]; B=np.c_[np.ones(len(B)),B]
    return B@np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@ytr)
def ev(cols):
    X = df[cols].replace([np.inf,-np.inf],np.nan)
    X = X.fillna(X[itr].median()).fillna(0.0).values.astype(float)
    p = ridge(X[itr],y[itr],X[iva],100.0)
    return np.abs(p-y[iva]).mean()
base = E5COLS
print('base %.3f' % ev(base))
print('+day_idx      %.3f' % ev(base+['day_idx']))
print('+sin/cos yr   %.3f' % ev(base+['sin_y','cos_y']))
print('+sin/cos q    %.3f' % ev(base+['sin_q','cos_q']))
print('+all season   %.3f' % ev(base+['sin_y','cos_y','sin_q','cos_q','day_idx']))
print('+yr+day       %.3f' % ev(base+['sin_y','cos_y','day_idx']))
# combos
C = ['avg_basket_28','basket_28_vs_84','lines_per_basket_84','units_per_basket_84','unit_price_84']
D = ['disc_share_84','zero_line_share_84','private_share_84']
E = ['dep_GROCER','dep_DRUGG','dep_PRODUC','dep_MEAT','dei_DELI','dep_PASTRY','dep_COSMET']
S = ['sin_y','cos_y','sin_q','cos_q','day_idx']
print('+C+D+E+S      %.3f' % ev(base+C+D+E+S))
print('+C+S          %.3f' % ev(base+C+S))
print('+D+E+S        %.3f' % ev(base+D+E+S))
# seasonal interactions with spend level
df['ix_ew28_siny'] = df['ew_28']*df['sin_y']; df['ix_ew28_cosy'] = df['ew_28']*df['cos_y']
print('+S+ix(ew28*siny,cosy) %.3f' % ev(base+S+['ix_ew28_siny','ix_ew28_cosy']))
# second harmonic of year
df['sin_y2'] = np.sin(4*np.pi*df['snapshot_day']/364); df['cos_y2'] = np.cos(4*np.pi*df['snapshot_day']/364)
print('+S+harmonic2  %.3f' % ev(base+S+['sin_y2','cos_y2']))
# lag364_4w style: same-window last year is in E5 (spend_lag364). check val-only trend: per-snapshot mean target vs day
print(df.groupby('snapshot_day')['future_spend_4w'].mean().round(1).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('e008_candidate.parquet')
tt = agent_api.train_targets()
df = F.merge(tt, on=['household_key','snapshot_day'])
itr = (df.snapshot_day<=347).values; iva = (df.snapshot_day>=375).values
y = df['future_spend_4w'].values
E5COLS = ['spend_7','spend_14','spend_28','spend_56','spend_84','spend_180','spend_365','spend_28_prior','spend_84_prior',
'baskets_28','baskets_84','days_since_last','days_since_first','avg_basket_84','trips_per_wk_84','spend_28_ratio',
'n_products_84','n_stores_84','spend_trend','active_28','ew_7','ew_14','ew_28','ew_56','ew_84','ew_180',
'spend_lag336','spend_lag364','spend_lag392','longrun_wk','ratio28_lr','ratio84_lr','basket_max_84','basket_std_84',
'basket_med_84','active_days_28','gap_cv']
def ridge(Xtr,ytr,Xva,lam,clip=8.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-8]=1
    A=np.clip((Xtr-mu)/sd,-clip,clip); B=np.clip((Xva-mu)/sd,-clip,clip)
    A=np.c_[np.ones(len(A)),A]; B=np.c_[np.ones(len(B)),B]
    return B@np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@ytr)
def ev(cols):
    X = df[cols].replace([np.inf,-np.inf],np.nan)
    X = X.fillna(X[itr].median()).fillna(0.0).values.astype(float)
    p = ridge(X[itr],y[itr],X[iva],100.0)
    return np.abs(p-y[iva]).mean()
base = E5COLS
C = ['avg_basket_28','basket_28_vs_84','lines_per_basket_84','units_per_basket_84','unit_price_84']
D = ['disc_share_84','zero_line_share_84','private_share_84']
E = ['dep_GROCER','dep_DRUGG','dep_PRODUC','dep_MEAT','dep_DELI','dep_PASTRY','dep_COSMET']
S = ['sin_y','cos_y','sin_q','cos_q','day_idx']
Sy = ['sin_y','cos_y','day_idx']
print('base %.3f' % ev(base))
print('+C+D+E+S %.3f' % ev(base+C+D+E+S))
print('+C+S     %.3f' % ev(base+C+S))
print('+D+E+S   %.3f' % ev(base+D+E+S))
print('+S+ix    %.3f' % ev(base+S+['ix_ew28_siny','ix_ew28_cosy']))
print('+S+har2  %.3f' % ev(base+S+['sin_y2','cos_y2']))
# per-snapshot mean target
print(df.groupby('snapshot_day')['future_spend_4w'].mean().round(1).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('e008_candidate.parquet')
tt = agent_api.train_targets()
df = F.merge(tt, on=['household_key','snapshot_day'])
itr = (df.snapshot_day<=347).values; iva = (df.snapshot_day>=375).values
y = df['future_spend_4w'].values
E5COLS = ['spend_7','spend_14','spend_28','spend_56','spend_84','spend_180','spend_365','spend_28_prior','spend_84_prior',
'baskets_28','baskets_84','days_since_last','days_since_first','avg_basket_84','trips_per_wk_84','spend_28_ratio',
'n_products_84','n_stores_84','spend_trend','active_28','ew_7','ew_14','ew_28','ew_56','ew_84','ew_180',
'spend_lag336','spend_lag364','spend_lag392','longrun_wk','ratio28_lr','ratio84_lr','basket_max_84','basket_std_84',
'basket_med_84','active_days_28','gap_cv']
def ridge(Xtr,ytr,Xva,lam,clip=8.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-8]=1
    A=np.clip((Xtr-mu)/sd,-clip,clip); B=np.clip((Xva-mu)/sd,-clip,clip)
    A=np.c_[np.ones(len(A)),A]; B=np.c_[np.ones(len(B)),B]
    return B@np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@ytr)
def ev(cols):
    X = df[cols].replace([np.inf,-np.inf],np.nan)
    X = X.fillna(X[itr].median()).fillna(0.0).values.astype(float)
    p = ridge(X[itr],y[itr],X[iva],100.0)
    return np.abs(p-y[iva]).mean()
base = E5COLS
C = ['avg_basket_28','basket_28_vs_84','lines_per_basket_84','units_per_basket_84','unit_price_84']
S = ['sin_y','cos_y','sin_q','cos_q','day_idx']
Sy = ['sin_y','cos_y','day_idx']
df['sin_y2']=np.sin(4*np.pi*df['snapshot_day']/364); df['cos_y2']=np.cos(4*np.pi*df['snapshot_day']/364)
df['ix1']=df['ew_28']*df['sin_y']; df['ix2']=df['ew_28']*df['cos_y']
df['ix3']=df['spend_84']*df['sin_y']; df['ix4']=df['spend_84']*df['cos_y']
print('base+C+S      %.3f' % ev(base+C+S))
print('base+C+S+har2 %.3f' % ev(base+C+S+['sin_y2','cos_y2']))
print('base+C+S+ix12 %.3f' % ev(base+C+S+['ix1','ix2']))
print('base+C+S+ix34 %.3f' % ev(base+C+S+['ix3','ix4']))
print('base+C+Sy     %.3f' % ev(base+C+Sy))
print('base+Sy       %.3f' % ev(base+Sy))


# ---- cell ----

import agent_api, pandas as pd
F = agent_api.load_saved('e008_candidate.parquet')
E5COLS = ['spend_7','spend_14','spend_28','spend_56','spend_84','spend_180','spend_365','spend_28_prior','spend_84_prior',
'baskets_28','baskets_84','days_since_last','days_since_first','avg_basket_84','trips_per_wk_84','spend_28_ratio',
'n_products_84','n_stores_84','spend_trend','active_28','ew_7','ew_14','ew_28','ew_56','ew_84','ew_180',
'spend_lag336','spend_lag364','spend_lag392','longrun_wk','ratio28_lr','ratio84_lr','basket_max_84','basket_std_84',
'basket_med_84','active_days_28','gap_cv']
NEW = ['avg_basket_28','basket_28_vs_84','lines_per_basket_84','units_per_basket_84','unit_price_84','sin_y','cos_y','day_idx']
cols = ['household_key','snapshot_day'] + E5COLS + NEW
out = F[cols]
agent_api.save_table(out, 'e008_season_basket')
print(out.shape, out.columns.tolist())
