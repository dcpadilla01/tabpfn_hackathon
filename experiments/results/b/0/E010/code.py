import pandas as pd, numpy as np, agent_api

e7 = agent_api.load_saved('e007_lagseq.parquet')
print('e007 shape:', e7.shape)
cols = list(e7.columns)
print('n cols:', len(cols))
for i in range(0, len(cols), 6):
    print('  ' + ' | '.join(cols[i:i+6]))
print('snapshots:', sorted(e7.snapshot_day.unique()))
print('rows per snapshot:', e7.groupby('snapshot_day').size().to_dict())

tt = agent_api.train_targets()
print('\ntrain_targets shape:', tt.shape)
print(tt.future_spend_4w.describe())
print('zero frac:', (tt.future_spend_4w == 0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean', 'median', 'max']))

md = agent_api.load_saved('mkt_demo.parquet')
print('\nmkt_demo shape:', md.shape)
print(list(md.columns)[:40])


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

e7 = agent_api.load_saved('e007_lagseq.parquet')
tt = agent_api.train_targets()
df = e7.merge(tt, on=['household_key','snapshot_day'], how='inner')
print(df.shape)

ycol='future_spend_4w'
def prep(df):
    X = df.drop(columns=['household_key','snapshot_day',ycol], errors='ignore')
    cats=[]
    for c in X.columns:
        if X[c].dtype==object or str(X[c].dtype)=='category':
            cats.append(c)
    X = pd.get_dummies(X, columns=cats, dummy_na=True)
    X = X.astype(np.float64)
    # fillna with train medians
    med = X.median()
    X = X.fillna(med).fillna(0)
    return X

def ridge_fit(Xtr, ytr, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = (Xtr-mu)/sd
    Z = np.c_[np.ones(len(Z)), Z]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[0,0]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return w, mu, sd

def ridge_pred(w, mu, sd, X):
    Z = (X-mu)/sd
    Z = np.c_[np.ones(len(Z)), Z]
    return Z@w

X = prep(df)
y = df[ycol].values
sd_ = df.snapshot_day.values
print('X shape', X.shape)

for alpha in [1.0, 10.0, 100.0]:
    # fit on train snaps <=403, eval on 431
    m = sd_<=403
    w,mu,s = ridge_fit(X[m], y[m], alpha)
    p = ridge_pred(w,mu,s,X[sd_==431])
    mae = np.abs(p - y[sd_==431]).mean()
    # log target variant
    w2,mu2,s2 = ridge_fit(X[m], np.log1p(y[m]), alpha)
    p2 = np.expm1(ridge_pred(w2,mu2,s2,X[sd_==431]))
    mae2 = np.abs(p2 - y[sd_==431]).mean()
    print(f'alpha={alpha}: 431 MAE raw={mae:.3f}  logtgt={mae2:.3f}')


# ---- cell ----
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

snaps = agent_api.snapshot_days()['train'] + agent_api.snapshot_days()['validation']
print(snaps)

def build(view, s):
    hh = view.households
    tx = view.transactions
    tx = tx[tx.household_key.isin(hh)]
    out = pd.DataFrame(index=hh)
    # A: decay-weighted spend/trips last 56d
    t = tx[(tx.day > s-56)]
    g = t.groupby('household_key')
    dec = np.exp(-(s - t.day)/14.0)
    out['dec_spend14'] = (t.sales_value*dec).groupby(t.household_key).sum()
    dec7 = np.exp(-(s - t.day)/7.0)
    out['dec_spend7'] = (t.sales_value*dec7).groupby(t.household_key).sum()
    out['dec_trips'] = t.drop_duplicates('basket_id').assign(d=dec7.groupby(t.loc[t.drop_duplicates('basket_id').index,'household_key']).sum().reindex(t.drop_duplicates('basket_id').household_key).values).groupby('household_key').d.sum() if False else np.nan
    # simpler: recency-weighted trips
    bt = t.drop_duplicates('basket_id')[['household_key','day']]
    out['dec_trips'] = bt.assign(w=np.exp(-(s-bt.day)/14.0)).groupby('household_key').w.sum()
    # B: trip cadence last 112d
    t2 = tx[(tx.day > s-112)]
    b2 = t2.drop_duplicates('basket_id')[['household_key','day']].sort_values(['household_key','day'])
    b2['gap'] = b2.groupby('household_key').day.diff()
    gapstat = b2.groupby('household_key').gap.agg(['mean','std','median'])
    out['gap_mean'] = gapstat['mean']; out['gap_std'] = gapstat['std']; out['gap_med'] = gapstat['median']
    out['ntrip112'] = b2.groupby('household_key').size()
    # C: brand/deal traits last 112d
    prod = view.products
    m = t2.merge(prod[['product_id','brand']], on='product_id', how='left')
    sp = m.groupby('household_key').sales_value.sum()
    pl = m[m.brand=='Private'].groupby('household_key').sales_value.sum()
    out['pl_share'] = (pl/sp).fillna(0)
    out['disc_share2'] = (-(t2.coupon_disc+t2.coupon_match_disc+t2.retail_disc).groupby(t2.household_key).sum() / sp.clip(lower=0.01))
    # quantity and unit price
    out['qty112'] = t2.groupby('household_key').quantity.sum()
    out['unit_price'] = (sp / t2.groupby('household_key').quantity.sum().clip(lower=1))
    # D: macro trend: retailer-wide avg weekly spend per active hh, last 4w vs prior 52w
    allt = view.transactions
    w4 = allt[(allt.day > s-28)].sales_value.sum()/28.0
    w52 = allt[(allt.day > s-364)&(allt.day<=s-28)].sales_value.sum()/336.0
    out['macro_ratio'] = w4/(w52+1e-9)
    # E: interactions
    out['inter_spend_season'] = out.index.map(lambda k: np.nan)  # placeholder, add later in merge
    return out

frames=[]
for s in snaps:
    v = agent_api.snapshot(s)
    f = build(v, s)
    f['snapshot_day']=s
    frames.append(f.reset_index().rename(columns={'index':'household_key'}))
cand = pd.concat(frames, ignore_index=True)
print(cand.shape)
print(cand.head())
agent_api.save_table(cand, 'cand1')


# ---- cell ----
import agent_api
v = agent_api.snapshot(459)
print(type(v.households), len(v.households), v.households[:3])
print(type(v.day), v.day)
print(type(v.week), v.week)
print(type(v.transactions), v.transactions.shape)
print(type(v.products), v.products.shape)


# ---- cell ----
import agent_api
v = agent_api.snapshot(459)
print('households:', v.households)
print('day:', v.day, 'week:', v.week)
print('tx:', type(v.transactions))
print(v.transactions.head(2))
print('products:', type(v.products))
print(v.products.head(2))
print([a for a in dir(v) if not a.startswith('_')])


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def build(view, s):
    hh = view.households
    hhset = set(hh)
    tx = view.transactions
    tx = tx[tx.household_key.isin(hhset)]
    out = pd.DataFrame(index=hh)
    # A: decay-weighted spend last 56d
    t = tx[tx.day > s-56]
    dec = np.exp(-(s - t.day)/14.0)
    out['dec_spend14'] = (t.sales_value*dec).groupby(t.household_key).sum()
    dec7 = np.exp(-(s - t.day)/7.0)
    out['dec_spend7'] = (t.sales_value*dec7).groupby(t.household_key).sum()
    bt = t.drop_duplicates('basket_id')[['household_key','day']]
    out['dec_trips'] = bt.assign(w=np.exp(-(s-bt.day)/14.0)).groupby('household_key').w.sum()
    # B: trip cadence last 112d
    t2 = tx[tx.day > s-112]
    b2 = t2.drop_duplicates('basket_id')[['household_key','day']].sort_values(['household_key','day'])
    b2['gap'] = b2.groupby('household_key').day.diff()
    gs = b2.groupby('household_key').gap.agg(['mean','std','median'])
    out['gap_mean']=gs['mean']; out['gap_std']=gs['std']; out['gap_med']=gs['median']
    out['ntrip112'] = b2.groupby('household_key').size()
    # C: brand/deal traits last 112d
    prod = view.products[['product_id','brand']]
    m = t2.merge(prod, on='product_id', how='left')
    sp = m.groupby('household_key').sales_value.sum()
    pl = m[m.brand=='Private'].groupby('household_key').sales_value.sum()
    out['pl_share'] = (pl/sp).fillna(0)
    out['disc_share2'] = (-(t2.coupon_disc+t2.coupon_match_disc+t2.retail_disc).groupby(t2.household_key).sum()/sp.clip(lower=0.01))
    out['qty112'] = t2.groupby('household_key').quantity.sum()
    out['unit_price'] = sp/t2.groupby('household_key').quantity.sum().clip(lower=1)
    # D: macro trend
    allt = view.transactions
    w4 = allt[allt.day > s-28].sales_value.sum()/28.0
    w52 = allt[(allt.day > s-364)&(allt.day<=s-28)].sales_value.sum()/336.0
    out['macro_ratio'] = w4/(w52+1e-9)
    out['snapshot_day']=s
    return out.reset_index().rename(columns={'index':'household_key'})

cand = agent_api.build_features(build)
print(cand.shape)
print(cand.head(3))
agent_api.save_table(cand, 'cand1')


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet')
tt = agent_api.train_targets()
df = e7.merge(cand.drop(columns=['snapshot_day']), on='household_key', how='left').merge(tt, on=['household_key','snapshot_day'])
print(df.shape)

def prep(d):
    X = d.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    cats=[c for c in X.columns if X[c].dtype==object or str(X[c].dtype)=='category']
    X = pd.get_dummies(X, columns=cats, dummy_na=True).astype(np.float64)
    return X.fillna(X.median()).fillna(0)

def ridge_fit(Xtr,ytr,alpha):
    mu,sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[0,0]-=alpha
    return np.linalg.solve(A, Z.T@ytr), mu, sd
def ridge_pred(w,mu,sd,X):
    return np.c_[np.ones(len(X)), (X-mu)/sd]@w

y = df.future_spend_4w.values
sd_ = df.snapshot_day.values
tr = sd_<=403; te = sd_==431
cand_cols = ['dec_spend14','dec_spend7','dec_trips','gap_mean','gap_std','gap_med','ntrip112','pl_share','disc_share2','qty112','unit_price','macro_ratio']

def evalset(cols, alpha=10.0):
    X = prep(df[cols+['household_key','snapshot_day','future_spend_4w']])
    w,mu,s = ridge_fit(X[tr], y[tr], alpha)
    return np.abs(ridge_pred(w,mu,s,X[te]) - y[te]).mean()

base = evalset([c for c in df.columns if c not in cand_cols+['future_spend_4w']])
print('base(e7) 431 MAE:', round(base,3))
for c in cand_cols:
    m = evalset([c for c in df.columns if c not in cand_cols+['future_spend_4w']] + [c])
    print(f'+{c:14s} {m:8.3f}  delta {m-base:+.3f}')
allc = evalset([c for c in df.columns if c not in ['future_spend_4w']])
print('all e7+cand:', round(allc,3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet')
tt = agent_api.train_targets()
cand = cand.rename(columns={'qty112':'c_qty112'})
df = e7.merge(cand.drop(columns=['snapshot_day']), on='household_key', how='left').merge(tt, on=['household_key','snapshot_day'])

def prep(d):
    X = d.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    cats=[c for c in X.columns if X[c].dtype==object or str(X[c].dtype)=='category']
    X = pd.get_dummies(X, columns=cats, dummy_na=True).astype(np.float64)
    return X.fillna(X.median()).fillna(0)
def rfit(Xtr,ytr,alpha):
    mu,sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[0,0]-=alpha
    return np.linalg.solve(A, Z.T@ytr), mu, sd
def rpred(w,mu,sd,X): return np.c_[np.ones(len(X)), (X-mu)/sd]@w

y = df.future_spend_4w.values; sd_ = df.snapshot_day.values
tr = sd_<=403; te = sd_==431
cand_cols = ['dec_spend14','dec_spend7','dec_trips','gap_mean','gap_std','gap_med','ntrip112','pl_share','disc_share2','c_qty112','unit_price','macro_ratio']
def ev(cols, alpha=10.0):
    X = prep(df[cols+['household_key','snapshot_day','future_spend_4w']])
    w,mu,s = rfit(X[tr], y[tr], alpha)
    return np.abs(rpred(w,mu,s,X[te]) - y[te]).mean()
base_cols = [c for c in df.columns if c not in cand_cols+['future_spend_4w']]
base = ev(base_cols); print('base:', round(base,3))
for c in ['c_qty112','unit_price','macro_ratio']:
    print(f'+{c:12s} {ev(base_cols+[c]):8.3f}  delta {ev(base_cols+[c])-base:+.3f}')
combos = {
 'dec14+dec7+dectrips': ['dec_spend14','dec_spend7','dec_trips'],
 'dec14+ntrip112': ['dec_spend14','ntrip112'],
 'allgood': ['dec_spend14','dec_spend7','dec_trips','ntrip112'],
 'allgood+macro': ['dec_spend14','dec_spend7','dec_trips','ntrip112','macro_ratio'],
 'all12': cand_cols,
}
for k,v in combos.items():
    m = ev(base_cols+v)
    print(f'{k:22s} {m:8.3f}  delta {m-base:+.3f}')


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet').rename(columns={'qty112':'c_qty112'})
tt = agent_api.train_targets()
df = e7.merge(cand.drop(columns=['snapshot_day']), on='household_key', how='left').merge(tt, on=['household_key','snapshot_day'])
def prep(d):
    X = d.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    cats=[c for c in X.columns if X[c].dtype==object or str(X[c].dtype)=='category']
    return pd.get_dummies(X, columns=cats, dummy_na=True).astype(np.float64).fillna(X.median()).fillna(0) if False else pd.get_dummies(X, columns=cats, dummy_na=True).astype(np.float64).replace([np.inf,-np.inf],np.nan).fillna(0)
def rfit(Xtr,ytr,alpha):
    mu,sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[0,0]-=alpha
    return np.linalg.solve(A, Z.T@ytr), mu, sd
def rpred(w,mu,sd,X): return np.c_[np.ones(len(X)), (X-mu)/sd]@w
y = df.future_spend_4w.values; sd_ = df.snapshot_day.values
cand_cols = ['dec_spend14','dec_spend7','dec_trips','gap_mean','gap_std','gap_med','ntrip112','pl_share','disc_share2','c_qty112','unit_price','macro_ratio']
base_cols = [c for c in df.columns if c not in cand_cols+['future_spend_4w']]
def ev(cols, fitmax, evals, alpha=10.0):
    X = prep(df[cols+['household_key','snapshot_day','future_spend_4w']])
    w,mu,s = rfit(X[sd_<=fitmax], y[sd_<=fitmax], alpha)
    m = sd_.isin(evals)
    return np.abs(rpred(w,mu,s,X[m]) - y[m]).mean()
for fitmax, evals in [(375,[403,431]), (403,[431])]:
    b = ev(base_cols, fitmax, evals)
    a = ev(base_cols+['dec_spend14','dec_spend7','dec_trips','ntrip112'], fitmax, evals)
    c = ev(base_cols+cand_cols, fitmax, evals)
    print(f'fit<={fitmax} eval{evals}: base {b:.3f} | +dec4 {a:.3f} ({a-b:+.3f}) | +all12 {c:.3f} ({c-b:+.3f})')


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet').rename(columns={'qty112':'c_qty112'})
tt = agent_api.train_targets()
df = e7.merge(cand.drop(columns=['snapshot_day']), on='household_key', how='left').merge(tt, on=['household_key','snapshot_day'])
def prep(d):
    X = d.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    cats=[c for c in X.columns if X[c].dtype==object or str(X[c].dtype)=='category']
    X = pd.get_dummies(X, columns=cats, dummy_na=True).astype(np.float64).replace([np.inf,-np.inf],np.nan)
    return X.fillna(0)
def rfit(Xtr,ytr,alpha):
    mu,sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[0,0]-=alpha
    return np.linalg.solve(A, Z.T@ytr), mu, sd
def rpred(w,mu,sd,X): return np.c_[np.ones(len(X)), (X-mu)/sd]@w
y = df.future_spend_4w.values; sd_ = df.snapshot_day.values
cand_cols = ['dec_spend14','dec_spend7','dec_trips','gap_mean','gap_std','gap_med','ntrip112','pl_share','disc_share2','c_qty112','unit_price','macro_ratio']
base_cols = [c for c in df.columns if c not in cand_cols+['future_spend_4w']]
def ev(cols, fitmax, evals, alpha=10.0):
    X = prep(df[cols+['household_key','snapshot_day','future_spend_4w']])
    w,mu,s = rfit(X[sd_<=fitmax], y[sd_<=fitmax], alpha)
    m = pd.Series(sd_).isin(evals).values
    return np.abs(rpred(w,mu,s,X[m]) - y[m]).mean()
for fitmax, evals in [(375,[403,431]), (403,[431])]:
    b = ev(base_cols, fitmax, evals)
    a = ev(base_cols+['dec_spend14','dec_spend7','dec_trips','ntrip112'], fitmax, evals)
    c = ev(base_cols+cand_cols, fitmax, evals)
    print(f'fit<={fitmax} eval{evals}: base {b:.3f} | +dec4 {a:.3f} ({a-b:+.3f}) | +all12 {c:.3f} ({c-b:+.3f})')


# ---- cell ----
import agent_api, pandas as pd
e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet').rename(columns={'qty112':'c_qty112'})
merged = e7.merge(cand.drop(columns=['snapshot_day']), on='household_key', how='left')
print(merged.shape)
print([c for c in merged.columns if c in ['dec_spend14','dec_spend7','dec_trips','gap_mean','gap_std','gap_med','ntrip112','pl_share','disc_share2','c_qty112','unit_price','macro_ratio']])
agent_api.save_table(merged, 'e010_decay')


# ---- cell ----
import agent_api, pandas as pd
e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet').rename(columns={'qty112':'c_qty112'})
merged = e7.merge(cand, on=['household_key','snapshot_day'], how='left')
print(merged.shape)
agent_api.save_table(merged, 'e010_decay')
