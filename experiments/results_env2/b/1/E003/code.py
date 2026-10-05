
import agent_api, pandas as pd, numpy as np

t = agent_api.train_targets()
print(t.shape, t.columns.tolist())
print(t['future_spend_4w'].describe())
print("zero share:", (t['future_spend_4w']==0).mean())

e2 = agent_api.load_saved('e002_mix.parquet')
print(e2.shape)
m = t.merge(e2, on=['household_key','snapshot_day'], how='left')
num = [c for c in e2.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
corr = m[num].corrwith(m['future_spend_4w']).sort_values()
print("Top positive corr:")
print(corr.tail(25))
print("Top negative corr:")
print(corr.head(10))
print("n features:", len(num))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
e2 = agent_api.load_saved('e002_mix.parquet')
cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
print(cols)


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def cand(view, sd):
    tx = view.table('transactions')
    hh = view.households
    f = pd.DataFrame(index=hh)
    d = tx['day']
    # candidate: spend in specific recent windows
    for w in (7, 14, 21, 28, 42, 56, 84, 112, 168):
        f[f'c_sp{w}'] = tx[d > sd-w].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    # lagged same-window spend a year ago (364-392 days back)
    f['c_splag1y'] = tx[(d > sd-392) & (d <= sd-364)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['c_splag2y'] = tx[(d > sd-756) & (d <= sd-728)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    # weekly spend volatility last 12 weeks
    t2 = tx[d > sd-84].copy()
    t2['wk'] = t2['day']//7
    ws = t2.groupby(['household_key','wk'])['sales_value'].sum()
    g = ws.groupby('household_key')
    f['c_wkstd'] = g.std().reindex(hh, fill_value=0.0)
    f['c_wkmean'] = g.mean().reindex(hh, fill_value=0.0)
    f['c_wkcv'] = (f['c_wkstd'] / (f['c_wkmean']+1e-9))
    # trip gap stats last 168d
    t3 = tx[d > sd-168].drop_duplicates(['household_key','day'])
    ds = t3.groupby('household_key')['day'].agg(['min','max','count'])
    f['c_span'] = (ds['max']-ds['min']).reindex(hh)
    f['c_gapmean'] = (f['c_span']/(ds['count']-1).clip(lower=1)).reindex(hh)
    # share of spend at top store (loyalty) last 168d
    ss = t3.groupby(['household_key','store_id'])['sales_value'].sum()
    f['c_topstore'] = ss.groupby('household_key').max().reindex(hh, fill_value=0.0)/ (f['c_sp168']+1e-9)
    # distinct products last 168
    f['c_nprod168'] = t3.groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0)
    # avg basket value last 84
    bs = tx[d > sd-84].groupby(['household_key','basket_id'])['sales_value'].sum()
    f['c_bsmean'] = bs.groupby('household_key').mean().reindex(hh, fill_value=0.0)
    f['c_bsmax'] = bs.groupby('household_key').max().reindex(hh, fill_value=0.0)
    # coupon redemption count last 84/168
    cr = view.table('coupon_redemptions')
    for w in (84,168):
        f[f'c_nred{w}'] = cr[cr['day']>sd-w].groupby('household_key').size().reindex(hh, fill_value=0)
    # active days ratio last 28
    f['c_act28'] = tx[d > sd-28].groupby('household_key')['day'].nunique().reindex(hh, fill_value=0)/28.0
    return f

X = agent_api.build_features(cand)
t = agent_api.train_targets()
m = t.merge(X.reset_index(), on=['household_key','snapshot_day'], how='left')
num = [c for c in X.columns if pd.api.types.is_numeric_dtype(m[c])]
corr = m[num].corrwith(m['future_spend_4w']).sort_values()
print(corr)


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = t.merge(e2, on=['household_key','snapshot_day'], how='left').dropna(subset=['future_spend_4w'])
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values

def ridge_eval(df, target, seed=0):
    X = df.copy()
    cat = [c for c in X.columns if X[c].dtype == object]
    X = pd.get_dummies(X, columns=cat, dummy_na=True)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    mu, sd = X.mean(), X.std().replace(0,1)
    X = ((X-mu)/sd).values
    ly = np.log1p(target)
    lam = 30.0
    A = X[vd==False]; ya = ly[vd==False]
    Xt = X[vd]; 
    I = np.eye(A.shape[1])
    w = np.linalg.solve(A.T@A + lam*I, A.T@ya)
    p = np.expm1(Xt@w)
    return np.abs(p - target[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print("E002 proxy MAE:", ridge_eval(m[base_cols], y))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = t.merge(e2, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values

def ridge_eval(df, target, seed=0):
    X = df.copy()
    cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    X = pd.get_dummies(X, columns=cat, dummy_na=True)
    X = X.astype(float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    mu, sd = X.mean(), X.std().replace(0,1)
    X = ((X-mu)/sd).values
    ly = np.log1p(target)
    lam = 30.0
    A = X[~vd]; ya = np.log1p(target[~vd])
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ya)
    p = np.expm1(X[vd]@w)
    return np.abs(p - target[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print("E002 proxy MAE:", ridge_eval(m[base_cols], y))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = t.merge(e2, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values

def ridge_eval(df, target, lam=30.0, seed=0):
    X = df.copy()
    cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    X = pd.get_dummies(X, columns=cat, dummy_na=True)
    X = X.astype(float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    mu, sd = X.mean(), X.std().replace(0,1)
    X = ((X-mu)/sd).values
    ly = np.log1p(target)
    A = X[~vd]; ya = ly[~vd]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ya)
    p = np.expm1(X[vd]@w)
    return np.abs(p - target[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print("E002 proxy MAE:", ridge_eval(m[base_cols], y))
print("nan count in base:", m[base_cols].isna().sum().sum())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = t.merge(e2, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values

def ridge_eval(df, target, lam=30.0, seed=0):
    X = df.copy()
    cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    X = pd.get_dummies(X, columns=cat, dummy_na=True)
    X = X.astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    mu, sd = X.mean(), X.std().replace(0,1)
    X = ((X-mu)/sd).values
    ly = np.log1p(target)
    A = X[~vd]; ya = ly[~vd]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ya)
    p = np.expm1(X[vd]@w)
    return np.abs(p - target[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print("E002 proxy MAE:", ridge_eval(m[base_cols], y))

# now test candidate additions
def cand(view, sd):
    tx = view.table('transactions'); d = tx['day']; hh = view.households
    f = pd.DataFrame(index=hh)
    for w in (7,14,21,42,112):
        f[f'c_sp{w}'] = tx[d>sd-w].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['c_splag1y'] = tx[(d>sd-392)&(d<=sd-364)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    t2 = tx[d>sd-84].copy(); t2['wk']=t2['day']//7
    ws = t2.groupby(['household_key','wk'])['sales_value'].sum(); g = ws.groupby('household_key')
    f['c_wkstd'] = g.std().reindex(hh, fill_value=0.0)
    bs = tx[d>sd-84].groupby(['household_key','basket_id'])['sales_value'].sum()
    f['c_bsmean'] = bs.groupby('household_key').mean().reindex(hh, fill_value=0.0)
    f['c_bsmax'] = bs.groupby('household_key').max().reindex(hh, fill_value=0.0)
    f['c_nprod168'] = tx[d>sd-168].groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0)
    f['c_act28'] = tx[d>sd-28].groupby('household_key')['day'].nunique().reindex(hh, fill_value=0)/28.0
    return f

X = agent_api.build_features(cand)
mm = m.merge(X.reset_index(), on=['household_key','snapshot_day'], how='left')
cand_cols = [c for c in X.columns if c!='household_key']
print("E2 + candidates:", ridge_eval(mm[base_cols+cand_cols], y))
for c in cand_cols:
    print(c, ridge_eval(mm[base_cols+[c]], y))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = t.merge(e2, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values

X = m[[c for c in e2.columns if c not in ('household_key','snapshot_day')]].copy()
cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
X = pd.get_dummies(X, columns=cat, dummy_na=True).astype(float)
X = X.replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0)
mu, sd = X.mean(), X.std().replace(0,1)
Xz = ((X-mu)/sd).values
A = Xz[~vd]; ya = np.log1p(y[~vd])
w = np.linalg.solve(A.T@A + 30*np.eye(A.shape[1]), A.T@ya)
p = np.expm1(Xz[vd]@w)
print("pred stats:", np.nanmin(p), np.nanmax(p), np.isfinite(p).mean())
print("MAE no clip:", np.abs(p - y[vd]).mean())
pc = np.clip(p, 0, None)
print("MAE clip0:", np.abs(pc - y[vd]).mean())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
print(e2.dtypes.head(5))
print(e2['snapshot_day'].dtype, e2['snapshot_day'].unique()[:20])
m = t.merge(e2, on=['household_key','snapshot_day'], how='left')
print(m['snapshot_day'].dtype, m['snapshot_day'].unique())
print(agent_api.snapshot_days())
print("vd count:", m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).sum())
print("rows:", len(m), "targets:", len(t))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = e2.merge(t, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values
print("val rows:", vd.sum(), "train rows:", (~vd).sum())

def prep(df):
    X = df.copy()
    cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    X = pd.get_dummies(X, columns=cat, dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return ((X - X.mean()) / X.std().replace(0,1)).values

def ridge_eval(Xdf, lam=30.0):
    Xz = prep(Xdf)
    A = Xz[~vd]; ya = np.log1p(y[~vd])
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ya)
    p = np.clip(np.expm1(Xz[vd]@w), 0, None)
    return np.abs(p - y[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print("E002 proxy MAE:", ridge_eval(m[base_cols]))
for lam in (3, 10, 100, 300):
    print(lam, ridge_eval(m[base_cols], lam))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = e2.merge(t, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values

X = m[[c for c in e2.columns if c not in ('household_key','snapshot_day')]].copy()
cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
X = pd.get_dummies(X, columns=cat, dummy_na=True).astype(float)
X = X.replace([np.inf,-np.inf], np.nan)
print("nan cols:", X.isna().sum()[X.isna().sum()>0])
X = X.fillna(X.median(numeric_only=True)).fillna(0)
mu, sd = X.mean(), X.std().replace(0,1)
Xz = ((X-mu)/sd).values
A = Xz[~vd]; ya = np.log1p(y[~vd])
w = np.linalg.solve(A.T@A + 30*np.eye(A.shape[1]), A.T@ya)
p = np.clip(np.expm1(Xz[vd]@w), 0, None)
print("finite preds:", np.isfinite(p).mean(), p.min(), p.max())
print("MAE:", np.abs(p - y[vd]).mean())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = e2.merge(t, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w'].values
vd = (m['snapshot_day']==431).values
print("pseudo-val rows:", vd.sum())

def prep(df):
    X = df.copy()
    cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    X = pd.get_dummies(X, columns=cat, dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return ((X - X.mean()) / X.std().replace(0,1)).values

def ridge_eval(Xdf, lam=30.0):
    Xz = prep(Xdf)
    A = Xz[~vd]; ya = np.log1p(y[~vd])
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ya)
    p = np.clip(np.expm1(Xz[vd]@w), 0, None)
    return np.abs(p - y[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print("E002 proxy MAE:", ridge_eval(m[base_cols]))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = e2.merge(t, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w'].values
vd = (m['snapshot_day']==431).values

def prep(df):
    X = df.copy()
    cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    X = pd.get_dummies(X, columns=cat, dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return ((X - X.mean()) / X.std().replace(0,1)).values

def ridge_eval(Xdf, lam=30.0):
    Xz = prep(Xdf)
    A = Xz[~vd]; ya = np.log1p(y[~vd])
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ya)
    p = np.clip(np.expm1(Xz[vd]@w), 0, None)
    return np.abs(p - y[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]

def cand(view, sd):
    tx = view.table('transactions'); d = tx['day']; hh = view.households
    f = pd.DataFrame(index=hh)
    for w in (7,14,21,42,112):
        f[f'c_sp{w}'] = tx[d>sd-w].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['c_splag1y'] = tx[(d>sd-392)&(d<=sd-364)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    t2 = tx[d>sd-84].copy(); t2['wk']=t2['day']//7
    ws = t2.groupby(['household_key','wk'])['sales_value'].sum(); g = ws.groupby('household_key')
    f['c_wkstd'] = g.std().reindex(hh, fill_value=0.0)
    f['c_wkmean'] = g.mean().reindex(hh, fill_value=0.0)
    bs = tx[d>sd-84].groupby(['household_key','basket_id'])['sales_value'].sum()
    f['c_bsmean'] = bs.groupby('household_key').mean().reindex(hh, fill_value=0.0)
    f['c_bsmax'] = bs.groupby('household_key').max().reindex(hh, fill_value=0.0)
    f['c_nprod168'] = tx[d>sd-168].groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0)
    f['c_act28'] = tx[d>sd-28].groupby('household_key')['day'].nunique().reindex(hh, fill_value=0)/28.0
    cr = view.table('coupon_redemptions')
    f['c_nred84'] = cr[cr['day']>sd-84].groupby('household_key').size().reindex(hh, fill_value=0)
    return f

X = agent_api.build_features(cand)
mm = m.merge(X.reset_index(), on=['household_key','snapshot_day'], how='left')
cand_cols = [c for c in X.columns if c!='household_key']
print("E2+cands:", ridge_eval(mm[base_cols+cand_cols]))
for c in cand_cols:
    print(c, round(ridge_eval(mm[base_cols+[c]]),2))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def fn(view, sd):
    hh = view.households
    tx = view.table('transactions'); d = tx['day']
    f = pd.DataFrame(index=hh); f.index.name = 'household_key'
    def sp(w): return tx[d>sd-w].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    def tr(w): return tx[d>sd-w].groupby('household_key')['basket_id'].nunique().reindex(hh, fill_value=0.0)
    def pr(w): return tx[d>sd-w].groupby('household_key')['product_id'].nunique().reindex(hh, fill_value=0.0)
    def na(w): return tx[d>sd-w].groupby('household_key')['day'].nunique().reindex(hh, fill_value=0)
    s28,s56,s84,s168,s364 = sp(28),sp(56),sp(84),sp(168),sp(364)
    t28,t84 = tr(28),tr(84); p28,p84 = pr(28),pr(84)
    # momentum / ratios / interactions
    f['m_sp28_56'] = s28 - 0.5*s56
    f['m_sp56_84'] = s56 - 0.5*s84
    f['r_sp28_84'] = s28/(s84+1.0)
    f['r_sp84_364'] = s84/(s364+1.0)
    f['r_tr28_84'] = t28/(t84+1.0)
    f['r_pr28_84'] = p28/(p84+1.0)
    f['spt_84'] = s84/(t84+1.0)
    f['spd_84'] = s84/(na(84)+1.0)
    f['i_sp28_rise'] = s28*s28/(s56+1.0)
    f['i_sp84_rise'] = s84*s84/(s364+1.0)
    # store loyalty share (168d)
    t3 = tx[d>sd-168]
    ss = t3.groupby(['household_key','store_id'])['sales_value'].sum()
    f['topstore'] = ss.groupby('household_key').max().reindex(hh, fill_value=0.0)/(s168+1e-9)
    # weekly volatility (12w)
    t2 = tx[d>sd-84].copy(); t2['wk'] = t2['day']//7
    ws = t2.groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key')
    f['wkstd'] = ws.std().reindex(hh, fill_value=0.0)
    f['wkmax'] = ws.max().reindex(hh, fill_value=0.0)
    # seasonal lag + recent activity
    f['splag1y'] = tx[(d>sd-392)&(d<=sd-364)].groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
    f['nact14'] = na(14)
    try:
        e2 = agent_api.load_saved('e002_mix.parquet')
        sub = e2[e2['snapshot_day']==sd].drop(columns=['snapshot_day']).set_index('household_key').reindex(hh)
        out = sub.join(f)
        return out
    except Exception as e:
        # fallback core
        g = tx.groupby('household_key')
        f['tenure'] = (sd - g['day'].min().reindex(hh)).clip(lower=0)
        f['days_since_last'] = sd - g['day'].max().reindex(hh)
        for w in (28,56,84,168,364,728):
            f[f'sp{w}'] = sp(w); f[f'trips{w}'] = tr(w); f[f'prods{w}'] = pr(w); f[f'nact{w}'] = na(w)
            f[f'stores{w}'] = tx[d>sd-w].groupby('household_key')['store_id'].nunique().reindex(hh, fill_value=0)
            f[f'qty{w}'] = tx[d>sd-w].groupby('household_key')['quantity'].sum().reindex(hh, fill_value=0.0)
            bs = tx[d>sd-w].groupby(['household_key','basket_id'])['sales_value'].sum()
            f[f'avgbs{w}'] = bs.groupby('household_key').mean().reindex(hh, fill_value=0.0)
            f[f'maxbs{w}'] = bs.groupby('household_key').max().reindex(hh, fill_value=0.0)
        for w in (84,364):
            f[f'cdisc{w}'] = tx[d>sd-w].groupby('household_key')['coupon_disc'].sum().reindex(hh, fill_value=0.0)
            f[f'rdisc{w}'] = tx[d>sd-w].groupby('household_key')['retail_disc'].sum().reindex(hh, fill_value=0.0)
        return f

X = agent_api.build_features(fn)
print(X.shape)
print([c for c in X.columns if c.startswith(('m_','r_','i_','top','wk','splag','nact14','spt','spd'))])
print(X.head(3).iloc[:, :8])
p = agent_api.save_table(X, 'e003_momentum.parquet')
print(p)


# ---- cell ----

import agent_api, pandas as pd
e2 = agent_api.load_saved('e002_mix.parquet')
e3 = agent_api.load_saved('e003_momentum.parquet')
new_cols = [c for c in e3.columns if c.startswith(('m_','r_','i_','topstore','wkstd','wkmax','splag1y','nact14','spt_','spd_'))]
print("new:", new_cols)
mm = e2.merge(e3[['household_key','snapshot_day']+new_cols], on=['household_key','snapshot_day'], how='inner')
print(mm.shape)
print("rows per snapshot:", mm.groupby('snapshot_day').size().to_dict())
p = agent_api.save_table(mm, 'e003_full.parquet')
print(p)
