import numpy as np, pandas as pd, agent_api
t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
print('shapes', t8.shape, t16.shape, tt.shape)
c8 = [c for c in t8.columns if c not in ('household_key','snapshot_day')]
c16 = [c for c in t16.columns if c not in ('household_key','snapshot_day')]
print('E008(%d):' % len(c8)); print(c8)
print('E016(%d):' % len(c16)); print(c16)
y = tt.future_spend_4w
print('tgt mean %.2f med %.2f zero%% %.3f p90 %.1f p99 %.1f max %.1f' % (y.mean(), y.median(), (y==0).mean(), y.quantile(.9), y.quantile(.99), y.max()))
print(t8.dtypes.value_counts().to_dict())
print('rows/snap:'); print(tt.groupby('snapshot_day').size().to_dict())


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def build_matrix(df):
    df = df.copy()
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool')]
    num_cols = [c for c in df.columns if c not in cat_cols + ['household_key','snapshot_day']]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df)))
        names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names

def ridge_eval(X, snap, y, lam=200.0, verbose=True):
    istr = np.isin(snap, [95,123,151,179,207,235,263,291,319,347,375,403,431])
    isv  = np.isin(snap, [459,487,515,543])
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd
    Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    pv = np.clip(Z[isv]@w, 0, None)
    mae = np.abs(pv-y[isv]).mean()
    if verbose: print('val MAE %.4f  nfeat %d' % (mae, X.shape[1]))
    return mae

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
m = tt.merge(t8, on=['household_key','snapshot_day']).merge(
    t16[[c for c in t16.columns if c.startswith('g_')]+['household_key','snapshot_day']], on=['household_key','snapshot_day'])
m = m.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = m.snapshot_day.values; y = m.future_spend_4w.values.astype(float)
X, names = build_matrix(m)
print('E008+grid screener:'); mae_all = ridge_eval(X, snap, y)
# per-snapshot MAE
istr = np.isin(snap,[95,123,151,179,207,235,263,291,319,347,375,403,431]); isv=np.isin(snap,[459,487,515,543])
mu,sd = X[istr].mean(0), X[istr].std(0)+1e-9; Z=(X-mu)/sd; Z=np.column_stack([np.ones(len(Z)),Z])
A=Z[istr].T@Z[istr]+200*np.eye(Z.shape[1]); A[0,0]-=200; w=np.linalg.solve(A,Z[istr].T@y[istr])
pv=np.clip(Z[isv]@w,0,None)
sv=snap[isv]
for s in [459,487,515,543]:
    k=sv==s; print('  snap %d MAE %.2f (n=%d)' % (s, np.abs(pv[k]-y[isv][k]).mean(), k.sum()))


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def build_matrix(df):
    df = df.copy()
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool')]
    num_cols = [c for c in df.columns if c not in cat_cols + ['household_key','snapshot_day']]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df)))
        names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names

def ridge_fit_eval(X, snap, y, fit_snaps, eval_snaps, lam=200.0):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    pv = np.clip(Z[isv]@w, 0, None)
    return np.abs(pv-y[isv]).mean()

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
gcols = ['household_key','snapshot_day']+[c for c in t16.columns if c.startswith('g_')]
m = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(t16[gcols], on=['household_key','snapshot_day'])
m = m.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = m.snapshot_day.values; y = m.future_spend_4w.values.astype(float)
FIT = [95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
print('rows', len(m))

X8, _ = build_matrix(m.drop(columns=[c for c in m.columns if c.startswith('g_')]))
print('E008 only      : %.4f' % ridge_fit_eval(X8, snap, y, FIT, EVAL))
Xall, _ = build_matrix(m)
print('E008 + g_ grid : %.4f' % ridge_fit_eval(Xall, snap, y, FIT, EVAL))
# g_ alone
Xg, _ = build_matrix(m[['household_key','snapshot_day']+[c for c in m.columns if c.startswith('g_')]])
print('g_ grid only   : %.4f' % ridge_fit_eval(Xg, snap, y, FIT, EVAL))
# lam sensitivity on E008
for lam in [50,100,200,400,800]:
    print('  E008 lam %4d: %.4f' % (lam, ridge_fit_eval(X8, snap, y, FIT, EVAL, lam)))


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
def build_matrix(df):
    df = df.copy()
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool')]
    num_cols = [c for c in df.columns if c not in cat_cols + ['household_key','snapshot_day']]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df))); names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names

t8 = agent_api.load_saved('e008_level_shape.parquet')
tt = agent_api.train_targets()
m = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = m.snapshot_day.values; y = m.future_spend_4w.values.astype(float)
print('y stats by snap:')
print(m.groupby('snapshot_day').future_spend_4w.agg(['mean','median','std','count']).round(1))
# check duplicates in merge
print('dupes:', m.duplicated(['household_key','snapshot_day']).sum())
X8,_ = build_matrix(m)
istr = np.isin(snap,[95,123,151,179,207,235,263,291,319,347,375]); isv=np.isin(snap,[403,431])
mu,sd = X8[istr].mean(0), X8[istr].std(0)+1e-9; Z=(X8-mu)/sd; Z=np.column_stack([np.ones(len(Z)),Z])
A=Z[istr].T@Z[istr]+50*np.eye(Z.shape[1]); A[0,0]-=50; w=np.linalg.solve(A,Z[istr].T@y[istr])
pv=np.clip(Z[isv]@w,0,None)
print('eval y: mean %.2f med %.2f' % (y[isv].mean(), np.median(y[isv])))
print('pred  : mean %.2f med %.2f min %.2f max %.2f' % (pv.mean(), np.median(pv), pv.min(), pv.max()))
print('MAE %.4f  corr %.4f' % (np.abs(pv-y[isv]).mean(), np.corrcoef(pv, y[isv])[0,1]))
# per snapshot
for s in [403,431]:
    k = snap[isv]==s
    print('snap', s, 'MAE %.3f' % np.abs(pv[k]-y[isv][k]).mean())


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
t8 = agent_api.load_saved('e008_level_shape.parquet')
tt = agent_api.train_targets()
m = t8.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = m.future_spend_4w.values.astype(float)
feats = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
best = []
for c in feats:
    v = pd.to_numeric(m[c], errors='coerce')
    if v.notna().sum() < len(m)*0.5: continue
    v = v.fillna(v.median()).values.astype(float)
    if v.std() < 1e-12: continue
    r = abs(np.corrcoef(v, y)[0,1])
    best.append((r, c))
best.sort(reverse=True)
for r,c in best[:15]: print('%.6f  %s' % (r,c))


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
def build_matrix(df):
    df = df.copy()
    drop = ['household_key','snapshot_day','future_spend_4w']
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool') and c not in drop]
    num_cols = [c for c in df.columns if c not in cat_cols + drop]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df))); names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names

def ridge(X, snap, y, fit_snaps, eval_snaps, lam=200.0, return_pred=False):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    pv = np.clip(Z[isv]@w, 0, None)
    return (np.abs(pv-y[isv]).mean(), pv) if return_pred else np.abs(pv-y[isv]).mean()

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
gcols = [c for c in t16.columns if c.startswith('g_')]
m = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(
    t16[['household_key','snapshot_day']+gcols], on=['household_key','snapshot_day'])
m = m.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = m.snapshot_day.values; y = m.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]

X8,_  = build_matrix(m.drop(columns=gcols))
Xa,_  = build_matrix(m)
Xg,_  = build_matrix(m[['household_key','snapshot_day']+gcols])
print('E008        : %.4f' % ridge(X8, snap, y, FIT, EVAL))
print('E008+g_grid : %.4f' % ridge(Xa, snap, y, FIT, EVAL))
print('g_grid only : %.4f' % ridge(Xg, snap, y, FIT, EVAL))
for lam in [50,100,200,400]:
    print('  E008+g lam %3d: %.4f' % (lam, ridge(Xa, snap, y, FIT, EVAL, lam)))
# per-snap for E008+g
mae, pv = ridge(Xa, snap, y, FIT, EVAL, 100, True)
isv = np.isin(snap, EVAL)
for s in EVAL:
    k = snap[isv]==s
    print('  snap %d MAE %.2f  predmean %.1f ymean %.1f' % (s, np.abs(pv[k]-y[isv][k]).mean(), pv[k].mean(), y[isv][k].mean()))


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def cand_fn(view, snapshot_day):
    hh = view.households
    tx = view.table('transactions')
    d0 = view.day
    tx = tx[tx.household_key.isin(set(hh))]
    tx = tx.assign(bs=tx.groupby('basket_id').sales_value.transform('sum'))
    out = pd.DataFrame(index=hh)

    def win(days):
        return tx[(tx.day > d0-days) & (tx.day <= d0)]

    # A: tail / premium
    w84 = win(84); w28 = win(28)
    g = w84.groupby('household_key')
    bs84 = w84.drop_duplicates('basket_id')[['household_key','bs']]
    gb = bs84.groupby('household_key').bs
    out['n_big84'] = gb.apply(lambda s: (s>100).sum())
    out['bigshare84'] = gb.apply(lambda s: s[s>100].sum()).fillna(0)
    out['bigshare84'] = out['bigshare84'] / (g.sales_value.sum()+1e-9)
    out['top3b84'] = gb.max() + gb.nlargest if False else gb.apply(lambda s: s.nlargest(3).sum() if len(s)>=1 else 0)
    out['top3b84'] = out['top3b84'] / (g.sales_value.sum()+1e-9)
    line84 = w84.assign(price=w84.sales_value/w84.quantity.clip(lower=0.1))
    out['maxline84'] = line84.groupby('household_key').sales_value.max()
    out['hi_item_share84'] = line84[line84.sales_value>=15].groupby('household_key').sales_value.sum().fillna(0)/(g.sales_value.sum()+1e-9)
    # B: absolute department spend 84d
    dep = w84.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
    for dd in ['GROCERY','MEAT','PRODU','DELI','SPIRI','COSME','FLORA','SEAFO']:
        if dd in dep.columns: out['dabs84_'+dd] = dep[dd]
        else: out['dabs84_'+dd] = 0.0
    # D: dormancy
    gap = tx[tx.day > d0-364].sort_values(['household_key','day']).groupby('household_key').day
    gapd = gap.diff().groupby(tx.loc[gap.gap.index if False else gap.apply(lambda x: x.index[0]) if False else slice(None),'household_key']) if False else None
    days = tx[tx.day > d0-364].sort_values(['household_key','day'])
    gd = days.groupby('household_key').day.diff()
    gap_med = gd.groupby(days.household_key).median()
    dsl = (d0 - g.day.max()).clip(lower=0)
    out['dorm_r'] = dsl/(gap_med+1.0)
    out['act_exp'] = np.exp(-dsl/(gap_med+7.0))
    # zeros in last 3 disjoint 28d windows
    z = []
    for k in range(3):
        wk = win(28*(k+1)).groupby('household_key').sales_value.sum()
        z.append((wk.reindex(hh).fillna(0)==0).astype(float))
    out['zeros_last3'] = sum(z)
    # E: discount intensity
    out['cd84_r'] = w84.groupby('household_key').coupon_disc.sum().abs()/(g.sales_value.sum()+1e-9)
    out['rd84_r'] = w84.groupby('household_key').retail_disc.sum().abs()/(g.sales_value.sum()+1e-9)
    # F: time of day
    tt = w84.assign(eve=w84.trans_time>=1700, morn=w84.trans_time<1000)
    out['eve_share84'] = tt[tt.eve].groupby('household_key').sales_value.sum().fillna(0)/(g.sales_value.sum()+1e-9)
    out['morn_share84'] = tt[tt.morn].groupby('household_key').sales_value.sum().fillna(0)/(g.sales_value.sum()+1e-9)
    # H: unit price
    q = w84.groupby('household_key').quantity.sum().clip(lower=0.1)
    out['up84'] = g.sales_value.sum()/q
    # C: cross-sectional ranks
    sp84 = g.sales_value.sum().reindex(hh).fillna(0)
    out['r_sp84'] = sp84.rank(pct=True)
    out['r_dsl'] = dsl.reindex(hh).fillna(999).rank(pct=True)
    tr84 = w84.groupby('household_key').basket_id.nunique().reindex(hh).fillna(0)
    out['r_tr84'] = tr84.rank(pct=True)
    return out

X = agent_api.build_features(cand_fn)
print(X.shape)
tt = agent_api.train_targets()
m = tt.merge(X.reset_index(), on=['household_key','snapshot_day'])
agent_api.save_table(X.reset_index(), 'cand_new1.parquet')
print('saved', m.shape)


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def cand_fn(view, snapshot_day):
    hh = view.households
    tx = view.table('transactions')
    d0 = view.day
    tx = tx[tx.household_key.isin(set(hh))]
    tx = tx.assign(bs=tx.groupby('basket_id').sales_value.transform('sum'))
    out = pd.DataFrame(index=hh)
    def win(days): return tx[(tx.day > d0-days) & (tx.day <= d0)]
    w84 = win(84)
    g = w84.groupby('household_key')
    tot = g.sales_value.sum()
    # A: tail / premium
    bs84 = w84.drop_duplicates('basket_id')[['household_key','bs']]
    gb = bs84.groupby('household_key').bs
    out['n_big84'] = gb.apply(lambda s: (s>100).sum())
    out['bigshare84'] = gb.apply(lambda s: s[s>100].sum()).fillna(0)/(tot+1e-9)
    out['top3b84'] = gb.apply(lambda s: s.nlargest(3).sum() if len(s)>=1 else 0)/(tot+1e-9)
    out['maxline84'] = w84.groupby('household_key').sales_value.max()
    hi = w84[w84.sales_value>=15]
    out['hi_item_share84'] = hi.groupby('household_key').sales_value.sum().fillna(0)/(tot+1e-9)
    # B: absolute department spend 84d (merge products)
    pr = view.table('products')[['product_id','department']]
    wdep = w84.merge(pr, on='product_id', how='left')
    dep = wdep.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
    for dd in ['GROCERY','MEAT','PRODU','DELI','SPIRI','COSME','FLORA','SEAFO']:
        out['dabs84_'+dd] = dep[dd] if dd in dep.columns else 0.0
    # D: dormancy
    days = tx[tx.day > d0-364].sort_values(['household_key','day'])
    gd = days.groupby('household_key').day.diff()
    gap_med = gd.groupby(days.household_key).median()
    dsl = (d0 - g.day.max()).clip(lower=0)
    out['dorm_r'] = dsl/(gap_med+1.0)
    out['act_exp'] = np.exp(-dsl/(gap_med+7.0))
    z = 0.0
    for k in range(3):
        wk = win(28*(k+1)).groupby('household_key').sales_value.sum()
        z = z + (wk.reindex(hh).fillna(0)==0).astype(float)
    out['zeros_last3'] = z
    # E: discount intensity
    out['cd84_r'] = w84.groupby('household_key').coupon_disc.sum().abs()/(tot+1e-9)
    out['rd84_r'] = w84.groupby('household_key').retail_disc.sum().abs()/(tot+1e-9)
    # F: time of day
    out['eve_share84'] = w84[w84.trans_time>=1700].groupby('household_key').sales_value.sum().fillna(0)/(tot+1e-9)
    out['morn_share84'] = w84[w84.trans_time<1000].groupby('household_key').sales_value.sum().fillna(0)/(tot+1e-9)
    # H: unit price
    q = w84.groupby('household_key').quantity.sum().clip(lower=0.1)
    out['up84'] = tot/q
    # C: cross-sectional ranks
    out['r_sp84'] = tot.reindex(hh).fillna(0).rank(pct=True)
    out['r_dsl'] = dsl.reindex(hh).fillna(999).rank(pct=True)
    tr84 = w84.groupby('household_key').basket_id.nunique().reindex(hh).fillna(0)
    out['r_tr84'] = tr84.rank(pct=True)
    return out

X = agent_api.build_features(cand_fn)
print(X.shape)
tt = agent_api.train_targets()
m = tt.merge(X.reset_index(), on=['household_key','snapshot_day'])
agent_api.save_table(X.reset_index(), 'cand_new1.parquet')
print('saved', m.shape)


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
def build_matrix(df, drop=('household_key','snapshot_day','future_spend_4w')):
    drop = list(drop)
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool') and c not in drop]
    num_cols = [c for c in df.columns if c not in cat_cols + drop]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df))); names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names
def ridge(X, snap, y, fit_snaps, eval_snaps, lam=200.0):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    return np.abs(np.clip(Z[isv]@w,0,None)-y[isv]).mean()

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
cn = agent_api.load_saved('cand_new1.parquet')
tt = agent_api.train_targets()
gcols = [c for c in t16.columns if c.startswith('g_')]
base = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(
    t16[['household_key','snapshot_day']+gcols], on=['household_key','snapshot_day'])
base = base.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = base.snapshot_day.values; y = base.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
Xb,_ = build_matrix(base)
print('base E008+g : %.4f' % ridge(Xb, snap, y, FIT, EVAL))

new = cn.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
ncols = [c for c in new.columns if c not in ('household_key','snapshot_day')]
# align
assert (new.household_key.values == base.household_key.values).all() and (new.snapshot_day.values == base.snapshot_day.values).all()
m2 = pd.concat([base, new[ncols]], axis=1)
X2,_ = build_matrix(m2)
print('base+cand_new1: %.4f' % ridge(X2, snap, y, FIT, EVAL))
# family-level screening: drop one family at a time
fams = {'A_tail':[c for c in ncols if c.startswith(('n_big','bigshare','top3b','maxline','hi_item'))],
        'B_dabs':[c for c in ncols if c.startswith('dabs84')],
        'D_dorm':['dorm_r','act_exp','zeros_last3'],
        'E_disc':['cd84_r','rd84_r'],
        'F_tod':['eve_share84','morn_share84'],
        'H_up':['up84'],
        'C_rank':['r_sp84','r_dsl','r_tr84']}
for k,v in fams.items():
    Xk,_ = build_matrix(m2.drop(columns=v))
    print('  drop %-7s: %.4f' % (k, ridge(Xk, snap, y, FIT, EVAL)))
# individual feature correlations with residual
mu,sd = Xb.mean(0), Xb.std(0)+1e-9
Zb=(Xb-mu)/sd; Zb=np.column_stack([np.ones(len(Zb)),Zb])
A=Zb[np.isin(snap,FIT)].T@Zb[np.isin(snap,FIT)]+200*np.eye(Zb.shape[1]); A[0,0]-=200
w=np.linalg.solve(A, Zb[np.isin(snap,FIT)].T@y[np.isin(snap,FIT)])
res = y - Zb@w
Xn,_ = build_matrix(new)
print('\n|corr| with residual:')
for i,c in enumerate(ncols):
    v = Xn[:,i]
    if v.std()<1e-12: print('  %-16s const' % c); continue
    print('  %-16s %.4f' % (c, abs(np.corrcoef(v, res)[0,1])))


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
def build_matrix(df, drop=('household_key','snapshot_day','future_spend_4w')):
    drop = list(drop)
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool') and c not in drop]
    num_cols = [c for c in df.columns if c not in cat_cols + drop]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df))); names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names
def ridge(X, snap, y, fit_snaps, eval_snaps, lam=200.0):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    return np.abs(np.clip(Z[isv]@w,0,None)-y[isv]).mean()

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
cn = agent_api.load_saved('cand_new1.parquet')
tt = agent_api.train_targets()
gcols = [c for c in t16.columns if c.startswith('g_')]
base = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(
    t16[['household_key','snapshot_day']+gcols], on=['household_key','snapshot_day'])
base = base.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = base.snapshot_day.values; y = base.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
Xb,_ = build_matrix(base)
print('base E008+g : %.4f' % ridge(Xb, snap, y, FIT, EVAL))
ncols = [c for c in cn.columns if c not in ('household_key','snapshot_day')]
m2 = base.merge(cn, on=['household_key','snapshot_day'])
X2,_ = build_matrix(m2)
print('base+cand   : %.4f' % ridge(X2, snap, y, FIT, EVAL))
fams = {'A_tail':[c for c in ncols if c.startswith(('n_big','bigshare','top3b','maxline','hi_item'))],
        'B_dabs':[c for c in ncols if c.startswith('dabs84')],
        'D_dorm':['dorm_r','act_exp','zeros_last3'],
        'E_disc':['cd84_r','rd84_r'],
        'F_tod':['eve_share84','morn_share84'],
        'H_up':['up84'],
        'C_rank':['r_sp84','r_dsl','r_tr84']}
for k,v in fams.items():
    Xk,_ = build_matrix(m2.drop(columns=v))
    print('  drop %-7s: %.4f' % (k, ridge(Xk, snap, y, FIT, EVAL)))
# residual correlations
istr = np.isin(snap, FIT)
mu,sd = Xb[istr].mean(0), Xb[istr].std(0)+1e-9
Zb=(Xb-mu)/sd; Zb=np.column_stack([np.ones(len(Zb)),Zb])
A=Zb[istr].T@Zb[istr]+200*np.eye(Zb.shape[1]); A[0,0]-=200
w=np.linalg.solve(A, Zb[istr].T@y[istr])
res = y - Zb@w
Xn,_ = build_matrix(m2[ncols])
print('\n|corr| with residual (full rows):')
for i,c in enumerate(ncols):
    v = Xn[:,i]
    if v.std()<1e-12: print('  %-16s const' % c); continue
    print('  %-16s %.4f' % (c, abs(np.corrcoef(v, res)[0,1])))


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
def build_matrix(df, drop=('household_key','snapshot_day','future_spend_4w')):
    drop = list(drop)
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool') and c not in drop]
    num_cols = [c for c in df.columns if c not in cat_cols + drop]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df))); names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names
def ridge(X, snap, y, fit_snaps, eval_snaps, lam=200.0):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    return np.abs(np.clip(Z[isv]@w,0,None)-y[isv]).mean()

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
gcols = [c for c in t16.columns if c.startswith('g_')]
base = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(
    t16[['household_key','snapshot_day']+gcols], on=['household_key','snapshot_day'])
base = base.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = base.snapshot_day.values; y = base.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
ALLSNAPS = [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]

# expanding per-household past-target mean (strictly earlier snapshots)
tt2 = tt.sort_values(['household_key','snapshot_day'])
g = tt2.groupby('household_key')
past_sum = g.future_spend_4w.cumsum() - tt2.future_spend_4w
past_cnt = g.cumcount()
te_mean = (past_sum/past_cnt.replace(0,np.nan))
# EWMA over past targets per household (halflife 2 snapshots)
def ewma(sub):
    v = sub.future_spend_4w.values.astype(float)
    out = np.empty(len(v)); acc=np.nan; wsum=0.0
    for i in range(len(v)):
        if i>0:
            wsum = wsum*0.5 + 1.0; acc = (np.nan_to_num(acc,nan=0.0)*0.5 + v[i-1])
            # proper: weights 0.5^k
        out[i] = np.nan if i==0 else np.average(v[:i], weights=0.5**np.arange(i-1,-1,-1))
    return pd.Series(out, index=sub.index)
te_ew = tt2.groupby('household_key', group_keys=False).apply(ewma)
te = pd.DataFrame({'household_key':tt2.household_key,'snapshot_day':tt2.snapshot_day,
                   'te_mean':te_mean.values,'te_cnt':past_cnt.values,'te_ew':te_ew.values})
gm = tt.future_spend_4w.mean()
te['te_mean'] = te.te_mean.fillna(gm); te['te_ew'] = te.te_ew.fillna(gm)

m = base.merge(te, on=['household_key','snapshot_day'], how='left')
m['snap_day'] = m.snapshot_day/100.0
Xb,_ = build_matrix(base)
print('base            : %.4f' % ridge(Xb, snap, y, FIT, EVAL))
X1,_ = build_matrix(m.drop(columns=['te_mean','te_cnt','te_ew']))
print('+snap_day trend : %.4f' % ridge(X1, snap, y, FIT, EVAL))
X2,_ = build_matrix(m.drop(columns=['snap_day']))
print('+te (mean/cnt/ew): %.4f' % ridge(X2, snap, y, FIT, EVAL))
X3,_ = build_matrix(m)
print('+both           : %.4f' % ridge(X3, snap, y, FIT, EVAL))
for lam in [100,400]:
    print('  +both lam %d: %.4f' % (lam, ridge(X3, snap, y, FIT, EVAL, lam)))
# corr of te_mean with y on eval snaps
isv = np.isin(snap, EVAL)
print('te_mean corr w/ y (eval rows): %.4f' % np.corrcoef(m.te_mean[isv], y[isv])[0,1])
print('sp84 corr w/ y (eval rows)   : %.4f' % np.corrcoef(pd.to_numeric(base.sp84,errors='coerce').fillna(0)[isv], y[isv])[0,1])


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
def build_matrix(df, drop=('household_key','snapshot_day','future_spend_4w')):
    drop = list(drop)
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool') and c not in drop]
    num_cols = [c for c in df.columns if c not in cat_cols + drop]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df))); names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names
def ridge(X, snap, y, fit_snaps, eval_snaps, lam=200.0):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    return np.abs(np.clip(Z[isv]@w,0,None)-y[isv]).mean()

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
gcols = [c for c in t16.columns if c.startswith('g_')]
base = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(
    t16[['household_key','snapshot_day']+gcols], on=['household_key','snapshot_day'])
base = base.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = base.snapshot_day.values; y = base.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
Xb,_ = build_matrix(base)
print('base: %.4f' % ridge(Xb, snap, y, FIT, EVAL))

def numcol(df, c):
    v = pd.to_numeric(df[c], errors='coerce')
    return v.fillna(v.median()).values.astype(float)

extras = []
# 1) binned dummies of log1p(sp84): 12 fixed-width bins on log scale 0..8
lsp = np.log1p(np.clip(numcol(base,'sp84'),0,None))
bins = np.clip(((lsp/0.7).astype(int)), 0, 11)
for b in range(12):
    extras.append(('bin84_%d'%b, (bins==b).astype(float)))
# same for sp28
lsp28 = np.log1p(np.clip(numcol(base,'sp28'),0,None))
b28 = np.clip((lsp28/0.7).astype(int),0,11)
for b in range(12):
    extras.append(('bin28_%d'%b, (b28==b).astype(float)))
# 2) interactions with snap_day
sdn = snap/400.0
for c in ['sp84','sp28','sp364','z_med4w_hist','g_wmean','newma56','days_since_last']:
    v = numcol(base,c); v = (v-v.mean())/(v.std()+1e-9)
    extras.append(('ia_%s'%c, v*sdn))
# 3) pairwise interactions of top-3
v1 = numcol(base,'sp84'); v1=(v1-v1.mean())/(v1.std()+1e-9)
v2 = numcol(base,'days_since_last'); v2=(v2-v2.mean())/(v2.std()+1e-9)
v3 = numcol(base,'g_wmean'); v3=(v3-v3.mean())/(v3.std()+1e-9)
extras.append(('ia_l_r', v1*v2)); extras.append(('ia_l_w', v1*v3)); extras.append(('ia_r_w', v2*v3))
E = np.column_stack([e[1] for e in extras]); enames=[e[0] for e in extras]
X1 = np.column_stack([Xb, E])
print('base+bins+interactions: %.4f' % ridge(X1, snap, y, FIT, EVAL))
# bins only
X2 = np.column_stack([Xb, E[:, :24]])
print('base+bins only        : %.4f' % ridge(X2, snap, y, FIT, EVAL))
# interactions only
X3 = np.column_stack([Xb, E[:, 24:]])
print('base+interactions only: %.4f' % ridge(X3, snap, y, FIT, EVAL))
for lam in [400, 800]:
    print('  all lam %d: %.4f' % (lam, ridge(X1, snap, y, FIT, EVAL, lam)))


# ---- cell ----
import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
t8  = agent_api.load_saved('e008_level_shape.parquet')
t6  = agent_api.load_saved('e006_cadence.parquet')
t7  = agent_api.load_saved('e007_temporal.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt  = agent_api.train_targets()
key = ['household_key','snapshot_day']
def uniq(t, base_cols):
    return [c for c in t.columns if c not in base_cols]
cols8  = uniq(t8,  key)
cols6  = [c for c in uniq(t6, key)  if c not in cols8]
cols7  = [c for c in uniq(t7, key)  if c not in cols8+cols6]
cols16 = [c for c in uniq(t16, key) if c not in cols8+cols6+cols7]
print('counts:', len(cols8), len(cols6), len(cols7), len(cols16))
m = t8[key+cols8].merge(t6[key+cols6], on=key).merge(t7[key+cols7], on=key).merge(t16[key+cols16], on=key)
print('grand table:', m.shape)
agent_api.save_table(m, 'e017_grand.parquet')

# screen on late-train holdout
def build_matrix(df, drop=('household_key','snapshot_day','future_spend_4w')):
    drop = list(drop)
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool') and c not in drop]
    num_cols = [c for c in df.columns if c not in cat_cols + drop]
    Xs = []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df)))
    for c in cat_cols:
        Xs.append(pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True).values.astype(float))
    return np.column_stack(Xs)
def ridge(X, snap, y, fit_snaps, eval_snaps, lam=200.0):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    return np.abs(np.clip(Z[isv]@w,0,None)-y[isv]).mean()
b = t8.merge(tt, on=key, how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = b.snapshot_day.values; y = b.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
g = m.merge(tt, on=key, how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
assert (g.household_key.values==b.household_key.values).all()
print('screen E008      : %.4f' % ridge(build_matrix(b.drop(columns=['future_spend_4w'])), snap, y, FIT, EVAL))
print('screen grand union: %.4f' % ridge(build_matrix(g.drop(columns=['future_spend_4w'])), snap, y, FIT, EVAL))
